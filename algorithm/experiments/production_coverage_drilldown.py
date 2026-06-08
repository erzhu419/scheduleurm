"""Production coverage drilldown for global Scheduleurm closure.

Module49 answers whether mapped task records fit inside measured capacity.  This
module answers the sharper reviewer question: which task population is being
claimed, and what remains outside the measured service map?
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .production_load_certificate import (
    build_production_load_certificate,
    classify_record,
    load_scheduler_records,
    _dedupe_records,
    _submitted_at,
    _window_end,
)


PopulationFilter = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]


def build_coverage_drilldown(
    *,
    records: Iterable[Mapping[str, Any]],
    window_days: float = 30.0,
    now_ts: float | None = None,
) -> dict[str, Any]:
    materialized = _dedupe_records(records)
    end_ts = _window_end(materialized, now_ts=now_ts)
    window_s = max(1.0, float(window_days) * 86400.0)
    start_ts = end_ts - window_s
    rows = [
        dict(row) for row in materialized
        if start_ts <= _submitted_at(row) <= end_ts
    ]
    views = {}
    for name, predicate in _population_views().items():
        selected = [row for row in rows if predicate(row, population_label(row))]
        strict = build_production_load_certificate(
            records=selected,
            window_days=window_days,
            include_representative=False,
            now_ts=now_ts,
        )
        representative = build_production_load_certificate(
            records=selected,
            window_days=window_days,
            include_representative=True,
            now_ts=now_ts,
        )
        views[name] = _view_summary(strict, representative)
    production_predicate = _population_views()["completed_active_production"]
    production_rows = [
        row for row in rows if production_predicate(row, population_label(row))
    ]
    obligations = _coverage_obligations(production_rows)
    return {
        "window_days": float(window_days),
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "window_s": window_s,
        },
        "view_count": len(views),
        "views": views,
        "completed_active_production_obligations": obligations,
        "global_theorem_closed": bool(
            views["completed_active_production"]["representative"]["usable_for_global_theorem"]
        ),
        "interpretation": (
            "Global production stability is not closed until the reviewer-facing "
            "production population has full strict measured-bucket coverage, or "
            "each remaining bucket has its own theorem-grade service certificate."
        ),
    }


def population_label(row: Mapping[str, Any]) -> dict[str, Any]:
    status = str(row.get("status") or "")
    text = _record_text(row)
    project = str(row.get("project") or "").strip().lower()
    signature = str(row.get("signature") or "").strip().lower()
    if status == "cancelled":
        return {"label": "excluded_cancelled", "include_attempted": False, "include_completed_active": False}
    if status == "forgotten":
        return {"label": "excluded_forgotten", "include_attempted": False, "include_completed_active": False}
    if "scheduleurmbench" in text:
        return {"label": "excluded_benchmark", "include_attempted": False, "include_completed_active": False}
    if project == "tmp" or signature.startswith("test/"):
        return {"label": "excluded_test_or_guard", "include_attempted": False, "include_completed_active": False}
    if "auto-adopted" in text and any(tok in text for tok in ("ptxas", "cuda_nvcc", "tempfile-")):
        return {"label": "excluded_adopted_compiler_aux", "include_attempted": False, "include_completed_active": False}
    return {
        "label": "included_production_like",
        "include_attempted": status not in ("cancelled", "forgotten"),
        "include_completed_active": status in ("done", "running", "queued", "launching"),
    }


def bucket_obligation(row: Mapping[str, Any]) -> dict[str, Any]:
    cls = classify_record(row, include_representative=True)
    pop = population_label(row)
    if cls.get("workload_key"):
        return {
            "bucket": str(cls.get("workload_key")),
            "status": "mapped",
            "reason": str(cls.get("reason") or ""),
            "population_label": pop["label"],
        }
    if not pop.get("include_attempted"):
        return {
            "bucket": str(pop["label"]),
            "status": "excluded_from_production_population",
            "reason": str(pop["label"]),
            "population_label": pop["label"],
        }
    text = _record_text(row)
    est_vram = _as_float(row.get("est_vram_mb", row.get("vram_mb", 0.0)))
    if est_vram > 0:
        if any(tok in text for tok in ("jax_experiments.train", "resac", "re-sac", "bapr", "sac", "hopper", "walker2d", "halfcheetah", "ant-v2")):
            bucket = "gpu_rl_unmeasured_variant"
        else:
            bucket = "gpu_unmeasured_production"
    elif any(tok in text for tok in (
        "sumo", "transit", "freqduet", "freqhrl", "h2oplus", "simplesac",
        "offline-sumo", "bamor",
    )):
        bucket = "cpu_sumo_transit_eval_or_control"
    elif any(tok in text for tok in ("eval", "evaluate", "rollout")):
        bucket = "cpu_eval_generic"
    elif any(tok in text for tok in ("lean", "lake", "proof")):
        bucket = "proof_build_cpu"
    elif any(tok in text for tok in ("tar", "rsync", "scp", "git", "download", "reference")):
        bucket = "artifact_io_control"
    elif any(tok in text for tok in ("scheduler", "scheduleurm", "dispatch", "status", "watch", "profile")):
        bucket = "scheduler_control_plane"
    elif "python" in text:
        bucket = "generic_cpu_python"
    else:
        bucket = "misc_unmeasured_cpu"
    return {
        "bucket": bucket,
        "status": "measurement_required",
        "reason": cls.get("reason") or "unmapped",
        "population_label": pop["label"],
    }


def _population_views() -> dict[str, PopulationFilter]:
    return {
        "raw_history_all": lambda _row, _pop: True,
        "attempted_production": lambda _row, pop: bool(pop.get("include_attempted")),
        "completed_active_production": lambda _row, pop: bool(pop.get("include_completed_active")),
    }


def _view_summary(strict: Mapping[str, Any], representative: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strict": _one_summary(strict),
        "representative": _one_summary(representative),
    }


def _one_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_count_window": report.get("record_count_window"),
        "mapped_task_count": report.get("mapped_task_count"),
        "representative_mapped_task_count": report.get("representative_mapped_task_count"),
        "unmapped_task_count": report.get("unmapped_task_count"),
        "mapped_fraction": report.get("mapped_fraction"),
        "strict_mapped_fraction": report.get("strict_mapped_fraction"),
        "lambda": report.get("lambda"),
        "delta": (report.get("capacity") or {}).get("delta"),
        "mapped_capacity_usable_for_theorem": report.get("mapped_capacity_usable_for_theorem"),
        "global_coverage_usable_for_theorem": report.get("global_coverage_usable_for_theorem"),
        "usable_for_global_theorem": report.get("usable_for_global_theorem"),
        "unmapped_reasons": report.get("unmapped_reasons"),
    }


def _coverage_obligations(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    bucket_counts: Counter[str] = Counter()
    bucket_status: dict[str, str] = {}
    project_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    signature_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    mapped = 0
    for row in rows:
        total += 1
        obligation = bucket_obligation(row)
        bucket = str(obligation["bucket"])
        bucket_counts[bucket] += 1
        bucket_status[bucket] = str(obligation["status"])
        project_by_bucket[bucket][str(row.get("project") or "")] += 1
        signature_by_bucket[bucket][_signature_prefix(row)] += 1
        if obligation["status"] == "mapped":
            mapped += 1
        if len(examples[bucket]) < 5:
            examples[bucket].append(_example(row, obligation))
    top = []
    for bucket, count in bucket_counts.most_common():
        top.append({
            "bucket": bucket,
            "count": count,
            "status": bucket_status.get(bucket, ""),
            "fraction_of_completed_active_production": count / total if total else 0.0,
            "top_projects": dict(project_by_bucket[bucket].most_common(8)),
            "top_signature_prefixes": dict(signature_by_bucket[bucket].most_common(8)),
            "examples": examples[bucket],
        })
    return {
        "record_count": total,
        "mapped_count": mapped,
        "measurement_required_count": sum(
            count for bucket, count in bucket_counts.items()
            if bucket_status.get(bucket) == "measurement_required"
        ),
        "mapped_fraction": mapped / total if total else 0.0,
        "top_buckets": top,
    }


def _example(row: Mapping[str, Any], obligation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row.get("id") or row.get("task_id"),
        "status": row.get("status"),
        "project": row.get("project"),
        "signature": row.get("signature"),
        "description": row.get("description"),
        "est_vram_mb": row.get("est_vram_mb"),
        "cpu_cores": row.get("cpu_cores"),
        "bucket": obligation.get("bucket"),
        "reason": obligation.get("reason"),
    }


def _signature_prefix(row: Mapping[str, Any]) -> str:
    raw = str(row.get("signature") or "").strip()
    if not raw:
        return ""
    parts = [part for part in re.split(r"[/|,:]+", raw) if part]
    return "/".join(parts[:3])


def _record_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("project", "signature", "description", "cmd", "cwd")
    ).lower()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Module51 Production Coverage Drilldown",
        "",
        "## Population Views",
        "",
        "| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for view_name, view in (report.get("views") or {}).items():
        for mode in ("strict", "representative"):
            row = view.get(mode) or {}
            lines.append(
                "| `{view}` | `{mode}` | {records} | {mapped} | {unmapped} | {fraction:.6f} | {delta:.9f} | {usable} |".format(
                    view=view_name,
                    mode=mode,
                    records=int(row.get("record_count_window") or 0),
                    mapped=int(row.get("mapped_task_count") or 0),
                    unmapped=int(row.get("unmapped_task_count") or 0),
                    fraction=float(row.get("mapped_fraction") or 0.0),
                    delta=float(row.get("delta") or 0.0),
                    usable=str(bool(row.get("usable_for_global_theorem"))).lower(),
                )
            )
    obligations = report.get("completed_active_production_obligations") or {}
    lines.extend([
        "",
        "## Completed/Active Production Obligations",
        "",
        "```text",
        f"record_count = {obligations.get('record_count')}",
        f"mapped_count = {obligations.get('mapped_count')}",
        f"measurement_required_count = {obligations.get('measurement_required_count')}",
        f"mapped_fraction = {obligations.get('mapped_fraction')}",
        "```",
        "",
        "| Bucket | Status | Count | Fraction | Top Projects |",
        "|---|---|---:|---:|---|",
    ])
    for row in obligations.get("top_buckets", [])[:20]:
        projects = ", ".join(f"{k}:{v}" for k, v in (row.get("top_projects") or {}).items())
        lines.append(
            "| `{bucket}` | `{status}` | {count} | {fraction:.6f} | {projects} |".format(
                bucket=row.get("bucket"),
                status=row.get("status"),
                count=int(row.get("count") or 0),
                fraction=float(row.get("fraction_of_completed_active_production") or 0.0),
                projects=projects,
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        str(report.get("interpretation") or ""),
        "",
        "The strict theorem-facing standard is full measured-bucket coverage of the",
        "declared production population. Representative mappings are diagnostic",
        "unless separately backed by service measurements or equivalence tests.",
        "",
    ])
    return "\n".join(lines)


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_coverage_drilldown(
        records=load_scheduler_records(queue_path=args.queue, archive_path=args.archive),
        window_days=args.window_days,
        now_ts=args.now_ts,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        Path(args.markdown_output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).expanduser().write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("global_theorem_closed") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.production_coverage_drilldown")
    sub = p.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build production coverage drilldown")
    build.add_argument("--output", required=True)
    build.add_argument("--markdown-output", default="")
    build.add_argument("--queue", default="")
    build.add_argument("--archive", default="")
    build.add_argument("--window-days", type=float, default=30.0)
    build.add_argument("--now-ts", type=float, default=None)
    build.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
