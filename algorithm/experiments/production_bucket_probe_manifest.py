"""Build probe manifests for unmapped production coverage buckets."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from .production_coverage_drilldown import bucket_obligation, population_label
from .production_load_certificate import (
    load_scheduler_records,
    _dedupe_records,
    _record_timestamp,
    _submitted_at,
    _window_end,
)


DEFAULT_BUCKET = "cpu_sumo_transit_eval_or_control"


def build_probe_manifest(
    *,
    records: Iterable[Mapping[str, Any]],
    bucket: str = DEFAULT_BUCKET,
    window_days: float = 30.0,
    now_ts: float | None = None,
    max_groups: int = 24,
) -> dict[str, Any]:
    materialized = _dedupe_records(records)
    end_ts = _window_end(materialized, now_ts=now_ts)
    window_s = max(1.0, float(window_days) * 86400.0)
    start_ts = end_ts - window_s
    rows = [
        dict(row) for row in materialized
        if start_ts <= _submitted_at(row) <= end_ts
        and population_label(row).get("include_completed_active")
        and bucket_obligation(row).get("bucket") == bucket
    ]
    sub_buckets = _sub_bucket_rows(rows)
    has_remaining_records = bool(rows)
    return {
        "bucket": bucket,
        "window": {"start_ts": start_ts, "end_ts": end_ts, "window_days": float(window_days)},
        "record_count": len(rows),
        "status_counts": dict(Counter(str(row.get("status") or "") for row in rows)),
        "project_counts": dict(Counter(str(row.get("project") or "") for row in rows).most_common(30)),
        "resource_summary": _resource_summary(rows),
        "sub_bucket_count": len(sub_buckets),
        "sub_buckets": sub_buckets[:max(1, int(max_groups))],
        "probe_grid": _probe_grid(rows),
        "theorem_status": "measurement_required" if has_remaining_records else "no_remaining_records",
        "interpretation": (
            "This manifest extracts real production templates for a missing "
            "coverage bucket. It does not certify service until the listed "
            "sub-buckets receive progress-bearing service curves."
            if has_remaining_records
            else "No completed/active production records remain in this bucket."
        ),
    }


def _sub_bucket_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_sub_bucket_key(row)].append(row)
    out = []
    for key, items in groups.items():
        out.append({
            "sub_bucket": key,
            "count": len(items),
            "fraction": len(items) / len(rows) if rows else 0.0,
            "project_counts": dict(Counter(str(row.get("project") or "") for row in items).most_common(8)),
            "signature_prefixes": dict(Counter(_signature_prefix(row) for row in items).most_common(8)),
            "resource_summary": _resource_summary(items),
            "examples": [_example(row) for row in items[:5]],
            "recommended_probe_profiles": _recommended_profiles(items),
        })
    return sorted(out, key=lambda row: (-int(row["count"]), str(row["sub_bucket"])))


def _sub_bucket_key(row: Mapping[str, Any]) -> str:
    text = _record_text(row)
    project = str(row.get("project") or "").lower()
    cpu = _as_float(row.get("cpu_cores"), 0.0)
    if "bamor" in project or "train_compare_baselines.py" in text:
        family = "bamor_cpu_training"
    elif any(tok in text for tok in ("freqduet", "run_freqduet_ablation")):
        family = "freqduet_cpu_ablation"
    elif any(tok in text for tok in ("freq_hrl", "native_promotion", "transit_hrl")):
        family = "transit_freqhrl_cpu_validation"
    elif any(tok in text for tok in ("simplesac", "h2oplus", "offline-sumo", "sumo")):
        family = "sumo_eval_cpu"
    else:
        family = "transit_misc_cpu"
    return f"{family}|{_cpu_bucket(cpu)}"


def _cpu_bucket(cpu: float) -> str:
    if cpu <= 2:
        return "c_le2"
    if cpu <= 8:
        return "c_3_8"
    if cpu <= 16:
        return "c_9_16"
    if cpu <= 32:
        return "c_17_32"
    if cpu <= 64:
        return "c_33_64"
    return "c_65p"


def _resource_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    cpu = [_as_float(row.get("cpu_cores"), 0.0) for row in rows]
    ram = [_as_float(row.get("ram_mb"), 0.0) for row in rows]
    return {
        "cpu_cores": _quantiles(cpu),
        "ram_mb": _quantiles(ram),
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return {"min": None, "median": None, "p90": None, "max": None}
    return {
        "min": clean[0],
        "median": float(median(clean)),
        "p90": clean[min(len(clean) - 1, int(0.90 * (len(clean) - 1)))],
        "max": clean[-1],
    }


def _recommended_profiles(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    cpu_values = sorted({_as_float(row.get("cpu_cores"), 0.0) for row in rows})
    cpus = [int(v) for v in cpu_values if v > 0]
    representative_cpus = sorted({cpus[0], cpus[len(cpus) // 2], cpus[-1]}) if cpus else []
    return {
        "task_concurrency_profiles": [1, 2, 4, 8],
        "representative_cpu_cores_per_task": representative_cpus,
        "requires_progress_unit": True,
        "suggested_units": [
            "completed_seed",
            "completed_episode",
            "completed_eval_pair",
            "completed_training_step",
        ],
    }


def _probe_grid(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    cpu_buckets = sorted({_cpu_bucket(_as_float(row.get("cpu_cores"), 0.0)) for row in rows})
    first_order = [
        key for key, _count in Counter(_sub_bucket_key(row) for row in rows).most_common(8)
    ]
    return {
        "cpu_buckets": cpu_buckets,
        "task_concurrency_profiles": [1, 2, 4, 8],
        "node_targets": ["local_cpu", "direct_hpc_cpu_node"],
        "first_probe_order": first_order,
    }


def _signature_prefix(row: Mapping[str, Any]) -> str:
    raw = str(row.get("signature") or "")
    parts = [part for part in re.split(r"[/|,:]+", raw) if part]
    return "/".join(parts[:4])


def _record_text(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in ("project", "signature", "description", "cmd", "cwd")).lower()


def _example(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row.get("id") or row.get("task_id"),
        "status": row.get("status"),
        "project": row.get("project"),
        "signature": row.get("signature"),
        "description": row.get("description"),
        "cwd": row.get("cwd"),
        "cmd": _shorten(str(row.get("cmd") or ""), 500),
        "cpu_cores": row.get("cpu_cores"),
        "ram_mb": row.get("ram_mb"),
        "eta_seconds": row.get("eta_seconds"),
        "eta_source": row.get("eta_source"),
    }


def _shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


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
        "# Module53 Production Bucket Probe Manifest",
        "",
        "```text",
        f"bucket = {report.get('bucket')}",
        f"record_count = {report.get('record_count')}",
        f"theorem_status = {report.get('theorem_status')}",
        "```",
        "",
        "## Resource Summary",
        "",
        "| Resource | Min | Median | P90 | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in (report.get("resource_summary") or {}).items():
        lines.append(
            "| `{name}` | {min} | {med} | {p90} | {max} |".format(
                name=name,
                min=_fmt(stats.get("min")),
                med=_fmt(stats.get("median")),
                p90=_fmt(stats.get("p90")),
                max=_fmt(stats.get("max")),
            )
        )
    lines.extend([
        "",
        "## Sub-Buckets",
        "",
        "| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |",
        "|---|---:|---:|---:|---|",
    ])
    for row in report.get("sub_buckets", []):
        cpu = ((row.get("resource_summary") or {}).get("cpu_cores") or {})
        projects = ", ".join(f"{k}:{v}" for k, v in (row.get("project_counts") or {}).items())
        lines.append(
            "| `{bucket}` | {count} | {fraction:.6f} | {cpu} | {projects} |".format(
                bucket=row.get("sub_bucket"),
                count=int(row.get("count") or 0),
                fraction=float(row.get("fraction") or 0.0),
                cpu=_fmt(cpu.get("median")),
                projects=projects,
            )
        )
    grid = report.get("probe_grid") or {}
    lines.extend([
        "",
        "## Probe Grid",
        "",
        "```text",
        f"task_concurrency_profiles = {grid.get('task_concurrency_profiles')}",
        f"node_targets = {grid.get('node_targets')}",
        f"first_probe_order = {grid.get('first_probe_order')}",
        "```",
        "",
        "## Interpretation",
        "",
        str(report.get("interpretation") or ""),
        "",
    ])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        out = float(value)
        return f"{out:.3f}" if abs(out) >= 1000 else f"{out:.6f}"
    except (TypeError, ValueError):
        return "NA"


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_probe_manifest(
        records=load_scheduler_records(queue_path=args.queue, archive_path=args.archive),
        bucket=args.bucket,
        window_days=args.window_days,
        now_ts=args.now_ts,
        max_groups=args.max_groups,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("record_count", 0) > 0 else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.production_bucket_probe_manifest")
    sub = p.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build probe manifest for a missing production bucket")
    build.add_argument("--bucket", default=DEFAULT_BUCKET)
    build.add_argument("--output", required=True)
    build.add_argument("--markdown-output", default="")
    build.add_argument("--queue", default="")
    build.add_argument("--archive", default="")
    build.add_argument("--window-days", type=float, default=30.0)
    build.add_argument("--now-ts", type=float, default=None)
    build.add_argument("--max-groups", type=int, default=24)
    build.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
