"""Organic production history completion gate.

This gate audits the scheduler's real production history and extracts the
strict theorem-facing population: rows launched by the controlled scheduler
surface, mapped by exact/signature service certificates, and not diagnostic,
auto-adopted, or read-only probe rows.  It is intentionally narrower than raw
history and broader than a single live trace file: it certifies what the
scheduler has already launched and completed organically without manufacturing
unsafe work.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.defaults import build_default_cache

from .production_live_theorem_trace_gate import _looks_like_production
from .production_load_certificate import _dedupe_records, load_scheduler_records
from ..theorem_dispatch.admission import task_admission_certificate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "organic_history_completion_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "organic_history_completion_gate_20260614.md"


def build_organic_history_completion_gate(
    *,
    records: Iterable[Mapping[str, Any]] | None = None,
    launch_threshold: int = 64,
    completion_threshold: int = 50,
    completion_fraction_threshold: float = 0.80,
    workload_domain_threshold: int = 3,
    node_threshold: int = 2,
    max_unadmitted_strict_launched: int = 0,
) -> dict[str, Any]:
    cache = build_default_cache()
    materialized = _dedupe_records(records if records is not None else load_scheduler_records())
    production_rows = [row for row in materialized if _looks_like_production(row)]

    raw_launched = []
    raw_completed = []
    excluded_rows = []
    strict_launched: list[tuple[Mapping[str, Any], Any]] = []
    strict_completed: list[tuple[Mapping[str, Any], Any]] = []
    strict_unadmitted: list[tuple[Mapping[str, Any], Any]] = []
    exclusion_reasons: Counter[str] = Counter()

    for row in production_rows:
        launched = _is_launched(row)
        completed = _is_completed(row)
        if launched:
            raw_launched.append(row)
        if completed:
            raw_completed.append(row)
        reason = _exclusion_reason(row)
        if reason:
            excluded_rows.append(row)
            exclusion_reasons[reason] += 1
            continue
        if not launched:
            continue
        cert = task_admission_certificate(row, cache=cache, admission_mode="strict")
        if cert.admitted:
            strict_launched.append((row, cert))
            if completed:
                strict_completed.append((row, cert))
        else:
            strict_unadmitted.append((row, cert))

    domains = Counter(cert.workload_key for _, cert in strict_launched)
    nodes = Counter(_node(row) for row, _ in strict_launched if _node(row))
    projects = Counter(str(row.get("project") or "") for row, _ in strict_launched)
    completion_fraction = (
        float(len(strict_completed)) / float(len(strict_launched))
        if strict_launched else 0.0
    )
    thresholds_ready = (
        len(strict_launched) >= int(launch_threshold)
        and len(strict_completed) >= int(completion_threshold)
        and completion_fraction >= float(completion_fraction_threshold)
        and len(domains) >= int(workload_domain_threshold)
        and len(nodes) >= int(node_threshold)
        and len(strict_unadmitted) <= int(max_unadmitted_strict_launched)
    )
    status = (
        "ORGANIC_HISTORY_COMPLETION_PASS"
        if thresholds_ready else "ORGANIC_HISTORY_COMPLETION_PENDING"
    )
    return {
        "gate": "organic_history_completion_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "gate_pass": thresholds_ready,
        "scoped_claim_ready": thresholds_ready,
        "strong_claim_ready": thresholds_ready,
        "large_scale_organic_history_completion_ready": thresholds_ready,
        "large_scale_organic_launched_completion_ready": thresholds_ready,
        "admission_mode": "strict_exact_or_signature_service_certificate",
        "raw_production_row_count": len(production_rows),
        "raw_production_launched_count": len(raw_launched),
        "raw_production_completed_count": len(raw_completed),
        "excluded_row_count": len(excluded_rows),
        "excluded_launched_count": sum(1 for row in excluded_rows if _is_launched(row)),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "strict_organic_launched_count": len(strict_launched),
        "strict_organic_completed_count": len(strict_completed),
        "strict_organic_completion_fraction": completion_fraction,
        "strict_unadmitted_launched_count": len(strict_unadmitted),
        "workload_domain_count": len(domains),
        "node_count": len(nodes),
        "project_count": len(projects),
        "launch_threshold": int(launch_threshold),
        "completion_threshold": int(completion_threshold),
        "completion_fraction_threshold": float(completion_fraction_threshold),
        "workload_domain_threshold": int(workload_domain_threshold),
        "node_threshold": int(node_threshold),
        "max_unadmitted_strict_launched": int(max_unadmitted_strict_launched),
        "top_workload_domains": _top(domains, 20),
        "top_nodes": _top(nodes, 20),
        "top_projects": _top(projects, 20),
        "strict_unadmitted_examples": [
            _row_summary(row, cert)
            for row, cert in strict_unadmitted[:20]
        ],
        "excluded_examples": [
            _row_summary(row, None, exclusion_reason=_exclusion_reason(row))
            for row in excluded_rows[:20]
        ],
        "blocker": "" if thresholds_ready else _blocker(
            strict_launched=strict_launched,
            strict_completed=strict_completed,
            strict_unadmitted=strict_unadmitted,
            domains=domains,
            nodes=nodes,
            launch_threshold=launch_threshold,
            completion_threshold=completion_threshold,
            completion_fraction_threshold=completion_fraction_threshold,
            workload_domain_threshold=workload_domain_threshold,
            node_threshold=node_threshold,
            max_unadmitted_strict_launched=max_unadmitted_strict_launched,
            completion_fraction=completion_fraction,
        ),
        "scope": (
            "Real scheduler history audit for the strict theorem-facing organic "
            "production population.  Raw history, attempted-only jobs, cancelled "
            "jobs, auto-adopted jobs, diagnostic probes, and read-only probes are "
            "reported but excluded.  The claim is not about arbitrary future "
            "workloads; future unknown jobs still require the admission/probe "
            "contract."
        ),
        "pass": thresholds_ready,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Organic History Completion Gate",
        "",
        "This gate audits real scheduler history under strict exact/signature service admission. It excludes raw-history rows that are not controlled theorem-facing production rows.",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `admission_mode` | `{report.get('admission_mode')}` |",
        f"| `raw_production_row_count` | {report.get('raw_production_row_count')} |",
        f"| `raw_production_launched_count` | {report.get('raw_production_launched_count')} |",
        f"| `raw_production_completed_count` | {report.get('raw_production_completed_count')} |",
        f"| `excluded_row_count` | {report.get('excluded_row_count')} |",
        f"| `excluded_launched_count` | {report.get('excluded_launched_count')} |",
        f"| `strict_organic_launched_count` | {report.get('strict_organic_launched_count')} |",
        f"| `strict_organic_completed_count` | {report.get('strict_organic_completed_count')} |",
        f"| `strict_organic_completion_fraction` | {report.get('strict_organic_completion_fraction')} |",
        f"| `strict_unadmitted_launched_count` | {report.get('strict_unadmitted_launched_count')} |",
        f"| `workload_domain_count` | {report.get('workload_domain_count')} |",
        f"| `node_count` | {report.get('node_count')} |",
        f"| `project_count` | {report.get('project_count')} |",
        "",
        "## Top Workload Domains",
        "",
        "| Workload key | Count |",
        "|---|---:|",
    ]
    for row in report.get("top_workload_domains") or []:
        lines.append(f"| `{row.get('name')}` | {row.get('count')} |")
    lines.extend([
        "",
        "## Exclusion Reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ])
    for reason, count in (report.get("exclusion_reason_counts") or {}).items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend([
        "",
        "## Blocker",
        "",
        str(report.get("blocker") or "none"),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _is_launched(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    return status in {"running", "done"} and bool(
        row.get("started_at") or row.get("node") or status in {"running", "done"}
    )


def _is_completed(row: Mapping[str, Any]) -> bool:
    return str(row.get("status") or "").lower() == "done"


def _node(row: Mapping[str, Any]) -> str:
    return str(row.get("node") or row.get("selected_node") or "")


def _exclusion_reason(row: Mapping[str, Any]) -> str:
    text = _row_text(row)
    project = str(row.get("project") or "").strip().lower()
    if project in {"tmp", "md", "proof", "sched", "python"}:
        return f"nonproduction_project:{project}"
    if "auto-adopted" in text:
        return "auto_adopted"
    if ("adopted" in text and "outside scheduler" in text) or "launched outside scheduler" in text:
        return "outside_scheduler_adopted"
    if "read-only probe" in text:
        return "read_only_probe"
    if "stdout probe" in text:
        return "stdout_probe"
    if "theorem trace probe" in text:
        return "theorem_trace_probe"
    if "controlled live theorem dispatch" in text:
        return "controlled_live_trace_probe"
    return ""


def _row_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "").lower()
        for key in ("id", "project", "signature", "description", "cmd", "cwd", "notes")
    )


def _top(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [
        {"name": str(name), "count": int(count)}
        for name, count in counter.most_common(int(limit))
    ]


def _row_summary(row: Mapping[str, Any], cert: Any, *, exclusion_reason: str = "") -> dict[str, Any]:
    return {
        "task_id": str(row.get("id") or row.get("task_id") or ""),
        "project": str(row.get("project") or ""),
        "signature": str(row.get("signature") or ""),
        "status": str(row.get("status") or ""),
        "node": _node(row),
        "workload_key": str(getattr(cert, "workload_key", "") or ""),
        "admission_reason": str(getattr(cert, "reason", "") or ""),
        "exclusion_reason": str(exclusion_reason or ""),
    }


def _blocker(
    *,
    strict_launched: list[tuple[Mapping[str, Any], Any]],
    strict_completed: list[tuple[Mapping[str, Any], Any]],
    strict_unadmitted: list[tuple[Mapping[str, Any], Any]],
    domains: Counter[str],
    nodes: Counter[str],
    launch_threshold: int,
    completion_threshold: int,
    completion_fraction_threshold: float,
    workload_domain_threshold: int,
    node_threshold: int,
    max_unadmitted_strict_launched: int,
    completion_fraction: float,
) -> str:
    blockers = []
    if len(strict_launched) < int(launch_threshold):
        blockers.append(f"strict launched count {len(strict_launched)} < {int(launch_threshold)}")
    if len(strict_completed) < int(completion_threshold):
        blockers.append(f"strict completed count {len(strict_completed)} < {int(completion_threshold)}")
    if completion_fraction < float(completion_fraction_threshold):
        blockers.append(
            f"completion fraction {completion_fraction:.6f} < {float(completion_fraction_threshold):.6f}"
        )
    if len(domains) < int(workload_domain_threshold):
        blockers.append(f"workload domains {len(domains)} < {int(workload_domain_threshold)}")
    if len(nodes) < int(node_threshold):
        blockers.append(f"nodes {len(nodes)} < {int(node_threshold)}")
    if len(strict_unadmitted) > int(max_unadmitted_strict_launched):
        blockers.append(
            f"strict unadmitted launched {len(strict_unadmitted)} > {int(max_unadmitted_strict_launched)}"
        )
    return "; ".join(blockers)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_organic_history_completion_gate(
        launch_threshold=args.launch_threshold,
        completion_threshold=args.completion_threshold,
        completion_fraction_threshold=args.completion_fraction_threshold,
        workload_domain_threshold=args.workload_domain_threshold,
        node_threshold=args.node_threshold,
        max_unadmitted_strict_launched=args.max_unadmitted_strict_launched,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.organic_history_completion_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build organic production history completion gate")
    build.add_argument("--launch-threshold", type=int, default=64)
    build.add_argument("--completion-threshold", type=int, default=50)
    build.add_argument("--completion-fraction-threshold", type=float, default=0.80)
    build.add_argument("--workload-domain-threshold", type=int, default=3)
    build.add_argument("--node-threshold", type=int, default=2)
    build.add_argument("--max-unadmitted-strict-launched", type=int, default=0)
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
