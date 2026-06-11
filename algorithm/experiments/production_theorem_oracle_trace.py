"""Build a theorem-grade oracle trace from the measured production service map.

This module is deliberately not a live scheduler trace.  It takes the
reviewer-facing completed-active production population, rebuilds the measured
production capacity slice, and emits a robust MaxWeight lower-service trace
slot over that finite service map.  The goal is to certify that the theorem
bridge and alpha0/alpha1 audit close on the same measured lower-service
objects used by Module49/51, while keeping the live-dispatch oracle trace as a
separate requirement.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.defaults import build_default_cache

from .production_coverage_drilldown import population_label
from .production_load_certificate import (
    DEFAULT_TASKSETS,
    build_production_load_certificate,
    load_scheduler_records,
    _dedupe_records,
    _dominating_product_actions,
    _members_for_tasksets,
    _profile_domain,
    _submitted_at,
    _window_end,
)
from .theorem_oracle_trace_bridge import build_theorem_oracle_audit_from_trace


def build_production_theorem_oracle_trace(
    *,
    records: Iterable[Mapping[str, Any]],
    window_days: float = 30.0,
    taskset_names: Iterable[str] = DEFAULT_TASKSETS,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Return a service-map trace and oracle audit for completed production.

    The trace contains one selected dominating measured action and two
    explicitly dominated alternatives.  This is enough to exercise the exact
    lower-service score semantics without pretending that a live greedy
    scheduler emitted the trace.
    """
    tasksets = tuple(str(name) for name in taskset_names)
    materialized = _dedupe_records(records)
    end_ts = _window_end(materialized, now_ts=now_ts)
    window_s = max(1.0, float(window_days) * 86400.0)
    start_ts = end_ts - window_s
    completed_active = [
        dict(row) for row in materialized
        if start_ts <= _submitted_at(row) <= end_ts
        and population_label(row).get("include_completed_active")
    ]
    load = build_production_load_certificate(
        records=completed_active,
        window_days=window_days,
        include_representative=False,
        taskset_names=tasksets,
        now_ts=end_ts,
    )
    trace_slot = _trace_slot_from_load_certificate(load, taskset_names=tasksets)
    audit = build_theorem_oracle_audit_from_trace([trace_slot])
    live_trace_default = str(Path.home() / ".claude" / "scheduler" / "oracle_trace.jsonl")
    report = {
        "status": _status(load, audit),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "window": load.get("window"),
        "completed_active_record_count": len(completed_active),
        "production_load_summary": {
            "mapped_task_count": load.get("mapped_task_count"),
            "representative_mapped_task_count": load.get("representative_mapped_task_count"),
            "unmapped_task_count": load.get("unmapped_task_count"),
            "mapped_capacity_usable_for_theorem": load.get("mapped_capacity_usable_for_theorem"),
            "global_coverage_usable_for_theorem": load.get("global_coverage_usable_for_theorem"),
            "usable_for_global_theorem": load.get("usable_for_global_theorem"),
            "delta": (load.get("capacity") or {}).get("delta"),
            "action_generation": load.get("action_generation"),
            "action_count_evaluated": load.get("action_count_evaluated"),
            "full_action_count": load.get("full_action_count"),
        },
        "oracle_audit": audit,
        "trace_slots": [trace_slot],
        "trace_slot_count": 1,
        "usable_for_service_map_oracle_bridge": (
            bool(load.get("usable_for_global_theorem"))
            and bool(audit.get("usable_for_theorem"))
        ),
        "usable_for_live_scheduler_oracle_trace": False,
        "live_scheduler_trace_path": live_trace_default,
        "live_scheduler_trace_remaining_requirement": (
            "Capture real scheduler candidate slots with "
            "SCHEDULEURM_ORACLE_TRACE_PATH and enrich every candidate with "
            "queue_vector, lower_service, penalty_units, and "
            "score_semantics=robust_maxweight_lower_service before claiming "
            "the live scheduler implements the theorem oracle."
        ),
        "certificate_scope": (
            "service-map theorem oracle bridge for the completed-active "
            "production population; not a live scheduler dispatch trace and "
            "not a raw-history theorem population"
        ),
    }
    return report


def _trace_slot_from_load_certificate(
    load: Mapping[str, Any],
    *,
    taskset_names: Iterable[str],
) -> dict[str, Any]:
    lam = {str(k): float(v) for k, v in (load.get("lambda") or {}).items()}
    members = _members_for_tasksets(taskset_names)
    cache = build_default_cache()
    domains = {member.workload_key: _profile_domain(cache, member) for member in members}
    action = _dominating_product_actions(cache, members, domains)[0]
    lower = {
        str(key): float(value)
        for key, value in (action.get("lower_service") or action.get("service_vector") or {}).items()
    }
    candidate_actions = [
        {
            "action_id": str(action.get("action_id")),
            "selected": True,
            "lower_service": lower,
            "penalty_units": 0.0,
        },
        {
            "action_id": "dominated_half_service",
            "selected": False,
            "lower_service": {key: 0.5 * value for key, value in lower.items()},
            "penalty_units": 0.0,
        },
        {
            "action_id": "dominated_zero_service",
            "selected": False,
            "lower_service": {key: 0.0 for key in lower},
            "penalty_units": 0.0,
        },
    ]
    return {
        "slot_id": "module100:completed_active_production:service_map",
        "score_semantics": "robust_maxweight_lower_service",
        "queue_vector": lam,
        "candidate_count": len(candidate_actions),
        "selected_action_id": str(action.get("action_id")),
        "best_score_exact_over_candidate_set": True,
        "candidates": candidate_actions,
        "source": "measured_production_service_map",
    }


def _status(load: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    if not load.get("usable_for_global_theorem"):
        return "PRODUCTION_LOAD_NOT_CLOSED"
    if not audit.get("usable_for_theorem"):
        return "THEOREM_ORACLE_BRIDGE_FAIL"
    return "SERVICE_MAP_THEOREM_ORACLE_PASS"


def markdown_report(report: Mapping[str, Any]) -> str:
    summary = report.get("production_load_summary") or {}
    audit = report.get("oracle_audit") or {}
    rows = [
        "# Module100 Production Theorem Oracle Bridge",
        "",
        "This certificate is built from the measured service map and the",
        "reviewer-facing completed-active production population. It is not a",
        "live scheduler dispatch trace.",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `status` | `{report.get('status')}` |",
        f"| `completed_active_record_count` | {report.get('completed_active_record_count', 0)} |",
        f"| `mapped_task_count` | {summary.get('mapped_task_count', 0)} |",
        f"| `representative_mapped_task_count` | {summary.get('representative_mapped_task_count', 0)} |",
        f"| `unmapped_task_count` | {summary.get('unmapped_task_count', 0)} |",
        f"| `delta` | {_fmt(summary.get('delta'))} |",
        f"| `alpha0` | {_fmt(audit.get('alpha0'))} |",
        f"| `alpha1` | {_fmt(audit.get('alpha1'))} |",
        f"| `usable_for_service_map_oracle_bridge` | {str(bool(report.get('usable_for_service_map_oracle_bridge'))).lower()} |",
        f"| `usable_for_live_scheduler_oracle_trace` | {str(bool(report.get('usable_for_live_scheduler_oracle_trace'))).lower()} |",
        "",
        "## Interpretation",
        "",
        str(report.get("certificate_scope") or ""),
        "",
        "The remaining live-oracle requirement is separate:",
        "",
        str(report.get("live_scheduler_trace_remaining_requirement") or ""),
        "",
    ]
    return "\n".join(rows)


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return "NA"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_production_theorem_oracle_trace(
        records=load_scheduler_records(
            queue_path=args.queue_path or None,
            archive_path=args.archive_path or None,
        ),
        window_days=args.window_days,
        taskset_names=[name.strip() for name in args.tasksets.split(",") if name.strip()],
    )
    _write_json(args.output, report)
    if args.trace_output:
        _write_jsonl(args.trace_output, report.get("trace_slots") or [])
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("usable_for_service_map_oracle_bridge") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.production_theorem_oracle_trace"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build service-map theorem oracle trace")
    build.add_argument("--queue-path", default="")
    build.add_argument("--archive-path", default="")
    build.add_argument("--window-days", type=float, default=30.0)
    build.add_argument("--tasksets", default=",".join(DEFAULT_TASKSETS))
    build.add_argument("--output", required=True)
    build.add_argument("--trace-output", default="")
    build.add_argument("--markdown-output", default="")
    build.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
