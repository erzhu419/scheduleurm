"""Close a live scheduler trace against lower-service oracle semantics.

The raw scheduler trace records the implementation's candidate family and
sort-key decision.  This module attaches a conservative measured lower-service
row to every observed candidate action, writes an enriched theorem trace, and
audits alpha0/alpha1 through the existing theorem bridge.

It is intentionally scoped to a live trace file: if the scheduler did not emit
candidate slots, this module fails instead of treating replay data as a live
dispatch certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.oracle_trace import audit_scheduler_score_trace, load_trace_slots
from simulation.defaults import build_default_cache

from .oracle_trace_enrichment import enrich_trace_slots
from .theorem_oracle_trace_bridge import build_theorem_oracle_audit_from_trace


def build_live_scheduler_oracle_closure(
    *,
    trace_path: str | Path,
    workload_key: str = "scheduleurm_control_plane_completed_history",
    profile: int = 1,
    queue_weight: float = 1.0,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    slots = load_trace_slots(trace) if trace.exists() else []
    scheduler_audit = audit_scheduler_score_trace(slots)
    if not trace.exists():
        return _blocked(
            trace=trace,
            status="NO_TRACE",
            reason="trace_file_does_not_exist",
            slots=slots,
            scheduler_audit=scheduler_audit,
        )
    if not slots:
        return _blocked(
            trace=trace,
            status="EMPTY_TRACE",
            reason="trace_file_exists_but_has_no_slots",
            slots=slots,
            scheduler_audit=scheduler_audit,
        )
    service_rate, service_source = _measured_service_rate(workload_key, profile)
    service_rows = _service_rows_for_slots(
        slots,
        workload_key=workload_key,
        service_rate=service_rate,
        service_source=service_source,
    )
    queue_vector = {workload_key: max(0.0, float(queue_weight))}
    enrich = enrich_trace_slots(
        slots,
        service_rows=service_rows,
        queue_vector=queue_vector,
    )
    enriched_slots = list(enrich.get("enriched_slots") or [])
    bridge = build_theorem_oracle_audit_from_trace(enriched_slots)
    status = (
        "LIVE_SCHEDULER_THEOREM_ORACLE_PASS"
        if scheduler_audit.get("usable_for_scheduler_score_audit")
        and enrich.get("usable_for_theorem")
        and bridge.get("usable_for_theorem")
        else "LIVE_SCHEDULER_THEOREM_ORACLE_FAIL"
    )
    return {
        "status": status,
        "input": str(trace),
        "trace_slot_count": len(slots),
        "candidate_count_total": sum(len(slot.get("candidates") or []) for slot in slots),
        "workload_key": workload_key,
        "profile": int(profile),
        "queue_vector": queue_vector,
        "service_rate": service_rate,
        "service_source": service_source,
        "service_rows": service_rows,
        "scheduler_score_audit": scheduler_audit,
        "enrichment": enrich,
        "theorem_oracle_bridge": bridge,
        "enriched_slots": enriched_slots,
        "usable_for_live_scheduler_oracle_trace": status == "LIVE_SCHEDULER_THEOREM_ORACLE_PASS",
        "certificate_scope": (
            "one emitted live scheduler candidate-family trace, enriched with "
            "a conservative measured lower-service row for the declared "
            "workload key; not a universal claim over future dispatches"
        ),
    }


def _blocked(
    *,
    trace: Path,
    status: str,
    reason: str,
    slots: list[Mapping[str, Any]],
    scheduler_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "input": str(trace),
        "trace_slot_count": len(slots),
        "candidate_count_total": 0,
        "blockers": [{"reason": reason, "path": str(trace)}],
        "scheduler_score_audit": dict(scheduler_audit),
        "usable_for_live_scheduler_oracle_trace": False,
        "certificate_scope": "live trace absent; no live scheduler oracle claim",
    }


def _measured_service_rate(workload_key: str, profile: int) -> tuple[float, str]:
    record = build_default_cache().get(workload_key, int(profile))
    if record is None or record.capacity_boundary or float(record.aggregate_rate) <= 0.0:
        raise KeyError(f"no positive measured service for {workload_key!r} profile {profile}")
    return float(record.aggregate_rate), str(record.source)


def _service_rows_for_slots(
    slots: Iterable[Mapping[str, Any]],
    *,
    workload_key: str,
    service_rate: float,
    service_source: str,
) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for slot in slots:
        for candidate in slot.get("candidates") or []:
            action_id = str(candidate.get("action_id") or "").strip()
            if not action_id or action_id in seen:
                continue
            seen.add(action_id)
            rows.append({
                "action_id": action_id,
                "lower_service": {workload_key: max(0.0, float(service_rate))},
                "penalty_units": 0.0,
                "source": service_source,
            })
    return rows


def markdown_report(report: Mapping[str, Any]) -> str:
    sched = report.get("scheduler_score_audit") or {}
    bridge = report.get("theorem_oracle_bridge") or {}
    lines = [
        "# Module101 Live Scheduler Oracle Closure",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `status` | `{report.get('status')}` |",
        f"| `trace_slot_count` | {report.get('trace_slot_count', 0)} |",
        f"| `candidate_count_total` | {report.get('candidate_count_total', 0)} |",
        f"| `workload_key` | `{report.get('workload_key', '')}` |",
        f"| `profile` | {report.get('profile', '')} |",
        f"| `service_rate` | {_fmt(report.get('service_rate'))} |",
        f"| `scheduler_score_pass` | {str(bool(sched.get('usable_for_scheduler_score_audit'))).lower()} |",
        f"| `alpha0` | {_fmt(bridge.get('alpha0'))} |",
        f"| `alpha1` | {_fmt(bridge.get('alpha1'))} |",
        f"| `theorem_bridge_pass` | {str(bool(bridge.get('usable_for_theorem'))).lower()} |",
        f"| `usable_for_live_scheduler_oracle_trace` | {str(bool(report.get('usable_for_live_scheduler_oracle_trace'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("certificate_scope") or ""),
        "",
    ]
    blockers = list(report.get("blockers") or [])
    if blockers:
        lines.extend(["## Blockers", "", "| Reason | Path |", "|---|---|"])
        for row in blockers:
            lines.append(f"| `{row.get('reason')}` | `{row.get('path')}` |")
        lines.append("")
    return "\n".join(lines)


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
    report = build_live_scheduler_oracle_closure(
        trace_path=args.input,
        workload_key=args.workload_key,
        profile=args.profile,
        queue_weight=args.queue_weight,
    )
    _write_json(args.output, report)
    if args.enriched_trace_output:
        _write_jsonl(args.enriched_trace_output, report.get("enriched_slots") or [])
    if args.service_lookup_output:
        _write_json(args.service_lookup_output, {"service_rows": report.get("service_rows") or []})
    if args.queue_vector_output:
        _write_json(args.queue_vector_output, {"queue_vector": report.get("queue_vector") or {}})
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("usable_for_live_scheduler_oracle_trace") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.live_scheduler_oracle_closure"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Enrich and audit a live scheduler trace")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--markdown-output", default="")
    build.add_argument("--enriched-trace-output", default="")
    build.add_argument("--service-lookup-output", default="")
    build.add_argument("--queue-vector-output", default="")
    build.add_argument("--workload-key", default="scheduleurm_control_plane_completed_history")
    build.add_argument("--profile", type=int, default=1)
    build.add_argument("--queue-weight", type=float, default=1.0)
    build.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
