"""Gate production-wide live theorem-trace claims against the real queue.

This module is intentionally conservative.  It does not manufacture a live
production trace from controlled benchmarks, dry-run probes, or completed
history.  It only answers two reviewer-facing questions:

* are there queued production tasks that can be admitted into the measured
  theorem service domain now?
* if a scheduler trace is supplied, does it have theorem oracle semantics and
  enough real slots to support a live-trace claim?
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.oracle_trace import load_trace_slots
from algorithm.theorem_dispatch.admission import task_admission_certificate

from .production_load_certificate import load_scheduler_records, _dedupe_records, _submitted_at
from .theorem_oracle_trace_bridge import build_theorem_oracle_audit_from_trace


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TRACE = Path.home() / ".claude" / "scheduler" / "oracle_trace.jsonl"

ACTIVE_STATUSES = {"queued", "launching", "running"}
TRACEABLE_QUEUED_STATUSES = {"queued"}
NON_PRODUCTION_PROJECT_TOKENS = {
    "scheduleurmbench",
    "q01-gpu-heavy",
    "q11-hybrid-rl",
    "q00-light-control",
    "q10-cpu-heavy",
    "live-theorem",
}


def build_production_live_theorem_trace_gate(
    *,
    records: Iterable[Mapping[str, Any]],
    trace_path: str | Path = DEFAULT_TRACE,
    min_trace_slots: int = 32,
    now_ts: float | None = None,
) -> dict[str, Any]:
    materialized = _dedupe_records(records)
    active = [dict(row) for row in materialized if str(row.get("status") or "") in ACTIVE_STATUSES]
    queued = [row for row in active if str(row.get("status") or "") in TRACEABLE_QUEUED_STATUSES]
    production_queued = [row for row in queued if _looks_like_production(row)]
    admitted_rows = [_admission_row(row) for row in production_queued]
    admitted = [row for row in admitted_rows if row.get("admitted")]
    trace_report = _trace_report(trace_path=trace_path, min_trace_slots=min_trace_slots)
    status = _gate_status(admitted_rows, admitted, trace_report)
    return {
        "gate": "production_wide_live_theorem_trace_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "now_ts": float(now_ts or time.time()),
        "status": status,
        "queue_summary": {
            "active_count": len(active),
            "queued_count": len(queued),
            "production_queued_count": len(production_queued),
            "admissible_production_queued_count": len(admitted),
            "status_counts": dict(Counter(str(row.get("status") or "") for row in materialized)),
            "active_project_counts": dict(Counter(str(row.get("project") or "") for row in active).most_common(20)),
        },
        "admissible_queued_production": admitted[:80],
        "queued_production_blockers": [row for row in admitted_rows if not row.get("admitted")][:80],
        "trace_report": trace_report,
        "production_wide_live_trace_closed": status == "PRODUCTION_WIDE_LIVE_TRACE_PASS",
        "ready_to_launch_when_admissible_queue_exists": status in {
            "WAIT_NO_QUEUED_PRODUCTION",
            "WAIT_NO_ADMISSIBLE_QUEUED_PRODUCTION",
        },
        "pass": status in {
            "PRODUCTION_WIDE_LIVE_TRACE_PASS",
            "WAIT_NO_QUEUED_PRODUCTION",
            "WAIT_NO_ADMISSIBLE_QUEUED_PRODUCTION",
        },
        "scope": (
            "This is a gate for production-wide live theorem trace claims. "
            "Controlled ScheduleurmBench traces and completed-history service-map "
            "bridges are not counted as production-wide live traces.  If no "
            "admissible queued production task exists, the correct status is WAIT, "
            "not a theorem-grade production dispatch claim."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    q = report.get("queue_summary") or {}
    trace = report.get("trace_report") or {}
    lines = [
        "# Production-Wide Live Theorem Trace Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `status` | `{report.get('status')}` |",
        f"| `active_count` | {q.get('active_count', 0)} |",
        f"| `queued_count` | {q.get('queued_count', 0)} |",
        f"| `production_queued_count` | {q.get('production_queued_count', 0)} |",
        f"| `admissible_production_queued_count` | {q.get('admissible_production_queued_count', 0)} |",
        f"| `trace_slot_count` | {trace.get('slot_count', 0)} |",
        f"| `trace_theorem_slot_count` | {trace.get('theorem_slot_count', 0)} |",
        f"| `trace_oracle_usable` | {str(bool(trace.get('oracle_usable_for_theorem'))).lower()} |",
        f"| `production_wide_live_trace_closed` | {str(bool(report.get('production_wide_live_trace_closed'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ]
    admissible = list(report.get("admissible_queued_production") or [])
    if admissible:
        lines.extend([
            "## Admissible Queued Production",
            "",
            "| Task | Project | Workload | Profiles | Submitted |",
            "|---|---|---|---|---:|",
        ])
        for row in admissible:
            lines.append(
                "| `{task}` | `{project}` | `{workload}` | `{profiles}` | {submitted:.3f} |".format(
                    task=row.get("task_id"),
                    project=row.get("project"),
                    workload=row.get("workload_key"),
                    profiles=",".join(str(x) for x in row.get("measured_profiles") or []),
                    submitted=float(row.get("submitted_at") or 0.0),
                )
            )
        lines.append("")
    blockers = list(report.get("queued_production_blockers") or [])
    if blockers:
        lines.extend([
            "## Queued Production Blockers",
            "",
            "| Task | Project | Reason | Inferred workload |",
            "|---|---|---|---|",
        ])
        for row in blockers:
            lines.append(
                "| `{task}` | `{project}` | `{reason}` | `{workload}` |".format(
                    task=row.get("task_id"),
                    project=row.get("project"),
                    reason=row.get("reason"),
                    workload=row.get("workload_key"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _admission_row(row: Mapping[str, Any]) -> dict[str, Any]:
    cert = task_admission_certificate(row)
    return {
        "task_id": row.get("id") or row.get("task_id"),
        "status": row.get("status"),
        "project": row.get("project"),
        "signature": row.get("signature"),
        "submitted_at": _submitted_at(row),
        "workload_key": cert.workload_key,
        "admitted": bool(cert.admitted),
        "reason": cert.reason,
        "measured_profiles": list(cert.measured_profiles),
    }


def _looks_like_production(row: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("project", "description", "signature", "cmd")
    )
    if not text.strip():
        return False
    if any(token in text for token in NON_PRODUCTION_PROJECT_TOKENS):
        return False
    if "controlled live theorem dispatch" in text or "theorem trace probe" in text:
        return False
    return True


def _trace_report(*, trace_path: str | Path, min_trace_slots: int) -> dict[str, Any]:
    path = Path(trace_path).expanduser()
    if not path.exists():
        return {
            "trace_path": str(path),
            "exists": False,
            "slot_count": 0,
            "theorem_slot_count": 0,
            "oracle_usable_for_theorem": False,
            "min_trace_slots": int(min_trace_slots),
            "reason": "trace_file_missing",
        }
    slots = load_trace_slots(path)
    audit = build_theorem_oracle_audit_from_trace(slots)
    theorem_slots = [
        slot for slot in slots
        if slot.get("score_semantics") == "robust_maxweight_lower_service"
    ]
    return {
        "trace_path": str(path),
        "exists": True,
        "slot_count": len(slots),
        "theorem_slot_count": len(theorem_slots),
        "min_trace_slots": int(min_trace_slots),
        "oracle_usable_for_theorem": bool(audit.get("usable_for_theorem")),
        "enough_slots": len(theorem_slots) >= int(min_trace_slots),
        "audit": audit,
    }


def _gate_status(
    rows: list[Mapping[str, Any]],
    admitted: list[Mapping[str, Any]],
    trace_report: Mapping[str, Any],
) -> str:
    if not rows:
        return "WAIT_NO_QUEUED_PRODUCTION"
    if not admitted:
        return "WAIT_NO_ADMISSIBLE_QUEUED_PRODUCTION"
    if not trace_report.get("exists"):
        return "WAIT_TRACE_FILE_MISSING"
    if not trace_report.get("oracle_usable_for_theorem"):
        return "TRACE_ORACLE_AUDIT_FAIL"
    if not trace_report.get("enough_slots"):
        return "WAIT_MORE_LIVE_TRACE_SLOTS"
    return "PRODUCTION_WIDE_LIVE_TRACE_PASS"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_production_live_theorem_trace_gate(
        records=load_scheduler_records(
            queue_path=args.queue_path or None,
            archive_path=args.archive_path or None,
        ),
        trace_path=args.trace_path,
        min_trace_slots=args.min_trace_slots,
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
        prog="python -m algorithm.experiments.production_live_theorem_trace_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Gate production-wide live theorem trace claims")
    build.add_argument("--queue-path", default="")
    build.add_argument("--archive-path", default="")
    build.add_argument("--trace-path", default=str(DEFAULT_TRACE))
    build.add_argument("--min-trace-slots", type=int, default=32)
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "production_live_theorem_trace_gate.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "production_live_theorem_trace_gate.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
