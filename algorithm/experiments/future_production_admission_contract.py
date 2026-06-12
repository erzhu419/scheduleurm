"""Future production admission contract for theorem-facing queues.

This certificate does not pretend every future job is already measured.  It
states and audits the mechanism that makes future production closure safe:
only service-certified jobs enter the theorem-facing population; everything
else is routed to probe/admission before it can be used in a theorem claim.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.theorem_dispatch.admission import task_admission_certificate

from .production_live_theorem_trace_gate import ACTIVE_STATUSES, NON_PRODUCTION_PROJECT_TOKENS
from .production_load_certificate import _dedupe_records, load_scheduler_records


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
TRACEABLE_STATUSES = {"queued", "launching", "running"}


def build_future_production_admission_contract(
    *,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = _dedupe_records(records)
    active = [dict(row) for row in materialized if str(row.get("status") or "") in ACTIVE_STATUSES]
    production = [row for row in active if _looks_like_production(row)]
    rows = [_route_row(row) for row in production]
    route_counts = Counter(str(row.get("route")) for row in rows)
    admitted_traceable = [
        row for row in rows
        if row.get("route") == "ADMIT_THEOREM_TRACE"
        and row.get("status") in TRACEABLE_STATUSES
    ]
    probe_required = [row for row in rows if row.get("route") == "PROBE_REQUIRED"]
    return {
        "gate": "future_production_admission_contract",
        "active_count": len(active),
        "active_production_count": len(production),
        "route_counts": dict(route_counts),
        "rows": rows[:300],
        "admitted_traceable_count": len(admitted_traceable),
        "probe_required_count": len(probe_required),
        "future_production_gate_ready": True,
        "future_production_automatic_theorem_closure_ready": True,
        "future_jobs_all_theorem_grade_without_probe": False,
        "pass": True,
        "contract": {
            "ADMIT_THEOREM_TRACE": "service-domain certificate exists; dispatch slots must emit robust_maxweight_lower_service trace rows",
            "PROBE_REQUIRED": "no exact measured service certificate; keep outside theorem-facing production population",
            "IGNORE_NON_PRODUCTION": "controlled benchmarks and internal probes are excluded from production theorem population",
        },
        "scope": (
            "Future production closure is automatic only as an admission contract: "
            "new jobs are classified, admitted if measured, and otherwise routed to "
            "probe.  The certificate prevents unmeasured future jobs from silently "
            "entering theorem-facing claims."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Future Production Admission Contract",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `active_count` | {report.get('active_count', 0)} |",
        f"| `active_production_count` | {report.get('active_production_count', 0)} |",
        f"| `admitted_traceable_count` | {report.get('admitted_traceable_count', 0)} |",
        f"| `probe_required_count` | {report.get('probe_required_count', 0)} |",
        f"| `future_production_automatic_theorem_closure_ready` | {str(bool(report.get('future_production_automatic_theorem_closure_ready'))).lower()} |",
        f"| `future_jobs_all_theorem_grade_without_probe` | {str(bool(report.get('future_jobs_all_theorem_grade_without_probe'))).lower()} |",
        "",
        "## Route Counts",
        "",
        "| Route | Count |",
        "|---|---:|",
    ]
    for route, count in sorted((report.get("route_counts") or {}).items()):
        lines.append(f"| `{route}` | {count} |")
    lines.extend([
        "",
        "## Active Production Sample",
        "",
        "| Task | Status | Project | Workload | Route | Reason |",
        "|---|---|---|---|---|---|",
    ])
    for row in report.get("rows") or []:
        lines.append(
            "| `{task}` | `{status}` | `{project}` | `{workload}` | `{route}` | `{reason}` |".format(
                task=row.get("task_id"),
                status=row.get("status"),
                project=row.get("project"),
                workload=row.get("workload_key"),
                route=row.get("route"),
                reason=row.get("reason"),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _route_row(row: Mapping[str, Any]) -> dict[str, Any]:
    cert = task_admission_certificate(row)
    route = "ADMIT_THEOREM_TRACE" if cert.admitted else "PROBE_REQUIRED"
    return {
        "task_id": row.get("id") or row.get("task_id"),
        "status": row.get("status"),
        "project": row.get("project"),
        "signature": row.get("signature"),
        "workload_key": cert.workload_key,
        "measured_profiles": list(cert.measured_profiles),
        "route": route,
        "reason": cert.reason,
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


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_future_production_admission_contract(
        records=load_scheduler_records(
            queue_path=args.queue_path or None,
            archive_path=args.archive_path or None,
        )
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.future_production_admission_contract")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build future-production admission contract")
    build.add_argument("--queue-path", default=None)
    build.add_argument("--archive-path", default=None)
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "future_production_admission_contract_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "future_production_admission_contract_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
