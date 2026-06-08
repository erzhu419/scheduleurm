"""Audit scheduler candidate-set oracle trace logs.

This CLI reads JSONL slots emitted by ``algorithm.oracle_trace`` and reports
whether the selected placement is the best candidate under the scheduler's
recorded sort key.  It intentionally does not convert scheduler sort-key audits
into theorem-grade MaxWeight alpha certificates unless the trace explicitly
contains robust lower-service semantics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.oracle_trace import audit_scheduler_score_trace, load_trace_slots


def audit_trace_file(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    slots = load_trace_slots(p) if p.exists() else []
    report = audit_scheduler_score_trace(slots)
    report.update({
        "input": str(p),
        "trace_slot_count": len(slots),
        "status": _status(report, p.exists()),
    })
    if not p.exists():
        report["theorem_blocker"] = "trace file does not exist"
    elif not slots:
        report["theorem_blocker"] = "trace file exists but contains no slots"
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Scheduler Oracle Trace Audit",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `status` | `{report.get('status')}` |",
        f"| `trace_slot_count` | {report.get('trace_slot_count', 0)} |",
        f"| `audited_slot_count` | {report.get('audited_slot_count', 0)} |",
        f"| `max_scheduler_score_gap` | {_fmt(report.get('max_scheduler_score_gap'))} |",
        f"| `usable_for_scheduler_score_audit` | {str(bool(report.get('usable_for_scheduler_score_audit'))).lower()} |",
        f"| `usable_for_theorem` | {str(bool(report.get('usable_for_theorem'))).lower()} |",
        "",
    ]
    blocker = str(report.get("theorem_blocker") or "")
    if blocker:
        lines.extend(["## Theorem Blocker", "", blocker, ""])
    rows = list(report.get("rows") or [])
    if rows:
        lines.extend([
            "## Audited Slots",
            "",
            "| Slot | Candidates | Selected | Best | Gap | Semantics |",
            "|---|---:|---|---|---:|---|",
        ])
        for row in rows[:50]:
            lines.append(
                "| `{slot}` | {count} | `{selected}` | `{best}` | {gap} | `{sem}` |".format(
                    slot=row.get("slot_id"),
                    count=row.get("candidate_count"),
                    selected=row.get("selected_action_id"),
                    best=row.get("best_action_id"),
                    gap=_fmt(row.get("scheduler_score_gap")),
                    sem=row.get("score_semantics"),
                )
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "A PASS here certifies that the live scheduler chose the best action under",
        "the recorded scheduler sort key for each traced candidate family.  It is",
        "not by itself a robust MaxWeight theorem certificate unless the trace",
        "rows carry lower-service vectors and declare",
        "`robust_maxweight_lower_service` semantics.",
        "",
    ])
    return "\n".join(lines)


def _status(report: Mapping[str, Any], exists: bool) -> str:
    if not exists:
        return "NO_TRACE"
    if not report.get("audited_slot_count"):
        return "EMPTY_TRACE"
    if report.get("usable_for_theorem"):
        return "THEOREM_PASS"
    if report.get("usable_for_scheduler_score_audit"):
        return "SCHEDULER_SCORE_PASS"
    return "FAIL"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return "NA"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_audit(args: argparse.Namespace) -> int:
    report = audit_trace_file(args.input)
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("usable_for_scheduler_score_audit") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.scheduler_oracle_trace_audit")
    sub = p.add_subparsers(dest="cmd", required=True)
    audit = sub.add_parser("audit", help="Audit a scheduler oracle trace JSONL file")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--markdown-output", default="")
    audit.set_defaults(func=_cmd_audit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
