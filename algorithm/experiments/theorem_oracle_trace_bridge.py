"""Bridge candidate-family trace logs to theorem-side oracle audits.

The live scheduler trace is only theorem-grade when every slot carries:

* queue_vector Q(k)
* per-candidate lower_service vectors
* per-candidate penalty_units
* selected action
* robust_maxweight_lower_service score semantics

This module refuses scheduler-sort-key-only traces instead of silently treating
them as approximate MaxWeight certificates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.oracle_trace import load_trace_slots

from .oracle_audit import audit_slots


def build_theorem_oracle_audit_from_trace(slots: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    slot_list = [dict(slot) for slot in slots]
    converted = []
    blockers: list[dict[str, Any]] = []
    for slot in slot_list:
        converted_slot, blocker = trace_slot_to_oracle_slot(slot)
        if blocker:
            blockers.append(blocker)
        elif converted_slot:
            converted.append(converted_slot)
    if blockers or not converted:
        return {
            "status": "NOT_THEOREM_TRACE" if blockers else "NO_THEOREM_SLOTS",
            "trace_slot_count": len(slot_list),
            "converted_slot_count": len(converted),
            "blocker_count": len(blockers),
            "blockers": blockers[:100],
            "usable_for_theorem": False,
            "interpretation": (
                "Trace cannot certify alpha0/alpha1 until each slot has robust "
                "MaxWeight lower-service semantics."
            ),
        }
    audit = audit_slots(converted)
    audit.update({
        "status": "THEOREM_ORACLE_PASS" if audit.get("usable_for_theorem") else "THEOREM_ORACLE_FAIL",
        "trace_slot_count": len(converted),
        "converted_slot_count": len(converted),
        "blocker_count": 0,
        "blockers": [],
    })
    return audit


def trace_slot_to_oracle_slot(slot: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    slot_id = slot.get("slot_id")
    semantics = str(slot.get("score_semantics") or "")
    if semantics != "robust_maxweight_lower_service":
        return None, _blocker(slot_id, "score_semantics_not_robust_maxweight_lower_service", semantics)
    queue = slot.get("queue_vector")
    if not isinstance(queue, Mapping) or not queue:
        return None, _blocker(slot_id, "missing_queue_vector", "")
    candidates = list(slot.get("candidates") or [])
    if not candidates:
        return None, _blocker(slot_id, "missing_candidates", "")
    selected = [row for row in candidates if row.get("selected")]
    if len(selected) != 1:
        return None, _blocker(slot_id, "selected_action_not_unique", str(len(selected)))
    actions = []
    for row in candidates:
        action_id = str(row.get("action_id") or "")
        lower = row.get("lower_service") or row.get("lower_service_vector")
        if not action_id:
            return None, _blocker(slot_id, "candidate_missing_action_id", "")
        if not isinstance(lower, Mapping) or not lower:
            return None, _blocker(slot_id, "candidate_missing_lower_service", action_id)
        actions.append({
            "action_id": action_id,
            "lower_service": dict(lower),
            "penalty_units": _as_float(row.get("penalty_units"), 0.0),
        })
    return {
        "slot_id": slot_id,
        "queue_vector": dict(queue),
        "chosen_action_id": str(selected[0].get("action_id") or ""),
        "candidate_actions": actions,
        "best_score_exact_over_candidate_set": bool(
            slot.get("best_score_exact_over_candidate_set", True)
        ),
    }, None


def audit_trace_file(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    slots = load_trace_slots(p) if p.exists() else []
    if not p.exists():
        return {
            "status": "NO_TRACE",
            "input": str(p),
            "trace_slot_count": 0,
            "converted_slot_count": 0,
            "blocker_count": 1,
            "blockers": [{"slot_id": None, "reason": "trace_file_does_not_exist"}],
            "usable_for_theorem": False,
        }
    report = build_theorem_oracle_audit_from_trace(slots)
    report["input"] = str(p)
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Module52 Theorem Oracle Trace Bridge",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `status` | `{report.get('status')}` |",
        f"| `trace_slot_count` | {report.get('trace_slot_count', 0)} |",
        f"| `converted_slot_count` | {report.get('converted_slot_count', 0)} |",
        f"| `blocker_count` | {report.get('blocker_count', 0)} |",
        f"| `alpha0` | {_fmt(report.get('alpha0'))} |",
        f"| `alpha1` | {_fmt(report.get('alpha1'))} |",
        f"| `usable_for_theorem` | {str(bool(report.get('usable_for_theorem'))).lower()} |",
        "",
    ]
    blockers = list(report.get("blockers") or [])
    if blockers:
        lines.extend([
            "## Blockers",
            "",
            "| Slot | Reason | Detail |",
            "|---|---|---|",
        ])
        for row in blockers[:50]:
            lines.append(
                "| `{slot}` | `{reason}` | `{detail}` |".format(
                    slot=row.get("slot_id"),
                    reason=row.get("reason"),
                    detail=row.get("detail", ""),
                )
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "This bridge only accepts robust MaxWeight lower-service trace slots.  A",
        "scheduler-sort-key trace can pass Module50 and still fail here; that is",
        "the intended separation between implementation audit and theorem",
        "alpha0/alpha1 calibration.",
        "",
    ])
    return "\n".join(lines)


def _blocker(slot_id: Any, reason: str, detail: Any) -> dict[str, Any]:
    return {"slot_id": slot_id, "reason": reason, "detail": str(detail)}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
    return 0 if report.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.theorem_oracle_trace_bridge")
    sub = p.add_subparsers(dest="cmd", required=True)
    audit = sub.add_parser("audit", help="Convert theorem-grade trace to alpha0/alpha1 audit")
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
