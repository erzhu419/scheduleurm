"""Experiment-side action-family reconstruction.

The top-level `algorithm.action_model` defines the serializable action schema.
This file validates and groups logged action rows into the per-slot full and
candidate families used by the statewise theorem.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _json_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_action_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    errors: list[str] = []
    if not row.get("action_id"):
        errors.append("missing action_id")
    assignments = row.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        errors.append("missing assignments")
    else:
        for idx, assignment in enumerate(assignments):
            if not isinstance(assignment, Mapping):
                errors.append(f"assignment {idx} is not an object")
                continue
            if not assignment.get("task_id"):
                errors.append(f"assignment {idx} missing task_id")
            if not assignment.get("class_key"):
                errors.append(f"assignment {idx} missing class_key")
            if not assignment.get("candidate_bucket"):
                errors.append(f"assignment {idx} missing candidate_bucket")
    return {
        "action_id": row.get("action_id"),
        "valid": not errors,
        "errors": errors,
    }


def candidate_families_by_slot(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"full_actions": [], "candidate_actions": [], "chosen_action_id": "", "chosen_count": 0})
    for row in rows:
        slot_id = str(row.get("slot_id") or "")
        if not slot_id:
            continue
        family = grouped[slot_id]
        source = str(row.get("candidate_source") or row.get("source") or "")
        is_full = source in (
            "exact_full",
            "full_exact",
            "sampled_full",
            "full_sampled",
            "historical_full",
            "all_feasible",
            "statewise_uniform",
        ) or row.get("full_action")
        if is_full:
            family["full_actions"].append(dict(row))
        if row.get("candidate", True):
            family["candidate_actions"].append(dict(row))
        if row.get("chosen"):
            family["chosen_action_id"] = str(row.get("action_id") or "")
            family["chosen_count"] += 1
    return dict(grouped)


def chosen_action_for_slot(rows: Iterable[Mapping[str, Any]], slot_id: str) -> Dict[str, Any]:
    matches = [
        dict(row) for row in rows
        if str(row.get("slot_id") or "") == str(slot_id) and row.get("chosen")
    ]
    if len(matches) != 1:
        raise ValueError(f"slot {slot_id!r} must have exactly one chosen action, got {len(matches)}")
    validation = validate_action_row(matches[0])
    if not validation["valid"]:
        raise ValueError(f"chosen action is invalid: {validation['errors']}")
    return matches[0]


def build_action_family_report(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    materialized = [dict(row) for row in rows]
    row_errors = []
    for idx, row in enumerate(materialized):
        validation = validate_action_row(row)
        if not validation["valid"]:
            row_errors.append({
                "row_index": idx,
                "action_id": validation["action_id"],
                "errors": validation["errors"],
            })
    families = candidate_families_by_slot(materialized)
    slot_errors = []
    for slot_id, family in sorted(families.items()):
        chosen = family.get("chosen_action_id") or ""
        candidate_ids = {
            str(action.get("action_id") or action.get("id") or "")
            for action in family.get("candidate_actions") or []
        }
        if not family.get("full_actions"):
            slot_errors.append({"slot_id": slot_id, "error": "missing_full_action_family"})
        if not family.get("candidate_actions"):
            slot_errors.append({"slot_id": slot_id, "error": "missing_candidate_action_family"})
        if int(family.get("chosen_count") or 0) != 1:
            slot_errors.append({
                "slot_id": slot_id,
                "error": "chosen_action_count_not_one",
                "chosen_count": int(family.get("chosen_count") or 0),
            })
        elif chosen not in candidate_ids:
            slot_errors.append({
                "slot_id": slot_id,
                "error": "chosen_action_not_in_candidate_family",
                "chosen_action_id": chosen,
            })
    valid = bool(materialized) and not row_errors and not slot_errors
    return {
        "valid": valid,
        "action_row_count": len(materialized),
        "slot_count": len(families),
        "row_error_count": len(row_errors),
        "slot_error_count": len(slot_errors),
        "row_errors": row_errors,
        "slot_errors": slot_errors,
        "slots": families,
        "usable_for_theorem": valid,
    }


def _cmd_build(args: argparse.Namespace) -> int:
    rows = _json_rows(Path(args.input).expanduser())
    out = build_action_family_report(rows)
    _write_json(Path(args.output).expanduser(), out)
    print(args.output)
    return 0 if out.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.action_model")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build", help="Validate and group statewise action-family rows")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
