"""Approximate robust MaxWeight oracle audit.

For each slot the theorem-side score is

    score(a; Q) = Q^T lower(a) - penalty(a).

The selected action may be produced by a greedy/local-search scheduler.  This
module measures the finite candidate-set oracle gap and fits constants

    oracle_gap(k) <= alpha0 + alpha1 * ||Q(k)||_1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .penalty_fit import fit_nonnegative_linear_envelope


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


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


def l1_norm(vec: Mapping[str, Any]) -> float:
    return sum(max(0.0, _as_float(v)) for v in vec.values())


def dot_nonnegative(queue: Mapping[str, Any], service: Mapping[str, Any]) -> float:
    total = 0.0
    for cls, q in queue.items():
        total += max(0.0, _as_float(q)) * _as_float(service.get(cls), 0.0)
    return total


def theorem_score(
    queue: Mapping[str, Any],
    lower_service: Mapping[str, Any],
    penalty: float = 0.0,
) -> float:
    return dot_nonnegative(queue, lower_service) - max(0.0, _as_float(penalty))


def action_lower_service(action: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("lower_service", "lower_service_vector", "service_lower", "service_vector"):
        value = action.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def action_penalty(action: Mapping[str, Any]) -> float:
    return _as_float(action.get("penalty_units", action.get("penalty", 0.0)))


def audit_slot(slot: Mapping[str, Any]) -> Dict[str, Any]:
    queue = slot.get("queue_vector") or slot.get("queue") or {}
    if not isinstance(queue, Mapping):
        raise ValueError("slot queue_vector must be a mapping")
    actions = list(slot.get("candidate_actions") or slot.get("actions") or [])
    if not actions:
        raise ValueError("slot has no candidate actions")

    chosen_id = str(slot.get("chosen_action_id") or slot.get("chosen_id") or "")
    scored: list[Dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        action_id = str(action.get("action_id") or action.get("id") or "")
        score = theorem_score(queue, action_lower_service(action), action_penalty(action))
        scored.append({
            "action_id": action_id,
            "score": score,
            "penalty_units": action_penalty(action),
        })
    if not scored:
        raise ValueError("slot has no scorable candidate actions")

    best = max(scored, key=lambda row: row["score"])
    chosen = next((row for row in scored if row["action_id"] == chosen_id), None)
    if chosen is None:
        raise ValueError(f"chosen action {chosen_id!r} not present in candidate set")

    gap = max(0.0, best["score"] - chosen["score"])
    q_norm = l1_norm(queue)
    return {
        "slot_id": slot.get("slot_id"),
        "q_norm": q_norm,
        "chosen_action_id": chosen_id,
        "best_action_id": best["action_id"],
        "chosen_score": chosen["score"],
        "best_candidate_score": best["score"],
        "oracle_gap": gap,
        "candidate_count": len(scored),
        "best_score_exact_over_candidate_set": bool(
            slot.get("best_score_exact_over_candidate_set", True)),
    }


def audit_slots(slots: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [audit_slot(slot) for slot in slots]
    exact = all(row["best_score_exact_over_candidate_set"] for row in rows)
    env = fit_nonnegative_linear_envelope(
        ((row["q_norm"], row["oracle_gap"]) for row in rows),
        intercept_name="alpha0",
        slope_name="alpha1",
    )
    env.update({
        "audited_slot_count": len(rows),
        "best_score_exact_over_candidate_set": exact,
        "usable_for_theorem": bool(rows) and exact and env["max_residual_violation"] <= 1e-9,
        "rows": rows,
    })
    return env


def _cmd_audit(args: argparse.Namespace) -> int:
    slots = _json_rows(Path(args.input).expanduser())
    out = audit_slots(slots)
    _write_json(Path(args.output).expanduser(), out)
    print(args.output)
    return 0 if out.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.oracle_audit")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit", help="Audit approximate oracle gaps")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=_cmd_audit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
