"""Candidate-set oracle trace hooks for scheduleurm.

The scheduler can only certify an approximate-oracle error when a decision
record contains the full candidate family, not only the selected placement.
This module keeps that trace format outside scheduler.py.  It is disabled unless
``SCHEDULEURM_ORACLE_AUDIT_LOG`` or ``SCHEDULEURM_ORACLE_TRACE_PATH`` is set.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


TRACE_ENV_NAMES = ("SCHEDULEURM_ORACLE_AUDIT_LOG", "SCHEDULEURM_ORACLE_TRACE_PATH")


def trace_path(env: Mapping[str, str] | None = None) -> str:
    env = env or os.environ
    for name in TRACE_ENV_NAMES:
        raw = str(env.get(name) or "").strip()
        if raw:
            return raw
    return ""


def trace_enabled(env: Mapping[str, str] | None = None) -> bool:
    return bool(trace_path(env))


def make_slot_id(task: Mapping[str, Any], *, now_ts: float | None = None) -> str:
    now = now_ts if now_ts is not None else time.time()
    task_id = task.get("id") or task.get("task_id") or task.get("signature") or "unknown"
    submitted = task.get("submitted_at", "")
    return f"slot:{task_id}:{submitted}:{int(float(now) * 1000)}"


def action_id(node: Any, gpu_idx: Any) -> str:
    if gpu_idx is None:
        return f"node={node}|cpu"
    return f"node={node}|gpu={gpu_idx}"


def normalize_score(score: Any) -> list[Any]:
    if isinstance(score, tuple):
        values = list(score)
    elif isinstance(score, list):
        values = list(score)
    else:
        values = [score]
    return [_jsonable(v) for v in values]


def primary_numeric_score(score: Any) -> float | None:
    values = normalize_score(score)
    if not values:
        return None
    try:
        out = float(values[0])
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def build_candidate_row(
    *,
    node: Any,
    gpu_idx: Any,
    score: Any,
    selected: bool,
    algorithm_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = dict(algorithm_audit or {})
    return {
        "action_id": action_id(node, gpu_idx),
        "node": node,
        "gpu_idx": gpu_idx,
        "selected": bool(selected),
        "score_sort_key": normalize_score(score),
        "primary_numeric_score": primary_numeric_score(score),
        "algorithm_audit": _jsonable(audit),
        "candidate_bucket": audit.get("candidate_bucket"),
        "class_key": audit.get("class_key"),
        "regime_key": audit.get("regime_key"),
    }


def build_decision_slot(
    *,
    slot_id: str,
    task: Mapping[str, Any],
    algorithm: str,
    phase: str,
    candidates: Iterable[Mapping[str, Any]],
    score_semantics: str = "scheduler_sort_key_minimization",
    hard_constraints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in candidates]
    selected = [row for row in rows if row.get("selected")]
    return {
        "slot_id": slot_id,
        "ts": time.time(),
        "task_id": task.get("id") or task.get("task_id"),
        "project": task.get("project"),
        "signature": task.get("signature"),
        "priority": task.get("priority"),
        "algorithm": algorithm,
        "phase": phase,
        "score_semantics": score_semantics,
        "candidate_count": len(rows),
        "selected_action_id": selected[0].get("action_id") if selected else None,
        "hard_constraints": dict(hard_constraints or {}),
        "candidates": rows,
        "theorem_semantics_available": score_semantics == "robust_maxweight_lower_service",
    }


def log_decision_slot(slot: Mapping[str, Any], path: str | Path | None = None) -> bool:
    target = str(path or trace_path() or "").strip()
    if not target:
        return False
    p = Path(target).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(dict(slot)), sort_keys=True) + "\n")
    return True


def load_trace_slots(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser()
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [dict(row) for row in payload if isinstance(row, Mapping)]
            if isinstance(payload, Mapping) and "candidates" in payload:
                return [dict(payload)]
            return []
        except json.JSONDecodeError:
            pass
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def audit_scheduler_score_trace(slots: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit that the selected action is best under scheduler's recorded sort key.

    This is an engineering audit, not a theorem-grade MaxWeight certificate,
    unless a slot explicitly declares robust_maxweight_lower_service semantics.
    """
    rows = []
    theorem_ready = True
    for slot in slots:
        candidates = list(slot.get("candidates") or [])
        if not candidates:
            continue
        scored = [
            row for row in candidates
            if row.get("primary_numeric_score") is not None
        ]
        if not scored:
            continue
        best = min(scored, key=lambda row: float(row["primary_numeric_score"]))
        selected = next((row for row in scored if row.get("selected")), None)
        if selected is None:
            continue
        gap = max(
            0.0,
            float(selected["primary_numeric_score"]) - float(best["primary_numeric_score"]),
        )
        semantics = str(slot.get("score_semantics") or "")
        theorem_ready = theorem_ready and semantics == "robust_maxweight_lower_service"
        rows.append({
            "slot_id": slot.get("slot_id"),
            "task_id": slot.get("task_id"),
            "algorithm": slot.get("algorithm"),
            "phase": slot.get("phase"),
            "score_semantics": semantics,
            "candidate_count": len(scored),
            "selected_action_id": selected.get("action_id"),
            "best_action_id": best.get("action_id"),
            "selected_primary_score": float(selected["primary_numeric_score"]),
            "best_primary_score": float(best["primary_numeric_score"]),
            "scheduler_score_gap": gap,
        })
    max_gap = max((row["scheduler_score_gap"] for row in rows), default=0.0)
    return {
        "audited_slot_count": len(rows),
        "max_scheduler_score_gap": max_gap,
        "usable_for_scheduler_score_audit": bool(rows) and max_gap <= 1e-9,
        "usable_for_theorem": bool(rows) and theorem_ready and max_gap <= 1e-9,
        "theorem_blocker": (
            "" if theorem_ready else
            "trace uses scheduler sort-key semantics, not robust MaxWeight lower-service semantics"
        ),
        "rows": rows,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return str(value)
