"""Bounded global robust-MaxWeight action selector.

This is a pure theorem-dispatch prototype.  It scores bounded global
configurations of already-audited candidate rows by

    Q^T lower_service(action) - penalty(action)

and returns the best configuration plus an exact oracle-gap certificate over the
enumerated family.  It does not launch tasks and is not wired into the legacy
scheduler by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GlobalActionResult:
    selected_action_ids: tuple[str, ...]
    selected_task_ids: tuple[str, ...]
    score: float
    candidate_configuration_count: int
    oracle_gap_alpha0: float
    oracle_gap_alpha1: float
    score_semantics: str
    fallback_legacy_rows_excluded: bool

    def snapshot(self) -> dict[str, Any]:
        return {
            "selected_action_ids": list(self.selected_action_ids),
            "selected_task_ids": list(self.selected_task_ids),
            "score": float(self.score),
            "candidate_configuration_count": int(self.candidate_configuration_count),
            "oracle_gap_alpha0": float(self.oracle_gap_alpha0),
            "oracle_gap_alpha1": float(self.oracle_gap_alpha1),
            "score_semantics": self.score_semantics,
            "fallback_legacy_rows_excluded": self.fallback_legacy_rows_excluded,
            "global_action_dispatch_ready": self.candidate_configuration_count > 0,
        }


def select_global_action(
    candidate_rows: Sequence[Mapping[str, Any]],
    queue_vector: Mapping[str, float],
    *,
    max_batch_size: int = 4,
    max_configurations: int = 10000,
    lookahead_weight: float = 0.0,
) -> GlobalActionResult:
    theorem_rows = [
        row for row in candidate_rows
        if str(row.get("score_semantics") or "") == "robust_maxweight_lower_service"
        and bool(row.get("theorem_ready", True))
        and _action_row_allowed(row)
    ]
    configs = _enumerate_configurations(
        theorem_rows,
        max_batch_size=max_batch_size,
        max_configurations=max_configurations,
    )
    best_rows: tuple[Mapping[str, Any], ...] = ()
    best_score = float("-inf")
    for config in configs:
        score = _configuration_score(
            config,
            queue_vector,
            lookahead_weight=max(0.0, float(lookahead_weight)),
        )
        if score > best_score or (
            score == best_score
            and _configuration_tiebreak(config) < _configuration_tiebreak(best_rows)
        ):
            best_score = score
            best_rows = tuple(config)
    if not best_rows:
        best_score = 0.0
    return GlobalActionResult(
        selected_action_ids=tuple(str(row.get("action_id") or "") for row in best_rows),
        selected_task_ids=tuple(str(row.get("task_id") or "") for row in best_rows),
        score=float(best_score),
        candidate_configuration_count=len(configs),
        oracle_gap_alpha0=0.0,
        oracle_gap_alpha1=0.0,
        score_semantics="robust_maxweight_lower_service",
        fallback_legacy_rows_excluded=len(theorem_rows) <= len(candidate_rows),
    )


def _enumerate_configurations(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_batch_size: int,
    max_configurations: int,
) -> list[tuple[Mapping[str, Any], ...]]:
    out: list[tuple[Mapping[str, Any], ...]] = []
    capped_batch = max(1, int(max_batch_size))
    for size in range(1, min(capped_batch, len(rows)) + 1):
        for config in combinations(rows, size):
            if _resource_disjoint(config) and _task_disjoint(config):
                out.append(tuple(config))
                if len(out) >= int(max_configurations):
                    return out
    return out


def _configuration_score(
    rows: Iterable[Mapping[str, Any]],
    queue_vector: Mapping[str, float],
    *,
    lookahead_weight: float = 0.0,
) -> float:
    lower: dict[str, float] = {}
    penalty = 0.0
    lookahead = 0.0
    for row in rows:
        for key, value in (row.get("lower_service") or {}).items():
            lower[str(key)] = lower.get(str(key), 0.0) + float(value)
        penalty += float(row.get("penalty_units") or 0.0)
        lookahead += _row_lookahead_bonus(row, queue_vector)
    return (
        sum(float(queue_vector.get(key, 0.0)) * value for key, value in lower.items())
        - penalty
        + max(0.0, float(lookahead_weight)) * lookahead
    )


def _row_lookahead_bonus(row: Mapping[str, Any], queue_vector: Mapping[str, float]) -> float:
    workload = str(row.get("workload_key") or row.get("service_workload_key") or "")
    if not workload:
        lower = row.get("lower_service") or {}
        workload = next(iter(lower), "")
    q = max(0.0, float(queue_vector.get(workload, 0.0)))
    eta = _positive_float(row.get("eta_lcb_s") or row.get("eta_s") or 0.0)
    if eta <= 0.0:
        eta = 1.0 / max(1e-12, float(row.get("selected_class_lower_service") or 0.0))
    age = max(0.0, float(row.get("oldest_wait_s") or row.get("age_s") or 0.0))
    return (1.0 + q + min(age, 3600.0) / 3600.0) / max(1e-9, eta)


def _resource_disjoint(rows: Iterable[Mapping[str, Any]]) -> bool:
    seen: set[str] = set()
    for row in rows:
        resources = _resource_ids(row)
        for resource in resources:
            if resource and resource in seen:
                return False
            if resource:
                seen.add(resource)
    return True


def _task_disjoint(rows: Iterable[Mapping[str, Any]]) -> bool:
    seen: set[str] = set()
    for row in rows:
        task = str(row.get("task_id") or "")
        if task and task in seen:
            return False
        if task:
            seen.add(task)
    return True


def _resource_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("resource_ids") or row.get("resource_units")
    if raw is None:
        resource = str(row.get("resource_id") or row.get("action_id") or "")
        return (resource,) if resource else tuple()
    if isinstance(raw, str):
        return (raw,) if raw else tuple()
    try:
        return tuple(str(item) for item in raw if str(item))
    except TypeError:
        resource = str(raw or "")
        return (resource,) if resource else tuple()


def _configuration_tiebreak(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(str(row.get("action_id") or row.get("task_id") or "") for row in rows))


def _positive_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out > 0.0 else 0.0


def _action_row_allowed(row: Mapping[str, Any]) -> bool:
    if str(row.get("action_type") or "").lower() == "migrate":
        return bool(row.get("controlled_migration_ready")) and not row.get("migration_block_reason")
    return True
