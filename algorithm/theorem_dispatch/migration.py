"""Controlled migration action rows for theorem-facing dispatch.

This module only models migration candidates.  It never moves a real process.
Rows are eligible for the global robust-MaxWeight selector only when they come
from controlled benchmark tasks with verified checkpoint/resume metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MigrationCost:
    checkpoint_flush_s: float = 0.0
    sync_s: float = 0.0
    environment_staging_s: float = 0.0
    resume_warmup_s: float = 0.0
    lost_work_s: float = 0.0
    risk_penalty_units: float = 0.0

    @property
    def total_time_s(self) -> float:
        return (
            max(0.0, float(self.checkpoint_flush_s))
            + max(0.0, float(self.sync_s))
            + max(0.0, float(self.environment_staging_s))
            + max(0.0, float(self.resume_warmup_s))
            + max(0.0, float(self.lost_work_s))
        )

    @property
    def total_penalty_units(self) -> float:
        return self.total_time_s + max(0.0, float(self.risk_penalty_units))

    def snapshot(self) -> dict[str, float]:
        return {
            "checkpoint_flush_s": float(self.checkpoint_flush_s),
            "sync_s": float(self.sync_s),
            "environment_staging_s": float(self.environment_staging_s),
            "resume_warmup_s": float(self.resume_warmup_s),
            "lost_work_s": float(self.lost_work_s),
            "risk_penalty_units": float(self.risk_penalty_units),
            "total_time_s": float(self.total_time_s),
            "total_penalty_units": float(self.total_penalty_units),
        }


def migration_is_beneficial(
    *,
    remaining_work: float,
    current_rate: float,
    target_rate: float,
    cost: MigrationCost,
) -> bool:
    return migration_time_savings(
        remaining_work=remaining_work,
        current_rate=current_rate,
        target_rate=target_rate,
    ) > cost.total_penalty_units


def migration_time_savings(
    *,
    remaining_work: float,
    current_rate: float,
    target_rate: float,
) -> float:
    """Return completion-time reduction before migration overhead, in seconds.

    Rates are work units per second, so the dimensionally valid comparison with
    checkpoint/sync/resume time is ``W / mu_old - W / mu_new``.  A nonpositive
    or non-improving target rate cannot provide a migration benefit.
    """

    work = max(0.0, float(remaining_work))
    old_rate = float(current_rate)
    new_rate = float(target_rate)
    if work <= 0.0 or old_rate <= 0.0 or new_rate <= old_rate:
        return 0.0
    return work / old_rate - work / new_rate


def controlled_migration_allowed(task: Mapping[str, Any]) -> tuple[bool, str]:
    if not _truthy(task.get("controlled_benchmark") or task.get("allow_controlled_migration")):
        return False, "not_controlled_benchmark"
    if _truthy(task.get("external_user_task") or task.get("auto_adopted") or task.get("production_running")):
        return False, "ordinary_running_task_not_touchable"
    if not _truthy(task.get("checkpoint_verified") or task.get("checkpoint_command")):
        return False, "checkpoint_missing"
    if not _truthy(task.get("resume_verified") or task.get("resume_command")):
        return False, "resume_missing"
    return True, ""


def build_migration_action_row(
    *,
    task: Mapping[str, Any],
    workload_key: str,
    from_node: str,
    to_node: str,
    current_rate: float,
    target_lower_service: float,
    remaining_work: float,
    progress_fraction: float,
    cost: MigrationCost,
) -> dict[str, Any]:
    allowed, reason = controlled_migration_allowed(task)
    beneficial = migration_is_beneficial(
        remaining_work=remaining_work,
        current_rate=current_rate,
        target_rate=target_lower_service,
        cost=cost,
    )
    time_savings_s = migration_time_savings(
        remaining_work=remaining_work,
        current_rate=current_rate,
        target_rate=target_lower_service,
    )
    task_id = str(task.get("id") or task.get("task_id") or task.get("signature") or "task")
    action_id = (
        f"migrate|task={task_id}|from={from_node}|to={to_node}|"
        f"p={float(progress_fraction):.2f}"
    )
    theorem_ready = bool(allowed and beneficial)
    return {
        "action_type": "migrate",
        "task_id": task_id,
        "action_id": action_id,
        "resource_id": f"migration:{task_id}:{from_node}->{to_node}",
        "resource_ids": (
            f"task:{task_id}",
            f"node:{from_node}",
            f"node:{to_node}",
        ),
        "workload_key": workload_key,
        "service_workload_key": workload_key,
        "lower_service": {str(workload_key): max(0.0, float(target_lower_service))} if theorem_ready else {},
        "selected_class_lower_service": max(0.0, float(target_lower_service)) if theorem_ready else 0.0,
        "penalty_units": float(cost.total_penalty_units),
        "migration_cost": cost.snapshot(),
        "progress_fraction": float(progress_fraction),
        "remaining_work": float(remaining_work),
        "current_rate": float(current_rate),
        "target_rate": float(target_lower_service),
        "keep_remaining_completion_s": (
            max(0.0, float(remaining_work)) / float(current_rate)
            if float(current_rate) > 0.0
            else None
        ),
        "migrated_remaining_completion_s_before_cost": (
            max(0.0, float(remaining_work)) / float(target_lower_service)
            if float(target_lower_service) > 0.0
            else None
        ),
        "migration_time_savings_before_cost_s": float(time_savings_s),
        "migration_net_time_savings_s": float(
            time_savings_s - cost.total_penalty_units
        ),
        "benefit_test": (
            "remaining_work/current_rate - remaining_work/target_rate "
            "> migration_total_penalty_seconds"
        ),
        "beneficial_by_threshold": bool(beneficial),
        "controlled_migration_ready": bool(allowed),
        "migration_block_reason": reason if not allowed else ("" if beneficial else "migration_cost_not_recovered"),
        "score_semantics": "robust_maxweight_lower_service",
        "theorem_ready": theorem_ready,
        "scheduler_hint_only": True,
    }


def build_keep_action_row(
    *,
    task: Mapping[str, Any],
    workload_key: str,
    current_node: str,
    current_lower_service: float,
) -> dict[str, Any]:
    task_id = str(task.get("id") or task.get("task_id") or task.get("signature") or "task")
    return {
        "action_type": "keep",
        "task_id": task_id,
        "action_id": f"keep|task={task_id}|node={current_node}",
        "resource_id": f"task:{task_id}",
        "resource_ids": (f"task:{task_id}",),
        "workload_key": workload_key,
        "service_workload_key": workload_key,
        "lower_service": {str(workload_key): max(0.0, float(current_lower_service))},
        "selected_class_lower_service": max(0.0, float(current_lower_service)),
        "penalty_units": 0.0,
        "score_semantics": "robust_maxweight_lower_service",
        "theorem_ready": True,
        "scheduler_hint_only": True,
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "verified"}
    return bool(value)
