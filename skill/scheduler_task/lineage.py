from __future__ import annotations

from typing import Callable


def task_is_descendant_of(task: dict, ancestor_id: str, by_id: dict) -> bool:
    """Return True if task's parent_id chain reaches ancestor_id."""
    if not ancestor_id:
        return False
    seen = set()
    cur = task
    while cur:
        parent_id = cur.get("parent_id")
        if not parent_id:
            return False
        if parent_id == ancestor_id:
            return True
        if parent_id in seen:
            return False
        seen.add(parent_id)
        cur = by_id.get(parent_id)
    return False


def mark_user_cancelled(
    task: dict,
    reason: str = "user cancel",
    *,
    actor_info: Callable[[str, str], dict],
    now: Callable[[], float],
) -> None:
    ts = now()
    actor = actor_info("cancel", reason)
    task["status"] = "cancelled"
    task["finished_at"] = ts
    task["cancelled_at"] = ts
    task["cancelled_by_user"] = True
    task["cancelled_by"] = actor["label"]
    task["cancel_actor"] = actor
    task["cancel_reason"] = reason
    task.pop("launching_started_at", None)
    task.pop("launch_token", None)
    task["last_block_reason"] = reason


def remember_last_placement(task: dict) -> None:
    """Keep enough placement history to audit/recover stale launch artifacts."""
    if task.get("node"):
        task["last_node"] = task.get("node")
    if task.get("gpu_idx") is not None:
        task["last_gpu_idx"] = task.get("gpu_idx")


def cancel_related_queued_retries(
    state: dict,
    task: dict,
    reason: str,
    *,
    task_run_identity: Callable[[dict], object],
    release_task_claims_and_intents: Callable[[dict], int],
    mark_user_cancelled: Callable[[dict, str], None],
) -> int:
    """Cancel queued/launching retry duplicates for the same exact run."""
    key = task_run_identity(task)
    if not key:
        return 0
    cancelled = 0
    target_id = task.get("id")
    for other in state.get("tasks", []):
        if other is task or other.get("id") == target_id:
            continue
        if other.get("status") not in ("queued", "launching"):
            continue
        if task_run_identity(other) != key:
            continue
        try:
            release_task_claims_and_intents(other)
        except Exception:
            pass
        mark_user_cancelled(
            other,
            f"{reason}; cancelling duplicate retry for same run identity as {target_id}",
        )
        other["node"] = None
        other["gpu_idx"] = None
        other["remote_pids"] = []
        cancelled += 1
    return cancelled


def has_user_cancelled_retry_descendant(
    parent: dict,
    state: dict,
    parent_key=None,
    *,
    task_run_identity: Callable[[dict], object],
) -> bool:
    """True if a retry descendant of parent was cancelled by the operator."""
    parent_id = parent.get("id")
    if not parent_id:
        return False
    key = parent_key if parent_key is not None else task_run_identity(parent)
    if not key:
        return False
    by_id = {task.get("id"): task for task in state.get("tasks", []) if task.get("id")}
    for task in state.get("tasks", []):
        if task.get("status") != "cancelled":
            continue
        if task_run_identity(task) != key:
            continue
        if task.get("parent_id") == parent_id or task_is_descendant_of(task, parent_id, by_id):
            return True
    return False


def reconcile_requeue_lineage_invariants(
    state: dict,
    *,
    release_task_claims_and_intents: Callable[[dict], int],
    mark_user_cancelled: Callable[[dict, str], None],
    now: Callable[[], float],
) -> int:
    """Repair impossible queued parents that have already spawned a retry."""
    by_id = {task.get("id"): task for task in state.get("tasks", []) if task.get("id")}
    changed = 0
    for task in state.get("tasks", []):
        if task.get("status") not in ("queued", "launching"):
            continue
        child_id = task.get("requeued_as")
        if not child_id:
            continue
        child = by_id.get(child_id)
        if not child:
            continue
        try:
            release_task_claims_and_intents(task)
        except Exception:
            pass
        task["node"] = None
        task["gpu_idx"] = None
        task["remote_pids"] = []
        task.pop("launching_started_at", None)
        task.pop("launch_token", None)
        if child.get("status") == "cancelled":
            mark_user_cancelled(
                task,
                f"superseded by retry {child_id}, which was cancelled; parent will not dispatch",
            )
        else:
            task["status"] = "failed"
            task["finished_at"] = task.get("finished_at") or now()
            task["last_block_reason"] = (
                f"superseded by retry {child_id}; parent kept terminal to avoid duplicate dispatch"
            )
        changed += 1
    return changed
