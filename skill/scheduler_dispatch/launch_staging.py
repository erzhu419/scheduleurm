"""Launch-staging decision helpers for dispatch."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DispatchLaunchStagingDeps:
    max_launch_retry: int
    launch_max_cwd_size_mb: int
    stage_failure_reason: Callable[[str, str], str | None]
    release_task_claims_and_intents: Callable[..., None]
    write_escalation: Callable[[dict, str, dict], None]
    now: Callable[[], float] = _time.time


@dataclass(frozen=True)
class DispatchLaunchStagingResult:
    blocked: bool
    event: dict | None = None


def _clear_launch_selection(task: dict) -> None:
    task["node"] = None
    task["gpu_idx"] = None
    task.pop("launching_started_at", None)
    task.pop("launch_token", None)


def apply_launch_staging_gate(
    task: dict,
    *,
    target: str,
    cwd_for_stage: str,
    stage_state: str,
    deps: DispatchLaunchStagingDeps,
) -> DispatchLaunchStagingResult:
    """Apply the dispatch action for a launch-staging cache state.

    Returns blocked=False only when dispatch should continue into the actual
    launch path. All blocked states mutate task exactly as the legacy dispatch
    branch did and return the event that must be appended to the dispatch log.
    """
    if stage_state == "ready":
        return DispatchLaunchStagingResult(blocked=False)

    if stage_state == "cap_exceeded":
        allowed = {
            str(node)
            for node in (task.get("allowed_nodes") or [])
            if str(node)
        }
        can_run_local = not allowed or "local" in allowed
        task["status"] = "queued" if can_run_local else "failed"
        if can_run_local:
            task["require_node"] = "local"
            resolution = "pinned require_node=local"
        else:
            task["require_node"] = None
            resolution = (
                "local is outside explicit allowed_nodes; refusing an "
                "inconsistent automatic pin"
            )
        task["last_block_reason"] = (
            f"launch staging: cwd > {deps.launch_max_cwd_size_mb}MB cap "
            f"for {target}; {resolution}"
        )
        deps.release_task_claims_and_intents(task, extra_nodes=[target])
        _clear_launch_selection(task)
        return DispatchLaunchStagingResult(
            blocked=True,
            event={
                "type": (
                    "launch_capped" if can_run_local
                    else "launch_capped_no_allowed_fallback"
                ),
                "task_id": task["id"],
                "task": task,
                "reason": task["last_block_reason"],
            },
        )

    if stage_state == "unsafe_path":
        allowed = {str(node) for node in (task.get("allowed_nodes") or []) if str(node)}
        can_run_local = not allowed or "local" in allowed
        task["status"] = "queued" if can_run_local else "failed"
        if can_run_local:
            task["require_node"] = "local"
        task["last_block_reason"] = (
            f"launch staging safety: cwd {cwd_for_stage} is a protected system/root "
            + ("path; pinned require_node=local" if can_run_local else "path and local is not allowed")
        )
        deps.release_task_claims_and_intents(task, extra_nodes=[target])
        _clear_launch_selection(task)
        return DispatchLaunchStagingResult(
            blocked=True,
            event={
                "type": "launch_unsafe_cwd",
                "task_id": task["id"],
                "task": task,
                "reason": task["last_block_reason"],
            },
        )

    if stage_state == "stage_failed":
        fail_msg = (
            deps.stage_failure_reason(target, cwd_for_stage)
            or "rsync to target failed"
        )
        attempted = target
        task["launch_fail_count"] = (task.get("launch_fail_count") or 0) + 1
        failed = task.setdefault("launch_failed_nodes", {})
        if not isinstance(failed, dict):
            failed = {}
            task["launch_failed_nodes"] = failed
        failed[attempted] = {
            "ts": deps.now(),
            "attempt": task["launch_fail_count"],
            "error": f"stage_cwd: {fail_msg[:280]}",
        }
        task["last_block_reason"] = (
            f"launch stage attempt {task['launch_fail_count']}/{deps.max_launch_retry}: "
            f"rsync to {target} failed: {fail_msg[:200]}"
        )
        deps.release_task_claims_and_intents(task, extra_nodes=[target])
        _clear_launch_selection(task)
        if task["launch_fail_count"] >= deps.max_launch_retry:
            task["status"] = "failed"
            try:
                deps.write_escalation(
                    task,
                    "LAUNCH_FAIL_CAP",
                    {"reason": fail_msg, "tail": fail_msg},
                )
            except Exception:
                pass
            return DispatchLaunchStagingResult(
                blocked=True,
                event={
                    "type": "launch_failed_terminal",
                    "task_id": task["id"],
                    "task": task,
                    "error": fail_msg,
                },
            )

        task["status"] = "queued"
        return DispatchLaunchStagingResult(
            blocked=True,
            event={
                "type": "launch_failed_retry",
                "task_id": task["id"],
                "task": task,
                "error": fail_msg,
            },
        )

    if stage_state == "needs_stage":
        task["status"] = "queued"
        task["last_block_reason"] = (
            f"launch staging: cwd not yet staged to {target}; "
            f"will retry next dispatch cycle"
        )
        deps.release_task_claims_and_intents(task, extra_nodes=[target])
        _clear_launch_selection(task)
        return DispatchLaunchStagingResult(
            blocked=True,
            event={
                "type": "launch_stage_deferred",
                "task_id": task["id"],
                "task": task,
                "reason": f"awaiting outside-lock staging to {target}",
            },
        )

    return DispatchLaunchStagingResult(blocked=False)


def apply_launch_input_staging_gate(
    task: dict,
    *,
    target: str,
    stage_state: str,
    deps: DispatchLaunchStagingDeps,
) -> DispatchLaunchStagingResult:
    """Defer launch until its declared immutable inputs reach the target."""
    if stage_state == "ready":
        task.pop("stage_input_target", None)
        return DispatchLaunchStagingResult(blocked=False)

    task["status"] = "queued"
    task["stage_input_target"] = target
    if stage_state == "missing_input":
        reason = "launch input staging: declared local input directory is missing"
        event_type = "launch_input_missing"
    else:
        reason = f"launch input staging: inputs not yet staged to {target}; will retry next dispatch cycle"
        event_type = "launch_input_stage_deferred"
    task["last_block_reason"] = reason
    deps.release_task_claims_and_intents(task, extra_nodes=[target])
    _clear_launch_selection(task)
    return DispatchLaunchStagingResult(
        blocked=True,
        event={
            "type": event_type,
            "task_id": task["id"],
            "task": task,
            "reason": reason,
        },
    )
