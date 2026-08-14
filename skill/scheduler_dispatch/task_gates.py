"""Per-task pre-placement gates for dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class DispatchTaskGateDeps:
    clear_disallowed_cpu_fallback_selection: Callable[[dict], bool]
    reconcile_queued_launch_artifacts_before_dispatch: Callable[[dict, dict], dict | None]
    task_run_identity: Callable[[dict], object]
    same_run_identity_live_artifact_reason: Callable[[dict, dict, object], str]
    eviction_cooldown_block_reason: Callable[[dict], str]
    queued_wait_for_file_block_reason: Callable[[dict], str]
    queued_cpu_training_block_reason: Callable[[dict], str]


@dataclass
class DispatchTaskGateResult:
    skip_task: bool = False
    events: list = field(default_factory=list)


def _blocked_event(task: dict, reason: str) -> dict:
    return {
        "type": "blocked",
        "task_id": task["id"],
        "task": task,
        "reason": reason,
    }


def apply_dispatch_task_gates(
    task: dict,
    *,
    state: dict,
    running_keys: set,
    deps: DispatchTaskGateDeps,
) -> DispatchTaskGateResult:
    """Apply pre-placement gates for one queued task.

    Returns skip_task=True when the caller should move to the next queued task.
    The function mutates task and running_keys exactly like the legacy loop did.
    """
    events = []
    if deps.clear_disallowed_cpu_fallback_selection(task):
        task["last_block_reason"] = (
            "cleared stale CPU fallback placement; task has vram>0 and "
            "will wait for GPU/local placement"
        )

    artifact_event = deps.reconcile_queued_launch_artifacts_before_dispatch(
        task,
        state,
    )
    if artifact_event:
        events.append(artifact_event)
        if task.get("status") in ("running", "launching"):
            key = deps.task_run_identity(task)
            if key:
                running_keys.add(key)
        return DispatchTaskGateResult(skip_task=True, events=events)

    sig = task.get("signature") or ""
    run_key = deps.task_run_identity(task)
    if run_key and run_key in running_keys:
        reason = (
            "run identity already has a running/launching task; "
            "refusing to dispatch a duplicate. Broad signatures are allowed: "
            f"different cmd/cwd/env/result identities with signature {sig!r} "
            "can still run in parallel."
        )
        task["last_block_reason"] = reason
        events.append(_blocked_event(task, reason))
        return DispatchTaskGateResult(skip_task=True, events=events)

    terminal_artifact_reason = deps.same_run_identity_live_artifact_reason(
        task,
        state,
        run_key,
    )
    if terminal_artifact_reason:
        task["last_block_reason"] = terminal_artifact_reason
        events.append(_blocked_event(task, terminal_artifact_reason))
        return DispatchTaskGateResult(skip_task=True, events=events)

    for reason in (
        deps.eviction_cooldown_block_reason(task),
        deps.queued_wait_for_file_block_reason(task),
        deps.queued_cpu_training_block_reason(task),
    ):
        if reason:
            task["last_block_reason"] = reason
            events.append(_blocked_event(task, reason))
            return DispatchTaskGateResult(skip_task=True, events=events)

    return DispatchTaskGateResult(skip_task=False, events=events)
