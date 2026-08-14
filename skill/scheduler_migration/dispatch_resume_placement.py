"""Resume-aware placement decisions for dispatch."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class DispatchResumePlacementDeps:
    task_requires_resume_scan: Callable[[dict], bool]
    cached_resume_locations_for_task: Callable[[dict, list], tuple[bool, list, dict]]
    allow_initial_resume_scan_error: Callable[[dict], bool]
    resume_checkpoint_stage_check: Callable[[dict, str, list], tuple[str, dict | None, str]]
    record_staged_resume_location: Callable[[dict, str, dict], None]
    checkpoint_migration_extra_allowed_nodes: Callable[[dict, list, list, dict], tuple[list, dict | None]]
    soft_require_nodes: Callable[[dict], list]
    release_task_claims_and_intents: Callable[..., Any]
    now: Callable[[], float] = _time.time


@dataclass(frozen=True)
class ResumePlacementResult:
    placement: tuple | None = None
    resume_locations: list = field(default_factory=list)
    resume_nodes: set = field(default_factory=set)
    extra_allowed_nodes: list = field(default_factory=list)
    skip_task: bool = False


def _blocked_event(task: dict, reason: str) -> dict:
    task["last_block_reason"] = reason
    return {"type": "blocked", "task_id": task["id"], "task": task, "reason": reason}


def _resume_nodes(resume_locations: list) -> set:
    return {loc.get("node") for loc in (resume_locations or []) if loc.get("node")}


def _scan_error_reason(resume_errors: dict) -> str:
    return (
        "resume checkpoint scan failed on all checked nodes; refusing to dispatch "
        "because a checkpoint may exist on an unchecked server: "
        + "; ".join(f"{n}: {msg}" for n, msg in sorted(resume_errors.items())[:3])
    )


def _scan_pending_reason(task: dict) -> str:
    inflight = task.get("resume_scan_inflight") or {}
    if inflight.get("token"):
        return (
            "resume checkpoint scan is running outside the scheduler state lock; "
            "will retry placement after the completed result is committed"
        )
    return (
        "resume checkpoint scan is queued for the outside-lock prefetch worker; "
        "will retry placement in the next dispatch pass"
    )


def _require_node_missing_ckpt_reason(
    stage_state: str,
    *,
    resume_nodes: set,
    require_node: str,
    stage_msg: str,
) -> str:
    nodes_text = ",".join(sorted(resume_nodes))
    if stage_state == "needs_stage":
        return (
            f"resume checkpoint exists on {nodes_text}, "
            f"but require_node={require_node} has no matching checkpoint yet; "
            "awaiting small-checkpoint staging before remote launch"
        )
    if stage_state == "cap_exceeded":
        return (
            f"resume checkpoint exists on {nodes_text}, "
            f"but require_node={require_node} has no checkpoint and ckpt_dir "
            "is too large for automatic staging; keep on checkpoint-local node"
        )
    if stage_state == "stage_failed":
        return (
            f"resume checkpoint staging to require_node={require_node} failed: "
            f"{stage_msg[:220]}"
        )
    return (
        f"resume checkpoint exists on {nodes_text}, "
        f"but require_node={require_node} has no matching checkpoint; "
        "refusing to launch there because it would likely restart from step 0"
    )


def _selected_node_missing_ckpt_reason(
    stage_state: str,
    *,
    resume_nodes: set,
    selected_node: str,
    stage_msg: str,
) -> str:
    nodes_text = ",".join(sorted(resume_nodes))
    if stage_state == "needs_stage":
        return (
            f"resume checkpoint exists on {nodes_text}, "
            f"but selected node {selected_node} has no matching checkpoint yet; "
            "awaiting small-checkpoint staging before remote launch"
        )
    if stage_state == "cap_exceeded":
        return (
            f"resume checkpoint exists on {nodes_text}, "
            f"but selected node {selected_node} lacks it and ckpt_dir is too "
            "large for automatic staging; waiting for checkpoint-local capacity"
        )
    if stage_state == "stage_failed":
        return (
            f"resume checkpoint staging to selected node {selected_node} failed: "
            f"{stage_msg[:220]}"
        )
    return (
        f"resume checkpoint exists on {nodes_text}, "
        f"but selected node {selected_node} has no matching checkpoint; "
        "waiting for a checkpoint-local node instead of launching a fresh run"
    )


def _ensure_required_resume_node(
    task: dict,
    resume_locations: list,
    resume_nodes: set,
    events: list,
    deps: DispatchResumePlacementDeps,
) -> tuple[bool, list, set]:
    require_node = task.get("require_node")
    if deps.soft_require_nodes(task):
        return True, resume_locations, resume_nodes
    if not resume_nodes or not require_node:
        return True, resume_locations, resume_nodes
    stage_state, source_loc, stage_msg = deps.resume_checkpoint_stage_check(
        task, require_node, resume_locations)
    if stage_state == "ready" and source_loc:
        deps.record_staged_resume_location(task, require_node, source_loc)
        resume_locations = list(task.get("resume_locations") or [])
        return True, resume_locations, _resume_nodes(resume_locations)
    reason = _require_node_missing_ckpt_reason(
        stage_state,
        resume_nodes=resume_nodes,
        require_node=require_node,
        stage_msg=stage_msg,
    )
    events.append(_blocked_event(task, reason))
    return False, resume_locations, resume_nodes


def resolve_resume_aware_placement(
    task: dict,
    *,
    state: dict,
    nodes: list,
    resume_scan_cache: dict,
    events: list,
    pick_placement: Callable[..., tuple | None],
    deps: DispatchResumePlacementDeps,
) -> ResumePlacementResult:
    """Choose placement while protecting resume checkpoint locality."""
    if not deps.task_requires_resume_scan(task):
        task.pop("resume_checkpoint_migration_plan", None)
        return ResumePlacementResult(placement=pick_placement(task, nodes))

    pre_scan_placement = pick_placement(task, nodes)
    has_known_resume_hint = task.get("resume_locations") or task.get("resume_checkpoint_node")
    if pre_scan_placement is None and not has_known_resume_hint:
        task.pop("resume_checkpoint_migration_plan", None)
        return ResumePlacementResult(placement=None)

    del resume_scan_cache
    scan_ready, resume_locations, resume_errors = deps.cached_resume_locations_for_task(
        task, nodes)
    if not scan_ready:
        events.append(_blocked_event(task, _scan_pending_reason(task)))
        return ResumePlacementResult(skip_task=True)
    if resume_errors and not resume_locations:
        if deps.allow_initial_resume_scan_error(task):
            task["resume_scan_errors_ignored_at"] = deps.now()
            task["resume_scan_errors_ignored"] = dict(resume_errors)
            task.pop("last_block_reason", None)
        else:
            events.append(_blocked_event(task, _scan_error_reason(resume_errors)))
            return ResumePlacementResult(skip_task=True)

    resume_nodes = _resume_nodes(resume_locations)
    ok, resume_locations, resume_nodes = _ensure_required_resume_node(
        task, resume_locations, resume_nodes, events, deps)
    if not ok:
        return ResumePlacementResult(
            resume_locations=resume_locations,
            resume_nodes=resume_nodes,
            skip_task=True,
        )

    extra_allowed_nodes, ckpt_migration_plan = (
        deps.checkpoint_migration_extra_allowed_nodes(
            task, nodes, resume_locations, state)
    )
    if ckpt_migration_plan:
        task["resume_checkpoint_migration_plan"] = ckpt_migration_plan
    else:
        task.pop("resume_checkpoint_migration_plan", None)
    # Prefer nodes that already have a generation-verified checkpoint
    # directory.  A raw resume file may exist on several nodes after an
    # interrupted rsync; selecting one of those first and rejecting it later
    # can starve a fully staged sibling indefinitely.
    ready_nodes = []
    has_exact_required_node = bool(
        task.get("require_node") and not deps.soft_require_nodes(task))
    if resume_nodes and not has_exact_required_node:
        for node in nodes:
            node_name = node.get("name")
            if not node_name:
                continue
            stage_state, source_loc, _ = deps.resume_checkpoint_stage_check(
                task, node_name, resume_locations)
            if stage_state == "ready" and source_loc:
                ready_nodes.append(node)
    placement = None
    if ready_nodes:
        placement = pick_placement(
            task, ready_nodes, extra_allowed_nodes=extra_allowed_nodes)
    if placement is None:
        placement = pick_placement(
            task, nodes, extra_allowed_nodes=extra_allowed_nodes)
    return ResumePlacementResult(
        placement=placement,
        resume_locations=resume_locations,
        resume_nodes=resume_nodes,
        extra_allowed_nodes=extra_allowed_nodes,
    )


def validate_selected_resume_checkpoint(
    task: dict,
    *,
    resume_locations: list,
    resume_nodes: set,
    events: list,
    deps: DispatchResumePlacementDeps,
) -> bool:
    """Reject a selected node that would launch fresh while a checkpoint exists elsewhere."""
    if not resume_nodes:
        return True
    selected_node = task.get("node")
    stage_state, source_loc, stage_msg = deps.resume_checkpoint_stage_check(
        task, selected_node, resume_locations)
    if stage_state == "ready" and source_loc:
        deps.record_staged_resume_location(task, selected_node, source_loc)
        return True
    reason = _selected_node_missing_ckpt_reason(
        stage_state,
        resume_nodes=resume_nodes,
        selected_node=selected_node,
        stage_msg=stage_msg,
    )
    task["last_block_reason"] = reason
    deps.release_task_claims_and_intents(task)
    task["node"] = None
    task["gpu_idx"] = None
    events.append({"type": "blocked", "task_id": task["id"], "task": task, "reason": reason})
    return False
