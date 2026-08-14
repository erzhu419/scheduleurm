"""Checkpoint-aware placement expansion for resume tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


_UNBOUNDED_CHECKPOINT_WAIT_S = 10 ** 9


@dataclass(frozen=True)
class CheckpointMigrationDeps:
    task_requires_resume_scan: Callable[[dict], bool]
    estimate_node_fit_wait_seconds: Callable[[dict, dict | None, dict], int | None]
    blocked_nodes_for_task: Callable[[dict], set]
    launch_failed_nodes_for_task: Callable[[dict], set]
    pick_placement: Callable[..., tuple | None]
    resume_checkpoint_stage_check: Callable[[dict, str, list], tuple]
    resume_ckpt_stage_estimate_seconds: Callable[[str, dict | None], int]
    min_wait_s: int
    safety_s: int


def checkpoint_migration_extra_allowed_nodes(
    task: dict,
    nodes: list,
    resume_locations: list,
    state: dict,
    *,
    deps: CheckpointMigrationDeps,
) -> tuple:
    """Allow staged checkpoint retries to use a faster non-locality node."""
    if not deps.task_requires_resume_scan(task) or not resume_locations:
        return [], None
    if task.get("require_node"):
        return [], None
    resume_nodes = {loc.get("node") for loc in resume_locations if loc.get("node")}
    if not resume_nodes:
        return [], None
    node_by_name = {node.get("name"): node for node in nodes if node.get("name")}
    wait_candidates = []
    for node_name in sorted(resume_nodes):
        wait = deps.estimate_node_fit_wait_seconds(task, node_by_name.get(node_name), state)
        if wait is not None:
            wait_candidates.append(wait)
    # None means the checkpoint-local node has no foreseeable fit (for
    # example, its host is reachable but its GPU driver is unavailable).  It
    # must not disable migration; treat it as an unbounded wait and let the
    # staging route fail closed if the checkpoint source itself is unreachable.
    checkpoint_wait_s = (
        min(wait_candidates)
        if wait_candidates
        else _UNBOUNDED_CHECKPOINT_WAIT_S
    )
    if checkpoint_wait_s < deps.min_wait_s:
        return [], None

    allowed = {str(node) for node in (task.get("allowed_nodes") or [])}
    blocked = deps.blocked_nodes_for_task(task)
    launch_failed = deps.launch_failed_nodes_for_task(task)
    extra = []
    targets = []
    for node_state in nodes:
        node_name = node_state.get("name")
        if not node_name or node_name in resume_nodes:
            continue
        if not node_state.get("alive"):
            continue
        if node_name in blocked or node_name in launch_failed:
            continue
        if allowed and node_name not in allowed and node_name != "local":
            continue
        if deps.pick_placement(task, [node_state], extra_allowed_nodes=[node_name]) is None:
            continue
        stage_state, source_loc, stage_msg = deps.resume_checkpoint_stage_check(
            task,
            node_name,
            resume_locations,
        )
        if stage_state not in ("ready", "needs_stage"):
            continue
        stage_s = deps.resume_ckpt_stage_estimate_seconds(stage_state, source_loc)
        if stage_s + deps.safety_s >= checkpoint_wait_s:
            continue
        extra.append(node_name)
        targets.append({
            "node": node_name,
            "source": (source_loc or {}).get("node"),
            "stage_state": stage_state,
            "stage_estimate_s": stage_s,
            "stage_msg": stage_msg,
        })
    if not extra:
        return [], None
    targets.sort(key=lambda item: (
        item.get("stage_estimate_s", 10 ** 9),
        item.get("node") != "local",
        item.get("node") or "",
    ))
    return extra, {
        "checkpoint_wait_s": int(checkpoint_wait_s),
        "safety_s": int(deps.safety_s),
        "targets": targets,
    }
