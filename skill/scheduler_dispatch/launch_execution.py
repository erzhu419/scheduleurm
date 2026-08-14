"""Launch lease, execution, and resource snapshot handling for dispatch."""

from __future__ import annotations

import copy
import time as _time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DispatchLaunchExecutionDeps:
    resource_snapshot_for_placement: Callable[..., dict]
    new_launch_token: Callable[[], str]
    debit_node_resources_for_launch: Callable[[list, dict], None]
    task_run_identity: Callable[[dict], Any]
    save_state: Callable[[dict], None]
    launch: Callable[..., tuple[bool, str]]
    apply_launch_result_to_task: Callable[[dict, bool, str, list], bool]
    notify: Callable[..., Any]
    now: Callable[[], float] = _time.time


def _annotate_crossed_thresholds(pre_snapshot: dict, post_snapshot: dict) -> None:
    pre_gpu = (pre_snapshot or {}).get("gpu") or {}
    post_gpu = (post_snapshot or {}).get("gpu") or {}
    pre_ram = (pre_snapshot or {}).get("node_ram") or {}
    post_ram = (post_snapshot or {}).get("node_ram") or {}
    post_snapshot["crossed_one_third_from_pre"] = (
        bool(post_gpu.get("over_one_third")) and not bool(pre_gpu.get("over_one_third"))
    )
    post_snapshot["crossed_ram_headroom_from_pre"] = (
        bool(post_ram.get("below_headroom")) and not bool(pre_ram.get("below_headroom"))
    )


def _post_launch_snapshot(
    state: dict,
    nodes: list,
    task: dict,
    pre_launch_snapshot: dict,
    deps: DispatchLaunchExecutionDeps,
) -> dict:
    post_launch_snapshot = deps.resource_snapshot_for_placement(
        state,
        nodes,
        task,
        task.get("node"),
        task.get("gpu_idx"),
        "post_launch_estimated",
    )
    _annotate_crossed_thresholds(pre_launch_snapshot, post_launch_snapshot)
    return post_launch_snapshot


def _mark_running_for_dispatch_pass(
    task: dict,
    nodes: list,
    running_keys: set,
    deps: DispatchLaunchExecutionDeps,
) -> None:
    deps.debit_node_resources_for_launch(nodes, task)
    launched_key = deps.task_run_identity(task)
    if launched_key:
        running_keys.add(launched_key)


def apply_dispatch_launch_execution(
    task: dict,
    *,
    state: dict,
    nodes: list,
    picked_state: dict | None,
    running_keys: set,
    events: list,
    defer_launches: bool,
    deps: DispatchLaunchExecutionDeps,
) -> bool:
    """Persist a launch lease, optionally execute launch, and append launch events."""
    pre_launch_snapshot = deps.resource_snapshot_for_placement(
        state,
        nodes,
        task,
        task.get("node"),
        task.get("gpu_idx"),
        "pre_launch",
    )
    task["last_launch_pre_snapshot"] = pre_launch_snapshot
    events.append({
        "type": "pre_launch_snapshot",
        "task_id": task["id"],
        "snapshot": pre_launch_snapshot,
    })

    task["status"] = "launching"
    task["launch_protocol_version"] = 2
    task["launching_started_at"] = deps.now()
    task["launch_token"] = deps.new_launch_token()
    if defer_launches:
        _mark_running_for_dispatch_pass(task, nodes, running_keys, deps)
        post_launch_snapshot = _post_launch_snapshot(
            state, nodes, task, pre_launch_snapshot, deps)
        events.append({
            "type": "launch_intent",
            "task_id": task["id"],
            "task": copy.deepcopy(task),
            "node_state": copy.deepcopy(picked_state),
            "post_launch_snapshot": copy.deepcopy(post_launch_snapshot),
        })
        return True

    try:
        deps.save_state(state)
    except Exception:
        pass
    ok, msg = deps.launch(task, node_state=picked_state)
    if not deps.apply_launch_result_to_task(task, ok, msg, events):
        return False
    try:
        deps.save_state(state)
    except Exception as exc:
        deps.notify(
            "save_state_after_launch_failed",
            {"id": task["id"], "error": str(exc)[:200]},
            feishu_enabled=False,
        )
    _mark_running_for_dispatch_pass(task, nodes, running_keys, deps)
    post_launch_snapshot = _post_launch_snapshot(
        state, nodes, task, pre_launch_snapshot, deps)
    task["last_launch_post_snapshot"] = post_launch_snapshot
    task["last_resource_snapshot"] = post_launch_snapshot
    events.append({
        "type": "post_launch_snapshot",
        "task_id": task["id"],
        "snapshot": post_launch_snapshot,
    })
    return True
