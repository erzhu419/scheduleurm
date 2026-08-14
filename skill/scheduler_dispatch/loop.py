"""Top-level dispatch loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time as _time
from typing import Any, Callable, Optional


GLOBAL_BATCH_EXECUTION_CONTRACT = "enforced_exact_placement_v1"


@dataclass(frozen=True)
class DispatchLoopDeps:
    reconcile_requeue_lineage_invariants: Callable[[dict], int]
    consider_migration: Callable[[dict, list], list]
    preempt_for_high_priority: Callable[[dict, list], list]
    prepare_dispatch_node_accounting: Callable[..., list]
    dispatch_accounting_deps: Callable[[], Any]
    refresh_queued_resource_estimates: Callable[..., Any]
    load_history: Callable[[], dict]
    resource_estimate_deps: Callable[[], Any]
    build_dispatch_queue_plan: Callable[..., Any]
    task_run_identity: Callable[[dict], Any]
    algorithm_global_batch_plan: Callable[[list, list], tuple[dict, dict | None]]
    apply_dispatch_task_gates: Callable[..., Any]
    dispatch_task_gate_deps: Callable[[], Any]
    resolve_resume_aware_placement: Callable[..., Any]
    dispatch_resume_placement_deps: Callable[[], Any]
    build_no_fit_reason: Callable[..., str]
    apply_dispatch_placement: Callable[..., dict | None]
    dispatch_placement_apply_deps: Callable[[], Any]
    validate_selected_resume_checkpoint: Callable[..., bool]
    release_task_claims_and_intents: Callable[..., Any]
    precheck_git: Callable[[dict], tuple[bool, str]]
    resume_location_for_node: Callable[[dict, str], dict | None]
    node_configs: dict
    stage_cwd_check: Callable[[str, str], str]
    launch_input_stage_state: Callable[[dict, str], str]
    apply_launch_staging_gate: Callable[..., Any]
    apply_launch_input_staging_gate: Callable[..., Any]
    dispatch_launch_staging_deps: Callable[[], Any]
    apply_dispatch_launch_execution: Callable[..., Any]
    dispatch_launch_execution_deps: Callable[[], Any]
    pick_placement: Callable[..., Any]
    now: Callable[[], float] = _time.time


def _dispatch_attempt_order(task: dict) -> tuple[int, float, float, str]:
    """Retry staged inputs promptly, then rotate fairly by last attempt."""
    stage_retry = int(not (
        task.get("stage_input_target")
        and str(task.get("last_block_reason") or "").startswith(
            "launch input staging:"
        )
    ))
    try:
        attempted_at = float(task.get("last_dispatch_attempt_at") or 0.0)
    except (TypeError, ValueError):
        attempted_at = 0.0
    try:
        submitted_at = float(task.get("submitted_at") or 0.0)
    except (TypeError, ValueError):
        submitted_at = 0.0
    return stage_retry, attempted_at, submitted_at, str(task.get("id") or "")


def _dispatch_resource_lane(task: dict) -> str:
    if int(task.get("est_vram_mb") or 0) > 0:
        return "gpu"
    widths = []
    plan = task.get("cpu_batch_plan")
    for value in (
        task.get("cpu_declared_cores"),
        task.get("cpu_auto_workers"),
        plan.get("workers") if isinstance(plan, dict) else None,
        task.get("cpu_cores"),
    ):
        try:
            parsed = int(value or 0)
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            widths.append(parsed)
    return "cpu_single" if max(widths or [1]) <= 1 else "cpu_multi"


def _interleave_resource_lanes(tasks: list[dict]) -> list[dict]:
    lane_order = ("cpu_single", "cpu_multi", "gpu")
    lanes = {lane: [] for lane in lane_order}
    for task in tasks:
        lanes[_dispatch_resource_lane(task)].append(task)
    for lane in lanes.values():
        lane.sort(key=_dispatch_attempt_order)
    selected = []
    offsets = {lane: 0 for lane in lane_order}
    while len(selected) < len(tasks):
        advanced = False
        for lane_name in lane_order:
            index = offsets[lane_name]
            lane = lanes[lane_name]
            if index >= len(lane):
                continue
            selected.append(lane[index])
            offsets[lane_name] = index + 1
            advanced = True
        if not advanced:
            break
    return selected


def select_dispatch_cycle_tasks(queued: list[dict], max_queued: Optional[int]) -> list[dict]:
    """Bound one pass without letting a saturated resource lane block another."""
    if max_queued is None or max_queued <= 0 or len(queued) <= max_queued:
        return list(queued)
    buckets = {"high": [], "normal": [], "low": []}
    for task in queued:
        priority = str(task.get("priority") or "normal")
        buckets[priority if priority in buckets else "normal"].append(task)
    for priority, bucket in list(buckets.items()):
        buckets[priority] = _interleave_resource_lanes(bucket)
    offsets = {key: 0 for key in buckets}
    weights = (("high", 4), ("normal", 2), ("low", 1))
    selected = []
    while len(selected) < max_queued:
        advanced = False
        for priority, weight in weights:
            bucket = buckets[priority]
            for _ in range(weight):
                idx = offsets[priority]
                if idx >= len(bucket) or len(selected) >= max_queued:
                    break
                selected.append(bucket[idx])
                offsets[priority] = idx + 1
                advanced = True
        if not advanced:
            break
    return selected


def _enforced_global_batch_event(event: dict | None) -> bool:
    return bool(
        isinstance(event, dict)
        and event.get("execution_contract") == GLOBAL_BATCH_EXECUTION_CONTRACT
        and event.get("fail_closed") is True
    )


def _placement_matches_batch_action(
    placement: tuple | None,
    action: dict | None,
) -> bool:
    if placement is None or not isinstance(action, dict):
        return False
    node, gpu_idx = placement
    if str(node or "") != str(action.get("node") or ""):
        return False
    expected_gpu = action.get("gpu_idx")
    if expected_gpu is None:
        return gpu_idx is None
    return str(gpu_idx) == str(expected_gpu)


def _launch_input_state_cache_key(task: dict, target: str) -> tuple:
    paths = [
        str(path).strip()
        for path in (task.get("stage_input_paths") or [])
        if str(path).strip()
    ]
    if not paths:
        paths = [
            os.path.dirname(str(path).strip())
            for path in (task.get("wait_for_files") or [])
            if str(path).strip()
        ]
    return str(target), tuple(dict.fromkeys(paths))


def do_dispatch(
    state: dict,
    nodes: list,
    *,
    target_task_ids: Optional[set] = None,
    defer_launches: bool = False,
    max_queued: Optional[int] = None,
    deps: DispatchLoopDeps,
) -> tuple[list, int]:
    """Place every fittable queued task. Mutates state and nodes in place."""
    events = []
    prio = {"high": 0, "normal": 1, "low": 2}
    target_ids = {str(x) for x in (target_task_ids or set()) if str(x)}
    repaired = deps.reconcile_requeue_lineage_invariants(state)
    if repaired:
        events.append({
            "type": "lineage_repaired",
            "count": repaired,
            "reason": "queued requeue parents were made terminal before dispatch",
        })

    migrated = [] if target_ids else deps.consider_migration(state, nodes)
    if migrated:
        by_id = {t["id"]: t for t in state.get("tasks", [])}
        for tid in migrated:
            cand = by_id.get(tid, {})
            events.append({
                "type": "migrated",
                "task_id": tid,
                "from_node": cand.get("migrated_from"),
                "to_node": cand.get("preferred_node"),
                "eta_seconds": int(cand.get("eta_seconds") or 0),
                "reason": cand.get("last_block_reason", ""),
            })

    preempted = [] if target_ids else deps.preempt_for_high_priority(state, nodes)
    events.extend(deps.prepare_dispatch_node_accounting(
        state,
        nodes,
        preempted,
        deps=deps.dispatch_accounting_deps(),
    ))
    deps.refresh_queued_resource_estimates(
        state,
        deps.load_history(),
        deps=deps.resource_estimate_deps(),
    )
    queue_plan = deps.build_dispatch_queue_plan(
        state,
        nodes,
        target_ids,
        prio,
        task_run_identity=deps.task_run_identity,
        algorithm_global_batch_plan=deps.algorithm_global_batch_plan,
    )
    running_keys = queue_plan.running_keys
    queued_all = queue_plan.queued
    queued = select_dispatch_cycle_tasks(queued_all, max_queued)
    global_batch_plan = queue_plan.global_batch_plan
    global_batch_enforced = _enforced_global_batch_event(
        queue_plan.global_batch_event)
    if queue_plan.global_batch_event:
        events.append(queue_plan.global_batch_event)
    if global_batch_enforced:
        selected_ids = set(global_batch_plan)
        # The global policy already applies its own bounded batch size. Applying
        # max_queued again could execute only a strict subset of one certified
        # action, so an enforced action is attempted atomically at task-ID level.
        queued = [
            task for task in queued_all
            if str(task.get("id") or "") in selected_ids
        ]
    resume_scan_cache = {}
    launch_cwd_state_cache = {}
    launch_input_state_cache = {}
    attempted_at = deps.now()

    def _global_batch_hint_for(task: dict) -> Optional[dict]:
        return global_batch_plan.get(str(task.get("id") or ""))

    def _pick_placement_with_global_batch_hint(task: dict, search_nodes: list, extra_allowed_nodes=None):
        hint = _global_batch_hint_for(task)
        if global_batch_enforced:
            if not hint or not hint.get("node"):
                return None
            required_node = str(task.get("require_node") or "")
            action_node = str(hint.get("node") or "")
            if required_node and required_node != action_node:
                return None
            allowed_nodes = {
                str(node) for node in (task.get("allowed_nodes") or []) if str(node)
            }
            if allowed_nodes and action_node not in allowed_nodes:
                return None
            staged_target = str(task.get("stage_input_target") or "")
            if staged_target and staged_target != action_node:
                return None
            required_gpu = task.get("require_gpu_idx")
            action_gpu = hint.get("gpu_idx")
            if required_gpu not in (None, "") and str(required_gpu) != str(action_gpu):
                return None
            exact = dict(task)
            exact["require_node"] = action_node
            if action_gpu is None:
                exact.pop("require_gpu_idx", None)
            else:
                exact["require_gpu_idx"] = action_gpu
            exact["global_batch_hint"] = hint
            if extra_allowed_nodes is not None:
                return deps.pick_placement(
                    exact, search_nodes, extra_allowed_nodes=extra_allowed_nodes)
            return deps.pick_placement(exact, search_nodes)
        staged_input_target = str(task.get("stage_input_target") or "")
        if staged_input_target and not task.get("require_node") and not task.get("preferred_node"):
            hinted = dict(task)
            hinted["preferred_node"] = staged_input_target
            if extra_allowed_nodes is not None:
                return deps.pick_placement(
                    hinted, search_nodes, extra_allowed_nodes=extra_allowed_nodes)
            return deps.pick_placement(hinted, search_nodes)
        if (
            hint
            and hint.get("node")
            and not task.get("require_node")
            and not task.get("preferred_node")
        ):
            hinted = dict(task)
            hinted["preferred_node"] = hint.get("node")
            hinted["global_batch_hint"] = hint
            if extra_allowed_nodes is not None:
                return deps.pick_placement(
                    hinted, search_nodes, extra_allowed_nodes=extra_allowed_nodes)
            return deps.pick_placement(hinted, search_nodes)
        if extra_allowed_nodes is not None:
            return deps.pick_placement(task, search_nodes, extra_allowed_nodes=extra_allowed_nodes)
        return deps.pick_placement(task, search_nodes)

    for task in queued:
        task["last_dispatch_attempt_at"] = attempted_at
        try:
            prior_attempts = int(task.get("dispatch_attempt_count") or 0)
        except (TypeError, ValueError):
            prior_attempts = 0
        task["dispatch_attempt_count"] = prior_attempts + 1
        gate_result = deps.apply_dispatch_task_gates(
            task,
            state=state,
            running_keys=running_keys,
            deps=deps.dispatch_task_gate_deps(),
        )
        events.extend(gate_result.events)
        if gate_result.skip_task:
            continue

        resume_result = deps.resolve_resume_aware_placement(
            task,
            state=state,
            nodes=nodes,
            resume_scan_cache=resume_scan_cache,
            events=events,
            pick_placement=_pick_placement_with_global_batch_hint,
            deps=deps.dispatch_resume_placement_deps(),
        )
        if resume_result.skip_task:
            continue
        placement = resume_result.placement
        resume_locations = resume_result.resume_locations
        resume_nodes = resume_result.resume_nodes
        extra_allowed_nodes = resume_result.extra_allowed_nodes
        if placement is None:
            task["last_block_reason"] = deps.build_no_fit_reason(
                task,
                nodes,
                extra_allowed_nodes=extra_allowed_nodes,
            )
            events.append({"type": "no_fit", "task_id": task["id"], "task": task})
            continue

        batch_action = _global_batch_hint_for(task)
        if (
            global_batch_enforced
            and not _placement_matches_batch_action(placement, batch_action)
        ):
            task["last_block_reason"] = (
                "global theorem action invalidated before launch; "
                "exact node/GPU placement is unavailable and requires replanning"
            )
            events.append({
                "type": "algorithm_global_batch_replan_required",
                "task_id": task["id"],
                "planned_node": (batch_action or {}).get("node"),
                "planned_gpu_idx": (batch_action or {}).get("gpu_idx"),
                "observed_placement": placement,
                "reason": task["last_block_reason"],
            })
            continue

        picked_state = deps.apply_dispatch_placement(
            task,
            placement=placement,
            nodes=nodes,
            state=state,
            batch_hint=_global_batch_hint_for(task),
            deps=deps.dispatch_placement_apply_deps(),
        )
        if not deps.validate_selected_resume_checkpoint(
            task,
            resume_locations=resume_locations,
            resume_nodes=resume_nodes,
            events=events,
            deps=deps.dispatch_resume_placement_deps(),
        ):
            continue

        deps.release_task_claims_and_intents(
            task, exclude_nodes={task["node"]}, clear_markers=False)
        ok, why = deps.precheck_git(task)
        if not ok:
            task["last_block_reason"] = why
            deps.release_task_claims_and_intents(task)
            task["node"] = None
            task["gpu_idx"] = None
            events.append({
                "type": "blocked",
                "task_id": task["id"],
                "task": task,
                "reason": why,
            })
            continue
        if why and why.startswith("warn:"):
            task["last_block_reason"] = why
            events.append({
                "type": "git_warn",
                "task_id": task["id"],
                "task": task,
                "reason": why,
            })

        loc = None
        if not task.get("skip_resume_scan"):
            loc = deps.resume_location_for_node(task, task.get("node"))
        resume = (loc or {}).get("path")
        if resume:
            task["resume_from"] = resume
            events.append({
                "type": "resume_found",
                "task_id": task["id"],
                "resume_from": resume,
            })

        target = task.get("node")
        cwd_for_stage = task.get("cwd")
        if (
            target
            and cwd_for_stage
            and not task.get("skip_launch_staging")
            and deps.node_configs.get(target, {}).get("host")
        ):
            cwd_cache_key = (str(target), str(cwd_for_stage))
            if cwd_cache_key not in launch_cwd_state_cache:
                launch_cwd_state_cache[cwd_cache_key] = (
                    deps.stage_cwd_check(target, cwd_for_stage)
                )
            stage_state = launch_cwd_state_cache[cwd_cache_key]
            stage_result = deps.apply_launch_staging_gate(
                task,
                target=target,
                cwd_for_stage=cwd_for_stage,
                stage_state=stage_state,
                deps=deps.dispatch_launch_staging_deps(),
            )
            if stage_result.blocked:
                if stage_result.event:
                    events.append(stage_result.event)
                continue

        if target and (task.get("stage_input_paths") or task.get("wait_for_files")):
            input_cache_key = _launch_input_state_cache_key(task, target)
            if input_cache_key not in launch_input_state_cache:
                launch_input_state_cache[input_cache_key] = (
                    deps.launch_input_stage_state(task, target)
                )
            input_result = deps.apply_launch_input_staging_gate(
                task,
                target=target,
                stage_state=launch_input_state_cache[input_cache_key],
                deps=deps.dispatch_launch_staging_deps(),
            )
            if input_result.blocked:
                if input_result.event:
                    events.append(input_result.event)
                continue

        deps.apply_dispatch_launch_execution(
            task,
            state=state,
            nodes=nodes,
            picked_state=picked_state,
            running_keys=running_keys,
            events=events,
            defer_launches=defer_launches,
            deps=deps.dispatch_launch_execution_deps(),
        )
    if global_batch_enforced:
        planned_ids = sorted(global_batch_plan)
        executed_ids = []
        unresolved_ids = []
        by_id = {
            str(task.get("id") or ""): task
            for task in queued_all
        }
        for task_id in planned_ids:
            task = by_id.get(task_id) or {}
            actual = (task.get("node"), task.get("gpu_idx"))
            if (
                task.get("status") in ("launching", "running")
                and _placement_matches_batch_action(
                    actual, global_batch_plan.get(task_id))
            ):
                executed_ids.append(task_id)
            else:
                unresolved_ids.append(task_id)
        exact_ready = bool(planned_ids) and not unresolved_ids
        events.append({
            "type": "algorithm_global_batch_execution",
            "execution_contract": GLOBAL_BATCH_EXECUTION_CONTRACT,
            "planned_task_ids": planned_ids,
            "executed_task_ids": executed_ids,
            "unresolved_task_ids": unresolved_ids,
            "exact_execution_ready": exact_ready,
            "oracle_gap_applies_to_execution": exact_ready,
        })
    return events, len(queued_all)
