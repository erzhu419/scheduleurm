from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DispatchCommandDeps:
    set_hard_rule_mode: Callable[[Any], Any]
    configure_algorithm: Callable[[Any], Any]
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    recover_stale_launching_tasks_outside_lock: Callable[..., int]
    notify: Callable[..., Any]
    preload_docker_images_outside_lock: Callable[..., Any]
    stage_migration_candidates_outside_lock: Callable[[], Any]
    stage_launch_candidates_outside_lock: Callable[..., Any]
    sync_completed_results_outside_lock: Callable[[], Any]
    running_probe_snapshot_outside_lock: Callable[[str], tuple[dict, set]]
    eta_tail_snapshot_outside_lock: Callable[[str], tuple[dict, set]]
    probe_all: Callable[[], list]
    update_running_tasks: Callable[..., Any]
    seed_pending_eta_from_history: Callable[[dict], Any]
    remember_running_resource_snapshots: Callable[[dict, list], Any]
    reconcile_aggregate_only_vram: Callable[[dict, list], Any]
    reserve_inflight_vram: Callable[[dict, list], Any]
    print_node_summary: Callable[[list], Any]
    do_dispatch: Callable[..., tuple[list, int]]
    execute_deferred_launches: Callable[..., list]
    load_state_shared_snapshot: Callable[[str], dict]
    dispatch_cycle_log_payload: Callable[[dict, list, list, int], dict]
    format_task_location: Callable[[dict], str]
    format_mem_gb: Callable[[int], str]
    max_launch_retry: int
    defer_launch_outside_lock: bool
    print_fn: Callable[..., Any] = print
    probe_all_cached: Callable[..., list] | None = None
    prefetch_resume_scans_outside_lock: Callable[..., Any] | None = None
    dispatch_probe_cache_max_age_s: int = 30
    refresh_escalations_outside_lock: Callable[[], Any] = lambda: None


def _notify_error(deps: DispatchCommandDeps, event_type: str, exc: Exception) -> None:
    deps.notify(event_type, {"error": str(exc)[:200]}, feishu_enabled=False)


def _dispatch_preflight(args, target_task_ids: set[str], lock_timeout, deps: DispatchCommandDeps) -> int:
    recovered_launching = 0
    try:
        recovered_launching = deps.recover_stale_launching_tasks_outside_lock(
            purpose="dispatch:recover-launching",
            lock_timeout=lock_timeout,
        )
    except Exception as exc:
        _notify_error(deps, "launching_recovery_error", exc)

    try:
        deps.preload_docker_images_outside_lock(task_ids=target_task_ids or None)
    except Exception as exc:
        _notify_error(deps, "preload_error", exc)

    if not target_task_ids:
        try:
            deps.stage_migration_candidates_outside_lock()
        except Exception as exc:
            _notify_error(deps, "migration_staging_error_outer", exc)

    try:
        deps.refresh_escalations_outside_lock()
    except Exception as exc:
        _notify_error(deps, "escalation_refresh_error_outer", exc)

    if not target_task_ids:
        try:
            deps.sync_completed_results_outside_lock()
        except Exception as exc:
            _notify_error(deps, "result_sync_error_outer", exc)

    return recovered_launching


def _print_dispatch_event(event: dict, deps: DispatchCommandDeps) -> None:
    event_type = event.get("type")
    if event_type == "lineage_repaired":
        deps.print_fn(f"  [lineage] repaired {event.get('count', 0)} queued parent retry record(s)")
        return
    if event_type == "algorithm_global_batch_plan":
        algorithm = str(event.get("algorithm") or "unknown")
        hook_status = "ready" if event.get("scheduler_hook_ready") else "not-ready"
        deps.print_fn(
            f"  [algorithm] GLOBAL-PLAN {algorithm}: "
            f"{event.get('planned_tasks', 0)}/{event.get('candidate_tasks', 0)} task(s), "
            f"hook={hook_status}"
        )
        return
    if event_type == "algorithm_global_batch_plan_error":
        algorithm = str(event.get("algorithm") or "unknown")
        reason = str(event.get("reason") or "unknown error")
        deps.print_fn(f"  [algorithm] GLOBAL-PLAN-ERROR {algorithm}: {reason[:200]}")
        return
    if event_type == "algorithm_global_batch_plan_rejected":
        algorithm = str(event.get("algorithm") or "unknown")
        reason = str(event.get("reason") or "uncertified action")
        deps.print_fn(f"  [algorithm] GLOBAL-PLAN-REJECTED {algorithm}: {reason[:200]}")
        return
    if event_type == "algorithm_global_batch_execution":
        planned = len(event.get("planned_task_ids") or [])
        executed = len(event.get("executed_task_ids") or [])
        status = "exact" if event.get("exact_execution_ready") else "incomplete"
        deps.print_fn(
            f"  [algorithm] GLOBAL-EXECUTION {executed}/{planned} task(s), "
            f"status={status}"
        )
        return

    task_id = event["task_id"]
    if event_type == "no_fit":
        task = event["task"]
        reason = str(task.get("last_block_reason") or "no fit")
        if len(reason) > 240:
            reason = reason[:237] + "..."
        deps.print_fn(f"  [{task_id}] WAIT  {reason} ({task['description'][:50]})")
    elif event_type == "blocked":
        deps.print_fn(f"  [{task_id}] BLOCK {event['reason']}")
    elif event_type == "resume_found":
        deps.print_fn(f"  [{task_id}] resume_from={event['resume_from']}")
    elif event_type == "launched":
        task = event["task"]
        deps.print_fn(
            f"  [{task_id}] LAUNCH on {deps.format_task_location(task)}  "
            f"{event['msg']}  log={task['log_path']}"
        )
    elif event_type == "launch_failed_retry":
        task = event["task"]
        deps.print_fn(
            f"  [{task_id}] RETRY launch failed "
            f"({task.get('launch_fail_count', '?')}/{deps.max_launch_retry}): "
            f"{event['error'][:200]}"
        )
    elif event_type == "launch_failed_terminal":
        deps.print_fn(f"  [{task_id}] FAIL  launch failed permanently: {event['error'][:200]}")
    elif event_type == "migrated":
        deps.print_fn(
            f"  [{task_id}] MIGRATE {event.get('from_node') or '?'} → "
            f"{event.get('to_node') or '?'}  (eta={event.get('eta_seconds', 0)}s, load-balance)"
        )
    elif event_type == "preempted":
        deps.print_fn(
            f"  [{task_id}] PREEMPT on {event.get('freed_node') or '?'} "
            f"(freed {event.get('cpu_freed', 0)}c / {deps.format_mem_gb(event.get('ram_freed', 0))})"
        )
    elif event_type == "claim_race":
        deps.print_fn(f"  [{task_id}] CLAIM-RACE {event['reason'][:200]}")
    elif event_type == "algorithm_global_batch_replan_required":
        deps.print_fn(
            f"  [{task_id}] REPLAN exact global action unavailable: "
            f"{event.get('reason', '')[:200]}"
        )
    elif event_type == "launch_commit_skipped":
        deps.print_fn(f"  [{task_id}] SKIP launch commit: {event.get('reason', '')[:200]}")


def run_dispatch_command(args, *, deps: DispatchCommandDeps) -> None:
    deps.set_hard_rule_mode(getattr(args, "hard_rule_mode", None))
    deps.configure_algorithm(getattr(args, "algorithm", None) or None)
    target_task_ids = {
        str(value)
        for value in (getattr(args, "dispatch_task_ids", None) or [])
        if str(value)
    }
    lock_timeout = getattr(args, "lock_timeout", None)

    recovered_launching = _dispatch_preflight(args, target_task_ids, lock_timeout, deps)

    targeted_dispatch = bool(target_task_ids)
    if targeted_dispatch:
        # Targeted dispatch is usually on the submit/repair hot path.  It only
        # needs current node capacity plus the requested queued records, not a
        # full running-task maintenance sweep over unrelated logs.
        running_probe_results, running_probe_ids = {}, set()
        eta_tail_outputs, eta_tail_ids = {}, set()
    else:
        running_probe_results, running_probe_ids = deps.running_probe_snapshot_outside_lock(
            "dispatch:running-probe"
        )
        eta_tail_outputs, eta_tail_ids = deps.eta_tail_snapshot_outside_lock("dispatch:eta-tail")
    if targeted_dispatch and deps.probe_all_cached is not None:
        nodes = deps.probe_all_cached(max_age_s=deps.dispatch_probe_cache_max_age_s)
    else:
        nodes = deps.probe_all()
    if deps.prefetch_resume_scans_outside_lock is not None:
        try:
            deps.prefetch_resume_scans_outside_lock(
                nodes,
                task_ids=target_task_ids or None,
            )
        except Exception as exc:
            _notify_error(deps, "resume_scan_prefetch_error_outer", exc)
    try:
        deps.stage_launch_candidates_outside_lock(
            task_ids=target_task_ids or None
        )
    except Exception as exc:
        _notify_error(deps, "launch_staging_error_outer", exc)

    with deps.state_lock(timeout_s=lock_timeout, purpose="dispatch:main"):
        state = deps.load_state()
        if not targeted_dispatch:
            deps.update_running_tasks(
                state,
                probe_results=running_probe_results,
                probed_task_ids=running_probe_ids,
                eta_tail_outputs=eta_tail_outputs,
                eta_probed_task_ids=eta_tail_ids,
            )
        deps.seed_pending_eta_from_history(state)
        deps.remember_running_resource_snapshots(state, nodes)
        deps.reconcile_aggregate_only_vram(state, nodes)
        deps.reserve_inflight_vram(state, nodes)
        deps.print_node_summary(nodes)
        events, queued_count = deps.do_dispatch(
            state,
            nodes,
            target_task_ids=target_task_ids or None,
            defer_launches=deps.defer_launch_outside_lock,
        )
        deps.save_state(state)

    had_deferred_launches = any(event.get("type") == "launch_intent" for event in events)
    events = deps.execute_deferred_launches(events, lock_timeout=lock_timeout)
    if had_deferred_launches:
        state = deps.load_state_shared_snapshot("dispatch:post-launch-snapshot")

    if recovered_launching:
        deps.notify(
            "launching_state_recovered",
            {"reverted_count": recovered_launching},
            feishu_enabled=False,
        )
    for event in events:
        if event.get("type") in ("pre_launch_snapshot", "post_launch_snapshot"):
            deps.notify(event["type"], event.get("snapshot") or {}, feishu_enabled=False)

    deps.notify(
        "dispatch_cycle",
        deps.dispatch_cycle_log_payload(state, nodes, events, queued_count),
        feishu_enabled=False,
    )

    if queued_count == 0 and not events:
        deps.print_fn("=== dispatch === (nothing queued)")
        return
    deps.print_fn(f"=== dispatch === ({queued_count} queued)")
    for event in events:
        _print_dispatch_event(event, deps)
