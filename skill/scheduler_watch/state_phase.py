"""State-locked watcher phase helpers."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WatchStatePhaseDeps:
    recover_queued_live_local_tasks: Callable[[dict], int]
    update_running_tasks: Callable[..., Any]
    seed_pending_eta_from_history: Callable[[dict], Any]
    detect_oom_kills_local: Callable[[dict], list]
    requeue_after_crash: Callable[[dict, dict], Any]
    notify: Callable[..., Any]
    classify_running_terminal_transitions: Callable[[list, dict], tuple[list, list]]
    remember_running_resource_snapshots: Callable[[dict, list], Any]
    reconcile_aggregate_only_vram: Callable[[dict, list], Any]
    build_crash_forensics_payload: Callable[[dict, dict, list], dict]
    build_oom_forensics_payload: Callable[[dict, dict], dict]
    is_oom_like_forensics: Callable[[dict], bool]
    enforce_post_dispatch_thresholds: Callable[[dict, list], list]
    reserve_inflight_vram: Callable[[dict, list], Any]
    do_dispatch: Callable[..., tuple[list, int]]
    reconcile_external_tasks: Callable[..., list]
    archive_terminal_tasks: Callable[[dict], int]
    detect_batch_completions: Callable[[dict, list], list]
    save_state: Callable[[dict], Any]
    watch_dispatch_max_queued_per_cycle: int = 0
    now: Callable[[], float] = _time.time
    recover_resolved_staging_failures: Callable[[dict], int] = lambda _state: 0


@dataclass
class WatchStatePhaseResult:
    recovered_launching_count: int
    recovered_queued_live_count: int
    newly_done: list
    newly_crashed: list
    crash_forensics: list
    oom_forensics: list
    resource_evictions: list
    events: list
    qcount: int
    auto_adopted: list
    auto_adopt_completed: bool
    archived_count: int
    batch_completions: list


def run_watch_state_phase(
    state: dict,
    nodes: list,
    *,
    recovered_launching_count: int,
    running_probe_results: dict | None,
    running_probe_ids: set | None,
    eta_tail_outputs: dict | None,
    eta_tail_ids: set | None,
    terminal_diagnostics: dict | None,
    should_auto_adopt: bool,
    auto_adopt_probe_data: Any,
    defer_launches: bool,
    record_phase: Callable[..., Any],
    deps: WatchStatePhaseDeps,
    defer_eta_refresh: bool = False,
) -> WatchStatePhaseResult:
    """Run the watcher state mutation phase while the caller holds state_lock."""
    recovered_queued_live_count = 0
    auto_adopted: list = []
    auto_adopt_completed = False

    phase_t0 = deps.now()
    recovered_queued_live_count += deps.recover_queued_live_local_tasks(state)
    recovered_environment_count = deps.recover_resolved_staging_failures(state)
    record_phase(
        "recover_stale",
        phase_t0,
        recovered_launching=recovered_launching_count,
        recovered_queued_live=recovered_queued_live_count,
        recovered_environment=recovered_environment_count,
    )

    phase_t0 = deps.now()
    pre_status = {t["id"]: t["status"] for t in state["tasks"]}
    deps.update_running_tasks(
        state,
        probe_results=running_probe_results,
        probed_task_ids=running_probe_ids,
        eta_tail_outputs=eta_tail_outputs,
        eta_probed_task_ids=eta_tail_ids,
        terminal_diagnostics=terminal_diagnostics,
        defer_eta_refresh=defer_eta_refresh,
    )
    record_phase(
        "update_running_tasks",
        phase_t0,
        probed=len(running_probe_results or {}),
    )

    phase_t0 = deps.now()
    deps.save_state(state)
    record_phase("save_after_running_update", phase_t0, tasks=len(state.get("tasks", [])))

    phase_t0 = deps.now()
    deps.seed_pending_eta_from_history(state)
    record_phase("seed_pending_eta", phase_t0)

    phase_t0 = deps.now()
    oom_flipped = deps.detect_oom_kills_local(state)
    record_phase("detect_oom_kills", phase_t0, flipped=len(oom_flipped or []))
    for task in oom_flipped or []:
        new_id = deps.requeue_after_crash(task, state)
        if new_id:
            task["requeued_as"] = new_id
        deps.notify(
            "task_oom_requeue",
            {
                "id": task["id"],
                "requeued_as": task.get("requeued_as"),
                "lifetime_s": max(
                    0,
                    (task.get("finished_at") or 0) - (task.get("started_at") or 0),
                ),
                "description": task.get("description", "")[:80],
            },
            feishu_enabled=False,
        )

    phase_t0 = deps.now()
    newly_done, newly_crashed = deps.classify_running_terminal_transitions(
        state["tasks"],
        pre_status,
    )
    record_phase(
        "classify_transitions",
        phase_t0,
        done=len(newly_done),
        crashed=len(newly_crashed),
    )

    phase_t0 = deps.now()
    deps.remember_running_resource_snapshots(state, nodes)
    deps.reconcile_aggregate_only_vram(state, nodes)
    crash_forensics = [
        deps.build_crash_forensics_payload(task, state, nodes)
        for task in newly_crashed
    ]
    oom_forensics = [
        deps.build_oom_forensics_payload(payload, state)
        for payload in crash_forensics
        if deps.is_oom_like_forensics(payload)
    ]
    by_id = {task.get("id"): task for task in state.get("tasks", [])}
    for payload in crash_forensics:
        if payload.get("id") in by_id:
            by_id[payload["id"]]["last_crash_forensics"] = payload
    for payload in oom_forensics:
        if payload.get("id") in by_id:
            by_id[payload["id"]]["last_oom_forensics"] = payload
    record_phase(
        "resource_and_forensics",
        phase_t0,
        crashed=len(newly_crashed),
    )

    phase_t0 = deps.now()
    evicted = deps.enforce_post_dispatch_thresholds(state, nodes)
    evicted_ids = set(evicted or [])
    resource_evictions = [
        task.get("last_resource_eviction")
        for task in state.get("tasks", [])
        if task.get("id") in evicted_ids and task.get("last_resource_eviction")
    ]
    deps.reserve_inflight_vram(state, nodes)
    record_phase("evict_and_reserve", phase_t0, evicted=len(evicted or []))

    phase_t0 = deps.now()
    events, qcount = deps.do_dispatch(
        state,
        nodes,
        defer_launches=defer_launches,
        max_queued=(
            deps.watch_dispatch_max_queued_per_cycle
            if deps.watch_dispatch_max_queued_per_cycle > 0
            else None
        ),
    )
    record_phase(
        "do_dispatch",
        phase_t0,
        queued=qcount,
        events=len(events or []),
        launch_intents=sum(
            1 for event in (events or []) if event.get("type") == "launch_intent"
        ),
    )

    phase_t0 = deps.now()
    for event in events:
        if event["type"] == "launched":
            event["task"]["notified_launch"] = True
    if should_auto_adopt and auto_adopt_probe_data is not None:
        auto_adopted = deps.reconcile_external_tasks(
            state,
            probe_data=auto_adopt_probe_data,
        )
        auto_adopt_completed = True
    record_phase(
        "mark_notified_auto_adopt",
        phase_t0,
        auto_adopted=len(auto_adopted or []),
    )

    phase_t0 = deps.now()
    archived_count = deps.archive_terminal_tasks(state)
    record_phase("archive_terminal", phase_t0, archived=archived_count)

    phase_t0 = deps.now()
    batch_completions = deps.detect_batch_completions(
        state,
        [task["id"] for task in newly_done + newly_crashed],
    )
    record_phase("batch_completions", phase_t0, batches=len(batch_completions or []))

    phase_t0 = deps.now()
    deps.save_state(state)
    record_phase("save_state", phase_t0, tasks=len(state.get("tasks", [])))

    return WatchStatePhaseResult(
        recovered_launching_count=recovered_launching_count,
        recovered_queued_live_count=recovered_queued_live_count,
        newly_done=newly_done,
        newly_crashed=newly_crashed,
        crash_forensics=crash_forensics,
        oom_forensics=oom_forensics,
        resource_evictions=resource_evictions,
        events=events,
        qcount=qcount,
        auto_adopted=auto_adopted,
        auto_adopt_completed=auto_adopt_completed,
        archived_count=archived_count,
        batch_completions=batch_completions,
    )
