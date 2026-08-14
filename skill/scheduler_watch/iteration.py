"""Top-level watcher tick orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WatchIterationDeps:
    watch_respects_dispatch_intent: bool
    auto_adopt_interval_s: float
    defer_launch_outside_lock: bool
    archive_age_days: float
    resource_log_interval_s: float
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    notify: Callable[..., Any]
    read_dispatch_intent: Callable[[], dict | None]
    dispatch_intent_message: Callable[[dict], str]
    recover_stale_launching_tasks_outside_lock: Callable[..., int]
    preload_docker_images_outside_lock: Callable[[], Any]
    stage_migration_candidates_outside_lock: Callable[..., Any]
    stage_launch_candidates_outside_lock: Callable[[], Any]
    probe_all: Callable[[], list]
    collect_external_task_probe_data: Callable[..., dict]
    running_probe_snapshot_outside_lock: Callable[[str], tuple]
    terminal_diagnostics_snapshot_outside_lock: Callable[[Any, Any, str], Any]
    eta_refresh_due: Callable[[dict], bool]
    start_eta_refresh_background: Callable[[], bool]
    watch_phase_recorder_factory: Callable[[float], Any]
    watch_phase_warn_s: float
    run_watch_state_phase: Callable[..., Any]
    watch_state_phase_deps: Callable[[], Any]
    execute_deferred_launches: Callable[..., list]
    load_state_shared_snapshot: Callable[[str], dict]
    notify_terminal_transitions: Callable[..., Any]
    notify_dispatch_events: Callable[..., Any]
    compact_task_event_payload: Callable[[dict], dict]
    dispatch_cycle_log_payload: Callable[[dict, list, list, int], dict]
    sync_completed_results_outside_lock: Callable[[], Any]
    tend_claims_for_watch: Callable[[], Any]
    watcher_state_exists: Callable[[], bool]
    read_watcher_state_text: Callable[[], str]
    write_watcher_state_text: Callable[[str], Any]
    plan_watcher_tick: Callable[..., Any]
    resource_accounting_payload: Callable[[dict, list], dict]
    build_heartbeat_payload: Callable[[dict, list], dict]
    get_last_auto_adopt_at: Callable[[], float]
    set_last_auto_adopt_at: Callable[[float], Any]
    now: Callable[[], float]
    getpid: Callable[[], int]
    prefetch_resume_scans_outside_lock: Callable[..., Any] | None = None
    begin_history_batch: Callable[[], Any] = lambda: None
    end_history_batch: Callable[[], Any] = lambda: None
    flush_pending_history_updates: Callable[[], dict] = lambda: {}
    refresh_escalations_outside_lock: Callable[[], Any] = lambda: None
    probe_all_cached: Callable[..., list] | None = None
    watch_probe_cache_max_age_s: int = 30
    watch_pre_dispatch_probe_max_age_s: int = 10
    stage_launch_candidates_background: Callable[[], Any] | None = None
    record_eta_analysis_tasks: Callable[..., Any] = lambda *_args, **_kwargs: {}


def _notify_outer_error(deps: WatchIterationDeps, event_type: str, exc: Exception) -> None:
    deps.notify(event_type, {"error": str(exc)[:200]}, feishu_enabled=False)


def _touch_watcher_progress(deps: WatchIterationDeps, phase: str) -> None:
    """Publish liveness without changing heartbeat notification cadence."""
    try:
        raw = json.loads(deps.read_watcher_state_text()) if deps.watcher_state_exists() else {}
        state = raw if isinstance(raw, dict) else {}
        now = deps.now()
        if state.get("phase") != phase:
            state["phase_started_at"] = now
        state.update(
            {
                "pid": deps.getpid(),
                "last_progress_ts": now,
                "phase": phase,
            }
        )
        state.setdefault("started_at", now)
        deps.write_watcher_state_text(json.dumps(state))
    except Exception:
        # Progress telemetry must never stop scheduling.
        pass


def _skip_for_dispatch_intent(args, deps: WatchIterationDeps, *, phase: str) -> bool:
    if not deps.watch_respects_dispatch_intent or getattr(args, "ignore_dispatch_intent", False):
        return False
    intent = deps.read_dispatch_intent()
    if not intent:
        return False
    deps.notify(
        "watch_skipped_for_dispatch_intent",
        {
            "reason": deps.dispatch_intent_message(intent),
            "phase": phase,
        },
        feishu_enabled=False,
    )
    return True


def _start_early_eta_refresh(deps: WatchIterationDeps) -> bool:
    """Refresh settled running work while slow node probes run in parallel."""
    state = deps.load_state_shared_snapshot("watch:eta-preflight-snapshot")
    tasks = state.get("tasks", [])
    if not any(task.get("status") == "running" for task in tasks):
        return False
    if any(task.get("status") in ("queued", "launching") for task in tasks):
        return False
    return bool(
        deps.eta_refresh_due(state)
        and deps.start_eta_refresh_background()
    )


def _recover_launching_before_watch(deps: WatchIterationDeps) -> int:
    try:
        return deps.recover_stale_launching_tasks_outside_lock(
            purpose="watch:recover-launching"
        )
    except Exception as exc:
        _notify_outer_error(deps, "launching_recovery_error", exc)
        return 0


def _run_outside_lock_preflight(
    deps: WatchIterationDeps,
    *,
    record_phase: Callable[..., Any] = lambda *_args, **_kwargs: None,
) -> tuple[int, list]:
    phase_t0 = deps.now()
    recovered_launching_count = _recover_launching_before_watch(deps)
    record_phase("recover_launching_outer", phase_t0, recovered=recovered_launching_count)
    for fn, event_type in (
        (deps.preload_docker_images_outside_lock, "preload_error"),
        (deps.refresh_escalations_outside_lock, "escalation_refresh_error_outer"),
    ):
        phase_t0 = deps.now()
        try:
            fn()
        except Exception as exc:
            _notify_outer_error(deps, event_type, exc)
        record_phase(event_type.removesuffix("_error_outer"), phase_t0)
    phase_t0 = deps.now()
    try:
        if deps.probe_all_cached is not None:
            nodes = deps.probe_all_cached(max_age_s=deps.watch_probe_cache_max_age_s)
        else:
            nodes = deps.probe_all()
    except Exception as exc:
        _notify_outer_error(deps, "probe_all_error_outer", exc)
        nodes = []
    record_phase("probe_all", phase_t0, nodes=len(nodes))
    return recovered_launching_count, nodes


def _run_outside_lock_staging(
    nodes: list,
    deps: WatchIterationDeps,
    *,
    record_phase: Callable[..., Any] = lambda *_args, **_kwargs: None,
) -> None:
    """Stage future launches only after live task state has been persisted."""
    if deps.prefetch_resume_scans_outside_lock is not None:
        phase_t0 = deps.now()
        try:
            deps.prefetch_resume_scans_outside_lock(nodes)
        except Exception as exc:
            _notify_outer_error(deps, "resume_scan_prefetch_error_outer", exc)
        record_phase("resume_scan_prefetch", phase_t0)
    phase_t0 = deps.now()
    try:
        stage_launch = (
            deps.stage_launch_candidates_background
            or deps.stage_launch_candidates_outside_lock
        )
        stage_launch()
    except Exception as exc:
        _notify_outer_error(deps, "launch_staging_error_outer", exc)
    record_phase("launch_staging", phase_t0)
    phase_t0 = deps.now()
    try:
        deps.stage_migration_candidates_outside_lock(nodes=nodes)
    except Exception as exc:
        _notify_outer_error(deps, "migration_staging_error_outer", exc)
    record_phase("migration_staging", phase_t0)


def _auto_adopt_probe_plan(
    deps: WatchIterationDeps,
    nodes: list,
) -> tuple[bool, dict | None]:
    should_auto_adopt = (
        deps.auto_adopt_interval_s <= 0
        or deps.now() - deps.get_last_auto_adopt_at() >= deps.auto_adopt_interval_s
    )
    if not should_auto_adopt:
        return False, None
    try:
        alive_node_names = {
            str(node.get("name"))
            for node in nodes
            if node.get("name") and node.get("alive") is True
        }
        return True, deps.collect_external_task_probe_data(
            eligible_node_names=alive_node_names
        )
    except Exception as exc:
        _notify_outer_error(deps, "auto_adopt_probe_error_outer", exc)
        deps.set_last_auto_adopt_at(deps.now())
        return False, None


def _node_snapshot_requires_predispatch_refresh(nodes: list) -> bool:
    """Refresh only cache-derived snapshots; never repeat a live probe in one tick."""
    if not nodes:
        return True
    return any(
        str((node or {}).get("probe_fallback") or "") == "stale_node_probe_cache"
        for node in nodes
    )


def _has_pending_dispatch_work(state: dict) -> bool:
    return any(
        task.get("status") == "queued"
        for task in state.get("tasks", [])
    )


def _run_state_phase(
    args,
    nodes: list,
    recovered_launching_count: int,
    should_auto_adopt: bool,
    auto_adopt_probe_data: dict | None,
    watch_phase_recorder,
    deps: WatchIterationDeps,
) -> tuple[Any, dict]:
    record_phase = watch_phase_recorder.record
    _touch_watcher_progress(deps, "running_task_probe")
    phase_t0 = deps.now()
    running_probe_results, running_probe_ids = deps.running_probe_snapshot_outside_lock(
        "watch:running-probe")
    record_phase(
        "running_probe_snapshot",
        phase_t0,
        probed=len(running_probe_results or {}),
    )
    _touch_watcher_progress(deps, "terminal_diagnostics")
    phase_t0 = deps.now()
    terminal_diagnostics = deps.terminal_diagnostics_snapshot_outside_lock(
        running_probe_results,
        running_probe_ids,
        "watch:terminal-diagnostics",
    )
    record_phase(
        "terminal_diagnostics_snapshot",
        phase_t0,
        diagnosed=len(terminal_diagnostics or {}),
    )
    # A cache-derived beginning-of-cycle snapshot may have crossed the tighter
    # dispatch freshness threshold while terminal checks ran. A live snapshot
    # must never trigger another full probe in the same tick: slow diagnostics
    # previously caused two consecutive multi-node SSH sweeps.
    needs_predispatch_refresh = False
    if (
        deps.probe_all_cached is not None
        and _node_snapshot_requires_predispatch_refresh(nodes)
    ):
        try:
            pending_state = deps.load_state_shared_snapshot(
                "watch:pre-dispatch-pending-snapshot"
            )
            needs_predispatch_refresh = _has_pending_dispatch_work(pending_state)
        except Exception as exc:
            # If the cheap queue snapshot fails, retain the conservative live
            # probe rather than dispatch from an old resource snapshot.
            needs_predispatch_refresh = True
            _notify_outer_error(deps, "pre_dispatch_pending_snapshot_error", exc)
    if needs_predispatch_refresh:
        _touch_watcher_progress(deps, "pre_dispatch_probe")
        phase_t0 = deps.now()
        try:
            refreshed_nodes = deps.probe_all_cached(
                max_age_s=deps.watch_pre_dispatch_probe_max_age_s
            )
            if refreshed_nodes:
                nodes[:] = refreshed_nodes
        except Exception as exc:
            _notify_outer_error(deps, "pre_dispatch_probe_error_outer", exc)
        record_phase(
            "pre_dispatch_probe",
            phase_t0,
            nodes=len(nodes),
            max_age_s=deps.watch_pre_dispatch_probe_max_age_s,
        )

    _touch_watcher_progress(deps, "state_update_and_dispatch_plan")
    main_t0 = deps.now()
    deps.begin_history_batch()
    try:
        with deps.state_lock(purpose="watch:main"):
            phase_t0 = deps.now()
            state = deps.load_state()
            record_phase("load_state", phase_t0)
            watch_state = deps.run_watch_state_phase(
                state,
                nodes,
                recovered_launching_count=recovered_launching_count,
                running_probe_results=running_probe_results,
                running_probe_ids=running_probe_ids,
                eta_tail_outputs=None,
                eta_tail_ids=None,
                terminal_diagnostics=terminal_diagnostics,
                should_auto_adopt=should_auto_adopt,
                auto_adopt_probe_data=auto_adopt_probe_data,
                defer_launches=deps.defer_launch_outside_lock,
                record_phase=record_phase,
                deps=deps.watch_state_phase_deps(),
                defer_eta_refresh=True,
            )
    finally:
        deps.end_history_batch()
    record_phase("watch_main_total", main_t0)
    return watch_state, state


def _notify_watch_outputs(
    watch_state,
    state: dict,
    nodes: list,
    events: list,
    qcount: int,
    watch_phase_recorder,
    deps: WatchIterationDeps,
) -> None:
    if watch_phase_recorder:
        deps.notify("watch_phase_slow", watch_phase_recorder.payload(), feishu_enabled=False)
    if watch_state.recovered_launching_count:
        deps.notify(
            "launching_state_recovered",
            {"reverted_count": watch_state.recovered_launching_count},
            feishu_enabled=False,
        )
    if watch_state.recovered_queued_live_count:
        deps.notify(
            "queued_live_state_recovered",
            {"recovered_count": watch_state.recovered_queued_live_count},
            feishu_enabled=False,
        )
    deps.notify_terminal_transitions(
        watch_state.newly_done,
        watch_state.newly_crashed,
        notify_fn=deps.notify,
        compact_task_payload_fn=deps.compact_task_event_payload,
    )
    for payload in watch_state.crash_forensics:
        deps.notify("crash_forensics", payload, feishu_enabled=False)
    for payload in watch_state.oom_forensics:
        deps.notify("oom_forensics", payload, feishu_enabled=False)
    for payload in watch_state.resource_evictions:
        deps.notify("resource_pressure_evicted", payload, feishu_enabled=False)
    deps.notify_dispatch_events(
        events,
        notify_fn=deps.notify,
        compact_task_payload_fn=deps.compact_task_event_payload,
    )
    deps.notify(
        "dispatch_cycle",
        deps.dispatch_cycle_log_payload(state, nodes, events, qcount),
        feishu_enabled=False,
    )
    try:
        deps.sync_completed_results_outside_lock()
    except Exception as exc:
        _notify_outer_error(deps, "result_sync_error_outer", exc)
    for task in watch_state.auto_adopted:
        deps.notify("task_auto_adopted", deps.compact_task_event_payload(task))
    for batch in watch_state.batch_completions:
        deps.notify("batch_complete", batch)
    if watch_state.archived_count:
        deps.notify(
            "archived_terminal_tasks",
            {"count": watch_state.archived_count, "age_days": deps.archive_age_days},
            feishu_enabled=False,
        )


def _emit_watcher_tick(args, nodes: list, deps: WatchIterationDeps) -> None:
    me = json.loads(deps.read_watcher_state_text()) if deps.watcher_state_exists() else {}
    tick = deps.plan_watcher_tick(
        me,
        now=deps.now(),
        pid=deps.getpid(),
        interval=getattr(args, "interval", 60),
        heartbeat=getattr(args, "heartbeat", 3600),
        resource_log_interval=getattr(args, "resource_log_interval", deps.resource_log_interval_s),
    )
    me = tick.state
    if tick.emit_resource_log:
        with deps.state_lock(shared=True, purpose="watch:resource-accounting-snapshot"):
            state = deps.load_state()
        deps.notify(
            "node_cpu_accounting",
            deps.resource_accounting_payload(state, nodes),
            feishu_enabled=False,
        )
    if tick.emit_heartbeat:
        with deps.state_lock(shared=True, purpose="watch:heartbeat-snapshot"):
            state = deps.load_state()
        deps.notify("heartbeat", deps.build_heartbeat_payload(state, nodes))
    if tick.dirty:
        try:
            deps.write_watcher_state_text(json.dumps(me))
        except Exception:
            pass


def _record_eta_lifecycle(watch_state, state: dict, deps: WatchIterationDeps) -> None:
    """Capture launch/terminal ETA records without holding scheduler state_lock."""
    targets = [
        task for task in state.get("tasks", [])
        if task.get("status") == "running"
    ]
    targets.extend(watch_state.newly_done)
    targets.extend(watch_state.newly_crashed)
    try:
        result = deps.record_eta_analysis_tasks(
            targets,
            include_periodic=False,
        ) or {}
    except Exception as exc:
        _notify_outer_error(deps, "eta_analysis_log_error", exc)
        return
    errors = result.get("errors") or []
    if errors:
        deps.notify(
            "eta_analysis_log_error",
            {"count": len(errors), "errors": errors[:5]},
            feishu_enabled=False,
        )


def run_watch_iteration(args, *, deps: WatchIterationDeps) -> None:
    """Run one watcher probe/update/notify heartbeat cycle."""
    if _skip_for_dispatch_intent(args, deps, phase="start"):
        _touch_watcher_progress(deps, "dispatch_intent_wait")
        return

    _touch_watcher_progress(deps, "preflight")
    watch_phase_recorder = deps.watch_phase_recorder_factory(deps.watch_phase_warn_s)
    phase_t0 = deps.now()
    try:
        eta_started_early = _start_early_eta_refresh(deps)
    except Exception as exc:
        eta_started_early = False
        _notify_outer_error(deps, "eta_refresh_background_early_start_error", exc)
    watch_phase_recorder.record(
        "eta_refresh_background_early_start",
        phase_t0,
        started=eta_started_early,
    )
    recovered_launching_count, nodes = _run_outside_lock_preflight(
        deps,
        record_phase=watch_phase_recorder.record,
    )
    if _skip_for_dispatch_intent(args, deps, phase="before_state_lock"):
        _touch_watcher_progress(deps, "dispatch_intent_wait")
        return
    phase_t0 = deps.now()
    should_auto_adopt, auto_adopt_probe_data = _auto_adopt_probe_plan(deps, nodes)
    watch_phase_recorder.record(
        "auto_adopt_probe",
        phase_t0,
        enabled=should_auto_adopt,
        alive_nodes=sum(1 for node in nodes if node.get("alive") is True),
    )
    watch_state, state = _run_state_phase(
        args,
        nodes,
        recovered_launching_count,
        should_auto_adopt,
        auto_adopt_probe_data,
        watch_phase_recorder,
        deps,
    )
    try:
        deps.flush_pending_history_updates()
    except Exception as exc:
        _notify_outer_error(deps, "history_batch_flush_error_outer", exc)
    if watch_state.auto_adopt_completed:
        deps.set_last_auto_adopt_at(deps.now())

    _touch_watcher_progress(deps, "launching")
    events = watch_state.events
    qcount = watch_state.qcount
    had_deferred_launches = any(event.get("type") == "launch_intent" for event in events)
    events = deps.execute_deferred_launches(events, mark_notified_launch=True)
    if had_deferred_launches:
        state = deps.load_state_shared_snapshot("watch:post-launch-snapshot")

    _record_eta_lifecycle(watch_state, state, deps)

    _touch_watcher_progress(deps, "eta_refresh_background")
    phase_t0 = deps.now()
    eta_started = eta_started_early
    if not eta_started:
        try:
            eta_started = bool(
                any(task.get("status") == "running" for task in state.get("tasks", []))
                and deps.eta_refresh_due(state)
                and deps.start_eta_refresh_background()
            )
        except Exception as exc:
            eta_started = False
            _notify_outer_error(deps, "eta_refresh_background_start_error", exc)
    watch_phase_recorder.record(
        "eta_refresh_background_start",
        phase_t0,
        started=eta_started,
        started_early=eta_started_early,
    )

    _touch_watcher_progress(deps, "staging")
    _run_outside_lock_staging(
        nodes,
        deps,
        record_phase=watch_phase_recorder.record,
    )

    _touch_watcher_progress(deps, "notifications_and_sync")
    _notify_watch_outputs(
        watch_state,
        state,
        nodes,
        events,
        qcount,
        watch_phase_recorder,
        deps,
    )
    _touch_watcher_progress(deps, "claim_maintenance")
    deps.tend_claims_for_watch()
    _emit_watcher_tick(args, nodes, deps)
    _touch_watcher_progress(deps, "sleeping")
