from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

from skill.scheduler_watch_iteration import WatchIterationDeps, run_watch_iteration


def _watch_state(**overrides):
    data = {
        "recovered_launching_count": 0,
        "recovered_queued_live_count": 0,
        "newly_done": [],
        "newly_crashed": [],
        "crash_forensics": [],
        "oom_forensics": [],
        "resource_evictions": [],
        "events": [],
        "qcount": 0,
        "auto_adopted": [],
        "auto_adopt_completed": False,
        "archived_count": 0,
        "batch_completions": [],
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class _Recorder:
    def __init__(self, warn_s, *, clock=lambda: 0.0):
        self.warn_s = warn_s
        self.clock = clock
        self.phases = []

    def record(self, *args, **kwargs):
        if len(args) < 2:
            return None
        name, started_at = args[0], args[1]
        seconds = self.clock() - started_at
        if seconds < self.warn_s:
            return None
        item = {"phase": name, "seconds": round(seconds, 3)}
        item.update({k: v for k, v in kwargs.items() if v is not None})
        self.phases.append(item)
        return item

    def payload(self):
        return {"threshold_s": self.warn_s, "phases": list(self.phases)}

    def __bool__(self):
        return bool(self.phases)


def _deps(
    calls,
    *,
    state=None,
    watch_state=None,
    intent=None,
    probe_data=None,
    last_auto_adopt_at=0.0,
    now_values=None,
    recover_count=0,
    launch_events=None,
    post_launch_state=None,
    tick=None,
    watcher_state_text="{}",
):
    state = state if state is not None else {"tasks": []}
    watch_state = watch_state if watch_state is not None else _watch_state()
    now_iter = iter(now_values or [100.0] * 100)
    last_holder = {"value": last_auto_adopt_at}
    launch_events = launch_events if launch_events is not None else None
    post_launch_state = post_launch_state if post_launch_state is not None else state
    tick = tick or SimpleNamespace(state={"tick": True}, emit_resource_log=False, emit_heartbeat=False, dirty=False)

    @contextmanager
    def state_lock(**kwargs):
        calls.append(("lock_enter", kwargs))
        try:
            yield
        finally:
            calls.append(("lock_exit", kwargs))

    def load_state():
        calls.append(("load_state", None))
        return state

    def notify(event, payload, **kwargs):
        calls.append(("notify", event, payload, kwargs))

    def run_phase(state_arg, nodes, **kwargs):
        calls.append(("run_phase", state_arg, nodes, kwargs))
        kwargs["record_phase"]("phase_inside")
        return watch_state

    return WatchIterationDeps(
        watch_respects_dispatch_intent=True,
        auto_adopt_interval_s=60.0,
        defer_launch_outside_lock=True,
        archive_age_days=7.0,
        resource_log_interval_s=60.0,
        state_lock=state_lock,
        load_state=load_state,
        save_state=lambda state_arg: calls.append(("save_state", state_arg)),
        notify=notify,
        read_dispatch_intent=lambda: calls.append(("read_intent", None)) or intent,
        dispatch_intent_message=lambda intent_arg: f"intent:{intent_arg['id']}",
        recover_stale_launching_tasks_outside_lock=lambda **kwargs: (
            calls.append(("recover_outside", kwargs)) or recover_count
        ),
        preload_docker_images_outside_lock=lambda: calls.append(("preload", None)),
        stage_migration_candidates_outside_lock=lambda **kwargs: calls.append(("stage_migration", kwargs)),
        stage_launch_candidates_outside_lock=lambda: calls.append(("stage_launch", None)),
        probe_all=lambda: calls.append(("probe_all", None)) or [{"name": "n1", "alive": True}],
        prefetch_resume_scans_outside_lock=lambda nodes: calls.append(
            ("prefetch_resume", list(nodes))
        ),
        collect_external_task_probe_data=lambda **kwargs: calls.append(("auto_probe", kwargs)) or probe_data,
        running_probe_snapshot_outside_lock=lambda purpose: calls.append(("running_probe", purpose)) or ({"r": 1}, {"t1"}),
        terminal_diagnostics_snapshot_outside_lock=lambda results, ids, purpose: (
            calls.append(("terminal_diag", results, ids, purpose)) or {"diag": 1}
        ),
        eta_refresh_due=lambda state_arg: True,
        start_eta_refresh_background=lambda: calls.append(("eta_background", None)) or True,
        watch_phase_recorder_factory=lambda warn_s: calls.append(("recorder", warn_s)) or _Recorder(
            warn_s,
            clock=lambda: next(now_iter),
        ),
        watch_phase_warn_s=9.0,
        run_watch_state_phase=run_phase,
        watch_state_phase_deps=lambda: calls.append(("phase_deps", None)) or object(),
        execute_deferred_launches=lambda events, **kwargs: (
            calls.append(("execute_deferred", list(events), kwargs)) or (
                launch_events if launch_events is not None else events
            )
        ),
        load_state_shared_snapshot=lambda purpose: calls.append(("shared_snapshot", purpose)) or post_launch_state,
        notify_terminal_transitions=lambda done, crashed, **kwargs: calls.append(("notify_terminal", done, crashed, kwargs)),
        notify_dispatch_events=lambda events, **kwargs: calls.append(("notify_dispatch_events", list(events), kwargs)),
        compact_task_event_payload=lambda task: {"id": task.get("id")},
        dispatch_cycle_log_payload=lambda state_arg, nodes, events, qcount: (
            calls.append(("dispatch_cycle_payload", state_arg, list(events), qcount)) or {"q": qcount}
        ),
        sync_completed_results_outside_lock=lambda: calls.append(("sync_results", None)),
        tend_claims_for_watch=lambda: calls.append(("tend_claims", None)),
        watcher_state_exists=lambda: calls.append(("watcher_exists", None)) or True,
        read_watcher_state_text=lambda: calls.append(("watcher_read", None)) or watcher_state_text,
        write_watcher_state_text=lambda text: calls.append(("watcher_write", text)),
        plan_watcher_tick=lambda me, **kwargs: calls.append(("plan_tick", me, kwargs)) or tick,
        resource_accounting_payload=lambda state_arg, nodes: calls.append(("resource_payload", state_arg, nodes)) or {"resource": True},
        build_heartbeat_payload=lambda state_arg, nodes: calls.append(("heartbeat_payload", state_arg, nodes)) or {"heartbeat": True},
        get_last_auto_adopt_at=lambda: last_holder["value"],
        set_last_auto_adopt_at=lambda value: calls.append(("set_last_auto", value)) or last_holder.__setitem__("value", value),
        now=lambda: next(now_iter),
        getpid=lambda: 1234,
        begin_history_batch=lambda: calls.append(("history_begin", None)),
        end_history_batch=lambda: calls.append(("history_end", None)),
        flush_pending_history_updates=lambda: calls.append(("history_flush", None)) or {},
    )


def test_dispatch_intent_skips_without_preflight_or_state_lock():
    calls = []

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=False),
        deps=_deps(calls, intent={"id": "bulk"}),
    )

    assert calls[:2] == [
        ("read_intent", None),
        (
            "notify",
            "watch_skipped_for_dispatch_intent",
            {"reason": "intent:bulk", "phase": "start"},
            {"feishu_enabled": False},
        ),
    ]
    assert all(call[0] != "lock_enter" for call in calls)
    assert all(call[0] != "probe_all" for call in calls)


def test_dispatch_intent_rechecked_after_preflight_before_state_lock():
    calls = []
    intents = iter([None, {"id": "bulk"}])

    deps = _deps(calls)
    deps = WatchIterationDeps(
        **{**deps.__dict__, "read_dispatch_intent": lambda: calls.append(("read_intent", None)) or next(intents)}
    )

    run_watch_iteration(SimpleNamespace(ignore_dispatch_intent=False), deps=deps)

    names = [call[0] for call in calls]
    assert "probe_all" in names
    assert any(
        call[0] == "notify"
        and call[1] == "watch_skipped_for_dispatch_intent"
        and call[2]["phase"] == "before_state_lock"
        for call in calls
    )
    assert not any(
        call[0] == "lock_enter" and call[1].get("purpose") == "watch:main"
        for call in calls
    )


def test_terminal_state_phase_runs_before_slow_staging_and_notifications_after():
    calls = []
    state = {"tasks": [{"id": "t1", "status": "running"}]}
    watch_state = _watch_state(
        recovered_launching_count=2,
        recovered_queued_live_count=1,
        events=[{"type": "launched", "task": {"id": "t1"}}],
        qcount=3,
        auto_adopted=[{"id": "adopted"}],
        auto_adopt_completed=True,
        archived_count=4,
        batch_completions=[{"prefix": "p"}],
    )

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=True, interval=60, heartbeat=3600),
        deps=_deps(
            calls,
            state=state,
            watch_state=watch_state,
            probe_data={"probe": True},
            recover_count=2,
            last_auto_adopt_at=0.0,
            now_values=[100.0] * 100,
        ),
    )

    names = [call[0] for call in calls]
    assert names.index("preload") < names.index("probe_all")
    assert names.index("eta_background") < names.index("probe_all")
    assert names.index("probe_all") < names.index("running_probe")
    auto_probe = next(call for call in calls if call[0] == "auto_probe")
    assert auto_probe[1] == {"eligible_node_names": {"n1"}}
    assert names.index("run_phase") < names.index("stage_launch")
    assert names.index("eta_background") < names.index("stage_launch")
    assert names.index("prefetch_resume") < names.index("stage_launch")
    assert names.index("stage_launch") < names.index("stage_migration")
    main_lock_index = next(
        i for i, call in enumerate(calls)
        if call[0] == "lock_enter" and call[1].get("purpose") == "watch:main"
    )
    assert names.index("running_probe") < main_lock_index
    assert main_lock_index < names.index("stage_launch")
    run_phase = next(call for call in calls if call[0] == "run_phase")
    assert run_phase[3]["defer_eta_refresh"] is True
    assert run_phase[3]["eta_tail_outputs"] is None
    assert run_phase[3]["eta_tail_ids"] is None
    assert ("set_last_auto", 100.0) in calls
    assert any(call[0] == "notify" and call[1] == "task_auto_adopted" for call in calls)
    assert any(call[0] == "notify" and call[1] == "archived_terminal_tasks" for call in calls)
    assert names.index("sync_results") < names.index("tend_claims")
    assert names.index("tend_claims") < names.index("plan_tick")
    assert all(call[1] != "watch_phase_slow" for call in calls if call[0] == "notify")


def test_watch_defers_eta_refresh_until_after_launch_when_pending_work_exists():
    calls = []
    state = {
        "tasks": [
            {"id": "running", "status": "running"},
            {"id": "queued", "status": "queued"},
        ]
    }

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=True),
        deps=_deps(calls, state=state),
    )

    names = [call[0] for call in calls]
    assert names.index("run_phase") < names.index("eta_background")
    assert names.count("eta_background") == 1


def test_watch_records_eta_lifecycle_outside_main_state_lock():
    calls = []
    running = {"id": "running", "status": "running", "started_at": 10.0}
    done = {"id": "done", "status": "done", "started_at": 1.0, "finished_at": 9.0}
    state = {"tasks": [running, done]}
    watch_state = _watch_state(newly_done=[done])
    deps = _deps(calls, state=state, watch_state=watch_state)

    def record_eta(tasks, *, include_periodic):
        calls.append(("eta_analysis", [task["id"] for task in tasks], include_periodic))
        return {"records": 2, "errors": []}

    deps = WatchIterationDeps(
        **{**deps.__dict__, "record_eta_analysis_tasks": record_eta}
    )
    run_watch_iteration(SimpleNamespace(ignore_dispatch_intent=True), deps=deps)

    eta_call = next(call for call in calls if call[0] == "eta_analysis")
    assert eta_call == ("eta_analysis", ["running", "done"], False)
    names = [call[0] for call in calls]
    assert names.index("execute_deferred") < names.index("eta_analysis")
    assert names.index("eta_analysis") < names.index("stage_launch")


def test_watch_refreshes_stale_probe_immediately_before_dispatch_state_phase():
    calls = []
    probes = []

    def cached_probe(*, max_age_s):
        probes.append(max_age_s)
        if len(probes) == 1:
            return [{
                "name": "n1",
                "snapshot": "initial",
                "probe_fallback": "stale_node_probe_cache",
            }]
        return [{"name": "n1", "snapshot": "pre-dispatch"}]

    deps = _deps(calls, state={"tasks": [{"id": "q1", "status": "queued"}]})
    deps = WatchIterationDeps(
        **{
            **deps.__dict__,
            "probe_all_cached": cached_probe,
            "watch_probe_cache_max_age_s": 30,
            "watch_pre_dispatch_probe_max_age_s": 10,
        }
    )

    run_watch_iteration(SimpleNamespace(ignore_dispatch_intent=True), deps=deps)

    assert probes == [30, 10]
    run_phase = next(call for call in calls if call[0] == "run_phase")
    assert run_phase[2] == [{"name": "n1", "snapshot": "pre-dispatch"}]


def test_watch_skips_strict_predispatch_probe_without_queued_work():
    calls = []
    probes = []

    def cached_probe(*, max_age_s):
        probes.append(max_age_s)
        return [{
            "name": "n1",
            "snapshot": "cached",
            "probe_fallback": "stale_node_probe_cache",
        }]

    deps = _deps(calls, state={"tasks": [{"id": "r1", "status": "running"}]})
    deps = WatchIterationDeps(
        **{
            **deps.__dict__,
            "probe_all_cached": cached_probe,
            "watch_probe_cache_max_age_s": 30,
            "watch_pre_dispatch_probe_max_age_s": 10,
        }
    )

    run_watch_iteration(SimpleNamespace(ignore_dispatch_intent=True), deps=deps)

    assert probes == [30]
    assert ("shared_snapshot", "watch:pre-dispatch-pending-snapshot") in calls


def test_watch_never_repeats_live_node_probe_in_same_iteration():
    calls = []
    probes = []

    def cached_probe(*, max_age_s):
        probes.append(max_age_s)
        return [{"name": "n1", "snapshot": "live"}]

    deps = _deps(calls)
    deps = WatchIterationDeps(
        **{
            **deps.__dict__,
            "probe_all_cached": cached_probe,
            "watch_probe_cache_max_age_s": 30,
            "watch_pre_dispatch_probe_max_age_s": 10,
        }
    )

    run_watch_iteration(SimpleNamespace(ignore_dispatch_intent=True), deps=deps)

    assert probes == [30]
    run_phase = next(call for call in calls if call[0] == "run_phase")
    assert run_phase[2] == [{"name": "n1", "snapshot": "live"}]


def test_watch_passes_dead_tasks_without_terminal_diagnostics_to_state_phase():
    calls = []
    probe_results = {
        "alive": {"state": "alive"},
        "dead1": {"state": "dead"},
        "dead2": {"state": "dead"},
    }

    deps = _deps(calls)
    deps = WatchIterationDeps(
        **{
            **deps.__dict__,
            "running_probe_snapshot_outside_lock": lambda purpose: (
                calls.append(("running_probe", purpose))
                or (probe_results, {"alive", "dead1", "dead2"})
            ),
            "terminal_diagnostics_snapshot_outside_lock": lambda results, ids, purpose: (
                calls.append(("terminal_diag", results, ids, purpose))
                or {"dead1": {"diagnosis": {"reason": "done"}}}
            ),
        }
    )

    run_watch_iteration(SimpleNamespace(ignore_dispatch_intent=True), deps=deps)

    run_phase = next(call for call in calls if call[0] == "run_phase")
    assert run_phase[3]["running_probe_ids"] == {"alive", "dead1", "dead2"}
    assert not any(
        call[0] == "notify"
        and call[1] == "terminal_diagnostics_deferred"
        for call in calls
    )


def test_auto_adopt_probe_error_throttles_and_runs_state_phase_without_adopt():
    calls = []

    def raising_deps():
        deps = _deps(calls, last_auto_adopt_at=0.0, now_values=[100.0] * 100)
        return WatchIterationDeps(
            **{
                **deps.__dict__,
                "collect_external_task_probe_data": (
                    lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
                ),
            }
        )

    run_watch_iteration(SimpleNamespace(ignore_dispatch_intent=True), deps=raising_deps())

    assert any(call[0] == "notify" and call[1] == "auto_adopt_probe_error_outer" for call in calls)
    assert ("set_last_auto", 100.0) in calls
    run_phase = next(call for call in calls if call[0] == "run_phase")
    assert run_phase[3]["should_auto_adopt"] is False
    assert run_phase[3]["auto_adopt_probe_data"] is None


def test_deferred_launch_reload_state_before_dispatch_cycle_payload():
    calls = []
    original = {"tasks": [{"id": "before"}]}
    reloaded = {"tasks": [{"id": "after"}]}
    watch_state = _watch_state(
        events=[{"type": "launch_intent", "task_id": "t1"}],
        qcount=1,
    )

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=True),
        deps=_deps(
            calls,
            state=original,
            post_launch_state=reloaded,
            watch_state=watch_state,
            launch_events=[{"type": "launched", "task": {"id": "t1"}}],
        ),
    )

    assert ("shared_snapshot", "watch:post-launch-snapshot") in calls
    payload_call = next(call for call in calls if call[0] == "dispatch_cycle_payload")
    assert payload_call[1] is reloaded
    assert payload_call[2] == [{"type": "launched", "task": {"id": "t1"}}]


def test_deferred_launch_execution_happens_after_watch_main_lock_exits():
    calls = []
    watch_state = _watch_state(
        events=[{"type": "launch_intent", "task_id": "t1"}],
        qcount=1,
    )

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=True),
        deps=_deps(calls, watch_state=watch_state),
    )

    execute_idx = next(i for i, call in enumerate(calls) if call[0] == "execute_deferred")
    main_exit_idx = next(
        i
        for i, call in enumerate(calls)
        if call[0] == "lock_exit" and call[1].get("purpose") == "watch:main"
    )
    assert main_exit_idx < execute_idx


def test_history_batch_flush_happens_after_watch_main_lock_exits():
    calls = []

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=True),
        deps=_deps(calls),
    )

    main_enter_idx = next(
        i for i, call in enumerate(calls)
        if call[0] == "lock_enter" and call[1].get("purpose") == "watch:main"
    )
    main_exit_idx = next(
        i for i, call in enumerate(calls)
        if call[0] == "lock_exit" and call[1].get("purpose") == "watch:main"
    )
    begin_idx = next(i for i, call in enumerate(calls) if call[0] == "history_begin")
    end_idx = next(i for i, call in enumerate(calls) if call[0] == "history_end")
    flush_idx = next(i for i, call in enumerate(calls) if call[0] == "history_flush")

    assert begin_idx < main_enter_idx < main_exit_idx < end_idx < flush_idx


def test_watch_publishes_in_cycle_progress_without_advancing_heartbeat():
    calls = []

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=True),
        deps=_deps(calls, watcher_state_text='{"last_heartbeat_ts": 77}'),
    )

    progress_states = [
        json.loads(call[1])
        for call in calls
        if call[0] == "watcher_write"
        and "last_progress_ts" in json.loads(call[1])
    ]
    phases = [state["phase"] for state in progress_states]
    assert phases[0] == "preflight"
    assert "running_task_probe" in phases
    assert "launching" in phases
    assert phases[-1] == "sleeping"
    assert all(state.get("last_heartbeat_ts") == 77 for state in progress_states)


def test_watch_main_total_slow_phase_is_reported():
    calls = []
    recorder = _Recorder(9.0)
    recorder.phases.append({"phase": "watch_main_total", "seconds": 10.0})
    deps = _deps(calls)
    deps = WatchIterationDeps(
        **{
            **deps.__dict__,
            "watch_phase_recorder_factory": lambda _warn_s: recorder,
        }
    )

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=True),
        deps=deps,
    )

    slow = [call for call in calls if call[0] == "notify" and call[1] == "watch_phase_slow"]
    assert len(slow) == 1
    assert slow[0][2]["phases"] == [{"phase": "watch_main_total", "seconds": 10.0}]


def test_tick_emits_resource_accounting_and_heartbeat_under_shared_locks():
    calls = []
    tick = SimpleNamespace(
        state={"last": 1},
        emit_resource_log=True,
        emit_heartbeat=True,
        dirty=True,
    )

    run_watch_iteration(
        SimpleNamespace(ignore_dispatch_intent=True, interval=5, heartbeat=10, resource_log_interval=15),
        deps=_deps(calls, tick=tick, watcher_state_text='{"old": 1}'),
    )

    assert any(
        call[0] == "lock_enter"
        and call[1].get("shared") is True
        and call[1].get("purpose") == "watch:resource-accounting-snapshot"
        for call in calls
    )
    assert any(
        call[0] == "lock_enter"
        and call[1].get("shared") is True
        and call[1].get("purpose") == "watch:heartbeat-snapshot"
        for call in calls
    )
    assert any(call[0] == "notify" and call[1] == "node_cpu_accounting" for call in calls)
    assert any(call[0] == "notify" and call[1] == "heartbeat" for call in calls)
    assert ("watcher_write", '{"last": 1}') in calls
