from __future__ import annotations

import copy
import threading
import time
from contextlib import contextmanager

from skill.scheduler_eta.background import (
    EtaBackgroundRefreshDeps,
    _background_target,
    refresh_eta_snapshot,
    start_eta_periodic_refresh,
    start_eta_refresh_background,
    stop_eta_periodic_refresh,
)
from skill.scheduler_watch.shutdown import WatcherShutdownRequested


def _deps(
    *,
    state_lock,
    load_state,
    save_state,
    tail_outputs,
    refresh,
    due=lambda state: True,
    notify=lambda *a, **k: None,
    record_eta=lambda *_args, **_kwargs: {},
):
    return EtaBackgroundRefreshDeps(
        state_lock=state_lock,
        load_state=load_state,
        save_state=save_state,
        eta_refresh_due=due,
        eta_tail_targets=lambda state: (
            {
                "node001": [
                    (task, task["log_path"])
                    for task in state["tasks"]
                    if task.get("status") == "running" and task.get("log_path")
                ]
            },
            [],
        ),
        eta_tail_outputs_by_node=tail_outputs,
        refresh_eta_from_logs=refresh,
        notify=notify,
        now=time.time,
        record_eta_analysis_tasks=record_eta,
    )


def test_eta_background_parses_outside_lock_and_commits_only_eta_fields():
    initial = {
        "tasks": [{
            "id": "t1",
            "status": "running",
            "node": "node001",
            "started_at": 10.0,
            "log_path": "/tmp/t1.log",
            "eta_seconds": 999,
        }]
    }
    latest = copy.deepcopy(initial)
    latest["tasks"][0]["eta_probe_error"] = "old error"
    lock_stack = []
    lock_events = []
    saved = []
    eta_records = []

    @contextmanager
    def state_lock(**kwargs):
        mode = "shared" if kwargs.get("shared") else "writer"
        lock_events.append(("enter", mode, kwargs.get("purpose")))
        lock_stack.append(mode)
        try:
            yield
        finally:
            lock_stack.pop()
            lock_events.append(("exit", mode, kwargs.get("purpose")))

    def load_state():
        return copy.deepcopy(initial) if lock_stack[-1] == "shared" else latest

    def tail_outputs(by_node):
        assert lock_stack == []
        assert list(by_node) == ["node001"]
        return {"node001": "tail"}

    def refresh(snapshot, *, tail_outputs, probed_task_ids):
        assert lock_stack == []
        assert tail_outputs == {"node001": "tail"}
        assert probed_task_ids == {"t1"}
        task = snapshot["tasks"][0]
        task["eta_seconds"] = 123
        task["eta_source"] = "progress_rate"
        task["runtime_total_s_est"] = 456
        task.pop("eta_probe_error", None)

    def record_eta(tasks, *, include_periodic, force_periodic):
        assert lock_stack == []
        eta_records.append((copy.deepcopy(tasks), include_periodic, force_periodic))
        return {"records": len(tasks), "errors": []}

    result = refresh_eta_snapshot(
        deps=_deps(
            state_lock=state_lock,
            load_state=load_state,
            save_state=lambda state: saved.append(copy.deepcopy(state)),
            tail_outputs=tail_outputs,
            refresh=refresh,
            record_eta=record_eta,
        )
    )

    assert result["status"] == "ok"
    assert result["updated"] == 1
    assert latest["tasks"][0]["eta_seconds"] == 123
    assert latest["tasks"][0]["eta_source"] == "progress_rate"
    assert latest["tasks"][0]["runtime_total_s_est"] == 456
    assert "eta_probe_error" not in latest["tasks"][0]
    assert latest["_last_eta_refresh_at"] > 0
    assert saved[-1] == latest
    assert eta_records[0][0][0]["eta_seconds"] == 123
    assert eta_records[0][1] is True
    assert eta_records[0][2] is False
    assert result["eta_analysis_records"] == 1
    assert lock_events[0] == ("enter", "shared", "watch:eta-background-snapshot")
    assert ("enter", "writer", "watch:eta-background-commit") in lock_events


def test_eta_background_does_not_overwrite_a_completed_attempt():
    snapshot = {
        "tasks": [{
            "id": "t1",
            "status": "running",
            "node": "node001",
            "started_at": 10.0,
            "log_path": "/tmp/t1.log",
        }]
    }
    latest = copy.deepcopy(snapshot)
    latest["tasks"][0].update(status="done", eta_seconds=7)
    shared = {"value": False}

    @contextmanager
    def state_lock(**kwargs):
        shared["value"] = bool(kwargs.get("shared"))
        try:
            yield
        finally:
            shared["value"] = False

    def load_state():
        return copy.deepcopy(snapshot) if shared["value"] else latest

    def refresh(state, **_kwargs):
        state["tasks"][0]["eta_seconds"] = 999

    result = refresh_eta_snapshot(
        deps=_deps(
            state_lock=state_lock,
            load_state=load_state,
            save_state=lambda state: None,
            tail_outputs=lambda by_node: {"node001": "tail"},
            refresh=refresh,
        )
    )

    assert result["updated"] == 0
    assert latest["tasks"][0]["status"] == "done"
    assert latest["tasks"][0]["eta_seconds"] == 7


def test_eta_background_start_is_single_flight():
    entered = threading.Event()
    release = threading.Event()

    @contextmanager
    def state_lock(**_kwargs):
        yield

    state = {
        "tasks": [{
            "id": "t1",
            "status": "running",
            "node": "node001",
            "started_at": 10.0,
            "log_path": "/tmp/t1.log",
        }]
    }

    def tail_outputs(_by_node):
        entered.set()
        release.wait(timeout=2)
        return {"node001": "tail"}

    deps = _deps(
        state_lock=state_lock,
        load_state=lambda: copy.deepcopy(state),
        save_state=lambda latest: None,
        tail_outputs=tail_outputs,
        refresh=lambda state, **kwargs: None,
    )

    assert start_eta_refresh_background(deps=deps) is True
    assert entered.wait(timeout=1)
    assert start_eta_refresh_background(deps=deps) is False
    release.set()


def test_eta_background_force_bypasses_global_refresh_due_gate():
    state = {
        "tasks": [{
            "id": "force",
            "status": "running",
            "node": "node001",
            "started_at": 1.0,
            "log_path": "/tmp/force.log",
        }]
    }

    @contextmanager
    def state_lock(**_kwargs):
        yield

    refreshed = []
    result = refresh_eta_snapshot(
        deps=_deps(
            state_lock=state_lock,
            load_state=lambda: state,
            save_state=lambda latest: None,
            tail_outputs=lambda by_node: {"node001": "tail"},
            refresh=lambda snapshot, **kwargs: refreshed.append(snapshot),
            due=lambda _state: False,
        ),
        force=True,
    )

    assert result["status"] == "ok"
    assert len(refreshed) == 1


def test_eta_background_consumes_watcher_shutdown_exception():
    state = {
        "tasks": [{
            "id": "shutdown",
            "status": "running",
            "node": "node001",
            "started_at": 1.0,
            "log_path": "/tmp/shutdown.log",
        }]
    }

    @contextmanager
    def state_lock(**_kwargs):
        yield

    deps = _deps(
        state_lock=state_lock,
        load_state=lambda: state,
        save_state=lambda latest: None,
        tail_outputs=lambda _by_node: (_ for _ in ()).throw(
            WatcherShutdownRequested("stop")
        ),
        refresh=lambda snapshot, **kwargs: None,
    )

    _background_target(deps, False)


def test_eta_periodic_refresh_uses_fixed_timer_and_is_single_instance():
    fired = threading.Event()
    calls = []

    def start_refresh():
        calls.append(time.monotonic())
        fired.set()
        return True

    assert start_eta_periodic_refresh(start_refresh=start_refresh, interval_s=0.02) is True
    assert start_eta_periodic_refresh(start_refresh=start_refresh, interval_s=0.02) is False
    assert fired.wait(timeout=1.0)
    time.sleep(0.055)
    assert stop_eta_periodic_refresh(timeout_s=1.0) is True
    count_after_stop = len(calls)
    time.sleep(0.04)

    assert count_after_stop >= 2
    assert len(calls) == count_after_stop
