from __future__ import annotations

import pytest

from skill.scheduler_watcher_state import (
    initial_watcher_state,
    live_watcher_pid,
    plan_watcher_tick,
)


def test_initial_watcher_state_matches_startup_payload_shape():
    assert initial_watcher_state(
        pid=1234,
        started_at=10.5,
        interval=60,
        heartbeat=3600,
        resource_log_interval=120,
    ) == {
        "pid": 1234,
        "started_at": 10.5,
        "last_progress_ts": 10.5,
        "phase": "starting",
        "phase_started_at": 10.5,
        "last_heartbeat_ts": 0,
        "last_resource_log_ts": 0,
        "interval": 60,
        "heartbeat": 3600,
        "resource_log_interval": 120,
    }


def test_live_watcher_pid_returns_alive_pid_and_preserves_raw_value():
    calls = []

    assert live_watcher_pid(
        {"pid": "1234"},
        pid_alive=lambda pid: calls.append(pid) or True,
    ) == "1234"
    assert calls == ["1234"]


def test_live_watcher_pid_ignores_dead_pid():
    assert live_watcher_pid({"pid": 1234}, pid_alive=lambda _pid: False) is None


def test_live_watcher_pid_ignores_missing_or_invalid_state_without_probe():
    calls = []

    assert live_watcher_pid(None, pid_alive=calls.append) is None
    assert live_watcher_pid(["bad"], pid_alive=calls.append) is None
    assert live_watcher_pid({}, pid_alive=calls.append) is None
    assert live_watcher_pid({"pid": 0}, pid_alive=calls.append) is None
    assert calls == []


def test_live_watcher_pid_propagates_probe_errors():
    def boom(_pid):
        raise TypeError("bad pid")

    with pytest.raises(TypeError, match="bad pid"):
        live_watcher_pid({"pid": "bad"}, pid_alive=boom)


def test_plan_watcher_tick_initial_state_emits_both_periodic_events():
    plan = plan_watcher_tick(
        {},
        now=4000.0,
        pid=1234,
        interval=60,
        heartbeat=3600,
        resource_log_interval=60,
    )

    assert plan.emit_resource_log is True
    assert plan.emit_heartbeat is True
    assert plan.dirty is True
    assert plan.state == {
        "pid": 1234,
        "started_at": 4000.0,
        "last_progress_ts": 4000.0,
        "phase": "starting",
        "phase_started_at": 4000.0,
        "interval": 60,
        "heartbeat": 3600,
        "resource_log_interval": 60,
        "last_resource_log_ts": 4000.0,
        "last_heartbeat_ts": 4000.0,
    }


def test_plan_watcher_tick_preserves_existing_state_when_not_due():
    state = {
        "pid": 1,
        "started_at": 10.0,
        "interval": 60,
        "heartbeat": 3600,
        "resource_log_interval": 60,
        "last_resource_log_ts": 95.0,
        "last_heartbeat_ts": 50.0,
    }

    plan = plan_watcher_tick(
        state,
        now=100.0,
        pid=1234,
        interval=30,
        heartbeat=3600,
        resource_log_interval=60,
    )

    assert plan.emit_resource_log is False
    assert plan.emit_heartbeat is False
    assert plan.dirty is False
    assert plan.state is state
    assert plan.state["pid"] == 1
    assert plan.state["interval"] == 60


def test_plan_watcher_tick_can_disable_resource_log_interval():
    plan = plan_watcher_tick(
        {"last_heartbeat_ts": 95.0},
        now=100.0,
        pid=1234,
        interval=60,
        heartbeat=10,
        resource_log_interval=0,
    )

    assert plan.emit_resource_log is False
    assert plan.emit_heartbeat is False
    assert plan.dirty is False


def test_plan_watcher_tick_rebuilds_non_dict_state():
    plan = plan_watcher_tick(
        ["bad"],
        now=4000.0,
        pid=1234,
        interval=60,
        heartbeat=3600,
        resource_log_interval=60,
    )

    assert isinstance(plan.state, dict)
    assert plan.state["pid"] == 1234
    assert plan.emit_resource_log is True
    assert plan.emit_heartbeat is True
