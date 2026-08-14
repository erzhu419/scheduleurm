from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from skill.scheduler_watch_command import WatchCommandDeps, cmd_watch_locked


class _Shutdown(BaseException):
    pass


def _deps(
    calls,
    *,
    watcher_state_exists=False,
    watcher_state_text="{}",
    live_pid=None,
    smoke_enabled=False,
    smoke_error=None,
    recover_count=0,
    iteration_plan=None,
    now_values=None,
):
    state = {"tasks": []}
    iteration_plan = list(iteration_plan if iteration_plan is not None else [_Shutdown()])
    handlers = {}
    now_iter = iter(now_values) if now_values is not None else None

    @contextmanager
    def state_lock(**kwargs):
        calls.append(("lock_enter", kwargs))
        try:
            yield
        finally:
            calls.append(("lock_exit", kwargs))

    def smoke_test_envs():
        calls.append(("smoke", None))
        if smoke_error is not None:
            raise smoke_error

    def watch_iteration(args):
        calls.append(("watch_iteration", args))
        action = iteration_plan.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    def register_signal(sig, handler):
        calls.append(("signal", sig))
        handlers[sig] = handler

    deps = WatchCommandDeps(
        resource_log_interval_s=60,
        reset_watcher_shutdown=lambda: calls.append(("reset_shutdown", None)),
        set_hard_rule_mode=lambda mode: calls.append(("hard_rule_mode", mode)),
        configure_algorithm=lambda name: calls.append(("algorithm", name)),
        watcher_state_exists=lambda: calls.append(("watcher_exists", None)) or watcher_state_exists,
        read_watcher_state_text=lambda: calls.append(("watcher_read", None)) or watcher_state_text,
        ensure_watcher_state_parent=lambda: calls.append(("watcher_parent", None)),
        write_watcher_state_text=lambda text: calls.append(("watcher_write", json.loads(text))),
        unlink_watcher_state=lambda: calls.append(("watcher_unlink", None)),
        live_watcher_pid=lambda existing, **kwargs: calls.append(("live_pid", existing, kwargs)) or live_pid,
        pid_alive=lambda pid: calls.append(("pid_alive", pid)) or True,
        register_signal_handler=register_signal,
        sigterm="TERM",
        sigint="INT",
        request_watcher_shutdown=lambda *args: calls.append(("request_shutdown", args)),
        initial_watcher_state=lambda **kwargs: calls.append(("initial_state", kwargs)) or dict(kwargs),
        getpid=lambda: 321,
        now=lambda: next(now_iter) if now_iter is not None else 100.0,
        notify=lambda event, payload, **kwargs: calls.append(("notify", event, payload, kwargs)),
        smoke_test_envs_enabled=lambda: calls.append(("smoke_enabled", None)) or smoke_enabled,
        smoke_test_envs=smoke_test_envs,
        state_lock=state_lock,
        load_state=lambda: calls.append(("load_state", None)) or state,
        save_state=lambda saved: calls.append(("save_state", saved)),
        recover_stale_launching_tasks_outside_lock=lambda **kwargs: (
            calls.append(("recover_outside", kwargs)) or recover_count
        ),
        post_reboot_triage_announce=lambda: calls.append(("post_reboot", None)),
        watch_iteration=watch_iteration,
        watcher_shutdown_requested_type=_Shutdown,
        sleep=lambda seconds: calls.append(("sleep", seconds)),
    )
    return deps, handlers


def _args(**overrides):
    data = {
        "interval": 2,
        "heartbeat": 30,
        "resource_log_interval": 40,
        "hard_rule_mode": "warn",
        "algorithm": "legacy",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_watch_locked_refuses_existing_live_watcher():
    calls = []
    deps, _handlers = _deps(
        calls,
        watcher_state_exists=True,
        watcher_state_text='{"pid": 123}',
        live_pid=123,
    )

    with pytest.raises(SystemExit) as exc:
        cmd_watch_locked(_args(), deps=deps)

    assert "another watcher is already running (pid=123)" in str(exc.value)
    assert all(call[0] != "watcher_write" for call in calls)
    assert all(call[:2] != ("notify", "watcher_started") for call in calls)


def test_watch_locked_starts_recovers_and_cleans_up_on_shutdown():
    calls = []
    deps, handlers = _deps(calls, recover_count=2)

    cmd_watch_locked(_args(), deps=deps)

    assert handlers.keys() == {"TERM", "INT"}
    written = [call[1] for call in calls if call[0] == "watcher_write"]
    assert written == [{
        "pid": 321,
        "started_at": 100.0,
        "interval": 2,
        "heartbeat": 30,
        "resource_log_interval": 40,
    }]
    assert (
        "recover_outside",
        {"purpose": "watch-start:recover-launching"},
    ) in calls
    assert not any(call[0] == "lock_enter" for call in calls)
    events = [call[1] for call in calls if call[0] == "notify"]
    assert events == [
        "watcher_started",
        "env_smoke_test_skipped",
        "launching_state_recovered",
        "watcher_stopped",
    ]
    assert ("watcher_unlink", None) in calls


def test_watch_locked_logs_iteration_errors_and_keeps_looping():
    calls = []
    deps, _handlers = _deps(
        calls,
        iteration_plan=[RuntimeError("boom"), _Shutdown()],
    )

    cmd_watch_locked(_args(interval=1), deps=deps)

    events = [call for call in calls if call[0] == "notify"]
    assert any(call[1] == "iteration_error" and call[2] == {"error": "boom"} for call in events)
    assert [call[0] for call in calls].count("watch_iteration") == 2
    assert ("sleep", 1) in calls


def test_watch_locked_uses_fixed_rate_and_skips_sleep_after_slow_iteration():
    calls = []
    deps, _handlers = _deps(
        calls,
        iteration_plan=[None, _Shutdown()],
        now_values=[100.0, 100.0, 105.0, 105.0],
    )

    cmd_watch_locked(_args(interval=2), deps=deps)

    assert [call[0] for call in calls].count("watch_iteration") == 2
    assert not any(call[0] == "sleep" for call in calls)


def test_watch_locked_starts_and_stops_independent_eta_timer():
    calls = []
    deps, _handlers = _deps(calls)
    deps = WatchCommandDeps(
        **{
            **deps.__dict__,
            "start_eta_periodic_refresh": lambda: calls.append(("eta_timer_start", None)) or True,
            "stop_eta_periodic_refresh": lambda: calls.append(("eta_timer_stop", None)),
        }
    )

    cmd_watch_locked(_args(), deps=deps)

    assert ("eta_timer_start", None) in calls
    assert ("eta_timer_stop", None) in calls
    assert any(
        call[0] == "notify" and call[1] == "eta_periodic_refresh_started"
        for call in calls
    )
    assert calls.index(("eta_timer_start", None)) < calls.index(("eta_timer_stop", None))
