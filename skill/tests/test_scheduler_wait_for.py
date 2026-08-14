from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from skill.scheduler_commands.wait_for import WaitForDeps, cmd_wait_for


def _deps(calls, *, states, now_values=None, owns_refresh=True):
    state_iter = iter(states)
    now_iter = iter(now_values or [0.0] * 100)
    output = []

    @contextmanager
    def state_lock(**kwargs):
        calls.append(("lock_enter", kwargs))
        try:
            yield
        finally:
            calls.append(("lock_exit", kwargs))

    def load_state():
        state = next(state_iter)
        calls.append(("load_state", state))
        return state

    @contextmanager
    def refresh_writer_lease():
        calls.append(("refresh_lease", owns_refresh))
        yield owns_refresh

    deps = WaitForDeps(
        state_lock=state_lock,
        load_state=load_state,
        save_state=lambda state: calls.append(("save_state", state)),
        recover_stale_launching_tasks=lambda state: calls.append(("recover", state)),
        update_running_tasks=lambda state, **kwargs: calls.append(("update_running", state, kwargs)),
        running_probe_snapshot_outside_lock=lambda purpose: calls.append(("running_probe", purpose)) or ({"r": 1}, {"t1"}),
        eta_tail_snapshot_outside_lock=lambda purpose: calls.append(("eta_tail", purpose)) or ({"eta": 1}, {"t1"}),
        refresh_writer_lease=refresh_writer_lease,
        now=lambda: next(now_iter),
        sleep=lambda seconds: calls.append(("sleep", seconds)),
        print_fn=lambda message: output.append(message),
    )
    return deps, output


def test_wait_for_returns_when_matching_signature_is_terminal():
    calls = []
    deps, output = _deps(
        calls,
        states=[{"tasks": [{"id": "t1", "signature": "kg/a", "status": "done"}]}],
    )

    rc = cmd_wait_for(
        SimpleNamespace(signature="kg/*", task_ids=[], timeout=30, poll=5, verbose=False, refresh=False),
        deps=deps,
    )

    assert rc == 0
    assert output == ["[wait-for kg/*] all 1 terminal: 1 done, 0 failed, 0 cancelled"]
    assert ("lock_enter", {"shared": True, "purpose": "wait-for:snapshot"}) in calls


def test_wait_for_refreshes_running_state_before_matching():
    calls = []
    deps, _output = _deps(
        calls,
        states=[{"tasks": [{"id": "t1", "signature": "kg/a", "status": "done"}]}],
    )

    rc = cmd_wait_for(
        SimpleNamespace(signature=None, task_ids=["t1"], timeout=30, poll=5, verbose=False, refresh=True),
        deps=deps,
    )

    assert rc == 0
    assert calls[:3] == [
        ("refresh_lease", True),
        ("running_probe", "wait-for:running-probe"),
        ("eta_tail", "wait-for:eta-tail"),
    ]
    assert ("lock_enter", {"purpose": "wait-for:update"}) in calls
    assert any(call[0] == "update_running" for call in calls)
    assert any(call[0] == "save_state" for call in calls)


def test_wait_for_refresh_is_readonly_while_watcher_owns_writer_lease():
    calls = []
    deps, _output = _deps(
        calls,
        states=[{"tasks": [{"id": "t1", "signature": "kg/a", "status": "done"}]}],
        owns_refresh=False,
    )

    rc = cmd_wait_for(
        SimpleNamespace(signature=None, task_ids=["t1"], timeout=30, poll=5, verbose=False, refresh=True),
        deps=deps,
    )

    assert rc == 0
    assert ("refresh_lease", False) in calls
    assert ("lock_enter", {"shared": True, "purpose": "wait-for:snapshot"}) in calls
    assert not any(call[0] in {"running_probe", "eta_tail", "update_running", "save_state"} for call in calls)


def test_wait_for_times_out_when_no_task_ever_matches():
    calls = []
    deps, output = _deps(
        calls,
        states=[{"tasks": []}, {"tasks": []}],
        now_values=[0.0, 0.0, 2.0],
    )

    rc = cmd_wait_for(
        SimpleNamespace(signature="missing", task_ids=[], timeout=1, poll=5, verbose=False, refresh=False),
        deps=deps,
    )

    assert rc == 2
    assert output == ["[wait-for] timeout: no matching tasks ever found"]
    assert ("sleep", 5) in calls
