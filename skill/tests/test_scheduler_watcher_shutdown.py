from __future__ import annotations

import pytest

from skill import scheduler_watcher_shutdown as shutdown


class FakeProc:
    pid = 12345


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    shutdown.reset_watcher_shutdown()
    shutdown.CONTROL_SUBPROCESS_PGIDS.clear()
    yield
    shutdown.reset_watcher_shutdown()
    shutdown.CONTROL_SUBPROCESS_PGIDS.clear()


def test_raise_if_watcher_shutdown_uses_base_exception_to_escape_broad_handlers():
    shutdown.request_watcher_shutdown()

    with pytest.raises(shutdown.WatcherShutdownRequested):
        try:
            shutdown.raise_if_watcher_shutdown()
        except Exception as e:  # pragma: no cover - BaseException should skip this
            raise AssertionError("shutdown was swallowed by Exception") from e


def test_register_unregister_control_subprocess_tracks_process_group(monkeypatch):
    monkeypatch.setattr(shutdown.os, "getpgid", lambda pid: 456)
    killed = []
    monkeypatch.setattr(
        shutdown,
        "terminate_control_process_group",
        lambda pgid, sig=None: killed.append(pgid),
    )

    pgid = shutdown.register_control_subprocess(FakeProc())
    assert pgid == 456
    assert shutdown.CONTROL_SUBPROCESS_PGIDS == {456}
    assert killed == []

    shutdown.unregister_control_subprocess(pgid)
    assert shutdown.CONTROL_SUBPROCESS_PGIDS == set()


def test_register_after_shutdown_immediately_terminates_group(monkeypatch):
    monkeypatch.setattr(shutdown.os, "getpgid", lambda pid: 456)
    killed = []
    monkeypatch.setattr(
        shutdown,
        "terminate_control_process_group",
        lambda pgid, sig=None: killed.append(pgid),
    )

    shutdown.request_watcher_shutdown()
    pgid = shutdown.register_control_subprocess(FakeProc())

    assert pgid == 456
    assert killed == [456]


def test_request_shutdown_terminates_all_registered_groups(monkeypatch):
    killed = []
    monkeypatch.setattr(
        shutdown,
        "terminate_control_process_group",
        lambda pgid, sig=None: killed.append(pgid),
    )
    shutdown.CONTROL_SUBPROCESS_PGIDS.update({1, 2})

    shutdown.request_watcher_shutdown()

    assert shutdown.watcher_shutdown_requested() is True
    assert sorted(killed) == [1, 2]
