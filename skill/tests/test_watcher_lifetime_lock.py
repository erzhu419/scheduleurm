from types import SimpleNamespace

import pytest


def test_cmd_watch_exits_when_watcher_lifetime_lock_is_held(monkeypatch, sch):
    def locked(*args, **kwargs):
        raise sch.SchedulerLockTimeout("timed out after 0.0s waiting for scheduler watcher lock")

    monkeypatch.setattr(sch, "watcher_lifetime_lock", locked, raising=False)

    with pytest.raises(SystemExit) as exc:
        sch.cmd_watch(SimpleNamespace())

    assert "scheduler watcher lock" in str(exc.value)
