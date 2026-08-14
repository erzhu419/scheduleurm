from __future__ import annotations

from contextlib import contextmanager

from skill.scheduler_result_sync.background import (
    ResultSyncBackgroundDeps,
    request_result_sync_background,
    run_result_sync_worker,
)


class _LockTimeout(Exception):
    pass


class _Process:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _deps(tmp_path, calls, *, worker_lock=None):
    @contextmanager
    def default_lock(**kwargs):
        calls.append(("lock_enter", kwargs))
        yield
        calls.append(("lock_exit", kwargs))

    def popen(args, **kwargs):
        calls.append(("popen", args, kwargs))
        return _Process()

    return ResultSyncBackgroundDeps(
        worker_lock=worker_lock or default_lock,
        sync_results=lambda: calls.append(("sync", None)),
        lock_timeout_type=_LockTimeout,
        scheduler_dir=tmp_path / "skill",
        python_executable="/usr/bin/python3",
        log_path=tmp_path / "logs" / "result_sync_worker.log",
        popen=popen,
        stderr_to_stdout="STDOUT",
    )


def test_background_request_deduplicates_live_worker(tmp_path):
    calls = []
    holder = {"process": None}
    deps = _deps(tmp_path, calls)

    assert request_result_sync_background(holder, deps=deps) is True
    assert request_result_sync_background(holder, deps=deps) is False

    popen_call = next(call for call in calls if call[0] == "popen")
    assert popen_call[1][0] == "/usr/bin/python3"
    assert "_run_result_sync_worker" in popen_call[1][2]
    assert popen_call[2]["start_new_session"] is True


def test_result_sync_worker_holds_dedicated_lock_while_syncing(tmp_path):
    calls = []

    assert run_result_sync_worker(deps=_deps(tmp_path, calls)) is True
    assert [call[0] for call in calls] == ["lock_enter", "sync", "lock_exit"]


def test_result_sync_worker_exits_when_another_worker_holds_lock(tmp_path):
    calls = []

    @contextmanager
    def busy_lock(**_kwargs):
        raise _LockTimeout("busy")
        yield

    assert run_result_sync_worker(
        deps=_deps(tmp_path, calls, worker_lock=busy_lock)
    ) is False
    assert calls == []
