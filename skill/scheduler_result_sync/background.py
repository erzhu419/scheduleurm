"""Detached, single-worker execution for result synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ResultSyncBackgroundDeps:
    worker_lock: Callable[..., Any]
    sync_results: Callable[[], Any]
    lock_timeout_type: type[BaseException]
    scheduler_dir: Path
    python_executable: str
    log_path: Path
    popen: Callable[..., Any]
    stderr_to_stdout: Any


def run_result_sync_worker(*, deps: ResultSyncBackgroundDeps) -> bool:
    """Run one bounded sync pass while holding the cross-process worker lock."""
    try:
        with deps.worker_lock(timeout_s=0, purpose="result-sync:worker"):
            deps.sync_results()
        return True
    except deps.lock_timeout_type:
        return False


def request_result_sync_background(holder: dict, *, deps: ResultSyncBackgroundDeps) -> bool:
    """Start a detached worker unless this watcher already has one alive."""
    current = holder.get("process")
    if current is not None and current.poll() is None:
        return False

    deps.log_path.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(deps.scheduler_dir)!r}); "
        "import scheduler as _scheduler; "
        "_scheduler._run_result_sync_worker()"
    )
    log_fh = open(deps.log_path, "ab")
    try:
        process = deps.popen(
            [deps.python_executable, "-c", code],
            stdin=None,
            stdout=log_fh,
            stderr=deps.stderr_to_stdout,
            cwd=str(deps.scheduler_dir),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()
    holder["process"] = process
    return True
