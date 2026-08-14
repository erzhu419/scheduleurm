"""Watcher shutdown helpers for control-plane subprocesses."""

from __future__ import annotations

import os
import signal
import threading


class WatcherShutdownRequested(BaseException):
    pass


SHUTDOWN_REQUESTED = False
CONTROL_SUBPROCESS_PGIDS: set[int] = set()
CONTROL_SUBPROCESS_LOCK = threading.Lock()


def reset_watcher_shutdown() -> None:
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = False


def watcher_shutdown_requested() -> bool:
    return bool(SHUTDOWN_REQUESTED)


def terminate_control_process_group(pgid: int, sig=signal.SIGTERM) -> None:
    try:
        os.killpg(int(pgid), sig)
        return
    except ProcessLookupError:
        return
    except Exception:
        pass
    try:
        os.kill(int(pgid), sig)
    except Exception:
        pass


def terminate_active_control_subprocesses(sig=signal.SIGTERM) -> None:
    with CONTROL_SUBPROCESS_LOCK:
        pgids = list(CONTROL_SUBPROCESS_PGIDS)
    for pgid in pgids:
        terminate_control_process_group(pgid, sig=sig)


def request_watcher_shutdown(_sig=None, _frame=None) -> None:
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    terminate_active_control_subprocesses(signal.SIGTERM)


def raise_if_watcher_shutdown() -> None:
    if watcher_shutdown_requested():
        raise WatcherShutdownRequested("watcher shutdown requested")


def register_control_subprocess(proc) -> int:
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = int(proc.pid)
    with CONTROL_SUBPROCESS_LOCK:
        CONTROL_SUBPROCESS_PGIDS.add(pgid)
        shutdown = SHUTDOWN_REQUESTED
    if shutdown:
        terminate_control_process_group(pgid, sig=signal.SIGTERM)
    return pgid


def unregister_control_subprocess(pgid: int | None) -> None:
    if pgid is None:
        return
    with CONTROL_SUBPROCESS_LOCK:
        CONTROL_SUBPROCESS_PGIDS.discard(int(pgid))
