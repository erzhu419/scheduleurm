from __future__ import annotations

import threading
import time

import pytest


def test_watcher_shutdown_terminates_active_control_subprocess_group(sch):
    sch._reset_watcher_shutdown()

    def request_shutdown():
        time.sleep(0.2)
        sch._request_watcher_shutdown()

    thread = threading.Thread(target=request_shutdown, daemon=True)
    start = time.monotonic()
    thread.start()
    try:
        with pytest.raises(sch._WatcherShutdownRequested):
            sch._run_ssh_subprocess(
                ["bash", "-lc", "sleep 30"],
                timeout=30,
                capture_output=True,
                text=True,
            )
    finally:
        sch._reset_watcher_shutdown()

    thread.join(timeout=2)
    assert time.monotonic() - start < 5
    assert sch._CONTROL_SUBPROCESS_PGIDS == set()


def test_run_ssh_subprocess_refuses_new_process_after_shutdown(sch):
    sch._request_watcher_shutdown()
    try:
        with pytest.raises(sch._WatcherShutdownRequested):
            sch._run_ssh_subprocess(
                ["bash", "-lc", "echo should-not-run"],
                timeout=5,
                capture_output=True,
                text=True,
            )
    finally:
        sch._reset_watcher_shutdown()
