from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skill import scheduler_locks as locks


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_lock_timeout_value_normalizes_invalid_values():
    assert locks.lock_timeout_value(None) is None
    assert locks.lock_timeout_value("bad") is None
    assert locks.lock_timeout_value(-1) is None
    assert locks.lock_timeout_value(0) == 0
    assert locks.lock_timeout_value("1.5") == 1.5


def test_state_lock_logs_when_thresholds_are_zero(tmp_path):
    lock_file = tmp_path / ".lock"
    log_dir = tmp_path / "logs"

    with locks.state_lock(
        lock_file=lock_file,
        log_dir=log_dir,
        purpose="unit-state",
        warn_wait_s=0,
        warn_hold_s=0,
        argv=["scheduler.py", "unit"],
    ):
        pass

    rows = [
        json.loads(line)
        for line in (log_dir / "lock.log").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["purpose"] == "unit-state"
    assert rows[0]["shared"] is False
    assert rows[0]["argv"] == ["scheduler.py", "unit"]
    assert lock_file.exists()


def test_state_lock_timeout_when_another_process_holds_lock(tmp_path):
    lock_file = tmp_path / ".lock"
    log_dir = tmp_path / "logs"
    ready_file = tmp_path / "holder.ready"
    holder = """
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from skill import scheduler_locks as locks

with locks.state_lock(lock_file=Path(sys.argv[1]), log_dir=Path(sys.argv[2]), purpose="holder"):
    Path(sys.argv[4]).write_text("ready", encoding="utf-8")
    time.sleep(0.5)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(lock_file), str(log_dir), str(REPO_ROOT), str(ready_file)],
    )
    try:
        deadline = time.time() + 2
        while not ready_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert ready_file.exists()

        with pytest.raises(locks.SchedulerLockTimeout):
            with locks.state_lock(
                lock_file=lock_file,
                log_dir=log_dir,
                timeout_s=0.05,
                purpose="contender",
            ):
                pass
    finally:
        proc.wait(timeout=5)


def test_launch_exec_lock_uses_default_timeout_and_logs(tmp_path):
    lock_file = tmp_path / ".launch_exec.lock"
    log_dir = tmp_path / "logs"

    with locks.launch_exec_lock(
        lock_file=lock_file,
        log_dir=log_dir,
        default_timeout_s=1,
        purpose="unit-launch",
        warn_wait_s=0,
        warn_hold_s=0,
    ):
        pass

    rows = [
        json.loads(line)
        for line in (log_dir / "lock.log").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["purpose"] == "unit-launch"
    assert lock_file.exists()


def test_counting_file_lock_blocks_only_matching_prefix(tmp_path):
    log_dir = tmp_path / "logs"
    node001 = tmp_path / ".launch.node001"
    node002 = tmp_path / ".launch.node002"

    with locks.counting_file_lock(
        lock_file_prefix=node001,
        slots=1,
        log_dir=log_dir,
        timeout_s=1,
        purpose="node001-holder",
    ):
        with locks.counting_file_lock(
            lock_file_prefix=node002,
            slots=1,
            log_dir=log_dir,
            timeout_s=0.05,
            purpose="node002-contender",
        ):
            pass
        with pytest.raises(locks.SchedulerLockTimeout):
            with locks.counting_file_lock(
                lock_file_prefix=node001,
                slots=1,
                log_dir=log_dir,
                timeout_s=0.05,
                purpose="node001-contender",
            ):
                pass


def test_watcher_lifetime_lock_blocks_second_process(tmp_path):
    lock_file = tmp_path / ".watcher.lock"
    log_dir = tmp_path / "logs"
    ready_file = tmp_path / "watcher.ready"
    holder = """
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from skill import scheduler_locks as locks

with locks.watcher_lifetime_lock(lock_file=Path(sys.argv[1]), log_dir=Path(sys.argv[2])):
    Path(sys.argv[4]).write_text("ready", encoding="utf-8")
    time.sleep(0.5)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(lock_file), str(log_dir), str(REPO_ROOT), str(ready_file)],
    )
    try:
        deadline = time.time() + 2
        while not ready_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert ready_file.exists()

        with pytest.raises(locks.SchedulerLockTimeout) as exc:
            with locks.watcher_lifetime_lock(
                lock_file=lock_file,
                log_dir=log_dir,
                timeout_s=0,
                purpose="watch:contender",
            ):
                pass
        assert "scheduler watcher lock" in str(exc.value)
    finally:
        proc.wait(timeout=5)
