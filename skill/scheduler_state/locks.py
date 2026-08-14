"""File-lock helpers for scheduler state and launch execution."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional


class SchedulerLockTimeout(TimeoutError):
    pass


def lock_timeout_value(timeout_s) -> Optional[float]:
    if timeout_s is None:
        return None
    try:
        value = float(timeout_s)
    except Exception:
        return None
    return value if value >= 0 else None


def log_lock_delay(
    *,
    log_dir: str | Path,
    purpose: str,
    shared: bool,
    wait_s: float,
    hold_s: float,
    warn_wait_s: float,
    warn_hold_s: float,
    argv: Optional[Iterable[str]] = None,
) -> None:
    if wait_s < warn_wait_s and hold_s < warn_hold_s:
        return
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "pid": os.getpid(),
            "purpose": purpose,
            "shared": bool(shared),
            "wait_s": round(float(wait_s), 3),
            "hold_s": round(float(hold_s), 3),
            "argv": list(sys.argv[:] if argv is None else argv),
        }
        with open(log_path / "lock.log", "a", encoding="utf-8") as lf:
            lf.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _ensure_lock_file(lock_file: str | Path, log_dir: str | Path) -> Path:
    path = Path(lock_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@contextmanager
def file_lock(
    *,
    lock_file: str | Path,
    log_dir: str | Path,
    timeout_s=None,
    shared: bool = False,
    purpose: str = "state",
    timeout_label: str = "scheduler state lock",
    warn_wait_s: float = 2.0,
    warn_hold_s: float = 5.0,
    argv: Optional[Iterable[str]] = None,
):
    path = _ensure_lock_file(lock_file, log_dir)
    f = open(path, "r+")
    op = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    timeout = lock_timeout_value(timeout_s)
    start = time.time()
    if timeout is None:
        fcntl.flock(f, op)
    else:
        while True:
            try:
                fcntl.flock(f, op | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start >= timeout:
                    f.close()
                    raise SchedulerLockTimeout(
                        f"timed out after {timeout:.1f}s waiting for {timeout_label}"
                        f" ({purpose})"
                    )
                time.sleep(min(0.25, max(0.01, timeout / 20.0)))
    acquired_at = time.time()
    waited = acquired_at - start
    exc_raised = False
    try:
        yield
    except Exception:
        exc_raised = True
        raise
    finally:
        hold_s = time.time() - acquired_at
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
        log_lock_delay(
            log_dir=log_dir,
            purpose=purpose + (":exception" if exc_raised else ""),
            shared=shared,
            wait_s=waited,
            hold_s=hold_s,
            warn_wait_s=warn_wait_s,
            warn_hold_s=warn_hold_s,
            argv=argv,
        )


def state_lock(
    *,
    lock_file: str | Path,
    log_dir: str | Path,
    timeout_s=None,
    shared: bool = False,
    purpose: str = "state",
    warn_wait_s: float = 2.0,
    warn_hold_s: float = 5.0,
    argv: Optional[Iterable[str]] = None,
):
    return file_lock(
        lock_file=lock_file,
        log_dir=log_dir,
        timeout_s=timeout_s,
        shared=shared,
        purpose=purpose,
        timeout_label="scheduler state lock",
        warn_wait_s=warn_wait_s,
        warn_hold_s=warn_hold_s,
        argv=argv,
    )


def launch_exec_lock(
    *,
    lock_file: str | Path,
    log_dir: str | Path,
    default_timeout_s,
    timeout_s=None,
    purpose: str = "launch-exec",
    warn_wait_s: float = 2.0,
    warn_hold_s: float = 5.0,
    argv: Optional[Iterable[str]] = None,
):
    timeout = lock_timeout_value(default_timeout_s if timeout_s is None else timeout_s)
    return file_lock(
        lock_file=lock_file,
        log_dir=log_dir,
        timeout_s=timeout,
        shared=False,
        purpose=purpose,
        timeout_label="scheduler launch lock",
        warn_wait_s=warn_wait_s,
        warn_hold_s=warn_hold_s,
        argv=argv,
    )


@contextmanager
def counting_file_lock(
    *,
    lock_file_prefix: str | Path,
    slots: int,
    log_dir: str | Path,
    timeout_s=None,
    purpose: str = "slot-lock",
    timeout_label: str = "scheduler slot lock",
    warn_wait_s: float = 2.0,
    warn_hold_s: float = 5.0,
    argv: Optional[Iterable[str]] = None,
):
    """Acquire one of several file-lock slots sharing the same prefix."""
    try:
        slot_count = int(slots)
    except Exception:
        slot_count = 1
    slot_count = max(1, slot_count)
    prefix = Path(lock_file_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timeout = lock_timeout_value(timeout_s)
    start = time.time()
    handle = None
    slot_idx = None
    while handle is None:
        for idx in range(slot_count):
            path = prefix.with_name(f"{prefix.name}.{idx}.lock")
            path.touch()
            f = open(path, "r+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle = f
                slot_idx = idx
                break
            except BlockingIOError:
                f.close()
            except Exception:
                f.close()
                raise
        if handle is not None:
            break
        if timeout is not None and time.time() - start >= timeout:
            raise SchedulerLockTimeout(
                f"timed out after {timeout:.1f}s waiting for {timeout_label}"
                f" ({purpose})"
            )
        if timeout is None:
            sleep_s = 0.25
        else:
            sleep_s = min(0.25, max(0.01, timeout / 20.0))
        time.sleep(sleep_s)

    acquired_at = time.time()
    waited = acquired_at - start
    exc_raised = False
    try:
        yield slot_idx
    except Exception:
        exc_raised = True
        raise
    finally:
        hold_s = time.time() - acquired_at
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()
        log_lock_delay(
            log_dir=log_dir,
            purpose=f"{purpose}:slot{slot_idx}" + (":exception" if exc_raised else ""),
            shared=False,
            wait_s=waited,
            hold_s=hold_s,
            warn_wait_s=warn_wait_s,
            warn_hold_s=warn_hold_s,
            argv=argv,
        )


def watcher_lifetime_lock(
    *,
    lock_file: str | Path,
    log_dir: str | Path,
    timeout_s=0,
    purpose: str = "watch:lifetime",
    warn_wait_s: float = 0.1,
    warn_hold_s: float = 30 * 86400,
    argv: Optional[Iterable[str]] = None,
):
    return file_lock(
        lock_file=lock_file,
        log_dir=log_dir,
        timeout_s=timeout_s,
        shared=False,
        purpose=purpose,
        timeout_label="scheduler watcher lock",
        warn_wait_s=warn_wait_s,
        warn_hold_s=warn_hold_s,
        argv=argv,
    )
