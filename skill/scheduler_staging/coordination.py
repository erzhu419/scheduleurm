"""Cross-process coordination for launch/checkpoint staging operations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable


def staging_key_digest(key: tuple) -> str:
    encoded = json.dumps(list(key), ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _marker_path(root: str | Path, key: tuple) -> Path:
    return Path(root) / "markers" / f"{staging_key_digest(key)}.done"


def persistent_staging_marker_hit(
    root: str | Path,
    key: tuple,
    *,
    ttl_s: int,
    now: Callable[[], float],
) -> bool:
    path = _marker_path(root, key)
    try:
        age = now() - path.stat().st_mtime
    except OSError:
        return False
    if age < 0 or age > ttl_s:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def mark_persistent_staging_success(
    root: str | Path,
    key: tuple,
    *,
    now: Callable[[], float],
) -> None:
    path = _marker_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = float(now())
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(f"{timestamp}\n", encoding="ascii")
    os.replace(tmp, path)
    os.utime(path, (timestamp, timestamp))


@contextmanager
def staging_key_guard(root: str | Path, key: tuple):
    """Yield False immediately when another process owns the same staging key."""
    lock_path = Path(root) / "locks" / f"{staging_key_digest(key)}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()
