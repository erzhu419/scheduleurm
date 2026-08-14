"""Dispatch-intent file helpers used to let watch yield to bulk dispatch."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional


PidAlive = Callable[[object], bool]


def read_dispatch_intent(
    path: str | Path,
    *,
    pid_alive: PidAlive,
    now: Optional[float] = None,
) -> Optional[dict]:
    intent_path = Path(path)
    try:
        raw = json.loads(intent_path.read_text())
    except FileNotFoundError:
        return None
    except Exception:
        _unlink_quietly(intent_path)
        return None
    if not isinstance(raw, dict):
        return None
    current_time = time.time() if now is None else float(now)
    try:
        expires_at = float(raw.get("expires_at") or 0)
    except Exception:
        expires_at = 0.0
    pid = raw.get("pid")
    if expires_at and expires_at < current_time:
        _unlink_quietly(intent_path)
        return None
    if pid and not pid_alive(pid):
        _unlink_quietly(intent_path)
        return None
    return raw


def write_dispatch_intent(
    path: str | Path,
    *,
    label: str = "",
    ttl_s: Optional[float] = None,
    default_ttl_s: float = 900,
    pid: Optional[int] = None,
    now_fn: Callable[[], float] = time.time,
) -> dict:
    ttl = default_ttl_s if ttl_s is None else max(1.0, float(ttl_s))
    payload = {
        "pid": os.getpid() if pid is None else int(pid),
        "started_at": now_fn(),
        "expires_at": now_fn() + ttl,
        "label": label or "dispatch",
    }
    _atomic_write_json(Path(path), payload)
    return payload


def clear_dispatch_intent(path: str | Path, *, pid: Optional[int] = None) -> None:
    intent_path = Path(path)
    current_pid = os.getpid() if pid is None else int(pid)
    try:
        raw = json.loads(intent_path.read_text())
        if raw.get("pid") not in (None, current_pid):
            return
    except FileNotFoundError:
        return
    except Exception:
        pass
    _unlink_quietly(intent_path)


def dispatch_intent_message(intent: dict, *, now: Optional[float] = None) -> str:
    current_time = time.time() if now is None else float(now)
    age = max(0.0, current_time - float(intent.get("started_at") or current_time))
    ttl = max(0.0, float(intent.get("expires_at") or current_time) - current_time)
    return (
        f"bulk dispatch active pid={intent.get('pid')} "
        f"label={intent.get('label') or 'dispatch'} age={age:.0f}s ttl={ttl:.0f}s"
    )


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except Exception:
        pass
