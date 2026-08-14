"""Scheduler JSON state persistence and queue-reset guards."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional


def load_json(path: str | Path, default, *, now_fn: Callable[[], float] = time.time):
    """Load JSON, quarantine empty/corrupt files, and never hide parse errors."""
    p = Path(path)
    if not p.exists():
        return default
    text = p.read_text()
    if not text.strip():
        try:
            p.rename(p.with_suffix(p.suffix + f".empty-{int(now_fn())}"))
        except Exception:
            pass
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            p.rename(p.with_suffix(p.suffix + f".corrupt-{int(now_fn())}"))
        except Exception:
            pass
        raise RuntimeError(
            f"corrupt JSON at {p} ({exc}); quarantined; restart watcher to start fresh"
        ) from exc


def atomic_write_json(path: str | Path, obj) -> None:
    """Write JSON via tmp + fsync + replace so interrupted writes keep old or new state."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def empty_queue_reset_allowed(
    allow_env: str,
    *,
    environ: Optional[dict[str, str]] = None,
) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(allow_env, "")).lower() in ("1", "true", "yes")


def state_counts(obj) -> tuple[int, int]:
    tasks = obj.get("tasks", []) if isinstance(obj, dict) else []
    total = len(tasks) if isinstance(tasks, list) else 0
    active = sum(
        1
        for task in tasks
        if isinstance(task, dict) and task.get("status") in ("queued", "launching", "running")
    )
    return total, active


def recent_queue_state_files(
    state_dir: str | Path,
    *,
    max_age_s: float,
    now_fn: Callable[[], float] = time.time,
) -> list[Path]:
    cutoff = now_fn() - max_age_s
    out: list[Path] = []
    for p in Path(state_dir).glob("queue.json.*"):
        if p.name.endswith(".tmp") or not p.is_file():
            continue
        try:
            if p.stat().st_size > 0 and p.stat().st_mtime >= cutoff:
                out.append(p)
        except Exception:
            continue
    return sorted(out, key=lambda x: x.stat().st_mtime, reverse=True)


def guard_missing_queue_state(
    *,
    queue_file: str | Path,
    state_dir: str | Path,
    allow_env: str,
    max_age_s: float,
    now_fn: Callable[[], float] = time.time,
) -> None:
    if empty_queue_reset_allowed(allow_env):
        return
    q = Path(queue_file)
    try:
        missing_or_empty = (not q.exists()) or q.stat().st_size == 0
    except Exception:
        missing_or_empty = False
    if not missing_or_empty:
        return
    recent = recent_queue_state_files(state_dir, max_age_s=max_age_s, now_fn=now_fn)
    if not recent:
        return
    names = ", ".join(p.name for p in recent[:3])
    raise RuntimeError(
        f"{q} is missing/empty while recent queue backups exist ({names}); "
        f"refusing to initialize an empty scheduler state. Restore a backup or set "
        f"{allow_env}=1 if this reset is intentional."
    )


def guard_empty_queue_save(
    new_state,
    *,
    queue_file: str | Path,
    allow_env: str,
    min_tasks: int,
    min_active: int,
) -> None:
    if empty_queue_reset_allowed(allow_env):
        return
    new_total, _new_active = state_counts(new_state)
    if new_total:
        return
    q = Path(queue_file)
    if not q.exists():
        return
    try:
        old_state = json.loads(q.read_text())
    except Exception:
        return
    old_total, old_active = state_counts(old_state)
    if old_total >= min_tasks or old_active >= min_active:
        raise RuntimeError(
            f"refusing to overwrite {q} with an empty task list "
            f"(previous total={old_total}, active={old_active}). Set "
            f"{allow_env}=1 if this reset is intentional."
        )


def load_scheduler_state(
    *,
    queue_file: str | Path,
    state_dir: str | Path,
    allow_env: str,
    missing_guard_max_age_s: float,
    default=None,
    canonicalize: Optional[Callable[[dict], object]] = None,
):
    guard_missing_queue_state(
        queue_file=queue_file,
        state_dir=state_dir,
        allow_env=allow_env,
        max_age_s=missing_guard_max_age_s,
    )
    state = load_json(queue_file, {"tasks": [], "next_id": 1} if default is None else default)
    if isinstance(state, dict) and canonicalize is not None:
        canonicalize(state)
    return state


def save_scheduler_state(
    state: dict,
    *,
    queue_file: str | Path,
    allow_env: str,
    empty_guard_min_tasks: int,
    empty_guard_min_active: int,
    canonicalize: Optional[Callable[[dict], object]] = None,
    compact: Optional[Callable[[dict], object]] = None,
) -> None:
    if canonicalize is not None:
        canonicalize(state)
    if compact is not None:
        compact(state)
    guard_empty_queue_save(
        state,
        queue_file=queue_file,
        allow_env=allow_env,
        min_tasks=empty_guard_min_tasks,
        min_active=empty_guard_min_active,
    )
    atomic_write_json(queue_file, state)
