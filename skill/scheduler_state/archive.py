"""Queue archive helpers for scheduleurm.

The scheduler keeps hot tasks in queue.json and appends older terminal records
to queue_archive.jsonl. These helpers are deliberately small and dependency
injected so CLI, TUI, and tests use one lookup path.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional


TERMINAL_ARCHIVE_STATUSES = frozenset({"done", "failed", "cancelled", "forgotten"})


def _has_pending_result_sync(task: dict) -> bool:
    """Keep completed remote artifacts hot until their pull is committed."""
    if task.get("status") != "done" or task.get("result_synced_at"):
        return False
    return bool(task.get("result_dir") or task.get("result_dirs"))


def archive_terminal_tasks(
    state: dict,
    *,
    archive_file: Path,
    age_days: float,
    max_hot_terminal: Optional[int] = None,
    extra_archive_predicate: Optional[Callable[[dict], bool]] = None,
    now_fn: Callable[[], float] = time.time,
) -> int:
    """Move terminal tasks out of hot state and append them to JSONL.

    A task is archived when it is older than ``age_days`` or when
    ``extra_archive_predicate`` selects it. When ``max_hot_terminal`` is set,
    only the newest N remaining terminal records are kept in hot state; older
    excess records are archived too. Caller owns any scheduler state lock and
    persists ``state`` after this returns. If writing the archive fails, the
    state is left untouched.
    """
    cutoff = now_fn() - age_days * 86400
    tasks = list(state.get("tasks", []))
    archive_task_obj_ids = set()

    def _terminal_ts(task: dict) -> float:
        try:
            return float(task.get("finished_at") or task.get("cancelled_at") or task.get("submitted_at") or 0)
        except Exception:
            return 0.0

    for task in tasks:
        if (
            task.get("status") in TERMINAL_ARCHIVE_STATUSES
            and not _has_pending_result_sync(task)
        ):
            ts = _terminal_ts(task)
            if ts and ts < cutoff:
                archive_task_obj_ids.add(id(task))
                continue
            if extra_archive_predicate is not None:
                try:
                    if extra_archive_predicate(task):
                        archive_task_obj_ids.add(id(task))
                except Exception:
                    pass

    if max_hot_terminal is not None:
        try:
            keep_limit = max(0, int(max_hot_terminal))
        except Exception:
            keep_limit = 0
        terminal_remaining = [
            task for task in tasks
            if task.get("status") in TERMINAL_ARCHIVE_STATUSES
            and id(task) not in archive_task_obj_ids
            and not _has_pending_result_sync(task)
        ]
        excess = len(terminal_remaining) - keep_limit
        if excess > 0:
            terminal_remaining.sort(
                key=lambda task: (_terminal_ts(task), str(task.get("id") or ""))
            )
            for task in terminal_remaining[:excess]:
                archive_task_obj_ids.add(id(task))

    keep, archived = [], []
    for task in tasks:
        if id(task) in archive_task_obj_ids:
            archived.append(task)
            continue
        keep.append(task)
    if not archived:
        return 0
    try:
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_file, "a") as f:
            for task in archived:
                f.write(json.dumps(task, default=str) + "\n")
        state["tasks"] = keep
    except Exception:
        return 0
    return len(archived)


def compact_age_days_from_args(args, *, default_age_days: float) -> float:
    if getattr(args, "all_terminal", False):
        return 0.0
    if getattr(args, "age_hours", None) is not None:
        return max(0.0, float(args.age_hours)) / 24.0
    if getattr(args, "age_days", None) is not None:
        return max(0.0, float(args.age_days))
    return max(0.0, float(default_age_days))


def terminal_archive_stats(
    state: dict,
    age_days: float,
    *,
    max_hot_terminal: Optional[int] = None,
    extra_archive_predicate: Optional[Callable[[dict], bool]] = None,
    now_fn: Callable[[], float] = time.time,
) -> dict:
    cutoff = now_fn() - max(0.0, float(age_days)) * 86400.0
    by_status: dict[str, int] = {}
    count = 0
    age_count = 0
    extra_count = 0
    excess_count = 0
    approx_bytes = 0
    terminal: list[tuple[float, dict]] = []
    pending_result_sync_count = 0
    for task in state.get("tasks", []):
        if task.get("status") not in TERMINAL_ARCHIVE_STATUSES:
            continue
        try:
            ts = float(task.get("finished_at") or task.get("cancelled_at") or task.get("submitted_at") or 0)
        except Exception:
            ts = 0.0
        terminal.append((ts, task))
        if _has_pending_result_sync(task):
            pending_result_sync_count += 1

    archive_ids = set()
    for ts, task in terminal:
        if _has_pending_result_sync(task):
            continue
        if ts and ts < cutoff:
            archive_ids.add(id(task))
            age_count += 1
            continue
        if extra_archive_predicate is not None:
            try:
                if extra_archive_predicate(task):
                    archive_ids.add(id(task))
                    extra_count += 1
            except Exception:
                pass

    if max_hot_terminal is not None:
        try:
            keep_limit = max(0, int(max_hot_terminal))
        except Exception:
            keep_limit = 0
        remaining = [
            (ts, task)
            for ts, task in terminal
            if id(task) not in archive_ids and not _has_pending_result_sync(task)
        ]
        excess = len(remaining) - keep_limit
        if excess > 0:
            for _ts, task in sorted(remaining, key=lambda item: (item[0], str(item[1].get("id") or "")))[:excess]:
                archive_ids.add(id(task))
                excess_count += 1

    for _ts, task in terminal:
        if id(task) not in archive_ids:
            continue
        count += 1
        status = task.get("status") or "?"
        by_status[status] = by_status.get(status, 0) + 1
        try:
            approx_bytes += len(json.dumps(task, ensure_ascii=False).encode("utf-8"))
        except Exception:
            pass

    return {
        "count": count,
        "age_count": age_count,
        "extra_count": extra_count,
        "excess_count": excess_count,
        "terminal_total": len(terminal),
        "pending_result_sync_count": pending_result_sync_count,
        "max_hot_terminal": max_hot_terminal,
        "approx_bytes": approx_bytes,
        "by_status": by_status,
    }


def load_archive_tasks(archive_file: Path, *, limit: Optional[int] = None) -> list[dict]:
    """Load archived task records from append-only JSONL.

    Bad lines are skipped so one torn archival write does not break diagnostics.
    A positive ``limit`` returns the most recent records.
    """
    if not archive_file.exists():
        return []
    tasks: list[dict] = []
    try:
        with open(archive_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if isinstance(record, dict):
                    tasks.append(record)
    except Exception:
        pass
    if limit and len(tasks) > limit:
        return tasks[-limit:]
    return tasks


def find_task_record(
    task_id: str,
    state: dict,
    *,
    include_archive: bool = True,
    archive_loader: Optional[Callable[[], Iterable[dict]]] = None,
) -> tuple[Optional[dict], str]:
    """Find a task in hot state first, then archive.

    If multiple archived records share an id, the newest appended record wins,
    matching the legacy JSONL scan behavior.
    """
    for task in state.get("tasks", []):
        if task.get("id") == task_id:
            return task, "queue"
    if include_archive and archive_loader is not None:
        found = None
        for task in archive_loader():
            if task.get("id") == task_id:
                found = task
        if found is not None:
            return found, "archive"
    return None, ""
