"""Pure helpers for scheduleurm task id parsing and allocation."""

from __future__ import annotations

import re
from typing import Optional


_TASK_ID_RE = re.compile(r"^t(\d+)$")


def task_id_number(task_id: str) -> Optional[int]:
    """Return the numeric suffix for scheduler task ids like t0001."""
    match = _TASK_ID_RE.match(str(task_id or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def format_task_id(n: int) -> str:
    """Format task ids with four digits as a minimum width, not a maximum."""
    try:
        num = int(n)
    except Exception:
        num = 1
    return f"t{max(1, num):04d}"


def allocate_task_id(state: dict) -> str:
    """Allocate one monotonically increasing task id without wrapping at t9999."""
    return allocate_task_ids(state, 1)[0]


def allocate_task_ids(state: dict, count: int) -> list[str]:
    """Allocate `count` monotonically increasing task ids in one queue scan."""
    count = max(0, int(count or 0))
    if count <= 0:
        return []
    tasks = state.get("tasks") or []
    used = {str(t.get("id")) for t in tasks if isinstance(t, dict) and t.get("id")}
    max_seen = 0
    for tid in used:
        num = task_id_number(tid)
        if num is not None:
            max_seen = max(max_seen, num)
    try:
        next_num = int(state.get("next_id") or 1)
    except Exception:
        next_num = 1
    next_num = max(1, next_num, max_seen + 1)
    allocated = []
    while len(allocated) < count:
        tid = format_task_id(next_num)
        next_num += 1
        if tid not in used:
            used.add(tid)
            allocated.append(tid)
    state["next_id"] = next_num
    return allocated
