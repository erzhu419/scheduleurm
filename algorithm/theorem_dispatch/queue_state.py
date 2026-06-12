"""Queue-vector extraction for theorem-facing MaxWeight dispatch."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from .service_registry import infer_workload_key


ACTIVE_STATUSES = {"queued", "launching", "running"}
DEFAULT_QUEUE_PATH = Path.home() / ".claude" / "scheduler" / "queue.json"

_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": None,
    "loaded_at": 0.0,
    "ttl_s": 1.0,
    "vector": {},
    "active_task_ids": set(),
    "error": "",
}


def load_queue_vector(
    path: str | Path | None = None,
    *,
    ttl_s: float = 1.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return active-task counts keyed by measured workload class."""

    p = Path(path or DEFAULT_QUEUE_PATH).expanduser()
    now = time.time()
    try:
        st = p.stat()
        mtime_ns = st.st_mtime_ns
    except OSError as exc:
        return {}, {
            "queue_path": str(p),
            "loaded": False,
            "error": f"stat_failed:{exc}",
            "active_task_count": 0,
            "unclassified_active_count": 0,
        }

    if (
        _CACHE.get("path") == str(p)
        and _CACHE.get("mtime_ns") == mtime_ns
        and now - float(_CACHE.get("loaded_at") or 0.0) <= max(0.0, float(ttl_s))
    ):
        return dict(_CACHE.get("vector") or {}), {
            "queue_path": str(p),
            "loaded": True,
            "cached": True,
            "active_task_count": int(_CACHE.get("active_task_count") or 0),
            "unclassified_active_count": int(_CACHE.get("unclassified_active_count") or 0),
            "error": str(_CACHE.get("error") or ""),
        }

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        tasks = payload.get("tasks") if isinstance(payload, Mapping) else payload
        if not isinstance(tasks, list):
            tasks = []
    except Exception as exc:
        return {}, {
            "queue_path": str(p),
            "loaded": False,
            "error": f"read_failed:{exc}",
            "active_task_count": 0,
            "unclassified_active_count": 0,
        }

    vector: dict[str, float] = {}
    active_ids: set[str] = set()
    active_count = 0
    unclassified = 0
    for row in tasks:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status") or "").strip().lower() not in ACTIVE_STATUSES:
            continue
        active_count += 1
        row_id = str(row.get("id") or row.get("task_id") or "").strip()
        if row_id:
            active_ids.add(row_id)
        key = infer_workload_key(row)
        if not key:
            unclassified += 1
            continue
        vector[key] = vector.get(key, 0.0) + 1.0

    _CACHE.update({
        "path": str(p),
        "mtime_ns": mtime_ns,
        "loaded_at": now,
        "ttl_s": float(ttl_s),
        "vector": dict(vector),
        "active_task_ids": set(active_ids),
        "active_task_count": active_count,
        "unclassified_active_count": unclassified,
        "error": "",
    })
    return vector, {
        "queue_path": str(p),
        "loaded": True,
        "cached": False,
        "active_task_count": active_count,
        "unclassified_active_count": unclassified,
        "error": "",
    }


def queue_vector_for_task(
    task: Mapping[str, Any],
    workload_key: str,
    *,
    queue_path: str | Path | None = None,
    ttl_s: float = 1.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    vector, meta = load_queue_vector(queue_path, ttl_s=ttl_s)
    if workload_key:
        task_id = str(task.get("id") or task.get("task_id") or "")
        if task_id and task_id not in set(_CACHE.get("active_task_ids") or set()):
            vector[workload_key] = vector.get(workload_key, 0.0) + 1.0
            meta = dict(meta)
            meta["included_current_task"] = True
        elif not task_id:
            vector[workload_key] = vector.get(workload_key, 0.0) + 1.0
            meta = dict(meta)
            meta["included_current_task"] = True
    return vector, meta
