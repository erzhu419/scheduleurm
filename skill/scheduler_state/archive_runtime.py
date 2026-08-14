"""Runtime-bound archive wrappers for scheduler.py."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_archive_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def archive_terminal_tasks(
        state,
        age_days=None,
        max_hot_terminal=None,
    ):
        if age_days is None:
            age_days = _ns(namespace, "ARCHIVE_AGE_DAYS")
        if max_hot_terminal is None:
            max_hot_terminal = _ns(namespace, "ARCHIVE_MAX_HOT_TERMINAL")
        return _ns(namespace, "_archive_terminal_tasks_impl")(
            state,
            archive_file=_ns(namespace, "ARCHIVE_FILE"),
            age_days=age_days,
            max_hot_terminal=max_hot_terminal,
            extra_archive_predicate=_ns(namespace, "_archive_control_plane_autoadopted_terminal"),
            now_fn=_ns(namespace, "time").time,
        )

    def load_archive_tasks(limit: Optional[int] = None):
        return _ns(namespace, "_load_archive_tasks_impl")(
            _ns(namespace, "ARCHIVE_FILE"),
            limit=limit,
        )

    def _archive_control_plane_autoadopted_terminal(task: dict) -> bool:
        if task.get("status") not in ("done", "failed", "cancelled", "forgotten"):
            return False
        if not task.get("auto_adopted"):
            return False
        if not str(task.get("signature") or "").startswith("scheduleurm/auto-adopted/"):
            return False
        return _ns(namespace, "_is_scheduler_control_process")(
            task.get("cwd") or "",
            task.get("cmd") or "",
        )

    def _find_task_record(task_id: str, include_archive: bool = True):
        return _ns(namespace, "_find_task_record_impl")(
            task_id,
            _ns(namespace, "load_state")(),
            include_archive=include_archive,
            archive_loader=_ns(namespace, "load_archive_tasks"),
        )

    return {
        "archive_terminal_tasks": archive_terminal_tasks,
        "load_archive_tasks": load_archive_tasks,
        "_archive_control_plane_autoadopted_terminal": _archive_control_plane_autoadopted_terminal,
        "_find_task_record": _find_task_record,
    }
