"""Watch-loop task transition helpers."""

from __future__ import annotations

from typing import Mapping, Sequence


def classify_running_terminal_transitions(
    tasks: Sequence[dict],
    previous_status_by_id: Mapping[str, str],
) -> tuple[list[dict], list[dict]]:
    """Return newly done/crashed tasks and mark them as notified.

    Only tasks that were ``running`` in the pre-update snapshot are classified.
    This preserves watcher semantics: already-terminal tasks and queued tasks
    that become terminal through repair logic do not emit duplicate completion
    notifications.
    """
    newly_done: list[dict] = []
    newly_crashed: list[dict] = []
    for task in tasks:
        tid = str(task.get("id") or "")
        if previous_status_by_id.get(tid) != "running":
            continue
        if task.get("notified_done"):
            continue
        status = task.get("status")
        if status == "failed":
            newly_crashed.append(task)
            task["notified_done"] = True
        elif status == "done":
            newly_done.append(task)
            task["notified_done"] = True
    return newly_done, newly_crashed
