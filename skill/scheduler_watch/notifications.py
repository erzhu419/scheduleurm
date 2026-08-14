"""Watch-loop notification dispatch helpers."""

from __future__ import annotations

from typing import Callable, Iterable


def crashed_task_payload(task: dict) -> dict:
    diag = task.get("_diagnosis") or {}
    return {
        "id": task["id"],
        "project": task.get("project"),
        "description": task.get("description", ""),
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "lifetime_s": diag.get("lifetime_s", 0),
        "log_size": diag.get("log_size", 0),
        "log_path": diag.get("log_path"),
        "reason": diag.get("reason", "?"),
        "tail": diag.get("tail", ""),
    }


def notify_terminal_transitions(
    newly_done: Iterable[dict],
    newly_crashed: Iterable[dict],
    *,
    notify_fn: Callable,
    compact_task_payload_fn: Callable[[dict], dict],
) -> None:
    for task in newly_done:
        notify_fn("task_done", compact_task_payload_fn(task))
    for task in newly_crashed:
        notify_fn("task_crashed", crashed_task_payload(task))


def notify_dispatch_events(
    events: Iterable[dict],
    *,
    notify_fn: Callable,
    compact_task_payload_fn: Callable[[dict], dict],
) -> None:
    """Emit watcher notifications for dispatch events.

    No-fit and resume-found events are intentionally ignored here; they are
    already present in watcher.log and are too noisy for Feishu.
    """
    for ev in events:
        event_type = ev["type"]
        if event_type == "launched":
            notify_fn("task_launched", compact_task_payload_fn(ev["task"]))
        elif event_type in ("pre_launch_snapshot", "post_launch_snapshot"):
            notify_fn(event_type, ev.get("snapshot") or {}, feishu_enabled=False)
        elif event_type == "blocked":
            notify_fn("task_blocked", {"task_id": ev["task_id"], "reason": ev["reason"]})
        elif event_type == "launch_failed_retry":
            notify_fn("task_launch_retry", {
                "task_id": ev["task_id"],
                "error": ev["error"],
                "attempt": ev["task"].get("launch_fail_count"),
            })
        elif event_type == "launch_failed_terminal":
            notify_fn("task_failed", {"task_id": ev["task_id"], "error": ev["error"]})
        elif event_type == "migrated":
            notify_fn("task_migrated", {
                "task_id": ev["task_id"],
                "from_node": ev.get("from_node"),
                "to_node": ev.get("to_node"),
                "eta_seconds": ev.get("eta_seconds", 0),
                "reason": ev.get("reason", ""),
            })
        elif event_type == "preempted":
            notify_fn("task_preempted", {
                "task_id": ev["task_id"],
                "freed_node": ev.get("freed_node"),
                "cpu_freed": ev.get("cpu_freed", 0),
                "ram_freed": ev.get("ram_freed", 0),
            })
        elif event_type == "claim_race":
            notify_fn(
                "task_claim_race",
                {"task_id": ev["task_id"], "reason": ev["reason"][:300]},
                feishu_enabled=False,
            )
        elif event_type == "launch_commit_skipped":
            notify_fn(
                "task_launch_commit_skipped",
                {"task_id": ev.get("task_id"), "reason": ev.get("reason", "")[:300]},
                feishu_enabled=False,
            )
