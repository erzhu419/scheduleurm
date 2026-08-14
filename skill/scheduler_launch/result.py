"""Apply launch backend results to scheduler task state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class LaunchResultDeps:
    max_launch_retry: int
    remember_claim_intent: Callable[[dict, str | None], Any]
    release_task_claims_and_intents: Callable[..., Any]
    write_escalation: Callable[[dict, str, dict], Any]
    clear_claim_intent_markers: Callable[[dict], Any]
    now: Callable[[], float] = time.time


def apply_launch_result_to_task(
    task: dict,
    ok: bool,
    msg: str,
    events: list,
    *,
    deps: LaunchResultDeps,
) -> bool:
    """Apply launch success/failure to a state task. Returns True on success."""
    if not ok:
        if isinstance(msg, str) and msg.startswith("CLAIM_RACE:"):
            attempted_node = task.get("node")
            deps.remember_claim_intent(task, attempted_node)
            task["status"] = "queued"
            task["last_block_reason"] = msg
            task["node"] = None
            task["gpu_idx"] = None
            task.pop("launching_started_at", None)
            task.pop("launch_token", None)
            events.append({
                "type": "claim_race",
                "task_id": task["id"],
                "task": task,
                "reason": msg,
            })
            return False

        attempted_node = task.get("node")
        if attempted_node:
            deps.release_task_claims_and_intents(
                task,
                extra_nodes=[attempted_node],
                clear_markers=False,
            )
        task["launch_fail_count"] = (task.get("launch_fail_count") or 0) + 1
        if attempted_node:
            failed_nodes = task.setdefault("launch_failed_nodes", {})
            if not isinstance(failed_nodes, dict):
                failed_nodes = {}
                task["launch_failed_nodes"] = failed_nodes
            failed_nodes[attempted_node] = {
                "ts": deps.now(),
                "attempt": task["launch_fail_count"],
                "error": str(msg)[:300],
            }
        task["last_block_reason"] = (
            f"launch attempt {task['launch_fail_count']}/{deps.max_launch_retry}: {msg}"
        )
        task["node"] = None
        task["gpu_idx"] = None
        task.pop("launching_started_at", None)
        task.pop("launch_token", None)
        if task["launch_fail_count"] >= deps.max_launch_retry:
            task["status"] = "failed"
            try:
                deps.write_escalation(task, "LAUNCH_FAIL_CAP", {"reason": msg, "tail": msg})
            except Exception:
                pass
            events.append({
                "type": "launch_failed_terminal",
                "task_id": task["id"],
                "task": task,
                "error": str(msg),
            })
        else:
            task["status"] = "queued"
            events.append({
                "type": "launch_failed_retry",
                "task_id": task["id"],
                "task": task,
                "error": str(msg),
            })
        return False

    task.pop("launch_fail_count", None)
    task.pop("launch_failed_nodes", None)
    task.pop("evict_cooldown_until", None)
    task.pop("evict_node_cooldowns", None)
    task.pop("launch_token", None)
    if not str(task.get("last_block_reason") or "").startswith("warn:"):
        task.pop("last_block_reason", None)
    deps.clear_claim_intent_markers(task)
    events.append({"type": "launched", "task_id": task["id"], "task": task, "msg": msg})
    return True
