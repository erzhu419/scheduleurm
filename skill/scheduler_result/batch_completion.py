from __future__ import annotations

from collections import Counter
from typing import Any


def signature_batch_key(signature: str | None) -> str:
    """Group signatures by their top two path components."""
    if not signature:
        return ""
    parts = signature.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else signature


def detect_batch_completions(state: dict[str, Any], transitioned_ids: list[str]) -> list[dict[str, Any]]:
    """Return batch completion summaries for prefixes newly made terminal.

    A batch fires only when one of this iteration's terminal transitions belongs
    to a prefix whose non-adopted siblings are all terminal. The per-prefix
    timestamp lets a later fresh batch with the same signature prefix notify
    again after newer tasks finish.
    """
    if not transitioned_ids:
        return []
    transitioned_set = set(transitioned_ids)

    candidate_prefixes = {
        signature_batch_key(task.get("signature") or "")
        for task in state.get("tasks", [])
        if task.get("id") in transitioned_set
    }
    completed: list[dict[str, Any]] = []
    notify_ts = state.setdefault("_batch_notify_ts", {})

    for prefix in candidate_prefixes:
        if not prefix:
            continue
        siblings = [
            task
            for task in state.get("tasks", [])
            if signature_batch_key(task.get("signature") or "") == prefix
            and not task.get("auto_adopted")
        ]
        if not siblings:
            continue
        if any(task.get("status") in ("queued", "running", "launching") for task in siblings):
            continue

        latest_finish = max((task.get("finished_at") or 0) for task in siblings)
        if latest_finish <= notify_ts.get(prefix, 0):
            continue

        counts = Counter(task.get("status") for task in siblings)
        completed.append(
            {
                "prefix": prefix,
                "total": len(siblings),
                "done": counts.get("done", 0),
                "failed": counts.get("failed", 0),
                "cancelled": counts.get("cancelled", 0),
                "task_ids": [task["id"] for task in siblings],
                "earliest_submit": min((task.get("submitted_at") or 0) for task in siblings),
                "latest_finish": latest_finish,
            }
        )
        notify_ts[prefix] = latest_finish
    return completed
