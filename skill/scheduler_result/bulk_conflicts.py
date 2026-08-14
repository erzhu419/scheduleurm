from __future__ import annotations

from typing import Any, Callable


def bulk_submit_result_conflict_message(
    state: dict,
    specs: list[dict],
    *,
    allow_shared_result_dir: bool,
    active_or_unsynced_result_status: Callable[[dict], bool],
    conflict_path_key: Callable[[Any], str],
    same_declared_path: Callable[[Any, Any], bool],
) -> str:
    if allow_shared_result_dir:
        return ""
    seen_new: dict[str, str] = {}
    existing = [
        (task.get("id"), task.get("local_result_dir") or task.get("result_dir"))
        for task in state.get("tasks", [])
        if active_or_unsynced_result_status(task)
    ]
    for spec in specs:
        dst = spec.get("local_result_dir") or spec.get("result_dir")
        key = conflict_path_key(dst)
        if not key:
            continue
        for existing_id, existing_dst in existing:
            if same_declared_path(existing_dst, dst):
                return (
                    "REFUSED: CPU batch result destination already in use by "
                    f"unfinished/unsynced task {existing_id}: {dst}"
                )
        previous_sig = seen_new.get(key)
        if previous_sig:
            return (
                "REFUSED: CPU batch shards share a local result destination "
                f"({dst}); first signature={previous_sig!r}, duplicate={spec.get('signature')!r}. "
                "Use a per-shard template or pass --allow-shared-result-dir."
            )
        seen_new[key] = str(spec.get("signature") or "")
    return ""
