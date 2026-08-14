"""Compact scheduler state before persisting queue.json."""

from __future__ import annotations


PERSISTED_SNAPSHOT_KEYS = (
    "last_launch_pre_snapshot",
    "last_launch_post_snapshot",
    "last_resource_snapshot",
)

RESOURCE_SNAPSHOT_TASK_FIELDS = (
    "id",
    "status",
    "project",
    "node",
    "gpu_idx",
    "started_at",
    "elapsed_s",
    "eta_seconds",
    "eta_source",
    "progress_ratio",
    "current_vram_mb",
    "peak_vram_mb",
    "current_ram_mb",
    "peak_ram_mb",
    "cpu_cores",
    "ram_pressure_mb",
    "evict_loss_protected",
)


def compact_resource_task_summary(summary: dict) -> dict:
    """Keep only fields needed for placement/forensics in persisted snapshots."""
    if not isinstance(summary, dict):
        return {}
    out = {}
    for key in RESOURCE_SNAPSHOT_TASK_FIELDS:
        if key not in summary:
            continue
        value = summary.get(key)
        if value is None or value == "":
            continue
        out[key] = value
    return out


def compact_resource_snapshot(
    snapshot: dict,
    *,
    max_same_gpu: int = 12,
    max_same_node: int = 0,
) -> dict:
    if not isinstance(snapshot, dict):
        return snapshot
    out = {}
    for key in (
        "stage",
        "ts",
        "target_node",
        "target_gpu_idx",
        "gpu",
        "node_ram",
        "crossed_one_third_from_pre",
        "crossed_ram_headroom_from_pre",
    ):
        if key in snapshot:
            out[key] = snapshot.get(key)
    if isinstance(snapshot.get("task"), dict):
        out["task"] = compact_resource_task_summary(snapshot.get("task") or {})
    same_gpu = snapshot.get("same_gpu_tasks")
    try:
        prior_same_gpu_count = max(0, int(snapshot.get("same_gpu_task_count") or 0))
    except Exception:
        prior_same_gpu_count = 0
    if isinstance(same_gpu, list):
        out["same_gpu_task_count"] = max(prior_same_gpu_count, len(same_gpu))
        out["same_gpu_tasks"] = [
            compact_resource_task_summary(x) for x in same_gpu[-max_same_gpu:]
            if isinstance(x, dict)
        ]
    elif prior_same_gpu_count:
        out["same_gpu_task_count"] = prior_same_gpu_count
    same_node = snapshot.get("same_node_running_tasks")
    try:
        prior_same_node_count = max(
            0, int(snapshot.get("same_node_running_task_count") or 0)
        )
    except Exception:
        prior_same_node_count = 0
    if isinstance(same_node, list):
        out["same_node_running_task_count"] = max(
            prior_same_node_count, len(same_node)
        )
        if max_same_node > 0:
            out["same_node_running_tasks"] = [
                compact_resource_task_summary(x) for x in same_node[-max_same_node:]
                if isinstance(x, dict)
            ]
    elif prior_same_node_count:
        out["same_node_running_task_count"] = prior_same_node_count
    return out


def compact_state_for_persistence(state: dict) -> dict:
    """Bound queue.json growth before writing.

    Resource snapshots are diagnostic breadcrumbs. Persisting full colocated
    task summaries for every running/done task can make queue.json tens of MB,
    so every watcher cycle holds the state lock while rewriting a huge file.
    Keep the fields that later crash/OOM/eviction logic reads, and drop verbose
    per-neighbor text such as signatures and progress lines.
    """
    if not isinstance(state, dict):
        return state
    for task in state.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        for key in PERSISTED_SNAPSHOT_KEYS:
            if key in task:
                task[key] = compact_resource_snapshot(task.get(key))
        crash = task.get("last_crash_forensics")
        if isinstance(crash, dict):
            for key in ("last_resource_snapshot", "crash_probe_snapshot"):
                if key in crash:
                    crash[key] = compact_resource_snapshot(crash.get(key))
            if isinstance(crash.get("same_gpu_tasks"), list):
                crash["same_gpu_tasks"] = [
                    compact_resource_task_summary(x)
                    for x in (crash.get("same_gpu_tasks") or [])[-12:]
                    if isinstance(x, dict)
                ]
        oom = task.get("last_oom_forensics")
        if isinstance(oom, dict) and isinstance(oom.get("suspected_trigger_task"), dict):
            oom["suspected_trigger_task"] = compact_resource_task_summary(
                oom.get("suspected_trigger_task") or {}
            )
    return state
