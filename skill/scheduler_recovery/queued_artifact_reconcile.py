from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class QueuedArtifactReconcileDeps:
    backend: Any
    task_pids: Callable[[dict], list]
    set_current_usage: Callable[[dict, int, int, float], Any]
    release_task_claims_and_intents: Callable[[dict], Any]
    diagnose_terminal: Callable[[dict], dict]
    requeue_after_crash: Callable[[dict, dict], Optional[str]]
    now: Callable[[], float]


def queued_launch_artifacts(task: dict, *, deps: QueuedArtifactReconcileDeps) -> list[str]:
    artifacts = []
    if deps.task_pids(task):
        artifacts.append("remote_pids")
    if task.get("process_group"):
        artifacts.append("process_group")
    if task.get("started_at") and not task.get("finished_at"):
        artifacts.append("started_at")
    if task.get("log_path") and task.get("started_at"):
        artifacts.append("log_path")
    if task.get("slurm_job_id"):
        artifacts.append("slurm_job_id")
    if task.get("local_ssh_pid"):
        artifacts.append("local_ssh_pid")
    return artifacts


def reconcile_queued_launch_artifacts_before_dispatch(
    task: dict,
    state: dict,
    *,
    deps: QueuedArtifactReconcileDeps,
) -> Optional[dict]:
    """Prevent same-id relaunch when a queued record still has old launch state."""
    if task.get("status") != "queued":
        return None
    artifacts = queued_launch_artifacts(task, deps=deps)
    if not artifacts:
        return None

    node = task.get("node") or task.get("last_node")
    if node and not task.get("node"):
        task["node"] = node
    if node:
        task["last_node"] = node
    if task.get("gpu_idx") is not None:
        task["last_gpu_idx"] = task.get("gpu_idx")

    if node and (deps.task_pids(task) or task.get("slurm_job_id")):
        probe_task = dict(task)
        probe_task["status"] = "running"
        probe_task["node"] = node
        try:
            result = deps.backend.batch_probe({"tasks": [probe_task]}).get(task.get("id"))
        except Exception:
            result = {"state": "unknown"}
        if not result or result.get("state") == "unknown":
            reason = (
                "queued task still has launch artifacts "
                f"({','.join(artifacts)}) but liveness probe is unknown; "
                "not relaunching same task id"
            )
            task["last_block_reason"] = reason
            return {"type": "blocked", "task_id": task.get("id"), "task": task, "reason": reason}
        if result.get("state") == "alive":
            task["status"] = "running"
            task["node"] = node
            task["alive_pids"] = result.get("alive_pids") or []
            deps.set_current_usage(
                task,
                result.get("vram_mb", 0),
                result.get("ram_mb", 0),
                result.get("pcpu", 0.0),
            )
            if result.get("vram_mb", 0) > 0:
                task["peak_vram_mb"] = max(task.get("peak_vram_mb", 0), result.get("vram_mb", 0))
            if result.get("ram_mb", 0) > 0:
                task["peak_ram_mb"] = max(task.get("peak_ram_mb", 0), result.get("ram_mb", 0))
            task["last_block_reason"] = (
                "recovered queued task with live launch artifacts; adopted as running "
                "to avoid same-id relaunch"
            )
            return {
                "type": "queued_artifact_adopted",
                "task_id": task.get("id"),
                "task": task,
                "reason": task["last_block_reason"],
            }

    if not node:
        reason = (
            "queued task still has launch artifacts "
            f"({','.join(artifacts)}) but no node/last_node to probe; "
            "not relaunching same task id"
        )
        task["last_block_reason"] = reason
        return {"type": "blocked", "task_id": task.get("id"), "task": task, "reason": reason}

    task["node"] = node
    task["finished_at"] = task.get("finished_at") or deps.now()
    task["started_at"] = task.get("started_at") or task["finished_at"]
    task["remote_pids"] = []
    task["alive_pids"] = []
    deps.set_current_usage(task, 0, 0, 0.0)
    try:
        deps.release_task_claims_and_intents(task)
    except Exception:
        pass
    diagnosis = deps.diagnose_terminal(task)
    task["_diagnosis"] = diagnosis
    if diagnosis.get("is_crash") and not task.get("auto_adopted"):
        task["status"] = "failed"
        task["last_block_reason"] = (
            "queued task carried dead launch artifacts; finalized as failed "
            f"instead of relaunching same id: {diagnosis.get('reason', '')}"
        )
        new_id = deps.requeue_after_crash(task, state)
        if new_id:
            task["requeued_as"] = new_id
        return {
            "type": "queued_artifact_finalized",
            "task_id": task.get("id"),
            "task": task,
            "requeued_as": task.get("requeued_as"),
            "reason": task["last_block_reason"],
        }
    task["status"] = "done"
    task["last_block_reason"] = (
        "queued task carried dead launch artifacts; finalized as done "
        "instead of relaunching same id"
    )
    return {
        "type": "queued_artifact_finalized",
        "task_id": task.get("id"),
        "task": task,
        "reason": task["last_block_reason"],
    }
