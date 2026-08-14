"""Run-identity and launch-artifact safety rules for scheduler tasks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class TaskIdentityDeps:
    task_pids: Callable[[dict], list]
    local_launch_transport_alive: Callable[[dict], bool]
    backend: Any


def conflict_path_key(path) -> str:
    """Normalize declared local/remote paths for equality checks without probing FS."""
    if not path:
        return ""
    return os.path.normpath(os.path.expanduser(str(path).rstrip("/")))


def same_declared_path(a, b) -> bool:
    ka = conflict_path_key(a)
    kb = conflict_path_key(b)
    return bool(ka and kb and ka == kb)


def active_or_unsynced_result_status(task: dict) -> bool:
    """True while a task can still write/sync into declared result directories."""
    st = task.get("status")
    if st in ("queued", "launching", "running"):
        return True
    if st == "done" and task.get("result_dir") and not task.get("result_synced_at"):
        return True
    return False


def task_run_identity(task: dict):
    """Identity used only for duplicate-run suppression.

    Broad signatures are allowed: many independent jobs may share one
    family-level signature. The key therefore includes fields that make a
    launch materially different, while intentionally excluding scheduling
    knobs so resubmitting the same run with a different estimate does not
    double-launch it.
    """
    sig = task.get("signature") or ""
    if not sig:
        return None
    extra_env = task.get("extra_env") or {}
    if not isinstance(extra_env, dict):
        extra_env = {}
    env_key = json.dumps(extra_env, sort_keys=True, separators=(",", ":"))
    return (
        sig,
        task.get("cmd") or "",
        conflict_path_key(task.get("cwd")),
        task.get("env_spec") or "none",
        task.get("image") or "",
        env_key,
        conflict_path_key(task.get("ckpt_dir")),
        task.get("ckpt_glob") or "*",
        task.get("resume_flag") or "",
        conflict_path_key(task.get("result_dir")),
        conflict_path_key(task.get("local_result_dir")),
        int(task.get("cpu_parallel_total_items") or task.get("cpu_parallel_items") or 0),
        int(task.get("cpu_parallel_start") or 0),
        int(task.get("cpu_parallel_end") or 0),
        int(task.get("cpu_auto_workers") or 0),
    )


def local_launch_transport_alive(
    task: dict,
    *,
    alive: Callable[[int], bool],
) -> bool:
    """True when the local SSH transport that owns a Windows foreground launch still exists."""
    pid = task.get("local_ssh_pid")
    if not pid:
        return False
    try:
        return alive(int(pid))
    except Exception:
        return False


LEGACY_EXTERNAL_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}


def legacy_external_state_is_terminal(state: object) -> bool:
    text = str(state or "").strip().upper()
    return bool(text in LEGACY_EXTERNAL_TERMINAL_STATES or text.startswith("CANCELLED"))


def slurm_state_is_terminal(state: object) -> bool:
    return legacy_external_state_is_terminal(state)


def task_has_recorded_launch_artifacts(task: dict, *, deps: TaskIdentityDeps) -> bool:
    return bool(
        deps.task_pids(task)
        or task.get("slurm_job_id")
        or task.get("local_ssh_pid")
        or (task.get("log_path") and task.get("started_at"))
    )


def recorded_launch_safety_state(task: dict, *, deps: TaskIdentityDeps) -> tuple[str, str]:
    """Fail-closed liveness check used before requeueing or duplicate dispatch."""
    if not task_has_recorded_launch_artifacts(task, deps=deps):
        return "dead", "no recorded launch artifacts"
    if deps.local_launch_transport_alive(task):
        return "alive", f"local launch transport pid {task.get('local_ssh_pid')} is alive"
    terminal = task.get("status") in ("done", "failed", "cancelled", "forgotten")
    if terminal and task.get("finished_at") and not (deps.task_pids(task) or task.get("slurm_job_id")):
        return "dead", "terminal task has no backend pid/job; log artifact alone is not live"
    if terminal and task.get("slurm_job_id") and legacy_external_state_is_terminal(task.get("slurm_state")):
        return "dead", f"terminal task has slurm_state={task.get('slurm_state')} recorded"
    has_backend_artifact = bool(
        deps.task_pids(task)
        or task.get("slurm_job_id")
        or (task.get("log_path") and task.get("started_at"))
    )
    if not has_backend_artifact:
        return "dead", "only recorded local launch transport is gone"
    node = task.get("node") or task.get("last_node")
    if not node:
        return "unknown", "recorded launch artifacts exist but node/last_node is missing"
    if not (deps.task_pids(task) or task.get("slurm_job_id")):
        return "unknown", "recorded launch artifacts exist but no backend pid/job id is available"
    probe_task = dict(task)
    probe_task["status"] = "running"
    probe_task["node"] = node
    try:
        res = deps.backend.batch_probe({"tasks": [probe_task]}).get(task.get("id"))
    except Exception as exc:
        return "unknown", f"backend probe raised {str(exc)[:160]}"
    if not res or res.get("state") == "unknown":
        err = (res or {}).get("error") or "backend probe returned unknown"
        return "unknown", str(err)[:200]
    if res.get("state") == "alive":
        fallback = str(res.get("probe_fallback") or "")
        if terminal and (fallback == "windows_queue_accounting" or fallback.startswith("windows_parse_failed")):
            return (
                "dead",
                f"terminal Windows task only has unverified liveness fallback {fallback}; "
                "treating recorded pid as stale",
            )
        return "alive", "backend probe reports recorded pid/job alive"
    if res.get("state") == "dead":
        return "dead", "backend probe reports recorded pid/job dead"
    return "unknown", f"unexpected backend state {res.get('state')!r}"


def same_run_identity_live_artifact_reason(
    task: dict,
    state: dict,
    run_key=None,
    *,
    deps: TaskIdentityDeps,
) -> Optional[str]:
    key = run_key if run_key is not None else task_run_identity(task)
    if not key:
        return None
    target_id = task.get("id")
    tasks = state.get("tasks", [])
    by_id = {t.get("id"): t for t in tasks if t.get("id")}
    ancestor_ids = set()
    cur = task
    seen = set()
    while cur:
        pid = cur.get("parent_id")
        if not pid or pid in seen:
            break
        seen.add(pid)
        ancestor_ids.add(pid)
        cur = by_id.get(pid)
    for other in tasks:
        if other is task or other.get("id") == target_id:
            continue
        if other.get("id") in ancestor_ids:
            # A retry clone intentionally represents the same run identity as
            # its parent. Old launch handles retained on the terminal lineage
            # are audit data, not a reason to block that lineage's next retry.
            continue
        if task_run_identity(other) != key:
            continue
        if other.get("status") in ("queued", "running", "launching"):
            continue
        if not task_has_recorded_launch_artifacts(other, deps=deps):
            continue
        launch_state, launch_reason = recorded_launch_safety_state(other, deps=deps)
        if launch_state in ("alive", "unknown"):
            return (
                f"run identity has {launch_state} launch artifacts in terminal task "
                f"{other.get('id')} ({launch_reason}); refusing duplicate dispatch"
            )
    return None
