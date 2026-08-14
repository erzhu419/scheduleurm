"""Crash requeue cloning and dedup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


HARD_FAIL_CATEGORIES = {
    "ARTIFACT_MISSING",
    "ENV_MISSING",
    "PYTHON_IMPORT",
    "CUDA_RUNTIME",
    "INVALID_FLAG",
    "OOM",
    "DISK_FULL",
    "PROTOCOL_INTEGRITY",
}


PROTOCOL_INTEGRITY_PATTERNS = (
    "protocol_integrity:",
    "source snapshot changed during/re-entering the paired protocol",
    "runtime/node/gpu identity changed during/re-entering the paired protocol",
    "refusing to mix sources",
    "source manifest differs from protocol checkpoint",
    "source archive differs from protocol checkpoint",
    "recovery is finalize-only and will not train",
    "partial completion artifacts exist; refusing recovery",
    "the two common-restart physical rollouts differ at iter 700",
    "the two common-restart physical rollout field hashes differ",
    "paired physical rollout proof is inconsistent",
    "paired physical rollout summary is invalid",
    "wrong paired physical rollout hash",
    "wrong paired physical rollout field hashes",
)


RETRY_CLEAR_KEYS = (
    "requeued_as",
    "cancelled_at",
    "cancelled_by_user",
    "cancel_reason",
    "cancelled_by",
    "cancel_actor",
    "last_killed_at",
    "last_killed_by",
    "last_kill_actor",
    "last_kill_action",
    "last_kill_reason",
    "claim_intent_node",
    "claim_intent_nodes",
    "claim_intent_at",
    "probe_unknown_since",
    "last_probe_unknown_at",
    "probe_unknown_count",
    "last_probe_unknown_duration_s",
    "last_probe_unknown_count",
    "last_probe_unknown_reason",
    "last_reconnected_at",
    "cpu_fallback_selected",
    "cpu_fallback_original_vram_mb",
    "cpu_fallback_capability",
    "windows_pin_base",
    "windows_pin_cores",
    "bootstrap_stdout_path",
    "bootstrap_stderr_path",
    "local_ssh_log_path",
    "launch_token",
    "wrapper_pid_path",
    "exit_status_path",
    "exit_status_token",
    "remote_pid_start_ticks",
    "backend_finished_at",
    "last_launch_pre_snapshot",
    "last_launch_post_snapshot",
    "last_node",
    "last_gpu_idx",
    "allow_initial_resume_scan_error",
    "staged_node",
    "migrated_from",
    "migrated_at",
    "resume_locations",
    "resume_scan_errors",
    "resume_preferred_nodes",
    "resume_checkpoint_node",
    "resume_scan_at",
    "resume_scan_key",
    "requeue_deferred_reason",
    "requeue_deferred_at",
    "terminal_transition_deferred",
    "terminal_transition_deferred_at",
    "resolved_environment_retry",
)


@dataclass(frozen=True)
class CrashRequeueDeps:
    max_auto_retry: int
    allocate_task_id: Callable[[dict], str]
    recorded_launch_safety_state: Callable[[dict], tuple[str, str]]
    task_run_identity: Callable[[dict], Any]
    has_user_cancelled_retry_descendant: Callable[[dict, dict, Any], bool]
    classify_failure: Callable[[dict], str]
    write_escalation: Callable[[dict, str, dict], None]
    clear_live_eta_fields: Callable[..., bool]
    local_user: Callable[[], str]
    local_host_short: Callable[[], str]
    scheduler_id: Callable[[], str]
    now: Callable[[], float]


def freqduet_keyerr25_bugfix_retry_allowed(task: dict, diag: dict) -> bool:
    if task.get("bugfix_requeue_reason"):
        return False
    signature = str(task.get("signature") or "")
    cmd = str(task.get("cmd") or "")
    if not signature.startswith("FreqDuet/paper_route_day_policy_v1b_ep100_wu10_20seed/"):
        return False
    if "run_freqduet_ablation.py" not in cmd:
        return False
    tail = str((diag or {}).get("tail") or "")
    reason = str((diag or {}).get("reason") or "")
    haystack = f"{tail}\n{reason}"
    return "KeyError: 25" in haystack and "action[bus.bus_id]" in haystack


def protocol_integrity_failure(diag: dict) -> bool:
    tail = str((diag or {}).get("tail") or "")
    reason = str((diag or {}).get("reason") or "")
    haystack = f"{tail}\n{reason}".lower()
    return any(pattern in haystack for pattern in PROTOCOL_INTEGRITY_PATTERNS)


def requeue_after_crash(parent: dict, state: dict, *, deps: CrashRequeueDeps):
    """Clone a crashed scheduler-owned task back into the queue.

    Returns the new task id, an existing active duplicate id, or None if the
    task is ineligible or should escalate instead of retrying.
    """
    cmd = parent.get("cmd") or ""
    if cmd.startswith("(auto-adopted"):
        return None

    launch_state, launch_reason = deps.recorded_launch_safety_state(parent)
    if launch_state == "alive":
        parent["last_block_reason"] = (
            f"not auto-requeued: recorded launch artifact is {launch_state} "
            f"({launch_reason}); refusing to duplicate the same run"
        )
        return None
    if launch_state == "unknown":
        now = deps.now()
        reason = (
            f"auto-requeue deferred: recorded launch artifact is unknown "
            f"({launch_reason}); will retry after probe reconnects"
        )
        parent["last_block_reason"] = reason
        parent["requeue_deferred_reason"] = reason
        parent["requeue_deferred_at"] = now
        parent["terminal_transition_deferred"] = True
        parent["terminal_transition_deferred_at"] = now
        parent["status"] = "running"
        parent.pop("finished_at", None)
        parent["node_probe_state"] = "unknown"
        if not parent.get("probe_unknown_since"):
            parent["probe_unknown_since"] = now
        parent["last_probe_unknown_at"] = now
        parent["probe_unknown_count"] = int(parent.get("probe_unknown_count") or 0) + 1
        parent["last_probe_unknown_reason"] = str(launch_reason or "backend probe returned unknown")[:300]
        parent["last_status_sync_at"] = now
        parent["last_status_sync_status"] = "running"
        parent["last_status_sync_reason"] = reason
        return None

    parent_key = deps.task_run_identity(parent)
    if parent_key:
        if deps.has_user_cancelled_retry_descendant(parent, state, parent_key):
            parent["last_block_reason"] = (
                "not auto-requeued: a retry descendant for this exact run "
                "identity was cancelled by the user"
            )
            return None
        for existing in state.get("tasks", []):
            if existing.get("id") == parent.get("id"):
                continue
            if existing.get("status") not in ("queued", "running", "launching"):
                continue
            if deps.task_run_identity(existing) != parent_key:
                continue
            return existing["id"]

    diag = parent.get("_diagnosis") or {}
    category = (
        "PROTOCOL_INTEGRITY"
        if protocol_integrity_failure(diag)
        else deps.classify_failure(diag)
    )
    parent["failure_category"] = category
    resolved_environment_retry = bool(parent.get("resolved_environment_retry"))
    if category in HARD_FAIL_CATEGORIES and not resolved_environment_retry:
        deps.write_escalation(parent, category, diag)
        return None

    retry_n = parent.get("retry_count", 0) + 1
    allow_freqduet_keyerr25_bugfix_retry = (
        retry_n == deps.max_auto_retry + 1
        and freqduet_keyerr25_bugfix_retry_allowed(parent, diag)
    )
    if (
        retry_n > deps.max_auto_retry
        and not allow_freqduet_keyerr25_bugfix_retry
        and not resolved_environment_retry
    ):
        deps.write_escalation(parent, "APP_BUG_CAP", diag)
        return None

    new_id = deps.allocate_task_id(state)
    new_task = {**parent}
    new_task.update(
        {
            "id": new_id,
            "status": "queued",
            "node": None,
            "gpu_idx": None,
            "remote_pids": [],
            "log_path": None,
            "started_at": None,
            "finished_at": None,
            "peak_vram_mb": 0,
            "peak_ram_mb": 0,
            "current_vram_mb": 0,
            "current_ram_mb": 0,
            "current_pcpu": 0.0,
            "alive_pids": [],
            "resume_from": None,
            "slurm_job_id": None,
            "slurm_state": None,
            "actual_started_at": None,
            "container_name": None,
            "container_main_pid": None,
            "adopted": False,
            "auto_adopted": False,
            "origin": "scheduleurm",
            "submitted_by": deps.local_user(),
            "submitted_host": deps.local_host_short(),
            "scheduler_id": deps.scheduler_id(),
            "process_group": None,
            "_diagnosis": None,
            "result_artifacts": [],
            "result_artifacts_discovered_at": None,
            "notified_launch": False,
            "notified_done": False,
            "retry_count": retry_n,
            "parent_id": parent["id"],
            "last_block_reason": (
                f"bugfix-requeue after {parent['id']} KeyError:25; "
                "FreqDuet env/sim.py now expands agent slots dynamically"
                if allow_freqduet_keyerr25_bugfix_retry
                else (
                    f"environment-recovery requeue after {parent['id']}; "
                    "project staging verified the previously missing script"
                    if resolved_environment_retry
                    else f"auto-requeue (retry {retry_n}/{deps.max_auto_retry}) after {parent['id']} crashed"
                )
            ),
        }
    )
    if allow_freqduet_keyerr25_bugfix_retry:
        new_task["bugfix_requeue_reason"] = "FreqDuet KeyError:25 max_agent_num/action slot fix"
        new_task["bugfix_requeue_at"] = deps.now()

    deps.clear_live_eta_fields(new_task, clear_runtime_projection=True)
    for key in RETRY_CLEAR_KEYS:
        new_task.pop(key, None)
    if resolved_environment_retry:
        new_task["recovered_from_environment_resolution"] = dict(
            parent.get("resolved_environment_retry") or {}
        )
    state["tasks"].append(new_task)
    return new_id
