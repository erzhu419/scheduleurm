"""Resource-pressure summaries and crash/OOM forensics payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ResourceForensicsDeps:
    now: Callable[[], float]
    effective_elapsed_s: Callable[[dict], float]
    discover_result_artifacts: Callable[..., list]
    compact_resource_task_summary: Callable[[dict], dict]
    gpu_threshold_snapshot: Callable[[Optional[dict]], Optional[dict]]
    node_ram_snapshot: Callable[[Optional[dict]], Optional[dict]]
    last_progress_line: Callable[[str], str]
    load_eta_tracker_module: Callable[[], Any]
    ram_evict_ckpt_protect_progress: float
    ram_evict_ckpt_protect_min_age_s: float


def task_progress_ratio(task: dict) -> Optional[float]:
    total = int(task.get("runtime_total_units") or 0)
    current = int(task.get("runtime_current_unit") or 0)
    if total <= 0 or current < 0:
        return None
    return max(0.0, min(1.0, float(current) / float(total)))


def task_has_progress_evidence(task: dict) -> bool:
    ratio = task_progress_ratio(task)
    if ratio is not None and ratio > 0.0:
        return True
    if int(task.get("runtime_current_unit") or 0) > 0:
        return True
    line = task.get("last_progress_line") or ""
    return bool(re.search(r"\b(?:Iter|Epoch|Step|Episode)\s*[#:]?\s*[1-9]\d*\b", line))


def task_ram_pressure_mb(task: dict) -> int:
    return max(
        int(task.get("current_ram_mb") or 0),
        int(task.get("peak_ram_mb") or 0),
        int(task.get("ram_mb") or 0),
    )


def task_has_resume_evidence(task: dict) -> bool:
    return bool(task.get("resume_locations") or task.get("resume_from") or task.get("resume_checkpoint_node"))


def ckpt_task_evict_protected(task: dict, *, deps: ResourceForensicsDeps) -> bool:
    """Avoid killing meaningful checkpoint work unless resume evidence exists."""
    if not task.get("ckpt_dir"):
        return False
    if task_has_resume_evidence(task):
        return False
    elapsed = deps.effective_elapsed_s(task)
    progress = task_progress_ratio(task)
    has_progress = (
        progress is not None
        and progress >= float(deps.ram_evict_ckpt_protect_progress)
    )
    old_enough = elapsed >= float(deps.ram_evict_ckpt_protect_min_age_s)
    return bool(old_enough or has_progress)


def task_has_incremental_result_resume(task: dict, *, deps: ResourceForensicsDeps) -> bool:
    """Detect eval-style resumability via incremental result artifacts."""
    cmd = task.get("cmd") or ""
    if "--skip_existing" not in cmd and "--skip-existing" not in cmd:
        return False
    artifacts = deps.discover_result_artifacts(task, include_log=False)
    return any((rec.get("kind") == "file" and rec.get("path")) for rec in artifacts)


def task_evict_loss_protected(task: dict, *, deps: ResourceForensicsDeps) -> bool:
    return bool(
        ckpt_task_evict_protected(task, deps=deps)
        or task_has_incremental_result_resume(task, deps=deps)
    )


def summarize_task_for_resource_log(task: dict, *, deps: ResourceForensicsDeps) -> dict:
    started = task.get("started_at")
    elapsed = max(0, int(deps.now() - started)) if started else 0
    ratio = task_progress_ratio(task)
    resume_locations = task.get("resume_locations") or []
    ckpt_evict_protected = ckpt_task_evict_protected(task, deps=deps)
    incremental_result_resume = task_has_incremental_result_resume(task, deps=deps)
    return {
        "id": task.get("id"),
        "status": task.get("status"),
        "project": task.get("project"),
        "signature": task.get("signature"),
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "started_at": started,
        "elapsed_s": elapsed,
        "eta_seconds": int(task.get("eta_seconds") or 0),
        "eta_source": task.get("eta_source") or task.get("runtime_est_source"),
        "runtime_current_unit": task.get("runtime_current_unit"),
        "runtime_total_units": task.get("runtime_total_units"),
        "progress_ratio": ratio,
        "current_vram_mb": int(task.get("current_vram_mb") or 0),
        "peak_vram_mb": int(task.get("peak_vram_mb") or 0),
        "current_ram_mb": int(task.get("current_ram_mb") or 0),
        "peak_ram_mb": int(task.get("peak_ram_mb") or 0),
        "cpu_cores": int(task.get("cpu_cores") or 0),
        "windows_pin_base": task.get("windows_pin_base"),
        "windows_pin_cores": task.get("windows_pin_cores"),
        "ram_pressure_mb": task_ram_pressure_mb(task),
        "ckpt_dir": task.get("ckpt_dir"),
        "resume_scan_at": task.get("resume_scan_at"),
        "resume_locations_count": len(resume_locations),
        "resume_checkpoint_node": task.get("resume_checkpoint_node"),
        "ckpt_evict_protected": ckpt_evict_protected,
        "incremental_result_resume": incremental_result_resume,
        "evict_loss_protected": ckpt_evict_protected or incremental_result_resume,
        "last_progress_line": (task.get("last_progress_line") or "")[-240:],
    }


def resource_snapshot_for_placement(
    state: dict,
    nodes: list,
    task: dict,
    node_name: Optional[str],
    gpu_idx,
    stage: str,
    *,
    deps: ResourceForensicsDeps,
) -> dict:
    node_state = next((n for n in nodes if n.get("name") == node_name), None)
    gpu_state = None
    if node_state and gpu_idx is not None:
        for gpu in node_state.get("gpus") or []:
            if gpu.get("idx") == gpu_idx:
                gpu_state = gpu
                break
    same_gpu_tasks = []
    same_node_tasks = []
    for other in state.get("tasks", []):
        if other.get("status") not in ("running", "launching"):
            continue
        if node_name and other.get("node") == node_name:
            summary = deps.compact_resource_task_summary(
                summarize_task_for_resource_log(other, deps=deps)
            )
            same_node_tasks.append(summary)
            if gpu_idx is not None and other.get("gpu_idx") == gpu_idx:
                same_gpu_tasks.append(summary)
    same_gpu_tasks.sort(key=lambda item: item.get("started_at") or 0)
    same_node_tasks.sort(key=lambda item: item.get("started_at") or 0)
    return {
        "stage": stage,
        "ts": deps.now(),
        "task": deps.compact_resource_task_summary(
            summarize_task_for_resource_log(task, deps=deps)
        ),
        "target_node": node_name,
        "target_gpu_idx": gpu_idx,
        "gpu": deps.gpu_threshold_snapshot(gpu_state),
        "node_ram": deps.node_ram_snapshot(node_state),
        "same_gpu_task_count": len(same_gpu_tasks),
        "same_gpu_tasks": same_gpu_tasks[-12:],
        "same_node_running_task_count": len(same_node_tasks),
    }


def remember_running_resource_snapshots(
    state: dict,
    nodes: list,
    *,
    deps: ResourceForensicsDeps,
) -> None:
    active_tasks = [
        task for task in state.get("tasks", [])
        if task.get("status") in ("running", "launching")
    ]
    if not active_tasks:
        return

    node_states = {node.get("name"): node for node in nodes or []}
    gpu_states = {
        (node.get("name"), gpu.get("idx")): gpu
        for node in nodes or []
        for gpu in node.get("gpus") or []
    }
    summaries = {
        id(task): deps.compact_resource_task_summary(
            summarize_task_for_resource_log(task, deps=deps)
        )
        for task in active_tasks
    }
    by_node: dict[Any, list[tuple[float, dict]]] = {}
    by_gpu: dict[tuple[Any, Any], list[tuple[float, dict]]] = {}
    for task in active_tasks:
        started_at = task.get("started_at") or 0
        summary = summaries[id(task)]
        node_name = task.get("node")
        by_node.setdefault(node_name, []).append((started_at, summary))
        by_gpu.setdefault((node_name, task.get("gpu_idx")), []).append(
            (started_at, summary)
        )
    for group in (*by_node.values(), *by_gpu.values()):
        group.sort(key=lambda item: item[0])

    now = deps.now()
    for task in active_tasks:
        if task.get("status") != "running":
            continue
        node_name = task.get("node")
        gpu_idx = task.get("gpu_idx")
        same_node = by_node.get(node_name, [])
        same_gpu = by_gpu.get((node_name, gpu_idx), [])
        task["last_resource_snapshot"] = {
            "stage": "running_probe",
            "ts": now,
            "task": summaries[id(task)],
            "target_node": node_name,
            "target_gpu_idx": gpu_idx,
            "gpu": deps.gpu_threshold_snapshot(
                gpu_states.get((node_name, gpu_idx))
            ),
            "node_ram": deps.node_ram_snapshot(node_states.get(node_name)),
            "same_gpu_task_count": len(same_gpu),
            "same_gpu_tasks": [summary for _, summary in same_gpu[-12:]],
            "same_node_running_task_count": len(same_node),
        }


def log_progress_forensics(
    task: dict,
    tail_text: str,
    elapsed_s: Optional[float] = None,
    *,
    deps: ResourceForensicsDeps,
) -> dict:
    elapsed = float(elapsed_s if elapsed_s is not None else deps.effective_elapsed_s(task))
    out = {
        "last_progress_line": deps.last_progress_line(tail_text),
        "training_started": any(
            marker in tail_text for marker in ("Starting training", "Iter ", "Epoch ", "Step ")
        ),
        "jax_device_seen": "JAX devices:" in tail_text,
    }
    eta_tracker = deps.load_eta_tracker_module()
    if eta_tracker:
        try:
            progress = eta_tracker.parse_progress(tail_text, cmd=task.get("cmd"))
        except Exception:
            progress = None
        if progress:
            cur, total = progress
            out.update({
                "progress_current": int(cur),
                "progress_total": int(total),
                "progress_ratio": float(cur) / float(total) if total else None,
            })
        try:
            projection = eta_tracker.runtime_projection(
                tail_text,
                elapsed_s=elapsed,
                cmd=task.get("cmd"),
            )
        except Exception:
            projection = None
        if projection:
            out.update({
                "eta_source": projection.get("source"),
                "eta_seconds": projection.get("eta_s"),
                "runtime_total_s_est": projection.get("total_s"),
                "runtime_unit_s_est": projection.get("unit_s"),
            })
    if (
        "Failed to create stream executor" in tail_text
        or "Unable to initialize backend 'cuda'" in tail_text
        or "no supported devices found for platform CUDA" in tail_text
    ):
        out["failure_stage"] = "cuda_init"
    elif out["training_started"] and not re.search(r"(?:^|\s)Iter\s+\d+", tail_text):
        out["failure_stage"] = "training_start_before_first_iter"
    elif out.get("progress_current") is not None or re.search(r"(?:^|\s)Iter\s+\d+", tail_text):
        out["failure_stage"] = "mid_training_or_after_progress"
    else:
        out["failure_stage"] = "pre_training_or_unknown"
    return out


def build_crash_forensics_payload(
    task: dict,
    state: dict,
    nodes: Optional[list] = None,
    *,
    deps: ResourceForensicsDeps,
) -> dict:
    diag = task.get("_diagnosis") or {}
    tail = diag.get("tail") or ""
    now = deps.now()
    elapsed = int(
        diag.get("lifetime_s")
        or max(0, (task.get("finished_at") or now) - (task.get("started_at") or now))
    )
    live_snapshot = (
        resource_snapshot_for_placement(
            state,
            nodes or [],
            task,
            task.get("node"),
            task.get("gpu_idx"),
            "crash_probe",
            deps=deps,
        )
        if nodes
        else None
    )
    last_snapshot = (
        task.get("last_resource_snapshot")
        or task.get("last_launch_post_snapshot")
        or task.get("last_launch_pre_snapshot")
    )
    progress = log_progress_forensics(task, tail, elapsed_s=elapsed, deps=deps)
    payload = {
        "id": task.get("id"),
        "project": task.get("project"),
        "signature": task.get("signature"),
        "description": task.get("description", "")[:160],
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "status": task.get("status"),
        "lifetime_s": elapsed,
        "reason": diag.get("reason"),
        "log_path": diag.get("log_path") or task.get("log_path"),
        "log_size": diag.get("log_size"),
        "progress": progress,
        "per_task_vram_known": bool(int(task.get("peak_vram_mb") or 0) > 0),
        "peak_vram_mb": int(task.get("peak_vram_mb") or 0),
        "current_vram_mb": int(task.get("current_vram_mb") or 0),
        "last_resource_snapshot": last_snapshot,
        "crash_probe_snapshot": live_snapshot,
    }
    gpu_snap = None
    if isinstance(last_snapshot, dict):
        gpu_snap = last_snapshot.get("gpu")
    if not gpu_snap and isinstance(live_snapshot, dict):
        gpu_snap = live_snapshot.get("gpu")
    if gpu_snap:
        payload["gpu_over_one_third"] = bool(gpu_snap.get("over_one_third"))
        payload["gpu_over_full"] = bool(gpu_snap.get("over_full"))
        payload["gpu_used_mb"] = gpu_snap.get("used_mb")
        payload["gpu_total_mb"] = gpu_snap.get("total_mb")
        payload["gpu_one_third_mb"] = gpu_snap.get("one_third_mb")
    same_gpu = []
    if isinstance(last_snapshot, dict):
        same_gpu = last_snapshot.get("same_gpu_tasks") or []
    if not same_gpu and isinstance(live_snapshot, dict):
        same_gpu = live_snapshot.get("same_gpu_tasks") or []
    payload["same_gpu_tasks"] = same_gpu
    return payload


def is_oom_like_forensics(payload: dict) -> bool:
    text = " ".join(str(payload.get(k) or "") for k in ("reason", "description"))
    progress = payload.get("progress") or {}
    text += " " + str(progress.get("failure_stage") or "")
    return any(
        needle in text.lower()
        for needle in ("oom", "out of memory", "cuda_error_out_of_memory", "resource_exhausted")
    )


def build_oom_forensics_payload(crash_payload: dict, state: dict) -> dict:
    task_id = crash_payload.get("id")
    same_gpu = crash_payload.get("same_gpu_tasks") or []
    trigger = None
    victim_started = 0.0
    for task in same_gpu:
        if task.get("id") == task_id:
            victim_started = float(task.get("started_at") or 0)
            break
    candidates = [task for task in same_gpu if task.get("id") != task_id]
    if candidates:
        recent = [
            task for task in candidates
            if float(task.get("started_at") or 0) >= victim_started - 60
        ]
        trigger = max(recent or candidates, key=lambda item: float(item.get("started_at") or 0))
    by_id = {task.get("id"): task for task in state.get("tasks", [])}
    trigger_status = None
    if trigger:
        cur = by_id.get(trigger.get("id"))
        trigger_status = cur.get("status") if cur else trigger.get("status")
    return {
        "id": task_id,
        "node": crash_payload.get("node"),
        "gpu_idx": crash_payload.get("gpu_idx"),
        "lifetime_s": crash_payload.get("lifetime_s"),
        "failure_stage": (crash_payload.get("progress") or {}).get("failure_stage"),
        "over_one_third": crash_payload.get("gpu_over_one_third"),
        "over_full": crash_payload.get("gpu_over_full"),
        "exact_per_task_vram_known": crash_payload.get("per_task_vram_known"),
        "gpu_used_mb": crash_payload.get("gpu_used_mb"),
        "gpu_total_mb": crash_payload.get("gpu_total_mb"),
        "gpu_one_third_mb": crash_payload.get("gpu_one_third_mb"),
        "suspected_trigger_task": trigger,
        "trigger_task_status_after_oom": trigger_status,
        "oom_victim_status": by_id.get(task_id, {}).get("status"),
        "reason": crash_payload.get("reason"),
        "log_path": crash_payload.get("log_path"),
    }
