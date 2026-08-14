"""Build and validate a single submitted task record."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from .task_common import (
        SubmitTaskDeps,
        SubmitTaskRefusal,
        SubmitTaskResult,
        _int_arg,
        _stage_excludes,
        _submit_parallel_fields,
        _submit_source_label,
    )
    from .task_resources import (
        _estimate_submit_resources,
        _submit_cpu_plan,
    )
    from .task_validation import (
        _validate_allowed_nodes,
        _validate_ckpt_conflict,
        _validate_duplicate_identity,
        _validate_result_conflict,
    )
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .task_common import (
        SubmitTaskDeps,
        SubmitTaskRefusal,
        SubmitTaskResult,
        _int_arg,
        _stage_excludes,
        _submit_parallel_fields,
        _submit_source_label,
    )
    from .task_resources import (
        _estimate_submit_resources,
        _submit_cpu_plan,
    )
    from .task_validation import (
        _validate_allowed_nodes,
        _validate_ckpt_conflict,
        _validate_duplicate_identity,
        _validate_result_conflict,
    )


_ETA_LOOKUP_METADATA_FIELDS = (
    "theorem_workload_key",
    "workload_key",
    "service_workload_key",
    "workload_env",
    "service_workload_env",
    "allocation_workers",
    "colocation_count",
    "profile",
    "profile_axis",
    "resource_state",
    "resident_mix",
    "resource_kind",
    "node_bucket",
    "hardware_class",
)


def _eta_lookup_metadata_from_args(args: Any) -> dict:
    return {
        field: getattr(args, field)
        for field in _ETA_LOOKUP_METADATA_FIELDS
        if hasattr(args, field) and getattr(args, field) is not None
    }


def _parse_submitted_env(raw_env: Any, deps: SubmitTaskDeps) -> dict:
    if not raw_env:
        return {}
    if isinstance(raw_env, dict):
        raw_env = [f"{key}={value}" for key, value in raw_env.items()]
    elif isinstance(raw_env, str):
        raw_env = [raw_env]
    return dict(deps.parse_env(raw_env) or {})


def _submitted_extra_env(args: Any, deps: SubmitTaskDeps) -> dict:
    extra_env = _parse_submitted_env(getattr(args, "extra_env", None), deps)
    extra_env.update(_parse_submitted_env(getattr(args, "env", None), deps))
    return extra_env


def _task_project(args: Any, sig: str, deps: SubmitTaskDeps) -> str:
    return (
        args.project
        or deps.project_from_path(args.cwd)
        or (sig.split("/", 1)[0] if "/" in sig else sig)
    )


def _parallel_task_fields(parallel: dict) -> dict:
    items = parallel["items"]
    return {
        "cpu_parallel_items": items or None,
        "cpu_parallel_total_items": parallel["total_items"] if items else None,
        "cpu_parallel_logical_items": parallel["logical_items"] if items else None,
        "cpu_parallel_item_multiplier": parallel["item_multiplier"] if items else None,
        "cpu_parallel_start": parallel["start"] if items else None,
        "cpu_parallel_end": parallel["end"] if items else None,
        "cpu_parallel_shard_index": parallel["shard_index"] if items else None,
        "cpu_parallel_num_shards": parallel["num_shards"] if items else None,
    }


def _cpu_plan_task_fields(cpu_auto_plan: dict | None) -> dict:
    return {
        "cpu_auto_workers": (
            (cpu_auto_plan or {}).get("workers") if cpu_auto_plan else None
        ),
        "cpu_parallel_waves": (
            (cpu_auto_plan or {}).get("waves") if cpu_auto_plan else None
        ),
        "cpu_parallel_physical_cores": (
            (cpu_auto_plan or {}).get("physical_cores") if cpu_auto_plan else None
        ),
        "cpu_parallel_total_physical_cores": (
            (cpu_auto_plan or {}).get("total_physical_cores")
            or ((cpu_auto_plan or {}).get("physical_cores") if cpu_auto_plan else None)
        ),
        "cpu_parallel_last_wave_items": (
            (cpu_auto_plan or {}).get("last_wave_items") if cpu_auto_plan else None
        ),
    }


def _build_task_record(
    state: dict,
    args: Any,
    preflight: Any,
    *,
    sig: str,
    extra_env: dict,
    allowed_nodes: list[str],
    stage_excludes: list[str],
    parallel: dict,
    est_vram: int,
    ram_mb: int,
    cpu_cores: int,
    cpu_auto_plan: dict | None,
    deps: SubmitTaskDeps,
) -> dict:
    cpu_batch_plan = getattr(args, "cpu_batch_plan", None)
    task = {
        "id": deps.allocate_task_id(state),
        "status": "queued",
        "description": args.description,
        "project": _task_project(args, sig, deps),
        "cmd": args.cmd,
        "cwd": args.cwd,
        "signature": sig,
        **_eta_lookup_metadata_from_args(args),
        "resource_family": (
            str(getattr(args, "resource_family", None) or "").strip() or None
        ),
        "vram_resource_family": (
            str(getattr(args, "vram_resource_family", None) or "").strip() or None
        ),
        "ram_resource_family": (
            str(getattr(args, "ram_resource_family", None) or "").strip() or None
        ),
        "est_vram_mb": int(est_vram),
        "ram_mb": int(ram_mb),
        "cpu_cores": int(cpu_cores),
        "cpu_declared_cores": int(cpu_cores),
        "est_vram_mb_explicit": args.vram is not None,
        "ram_mb_explicit": args.ram_mb is not None,
        "cpu_cores_explicit": args.cpu is not None,
        "priority": args.priority,
        "preferred_node": args.preferred_node,
        "require_node": args.require_node,
        "allowed_nodes": allowed_nodes or None,
        "allowed_nodes_user_explicit": bool(allowed_nodes),
        "allowed_nodes_submitted": list(allowed_nodes) if allowed_nodes else None,
        "stage_excludes": stage_excludes or None,
        "stage_input_paths": list(
            getattr(args, "stage_input_paths", None) or []
        ) or None,
        "skip_launch_staging": bool(getattr(args, "skip_launch_staging", False)),
        "reroute_on_node_down": bool(getattr(args, "reroute_on_node_down", False)),
        "node_down_requeue_s": _int_arg(args, "node_down_requeue_s") or None,
        "git_repo": args.git_repo,
        "ckpt_dir": args.ckpt_dir,
        "ckpt_dir_inferred": bool(preflight.ckpt_dir_was_inferred),
        "ckpt_inferred_source": (
            preflight.inferred_ckpt_source if preflight.ckpt_dir_was_inferred else ""
        ),
        "ckpt_glob": args.ckpt_glob,
        "resume_flag": args.resume_flag or "",
        "resume_managed_by_cmd": bool(
            preflight.inferred_resume_managed and not args.resume_flag
        ),
        "skip_resume_scan": bool(
            preflight.inferred_resume_managed and not args.resume_flag
        ),
        "allow_initial_resume_scan_error": bool(
            getattr(args, "allow_initial_resume_scan_error", False)
        ),
        "result_dir": getattr(args, "result_dir", None) or None,
        "local_result_dir": getattr(args, "local_result_dir", None) or None,
        "wait_for_files": list(getattr(args, "wait_for_files", None) or []),
        "result_synced_at": None,
        "result_sync_error": None,
        "result_sync_attempts": 0,
        "result_syncing_at": None,
        "extra_env": extra_env,
        "origin": "scheduleurm",
        "submitted_by": deps.local_user(),
        "submitted_host": deps.local_host_short(),
        "scheduler_id": deps.scheduler_id(),
        **_parallel_task_fields(parallel),
        **_cpu_plan_task_fields(cpu_auto_plan),
        "cpu_batch_plan": dict(cpu_batch_plan) if isinstance(cpu_batch_plan, dict) else None,
        "allow_cpu_training": bool(getattr(args, "allow_cpu_training", False)),
        "cpu_training_justification": (
            getattr(args, "cpu_training_justification", "") or ""
        ).strip(),
        "allow_remote_large_data": bool(getattr(args, "allow_remote_large_data", False)),
        "test_log_path": os.path.expanduser(getattr(args, "test_log", "") or "") or None,
        "test_log_profile_loaded": False,
        "env_spec": getattr(args, "env_spec", None) or "none",
        "image": getattr(args, "image", None) or "",
        "node": None,
        "gpu_idx": None,
        "remote_pids": [],
        "log_path": None,
        "submitted_at": deps.now(),
        "started_at": None,
        "finished_at": None,
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "current_vram_mb": 0,
        "current_ram_mb": 0,
        "current_pcpu": 0.0,
        "resume_from": None,
    }
    if getattr(args, "test_log", None):
        task["test_log_profile_loaded"] = deps.apply_test_log_runtime_profile(
            task, args.test_log)
        if not task["test_log_profile_loaded"]:
            task["last_block_reason"] = (
                "test log was provided but no tqdm/progress runtime profile "
                f"could be parsed: {args.test_log}"
            )
    return task


def build_submit_task_for_state(
    state: dict,
    args: Any,
    preflight: Any,
    *,
    deps: SubmitTaskDeps,
    default_ram_mb: int,
    default_cpu_cores: int,
) -> SubmitTaskResult:
    """Validate active-state conflicts and return the queued task record."""
    sig = args.signature
    extra_env = _submitted_extra_env(args, deps)
    parallel = _submit_parallel_fields(args)
    allowed_nodes = list(getattr(args, "allowed_nodes", None) or [])
    _validate_allowed_nodes(allowed_nodes, deps)

    _validate_duplicate_identity(
        state,
        args,
        sig=sig,
        extra_env=extra_env,
        allowed_nodes=allowed_nodes,
        parallel=parallel,
        deps=deps,
    )
    _validate_ckpt_conflict(state, args, sig=sig, deps=deps)
    _validate_result_conflict(state, args, deps)

    hist, est_vram, ram_mb = _estimate_submit_resources(
        state,
        args,
        sig=sig,
        deps=deps,
        default_ram_mb=default_ram_mb,
    )
    cpu_cores, cpu_auto_plan = _submit_cpu_plan(
        args,
        hist,
        parallel,
        deps=deps,
        default_cpu_cores=default_cpu_cores,
    )
    task = _build_task_record(
        state,
        args,
        preflight,
        sig=sig,
        extra_env=extra_env,
        allowed_nodes=allowed_nodes,
        stage_excludes=_stage_excludes(args),
        parallel=parallel,
        est_vram=est_vram,
        ram_mb=ram_mb,
        cpu_cores=cpu_cores,
        cpu_auto_plan=cpu_auto_plan,
        deps=deps,
    )
    deps.seed_pending_eta_from_history({"tasks": [task]})
    return SubmitTaskResult(
        task=task,
        hist=hist,
        est_vram=est_vram,
        ram_mb=ram_mb,
        cpu_cores=cpu_cores,
        source_label=_submit_source_label(args, hist, parallel["items"]),
    )
