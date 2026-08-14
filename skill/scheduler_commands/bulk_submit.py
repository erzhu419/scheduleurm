"""Trusted bulk submit helpers for scheduler task specs."""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from typing import Any, Callable

try:
    from scheduler_result.bulk_conflicts import bulk_submit_result_conflict_message
    from scheduler_cpu.batch_submit_spec import (
        CpuBatchSubmitDeps,
        build_cpu_batch_submit_spec,
    )
except ModuleNotFoundError:
    from .scheduler_result.bulk_conflicts import bulk_submit_result_conflict_message
    from .scheduler_cpu.batch_submit_spec import (
        CpuBatchSubmitDeps,
        build_cpu_batch_submit_spec,
    )


@dataclass(frozen=True)
class BulkSubmitDeps:
    known_nodes: Any
    default_vram_mb: int
    default_ram_mb: int
    default_cpu_cores: int
    canonical_node_name: Callable[[Any], str]
    canonicalize_node_list: Callable[[Any], list]
    parse_env: Callable[[Any], dict]
    task_run_identity: Callable[[dict], Any]
    history_get: Callable[[str], dict]
    project_from_path: Callable[[str], str]
    allocate_task_id: Callable[[dict], str]
    local_user: Callable[[], str]
    local_host_short: Callable[[], str]
    scheduler_id: Callable[[], str]
    seed_pending_eta_from_history: Callable[..., Any]
    now: Callable[[], float] = _time.time


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


def _eta_lookup_metadata_from_spec(spec: dict) -> dict:
    return {
        field: spec[field]
        for field in _ETA_LOOKUP_METADATA_FIELDS
        if field in spec and spec[field] is not None
    }


def _parse_submitted_env(raw_env: Any, deps: BulkSubmitDeps) -> dict:
    if not raw_env:
        return {}
    if isinstance(raw_env, dict):
        raw_env = [f"{key}={value}" for key, value in raw_env.items()]
    elif isinstance(raw_env, str):
        raw_env = [raw_env]
    return dict(deps.parse_env(raw_env) or {})


def _submitted_extra_env(spec: dict, deps: BulkSubmitDeps) -> dict:
    extra_env = _parse_submitted_env(spec.get("extra_env"), deps)
    extra_env.update(_parse_submitted_env(spec.get("env"), deps))
    return extra_env


def load_submit_jsonl_specs_from_text(text: str, *, source: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except Exception as e:
            raise ValueError(f"submit-jsonl {source}: invalid JSON: {e}") from e
        if not isinstance(data, list):
            raise ValueError(f"submit-jsonl {source}: top-level JSON must be a list")
        specs = data
    else:
        specs = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                specs.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"submit-jsonl {source}:{lineno}: invalid JSON: {e}") from e
    for idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"submit-jsonl {source}: item {idx} is not an object")
    return specs


def spec_bool(spec: dict, key: str, default: bool = False) -> bool:
    val = spec.get(key, default)
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def spec_int(spec: dict, key: str, default=None):
    val = spec.get(key, default)
    if val is None or val == "":
        return default
    return int(val)


def build_bulk_submit_task(
    state: dict,
    spec: dict,
    *,
    deps: BulkSubmitDeps,
    task_id: str | None = None,
    resource_history: dict | None = None,
    runtime_history: dict | None = None,
    runtime_closest_index: list[dict] | None = None,
) -> dict:
    missing = [k for k in ("description", "cmd", "cwd", "signature") if not spec.get(k)]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")

    sig = str(spec["signature"])
    cmd = str(spec["cmd"])
    cwd = str(spec["cwd"])
    description = str(spec["description"])
    resource_family = str(
        spec.get("resource_family") or spec.get("resource_class") or ""
    ).strip() or None
    vram_resource_family = str(
        spec.get("vram_resource_family") or ""
    ).strip() or None
    ram_resource_family = str(
        spec.get("ram_resource_family") or ""
    ).strip() or None
    require_node = spec.get("require_node")
    preferred_node = spec.get("preferred_node")
    require_gpu_idx = spec_int(spec, "require_gpu_idx", None)
    if require_gpu_idx is not None and require_gpu_idx < 0:
        raise ValueError("require_gpu_idx must be a non-negative integer")
    if require_node:
        require_node = deps.canonical_node_name(require_node)
    if preferred_node:
        preferred_node = deps.canonical_node_name(preferred_node)
    raw_allowed = spec.get("allowed_nodes")
    if raw_allowed is None:
        raw_allowed = spec.get("allowed_node")
    if isinstance(raw_allowed, str):
        raw_allowed = [x.strip() for x in raw_allowed.split(",") if x.strip()]
    allowed_nodes = deps.canonicalize_node_list(raw_allowed or [])
    bad_nodes = [
        n for n in [require_node, preferred_node, *allowed_nodes]
        if n and n not in deps.known_nodes
    ]
    if bad_nodes:
        raise ValueError(f"unknown node(s): {bad_nodes}; known={list(deps.known_nodes)}")

    extra_env = _submitted_extra_env(spec, deps)
    env_spec = spec.get("env_spec") or "none"
    image = spec.get("image") or ""
    ckpt_dir = spec.get("ckpt_dir")
    ckpt_glob = spec.get("ckpt_glob") or "*"
    resume_flag = spec.get("resume_flag") or ""
    result_dir = spec.get("result_dir") or None
    local_result_dir = spec.get("local_result_dir") or None
    cpu_parallel_items = spec_int(spec, "cpu_parallel_items", 0) or 0
    cpu_parallel_total_items = (
        spec_int(spec, "cpu_parallel_total_items", 0) or cpu_parallel_items
    )
    cpu_parallel_logical_items = spec_int(spec, "cpu_parallel_logical_items", 0) or 0
    if cpu_parallel_items and not cpu_parallel_logical_items:
        cpu_parallel_logical_items = cpu_parallel_total_items
    cpu_parallel_item_multiplier = spec_int(spec, "cpu_parallel_item_multiplier", 1) or 1
    cpu_parallel_start = spec_int(spec, "cpu_parallel_start", 0) or 0
    cpu_parallel_end = spec_int(spec, "cpu_parallel_end", 0) or cpu_parallel_items
    cpu_parallel_shard_index = spec_int(spec, "cpu_parallel_shard_index", 0) or 0
    cpu_parallel_num_shards = spec_int(spec, "cpu_parallel_num_shards", 1) or 1
    cpu_batch_plan = (
        spec.get("cpu_batch_plan") if isinstance(spec.get("cpu_batch_plan"), dict) else None
    )

    allow_duplicate = spec_bool(spec, "allow_duplicate", False)
    if not allow_duplicate:
        submit_identity = deps.task_run_identity({
            "signature": sig,
            "cmd": cmd,
            "cwd": cwd,
            "extra_env": extra_env,
            "env_spec": env_spec,
            "image": image,
            "ckpt_dir": ckpt_dir,
            "ckpt_glob": ckpt_glob,
            "resume_flag": resume_flag,
            "result_dir": result_dir,
            "local_result_dir": local_result_dir,
            "cpu_parallel_items": cpu_parallel_items,
            "cpu_parallel_total_items": cpu_parallel_total_items,
            "cpu_parallel_logical_items": cpu_parallel_logical_items,
            "cpu_parallel_item_multiplier": cpu_parallel_item_multiplier,
            "cpu_parallel_start": cpu_parallel_start,
            "cpu_parallel_end": cpu_parallel_end,
            "allowed_nodes": allowed_nodes,
            "require_node": require_node,
            "preferred_node": preferred_node,
            "require_gpu_idx": require_gpu_idx,
        })
        for existing in state["tasks"]:
            if deps.task_run_identity(existing) != submit_identity:
                continue
            if existing.get("status") in ("queued", "running", "launching"):
                raise ValueError(
                    f"duplicate active task {existing['id']} for signature {sig!r}; "
                    "set allow_duplicate=true to override"
                )

    if resource_history is not None:
        hist = resource_history.get(sig) or {}
        if isinstance(hist, int):
            hist = {"vram_mb": hist}
    else:
        hist = deps.history_get(sig) or {}
    vram_given = spec.get("vram") is not None
    ram_given = spec.get("ram_mb") is not None
    cpu_given = spec.get("cpu") is not None
    if vram_given:
        est_vram = int(spec.get("vram"))
    elif hist.get("vram_mb"):
        est_vram = int(hist["vram_mb"])
    else:
        est_vram = deps.default_vram_mb
    if ram_given:
        ram_mb = int(spec.get("ram_mb"))
    elif hist.get("ram_mb"):
        ram_mb = int(hist["ram_mb"])
    else:
        ram_mb = deps.default_ram_mb
    if cpu_given:
        cpu_cores = int(spec.get("cpu"))
    elif hist.get("cpu_cores"):
        cpu_cores = int(hist["cpu_cores"])
    else:
        cpu_cores = deps.default_cpu_cores
    project = (
        spec.get("project")
        or deps.project_from_path(cwd)
        or (sig.split("/", 1)[0] if "/" in sig else sig)
    )
    stage_excludes = [
        str(x).strip()
        for x in (spec.get("stage_excludes") or spec.get("stage_exclude") or [])
        if str(x).strip()
    ]
    stage_input_paths = [
        str(x).strip()
        for x in (spec.get("stage_input_paths") or spec.get("stage_input_path") or [])
        if str(x).strip()
    ]
    task_id = task_id or deps.allocate_task_id(state)
    task = {
        "id": task_id,
        "status": "queued",
        "parent_id": str(spec.get("parent_id") or "").strip() or None,
        "retry_count": max(0, spec_int(spec, "retry_count", 0) or 0),
        "description": description,
        "project": project,
        "cmd": cmd,
        "cwd": cwd,
        "signature": sig,
        **_eta_lookup_metadata_from_spec(spec),
        "resource_family": resource_family,
        "vram_resource_family": vram_resource_family,
        "ram_resource_family": ram_resource_family,
        "est_vram_mb": int(est_vram),
        "ram_mb": int(ram_mb),
        "cpu_cores": int(cpu_cores),
        "cpu_declared_cores": int(cpu_cores),
        "est_vram_mb_explicit": bool(vram_given),
        "ram_mb_explicit": bool(ram_given),
        "cpu_cores_explicit": bool(cpu_given),
        "priority": spec.get("priority") or "normal",
        "preferred_node": preferred_node,
        "require_node": require_node,
        "require_gpu_idx": require_gpu_idx,
        "allowed_nodes": allowed_nodes or None,
        "allowed_nodes_user_explicit": bool(allowed_nodes),
        "allowed_nodes_submitted": list(allowed_nodes) if allowed_nodes else None,
        "stage_excludes": stage_excludes or None,
        "stage_input_paths": stage_input_paths or None,
        "skip_launch_staging": spec_bool(spec, "skip_launch_staging", False),
        "reroute_on_node_down": spec_bool(spec, "reroute_on_node_down", False),
        "node_down_requeue_s": spec_int(spec, "node_down_requeue_s", 0) or None,
        "git_repo": spec.get("git_repo"),
        "ckpt_dir": ckpt_dir,
        "ckpt_dir_inferred": False,
        "ckpt_inferred_source": "",
        "ckpt_glob": ckpt_glob,
        "resume_flag": resume_flag,
        "resume_managed_by_cmd": spec_bool(
            spec, "resume_managed_by_cmd", False
        ),
        "allow_initial_resume_scan_error": spec_bool(
            spec, "allow_initial_resume_scan_error", False
        ),
        "skip_resume_scan": spec_bool(spec, "skip_resume_scan", False),
        "result_dir": result_dir,
        "local_result_dir": local_result_dir,
        "wait_for_files": list(spec.get("wait_for_files") or []),
        "result_synced_at": None,
        "result_sync_error": None,
        "result_sync_attempts": 0,
        "result_syncing_at": None,
        "extra_env": extra_env,
        "origin": "scheduleurm",
        "submitted_by": deps.local_user(),
        "submitted_host": deps.local_host_short(),
        "scheduler_id": deps.scheduler_id(),
        "cpu_parallel_items": cpu_parallel_items or None,
        "cpu_parallel_total_items": cpu_parallel_total_items if cpu_parallel_items else None,
        "cpu_parallel_logical_items": cpu_parallel_logical_items if cpu_parallel_items else None,
        "cpu_parallel_item_multiplier": (
            cpu_parallel_item_multiplier if cpu_parallel_items else None
        ),
        "cpu_parallel_start": cpu_parallel_start if cpu_parallel_items else None,
        "cpu_parallel_end": cpu_parallel_end if cpu_parallel_items else None,
        "cpu_parallel_shard_index": cpu_parallel_shard_index if cpu_parallel_items else None,
        "cpu_parallel_num_shards": cpu_parallel_num_shards if cpu_parallel_items else None,
        "cpu_auto_workers": (cpu_batch_plan or {}).get("workers") if cpu_batch_plan else None,
        "cpu_parallel_waves": (cpu_batch_plan or {}).get("waves") if cpu_batch_plan else None,
        "cpu_parallel_physical_cores": (
            (cpu_batch_plan or {}).get("physical_cores") if cpu_batch_plan else None
        ),
        "cpu_parallel_total_physical_cores": (
            (cpu_batch_plan or {}).get("total_physical_cores")
            or ((cpu_batch_plan or {}).get("physical_cores") if cpu_batch_plan else None)
        ),
        "cpu_parallel_last_wave_items": (
            (cpu_batch_plan or {}).get("last_wave_items") if cpu_batch_plan else None
        ),
        "cpu_batch_plan": dict(cpu_batch_plan) if isinstance(cpu_batch_plan, dict) else None,
        "allow_cpu_training": spec_bool(spec, "allow_cpu_training", False),
        "allow_gpu_over_one_third": spec_bool(
            spec, "allow_gpu_over_one_third", False
        ),
        "cpu_training_justification": str(
            spec.get("cpu_training_justification") or ""
        ).strip(),
        "allow_remote_large_data": spec_bool(spec, "allow_remote_large_data", False),
        "test_log_path": spec.get("test_log") or None,
        "test_log_profile_loaded": False,
        "env_spec": env_spec,
        "image": image,
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
    deps.seed_pending_eta_from_history(
        {"tasks": [task]},
        runtime_history_cache=runtime_history,
        runtime_closest_index=runtime_closest_index,
        resource_history_cache=resource_history,
    )
    return task


def build_bulk_submit_tasks(
    state: dict,
    specs: list[dict],
    *,
    deps: BulkSubmitDeps,
    task_ids: list[str],
    resource_history: dict | None = None,
    runtime_history: dict | None = None,
    runtime_closest_index: list[dict] | None = None,
) -> list[dict]:
    if len(task_ids) != len(specs):
        raise ValueError(
            f"bulk task id count mismatch: ids={len(task_ids)} specs={len(specs)}"
        )
    working_state = dict(state)
    working_state["tasks"] = list(state.get("tasks", []))
    submitted: list[dict] = []
    for idx, (spec, task_id) in enumerate(zip(specs, task_ids)):
        try:
            task = build_bulk_submit_task(
                working_state,
                spec,
                deps=deps,
                task_id=task_id,
                resource_history=resource_history,
                runtime_history=runtime_history,
                runtime_closest_index=runtime_closest_index,
            )
        except Exception as e:
            raise RuntimeError(f"bulk spec #{idx} failed: {e}") from e
        working_state["tasks"].append(task)
        submitted.append(task)
    state.setdefault("tasks", []).extend(submitted)
    return submitted
