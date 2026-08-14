"""Auto-adopt reconciliation for external processes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ExternalReconcileDeps:
    collect_external_task_probe_data: Callable[[Optional[dict]], dict]
    local_user: Callable[[], str]
    task_pids: Callable[[dict], list]
    descendants_of: Callable[[set, dict], set]
    is_scheduler_control_process: Callable[[str, str], bool]
    refresh_adopted_resources: Callable[[dict, list, list], Any]
    infer_adopt_cwd_from_cmdline: Callable[[str, list[str]], str]
    path_under_roots: Callable[[str, list[str]], bool]
    project_from_path: Callable[[str], str]
    history_get: Callable[[str], dict | None]
    allocate_task_id: Callable[[dict], str]
    default_ram_mb: int
    default_vram_mb: int
    now: Callable[[], float]


def _tracked_running_pids(state: dict, deps: ExternalReconcileDeps) -> set:
    tracked = set()
    for task in state["tasks"]:
        if task.get("status") != "running" or not task.get("node"):
            continue
        for pid in deps.task_pids(task):
            tracked.add((task["node"], int(pid)))
    return tracked


def _scheduler_roots_and_pgroups(
    state: dict,
    deps: ExternalReconcileDeps,
) -> tuple[dict, dict]:
    roots_by_node = {}
    pgroups_by_node = {}
    for task in state["tasks"]:
        if task.get("status") != "running" or task.get("auto_adopted") or not task.get("node"):
            continue
        for pid in deps.task_pids(task):
            roots_by_node.setdefault(task["node"], set()).add(int(pid))
        try:
            pgid = int(task.get("process_group") or 0)
        except Exception:
            pgid = 0
        if pgid > 1:
            pgroups_by_node.setdefault(task["node"], set()).add(pgid)
    return roots_by_node, pgroups_by_node


def _forget_misclassified_adopts(
    state: dict,
    scheduler_descendants_by_node: dict,
    scheduler_pgroups_by_node: dict,
    deps: ExternalReconcileDeps,
) -> None:
    for task in state["tasks"]:
        if task.get("status") != "running" or not task.get("auto_adopted") or not task.get("node"):
            continue
        if deps.is_scheduler_control_process(task.get("cwd") or "", task.get("cmd") or ""):
            task["status"] = "forgotten"
            task["finished_at"] = deps.now()
            task["last_block_reason"] = (
                "auto-forgotten: scheduler control process was misclassified by auto-adopt"
            )
            continue
        pids = set(int(pid) for pid in deps.task_pids(task))
        try:
            pgid = int(task.get("process_group") or 0)
        except Exception:
            pgid = 0
        same_scheduler_group = (
            pgid > 1 and pgid in scheduler_pgroups_by_node.get(task["node"], set())
        )
        child_of_scheduler = (
            pids and pids.issubset(scheduler_descendants_by_node.get(task["node"], set()))
        )
        if same_scheduler_group or child_of_scheduler:
            task["status"] = "forgotten"
            task["finished_at"] = deps.now()
            task["last_block_reason"] = (
                "auto-forgotten: duplicate child/process-group of a scheduler-launched task"
            )


def _all_probe_processes(adopt_node_names: list, gpu_proc_lists: list, cpu_proc_lists: list) -> list:
    all_procs = []
    for procs in gpu_proc_lists:
        all_procs.extend(procs)
    gpu_pids_per_node = {
        node: {proc["pid"] for proc in plist}
        for node, plist in zip(adopt_node_names, gpu_proc_lists)
    }
    for procs in cpu_proc_lists:
        for proc in procs:
            if proc["pid"] in gpu_pids_per_node.get(proc["node"], set()):
                continue
            all_procs.append(proc)
    return all_procs


def _candidate_processes(
    all_procs: list,
    tracked: set,
    descendants_by_node: dict,
    scheduler_pgroups_by_node: dict,
    adopt_owners_by_node: dict,
    adopt_roots_by_node: dict,
    adopt_skip_projects_by_node: dict,
    me: str,
    deps: ExternalReconcileDeps,
) -> list[tuple[dict, str]]:
    candidates = []
    for proc in all_procs:
        if (proc["node"], proc["pid"]) in tracked:
            continue
        if proc["pid"] in descendants_by_node.get(proc["node"], set()):
            continue
        try:
            proc_pgid = int(proc.get("pgid") or 0)
        except Exception:
            proc_pgid = 0
        if proc_pgid > 1 and proc_pgid in scheduler_pgroups_by_node.get(proc["node"], set()):
            continue
        if proc.get("is_slurm"):
            continue
        if proc["owner"] not in adopt_owners_by_node.get(proc["node"], {me}):
            continue
        roots = adopt_roots_by_node.get(proc["node"], [])
        adopt_cwd = proc.get("cwd") or deps.infer_adopt_cwd_from_cmdline(
            proc.get("cmdline") or "", roots)
        if not deps.path_under_roots(adopt_cwd, roots):
            continue
        if deps.is_scheduler_control_process(adopt_cwd, proc.get("cmdline") or ""):
            continue
        proc["cwd"] = adopt_cwd
        project = deps.project_from_path(adopt_cwd)
        if not project or project in adopt_skip_projects_by_node.get(proc["node"], set()):
            continue
        candidates.append((proc, project))
    return candidates


def _group_candidates(candidates: list[tuple[dict, str]]) -> dict:
    groups = {}
    for proc, project in candidates:
        key = (proc["node"], proc["gpu_idx"], project, proc.get("pgid") or proc["pid"])
        groups.setdefault(key, []).append(proc)
    return groups


def _adopt_group(
    state: dict,
    node: str,
    gpu_idx,
    project: str,
    pgid,
    procs: list[dict],
    me: str,
    deps: ExternalReconcileDeps,
) -> dict:
    pids = sorted(proc["pid"] for proc in procs)
    sum_vram = sum(proc["used_mb"] for proc in procs)
    sum_rss = sum(proc.get("rss_mb", 0) for proc in procs)
    sum_pcpu = sum(proc.get("pcpu", 0.0) for proc in procs)
    owners = sorted({proc.get("owner") or me for proc in procs})
    proc_owner = owners[0] if len(owners) == 1 else ",".join(owners[:3])
    cwd_sample = procs[0]["cwd"]
    cmdlines = [proc.get("cmdline", "") for proc in procs if proc.get("cmdline")]
    captured_cmd = max(cmdlines, key=len) if cmdlines else ""
    sig = f"{project}/auto-adopted/p{min(pids)}"
    hist = deps.history_get(sig) or {}
    measured_cpu = max(1, math.ceil(sum_pcpu / 100.0)) if sum_pcpu > 0 else len(pids)
    cpu_cores = hist.get("cpu_cores") or measured_cpu
    ram_mb = sum_rss or hist.get("ram_mb") or deps.default_ram_mb
    is_cpu_only_group = gpu_idx is None
    est_vram_for_record = 0 if is_cpu_only_group else (sum_vram or deps.default_vram_mb)
    desc_loc = f"{node}:CPU-only" if is_cpu_only_group else f"{node}:GPU{gpu_idx}"
    task_id = deps.allocate_task_id(state)
    now = deps.now()
    task = {
        "id": task_id,
        "status": "running",
        "description": f"auto-adopted: {project} on {desc_loc} ({len(pids)} procs)",
        "project": project,
        "cmd": captured_cmd or "(auto-adopted by watcher — cmdline not captured)",
        "cwd": cwd_sample,
        "signature": sig,
        "process_group": pgid,
        "est_vram_mb": est_vram_for_record,
        "ram_mb": ram_mb,
        "cpu_cores": cpu_cores,
        "priority": "normal",
        "preferred_node": None,
        "git_repo": None,
        "ckpt_dir": None,
        "ckpt_glob": "*",
        "resume_flag": "",
        "extra_env": {},
        "origin": "external",
        "submitted_by": proc_owner,
        "process_owner": proc_owner,
        "submitted_host": node,
        "scheduler_id": None,
        "shared_account_suspect": True,
        "node": node,
        "gpu_idx": gpu_idx,
        "remote_pids": pids,
        "log_path": None,
        "submitted_at": now,
        "started_at": deps.now(),
        "finished_at": None,
        "peak_vram_mb": sum_vram,
        "peak_ram_mb": sum_rss,
        "current_vram_mb": sum_vram,
        "current_ram_mb": sum_rss,
        "current_pcpu": sum_pcpu,
        "resume_from": None,
        "adopted": True,
        "auto_adopted": True,
        "notified_launch": True,
    }
    state["tasks"].append(task)
    return task


def reconcile_external_tasks(
    state: dict,
    probe_data: Optional[dict] = None,
    *,
    deps: ExternalReconcileDeps,
) -> list:
    """Auto-adopt external GPU/CPU processes and forget duplicate phantom adopts."""
    probe_data = probe_data or deps.collect_external_task_probe_data(state)
    me = probe_data.get("me") or deps.local_user()
    tracked = _tracked_running_pids(state, deps)
    adopt_node_names = list(probe_data.get("adopt_node_names") or [])
    adopt_owners_by_node = dict(probe_data.get("adopt_owners_by_node") or {})
    adopt_roots_by_node = dict(probe_data.get("adopt_roots_by_node") or {})
    adopt_skip_projects_by_node = dict(probe_data.get("adopt_skip_projects_by_node") or {})
    cleanup_node_names = list(probe_data.get("cleanup_node_names") or adopt_node_names)
    gpu_proc_lists = list(probe_data.get("gpu_proc_lists") or [])
    cpu_proc_lists = list(probe_data.get("cpu_proc_lists") or [])
    ppid_maps = list(probe_data.get("ppid_maps") or [])
    ppid_by_node = dict(zip(cleanup_node_names, ppid_maps))

    tracked_pids_by_node = {}
    for node, pid in tracked:
        tracked_pids_by_node.setdefault(node, set()).add(pid)
    scheduler_roots_by_node, scheduler_pgroups_by_node = _scheduler_roots_and_pgroups(
        state, deps)

    descendants_by_node = {
        node: deps.descendants_of(tracked_pids_by_node.get(node, set()), ppid_by_node.get(node, {}))
        for node in adopt_node_names
    }
    scheduler_descendants_by_node = {
        node: deps.descendants_of(
            scheduler_roots_by_node.get(node, set()), ppid_by_node.get(node, {}))
        for node in cleanup_node_names
    }

    _forget_misclassified_adopts(
        state, scheduler_descendants_by_node, scheduler_pgroups_by_node, deps)
    deps.refresh_adopted_resources(state, gpu_proc_lists, cpu_proc_lists)

    candidates = _candidate_processes(
        _all_probe_processes(adopt_node_names, gpu_proc_lists, cpu_proc_lists),
        tracked,
        descendants_by_node,
        scheduler_pgroups_by_node,
        adopt_owners_by_node,
        adopt_roots_by_node,
        adopt_skip_projects_by_node,
        me,
        deps,
    )
    if not candidates:
        return []

    adopted = []
    for (node, gpu_idx, project, pgid), procs in _group_candidates(candidates).items():
        adopted.append(_adopt_group(state, node, gpu_idx, project, pgid, procs, me, deps))
    return adopted
