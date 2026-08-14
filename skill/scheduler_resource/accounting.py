"""Resource accounting payloads for watcher and dispatch logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ResourceAccountingDeps:
    scheduler_id: Callable[[], str]
    reserved_cpu_slots_for_task: Callable[[dict], int]
    node_physical_cores: Callable[..., int]
    algorithm_name: Callable[[], str]
    algorithm_config_snapshot: Callable[[], dict]


@dataclass(frozen=True)
class AggregateVramReconcileDeps:
    is_slurm_managed: Callable[[dict], bool]


@dataclass(frozen=True)
class InflightVramReservationDeps:
    node_configs: dict
    startup_floor_mb: int
    is_slurm_managed: Callable[[dict], bool]
    hard_rule_bypassed: Callable[..., bool]


def cpu_ownership_snapshot(state: dict, nodes: list, *, deps: ResourceAccountingDeps) -> list:
    """Estimate per-node CPU attribution for JSONL logs."""
    own_sid = deps.scheduler_id()
    tracked = {}
    for task in state.get("tasks", []):
        if task.get("status") not in ("running", "launching"):
            continue
        node = task.get("node") or task.get("assigned_node")
        if not node:
            continue
        cpu = deps.reserved_cpu_slots_for_task(task)
        sid = task.get("scheduler_id")
        ours = (
            task.get("origin") == "scheduleurm"
            and not task.get("auto_adopted")
            and (not sid or sid == own_sid)
        )
        bucket = tracked.setdefault(node, {
            "ours_cpu": 0,
            "ours_tasks": [],
            "external_cpu": 0,
            "external_tasks": [],
        })
        rec = {
            "id": task.get("id"),
            "cpu_cores": cpu,
            "status": task.get("status"),
            "owner": task.get("process_owner") or task.get("submitted_by"),
            "scheduler_id": sid,
            "description": (task.get("description") or "")[:80],
        }
        if ours:
            bucket["ours_cpu"] += cpu
            bucket["ours_tasks"].append(rec)
        else:
            bucket["external_cpu"] += cpu
            bucket["external_tasks"].append(rec)

    out = []
    for node_state in nodes or []:
        name = node_state.get("name")
        bucket = tracked.get(name, {})
        try:
            total = int(node_state.get("total_cpu") or deps.node_physical_cores(name, node_state) or 0)
        except Exception:
            total = 0
        try:
            free = max(0, int(node_state.get("free_cpu") or 0))
        except Exception:
            free = 0
        if node_state.get("alive") is False:
            free = 0
        used_est = max(0, total - free)
        ours_cpu = int(bucket.get("ours_cpu") or 0)
        external_cpu = int(bucket.get("external_cpu") or 0)
        untracked_other = max(0, used_est - ours_cpu - external_cpu)
        over_reserved = max(0, ours_cpu + external_cpu - used_est)
        out.append({
            "node": name,
            "alive": node_state.get("alive"),
            "total_cpu": total,
            "logical_cpu": node_state.get("logical_cpu") or node_state.get("logical_cores"),
            "free_cpu": free,
            "used_cpu_est": used_est,
            "scheduleurm_cpu_reserved": ours_cpu,
            "scheduleurm_task_count": len(bucket.get("ours_tasks") or []),
            "scheduleurm_task_ids": [x.get("id") for x in (bucket.get("ours_tasks") or [])[:50]],
            "external_tracked_cpu_reserved": external_cpu,
            "external_tracked_task_count": len(bucket.get("external_tasks") or []),
            "external_tracked_task_ids": [x.get("id") for x in (bucket.get("external_tasks") or [])[:50]],
            "untracked_or_other_user_cpu_est": untracked_other,
            "over_reserved_cpu_est": over_reserved,
            "host_cpu_load_pct": node_state.get("host_cpu_load_pct"),
            "host_cpu_used_cores": node_state.get("host_cpu_used_cores"),
            "wsl_loadavg": node_state.get("wsl_loadavg"),
            "observed_free_cpu": node_state.get("observed_free_cpu"),
            "cpu_slot_reserved": node_state.get("cpu_slot_reserved"),
            "cpu_slot_overcommitted": node_state.get("cpu_slot_overcommitted"),
            "cpu_hard_reserved": node_state.get("cpu_hard_reserved"),
            "cpu_hard_free": node_state.get("cpu_hard_free"),
            "cpu_backfillable_reserved": node_state.get("cpu_backfillable_reserved"),
            "cpu_startup_hold_remaining_s": node_state.get(
                "cpu_startup_hold_remaining_s"
            ),
            "free_ram_mb": node_state.get("free_ram_mb"),
            "total_ram_mb": node_state.get("total_ram_mb"),
            "running_count": node_state.get("running_count"),
            "user_threads": node_state.get("user_threads"),
            "user_thread_limit": node_state.get("user_thread_limit"),
            "free_user_threads": node_state.get("free_user_threads"),
            "cgroup_pids_current": node_state.get("cgroup_pids_current"),
            "cgroup_pids_max": node_state.get("cgroup_pids_max"),
            "free_cgroup_pids": node_state.get("free_cgroup_pids"),
        })
    return out


def resource_accounting_payload(state: dict, nodes: list, *, deps: ResourceAccountingDeps) -> dict:
    return {
        "running": sum(1 for task in state.get("tasks", []) if task.get("status") == "running"),
        "launching": sum(1 for task in state.get("tasks", []) if task.get("status") == "launching"),
        "queued": sum(1 for task in state.get("tasks", []) if task.get("status") == "queued"),
        "nodes": cpu_ownership_snapshot(state, nodes, deps=deps),
    }


def dispatch_cycle_log_payload(
    state: dict,
    nodes: list,
    events: list,
    queued_count: int,
    *,
    deps: ResourceAccountingDeps,
) -> dict:
    event_counts = {}
    launched = []
    blocked = []
    no_fit = []
    for event in events or []:
        typ = event.get("type")
        event_counts[typ] = event_counts.get(typ, 0) + 1
        if typ == "launched":
            task = event.get("task") or {}
            launched.append({
                "task_id": event.get("task_id") or task.get("id"),
                "node": task.get("node"),
                "gpu_idx": task.get("gpu_idx"),
                "cpu_cores": task.get("cpu_cores"),
                "ram_mb": task.get("ram_mb"),
                "est_vram_mb": task.get("est_vram_mb"),
                "placement_algorithm": task.get("placement_algorithm"),
                "placement_algorithm_audit": task.get("placement_algorithm_audit"),
                "description": (task.get("description") or "")[:100],
            })
        elif typ == "blocked":
            blocked.append({"task_id": event.get("task_id"), "reason": event.get("reason")})
        elif typ == "no_fit":
            task = event.get("task") or {}
            no_fit.append({
                "task_id": event.get("task_id") or task.get("id"),
                "reason": task.get("last_block_reason") or "no fit",
                "cpu_cores": task.get("cpu_cores"),
                "ram_mb": task.get("ram_mb"),
                "est_vram_mb": task.get("est_vram_mb"),
            })
    payload = resource_accounting_payload(state, nodes, deps=deps)
    payload.update({
        "placement_algorithm": deps.algorithm_name(),
        "placement_algorithm_config": deps.algorithm_config_snapshot(),
        "queued_seen_by_dispatch": int(queued_count or 0),
        "event_counts": event_counts,
        "launched": launched[:100],
        "blocked": blocked[:100],
        "no_fit": no_fit[:100],
    })
    return payload


def reconcile_aggregate_only_vram(state: dict, nodes: list, *, deps: AggregateVramReconcileDeps) -> int:
    """Approximate task VRAM when aggregate GPU memory is known but per-PID VRAM is not."""
    changed = 0
    for node_state in nodes or []:
        if not node_state.get("alive"):
            continue
        node_name = node_state.get("name")
        for gpu in node_state.get("gpus") or []:
            gpu_idx = gpu.get("idx")
            try:
                aggregate_used = int(gpu.get("observed_used_mb", gpu.get("used_mb") or 0) or 0)
            except (TypeError, ValueError):
                continue
            tasks = [
                task for task in state.get("tasks", [])
                if task.get("status") == "running"
                and task.get("node") == node_name
                and task.get("gpu_idx") == gpu_idx
                and not deps.is_slurm_managed(task)
            ]
            if not tasks:
                continue
            baselines = []
            for task in tasks:
                snap = task.get("last_launch_pre_snapshot") or {}
                if snap.get("target_node") != node_name or snap.get("target_gpu_idx") != gpu_idx:
                    continue
                try:
                    baselines.append(int(((snap.get("gpu") or {}).get("used_mb")) or 0))
                except (TypeError, ValueError):
                    pass
            baseline = min(baselines) if baselines else 0
            known = 0
            unknown = []
            for task in tasks:
                cur = int(task.get("current_vram_mb") or 0)
                if cur > 0 and task.get("vram_estimation_source") != "aggregate_residual":
                    known += cur
                else:
                    unknown.append(task)
            if not unknown:
                continue
            residual = max(0, aggregate_used - baseline - known)
            if residual < 100:
                for task in unknown:
                    if task.get("vram_estimation_source") in ("aggregate_residual", "aggregate_observed_zero", None):
                        task["current_vram_mb"] = 0
                        task["peak_vram_mb"] = 0
                        task["vram_estimation_source"] = "aggregate_observed_zero"
                        task["vram_estimation_note"] = (
                            f"aggregate GPU memory on {node_name}:GPU{gpu_idx} has no residual for this task"
                        )
                        changed += 1
                continue
            weights = [
                max(1, int(task.get("est_vram_mb") or task.get("peak_vram_mb") or 1))
                for task in unknown
            ]
            total_weight = sum(weights) or len(unknown)
            remaining = residual
            for idx, (task, weight) in enumerate(zip(unknown, weights)):
                if idx == len(unknown) - 1:
                    share = remaining
                else:
                    share = int(round(residual * weight / total_weight))
                    remaining -= share
                if share < 100:
                    continue
                task["current_vram_mb"] = share
                task["peak_vram_mb"] = max(int(task.get("peak_vram_mb") or 0), share)
                task["vram_estimation_source"] = "aggregate_residual"
                task["vram_estimation_note"] = (
                    f"aggregate GPU memory attribution on {node_name}:GPU{gpu_idx}; "
                    "per-PID VRAM unavailable"
                )
                changed += 1
    return changed


def reserve_inflight_vram(state: dict, nodes: list, *, deps: InflightVramReservationDeps) -> None:
    """Reserve VRAM that active tasks may still need later in their run.

    An explicitly supplied ``est_vram_mb`` is a peak budget, not a snapshot of
    current allocator use. Keep that peak reserved after startup/progress so a
    low-memory early phase cannot admit a second task that will collide with a
    late peak. Tasks without an explicit estimate retain the smaller startup
    floor behavior.
    """
    by_node = {node["name"]: node for node in nodes}
    for task in state.get("tasks", []):
        if task.get("status") not in ("running", "launching"):
            continue
        if deps.is_slurm_managed(task):
            continue
        if task.get("gpu_idx") is None:
            continue
        if deps.hard_rule_bypassed(
            "startup_vram_reserve",
            task,
            node_info=deps.node_configs.get(task.get("node") or "", {}),
        ):
            continue
        node = by_node.get(task.get("node"))
        if not node or not node.get("alive"):
            continue
        est = int(task.get("est_vram_mb") or 0)
        peak = int(task.get("peak_vram_mb") or 0)
        current = max(0, int(task.get("current_vram_mb") or 0))
        if task.get("est_vram_mb_explicit") and est > 0:
            reserve = max(0, est - current)
        else:
            if peak >= 100:
                continue
            if task.get("last_progress_line") or int(task.get("runtime_current_unit") or 0) > 0:
                continue
            reserve = min(est, deps.startup_floor_mb) if est > 0 else deps.startup_floor_mb
        if reserve < 100:
            if reserve <= 0:
                continue
            reserve = 100
        for gpu in node["gpus"]:
            if gpu["idx"] != task["gpu_idx"]:
                continue
            gpu.setdefault("observed_used_mb", int(gpu.get("used_mb") or 0))
            gpu["used_mb"] = min(gpu["total_mb"], gpu["used_mb"] + reserve)
            gpu["free_mb"] = max(0, gpu["free_mb"] - reserve)
            gpu["reserved_vram_mb"] = (
                int(gpu.get("reserved_vram_mb") or 0) + reserve
            )
            break
