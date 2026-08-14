"""Status command data-shaping helpers and CLI implementation."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable


STATUS_COMPACT_KEYS = (
    "id", "status", "description", "project", "signature", "node", "gpu_idx",
    "remote_pids", "log_path", "submitted_at", "started_at", "finished_at",
    "slurm_job_id", "slurm_state",
    "est_vram_mb", "current_vram_mb", "peak_vram_mb", "ram_mb",
    "current_ram_mb", "peak_ram_mb", "cpu_cores", "current_pcpu",
    "eta_seconds", "eta_source", "eta_confidence", "runtime_current_unit",
    "runtime_total_units", "progress_ratio", "last_progress_line",
    "last_block_reason", "last_eviction_kind", "last_resource_eviction",
    "last_kill_action", "last_kill_reason", "launch_error",
    "placement_algorithm", "placement_algorithm_config",
    "placement_algorithm_audit", "require_node", "require_gpu_idx",
    "allow_gpu_over_one_third", "node_probe_state", "probe_unknown_since",
    "last_probe_unknown_at", "last_probe_unknown_duration_s",
    "last_probe_unknown_count", "last_probe_unknown_reason",
    "last_reconnected_at", "last_status_sync_at", "last_status_sync_status",
    "last_status_sync_reason",
)

ACTIVE_STATUS_ROWS = frozenset({"queued", "launching", "running"})


@dataclass(frozen=True)
class StatusCommandDeps:
    status_readonly_lock_timeout_s: float
    lock_timeout_error: type[Exception]
    running_probe_snapshot_outside_lock: Callable[[str], tuple[Any, Any]]
    eta_tail_snapshot_outside_lock: Callable[[str], tuple[Any, Any]]
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    recover_stale_launching_tasks: Callable[[dict], Any]
    update_running_tasks: Callable[..., Any]
    seed_pending_eta_from_history: Callable[[dict], Any]
    reconcile_requeue_lineage_invariants: Callable[[dict], Any]
    save_state: Callable[[dict], Any]
    compute_node_load_seconds: Callable[[dict], dict]
    probe_all: Callable[[], list]
    apply_cpu_slot_accounting_to_nodes: Callable[[dict, list], Any]
    iter_node_summary_nodes: Callable[[list], list]
    node_display_name: Callable[[str], str]
    format_mem_gb: Callable[[int], str]
    node_configs: dict
    format_node_claim_summary: Callable[[dict], str]
    format_node_ram_summary: Callable[[dict], str]
    format_task_location: Callable[[dict], str]
    format_task_vram_usage: Callable[[dict], str]
    format_task_ram_usage: Callable[[dict], str]
    format_task_eta: Callable[[dict], str]
    format_task_owner: Callable[[dict], str]


def parse_status_filter_ids(raw_ids) -> set[str]:
    filter_ids: set[str] = set()
    for raw in raw_ids or []:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                filter_ids.add(part)
    return filter_ids


def compact_status_task(task: dict) -> dict:
    return {key: task.get(key) for key in STATUS_COMPACT_KEYS if key in task}


def filter_status_tasks(tasks: list[dict], filter_ids: set[str]) -> list[dict]:
    if not filter_ids:
        return list(tasks)
    return [task for task in tasks if str(task.get("id")) in filter_ids]


def status_rows_for_display(tasks: list[dict], *, show_done: bool) -> list[dict]:
    if show_done:
        return list(tasks)
    return [task for task in tasks if task["status"] in ACTIVE_STATUS_ROWS]


def status_json_payload(tasks: list[dict], *, brief: bool) -> dict:
    tasks_out = [compact_status_task(task) for task in tasks] if brief else list(tasks)
    return {"tasks": tasks_out}


def format_eta_load_seconds(seconds: float | int | None) -> str:
    etaload = float(seconds or 0)
    if etaload <= 0:
        return ""
    if etaload < 3600:
        return f"  eta_load={etaload / 60:.0f}m"
    if etaload < 86400:
        return f"  eta_load={etaload / 3600:.1f}h"
    return f"  eta_load={etaload / 86400:.1f}d"


def format_task_runtime_minutes(task: dict, *, now_fn=time.time) -> str:
    if not task.get("started_at"):
        return ""
    end = task.get("finished_at") or now_fn()
    return f" {(end - task['started_at']) / 60:.1f}m"


def format_measurement_reservation(node: dict) -> str:
    if not node.get("measurement_reservation_active"):
        return ""
    reservation = node.get("measurement_reservation") or {}
    purpose = str(reservation.get("purpose") or "calibration")
    return f"  MEASUREMENT-RESERVED({purpose})"


def cmd_status(args, *, deps: StatusCommandDeps) -> None:
    filter_ids = parse_status_filter_ids(getattr(args, "ids", None))
    readonly = bool(getattr(args, "readonly", False))
    lock_timeout = getattr(args, "lock_timeout", None)
    if readonly and lock_timeout is None:
        lock_timeout = deps.status_readonly_lock_timeout_s
    if not readonly:
        running_probe_results, running_probe_ids = deps.running_probe_snapshot_outside_lock(
            "status:running-probe"
        )
        eta_tail_outputs, eta_tail_ids = deps.eta_tail_snapshot_outside_lock("status:eta-tail")
    else:
        running_probe_results = running_probe_ids = None
        eta_tail_outputs = eta_tail_ids = None
    try:
        with deps.state_lock(
            timeout_s=lock_timeout,
            shared=readonly,
            purpose="status:readonly" if readonly else "status:update",
        ):
            state = deps.load_state()
            if not readonly:
                deps.recover_stale_launching_tasks(state)
                deps.update_running_tasks(
                    state,
                    probe_results=running_probe_results,
                    probed_task_ids=running_probe_ids,
                    eta_tail_outputs=eta_tail_outputs,
                    eta_probed_task_ids=eta_tail_ids,
                )
                deps.seed_pending_eta_from_history(state)
                deps.reconcile_requeue_lineage_invariants(state)
                deps.save_state(state)
    except deps.lock_timeout_error as exc:
        sys.exit(str(exc) + "; try `status --readonly` or wait for dispatch/cancel to finish")

    filtered_tasks = filter_status_tasks(state["tasks"], filter_ids)
    if args.json:
        print(json.dumps(status_json_payload(
            filtered_tasks,
            brief=bool(getattr(args, "brief", False)),
        ), indent=2))
        return

    print("=== nodes ===")
    node_loads = deps.compute_node_load_seconds(state)
    if readonly:
        nodes = []
        print("  (readonly: node telemetry/probe skipped)")
    else:
        nodes = deps.probe_all()
        deps.apply_cpu_slot_accounting_to_nodes(state, nodes)

    for node in deps.iter_node_summary_nodes(nodes):
        display_name = deps.node_display_name(str(node.get("name") or "?"))
        if not node["alive"]:
            print(f"  {display_name:11s} DOWN ({node.get('error','?')})")
            continue
        gpu_parts = []
        for gpu in node["gpus"]:
            mem_pct = int(round(100 * gpu["used_mb"] / max(gpu["total_mb"], 1)))
            gpu_parts.append(
                f"GPU{gpu['idx']}={deps.format_mem_gb(gpu['used_mb'])}/"
                f"{deps.format_mem_gb(gpu['total_mb'])}"
                f"(free={deps.format_mem_gb(gpu.get('free_mb', 0))}, "
                f"mem:{mem_pct}%, util:{gpu['util_pct']}%)"
            )
        gpu_str = ", ".join(gpu_parts)
        load = node.get("loadavg", 0)
        reserve = int(deps.node_configs.get(node.get("name"), {}).get("reserved_cpu_cores") or 0)
        reserve_str = f", reserve {reserve}" if reserve else ""
        cpu_str = (
            f"cpu={node.get('free_cpu', '?')}/{node.get('total_cpu', '?')}"
            f"(load {load:.1f}{reserve_str})"
        )
        etaload_str = format_eta_load_seconds(node_loads.get(node["name"], 0))
        claim_str = deps.format_node_claim_summary(node)
        ram_str = deps.format_node_ram_summary(node)
        reservation_str = format_measurement_reservation(node)
        print(
            f"  {display_name:11s} {gpu_str}  {cpu_str}  {ram_str}"
            f"{reservation_str}{etaload_str}{claim_str}"
        )

    print("\n=== tasks ===")
    rows = status_rows_for_display(filtered_tasks, show_done=args.all)
    if not rows:
        print("  (no active tasks; pass --all to see history)")
    for task in rows:
        loc = deps.format_task_location(task)
        peak = deps.format_task_vram_usage(task)
        pram = deps.format_task_ram_usage(task)
        runtime = format_task_runtime_minutes(task)
        eta = deps.format_task_eta(task)
        eta = f" {eta:15s}" if eta else " " * 16
        proj = (task.get("project") or "?")[:14]
        owner = deps.format_task_owner(task)[:18]
        print(
            f"  [{task['id']}] {task['status']:9s} {loc:20s} {proj:14s} "
            f"{owner:18s} {peak:14s} {pram:13s}{runtime}{eta} "
            f"{task['description'][:55]}"
        )
