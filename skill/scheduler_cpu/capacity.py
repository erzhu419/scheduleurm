"""CPU capacity, platform limits, and slot accounting helpers."""

from __future__ import annotations

from collections import Counter
import math
import time
from typing import Callable, Optional


def ceil_div(a: int, b: int) -> int:
    a = int(a)
    b = max(1, int(b))
    return (a + b - 1) // b


def node_physical_cores(
    node: Optional[str],
    *,
    node_state: Optional[dict] = None,
    node_info: Optional[dict] = None,
    default_cpu_cores: int = 1,
) -> int:
    """Best estimate of schedulable physical cores for CPU-worker planning."""
    info = node_info or {}
    state = node_state or {}
    for key in ("physical_cores", "physical_cpu", "physical_cpus"):
        try:
            value = int(info.get(key) or state.get(key) or 0)
            if value > 0:
                return value
        except Exception:
            pass
    try:
        sched = int(info.get("cpu_cores") or state.get("total_cpu") or 0)
    except Exception:
        sched = 0
    try:
        logical = int(state.get("logical_cpu") or state.get("logical_cores") or state.get("cores") or 0)
    except Exception:
        logical = 0
    if sched > 0:
        return sched
    if logical > 1 and (info.get("windows_skip_ht_pair", True) or logical > 64):
        return max(1, logical // 2)
    return max(1, logical or default_cpu_cores)


def cpu_planning_cores_for_node(
    node: str,
    *,
    node_state: Optional[dict] = None,
    node_info: Optional[dict] = None,
    default_cpu_cores: int = 1,
) -> tuple[int, int]:
    """Return (available_physical_cores, total_physical_cores)."""
    total = node_physical_cores(
        node,
        node_state=node_state,
        node_info=node_info,
        default_cpu_cores=default_cpu_cores,
    )
    if isinstance(node_state, dict):
        if node_state.get("alive") is False:
            return 0, total
        if "free_cpu" in node_state:
            try:
                free = max(0, int(node_state.get("free_cpu") or 0))
                return min(total, free), total
            except Exception:
                return 0, total
    return total, total


def cpu_max_workers_per_process(
    node: Optional[str],
    *,
    node_info: Optional[dict] = None,
    windows_cpu_max_workers_per_process: int = 0,
) -> int:
    """Per-process worker cap for runtime/platform limits."""
    info = node_info or {}
    try:
        explicit = int(info.get("max_cpu_workers_per_process") or 0)
        if explicit > 0:
            return explicit
    except Exception:
        pass
    if str(info.get("os") or "").lower() == "windows":
        return max(1, int(windows_cpu_max_workers_per_process or 1))
    return 0


def declared_cpu_slot_floor(task: dict) -> int:
    """Minimum CPU slots a scheduler-owned batch task asked us to reserve."""
    floors = []
    for key in ("cpu_declared_cores", "cpu_auto_workers"):
        try:
            value = int((task or {}).get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            floors.append(value)
    plan = (task or {}).get("cpu_batch_plan")
    if isinstance(plan, dict):
        try:
            value = int(plan.get("workers") or 0)
        except Exception:
            value = 0
        if value > 0:
            floors.append(value)
    return max(floors or [0])


def reserved_cpu_slots_for_task(task: dict, *, default_cpu_cores: int) -> int:
    try:
        current = int((task or {}).get("cpu_cores") or default_cpu_cores)
    except Exception:
        current = default_cpu_cores
    return max(0, current, declared_cpu_slot_floor(task))


def cpu_startup_grace_s_for_task(
    task: dict,
    node_info: dict,
    *,
    reserved_cpu_slots: int,
) -> float:
    """Keep long ramp protection only for tasks that can create wide CPU load."""
    if int(reserved_cpu_slots or 0) <= 1:
        configured = node_info.get(
            "live_cpu_backfill_single_thread_startup_grace_s"
        )
        if configured is None:
            configured = min(
                10.0,
                float(node_info.get("live_cpu_backfill_startup_grace_s", 300.0)),
            )
        return max(0.0, float(configured))
    configured = node_info.get("live_cpu_backfill_startup_grace_s")
    return max(0.0, float(300.0 if configured is None else configured))


def apply_cpu_slot_accounting_to_nodes(
    state: dict,
    nodes: list,
    *,
    node_configs: dict,
    cpu_slot_accounting_node_names: list[str],
    reserved_cpu_slots_for_task: Callable[[dict], int],
    now_fn: Callable[[], float] = time.time,
) -> None:
    """Attach scheduling reservations without replacing live CPU telemetry."""
    cpu_slot_names = set(cpu_slot_accounting_node_names)
    if "local" in node_configs:
        cpu_slot_names.add("local")
    if not cpu_slot_names:
        return
    cpu_slot_reserved = Counter()
    cpu_hard_reserved = Counter()
    cpu_startup_declared = Counter()
    cpu_startup_observed = Counter()
    cpu_startup_hold_remaining_s = Counter()
    now = now_fn()
    for task in state.get("tasks", []):
        if task.get("status") not in ("running", "launching"):
            continue
        node = task.get("node") or task.get("assigned_node")
        if node not in cpu_slot_names:
            continue
        reserved = reserved_cpu_slots_for_task(task)
        cpu_slot_reserved[node] += reserved

        node_info = node_configs.get(node, {}) or {}
        startup_grace_s = cpu_startup_grace_s_for_task(
            task,
            node_info,
            reserved_cpu_slots=reserved,
        )
        started_at = task.get("started_at") or task.get("launching_started_at")
        try:
            age_s = max(0.0, now - float(started_at)) if started_at else None
        except Exception:
            age_s = None
        startup_reserved = (
            task.get("status") == "launching"
            or age_s is None
            or age_s < startup_grace_s
        )
        if startup_reserved:
            observed_cores = 0.0
            if task.get("status") == "running":
                try:
                    observed_cores = max(
                        0.0, min(float(reserved), float(task.get("current_pcpu") or 0) / 100.0)
                    )
                except Exception:
                    observed_cores = 0.0
            decay_factor = 1.0
            if task.get("status") == "running" and age_s is not None:
                if startup_grace_s <= 0:
                    decay_factor = 0.0
                else:
                    decay_factor = max(0.0, 1.0 - age_s / startup_grace_s)
            unobserved_reservation = int(math.ceil(
                max(0.0, float(reserved) - observed_cores) * decay_factor
            ))
            cpu_hard_reserved[node] += unobserved_reservation
            cpu_startup_declared[node] += reserved
            cpu_startup_observed[node] += observed_cores
            if unobserved_reservation > 0:
                if task.get("status") == "launching" or age_s is None:
                    remaining_s = startup_grace_s
                else:
                    remaining_s = max(0.0, startup_grace_s - age_s)
                cpu_startup_hold_remaining_s[node] = max(
                    float(cpu_startup_hold_remaining_s.get(node, 0.0)),
                    remaining_s,
                )
    for node_state in nodes:
        name = node_state.get("name")
        if name not in cpu_slot_names:
            continue
        total_cpu = int(node_configs.get(name, {}).get("cpu_cores") or node_state.get("total_cpu") or 0)
        if total_cpu <= 0:
            continue
        raw_reserved_cpu = int(cpu_slot_reserved.get(name, 0))
        raw_hard_reserved_cpu = int(cpu_hard_reserved.get(name, 0))
        reserved_cpu = min(total_cpu, raw_reserved_cpu)
        hard_reserved_cpu = min(total_cpu, raw_hard_reserved_cpu)
        observed_free_cpu = node_state.get("free_cpu")
        try:
            observed_free_cpu_int = max(0, int(observed_free_cpu))
        except Exception:
            observed_free_cpu_int = max(0, total_cpu - hard_reserved_cpu)
        node_state["observed_free_cpu"] = observed_free_cpu
        node_state["observed_loadavg"] = node_state.get("loadavg")
        node_state["cpu_slot_accounting"] = True
        node_state["cpu_slot_reserved"] = raw_reserved_cpu
        node_state["cpu_slot_overcommitted"] = max(0, raw_reserved_cpu - total_cpu)
        node_state["cpu_hard_reserved"] = raw_hard_reserved_cpu
        node_state["cpu_startup_declared_reserved"] = int(
            cpu_startup_declared.get(name, 0)
        )
        node_state["cpu_startup_observed_cores"] = round(
            float(cpu_startup_observed.get(name, 0.0)), 3
        )
        node_state["cpu_startup_hold_remaining_s"] = int(math.ceil(
            float(cpu_startup_hold_remaining_s.get(name, 0.0))
        ))
        node_state["cpu_hard_free"] = max(
            0, observed_free_cpu_int - hard_reserved_cpu
        )
        node_state["cpu_backfillable_reserved"] = max(
            0, raw_reserved_cpu - raw_hard_reserved_cpu
        )
        node_state["total_cpu"] = total_cpu
        if name != "local":
            slot_free_cpu = max(0, total_cpu - reserved_cpu)
            node_state["cpu_slot_free"] = slot_free_cpu
            # `free_cpu` remains the conservative placement budget for compatibility.
            # Live telemetry stays in observed_free_cpu/observed_loadavg/loadavg.
            node_state["free_cpu"] = slot_free_cpu
