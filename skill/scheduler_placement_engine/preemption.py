from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class PreemptionDeps:
    node_configs: dict
    queue_wait_min: int
    victim_min_age_min: int
    victim_max_age_min: int
    max_victims_per_dispatch: int
    default_cpu_cores: int
    default_ram_mb: int
    node_ram_headroom_mb: Callable[[dict, dict], int]
    ignore_cpu_for_server_gpu_task: Callable[..., bool]
    is_slurm_managed: Callable[[dict], bool]
    task_evict_loss_protected: Callable[[dict], bool]
    task_ram_pressure_mb: Callable[[dict], int]
    task_progress_ratio: Callable[[dict], Optional[float]]
    evict_to_queue: Callable[[dict, dict, str, str], None]
    format_mem_gb: Callable[[int], str]
    now: Callable[[], float]


def _queued_high_priority_tasks(state: dict, deps: PreemptionDeps) -> list[dict]:
    now = deps.now()
    queued_high = [
        task for task in state.get("tasks", [])
        if task.get("status") == "queued"
        and task.get("priority") == "high"
        and task.get("submitted_at")
        and (now - task["submitted_at"]) > deps.queue_wait_min * 60
    ]
    queued_high.sort(key=lambda task: task.get("submitted_at", 0))
    return queued_high


def _eligible_victims(state: dict, node_name: str, deps: PreemptionDeps) -> list[dict]:
    now = deps.now()
    return [
        task for task in state.get("tasks", [])
        if task.get("status") == "running"
        and task.get("node") == node_name
        and task.get("priority") == "normal"
        and not task.get("auto_adopted")
        and not deps.is_slurm_managed(task)
        and task.get("started_at")
        and (now - task["started_at"]) > deps.victim_min_age_min * 60
        and (now - task["started_at"]) < deps.victim_max_age_min * 60
    ]


def _pick_victim(
    victims: list[dict],
    remaining_cpu: int,
    remaining_ram: int,
    deps: PreemptionDeps,
) -> tuple[dict, set[int]]:
    protected_ids = {id(task) for task in victims if deps.task_evict_loss_protected(task)}
    pool = [task for task in victims if id(task) not in protected_ids] or victims

    def key(task: dict):
        cpu = int(task.get("cpu_cores") or deps.default_cpu_cores)
        ram = deps.task_ram_pressure_mb(task)
        progress = deps.task_progress_ratio(task)
        progress_rank = -(progress if progress is not None else 0.0)
        started = float(task.get("started_at") or 0)
        if remaining_ram > 0:
            return (
                min(ram, remaining_ram),
                min(cpu, remaining_cpu) if remaining_cpu > 0 else 0,
                ram,
                progress_rank,
                started,
            )
        return (
            min(cpu, remaining_cpu),
            progress_rank,
            -ram,
            started,
        )

    return max(pool, key=key), protected_ids


def _node_resource_deficit(hi_task: dict, nodes: list, node_name: str, deps: PreemptionDeps) -> tuple[int, int]:
    cpu_need = hi_task.get("cpu_cores") or deps.default_cpu_cores
    ram_need = hi_task.get("ram_mb") or deps.default_ram_mb
    node_state = next((node for node in nodes if node.get("name") == node_name), None)
    node_info = deps.node_configs.get(node_name, {})
    free_cpu = int((node_state or {}).get("free_cpu") or 0)
    free_ram = int((node_state or {}).get("free_ram_mb") or 0)
    headroom = deps.node_ram_headroom_mb(node_state or {}, node_info) if node_info else 0
    schedulable_ram = max(0, free_ram - headroom)
    ignore_cpu = deps.ignore_cpu_for_server_gpu_task(
        hi_task,
        node_state=node_state,
        node_info=node_info,
        node_name=node_name,
        gpu_idx=(
            hi_task.get("require_gpu_idx")
            if hi_task.get("require_gpu_idx") is not None
            else hi_task.get("gpu_idx")
        ),
    )
    cpu_deficit = 0 if ignore_cpu else max(0, int(cpu_need) - free_cpu)
    ram_deficit = max(0, int(ram_need) - schedulable_ram)
    return cpu_deficit, ram_deficit


def preempt_for_high_priority(state: dict, nodes: list, *, deps: PreemptionDeps) -> list[dict]:
    """Evict eligible normal-priority tasks so long-waiting high-priority tasks can fit."""
    now = deps.now()
    evicted = []
    nodes_done = set()
    reserved_nodes = {
        str(node.get("name"))
        for node in nodes
        if node.get("measurement_reservation_active")
    }
    for hi_task in _queued_high_priority_tasks(state, deps):
        if len(evicted) >= deps.max_victims_per_dispatch:
            break
        node_name = hi_task.get("require_node") or hi_task.get("preferred_node")
        if not node_name or node_name in nodes_done or node_name in reserved_nodes:
            continue
        cpu_deficit, ram_deficit = _node_resource_deficit(hi_task, nodes, node_name, deps)
        if cpu_deficit <= 0 and ram_deficit <= 0:
            continue
        cpu_acc = 0
        ram_acc = 0
        wait_min = int((now - hi_task["submitted_at"]) / 60)
        for _ in range(deps.max_victims_per_dispatch - len(evicted)):
            remaining_cpu = max(0, cpu_deficit - cpu_acc)
            remaining_ram = max(0, ram_deficit - ram_acc)
            if remaining_cpu <= 0 and remaining_ram <= 0:
                break
            victims = _eligible_victims(state, node_name, deps)
            if not victims:
                break
            victim, protected_ids = _pick_victim(victims, remaining_cpu, remaining_ram, deps)
            cpu_freed = victim.get("cpu_cores") or deps.default_cpu_cores
            ram_freed = deps.task_ram_pressure_mb(victim) or deps.default_ram_mb
            deps.evict_to_queue(
                victim,
                state,
                f"preempted by {hi_task['id']} (high-prio waited {wait_min}min; "
                f"deficit cpu={cpu_deficit}, ram={deps.format_mem_gb(ram_deficit)})",
                "preempt_high_priority",
            )
            evicted.append({
                "id": victim["id"],
                "node": node_name,
                "cpu_freed": cpu_freed,
                "ram_freed": ram_freed,
                "target_id": hi_task.get("id"),
                "cpu_deficit": cpu_deficit,
                "ram_deficit": ram_deficit,
                "protected_skipped": [
                    task.get("id") for task in victims if id(task) in protected_ids
                ],
            })
            cpu_acc += cpu_freed
            ram_acc += ram_freed
        nodes_done.add(node_name)
    return evicted
