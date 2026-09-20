"""Core dispatch bookkeeping helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DispatchAccountingDeps:
    counts_against_node_concurrency: Callable[[dict], bool]
    apply_cpu_slot_accounting_to_nodes: Callable[[dict, list], None]


@dataclass(frozen=True)
class DispatchQueuePlan:
    running_keys: set
    queued: list
    global_batch_plan: dict
    global_batch_event: dict | None
    identity_index: dict


def build_dispatch_queue_plan(
    state: dict,
    nodes: list,
    target_ids: set[str],
    priority_order: dict,
    *,
    task_run_identity: Callable[[dict], object],
    algorithm_global_batch_plan: Callable[[list, list], tuple[dict, dict | None]],
) -> DispatchQueuePlan:
    """Build the ordered dispatch queue plus identity/global-batch guards."""
    running_keys = set()
    tasks_by_id = {}
    terminal_by_run_key = {}
    for task in state["tasks"]:
        task_id = task.get("id")
        if task_id:
            tasks_by_id[task_id] = task
        status = task.get("status")
        if status == "queued":
            continue
        run_key = task_run_identity(task)
        if not run_key:
            continue
        if status in ("running", "launching"):
            running_keys.add(run_key)
        else:
            terminal_by_run_key.setdefault(run_key, []).append(task)
    queued = sorted(
        [t for t in state["tasks"] if t["status"] == "queued"],
        key=lambda t: (priority_order.get(t["priority"], 1), t["submitted_at"]),
    )
    if target_ids:
        queued = [t for t in queued if str(t.get("id") or "") in target_ids]
    global_batch_plan, global_batch_event = algorithm_global_batch_plan(queued, nodes)
    return DispatchQueuePlan(
        running_keys=running_keys,
        queued=queued,
        global_batch_plan=global_batch_plan,
        global_batch_event=global_batch_event,
        identity_index={
            "tasks_by_id": tasks_by_id,
            "terminal_by_run_key": terminal_by_run_key,
        },
    )


def prepare_dispatch_node_accounting(
    state: dict,
    nodes: list,
    preempted: list,
    *,
    deps: DispatchAccountingDeps,
) -> list:
    """Seed mutable node counters used by placement during one dispatch pass."""
    running_per_node = Counter(
        t.get("node") for t in state["tasks"]
        if deps.counts_against_node_concurrency(t)
    )
    running_per_gpu = Counter(
        (t.get("node"), t.get("gpu_idx")) for t in state["tasks"]
        if deps.counts_against_node_concurrency(t) and t.get("gpu_idx") is not None
    )
    deps.apply_cpu_slot_accounting_to_nodes(state, nodes)
    for n in nodes:
        n["running_count"] = running_per_node.get(n["name"], 0)
        for g in n.get("gpus") or []:
            g["running_task_count"] = running_per_gpu.get((n["name"], g.get("idx")), 0)

    events = []
    for ev in preempted:
        for n in nodes:
            if n["name"] == ev["node"]:
                n["free_cpu"] = n.get("free_cpu", 0) + ev["cpu_freed"]
                n["free_ram_mb"] = n.get("free_ram_mb", 0) + ev["ram_freed"]
                n["running_count"] = max(0, n.get("running_count", 0) - 1)
                break
        events.append({
            "type": "preempted",
            "task_id": ev["id"],
            "freed_node": ev["node"],
            "cpu_freed": ev["cpu_freed"],
            "ram_freed": ev["ram_freed"],
            "target_id": ev.get("target_id"),
            "cpu_deficit": ev.get("cpu_deficit"),
            "ram_deficit": ev.get("ram_deficit"),
            "protected_skipped": ev.get("protected_skipped") or [],
        })
    return events
