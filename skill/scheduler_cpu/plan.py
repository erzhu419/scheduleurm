"""CPU batch planning helpers for scheduleurm."""
from __future__ import annotations

import re
from typing import Optional

from .batch_plan import (
    cpu_batch_log_payload,
    cpu_batch_plan,
    cpu_wave_summary,
    cpu_worker_plan_for_items,
)
from .capacity import (
    apply_cpu_slot_accounting_to_nodes,
    ceil_div,
    cpu_max_workers_per_process,
    cpu_planning_cores_for_node,
    declared_cpu_slot_floor,
    node_physical_cores,
    reserved_cpu_slots_for_task,
)



def cpu_parallel_template_values(
    plan: dict,
    *,
    total_items: Optional[int] = None,
    logical_items: Optional[int] = None,
    item_multiplier: int = 1,
) -> dict[str, str]:
    total = total_items if total_items is not None else plan.get("items", 0)
    logical = logical_items if logical_items is not None else total
    multiplier = item_multiplier or 1
    values = {
        "node": plan.get("node", ""),
        "start": plan.get("start", 0),
        "end": plan.get("end", plan.get("items", 0)),
        "items": plan.get("items", 0),
        "shard_items": plan.get("items", 0),
        "total_items": total,
        "total_work_items": total,
        "logical_items": logical,
        "base_items": logical,
        "item_multiplier": multiplier,
        "items_per_unit": multiplier,
        "episodes_per_item": multiplier,
        "workers": plan.get("workers", 0),
        "n_workers": plan.get("workers", 0),
        "num_workers": plan.get("workers", 0),
        "waves": plan.get("waves", 0),
        "rounds": plan.get("waves", 0),
        "physical_cores": plan.get("physical_cores", 0),
        "available_cores": plan.get("available_cores", plan.get("physical_cores", 0)),
        "total_physical_cores": plan.get("total_physical_cores", plan.get("physical_cores", 0)),
        "last_wave_items": plan.get("last_wave_items", 0),
        "shard_index": plan.get("shard_index", 0),
        "shard_id": plan.get("shard_index", 0),
        "num_shards": plan.get("num_shards", 1),
    }
    return {key: str(value) for key, value in values.items()}


def format_cpu_parallel_template(
    text: Optional[str],
    plan: dict,
    *,
    total_items: Optional[int] = None,
    logical_items: Optional[int] = None,
    item_multiplier: int = 1,
) -> Optional[str]:
    if text is None:
        return None
    out = str(text)
    for key, value in cpu_parallel_template_values(
        plan,
        total_items=total_items,
        logical_items=logical_items,
        item_multiplier=item_multiplier,
    ).items():
        out = out.replace("{" + key + "}", value)
    return out


CPU_WORKER_FLAG_RE = re.compile(
    r"(?P<flag>--(?:workers|n-workers|n_workers|num-workers|num_workers|jobs|n-jobs|n_jobs|num-jobs|num_jobs))"
    r"(?P<sep>=|\s+)"
    r"(?P<value>auto|AUTO|\{workers\}|\{n_workers\}|\{num_workers\})"
)




def rewrite_cpu_parallel_cmd(
    cmd: str,
    plan: dict,
    *,
    total_items: Optional[int] = None,
    logical_items: Optional[int] = None,
    item_multiplier: int = 1,
) -> str:
    out = format_cpu_parallel_template(
        cmd or "",
        plan,
        total_items=total_items,
        logical_items=logical_items,
        item_multiplier=item_multiplier,
    ) or ""
    workers = str(plan.get("workers", 0))

    def replace(match):
        return f"{match.group('flag')}{match.group('sep')}{workers}"

    return CPU_WORKER_FLAG_RE.sub(replace, out)


def cpu_parallel_env(task: dict) -> dict[str, str]:
    keys = {
        "SCHEDULEURM_CPU_TOTAL_ITEMS": task.get("cpu_parallel_total_items")
        or task.get("cpu_parallel_items"),
        "SCHEDULEURM_CPU_TOTAL_WORK_ITEMS": task.get("cpu_parallel_total_items")
        or task.get("cpu_parallel_items"),
        "SCHEDULEURM_CPU_LOGICAL_ITEMS": task.get("cpu_parallel_logical_items"),
        "SCHEDULEURM_CPU_ITEM_MULTIPLIER": task.get("cpu_parallel_item_multiplier"),
        "SCHEDULEURM_CPU_ITEMS_PER_UNIT": task.get("cpu_parallel_item_multiplier"),
        "SCHEDULEURM_CPU_EPISODES_PER_ITEM": task.get("cpu_parallel_item_multiplier"),
        "SCHEDULEURM_CPU_SHARD_START": task.get("cpu_parallel_start"),
        "SCHEDULEURM_CPU_SHARD_END": task.get("cpu_parallel_end"),
        "SCHEDULEURM_CPU_SHARD_ITEMS": task.get("cpu_parallel_items"),
        "SCHEDULEURM_CPU_WORKERS": task.get("cpu_auto_workers"),
        "SCHEDULEURM_CPU_WAVES": task.get("cpu_parallel_waves"),
        "SCHEDULEURM_CPU_PHYSICAL_CORES": task.get("cpu_parallel_physical_cores"),
        "SCHEDULEURM_CPU_TOTAL_PHYSICAL_CORES": task.get("cpu_parallel_total_physical_cores"),
        "SCHEDULEURM_CPU_LAST_WAVE_ITEMS": task.get("cpu_parallel_last_wave_items"),
        "SCHEDULEURM_CPU_SHARD_INDEX": task.get("cpu_parallel_shard_index"),
        "SCHEDULEURM_CPU_NUM_SHARDS": task.get("cpu_parallel_num_shards"),
    }
    return {key: str(value) for key, value in keys.items() if value is not None and value != ""}


def apply_cpu_parallel_plan_to_task(
    task: dict,
    *,
    node_state: Optional[dict] = None,
    node_info: Optional[dict] = None,
    default_cpu_cores: int = 1,
) -> Optional[dict]:
    items = int(task.get("cpu_parallel_items") or 0)
    if items <= 0:
        return None
    batch_plan = task.get("cpu_batch_plan") if isinstance(task.get("cpu_batch_plan"), dict) else {}
    if batch_plan:
        physical = int(
            batch_plan.get("physical_cores")
            or batch_plan.get("available_cores")
            or node_physical_cores(
                task.get("node"),
                node_state=node_state,
                node_info=node_info,
                default_cpu_cores=default_cpu_cores,
            )
        )
        plan = cpu_worker_plan_for_items(items, physical)
        workers = int(batch_plan.get("workers") or task.get("cpu_auto_workers") or plan["workers"])
        waves = int(batch_plan.get("waves") or ceil_div(items, max(1, workers)))
        last = int(batch_plan.get("last_wave_items") or (items - workers * max(0, waves - 1)))
        if last <= 0:
            last = workers
        plan.update({
            "workers": workers,
            "waves": waves,
            "last_wave_items": last,
            "physical_cores": physical,
            "total_physical_cores": int(batch_plan.get("total_physical_cores") or physical),
        })
    else:
        physical = node_physical_cores(
            task.get("node"),
            node_state=node_state,
            node_info=node_info,
            default_cpu_cores=default_cpu_cores,
        )
        plan = cpu_worker_plan_for_items(items, physical)
    plan.update({
        "node": task.get("node") or "",
        "start": int(task.get("cpu_parallel_start") or 0),
        "end": int(task.get("cpu_parallel_end") or items),
        "shard_index": int(task.get("cpu_parallel_shard_index") or 0),
        "num_shards": int(task.get("cpu_parallel_num_shards") or 1),
    })
    task["cpu_auto_workers"] = int(plan["workers"])
    task["cpu_parallel_waves"] = int(plan["waves"])
    task["cpu_parallel_physical_cores"] = int(plan["physical_cores"])
    task["cpu_parallel_total_physical_cores"] = int(
        plan.get("total_physical_cores") or plan["physical_cores"]
    )
    task["cpu_parallel_last_wave_items"] = int(plan["last_wave_items"])
    if int(task.get("cpu_cores") or 0) < int(plan["workers"]):
        task["cpu_cores"] = int(plan["workers"])
    old_cmd = task.get("cmd") or ""
    new_cmd = rewrite_cpu_parallel_cmd(
        old_cmd,
        plan,
        total_items=task.get("cpu_parallel_total_items") or items,
        logical_items=task.get("cpu_parallel_logical_items"),
        item_multiplier=int(task.get("cpu_parallel_item_multiplier") or 1),
    )
    if new_cmd != old_cmd:
        task["cmd"] = new_cmd
        task["cpu_auto_worker_cmd_rewritten"] = True
    return plan


__all__ = [
    "CPU_WORKER_FLAG_RE",
    "apply_cpu_parallel_plan_to_task",
    "apply_cpu_slot_accounting_to_nodes",
    "ceil_div",
    "cpu_batch_log_payload",
    "cpu_batch_plan",
    "cpu_max_workers_per_process",
    "cpu_parallel_env",
    "cpu_parallel_template_values",
    "cpu_planning_cores_for_node",
    "cpu_wave_summary",
    "cpu_worker_plan_for_items",
    "declared_cpu_slot_floor",
    "format_cpu_parallel_template",
    "node_physical_cores",
    "reserved_cpu_slots_for_task",
    "rewrite_cpu_parallel_cmd",
]
