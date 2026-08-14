"""Runtime-bound CPU planning wrappers for scheduler.py."""

from __future__ import annotations

import re
import sys
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_cpu_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _ceil_div(a: int, b: int) -> int:
        return _ns(namespace, "_ceil_div_impl")(a, b)

    def _cpu_labor_node_names() -> list:
        return [
            name for name, info in _ns(namespace, "NODES").items()
            if info.get("cpu_labor_node") and not info.get("retired")
        ]

    def _cpu_slot_accounting_node_names() -> list:
        return [
            name for name, info in _ns(namespace, "NODES").items()
            if (info.get("cpu_labor_node") or info.get("cpu_slot_accounting"))
            and not info.get("retired")
        ]

    def _declared_cpu_slot_floor(task: dict) -> int:
        return _ns(namespace, "_declared_cpu_slot_floor_impl")(task)

    def _reserved_cpu_slots_for_task(task: dict) -> int:
        return _ns(namespace, "_reserved_cpu_slots_for_task_impl")(
            task,
            default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        )

    def _apply_cpu_slot_accounting_to_nodes(state: dict, nodes: list) -> None:
        return _ns(namespace, "_apply_cpu_slot_accounting_to_nodes_impl")(
            state,
            nodes,
            node_configs=_ns(namespace, "NODES"),
            cpu_slot_accounting_node_names=_ns(namespace, "_cpu_slot_accounting_node_names")(),
            reserved_cpu_slots_for_task=_ns(namespace, "_reserved_cpu_slots_for_task"),
        )

    def _node_physical_cores(node: Optional[str], node_state: Optional[dict] = None) -> int:
        return _ns(namespace, "_node_physical_cores_impl")(
            node,
            node_state=node_state,
            node_info=_ns(namespace, "NODES").get(node or "", {}) if node else {},
            default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        )

    def _cpu_planning_cores_for_node(node: str, node_state: Optional[dict] = None) -> tuple:
        return _ns(namespace, "_cpu_planning_cores_for_node_impl")(
            node,
            node_state=node_state,
            node_info=_ns(namespace, "NODES").get(node or "", {}),
            default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        )

    def _cpu_worker_plan_for_items(total_items: int, physical_cores: int) -> dict:
        return _ns(namespace, "_cpu_worker_plan_for_items_impl")(total_items, physical_cores)

    def _cpu_max_workers_per_process(node: Optional[str]) -> int:
        return _ns(namespace, "_cpu_max_workers_per_process_impl")(
            node,
            node_info=_ns(namespace, "NODES").get(node or "", {}) if node else {},
            windows_cpu_max_workers_per_process=_ns(
                namespace, "WINDOWS_CPU_MAX_WORKERS_PER_PROCESS"
            ),
        )

    def _cpu_batch_plan(
        total_items: int,
        node_names: Optional[list] = None,
        node_states: Optional[dict] = None,
    ) -> list:
        names = list(node_names or _ns(namespace, "_cpu_labor_node_names")())
        if not names:
            names = ["local"] if "local" in _ns(namespace, "NODES") else list(_ns(namespace, "NODES").keys())[:1]
        return _ns(namespace, "_cpu_batch_plan_impl")(
            total_items,
            names,
            node_states=node_states,
            node_info_by_name=_ns(namespace, "NODES"),
            default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
            windows_cpu_max_workers_per_process=_ns(
                namespace, "WINDOWS_CPU_MAX_WORKERS_PER_PROCESS"
            ),
        )

    def _cpu_batch_item_counts(args) -> tuple:
        try:
            logical_items = int(getattr(args, "items", 0) or 0)
            item_multiplier = int(getattr(args, "item_multiplier", 1) or 1)
        except Exception:
            sys.exit("--items and --item-multiplier must be integers")
        if logical_items < 0:
            sys.exit("--items must be >= 0")
        if item_multiplier < 1:
            sys.exit("--item-multiplier must be >= 1")
        return logical_items, item_multiplier, logical_items * item_multiplier

    def _cpu_wave_summary(items: int, workers: int) -> dict:
        return _ns(namespace, "_cpu_wave_summary_impl")(items, workers)

    def _cpu_batch_log_payload(
        total_items: int,
        plan: list,
        node_states: Optional[dict] = None,
        use_total_cores: bool = False,
        templates: Optional[dict] = None,
        logical_items: Optional[int] = None,
        item_multiplier: int = 1,
    ) -> dict:
        return _ns(namespace, "_cpu_batch_log_payload_impl")(
            total_items,
            plan,
            node_states=node_states,
            use_total_cores=use_total_cores,
            templates=templates,
            logical_items=logical_items,
            item_multiplier=item_multiplier,
        )

    def _cpu_parallel_template_values(
        plan: dict,
        total_items: Optional[int] = None,
        logical_items: Optional[int] = None,
        item_multiplier: int = 1,
    ) -> dict:
        return _ns(namespace, "_cpu_parallel_template_values_impl")(
            plan,
            total_items=total_items,
            logical_items=logical_items,
            item_multiplier=item_multiplier,
        )

    def _format_cpu_parallel_template(
        text: Optional[str],
        plan: dict,
        total_items: Optional[int] = None,
        logical_items: Optional[int] = None,
        item_multiplier: int = 1,
    ) -> Optional[str]:
        return _ns(namespace, "_format_cpu_parallel_template_impl")(
            text,
            plan,
            total_items=total_items,
            logical_items=logical_items,
            item_multiplier=item_multiplier,
        )

    def _rewrite_cpu_parallel_cmd(
        cmd: str,
        plan: dict,
        total_items: Optional[int] = None,
        logical_items: Optional[int] = None,
        item_multiplier: int = 1,
    ) -> str:
        return _ns(namespace, "_rewrite_cpu_parallel_cmd_impl")(
            cmd,
            plan,
            total_items=total_items,
            logical_items=logical_items,
            item_multiplier=item_multiplier,
        )

    def _cpu_parallel_env(task: dict) -> dict:
        return _ns(namespace, "_cpu_parallel_env_impl")(task)

    def _apply_cpu_parallel_plan_to_task(
        task: dict,
        node_state: Optional[dict] = None,
    ) -> Optional[dict]:
        return _ns(namespace, "_apply_cpu_parallel_plan_to_task_impl")(
            task,
            node_state=node_state,
            node_info=_ns(namespace, "NODES").get(task.get("node") or "", {}),
            default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        )

    return {
        "_CPU_WORKER_FLAG_RE": re.compile(
            r"(?P<flag>--(?:workers|n-workers|n_workers|num-workers|num_workers|jobs|n-jobs|n_jobs|num-jobs|num_jobs))"
            r"(?P<sep>=|\s+)"
            r"(?P<value>auto|AUTO|\{workers\}|\{n_workers\}|\{num_workers\})"
        ),
        "_ceil_div": _ceil_div,
        "_cpu_labor_node_names": _cpu_labor_node_names,
        "_cpu_slot_accounting_node_names": _cpu_slot_accounting_node_names,
        "_declared_cpu_slot_floor": _declared_cpu_slot_floor,
        "_reserved_cpu_slots_for_task": _reserved_cpu_slots_for_task,
        "_apply_cpu_slot_accounting_to_nodes": _apply_cpu_slot_accounting_to_nodes,
        "_node_physical_cores": _node_physical_cores,
        "_cpu_planning_cores_for_node": _cpu_planning_cores_for_node,
        "_cpu_worker_plan_for_items": _cpu_worker_plan_for_items,
        "_cpu_max_workers_per_process": _cpu_max_workers_per_process,
        "_cpu_batch_plan": _cpu_batch_plan,
        "_cpu_batch_item_counts": _cpu_batch_item_counts,
        "_cpu_wave_summary": _cpu_wave_summary,
        "_cpu_batch_log_payload": _cpu_batch_log_payload,
        "_cpu_parallel_template_values": _cpu_parallel_template_values,
        "_format_cpu_parallel_template": _format_cpu_parallel_template,
        "_rewrite_cpu_parallel_cmd": _rewrite_cpu_parallel_cmd,
        "_cpu_parallel_env": _cpu_parallel_env,
        "_apply_cpu_parallel_plan_to_task": _apply_cpu_parallel_plan_to_task,
    }
