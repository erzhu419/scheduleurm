from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_placement_gates_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _task_is_gpu_capacity_task(task: dict) -> bool:
        return _ns(namespace, "_task_is_gpu_capacity_task_impl")(
            task,
            default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
        )

    def _flag_enabled(value, default: bool) -> bool:
        return _ns(namespace, "_flag_enabled_impl")(value, default)

    def _ignore_cpu_for_server_gpu_task(
        task: dict,
        node_state: Optional[dict] = None,
        node_info: Optional[dict] = None,
        node_name: Optional[str] = None,
        gpu_idx: Optional[int] = None,
    ) -> bool:
        return _ns(namespace, "_ignore_cpu_for_server_gpu_task_impl")(
            task,
            node_state=node_state,
            node_info=node_info,
            node_name=node_name,
            gpu_idx=gpu_idx,
            task_is_gpu_capacity_task=_ns(namespace, "_task_is_gpu_capacity_task"),
            node_is_windows=_ns(namespace, "_node_is_windows"),
            node_configs=_ns(namespace, "NODES"),
        )

    def _task_ignores_one_third_pack_rule(
        task: dict,
        node_info: Optional[dict] = None,
    ) -> bool:
        return _ns(namespace, "_task_ignores_one_third_pack_rule_impl")(
            task,
            node_info,
            hard_rule_bypassed=_ns(namespace, "_hard_rule_bypassed"),
        )

    def _task_required_gpu_idx(task: dict) -> Optional[int]:
        return _ns(namespace, "_task_required_gpu_idx_impl")(task)

    def _node_numeric(node_state: Optional[dict], *keys, default=None):
        return _ns(namespace, "_node_numeric_impl")(node_state, *keys, default=default)

    def _local_cpu_pressure_high(
        node_state: Optional[dict],
        threshold_pct: int,
    ) -> tuple[bool, str]:
        return _ns(namespace, "_local_cpu_pressure_high_impl")(node_state, threshold_pct)

    def _local_gpu_cpu_block_reason(task: dict, node_state: Optional[dict]) -> str:
        return _ns(namespace, "_local_gpu_cpu_block_reason_impl")(
            task,
            node_state,
            task_is_gpu_capacity_task=_ns(namespace, "_task_is_gpu_capacity_task"),
            threshold_pct=_ns(namespace, "LOCAL_GPU_HOST_CPU_BLOCK_PCT"),
        )

    def _node_thread_pressure_block_reason(
        task: dict,
        node_state: Optional[dict],
        node_info: Optional[dict],
    ) -> str:
        return _ns(namespace, "_node_thread_pressure_block_reason_impl")(
            task,
            node_state,
            node_info,
        )

    def _node_concurrency_cap_for_task(task: dict, node_info: dict):
        return _ns(namespace, "_node_concurrency_cap_for_task_impl")(task, node_info)

    def _gpu_task_cap_for_task(task: dict, node_info: dict):
        return _ns(namespace, "_gpu_task_cap_for_task_impl")(task, node_info)

    def _node_schedulable_cpu_free(
        task: dict,
        node_state: dict,
        node_info: dict,
    ) -> tuple[int, str, int, int | None]:
        return _ns(namespace, "_node_schedulable_cpu_free_impl")(task, node_state, node_info)

    def _node_resource_gate_deps():
        return _ns(namespace, "_build_node_resource_gate_deps")(namespace)

    def _node_resources_ok(task, node_state, node_info):
        return _ns(namespace, "_node_resources_ok_impl")(
            task,
            node_state,
            node_info,
            deps=_ns(namespace, "_node_resource_gate_deps")(),
        )

    def _node_gpu_util_limit(node_info):
        return _ns(namespace, "_node_gpu_util_limit_impl")(
            node_info,
            default_limit=_ns(namespace, "GPU_UTIL_SATURATION_PCT"),
        )

    def _task_capability_text(task: dict) -> str:
        return _ns(namespace, "_task_capability_text_impl")(task)

    def _hpc_cpu_pool_soft_require_nodes(task: dict) -> list:
        return _ns(namespace, "_hpc_cpu_pool_soft_require_nodes_impl")(task)

    def _task_gpu_capability(task: dict) -> str:
        return _ns(namespace, "_task_gpu_capability_impl")(task)

    def _task_cpu_fallback_capability(task: dict) -> str:
        return _ns(namespace, "_task_cpu_fallback_capability_impl")(task)

    def _node_cpu_fallback_block_reason(task, node_name, node_info):
        return _ns(namespace, "_node_cpu_fallback_block_reason_impl")(
            task,
            node_name,
            node_info,
            task_cpu_fallback_capability=_ns(namespace, "_task_cpu_fallback_capability"),
        )

    def _clear_disallowed_cpu_fallback_selection(task: dict) -> bool:
        return _ns(namespace, "_clear_disallowed_cpu_fallback_selection_impl")(task)

    def _task_launch_cpu_mode(task: dict) -> bool:
        return _ns(namespace, "_task_launch_cpu_mode_impl")(task)

    def _node_gpu_task_block_reason(task, node_name, node_info):
        return _ns(namespace, "_node_gpu_task_block_reason_impl")(task, node_name, node_info)

    return {
        "_task_is_gpu_capacity_task": _task_is_gpu_capacity_task,
        "_flag_enabled": _flag_enabled,
        "_ignore_cpu_for_server_gpu_task": _ignore_cpu_for_server_gpu_task,
        "_task_ignores_one_third_pack_rule": _task_ignores_one_third_pack_rule,
        "_task_required_gpu_idx": _task_required_gpu_idx,
        "_node_numeric": _node_numeric,
        "_local_cpu_pressure_high": _local_cpu_pressure_high,
        "_local_gpu_cpu_block_reason": _local_gpu_cpu_block_reason,
        "_node_thread_pressure_block_reason": _node_thread_pressure_block_reason,
        "_node_concurrency_cap_for_task": _node_concurrency_cap_for_task,
        "_gpu_task_cap_for_task": _gpu_task_cap_for_task,
        "_node_schedulable_cpu_free": _node_schedulable_cpu_free,
        "_node_resource_gate_deps": _node_resource_gate_deps,
        "_node_resources_ok": _node_resources_ok,
        "_node_gpu_util_limit": _node_gpu_util_limit,
        "_task_capability_text": _task_capability_text,
        "_hpc_cpu_pool_soft_require_nodes": _hpc_cpu_pool_soft_require_nodes,
        "_task_gpu_capability": _task_gpu_capability,
        "_task_cpu_fallback_capability": _task_cpu_fallback_capability,
        "_node_cpu_fallback_block_reason": _node_cpu_fallback_block_reason,
        "_clear_disallowed_cpu_fallback_selection": _clear_disallowed_cpu_fallback_selection,
        "_task_launch_cpu_mode": _task_launch_cpu_mode,
        "_node_gpu_task_block_reason": _node_gpu_task_block_reason,
    }
