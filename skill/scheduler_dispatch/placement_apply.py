"""Apply placement side effects to a queued task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DispatchPlacementApplyDeps:
    node_configs: dict
    algorithm_name: Callable[[], str]
    algorithm_config_snapshot: Callable[[], dict]
    node_cpu_fallback_block_reason: Callable[[dict, str, dict], str]
    task_cpu_fallback_capability: Callable[[dict], str]
    algorithm_selected_gpu_audit: Callable[[dict, dict, dict], dict | None]
    assign_windows_pin_plan: Callable[[dict, dict, dict | None], None]


def _selected_gpu(picked_state: dict | None, gpu_idx) -> dict | None:
    if picked_state is None or gpu_idx is None:
        return None
    for gpu in picked_state.get("gpus") or []:
        try:
            if int(gpu.get("idx")) == int(gpu_idx):
                return gpu
        except Exception:
            continue
    return None


def apply_dispatch_placement(
    task: dict,
    *,
    placement: tuple,
    nodes: list,
    state: dict,
    batch_hint: dict | None,
    deps: DispatchPlacementApplyDeps,
) -> dict | None:
    """Write placement metadata/audit fields and return the selected node state."""
    task["node"], task["gpu_idx"] = placement
    task["placement_algorithm"] = deps.algorithm_name()
    task["placement_algorithm_config"] = deps.algorithm_config_snapshot()

    if batch_hint:
        task["placement_algorithm_global_batch_hint"] = {
            "node": batch_hint.get("node"),
            "gpu_idx": batch_hint.get("gpu_idx"),
            "execution_contract": batch_hint.get("execution_contract"),
            "matched": (
                str(batch_hint.get("node") or "") == str(task.get("node") or "")
                and (
                    batch_hint.get("gpu_idx") is None
                    or str(batch_hint.get("gpu_idx")) == str(task.get("gpu_idx"))
                )
            ),
        }
    else:
        task.pop("placement_algorithm_global_batch_hint", None)

    selected_info = deps.node_configs.get(task.get("node"), {})
    selected_cpu_fallback = (
        int(task.get("est_vram_mb") or 0) > 0
        and task.get("gpu_idx") is None
        and not deps.node_cpu_fallback_block_reason(
            task,
            task.get("node"),
            selected_info,
        )
    )
    if selected_cpu_fallback:
        task["cpu_fallback_selected"] = True
        task["cpu_fallback_original_vram_mb"] = int(task.get("est_vram_mb") or 0)
        task["cpu_fallback_capability"] = deps.task_cpu_fallback_capability(task)
    else:
        task.pop("cpu_fallback_selected", None)
        task.pop("cpu_fallback_original_vram_mb", None)
        task.pop("cpu_fallback_capability", None)

    picked_state = next(
        (node for node in nodes if node.get("name") == task.get("node")),
        None,
    )
    selected_gpu = _selected_gpu(picked_state, task.get("gpu_idx"))
    if selected_gpu is not None:
        algo_audit = deps.algorithm_selected_gpu_audit(
            task,
            picked_state,
            selected_gpu,
        )
        if algo_audit:
            task["placement_algorithm_audit"] = algo_audit
        else:
            task.pop("placement_algorithm_audit", None)
    else:
        task.pop("placement_algorithm_audit", None)

    deps.assign_windows_pin_plan(task, state, picked_state)
    return picked_state
