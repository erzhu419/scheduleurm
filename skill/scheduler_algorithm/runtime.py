"""Runtime-bound /algorithm bridge exports for the scheduler facade."""

from __future__ import annotations

import os
from typing import Any, Optional

from .bridge import AlgorithmBridge


def _ns(namespace: dict[str, Any], name: str) -> Any:
    return namespace[name]


def build_algorithm_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _scheduler_algorithm_runtime_context() -> dict:
        return {
            "gpu_empty_used_mb": _ns(namespace, "GPU_EMPTY_USED_MB"),
            "vram_margin_mb": _ns(namespace, "VRAM_MARGIN_MB"),
            "one_third_pack_rule": _ns(namespace, "ONE_THIRD_PACK_RULE"),
            "one_third_pack_grace_mb": _ns(namespace, "ONE_THIRD_PACK_GRACE_MB"),
            "gpu_util_saturation_pct": _ns(namespace, "GPU_UTIL_SATURATION_PCT"),
            "state_dir": str(_ns(namespace, "STATE_DIR")),
            "queue_file": str(_ns(namespace, "QUEUE_FILE")),
        }

    bridge = AlgorithmBridge(
        runtime_context_factory=_scheduler_algorithm_runtime_context,
        node_configs_factory=lambda: _ns(namespace, "NODES"),
        environ=os.environ,
    )

    def _configure_algorithm(name: Optional[str] = None):
        return bridge.configure(name)

    def _algorithm_runtime_context() -> dict:
        return bridge.runtime_context()

    def _algorithm_name() -> str:
        return bridge.name()

    def _algorithm_config_snapshot() -> dict:
        return bridge.config_snapshot()

    def _algorithm_gpu_fit_block_reason(task: dict, gpu: dict, node_info: dict) -> str:
        return bridge.gpu_fit_block_reason(task, gpu, node_info)

    def _algorithm_gpu_score(task: dict, node_state: dict, gpu: dict, legacy_score):
        return bridge.gpu_score(task, node_state, gpu, legacy_score)

    def _algorithm_selected_gpu_audit(task: dict, node_state: dict, gpu: dict) -> dict:
        return bridge.selected_gpu_audit(task, node_state, gpu)

    def _algorithm_global_batch_hook_enabled() -> bool:
        return bridge.global_batch_hook_enabled()

    def _algorithm_global_batch_plan(tasks: list, nodes: list) -> tuple[dict, dict]:
        return bridge.global_batch_plan(tasks, nodes)

    def _hard_rule_bypassed(
        rule: str,
        task: Optional[dict] = None,
        node_info: Optional[dict] = None,
        node_state: Optional[dict] = None,
        gpu: Optional[dict] = None,
    ) -> bool:
        return bridge.hard_rule_bypassed(
            rule,
            task=task,
            node_info=node_info,
            node_state=node_state,
            gpu=gpu,
        )

    def _hard_rule_override_snapshot(task: Optional[dict] = None) -> dict:
        return bridge.hard_rule_override_snapshot(task)

    def _set_hard_rule_mode(mode: Optional[str]) -> None:
        bridge.set_hard_rule_mode(mode)

    _configure_algorithm()

    return {
        "_ALGORITHM_BRIDGE": bridge,
        "_scheduler_algorithm_runtime_context": _scheduler_algorithm_runtime_context,
        "_configure_algorithm": _configure_algorithm,
        "_algorithm_runtime_context": _algorithm_runtime_context,
        "_algorithm_name": _algorithm_name,
        "_algorithm_config_snapshot": _algorithm_config_snapshot,
        "_algorithm_gpu_fit_block_reason": _algorithm_gpu_fit_block_reason,
        "_algorithm_gpu_score": _algorithm_gpu_score,
        "_algorithm_selected_gpu_audit": _algorithm_selected_gpu_audit,
        "_algorithm_global_batch_hook_enabled": _algorithm_global_batch_hook_enabled,
        "_algorithm_global_batch_plan": _algorithm_global_batch_plan,
        "_hard_rule_bypassed": _hard_rule_bypassed,
        "_hard_rule_override_snapshot": _hard_rule_override_snapshot,
        "_set_hard_rule_mode": _set_hard_rule_mode,
    }
