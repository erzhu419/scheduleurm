"""Runtime-bound cross-scheduler claim wrappers for scheduler.py."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_claims_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _claims_setup_cmd():
        return _ns(namespace, "_claims_setup_cmd_impl")()

    def _claims_remote_op(node, op, payload, capacity=None, timeout_s=30):
        return _ns(namespace, "_claims_remote_op_impl")(
            node,
            op,
            payload,
            capacity,
            timeout_s,
            run_on=_ns(namespace, "run_on"),
        )

    def _claim_manager_deps():
        return _ns(namespace, "_build_claim_manager_deps")(namespace)

    class _ClaimManager(_ns(namespace, "_ClaimManagerImpl")):
        pass

    _ClaimManager.__module__ = "scheduler"
    _ClaimManager.configure(_claim_manager_deps)

    def _claim_intent_nodes(task: dict) -> list:
        return _ns(namespace, "_claim_intent_nodes_impl")(task)

    def _remember_claim_intent(task: dict, node: Optional[str]) -> None:
        return _ns(namespace, "_remember_claim_intent_impl")(
            task,
            node,
            claim_enabled_for=_ClaimManager.enabled_for,
            now=_ns(namespace, "time").time,
        )

    def _clear_claim_intent_markers(task: dict) -> None:
        return _ns(namespace, "_clear_claim_intent_markers_impl")(task)

    def _claim_resource_record_for_task(task: dict) -> dict:
        return _ns(namespace, "_claim_resource_record_for_task_impl")(
            task,
            node_configs=_ns(namespace, "NODES"),
            ignore_cpu_for_server_gpu_task=_ns(namespace, "_ignore_cpu_for_server_gpu_task"),
            task_ignores_one_third_pack_rule=_ns(namespace, "_task_ignores_one_third_pack_rule"),
            default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
            default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
            startup_floor_mb=_ns(namespace, "STARTUP_FLOOR_MB"),
        )

    def _release_task_claims_and_intents(
        task: dict,
        extra_nodes=None,
        exclude_nodes=None,
        clear_markers: bool = True,
    ) -> int:
        return _ns(namespace, "_release_task_claims_and_intents_impl")(
            task,
            claim_enabled_for=_ClaimManager.enabled_for,
            claim_release=_ClaimManager.release,
            extra_nodes=extra_nodes,
            exclude_nodes=exclude_nodes,
            clear_markers=clear_markers,
        )

    return {
        "CLAIMS_REMOTE_SCRIPT_ALIAS": _ns(namespace, "CLAIMS_REMOTE_SCRIPT"),
        "_CLAIMS_REMOTE_SCRIPT": _ns(namespace, "CLAIMS_REMOTE_SCRIPT"),
        "_claims_setup_cmd": _claims_setup_cmd,
        "_claims_remote_op": _claims_remote_op,
        "_claim_manager_deps": _claim_manager_deps,
        "_ClaimManager": _ClaimManager,
        "_claim_intent_nodes": _claim_intent_nodes,
        "_remember_claim_intent": _remember_claim_intent,
        "_clear_claim_intent_markers": _clear_claim_intent_markers,
        "_claim_resource_record_for_task": _claim_resource_record_for_task,
        "_release_task_claims_and_intents": _release_task_claims_and_intents,
    }
