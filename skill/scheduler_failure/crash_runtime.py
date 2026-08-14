"""Runtime wrappers for crash requeue handling."""

from __future__ import annotations

from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_failure_crash_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _launch_failed_nodes_for_task(task):
        return _ns(namespace, "_launch_failed_nodes_for_task_impl")(
            task,
            known_nodes=_ns(namespace, "NODES"),
            soft_block_ttl_s=_ns(namespace, "LAUNCH_FAILED_NODE_TTL_S"),
            now=_ns(namespace, "time").time,
        )

    def _crash_requeue_deps():
        return _ns(namespace, "_build_crash_requeue_deps")(namespace)

    def _requeue_after_crash(parent, state):
        return _ns(namespace, "_requeue_after_crash_impl")(
            parent,
            state,
            deps=_ns(namespace, "_crash_requeue_deps")(),
        )

    return {
        "_launch_failed_nodes_for_task": _launch_failed_nodes_for_task,
        "_crash_requeue_deps": _crash_requeue_deps,
        "_requeue_after_crash": _requeue_after_crash,
    }
