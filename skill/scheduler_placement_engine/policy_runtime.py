from __future__ import annotations

from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_placement_policy_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _placement_runtime():
        return _ns(namespace, "_build_placement_runtime")(namespace)

    def _gpu_fits(task, gpu, node_info):
        return _ns(namespace, "_placement_gpu_fits")(
            task,
            gpu,
            node_info,
            _ns(namespace, "_placement_runtime")(),
        )

    def pick_placement(task, nodes, extra_allowed_nodes=None):
        return _ns(namespace, "_pick_placement_impl")(
            task,
            nodes,
            _ns(namespace, "_placement_runtime")(),
            extra_allowed_nodes=extra_allowed_nodes,
        )

    def _build_no_fit_reason(task, nodes, extra_allowed_nodes=None):
        return _ns(namespace, "_build_no_fit_reason_impl")(
            task,
            nodes,
            _ns(namespace, "_placement_runtime")(),
            extra_allowed_nodes=extra_allowed_nodes,
        )

    def fits(task, gpu, node_info):
        return _ns(namespace, "_gpu_fits")(task, gpu, node_info)

    return {
        "_placement_runtime": _placement_runtime,
        "_gpu_fits": _gpu_fits,
        "pick_placement": pick_placement,
        "_build_no_fit_reason": _build_no_fit_reason,
        "fits": fits,
    }
