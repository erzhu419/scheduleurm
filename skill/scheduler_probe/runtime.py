"""Runtime-bound node probe wrappers for scheduler.py."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from algorithm.experiments.theorem_measurement_reservation import (
        fold_measurement_reservations_into_probe as _fold_measurement_reservations,
    )
except Exception:
    def _fold_measurement_reservations(nodes):
        return nodes


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_probe_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _windows_host_extras_deps():
        return _ns(namespace, "_build_windows_host_extras_deps")(namespace)

    def _probe_windows_host_extras(timeout_s: float = 4.0) -> dict:
        return _ns(namespace, "_probe_windows_host_extras_impl")(
            timeout_s,
            deps=_ns(namespace, "_windows_host_extras_deps")(),
        )

    def _windows_node_probe_deps():
        return _ns(namespace, "_build_windows_node_probe_deps")(namespace)

    def _probe_windows_node(name: str) -> dict:
        return _ns(namespace, "_probe_windows_node_impl")(
            name,
            deps=_ns(namespace, "_windows_node_probe_deps")(),
        )

    def _cpu_only_sudo_probe_cmd() -> str:
        return _ns(namespace, "_cpu_only_sudo_probe_cmd_impl")()

    def _parse_cpu_only_sudo_probe(name: str, body: str) -> dict:
        return _ns(namespace, "_parse_cpu_only_sudo_probe_impl")(
            name,
            body,
            node_configs=_ns(namespace, "NODES"),
        )

    def _sudo_cpu_batch_probe_deps():
        return _ns(namespace, "_build_sudo_cpu_batch_probe_deps")(namespace)

    def _probe_sudo_cpu_nodes_batch(names: list[str]) -> dict:
        return _ns(namespace, "_probe_sudo_cpu_nodes_batch_impl")(
            names,
            deps=_ns(namespace, "_sudo_cpu_batch_probe_deps")(),
        )

    def probe_node(name):
        return _ns(namespace, "_probe_node_impl")(
            name,
            deps=_ns(namespace, "_probe_node_deps")(),
        )

    def _probe_node_deps():
        return _ns(namespace, "_build_probe_node_deps")(namespace)

    def _probe_all_deps():
        return _ns(namespace, "_build_probe_all_deps")(namespace)

    def probe_all():
        nodes = _ns(namespace, "_probe_all_nodes_impl")(
            deps=_ns(namespace, "_probe_all_deps")(),
        )
        _ns(namespace, "_save_node_probe_cache")({
            str(node.get("name")): node
            for node in nodes
            if node.get("name")
        })
        return nodes

    def probe_all_cached(max_age_s: int = 30):
        cached = _ns(namespace, "_load_node_probe_cache")(max_age_s=max_age_s)
        expected = [
            name
            for name, info in _ns(namespace, "NODES").items()
            if not (info or {}).get("monitor_only")
            and not (info or {}).get("retired")
        ]
        if cached and all(name in cached for name in expected):
            return [cached[name] for name in expected]
        return probe_all()

    def _claim_probe_fold_deps():
        return _ns(namespace, "_build_claim_probe_fold_deps")(namespace)

    def _fold_claims_into_probe(nodes: list) -> list:
        folded = _ns(namespace, "_fold_claims_into_probe_impl")(
            nodes,
            deps=_ns(namespace, "_claim_probe_fold_deps")(),
        )
        return _fold_measurement_reservations(folded)

    return {
        "_windows_host_extras_deps": _windows_host_extras_deps,
        "_probe_windows_host_extras": _probe_windows_host_extras,
        "_windows_node_probe_deps": _windows_node_probe_deps,
        "_probe_windows_node": _probe_windows_node,
        "_cpu_only_sudo_probe_cmd": _cpu_only_sudo_probe_cmd,
        "_parse_cpu_only_sudo_probe": _parse_cpu_only_sudo_probe,
        "_sudo_cpu_batch_probe_deps": _sudo_cpu_batch_probe_deps,
        "_probe_sudo_cpu_nodes_batch": _probe_sudo_cpu_nodes_batch,
        "probe_node": probe_node,
        "_probe_node_deps": _probe_node_deps,
        "_probe_all_deps": _probe_all_deps,
        "probe_all": probe_all,
        "probe_all_cached": probe_all_cached,
        "_claim_probe_fold_deps": _claim_probe_fold_deps,
        "_fold_claims_into_probe": _fold_claims_into_probe,
    }
