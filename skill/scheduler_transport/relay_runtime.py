"""Runtime-bound relay and rsync routing wrappers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from scheduler_result_sync.routing import (
    configured_relay_nodes as _configured_relay_nodes_impl,
    relay_node_for_node as _relay_node_for_node_impl,
    relay_node_reachable as _relay_node_reachable_impl,
    relay_path_for_node as _relay_path_for_node_impl,
    relay_ssh_target_for_node as _relay_ssh_target_for_node_impl,
    rsync_path_for_node as _rsync_path_for_node_impl,
    rsync_shell_for_pair as _rsync_shell_for_pair_impl,
)


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_transport_relay_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    def _rsync_path_for_node(node: str, path: str) -> str:
        return _rsync_path_for_node_impl(
            node,
            path,
            node_configs=_ns(namespace, "NODES"),
            ssh_target_for_node=_ns(namespace, "_ssh_target_for_node"),
        )

    def _configured_relay_nodes(info: dict) -> list:
        return _configured_relay_nodes_impl(info, route_list=_ns(namespace, "_route_list"))

    def _relay_node_reachable(relay_node: str) -> bool:
        return _relay_node_reachable_impl(
            relay_node,
            node_configs=_ns(namespace, "NODES"),
            run_on=_ns(namespace, "run_on"),
        )

    def _relay_node_for_node(node: str) -> Optional[str]:
        return _relay_node_for_node_impl(
            node,
            node_configs=_ns(namespace, "NODES"),
            route_list=_ns(namespace, "_route_list"),
            relay_node_reachable=_ns(namespace, "_relay_node_reachable"),
            relay_node_cache=_ns(namespace, "_RELAY_NODE_CACHE"),
            route_cache_ttl_s=_ns(namespace, "SSH_ROUTE_CACHE_TTL_S"),
            now=_ns(namespace, "time").time,
        )

    def _relay_path_for_node(node: str, remote_path: str) -> str:
        return _relay_path_for_node_impl(
            node,
            remote_path,
            node_configs=_ns(namespace, "NODES"),
        )

    def _relay_ssh_target_for_node(node: str) -> str:
        return _relay_ssh_target_for_node_impl(node, node_configs=_ns(namespace, "NODES"))

    def _rsync_shell_for_pair(source_node: str, target_node: str) -> Optional[str]:
        return _rsync_shell_for_pair_impl(
            source_node,
            target_node,
            node_configs=_ns(namespace, "NODES"),
            ssh_rsync_shell_for_node=_ns(namespace, "_ssh_rsync_shell_for_node"),
        )

    return {
        "_rsync_path_for_node": _rsync_path_for_node,
        "_configured_relay_nodes": _configured_relay_nodes,
        "_relay_node_reachable": _relay_node_reachable,
        "_relay_node_for_node": _relay_node_for_node,
        "_relay_path_for_node": _relay_path_for_node,
        "_relay_ssh_target_for_node": _relay_ssh_target_for_node,
        "_rsync_shell_for_pair": _rsync_shell_for_pair,
    }
