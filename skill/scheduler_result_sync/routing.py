from __future__ import annotations

import os
from typing import Callable, Mapping, Optional


def rsync_path_for_node(
    node: str,
    path: str,
    *,
    node_configs: Mapping,
    ssh_target_for_node: Callable[[str], str],
) -> str:
    host = (node_configs.get(node, {}) or {}).get("host")
    if not host:
        return path
    return f"{ssh_target_for_node(node)}:{path}"


def configured_relay_nodes(info: dict, *, route_list: Callable[[object], list]) -> list:
    relays = route_list(info.get("relay_nodes"))
    if relays:
        return relays
    return route_list(info.get("relay_node"))


def relay_node_reachable(
    relay_node: str,
    *,
    node_configs: Mapping,
    run_on: Callable[..., tuple[int, str, str]],
) -> bool:
    if relay_node not in node_configs:
        return False
    try:
        rc, _, _ = run_on(relay_node, "true", timeout=8, check=False)
        return rc == 0
    except Exception:
        return False


def relay_node_for_node(
    node: str,
    *,
    node_configs: Mapping,
    route_list: Callable[[object], list],
    relay_node_reachable: Callable[[str], bool],
    relay_node_cache: dict,
    route_cache_ttl_s: int,
    now: Callable[[], float],
) -> Optional[str]:
    info = node_configs.get(node, {}) or {}
    relays = configured_relay_nodes(info, route_list=route_list)
    if not relays:
        return None
    if len(relays) == 1:
        return str(relays[0])
    current = now()
    cached = relay_node_cache.get(node)
    if cached:
        relay, ts = cached
        if relay in relays and current - float(ts or 0) <= route_cache_ttl_s:
            return str(relay)
    for relay in relays:
        if relay_node_reachable(relay):
            relay_node_cache[node] = (relay, current)
            return str(relay)
    relay_node_cache.pop(node, None)
    return str(relays[0])


def relay_path_for_node(node: str, remote_path: str, *, node_configs: Mapping) -> str:
    info = node_configs.get(node, {}) or {}
    root = str(info.get("relay_root") or f"/tmp/scheduleurm-relay/{node}").rstrip("/")
    clean = os.path.normpath(str(remote_path or "")).lstrip("/")
    if not clean or clean == ".":
        clean = "_root"
    return f"{root}/{clean}"


def relay_ssh_target_for_node(node: str, *, node_configs: Mapping) -> str:
    info = node_configs.get(node, {}) or {}
    explicit = info.get("relay_ssh_target")
    if explicit:
        return str(explicit)
    host = info.get("host")
    user = info.get("ssh_user")
    return f"{user}@{host}" if user and "@" not in str(host) else str(host)


def rsync_shell_for_pair(
    source_node: str,
    target_node: str,
    *,
    node_configs: Mapping,
    ssh_rsync_shell_for_node: Callable[[str], str],
) -> Optional[str]:
    src_host = (node_configs.get(source_node or "", {}) or {}).get("host")
    tgt_host = (node_configs.get(target_node or "", {}) or {}).get("host")
    if src_host and tgt_host:
        return None
    if src_host:
        return ssh_rsync_shell_for_node(source_node)
    if tgt_host:
        return ssh_rsync_shell_for_node(target_node)
    return None
