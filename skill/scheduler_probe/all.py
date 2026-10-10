from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

try:
    from scheduler_transport.budget import TransportBusy
except ModuleNotFoundError:
    from ..scheduler_transport.budget import TransportBusy


@dataclass(frozen=True)
class ProbeAllDeps:
    node_configs: dict
    max_workers: int
    route_max_workers: int
    outer_ssh_route_key: Callable[[str], object]
    probe_all_route_failures: Callable[[], dict]
    probe_node: Callable[[str], dict]
    fold_claims_into_probe: Callable[[list], list]
    load_probe_cache: Callable[[], dict] = lambda: {}


def probe_all_nodes(*, deps: ProbeAllDeps) -> list:
    try:
        route_failures = deps.probe_all_route_failures()
    except TransportBusy:
        route_failures = {}
    precomputed = {}
    route_limits = {
        key: threading.Semaphore(deps.route_max_workers)
        for key in {deps.outer_ssh_route_key(name) for name in deps.node_configs}
        if key
    }
    probe_names = [
        name for name, info in deps.node_configs.items()
        if not (info or {}).get("monitor_only")
        and not (info or {}).get("retired")
    ]
    workers = max(1, min(len(probe_names), deps.max_workers))

    def probe_or_route_down(name):
        if name in precomputed:
            return precomputed[name]
        key = deps.outer_ssh_route_key(name)
        if key in route_failures:
            return {
                "name": name,
                "alive": False,
                "error": f"ssh login route down: {route_failures[key]}",
            }
        if key in route_limits:
            with route_limits[key]:
                return deps.probe_node(name)
        return deps.probe_node(name)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        nodes = list(executor.map(probe_or_route_down, probe_names))
    cached = deps.load_probe_cache()
    for index, node in enumerate(nodes):
        if node.get("transport_waiting") and node["name"] in cached:
            nodes[index] = {**cached[node["name"]],
                            "transport_waiting": True, "error": node["error"]}
    return deps.fold_claims_into_probe(nodes)
