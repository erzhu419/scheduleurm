from __future__ import annotations

from skill.scheduler_probe.all import ProbeAllDeps, probe_all_nodes


def _deps(
    *,
    node_configs,
    route_failures=None,
    probe_results=None,
    calls=None,
    folded=None,
    max_workers=1,
):
    route_failures = route_failures or {}
    probe_results = probe_results or {}
    calls = calls if calls is not None else []
    folded = folded if folded is not None else []

    def probe_node(name):
        calls.append(name)
        return probe_results.get(name, {"name": name, "alive": True, "gpus": []})

    def fold(nodes):
        folded.append(list(nodes))
        return [{"folded": True, **node} for node in nodes]

    return ProbeAllDeps(
        node_configs=node_configs,
        max_workers=max_workers,
        route_max_workers=2,
        outer_ssh_route_key=lambda name: (node_configs.get(name) or {}).get("route_key"),
        probe_all_route_failures=lambda: dict(route_failures),
        probe_node=probe_node,
        fold_claims_into_probe=fold,
    )


def test_probe_all_skips_monitor_only_and_folds_probe_results():
    calls = []
    folded = []
    deps = _deps(
        node_configs={
            "local": {},
            "dashboard-only": {"monitor_only": True},
        },
        calls=calls,
        folded=folded,
    )

    result = probe_all_nodes(deps=deps)

    assert calls == ["local"]
    assert folded == [[{"name": "local", "alive": True, "gpus": []}]]
    assert result == [{"folded": True, "name": "local", "alive": True, "gpus": []}]


def test_probe_all_collapses_outer_route_failures_without_calling_probe_node():
    calls = []
    folded = []
    deps = _deps(
        node_configs={
            "node001": {"route_key": ("jump", "hpc")},
            "node002": {"route_key": ("jump", "hpc")},
            "node007": {"route_key": ("direct", "node007")},
        },
        route_failures={("jump", "hpc"): "outer ssh route timed out"},
        calls=calls,
        folded=folded,
    )

    result = probe_all_nodes(deps=deps)

    assert calls == ["node007"]
    assert result == [
        {
            "folded": True,
            "name": "node001",
            "alive": False,
            "error": "ssh login route down: outer ssh route timed out",
        },
        {
            "folded": True,
            "name": "node002",
            "alive": False,
            "error": "ssh login route down: outer ssh route timed out",
        },
        {"folded": True, "name": "node007", "alive": True, "gpus": []},
    ]
    assert folded and len(folded[0]) == 3


def test_probe_all_handles_empty_inventory():
    folded = []
    deps = _deps(node_configs={}, folded=folded)

    assert probe_all_nodes(deps=deps) == []
    assert folded == [[]]
