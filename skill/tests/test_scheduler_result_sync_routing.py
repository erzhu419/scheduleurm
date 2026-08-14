from __future__ import annotations

from skill.scheduler_result_sync import routing


def test_rsync_path_for_node_returns_plain_local_path_and_remote_target_path():
    configs = {
        "local": {"host": None},
        "remote": {"host": "host"},
    }

    assert routing.rsync_path_for_node(
        "local",
        "/data/run",
        node_configs=configs,
        ssh_target_for_node=lambda node: f"user@{node}",
    ) == "/data/run"
    assert routing.rsync_path_for_node(
        "remote",
        "/data/run",
        node_configs=configs,
        ssh_target_for_node=lambda node: f"user@{node}",
    ) == "user@remote:/data/run"


def test_configured_relay_nodes_prefers_list_over_single_value():
    assert routing.configured_relay_nodes(
        {"relay_nodes": ["gpu1", "gpu2"], "relay_node": "gpu3"},
        route_list=lambda value: list(value) if isinstance(value, list) else ([value] if value else []),
    ) == ["gpu1", "gpu2"]
    assert routing.configured_relay_nodes(
        {"relay_node": "gpu3"},
        route_list=lambda value: list(value) if isinstance(value, list) else ([value] if value else []),
    ) == ["gpu3"]


def test_relay_node_for_node_uses_reachable_relay_and_short_ttl_cache():
    configs = {
        "target": {"relay_nodes": ["gpu2", "gpu1"]},
        "gpu2": {"host": "gpu2"},
        "gpu1": {"host": "gpu1"},
    }
    cache = {}
    reachable = {"gpu2": False, "gpu1": True}
    calls = []

    selected = routing.relay_node_for_node(
        "target",
        node_configs=configs,
        route_list=lambda value: list(value or []),
        relay_node_reachable=lambda node: calls.append(node) or reachable[node],
        relay_node_cache=cache,
        route_cache_ttl_s=30,
        now=lambda: 100.0,
    )

    assert selected == "gpu1"
    assert calls == ["gpu2", "gpu1"]
    assert cache["target"] == ("gpu1", 100.0)

    calls.clear()
    assert routing.relay_node_for_node(
        "target",
        node_configs=configs,
        route_list=lambda value: list(value or []),
        relay_node_reachable=lambda node: calls.append(node) or False,
        relay_node_cache=cache,
        route_cache_ttl_s=30,
        now=lambda: 120.0,
    ) == "gpu1"
    assert calls == []

    assert routing.relay_node_for_node(
        "target",
        node_configs=configs,
        route_list=lambda value: list(value or []),
        relay_node_reachable=lambda node: False,
        relay_node_cache=cache,
        route_cache_ttl_s=30,
        now=lambda: 200.0,
    ) == "gpu2"
    assert "target" not in cache


def test_relay_node_reachable_requires_known_node_and_successful_probe():
    configs = {"gpu1": {"host": "gpu1"}}
    calls = []

    assert routing.relay_node_reachable(
        "missing",
        node_configs=configs,
        run_on=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    ) is False
    assert routing.relay_node_reachable(
        "gpu1",
        node_configs=configs,
        run_on=lambda *args, **kwargs: calls.append((args, kwargs)) or (0, "", ""),
    ) is True
    assert calls[0][0][0:2] == ("gpu1", "true")
    assert routing.relay_node_reachable(
        "gpu1",
        node_configs=configs,
        run_on=lambda *args, **kwargs: (255, "", "down"),
    ) is False


def test_relay_paths_and_targets_are_normalized():
    configs = {
        "node": {"host": "10.0.0.7", "ssh_user": "user", "relay_root": "/relay/root/"},
        "explicit": {"host": "10.0.0.8", "relay_ssh_target": "jump-target"},
        "already_user": {"host": "user@host", "ssh_user": "ignored"},
    }

    assert routing.relay_path_for_node("node", "/a/../b/run", node_configs=configs) == "/relay/root/b/run"
    assert routing.relay_path_for_node("node", "", node_configs=configs) == "/relay/root/_root"
    assert routing.relay_ssh_target_for_node("node", node_configs=configs) == "user@10.0.0.7"
    assert routing.relay_ssh_target_for_node("explicit", node_configs=configs) == "jump-target"
    assert routing.relay_ssh_target_for_node("already_user", node_configs=configs) == "user@host"


def test_rsync_shell_for_pair_returns_shell_only_when_one_side_is_remote():
    configs = {
        "local": {"host": None},
        "r1": {"host": "h1"},
        "r2": {"host": "h2"},
    }

    shell = lambda node: f"ssh-shell-{node}"

    assert routing.rsync_shell_for_pair("local", "local", node_configs=configs, ssh_rsync_shell_for_node=shell) is None
    assert routing.rsync_shell_for_pair("r1", "r2", node_configs=configs, ssh_rsync_shell_for_node=shell) is None
    assert routing.rsync_shell_for_pair("r1", "local", node_configs=configs, ssh_rsync_shell_for_node=shell) == "ssh-shell-r1"
    assert routing.rsync_shell_for_pair("local", "r2", node_configs=configs, ssh_rsync_shell_for_node=shell) == "ssh-shell-r2"
