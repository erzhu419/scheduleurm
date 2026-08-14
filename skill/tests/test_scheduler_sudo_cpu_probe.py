from __future__ import annotations

from types import SimpleNamespace

from skill.scheduler_probe import sudo_cpu as sudo_probe


def test_parse_cpu_only_sudo_probe_uses_declared_caps_and_thread_limits():
    body = "90000\n128000\n__SEP__\n64\n__SEP__\n5.4\n__SEP__\n100 200 50 300\n"

    got = sudo_probe.parse_cpu_only_sudo_probe(
        "node001",
        body,
        node_configs={"node001": {"ram_mb": 100000, "cpu_cores": 32}},
    )

    assert got["alive"] is True
    assert got["free_ram_mb"] == 90000
    assert got["total_ram_mb"] == 100000
    assert got["free_cpu"] == 27
    assert got["cores"] == 64
    assert got["user_threads"] == 100
    assert got["free_user_threads"] == 100
    assert got["cgroup_pids_current"] == 50
    assert got["free_cgroup_pids"] == 250
    assert got["probe_source"] == "sudo_batch"


def _deps(stdout, *, probe_jump=True):
    route_jump_cache = {}
    outer_route_cache = {}
    return sudo_probe.SudoCpuBatchProbeDeps(
        node_configs={
            "zhengliang-hpc": {},
            "node001": {"sudo_ssh_host": "node001", "max_vram_per_task": 0, "ram_mb": 100000, "cpu_cores": 32},
            "gpu-node": {"sudo_ssh_host": "gpu", "max_vram_per_task": 12000},
        },
        route_jump_cache=route_jump_cache,
        outer_route_cache=outer_route_cache,
        route_cache_ttl_s=60,
        login_prefight_timeout_s=2,
        now=lambda: 100.0,
        outer_ssh_route_key=lambda name: "route",
        configured_proxy_jumps=lambda info: ["gpu2"],
        probe_ssh_proxy_jump=lambda name, jump, timeout: probe_jump,
        remote_bash_command_for_node=lambda name, cmd, timeout=None: f"REMOTE[{name}]",
        ssh_base_args_with_proxy=lambda node, jump: ["ssh", jump, node],
        run_ssh_subprocess=lambda args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )


def test_probe_sudo_cpu_nodes_batch_parses_success_and_missing_results():
    stdout = (
        "__SCHED_NODE_BEGIN__ node001\n"
        "90000\n128000\n__SEP__\n64\n__SEP__\n5.4\n__SEP__\n100 200 50 300\n"
        "__SCHED_NODE_END__ node001 0\n"
    )

    got = sudo_probe.probe_sudo_cpu_nodes_batch(["node001", "gpu-node"], deps=_deps(stdout))

    assert set(got) == {"node001"}
    assert got["node001"]["alive"] is True
    assert got["node001"]["free_cpu"] == 27
    assert got["node001"]["probe_source"] == "sudo_batch"


def test_probe_sudo_cpu_nodes_batch_marks_down_when_no_proxy_jump():
    got = sudo_probe.probe_sudo_cpu_nodes_batch(
        ["node001"],
        deps=_deps("", probe_jump=False),
    )

    assert got["node001"]["alive"] is False
    assert "no working sudo-hop jump" in got["node001"]["error"]
