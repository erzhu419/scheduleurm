from __future__ import annotations

from skill.scheduler_probe.node import ProbeNodeDeps, parse_probe_output, probe_node


def _sample_output(*, loadavg="4.0", threads="20 4096 30 1000"):
    return (
        "0, 1000, 24000, 23000, 10\n"
        "1, 2000, 24000, 22000, 20\n"
        "===SEP===\n"
        "30000\n"
        "64000\n"
        "===SEP===\n"
        "16\n"
        "===SEP===\n"
        f"{loadavg}\n"
        "===SEP===\n"
        "0, 40\n"
        "1, 90\n"
        "---SAMPLE---\n"
        "0, 70\n"
        "1, 30\n"
        "---SAMPLE---\n"
        "===SEP===\n"
        f"{threads}\n"
    )


def _deps(
    *,
    nodes=None,
    run_calls=None,
    run_results=None,
    path_exists=False,
    windows_extras=None,
    sleep_calls=None,
):
    nodes = nodes or {}
    run_calls = run_calls if run_calls is not None else []
    run_results = list(run_results or [(0, _sample_output(), "")])
    sleep_calls = sleep_calls if sleep_calls is not None else []

    def run_on(node, cmd, **kwargs):
        run_calls.append((node, cmd, kwargs))
        if run_results:
            return run_results.pop(0)
        return 0, _sample_output(), ""

    return ProbeNodeDeps(
        node_configs=nodes,
        canonical_node_name=lambda name: {"alias": "node001"}.get(name, name),
        node_is_windows=lambda name: name == "win",
        probe_windows_node=lambda name: {"name": name, "alive": True, "windows": True},
        path_exists=lambda path: path_exists,
        run_on=run_on,
        probe_windows_host_extras=lambda: windows_extras,
        sleep=lambda seconds: sleep_calls.append(seconds),
    )


def test_parse_probe_output_uses_memavailable_first_caps_declared_ram_and_averages_gpu_util():
    deps = _deps()

    result = parse_probe_output(
        "node001",
        {"ram_mb": 56000, "cpu_cores": 12},
        _sample_output(loadavg="4.0"),
        deps=deps,
    )

    assert result["name"] == "node001"
    assert result["alive"] is True
    assert result["actual_free_ram_mb"] == 30000
    assert result["total_ram_mb"] == 56000
    assert result["free_ram_mb"] == 30000
    assert result["total_cpu"] == 12
    assert result["free_cpu"] == 8
    assert result["cores"] == 16
    assert result["user_threads"] == 20
    assert result["user_thread_limit"] == 4096
    assert result["free_user_threads"] == 4076
    assert result["cgroup_pids_current"] == 30
    assert result["cgroup_pids_max"] == 1000
    assert result["free_cgroup_pids"] == 970
    assert result["gpus"][0]["util_pct"] == 40
    assert result["gpus"][1]["util_pct"] == 46


def test_local_probe_prefers_windows_nvidia_smi_and_folds_host_extras():
    run_calls = []
    deps = _deps(
        nodes={"local": {"cpu_cores": 16, "ram_mb": 64000}},
        run_calls=run_calls,
        path_exists=True,
        windows_extras={
            "host_free_ram_mb": 20000,
            "host_total_ram_mb": 128000,
            "host_cpu_load_pct": 75,
            "gpu_compute_util_pct": 88,
        },
    )

    result = probe_node("local", deps=deps)

    assert "/mnt/c/WINDOWS/system32/nvidia-smi.exe" in run_calls[0][1]
    assert result["wsl_free_ram_mb"] == 30000
    assert result["host_free_ram_mb"] == 20000
    assert result["free_ram_mb"] == 20000
    assert result["host_total_ram_mb"] == 128000
    assert result["host_cpu_load_pct"] == 75
    assert result["host_cpu_used_cores"] == 12
    assert result["wsl_free_cpu"] == 12
    assert result["free_cpu"] == 4
    assert result["gpus"][0]["util_pct_compute"] == 88


def test_probe_node_retries_sudo_route_failures_then_succeeds():
    run_calls = []
    sleep_calls = []
    deps = _deps(
        nodes={"node001": {"sudo_ssh_host": "jump", "probe_attempts": 3}},
        run_calls=run_calls,
        run_results=[
            (255, "", "temporary ssh error"),
            (0, _sample_output(loadavg="1.0"), ""),
        ],
        sleep_calls=sleep_calls,
    )

    result = probe_node("alias", deps=deps)

    assert result["name"] == "node001"
    assert result["alive"] is True
    assert len(run_calls) == 2
    assert sleep_calls == [0.4]


def test_probe_node_ignores_removed_cluster_probe_config():
    run_calls = []
    deps = _deps(
        nodes={"node001": {"probe_slurm_cluster": True}},
        run_calls=run_calls,
    )

    result = probe_node("node001", deps=deps)

    assert result["name"] == "node001"
    assert result["alive"] is True
    assert run_calls and run_calls[0][0] == "node001"


def test_probe_node_reports_down_after_all_attempts_fail():
    deps = _deps(
        nodes={"node001": {"probe_attempts": 2}},
        run_results=[
            (1, "", "bad route"),
            (1, "", "still bad"),
        ],
        sleep_calls=[],
    )

    result = probe_node("node001", deps=deps)

    assert result == {
        "name": "node001",
        "alive": False,
        "error": "ssh/cmd failed: still bad",
    }


def test_probe_node_delegates_windows_but_ignores_removed_cluster_probe():
    run_calls = []
    deps = _deps(
        nodes={
            "win": {},
            "old-probe": {"probe_slurm_cluster": True},
        },
        run_calls=run_calls,
    )

    assert probe_node("win", deps=deps) == {"name": "win", "alive": True, "windows": True}
    result = probe_node("old-probe", deps=deps)
    assert result["name"] == "old-probe"
    assert result["alive"] is True
    assert "gpus" in result
    assert run_calls and run_calls[0][0] == "old-probe"


def test_probe_node_removed_backend_config_uses_normal_resource_probe():
    run_calls = []
    deps = _deps(
        nodes={"old-login": {"slurm_backend": "slurm"}},
        run_calls=run_calls,
    )

    result = probe_node("old-login", deps=deps)

    assert result["name"] == "old-login"
    assert result["alive"] is True
    assert "gpus" in result
    assert len(run_calls) == 1
