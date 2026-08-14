from skill.scheduler_windows_probe import (
    WindowsNodeProbeDeps,
    collect_windows_probe_targets,
    normalize_windows_json_rows,
    probe_windows_node,
    schedulable_windows_cpu_from_logical,
    windows_log_exit_results_from_rows,
    windows_process_results_from_rows,
    windows_queue_accounting_fallback_result,
    windows_terminal_log_result,
)


class _Proc:
    def __init__(self, returncode=0, stdout=b"OK\r\n"):
        self.returncode = returncode
        self.stdout = stdout


def _node_probe_deps(
    *,
    nodes=None,
    tasks=None,
    ps_results=None,
    subprocess_result=None,
    env_cpu_load_source=None,
):
    nodes = nodes or {"win": {"cpu_cores": 128, "ram_mb": 524288}}
    tasks = tasks or []
    ps_results = list(ps_results or [(0, "500000|524288|256\n", "")])
    calls = []

    def run_windows_ps(node, script, **kwargs):
        calls.append(("ps", node, script, kwargs))
        if ps_results:
            return ps_results.pop(0)
        return 0, "500000|524288|256\n", ""

    def run_subprocess(args, **kwargs):
        calls.append(("subprocess", args, kwargs))
        return subprocess_result if subprocess_result is not None else _Proc()

    deps = WindowsNodeProbeDeps(
        node_configs=nodes,
        queue_tasks=lambda: list(tasks),
        run_windows_ps=run_windows_ps,
        run_subprocess=run_subprocess,
        ssh_base_args=lambda node: ["ssh", node],
        windows_probe_error_hint=lambda node, msg: f"hint:{node}:{msg}",
        env_cpu_load_source=env_cpu_load_source,
    )
    return deps, calls


def test_collect_windows_probe_targets_filters_running_windows_with_pids():
    state = {"tasks": [
        {"id": "w1", "status": "running", "node": "win", "remote_pids": [11]},
        {"id": "l1", "status": "running", "node": "linux", "remote_pids": [12]},
        {"id": "w2", "status": "done", "node": "win", "remote_pids": [13]},
        {"id": "w3", "status": "running", "node": "win", "remote_pids": []},
    ]}
    out = collect_windows_probe_targets(
        state,
        node_is_windows=lambda node: node == "win",
        task_pids=lambda task: task.get("remote_pids") or [],
    )

    assert list(out) == ["win"]
    assert [task["id"] for task, _pids in out["win"]] == ["w1"]


def test_schedulable_windows_cpu_from_logical_honors_declared_cap_and_ht_skip():
    assert schedulable_windows_cpu_from_logical(256, {"cpu_cores": 128}) == 128
    assert schedulable_windows_cpu_from_logical(64, {"cpu_cores": 128}) == 32
    assert schedulable_windows_cpu_from_logical(
        64,
        {"cpu_cores": 128, "windows_skip_ht_pair": False},
    ) == 64


def test_probe_windows_node_queue_mode_uses_live_ram_and_queue_cpu():
    deps, calls = _node_probe_deps(
        tasks=[
            {"id": "t1", "status": "running", "node": "win", "cpu_cores": 20, "ram_mb": 1000},
            {"id": "t2", "status": "running", "node": "win", "cpu_cores": 12, "current_ram_mb": 2000},
            {"id": "t3", "status": "running", "node": "other", "cpu_cores": 99},
        ],
        ps_results=[(0, "#< CLIXML noise\n500000|524288|256\n", "")],
    )

    out = probe_windows_node("win", deps=deps)

    assert out["alive"] is True
    assert out["probe_fallback"] == "queue_cpu_live_ram"
    assert out["total_cpu"] == 128
    assert out["free_cpu"] == 96
    assert out["cpu_load_pct"] == 25
    assert out["free_ram_mb"] == 500000
    assert out["actual_free_ram_mb"] == 500000
    assert out["total_ram_mb"] == 524288
    assert len([call for call in calls if call[0] == "ps"]) == 1


def test_probe_windows_node_live_mode_uses_full_probe_load_pct():
    deps, _calls = _node_probe_deps(
        nodes={"win": {"cpu_cores": 8, "ram_mb": 1000, "windows_cpu_load_source": "live"}},
        ps_results=[(0, "noise\n700|1200|16|25\n", "")],
    )

    out = probe_windows_node("win", deps=deps)

    assert out["alive"] is True
    assert "probe_fallback" not in out
    assert out["total_cpu"] == 8
    assert out["free_cpu"] == 6
    assert out["loadavg"] == 2.0
    assert out["free_ram_mb"] == 700
    assert out["total_ram_mb"] == 1000


def test_probe_windows_node_full_probe_parse_failure_falls_back_to_queue_accounting():
    deps, calls = _node_probe_deps(
        nodes={"win": {"cpu_cores": 128, "ram_mb": 524288, "windows_cpu_load_source": "live"}},
        tasks=[{"id": "t1", "status": "running", "node": "win", "cpu_cores": 40, "current_ram_mb": 1000}],
        ps_results=[
            (0, "not-a-probe-row\n", ""),
            (0, "400000|524288|256\n", ""),
            (0, "400000|524288|256\n", ""),
        ],
    )

    out = probe_windows_node("win", deps=deps)

    assert out["alive"] is True
    assert out["probe_fallback"] == "queue_accounting"
    assert out["free_cpu"] == 88
    assert out["free_ram_mb"] == 400000
    assert out["probe_fallback_reason"] == "not-a-probe-row"
    assert any(call[0] == "subprocess" for call in calls)


def test_probe_windows_node_fallback_marks_down_when_ping_fails():
    deps, _calls = _node_probe_deps(
        nodes={"win": {"cpu_cores": 128, "ram_mb": 524288, "windows_cpu_load_source": "live"}},
        ps_results=[(0, "bad\n", "")],
        subprocess_result=_Proc(returncode=255, stdout=b""),
    )

    out = probe_windows_node("win", deps=deps)

    assert out == {
        "name": "win",
        "alive": False,
        "error": "hint:win:bad",
    }


def test_windows_terminal_log_result_marks_rc_and_reason():
    result = windows_terminal_log_result(2)

    assert result["state"] == "dead"
    assert result["terminal_ok"] is False
    assert result["backend_state"] == "WINDOWS_LOG_RC_2"
    assert result["exit_code"] == 2


def test_windows_log_exit_results_from_rows_handles_exit_alive_and_dead_candidates():
    rows = [
        {"id": "exit", "log_exit": True, "exit_code": 0},
        {"id": "alive", "alive": [10, 11], "ram_mb": 2048, "candidate_count": 2},
        {"id": "dead", "alive": [], "candidate_count": 1},
    ]
    out = windows_log_exit_results_from_rows(rows)

    assert out["exit"]["terminal_ok"] is True
    assert out["alive"] == {
        "state": "alive",
        "alive_pids": [10, 11],
        "vram_mb": 0,
        "ram_mb": 2048,
        "pcpu": 0.0,
        "probe_fallback": "windows_log_pid_liveness",
    }
    assert out["dead"]["state"] == "dead"
    assert out["dead"]["probe_fallback"] == "windows_log_pid_liveness"


def test_windows_queue_accounting_fallback_result_uses_known_and_declared_usage():
    task = {
        "id": "t",
        "alive_pids": [21],
        "current_ram_mb": 3000,
        "peak_ram_mb": 2000,
        "ram_mb": 1000,
        "cpu_cores": 4,
    }
    out = windows_queue_accounting_fallback_result(
        task, [20, 21], "windows_queue_accounting")

    assert out["state"] == "alive"
    assert out["alive_pids"] == [20, 21]
    assert out["ram_mb"] == 3000
    assert out["pcpu"] == 400.0
    assert out["probe_fallback"] == "windows_queue_accounting"


def test_windows_process_results_from_rows_maps_alive_log_exit_unknown_and_dead():
    tasks = [
        ({"id": "alive", "cpu_cores": 2}, [100]),
        ({"id": "exit"}, [200]),
        ({"id": "unknown"}, [300]),
        ({"id": "dead"}, [400]),
    ]
    rows = [
        {"root": 100, "alive": [100, 101], "ram_mb": 1024, "pcpu": 0.0},
        {"root": 200, "log_exit": True, "exit_code": 7},
    ]
    out = windows_process_results_from_rows(
        tasks,
        rows,
        local_launch_transport_alive=lambda task: task.get("id") == "unknown",
    )

    assert out["alive"] == {
        "state": "alive",
        "alive_pids": [100, 101],
        "vram_mb": 0,
        "ram_mb": 1024,
        "pcpu": 200.0,
    }
    assert out["exit"]["backend_state"] == "WINDOWS_LOG_RC_7"
    assert out["unknown"]["state"] == "unknown"
    assert out["dead"]["state"] == "dead"


def test_normalize_windows_json_rows_accepts_object_and_rejects_empty_when_required():
    assert normalize_windows_json_rows('{"root": 1}', require_nonempty=True) == [{"root": 1}]
    assert normalize_windows_json_rows("[]", require_nonempty=False) == []
    try:
        normalize_windows_json_rows("[]", require_nonempty=True)
    except ValueError as exc:
        assert "no rows" in str(exc)
    else:
        raise AssertionError("expected empty list rejection")
