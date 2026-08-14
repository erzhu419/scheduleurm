import base64
import json
from pathlib import Path

from skill.scheduler_windows_backend import (
    WindowsBatchProbeDeps,
    WindowsLaunchDeps,
    windows_backend_batch_probe,
    windows_backend_launch,
)


def _launch_deps(tmp_path: Path, *, ps_results=None, prepare_command=None):
    calls = []
    payloads = []
    ps_results = list(ps_results or [
        (0, "OK\n", ""),
        (0, "WROTE\n", ""),
        (0, "PID=4321\n", ""),
    ])

    def run_windows_ps(node, script, **kwargs):
        calls.append((node, script, kwargs))
        if kwargs.get("input_data"):
            payloads.append(json.loads(base64.b64decode(kwargs["input_data"]).decode("utf-8")))
        return ps_results.pop(0)

    deps = WindowsLaunchDeps(
        node_configs={"win": {
            "windows_python": r"C:\Python\python.exe",
            "windows_auto_pin": True,
            "windows_skip_ht_pair": True,
        }},
        log_dir=tmp_path / "logs",
        wrapper_resource_log_interval_s=3.0,
        node_cpu_fallback_block_reason=lambda task, node, info: "fallback missing",
        windows_env_spec_error=lambda task: None,
        ensure_launcher=lambda node: (True, r"F:\sched\launcher.py"),
        ensure_detacher=lambda node: (True, r"F:\sched\detacher.py"),
        sched_dir=lambda node: r"F:\sched",
        log_path=lambda task: rf"F:\sched\logs\{task['id']}.log",
        windows_path_for_node=lambda node, path: path if ":" in path else r"F:\repo",
        ps_quote=lambda value: "'" + value.replace("'", "''") + "'",
        run_windows_ps=run_windows_ps,
        apply_cpu_parallel_plan_to_task=lambda task, node_state: bool(task.get("cpu_plan")),
        cpu_parallel_env=lambda task: {"SCHEDULEURM_CPU_WORKERS": "4"},
        safe_extra_env_items=lambda env: list(env.items()),
        launch_extra_env=lambda task: {"EXTRA": "yes", "CUDA_VISIBLE_DEVICES": "bad"},
        windows_prepare_command=prepare_command or (lambda task: {
            "argv": [r"C:\Python\python.exe", "-u", r"F:\repo\train.py"],
            "env": {"FROM_CMD": "1"},
        }),
        remember_last_placement=lambda task: task.__setitem__("last_node", task.get("node")),
        set_current_usage=lambda task, vram, ram, pcpu: task.update({
            "current_vram_mb": vram,
            "current_ram_mb": ram,
            "current_pcpu": pcpu,
        }),
        now=lambda: 123.0,
        time_ns=lambda: 999,
        getpid=lambda: 77,
    )
    return deps, calls, payloads


def test_windows_backend_launch_writes_payload_and_marks_running(tmp_path):
    deps, calls, payloads = _launch_deps(tmp_path)
    task = {
        "id": "tWin",
        "node": "win",
        "cwd": "/repo",
        "est_vram_mb": 0,
        "cpu_cores": 4,
        "cpu_plan": True,
    }

    ok, msg = windows_backend_launch(task, node_state={"free_cpu": 8}, deps=deps)

    assert ok is True
    assert msg == "win_pid=4321"
    assert task["status"] == "running"
    assert task["remote_pids"] == [4321]
    assert task["process_group"] == 4321
    assert task["wrapper_pid_path"] == r"F:\sched\pids\tWin_999_77.pid"
    assert task["bootstrap_stdout_path"].endswith(".999_77.boot.out")
    assert task["started_at"] == 123.0
    assert payloads
    payload = payloads[0]
    assert payload["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert payload["env"]["SCHEDULEURM_TASK_ID"] == "tWin"
    assert payload["env"]["EXTRA"] == "yes"
    assert payload["env"]["FROM_CMD"] == "1"
    assert payload["env"]["SCHEDULEURM_CPU_WORKERS"] == "4"
    assert payload["resource_log_interval_s"] == 3.0
    assert len(calls) == 3
    assert r"F:\sched\payloads\tWin_999_77.b64" in calls[1][1]
    assert (tmp_path / "logs" / "tWin.winssh.log").read_text() == "PID=4321\n"


def test_windows_backend_launch_rejects_gpu_before_network(tmp_path):
    deps, calls, _payloads = _launch_deps(tmp_path)
    task = {"id": "tGpu", "node": "win", "est_vram_mb": 1}

    ok, msg = windows_backend_launch(task, node_state=None, deps=deps)

    assert ok is False
    assert "WindowsBackend is CPU-only" in msg
    assert calls == []


def test_windows_backend_launch_reports_cwd_missing(tmp_path):
    deps, _calls, _payloads = _launch_deps(tmp_path, ps_results=[(0, "MISSING\n", "")])
    task = {"id": "tCwd", "node": "win", "cwd": "/repo", "est_vram_mb": 0}

    ok, msg = windows_backend_launch(task, node_state=None, deps=deps)

    assert ok is False
    assert msg == r"cwd missing on win: F:\repo"


def test_windows_backend_batch_probe_uses_log_fallback_on_high_fanout():
    tasks = [
        ({"id": "done", "remote_pids": [1]}, [1]),
        ({"id": "fallback", "remote_pids": [2], "cpu_cores": 3}, [2]),
    ]

    deps = WindowsBatchProbeDeps(
        collect_windows_probe_targets=lambda state, node_is_windows, task_pids: {"win": tasks},
        node_is_windows=lambda node: node == "win",
        task_pids=lambda task: task.get("remote_pids") or [],
        windows_path_for_task=lambda task, path: path,
        success_patterns=["done"],
        windows_log_exit_probe_specs=lambda items, **kwargs: [{"id": item[0]["id"]} for item in items],
        windows_log_exit_probe_script=lambda quoted_specs: "log-probe " + quoted_specs,
        windows_log_exit_results_from_rows=lambda rows: {
            row["id"]: {"state": "dead", "terminal_ok": True}
            for row in rows if row["id"] == "done"
        },
        windows_process_probe_specs=lambda items, **kwargs: [{"root": pids[0]} for _task, pids in items],
        windows_process_probe_script=lambda roots, specs: "process-probe",
        normalize_windows_json_rows=lambda out, require_nonempty=False: json.loads(out),
        windows_queue_accounting_fallback_result=lambda task, pids, reason: {
            "state": "alive",
            "alive_pids": pids,
            "pcpu": float(task.get("cpu_cores", 1) * 100),
            "probe_fallback": reason,
        },
        windows_process_results_from_rows=lambda items, rows, **kwargs: {},
        local_launch_transport_alive=lambda task: False,
        ps_quote=lambda text: text,
        run_windows_ps=lambda node, script, **kwargs: (
            0,
            json.dumps([{"id": "done"}, {"id": "fallback"}]),
            "",
        ),
        environ={"SCHEDULEURM_WINDOWS_PROCESS_PROBE_HIGH_FANOUT": "1"},
    )

    out = windows_backend_batch_probe({"tasks": []}, deps=deps)

    assert out["done"] == {"state": "dead", "terminal_ok": True}
    assert out["fallback"]["state"] == "alive"
    assert out["fallback"]["alive_pids"] == [2]
    assert out["fallback"]["probe_fallback"] == "windows_queue_accounting"
