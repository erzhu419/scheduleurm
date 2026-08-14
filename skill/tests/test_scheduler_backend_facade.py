from __future__ import annotations

import os
import signal
import subprocess

import pytest

from skill.scheduler_backend_facade import (
    Backend,
    LegacyExternalBackend,
)


def test_backend_base_requires_local_capacity_check_by_default():
    backend = Backend()

    assert backend.requires_local_capacity_check("node001") is True
    with pytest.raises(NotImplementedError):
        backend.launch({"id": "t1"})
    with pytest.raises(NotImplementedError):
        backend.kill({"id": "t1"})
    with pytest.raises(NotImplementedError):
        backend.batch_probe({"tasks": []})


def test_legacy_external_backend_refuses_launch_and_kill_and_reports_legacy_probe_unknown():
    backend = LegacyExternalBackend(lambda: "Legacy external scheduler records are read-only")
    running = {"id": "old", "status": "running", "slurm_job_id": 123}
    ignored = {"id": "new", "status": "queued", "slurm_job_id": 456}

    assert backend.name == "legacy-external"
    assert backend.requires_local_capacity_check("node001") is True
    assert backend.launch({"id": "new"}) == (
        False,
        "Legacy external scheduler records are read-only",
    )
    assert backend.kill(running) == (
        False,
        "Legacy external scheduler records are read-only",
    )

    result = backend.batch_probe({"tasks": [running, ignored, {"id": "local", "status": "running"}]})

    assert set(result) == {"old"}
    assert result["old"]["state"] == "unknown"
    assert result["old"]["alive_pids"] == []
    assert result["old"]["vram_mb"] == 0
    assert result["old"]["ram_mb"] == 0
    assert result["old"]["pcpu"] == 0.0
    assert result["old"]["error"] == "Legacy external scheduler records are read-only"


def test_local_backend_kill_many_uses_one_remote_command_per_node(sch, monkeypatch):
    calls = []

    def run_on(node, cmd, **kwargs):
        calls.append((node, cmd, kwargs))
        return 0, "", ""

    monkeypatch.setattr(sch, "run_on", run_on)
    backend = sch.LocalBackend()
    tasks = [
        {
            "id": "t1",
            "node": "node001",
            "remote_pids": [101],
            "process_group": 101,
        },
        {
            "id": "t2",
            "node": "node001",
            "remote_pids": [202],
            "process_group": 202,
            "container_name": "sched-t2",
        },
        {"id": "t3", "node": "node001", "remote_pids": []},
    ]

    results = backend.kill_many(tasks, timeout=9)

    assert len(calls) == 1
    node, command, kwargs = calls[0]
    assert node == "node001"
    assert command.count("sleep 1") == 2
    assert "kill -- -101" in command
    assert "kill -- -202" in command
    assert "kill -0 -- -101" in command
    assert "kill -0 -- -202" in command
    assert "docker stop -t 5 sched-t2" in command
    assert kwargs == {"timeout": 9, "check": False}
    assert results["t1"][0] is True
    assert results["t2"][0] is True
    assert results["t3"] == (False, "no pids")


def test_local_backend_kill_many_chunks_large_node_batches(sch, monkeypatch):
    calls = []
    monkeypatch.setattr(
        sch,
        "run_on",
        lambda node, cmd, **kwargs: calls.append((node, cmd, kwargs)) or (0, "", ""),
    )
    backend = sch.LocalBackend()
    tasks = [
        {
            "id": f"t{index}",
            "node": "node001",
            "remote_pids": [1000 + index],
            "process_group": 1000 + index,
        }
        for index in range(300)
    ]

    results = backend.kill_many(tasks)

    assert len(calls) == 2
    assert len(results) == 300
    assert all(result[0] is True for result in results.values())


@pytest.mark.parametrize("returncode", [75, 255])
def test_local_backend_kill_many_fails_closed_on_survivor_or_transport_error(
    sch,
    monkeypatch,
    returncode,
):
    monkeypatch.setattr(
        sch,
        "run_on",
        lambda *args, **kwargs: (
            returncode,
            "SCHEDULEURM_KILL_SURVIVOR=pgid:101" if returncode == 75 else "",
            "connection refused" if returncode == 255 else "",
        ),
    )
    backend = sch.LocalBackend()
    results = backend.kill_many([{
        "id": "t1",
        "node": "node001",
        "remote_pids": [101],
        "process_group": 101,
    }])

    ok, message = results["t1"]
    assert ok is False
    assert ("surviving process identities" if returncode == 75 else "rc=255") in message


def test_local_backend_kill_terminates_and_verifies_real_process_group(sch, monkeypatch):
    launched = subprocess.Popen(
        ["bash", "-c", "sleep 300 & wait"],
        start_new_session=True,
    )

    def run_on(_node, command, **_kwargs):
        completed = subprocess.run(
            ["bash", "-c", command],
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr

    monkeypatch.setattr(sch, "run_on", run_on)
    backend = sch.LocalBackend()
    try:
        ok, message = backend.kill({
            "id": "t-real",
            "node": "local",
            "remote_pids": [launched.pid],
            "process_group": launched.pid,
        })
        launched.wait(timeout=5)
        assert ok is True, message
        assert launched.poll() is not None
    finally:
        try:
            os.killpg(launched.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_scheduler_backend_runtime_deps_are_built_from_scheduler_namespace(sch):
    windows = sch._windows_backend_runtime_deps()
    assert windows.node_configs is sch.NODES
    assert windows.log_dir is sch.LOG_DIR
    assert windows.launcher_text is sch._WINDOWS_LAUNCHER
    assert windows.detacher_text is sch._WINDOWS_DETACHER
    assert windows.ssh_base_args is sch._ssh_base_args
    assert windows.run_subprocess is sch.subprocess.run
    assert windows.node_cpu_fallback_block_reason is sch._node_cpu_fallback_block_reason
    assert windows.windows_env_spec_error is sch._windows_env_spec_error
    assert windows.windows_path_for_node is sch._windows_path_for_node
    assert windows.ps_quote is sch._ps_quote
    assert windows.run_windows_ps is sch._run_windows_ps
    assert windows.apply_cpu_parallel_plan_to_task is sch._apply_cpu_parallel_plan_to_task
    assert windows.cpu_parallel_env is sch._cpu_parallel_env
    assert windows.safe_extra_env_items is sch._safe_extra_env_items
    assert windows.launch_extra_env is sch._launch_extra_env
    assert windows.windows_prepare_command is sch._windows_prepare_command
    assert windows.remember_last_placement is sch._remember_last_placement
    assert windows.set_current_usage is sch._set_current_usage
    assert windows.collect_windows_probe_targets is sch._collect_windows_probe_targets
    assert windows.node_is_windows is sch._node_is_windows
    assert windows.task_pids is sch._task_pids
    assert windows.windows_path_for_task is sch._windows_path_for_task
    assert windows.success_patterns is sch.SUCCESS_PATTERNS
    assert windows.now is sch.time.time
    assert windows.time_ns is sch.time.time_ns
    assert windows.getpid is sch.os.getpid
    assert windows.environ is sch.os.environ

    local = sch._local_backend_runtime_deps()
    assert local.state_dir is sch.STATE_DIR
    assert local.node_configs is sch.NODES
    assert local.remote_path_for_node is sch._remote_path_for_node
    assert local.apply_node_cmd_rewrites is sch._apply_node_cmd_rewrites
    assert local.rewrite_command_paths_for_node is sch._rewrite_command_paths_for_node
    assert local.inject_python_u is sch._inject_python_u
    assert local.maybe_wrap_docker is sch._maybe_wrap_docker
    assert local.run_on is sch.run_on
    assert local.release_task_claims_and_intents is sch._release_task_claims_and_intents
    assert local.task_required_gpu_idx is sch._task_required_gpu_idx
    assert local.gpu_fits is sch._gpu_fits
    assert local.node_launch_extra_env is sch._node_launch_extra_env
    assert local.safe_extra_env_items is sch._safe_extra_env_items
    assert local.remember_last_placement is sch._remember_last_placement
    assert local.set_current_usage is sch._set_current_usage
    assert local.task_pids is sch._task_pids
    assert local.task_process_groups is sch._task_process_groups
    assert local.descendants_of is sch._descendants_of
    assert local.now is sch.time.time
    assert local.sleep is sch.time.sleep

    hybrid = sch._hybrid_backend_deps()
    assert hybrid.node_is_windows is sch._node_is_windows
    assert hybrid.local_backend_factory is sch.LocalBackend
    assert hybrid.legacy_backend_factory is sch.LegacyExternalBackend
    assert hybrid.windows_backend_factory is sch.WindowsBackend
