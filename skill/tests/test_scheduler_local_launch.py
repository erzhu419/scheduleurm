from __future__ import annotations

import subprocess
import copy
import shlex
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Lock

from skill.scheduler_local_launch import LocalLaunchDeps, local_backend_launch
from skill.scheduler_backend.local_launch import _probe_cwd


def _task(**overrides):
    task = {
        "id": "t1",
        "node": "nodeA",
        "cwd": "/work",
        "cmd": "python train.py",
        "gpu_idx": 0,
        "est_vram_mb": 1000,
        "cpu_cores": 1,
        "ram_mb": 1000,
        "remote_pids": [],
        "extra_env": {},
    }
    task.update(overrides)
    return task


def _deps(
    tmp_path: Path,
    *,
    calls=None,
    node_configs=None,
    run_results=None,
    claim_enabled=False,
    claim_results=None,
    docker_wrapper=None,
    required_gpu_idx=None,
    gpu_fits_result=True,
):
    calls = calls if calls is not None else []
    node_configs = node_configs or {"nodeA": {"host": "remote", "enable_claims": claim_enabled}}
    run_results = list(run_results or [])
    claim_results = list(claim_results or [])

    def remote_path(node, path):
        calls.append(("remote_path", node, path))
        return f"/remote{path}" if node == "nodeA" else path

    def run_on(node, cmd, **kwargs):
        calls.append(("run_on", node, cmd, kwargs))
        if run_results:
            result = run_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if "test -d" in cmd:
            return 0, "", ""
        if "docker inspect" in cmd:
            return 1, "", ""
        if "setsid bash" in cmd:
            return 0, "PID=4242\nSTART_TICKS=777\n", ""
        return 0, "", ""

    def claim(node, task, gpu_idx, node_state):
        calls.append(("claim", node, gpu_idx))
        if claim_results:
            return claim_results.pop(0)
        return True, {"task_id": task["id"], "gpu_idx": gpu_idx}, "ok"

    def update_claim_pid(node, task_id, pid):
        calls.append(("update_pid", node, task_id, pid))
        return True

    def release(task, **kwargs):
        calls.append(("release", task["id"], kwargs))

    def maybe_wrap(task, inner, remote_cwd):
        calls.append(("wrap", inner, remote_cwd))
        if docker_wrapper:
            return docker_wrapper(task, inner, remote_cwd)
        return inner, None

    return LocalLaunchDeps(
        state_dir=tmp_path,
        node_configs=node_configs,
        remote_path_for_node=remote_path,
        apply_node_cmd_rewrites=lambda node, cmd: cmd.replace("train.py", "rewritten.py"),
        rewrite_command_paths_for_node=lambda node, cmd: cmd.replace("/data", "/remote/data"),
        inject_python_u=lambda cmd: cmd.replace("python ", "python -u "),
        maybe_wrap_docker=maybe_wrap,
        run_on=run_on,
        claim_enabled_for=lambda node: claim_enabled,
        claim=claim,
        update_claim_pid=update_claim_pid,
        release_task_claims_and_intents=release,
        task_required_gpu_idx=lambda task: required_gpu_idx,
        gpu_fits=lambda task, gpu, node_info: gpu_fits_result,
        node_launch_extra_env=lambda task: task.get("extra_env") or {},
        safe_extra_env_items=lambda env: [(k, str(v)) for k, v in env.items() if k != "CUDA_VISIBLE_DEVICES"],
        remember_last_placement=lambda task: calls.append(("remember", task["id"], task.get("node"), task.get("gpu_idx"))),
        set_current_usage=lambda task, vram, ram, pcpu: task.update(
            {"current_vram_mb": vram, "current_ram_mb": ram, "current_pcpu": pcpu}
        ),
        now=lambda: 123.0,
        sleep=lambda seconds: calls.append(("sleep", seconds)),
    )


def test_happy_path_sets_runtime_fields_and_builds_launch_command(tmp_path: Path):
    calls = []
    task = _task(extra_env={"FOO": "bar", "CUDA_VISIBLE_DEVICES": "bad"})

    ok, msg = local_backend_launch(task, node_state={"gpus": []}, deps=_deps(tmp_path, calls=calls))

    assert ok is True
    assert msg == "pid=4242"
    assert task["status"] == "running"
    assert task["remote_pids"] == [4242]
    assert task["process_group"] == 4242
    assert task["remote_pid_start_ticks"] == {"4242": 777}
    assert task["exit_status_token"]
    assert task["exit_status_path"].startswith("/tmp/scheduleurm_exit_t1_")
    assert task["exit_status_path"].endswith(".status")
    assert task["log_path"] == "/tmp/sched_t1.log"
    assert task["started_at"] == 123.0
    assert task["peak_vram_mb"] == 0
    assert task["current_ram_mb"] == 0
    launch_cmd = next(call[2] for call in calls if call[0] == "run_on" and "setsid bash" in call[2])
    assert "export CUDA_VISIBLE_DEVICES=0;" in launch_cmd
    assert "export SCHEDULEURM_TASK_ID=t1;" in launch_cmd
    assert f"export SCHEDULEURM_LAUNCH_TOKEN={task['exit_status_token']};" in launch_cmd
    assert "export FOO=bar;" in launch_cmd
    assert "export CUDA_VISIBLE_DEVICES=bad;" not in launch_cmd
    assert "python -u rewritten.py" in launch_cmd
    assert "START_TICKS=" in launch_cmd
    assert task["exit_status_path"] in launch_cmd
    assert task["exit_status_token"] in launch_cmd
    assert "mv -f --" in launch_cmd
    assert ("remember", "t1", "nodeA", 0) in calls


def test_local_node_uses_state_dir_log_path(tmp_path: Path):
    task = _task()
    deps = _deps(tmp_path, node_configs={"nodeA": {"host": None}})

    ok, _ = local_backend_launch(task, node_state=None, deps=deps)

    assert ok is True
    assert task["log_path"] == f"{tmp_path}/logs/t1.log"


def test_launch_transport_replay_runs_once_and_preserves_terminal_evidence(tmp_path, monkeypatch):
    from skill.scheduler_backend import local_launch

    status_path = tmp_path / "exit.status"
    monkeypatch.setattr(local_launch, "runtime_exit_status_path", lambda *_: str(status_path))
    births = tmp_path / "executions"
    command = f"printf 'once\\n' >> {shlex.quote(str(births))}; sleep 0.2; printf 'canonical log\\n'"
    task = _task(cwd=str(tmp_path), cmd=command, launch_token="same-transport-attempt")

    def run_on(_node, command, *, timeout, check):
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr

    deps = replace(_deps(tmp_path, node_configs={"nodeA": {"host": None}}),
                   remote_path_for_node=lambda _node, path: path, run_on=run_on,
                   now=time.time)
    attempts = [copy.deepcopy(task), copy.deepcopy(task)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda attempt: local_backend_launch(attempt, node_state=None, deps=deps), attempts))
    assert all(ok for ok, _ in results)
    assert attempts[0]["remote_pids"] == attempts[1]["remote_pids"]
    deadline = time.monotonic() + 3
    while not status_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert status_path.exists()
    original_status = status_path.read_bytes()
    log_path = Path(attempts[0]["log_path"])
    original_log = log_path.read_bytes()
    assert original_status.split(b"\t")[0] == b"0"
    assert original_log == b"canonical log\n"
    assert births.read_text() == "once\n"

    after_exit = copy.deepcopy(task)
    assert local_backend_launch(after_exit, node_state=None, deps=deps)[0]
    assert after_exit["remote_pids"] == attempts[0]["remote_pids"]
    assert births.read_text() == "once\n"
    assert status_path.read_bytes() == original_status
    assert log_path.read_bytes() == original_log


def test_resume_flag_appends_remote_checkpoint_path(tmp_path: Path):
    calls = []
    task = _task(resume_from="/ckpt/latest.pt", resume_flag="--resume")

    ok, _ = local_backend_launch(task, node_state=None, deps=_deps(tmp_path, calls=calls))

    assert ok is True
    launch_cmd = next(call[2] for call in calls if call[0] == "run_on" and "setsid bash" in call[2])
    assert "--resume /remote/ckpt/latest.pt" in launch_cmd


def test_docker_wrapper_error_fails_before_cwd_probe(tmp_path: Path):
    calls = []

    def wrapper(task, inner, remote_cwd):
        return inner, "docker unavailable"

    ok, msg = local_backend_launch(
        _task(),
        node_state=None,
        deps=_deps(tmp_path, calls=calls, docker_wrapper=wrapper),
    )

    assert ok is False
    assert msg == "docker unavailable"
    assert not any(call[0] == "run_on" for call in calls)


def test_cwd_probe_retries_and_reports_transport_failure(tmp_path: Path):
    calls = []
    timeout = subprocess.TimeoutExpired(cmd="test -d", timeout=10)

    ok, msg = local_backend_launch(
        _task(),
        node_state=None,
        deps=_deps(
            tmp_path,
            calls=calls,
            run_results=[timeout, (1, "", "ssh down")],
        ),
    )

    assert ok is False
    assert msg.startswith("cwd probe failed on nodeA")
    assert "timeout after 10s" in msg
    assert "ssh down" in msg
    assert ("sleep", 0.5) in calls


def test_batch_shares_successful_cwd_probe_until_cache_expires(tmp_path):
    calls = []
    clock = [123.0]
    deps = replace(_deps(tmp_path, calls=calls), cwd_probe_cache={},
                   cwd_probe_locks=defaultdict(Lock), now=lambda: clock[0])
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: _probe_cwd(_task(), "/work", deps), range(30)))
    assert all(result == (True, "") for result in results)
    assert len([call for call in calls if call[0] == "run_on"]) == 1
    clock[0] += 61
    assert _probe_cwd(_task(), "/work", deps) == (True, "")
    assert len([call for call in calls if call[0] == "run_on"]) == 2


def test_failed_cwd_probe_is_not_cached(tmp_path):
    calls = []
    deps = replace(_deps(tmp_path, calls=calls, run_results=[(1, "", ""), (1, "", "")]),
                   cwd_probe_cache={}, cwd_probe_locks=defaultdict(Lock))
    assert _probe_cwd(_task(), "/work", deps)[0] is False
    assert deps.cwd_probe_cache == {}
    assert _probe_cwd(_task(), "/work", deps) == (True, "")
    assert len([call for call in calls if call[0] == "run_on"]) == 3


def test_claim_conflict_restores_original_gpu_and_returns_claim_race(tmp_path: Path):
    calls = []
    task = _task(gpu_idx=0)

    ok, msg = local_backend_launch(
        task,
        node_state={"gpus": []},
        deps=_deps(
            tmp_path,
            calls=calls,
            claim_enabled=True,
            claim_results=[(False, "busy", "conflict")],
        ),
    )

    assert ok is False
    assert msg == "CLAIM_RACE: gpu0: busy"
    assert task["gpu_idx"] == 0
    assert not any(call[0] == "run_on" and "setsid bash" in call[2] for call in calls)


def test_claim_error_returns_claim_error_without_launch(tmp_path: Path):
    task = _task()

    ok, msg = local_backend_launch(
        task,
        node_state={"gpus": []},
        deps=_deps(
            tmp_path,
            claim_enabled=True,
            claim_results=[(False, "ssh blip", "error")],
        ),
    )

    assert ok is False
    assert msg == "CLAIM_ERROR: ssh blip"
    assert task["gpu_idx"] == 0


def test_alternate_gpu_claim_changes_cuda_visible_devices(tmp_path: Path):
    calls = []
    task = _task(gpu_idx=0)

    ok, msg = local_backend_launch(
        task,
        node_state={
            "gpus": [
                {"idx": 0, "total_mb": 12000},
                {"idx": 1, "total_mb": 12000},
            ]
        },
        deps=_deps(
            tmp_path,
            calls=calls,
            claim_enabled=True,
            claim_results=[
                (False, "gpu0 busy", "conflict"),
                (True, {"task_id": "t1", "gpu_idx": 1}, "ok"),
            ],
        ),
    )

    assert ok is True
    assert msg == "pid=4242"
    assert task["gpu_idx"] == 1
    assert ("claim", "nodeA", 0) in calls
    assert ("claim", "nodeA", 1) in calls
    launch_cmd = next(call[2] for call in calls if call[0] == "run_on" and "setsid bash" in call[2])
    assert "export CUDA_VISIBLE_DEVICES=1;" in launch_cmd
    assert ("update_pid", "nodeA", "t1", 4242) in calls


def test_launch_failure_releases_claim(tmp_path: Path):
    calls = []
    task = _task()

    ok, msg = local_backend_launch(
        task,
        node_state={"gpus": []},
        deps=_deps(
            tmp_path,
            calls=calls,
            claim_enabled=True,
            run_results=[
                (0, "", ""),  # cwd probe
                (7, "", "launch failed"),  # setsid launch
            ],
        ),
    )

    assert ok is False
    assert msg == "launch rc=7: launch failed"
    assert ("release", "t1", {}) in calls


def test_docker_container_pid_replaces_launcher_pid_and_updates_claim(tmp_path: Path):
    calls = []

    def wrapper(task, inner, remote_cwd):
        task["container_name"] = "sched-t1"
        return f"docker run {inner}", None

    task = _task()
    ok, msg = local_backend_launch(
        task,
        node_state={"gpus": []},
        deps=_deps(
            tmp_path,
            calls=calls,
            claim_enabled=True,
            docker_wrapper=wrapper,
            run_results=[
                (0, "", ""),  # cwd
                (0, "PID=4242\n", ""),  # launch
                (0, "9999\n", ""),  # docker inspect
            ],
        ),
    )

    assert ok is True
    assert msg == "pid=4242 container=sched-t1@9999"
    assert task["process_group"] == 4242
    assert task["remote_pids"] == [9999]
    assert task["container_main_pid"] == 9999
    assert ("update_pid", "nodeA", "t1", 4242) in calls
    assert ("update_pid", "nodeA", "t1", 9999) in calls
