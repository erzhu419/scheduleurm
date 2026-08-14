from __future__ import annotations

from skill.scheduler_task_identity import (
    TaskIdentityDeps,
    active_or_unsynced_result_status,
    legacy_external_state_is_terminal,
    recorded_launch_safety_state,
    same_declared_path,
    same_run_identity_live_artifact_reason,
    slurm_state_is_terminal,
    task_has_recorded_launch_artifacts,
    task_run_identity,
)


class _Backend:
    def __init__(self, result):
        self.result = result

    def batch_probe(self, state):
        task_id = state["tasks"][0]["id"]
        return {task_id: self.result}


def _deps(result=None, pids=None, transport_alive=False):
    return TaskIdentityDeps(
        task_pids=lambda task: pids if pids is not None else task.get("remote_pids", []),
        local_launch_transport_alive=lambda task: transport_alive,
        backend=_Backend(result or {"state": "dead"}),
    )


def _base_task(**overrides):
    task = {
        "id": "t1",
        "signature": "BAPR/demo",
        "cmd": "python train.py",
        "cwd": "/work/../work",
        "extra_env": {"B": "2", "A": "1"},
        "ckpt_dir": "/tmp/ckpt/",
        "result_dir": "/tmp/result/",
    }
    task.update(overrides)
    return task


def test_task_run_identity_is_exact_but_ignores_resource_knobs():
    first = _base_task(cpu_cores=1, priority="low")
    second = _base_task(cpu_cores=64, priority="high")
    deprecated_backend_knobs = _base_task(
        slurm_partition="gpu",
        slurm_account="legacy",
        slurm_qos="legacy",
    )

    assert task_run_identity(first) == task_run_identity(second)
    assert task_run_identity(first) == task_run_identity(deprecated_backend_knobs)
    assert task_run_identity(dict(first, cmd="python eval.py")) != task_run_identity(first)
    assert task_run_identity({"cmd": "python train.py"}) is None
    assert same_declared_path("/tmp/result/", "/tmp/result")


def test_active_or_unsynced_result_status_tracks_writers_and_pending_sync():
    assert active_or_unsynced_result_status({"status": "queued"})
    assert active_or_unsynced_result_status({"status": "done", "result_dir": "/r"})
    assert not active_or_unsynced_result_status(
        {"status": "done", "result_dir": "/r", "result_synced_at": 1}
    )


def test_recorded_launch_safety_state_fails_closed_on_unknown_probe():
    task = _base_task(id="t2", status="done", node="node001", remote_pids=[123])

    assert task_has_recorded_launch_artifacts(task, deps=_deps())
    assert recorded_launch_safety_state(
        task,
        deps=_deps(result={"state": "unknown", "error": "ssh timeout"}),
    ) == ("unknown", "ssh timeout")
    assert recorded_launch_safety_state(
        task,
        deps=_deps(result={"state": "alive"}),
    ) == ("alive", "backend probe reports recorded pid/job alive")


def test_recorded_launch_safety_state_treats_terminal_slurm_state_as_dead():
    assert legacy_external_state_is_terminal("FAILED")
    assert slurm_state_is_terminal("CANCELLED by 123")
    state, reason = recorded_launch_safety_state(
        {"id": "t3", "status": "failed", "finished_at": 1, "slurm_job_id": "99", "slurm_state": "FAILED"},
        deps=_deps(),
    )

    assert state == "dead"
    assert "slurm_state=FAILED" in reason


def test_same_run_identity_live_artifact_reason_ignores_ancestor_artifacts():
    parent = _base_task(id="parent", status="failed", remote_pids=[11], node="node001")
    child = _base_task(id="child", status="queued", parent_id="parent")
    state = {"tasks": [parent, child]}

    assert same_run_identity_live_artifact_reason(child, state, deps=_deps(result={"state": "alive"})) is None

    sibling = _base_task(id="sibling", status="failed", remote_pids=[12], node="node001")
    state["tasks"].append(sibling)
    reason = same_run_identity_live_artifact_reason(child, state, deps=_deps(result={"state": "alive"}))

    assert "terminal task sibling" in reason
