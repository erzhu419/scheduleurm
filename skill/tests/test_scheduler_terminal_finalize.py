from __future__ import annotations

from pathlib import Path

from skill.scheduler_failure.terminal_finalize import TerminalFinalizeDeps, try_finalize_terminal_local_task


def _task(task_id="t1", **overrides):
    task = {
        "id": task_id,
        "status": "launching",
        "node": "localnode",
        "launching_started_at": 90.0,
        "launch_token": "old-token",
        "remote_pids": [123],
        "alive_pids": [123],
        "peak_vram_mb": 10,
        "peak_ram_mb": 20,
        "cmd": "python train.py",
        "retry_count": 0,
        "signature": f"sig/{task_id}",
    }
    task.update(overrides)
    return task


def _deps(
    tmp_path: Path,
    *,
    windows=False,
    local=True,
    diag=None,
    requeue_id=None,
    claim_enabled=True,
    remote_sizes=None,
    calls=None,
):
    calls = calls if calls is not None else []
    diag = diag if diag is not None else {"is_crash": False, "reason": "Training complete"}
    remote_sizes = remote_sizes or {}

    def run_on(node, cmd, **kwargs):
        calls.append(("run_on", node, cmd, kwargs))
        for path, size in remote_sizes.items():
            if path in cmd:
                return 0, str(size), ""
        return 1, "", "missing"

    def diagnose(task):
        calls.append(("diagnose", task["id"], task.get("log_path")))
        return dict(diag)

    def requeue(task, state):
        calls.append(("requeue", task["id"]))
        if requeue_id:
            state["tasks"].append({"id": requeue_id, "status": "queued"})
        return requeue_id

    def record(task):
        calls.append(("record", task["id"]))
        return []

    def release(task, **kwargs):
        calls.append(("release", task["id"], kwargs))

    def set_usage(task, vram, ram, pcpu):
        calls.append(("usage", task["id"], vram, ram, pcpu))
        task["current_vram_mb"] = vram
        task["current_ram_mb"] = ram
        task["current_pcpu"] = pcpu

    return TerminalFinalizeDeps(
        state_dir=tmp_path,
        node_is_windows=lambda node: windows,
        is_local_node=lambda node: local,
        run_on=run_on,
        diagnose_terminal=diagnose,
        requeue_after_crash=requeue,
        record_result_artifacts=record,
        claim_enabled_for=lambda node: claim_enabled,
        release_task_claims_and_intents=release,
        set_current_usage=set_usage,
        now=lambda: 100.0,
    )


def test_local_wrapper_log_clean_finalizes_done_and_releases_claim(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "t1.log").write_text("Training complete\n")
    task = _task("t1")
    state = {"tasks": [task]}
    calls = []

    assert try_finalize_terminal_local_task(task, "localnode", state, deps=_deps(tmp_path, calls=calls)) is True

    assert task["status"] == "done"
    assert task["log_path"] == f"{tmp_path}/logs/t1.log"
    assert task["started_at"] == 90.0
    assert task["finished_at"] == 100.0
    assert task["remote_pids"] == []
    assert task["alive_pids"] == []
    assert task["peak_vram_mb"] == 0
    assert task["peak_ram_mb"] == 0
    assert task["current_vram_mb"] == 0
    assert "launching_started_at" not in task
    assert "launch_token" not in task
    assert ("record", "t1") in calls
    assert ("release", "t1", {"extra_nodes": ["localnode"]}) in calls


def test_local_crash_finalizes_failed_and_links_retry(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "t2.log").write_text("Traceback\nAssertionError\n")
    task = _task("t2")
    state = {"tasks": [task]}

    ok = try_finalize_terminal_local_task(
        task,
        "localnode",
        state,
        deps=_deps(
            tmp_path,
            diag={"is_crash": True, "reason": "AssertionError"},
            requeue_id="t-retry",
        ),
    )

    assert ok is True
    assert task["status"] == "failed"
    assert task["requeued_as"] == "t-retry"
    assert state["tasks"][-1] == {"id": "t-retry", "status": "queued"}


def test_empty_wrapper_with_user_redirect_content_finalizes(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "t3.log").write_text("")
    real_log = log_dir / "real_t3.log"
    real_log.write_text("Training complete\n")
    task = _task("t3", cmd=f"python train.py > {real_log} 2>&1")
    state = {"tasks": [task]}

    assert try_finalize_terminal_local_task(task, "localnode", state, deps=_deps(tmp_path)) is True
    assert task["status"] == "done"


def test_empty_wrapper_without_redirect_evidence_returns_false(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "t4.log").write_text("")
    task = _task("t4", cmd="python train.py")
    state = {"tasks": [task]}

    assert try_finalize_terminal_local_task(task, "localnode", state, deps=_deps(tmp_path)) is False
    assert task["status"] == "launching"


def test_empty_wrapper_with_empty_redirect_returns_false(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "t5.log").write_text("")
    real_log = log_dir / "real_t5.log"
    real_log.write_text("")
    task = _task("t5", cmd=f"python train.py > {real_log} 2>&1")
    state = {"tasks": [task]}

    assert try_finalize_terminal_local_task(task, "localnode", state, deps=_deps(tmp_path)) is False
    assert task["status"] == "launching"


def test_remote_node_probes_tmp_sched_log(tmp_path: Path):
    task = _task("t6")
    state = {"tasks": [task]}
    calls = []

    assert try_finalize_terminal_local_task(
        task,
        "node001",
        state,
        deps=_deps(
            tmp_path,
            local=False,
            remote_sizes={"/tmp/sched_t6.log": 42},
            calls=calls,
        ),
    ) is True

    assert task["status"] == "done"
    assert task["log_path"] == "/tmp/sched_t6.log"
    assert any(call[0] == "run_on" and "wc -c < /tmp/sched_t6.log" in call[2] for call in calls)


def test_windows_or_missing_task_id_are_not_finalized(tmp_path: Path):
    state = {"tasks": []}
    assert try_finalize_terminal_local_task(_task("tw"), "win", state, deps=_deps(tmp_path, windows=True)) is False
    assert try_finalize_terminal_local_task(_task("", id=""), "localnode", state, deps=_deps(tmp_path)) is False
