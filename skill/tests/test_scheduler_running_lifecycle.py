from skill.scheduler_running.lifecycle import (
    RunningLifecycleDeps,
    apply_alive_probe_result,
    check_running_probe_result,
    handle_dead_probe_result,
    handle_unknown_probe_result,
)


def _deps(**overrides):
    calls = overrides.pop("calls", {})

    def set_current_usage(task, vram, ram, pcpu):
        task["current_vram_mb"] = vram
        task["current_ram_mb"] = ram
        task["current_pcpu"] = pcpu

    def mark_probe_unknown(task, res=None):
        task.setdefault("probe_unknown_since", 90.0)
        task["last_probe_unknown_reason"] = (res or {}).get("error") or "unknown"

    def release(task):
        calls.setdefault("release", []).append(task.get("id"))

    def record_result(task):
        calls.setdefault("record_result", []).append(task.get("id"))

    def requeue(task, state):
        calls.setdefault("requeue", []).append(task.get("id"))
        return overrides.get("requeue_id", "t-retry")

    def history_record(signature, **kwargs):
        calls.setdefault("history", []).append((signature, kwargs))

    def runtime_history_record(task, **kwargs):
        calls.setdefault("runtime", []).append((task.get("id"), kwargs))

    def apply_artifacts(task, artifacts):
        calls.setdefault("artifacts", []).append((task.get("id"), artifacts))

    diagnose_terminal = overrides.get("diagnose_terminal")
    if diagnose_terminal is None:
        diagnose_terminal = lambda task: overrides.get("diagnosis", {
            "is_crash": False,
            "reason": "normal exit",
            "tail": "",
            "success_marker": None,
        })

    base = RunningLifecycleDeps(
        set_current_usage=set_current_usage,
        mark_probe_unknown=mark_probe_unknown,
        clear_probe_unknown=lambda task: None,
        annotate_diag_after_unknown=lambda diag, sync: diag,
        cached_task_success_marker=lambda task: overrides.get("cached_success"),
        diagnose_terminal=diagnose_terminal,
        release_task_claims_and_intents=release,
        record_result_artifacts=record_result,
        requeue_after_crash=requeue,
        local_launch_transport_alive=lambda task: overrides.get("transport_alive", False),
        terminal_diagnostic_matches=lambda task, diag: bool(diag),
        scan_completed_log_for_crash=lambda task: overrides.get("completed_crash", (False, "")),
        mark_user_cancelled=lambda task, reason: task.update(status="cancelled", last_block_reason=reason),
        history_get=lambda sig: overrides.get("history_get", {}),
        untrusted_startup_oom_sample=lambda task, duration_s: overrides.get("startup_oom", False),
        history_record=history_record,
        runtime_history_record=runtime_history_record,
        apply_discovered_result_artifacts=apply_artifacts,
        remember_last_placement=lambda task: calls.setdefault("remember", []).append(task.get("id")),
        declared_cpu_slot_floor=lambda task: overrides.get("cpu_floor", 0),
        default_ram_mb=2048,
        default_cpu_cores=1,
        node_down_requeue_s=1,
        early_death_seconds=120,
        now=lambda: overrides.get("now", 100.0),
    )
    return base, calls


def test_apply_alive_probe_result_tracks_usage_and_resources():
    deps, calls = _deps()
    task = {
        "id": "t-alive",
        "ram_mb": 1000,
        "cpu_cores": 1,
        "peak_vram_mb": 128,
        "peak_ram_mb": 900,
    }
    apply_alive_probe_result(
        task,
        {"alive_pids": [11, 12], "vram_mb": 512, "ram_mb": 1500, "pcpu": 250.0},
        {"duration_s": 7},
        deps=deps,
    )

    assert calls["remember"] == ["t-alive"]
    assert task["alive_pids"] == [11, 12]
    assert task["current_vram_mb"] == 512
    assert task["peak_vram_mb"] == 512
    assert task["peak_ram_mb"] == 1500
    assert task["ram_mb"] == 1500
    assert task["cpu_cores"] == 3
    assert task["last_status_sync_status"] == "running"


def test_check_running_probe_result_interprets_backend_states():
    calls = []

    def set_current_usage(task, vram, ram, pcpu):
        calls.append((task["id"], vram, ram, pcpu))
        task["current_vram_mb"] = vram
        task["current_ram_mb"] = ram
        task["current_pcpu"] = pcpu

    assert check_running_probe_result(
        {"id": "missing"},
        None,
        set_current_usage=set_current_usage,
    ) == "dead"
    assert check_running_probe_result(
        {"id": "unknown"},
        {"state": "unknown"},
        set_current_usage=set_current_usage,
    ) == "unknown"
    dead = {"id": "dead"}
    assert check_running_probe_result(
        dead,
        {"state": "dead"},
        set_current_usage=set_current_usage,
    ) == "dead"
    assert calls == [("dead", 0, 0, 0.0)]

    alive = {"id": "alive", "peak_vram_mb": 100, "peak_ram_mb": 200}
    assert check_running_probe_result(
        alive,
        {
            "state": "alive",
            "alive_pids": [1, 2],
            "vram_mb": 512,
            "ram_mb": 1024,
            "pcpu": 250.0,
        },
        set_current_usage=set_current_usage,
    ) == "alive"
    assert alive["alive_pids"] == [1, 2]
    assert alive["peak_vram_mb"] == 512
    assert alive["peak_ram_mb"] == 1024
    assert alive["current_pcpu"] == 250.0


def test_handle_unknown_probe_result_marks_done_when_cached_success_exists():
    deps, calls = _deps(cached_success="Training complete", now=100.0)
    task = {
        "id": "t-unknown",
        "status": "running",
        "node": "node001",
        "reroute_on_node_down": True,
        "probe_unknown_since": 90.0,
        "started_at": 40.0,
        "log_path": "/tmp/log",
    }
    handle_unknown_probe_result(task, {"tasks": [task]}, {"error": "ssh failed"}, deps=deps)

    assert task["status"] == "done"
    assert task["finished_at"] == 100.0
    assert task["_diagnosis"]["success_marker"] == "Training complete"
    assert calls["release"] == ["t-unknown"]
    assert calls["record_result"] == ["t-unknown"]


def test_handle_unknown_probe_result_reuses_inconclusive_batch_probe():
    deps, calls = _deps(now=100.0)
    task = {
        "id": "t-unknown",
        "status": "running",
        "node": "node001",
        "reroute_on_node_down": True,
        "probe_unknown_since": 90.0,
        "started_at": 40.0,
        "log_path": "/tmp/log",
    }

    handle_unknown_probe_result(
        task,
        {"tasks": [task]},
        {"state": "unknown", "error": "ssh failed"},
        {},
        deps=deps,
    )

    assert task["status"] == "running"
    assert "recorded launch artifact is unknown" in task["last_block_reason"]
    assert "requeue" not in calls


def test_handle_unknown_probe_result_defers_missing_lock_free_diagnosis():
    def should_not_run(task):
        raise AssertionError("diagnose_terminal should stay outside the state lock")

    deps, calls = _deps(diagnose_terminal=should_not_run, now=100.0)
    task = {
        "id": "t-unknown",
        "status": "running",
        "node": "node001",
        "reroute_on_node_down": True,
        "probe_unknown_since": 90.0,
        "started_at": 40.0,
        "log_path": "/tmp/log",
    }

    handle_unknown_probe_result(
        task,
        {"tasks": [task]},
        {
            "state": "unknown",
            "error": "ssh failed",
            "launch_safety_state": "dead",
        },
        {},
        deps=deps,
    )

    assert task["status"] == "running"
    assert task["unknown_terminal_diagnosis_deferred"] is True
    assert "outside state lock" in task["last_block_reason"]
    assert "requeue" not in calls


def test_handle_unknown_probe_result_uses_precomputed_terminal_diagnosis():
    def should_not_run(task):
        raise AssertionError("diagnose_terminal should have been precomputed")

    deps, calls = _deps(diagnose_terminal=should_not_run, now=100.0)
    task = {
        "id": "t-unknown",
        "status": "running",
        "node": "node001",
        "reroute_on_node_down": True,
        "probe_unknown_since": 90.0,
        "started_at": 40.0,
        "log_path": "/tmp/log",
    }
    precomputed = {
        "diagnosis": {
            "is_crash": False,
            "reason": "normal exit",
            "tail": "DONE",
            "success_marker": "DONE",
        },
        "result_artifacts": [{"path": "/tmp/result.json"}],
    }

    handle_unknown_probe_result(
        task,
        {"tasks": [task]},
        {
            "state": "unknown",
            "error": "ssh failed",
            "launch_safety_state": "dead",
        },
        {"t-unknown": precomputed},
        deps=deps,
    )

    assert task["status"] == "done"
    assert task["_diagnosis"]["success_marker"] == "DONE"
    assert calls["artifacts"] == [("t-unknown", precomputed["result_artifacts"])]


def test_handle_dead_probe_result_legacy_external_completed_records_history_and_runtime():
    deps, calls = _deps(now=100.0)
    task = {
        "id": "t-done",
        "status": "running",
        "signature": "sig",
        "started_at": 80.0,
        "peak_vram_mb": 256,
        "peak_ram_mb": 1024,
        "cpu_cores": 2,
    }
    handle_dead_probe_result(
        task,
        {"tasks": [task]},
        {
            "state": "dead",
            "terminal_ok": True,
            "backend_state": "COMPLETED",
            "exit_code": 0,
            "backend_finished_at": 99.5,
        },
        None,
        None,
        deps=deps,
    )

    assert task["status"] == "done"
    assert task["exit_code"] == 0
    assert task["backend_finished_at"] == 99.5
    assert task["_diagnosis"]["success_marker"] == "LEGACY_EXTERNAL_COMPLETED"
    assert calls["release"] == ["t-done"]
    assert calls["history"][0][0] == "sig"
    history_kwargs = calls["history"][0][1]
    assert history_kwargs["task"] is task
    assert {key: value for key, value in history_kwargs.items() if key != "task"} == {
        "peak_vram_mb": 256,
        "peak_ram_mb": 1024,
        "cpu_cores": 2,
        "duration_s": 19,
    }
    assert calls["runtime"] == [("t-done", {"duration_s": 19})]


def test_handle_dead_probe_result_completed_log_crash_requeues():
    deps, calls = _deps(completed_crash=(True, "ModuleNotFoundError"), now=100.0)
    task = {
        "id": "t-crash",
        "status": "running",
        "signature": "sig",
        "started_at": 80.0,
        "cmd": "python train.py",
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "cpu_cores": 1,
    }
    handle_dead_probe_result(
        task,
        {"tasks": [task]},
        {"state": "dead", "terminal_ok": True, "backend_state": "COMPLETED"},
        None,
        None,
        deps=deps,
    )

    assert task["status"] == "failed"
    assert task["last_block_reason"] == "ModuleNotFoundError"
    assert task["requeued_as"] == "t-retry"
    assert calls["requeue"] == ["t-crash"]


def test_handle_dead_probe_result_deferred_local_pid_dead_avoids_inline_log_diagnosis():
    def should_not_run(task):
        raise AssertionError("diagnose_terminal should stay outside watch main lock")

    deps, calls = _deps(diagnose_terminal=should_not_run, now=201.0)
    task = {
        "id": "t-local-dead",
        "status": "running",
        "signature": "sig",
        "started_at": 1.0,
        "log_path": "/tmp/sched_t-local-dead.log",
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "cpu_cores": 1,
    }

    handle_dead_probe_result(
        task,
        {"tasks": [task]},
        {
            "state": "dead",
            "backend_state": "LOCAL_PID_DEAD",
            "terminal_reason": "local backend pid probe found no live tracked process",
            "terminal_diagnosis_deferred": True,
            "exit_code": 0,
            "backend_finished_at": 199.5,
        },
        None,
        None,
        deps=deps,
    )

    assert task["status"] == "running"
    assert task["exit_code"] == 0
    assert task["backend_finished_at"] == 199.5
    assert task["terminal_transition_deferred"] is True
    assert "diagnosis pending" in task["last_block_reason"]
    assert "release" not in calls
    assert "record_result" not in calls
    assert "history" not in calls


def test_handle_dead_probe_result_cached_success_avoids_refresh_deferral_livelock():
    def should_not_run(task):
        raise AssertionError("diagnose_terminal should stay outside the state lock")

    deps, calls = _deps(
        cached_success="DONE",
        diagnose_terminal=should_not_run,
        now=201.0,
    )
    task = {
        "id": "t-local-done",
        "status": "running",
        "signature": "sig",
        "started_at": 1.0,
        "log_path": "/tmp/sched_t-local-done.log",
        "last_progress_line": "DONE",
        "last_block_reason": (
            "process exited; terminal log/result diagnosis pending outside state lock"
        ),
        "result_dir": "/tmp/result",
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "cpu_cores": 1,
    }

    handle_dead_probe_result(
        task,
        {"tasks": [task]},
        {
            "state": "dead",
            "backend_state": "LOCAL_PID_DEAD",
            "terminal_reason": "local backend pid probe found no live tracked process",
            "terminal_diagnosis_deferred": True,
        },
        None,
        None,
        deps=deps,
    )

    assert task["status"] == "done"
    assert task["_diagnosis"]["success_marker"] == "DONE"
    assert "terminal_transition_deferred" not in task
    assert "last_block_reason" not in task
    assert calls["release"] == ["t-local-done"]
    assert calls["record_result"] == ["t-local-done"]


def test_handle_dead_probe_result_honors_pending_batch_cancel_without_retry():
    deps, calls = _deps(now=201.0)
    task = {
        "id": "t-cancel-pending",
        "status": "running",
        "signature": "sig",
        "started_at": 1.0,
        "log_path": "/tmp/sched_t-cancel-pending.log",
        "cancel_request_token": "request-1",
        "cancel_requested_at": 200.0,
        "cancel_request_reason": "user batch force-cancel",
        "cancel_request_actor": {"label": "operator", "action": "user batch force-cancel"},
    }

    handle_dead_probe_result(
        task,
        {"tasks": [task]},
        {
            "state": "dead",
            "backend_state": "LOCAL_PID_DEAD",
            "terminal_diagnosis_deferred": True,
        },
        None,
        None,
        deps=deps,
    )

    assert task["status"] == "cancelled"
    assert task["cancelled_by"] == "operator"
    assert task["last_cancel_request_token"] == "request-1"
    assert "cancel_request_token" not in task
    assert calls["release"] == ["t-cancel-pending"]
    assert "requeue" not in calls


def test_handle_dead_probe_result_success_marker_beats_short_runtime_history():
    deps, calls = _deps(
        now=201.0,
        diagnosis={
            "is_crash": False,
            "reason": "normal exit (success marker found)",
            "tail": "DONE",
            "success_marker": "DONE",
        },
        history_get={"dur_s_ewma": 1000, "dur_s_runs": 4},
    )
    task = {
        "id": "t-fast-success",
        "status": "running",
        "signature": "sig",
        "started_at": 1.0,
        "log_path": "/tmp/sched_t-fast-success.log",
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "cpu_cores": 1,
    }

    handle_dead_probe_result(
        task,
        {"tasks": [task]},
        {"state": "dead"},
        None,
        None,
        deps=deps,
    )

    assert task["status"] == "done"
    assert "requeue" not in calls
    assert calls["history"][0][1]["duration_s"] == 200


def test_handle_dead_probe_result_deferred_terminal_diagnosis_skips_short_runtime_failure():
    def should_not_run(task):
        raise AssertionError("diagnose_terminal should stay outside watch main lock")

    deps, calls = _deps(
        diagnose_terminal=should_not_run,
        now=201.0,
        history_get={"dur_s_ewma": 1000, "dur_s_runs": 4},
    )
    task = {
        "id": "t-deferred-fast",
        "status": "running",
        "signature": "sig",
        "started_at": 1.0,
        "log_path": "/tmp/sched_t-deferred-fast.log",
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "cpu_cores": 1,
    }

    handle_dead_probe_result(
        task,
        {"tasks": [task]},
        {
            "state": "dead",
            "backend_state": "LOCAL_PID_DEAD",
            "terminal_reason": "local backend pid probe found no live tracked process",
            "terminal_diagnosis_deferred": True,
        },
        None,
        None,
        deps=deps,
    )

    assert task["status"] == "running"
    assert task["terminal_transition_deferred"] is True
    assert "requeue" not in calls


def test_apply_alive_probe_result_expires_abandoned_batch_cancel_request():
    deps, _calls = _deps(now=5000.0)
    task = {
        "id": "t-alive-cancel-stale",
        "ram_mb": 1000,
        "cpu_cores": 1,
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "cancel_request_token": "stale-request",
        "cancel_requested_at": 1.0,
        "cancel_request_actor": {"label": "operator"},
        "cancel_request_reason": "user batch force-cancel",
    }

    apply_alive_probe_result(
        task,
        {"alive_pids": [11], "vram_mb": 0, "ram_mb": 100, "pcpu": 10.0},
        None,
        deps=deps,
    )

    assert "cancel_request_token" not in task
    assert task["expired_cancel_request_at"] == 5000.0
