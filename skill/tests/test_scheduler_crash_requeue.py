from __future__ import annotations

from skill.scheduler_failure.crash_requeue import CrashRequeueDeps, requeue_after_crash


def _task(task_id="tParent", **overrides):
    task = {
        "id": task_id,
        "status": "failed",
        "signature": "proj/exp/s1",
        "cmd": "python train.py --seed 1",
        "submitted_at": 10.0,
        "priority": "normal",
        "retry_count": 0,
        "_diagnosis": {"is_crash": True, "reason": "RuntimeError", "tail": "Traceback", "lifetime_s": 100},
    }
    task.update(overrides)
    return task


def _deps(
    *,
    safety=("dead", "no recorded launch artifacts"),
    category="UNKNOWN",
    cancelled_descendant=False,
    max_auto_retry=3,
    escalations=None,
    cleared=None,
):
    escalations = escalations if escalations is not None else []
    cleared = cleared if cleared is not None else []

    def allocate(state):
        next_id = int(state.get("next_id", 1))
        state["next_id"] = next_id + 1
        return f"t{next_id:04d}"

    def identity(task):
        sig = task.get("signature")
        if not sig:
            return None
        return (sig, task.get("cmd") or "", task.get("cwd") or "")

    def classify(diag):
        return category

    def write_escalation(task, cat, diag):
        escalations.append((task["id"], cat, diag))

    def clear_eta(task, *, clear_runtime_projection=False):
        cleared.append((task["id"], clear_runtime_projection))
        for key in (
            "eta_seconds",
            "eta_source",
            "eta_confidence",
            "last_progress_line",
            "runtime_total_s_est",
            "runtime_eta_s_est",
            "runtime_est_source",
            "runtime_current_unit",
            "runtime_total_units",
            "runtime_unit_s_est",
        ):
            task.pop(key, None)
        return True

    return CrashRequeueDeps(
        max_auto_retry=max_auto_retry,
        allocate_task_id=allocate,
        recorded_launch_safety_state=lambda task: safety,
        task_run_identity=identity,
        has_user_cancelled_retry_descendant=lambda parent, state, key: cancelled_descendant,
        classify_failure=classify,
        write_escalation=write_escalation,
        clear_live_eta_fields=clear_eta,
        local_user=lambda: "tester",
        local_host_short=lambda: "host",
        scheduler_id=lambda: "sched-1",
        now=lambda: 1234.0,
    )


def test_retry_clone_clears_runtime_artifacts_and_becomes_scheduler_owned():
    parent = _task(
        node="n1",
        gpu_idx=0,
        remote_pids=[111],
        log_path="/tmp/old.log",
        started_at=1.0,
        finished_at=2.0,
        slurm_job_id=42,
        slurm_state="FAILED",
        actual_started_at=1.5,
        container_name="old-container",
        container_main_pid=222,
        adopted=True,
        auto_adopted=True,
        process_group=333,
        exit_status_path="/tmp/old.status",
        exit_status_token="old-token",
        remote_pid_start_ticks={"111": 999},
        backend_finished_at=1.9,
        requeued_as="old-child",
        migrated_from="n0",
        resume_locations=["stale"],
        eta_seconds=999,
        runtime_total_s_est=999,
    )
    state = {"tasks": [parent], "next_id": 7}
    cleared = []

    new_id = requeue_after_crash(parent, state, deps=_deps(cleared=cleared))

    assert new_id == "t0007"
    clone = state["tasks"][-1]
    assert clone["id"] == "t0007"
    assert clone["status"] == "queued"
    assert clone["parent_id"] == "tParent"
    assert clone["retry_count"] == 1
    assert clone["submitted_at"] == parent["submitted_at"]
    assert clone["node"] is None
    assert clone["gpu_idx"] is None
    assert clone["remote_pids"] == []
    assert clone["log_path"] is None
    assert clone["started_at"] is None
    assert clone["finished_at"] is None
    assert clone["slurm_job_id"] is None
    assert clone["slurm_state"] is None
    assert clone["actual_started_at"] is None
    assert clone["container_name"] is None
    assert clone["container_main_pid"] is None
    assert clone["adopted"] is False
    assert clone["auto_adopted"] is False
    assert clone["origin"] == "scheduleurm"
    assert clone["submitted_by"] == "tester"
    assert clone["submitted_host"] == "host"
    assert clone["scheduler_id"] == "sched-1"
    assert clone["process_group"] is None
    assert "exit_status_path" not in clone
    assert "exit_status_token" not in clone
    assert "remote_pid_start_ticks" not in clone
    assert "backend_finished_at" not in clone
    assert clone["_diagnosis"] is None
    assert "requeued_as" not in clone
    assert "migrated_from" not in clone
    assert "resume_locations" not in clone
    assert "eta_seconds" not in clone
    assert "runtime_total_s_est" not in clone
    assert cleared == [("t0007", True)]


def test_active_duplicate_identity_returns_existing_task_without_appending():
    parent = _task("failed")
    duplicate = _task("queued", status="queued")
    state = {"tasks": [parent, duplicate], "next_id": 10}

    new_id = requeue_after_crash(parent, state, deps=_deps())

    assert new_id == "queued"
    assert len(state["tasks"]) == 2


def test_alive_launch_artifact_blocks_duplicate_retry():
    parent = _task(remote_pids=[123], node="n1")
    state = {"tasks": [parent], "next_id": 10}

    new_id = requeue_after_crash(
        parent,
        state,
        deps=_deps(safety=("alive", "backend probe reports recorded pid/job alive")),
    )

    assert new_id is None
    assert len(state["tasks"]) == 1
    assert "recorded launch artifact is alive" in parent["last_block_reason"]
    assert parent["status"] == "failed"


def test_unknown_launch_artifact_defers_terminal_transition_instead_of_failing():
    parent = _task(
        remote_pids=[123],
        node="n1",
        finished_at=120.0,
        probe_unknown_since=90.0,
    )
    state = {"tasks": [parent], "next_id": 10}

    new_id = requeue_after_crash(
        parent,
        state,
        deps=_deps(safety=("unknown", "backend probe returned unknown")),
    )

    assert new_id is None
    assert len(state["tasks"]) == 1
    assert parent["status"] == "running"
    assert "finished_at" not in parent
    assert parent["node_probe_state"] == "unknown"
    assert parent["probe_unknown_since"] == 90.0
    assert parent["probe_unknown_count"] == 1
    assert "auto-requeue deferred" in parent["last_block_reason"]
    assert "backend probe returned unknown" in parent["last_probe_unknown_reason"]


def test_hard_fail_escalates_without_retry():
    parent = _task()
    state = {"tasks": [parent], "next_id": 10}
    escalations = []

    new_id = requeue_after_crash(
        parent,
        state,
        deps=_deps(category="DISK_FULL", escalations=escalations),
    )

    assert new_id is None
    assert parent["failure_category"] == "DISK_FULL"
    assert escalations == [("tParent", "DISK_FULL", parent["_diagnosis"])]
    assert len(state["tasks"]) == 1


def test_protocol_source_mismatch_escalates_without_retry():
    parent = _task(_diagnosis={
        "is_crash": True,
        "reason": "RuntimeError",
        "tail": (
            "RuntimeError: source snapshot changed during/re-entering the "
            "paired protocol; refusing to mix sources"
        ),
        "lifetime_s": 14_000,
    })
    state = {"tasks": [parent], "next_id": 10}
    escalations = []

    new_id = requeue_after_crash(
        parent,
        state,
        deps=_deps(category="APP_BUG", escalations=escalations),
    )

    assert new_id is None
    assert parent["failure_category"] == "PROTOCOL_INTEGRITY"
    assert escalations == [
        ("tParent", "PROTOCOL_INTEGRITY", parent["_diagnosis"]),
    ]
    assert len(state["tasks"]) == 1


def test_protocol_rollout_mismatch_escalates_without_retry():
    parent = _task(_diagnosis={
        "is_crash": True,
        "reason": "RuntimeError",
        "tail": (
            "RuntimeError: the two common-restart physical rollouts differ "
            "at iter 700"
        ),
        "lifetime_s": 8_000,
    })
    state = {"tasks": [parent], "next_id": 10}
    escalations = []

    new_id = requeue_after_crash(
        parent,
        state,
        deps=_deps(category="APP_BUG", escalations=escalations),
    )

    assert new_id is None
    assert parent["failure_category"] == "PROTOCOL_INTEGRITY"
    assert escalations == [
        ("tParent", "PROTOCOL_INTEGRITY", parent["_diagnosis"]),
    ]
    assert len(state["tasks"]) == 1


def test_resolved_environment_failure_gets_one_recovery_retry():
    parent = _task(
        retry_count=3,
        resolved_environment_retry={
            "ts": 123.0,
            "node": "node001",
            "resolution": "staging_success",
        },
    )
    state = {"tasks": [parent], "next_id": 10}
    escalations = []

    new_id = requeue_after_crash(
        parent,
        state,
        deps=_deps(category="ENV_MISSING", escalations=escalations),
    )

    assert new_id == "t0010"
    assert escalations == []
    clone = state["tasks"][-1]
    assert clone["status"] == "queued"
    assert "resolved_environment_retry" not in clone
    assert clone["recovered_from_environment_resolution"]["node"] == "node001"
    assert "environment-recovery requeue" in clone["last_block_reason"]


def test_retry_cap_escalates_app_bug_cap():
    parent = _task(retry_count=3)
    state = {"tasks": [parent], "next_id": 10}
    escalations = []

    new_id = requeue_after_crash(parent, state, deps=_deps(escalations=escalations))

    assert new_id is None
    assert escalations == [("tParent", "APP_BUG_CAP", parent["_diagnosis"])]
    assert len(state["tasks"]) == 1


def test_freqduet_keyerr25_gets_one_bugfix_retry_past_cap():
    parent = _task(
        retry_count=3,
        signature="FreqDuet/paper_route_day_policy_v1b_ep100_wu10_20seed/s42",
        cmd="python run_freqduet_ablation.py --seed 42",
        _diagnosis={
            "is_crash": True,
            "reason": "KeyError: 25",
            "tail": "KeyError: 25\nx = action[bus.bus_id]",
        },
    )
    state = {"tasks": [parent], "next_id": 10}

    new_id = requeue_after_crash(parent, state, deps=_deps())

    assert new_id == "t0010"
    clone = state["tasks"][-1]
    assert clone["retry_count"] == 4
    assert clone["bugfix_requeue_reason"] == "FreqDuet KeyError:25 max_agent_num/action slot fix"
    assert clone["bugfix_requeue_at"] == 1234.0
    assert "bugfix-requeue" in clone["last_block_reason"]


def test_placeholder_auto_adopted_cmd_is_not_requeued():
    parent = _task(cmd="(auto-adopted pid 123)")
    state = {"tasks": [parent], "next_id": 10}

    assert requeue_after_crash(parent, state, deps=_deps()) is None
    assert len(state["tasks"]) == 1


def test_cancelled_retry_descendant_suppresses_new_retry():
    parent = _task()
    state = {"tasks": [parent], "next_id": 10}

    assert requeue_after_crash(parent, state, deps=_deps(cancelled_descendant=True)) is None
    assert len(state["tasks"]) == 1
    assert "cancelled by the user" in parent["last_block_reason"]
