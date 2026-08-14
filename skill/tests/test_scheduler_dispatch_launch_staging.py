from __future__ import annotations

from skill.scheduler_dispatch_launch_staging import (
    DispatchLaunchStagingDeps,
    apply_launch_input_staging_gate,
    apply_launch_staging_gate,
)


def _deps(released: list, escalations: list, *, max_retry: int = 3):
    return DispatchLaunchStagingDeps(
        max_launch_retry=max_retry,
        launch_max_cwd_size_mb=2048,
        stage_failure_reason=lambda target, cwd: f"failed {target} {cwd}",
        release_task_claims_and_intents=lambda task, **kwargs: released.append(
            (task["id"], kwargs)
        ),
        write_escalation=lambda task, category, payload: escalations.append(
            (task["id"], category, payload)
        ),
        now=lambda: 123.0,
    )


def _task(**overrides):
    task = {
        "id": "t1",
        "status": "queued",
        "node": "node001",
        "gpu_idx": None,
        "launching_started_at": 1.0,
        "launch_token": "token",
    }
    task.update(overrides)
    return task


def test_launch_staging_gate_allows_ready_state():
    released = []
    escalations = []
    task = _task()

    result = apply_launch_staging_gate(
        task,
        target="node001",
        cwd_for_stage="/work",
        stage_state="ready",
        deps=_deps(released, escalations),
    )

    assert result.blocked is False
    assert result.event is None
    assert task["node"] == "node001"
    assert released == []
    assert escalations == []


def test_launch_staging_gate_cap_exceeded_pins_local_without_fail_count():
    released = []
    escalations = []
    task = _task()

    result = apply_launch_staging_gate(
        task,
        target="node001",
        cwd_for_stage="/work",
        stage_state="cap_exceeded",
        deps=_deps(released, escalations),
    )

    assert result.blocked is True
    assert result.event["type"] == "launch_capped"
    assert "cwd > 2048MB cap" in result.event["reason"]
    assert task["status"] == "queued"
    assert task["require_node"] == "local"
    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert "launch_fail_count" not in task
    assert "launch_token" not in task
    assert released == [("t1", {"extra_nodes": ["node001"]})]
    assert escalations == []


def test_launch_staging_cap_does_not_violate_explicit_remote_whitelist():
    released = []
    escalations = []
    task = _task(allowed_nodes=["node001", "node002"])

    result = apply_launch_staging_gate(
        task,
        target="node001",
        cwd_for_stage="/work",
        stage_state="cap_exceeded",
        deps=_deps(released, escalations),
    )

    assert result.blocked is True
    assert result.event["type"] == "launch_capped_no_allowed_fallback"
    assert task["status"] == "failed"
    assert task["require_node"] is None
    assert task["allowed_nodes"] == ["node001", "node002"]
    assert "refusing an inconsistent automatic pin" in task["last_block_reason"]
    assert task["node"] is None
    assert released == [("t1", {"extra_nodes": ["node001"]})]
    assert escalations == []


def test_launch_staging_gate_needs_stage_defers_without_fail_count():
    released = []
    escalations = []
    task = _task()

    result = apply_launch_staging_gate(
        task,
        target="node002",
        cwd_for_stage="/work",
        stage_state="needs_stage",
        deps=_deps(released, escalations),
    )

    assert result.blocked is True
    assert result.event == {
        "type": "launch_stage_deferred",
        "task_id": "t1",
        "task": task,
        "reason": "awaiting outside-lock staging to node002",
    }
    assert task["status"] == "queued"
    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert "launch_fail_count" not in task
    assert "not yet staged to node002" in task["last_block_reason"]
    assert released == [("t1", {"extra_nodes": ["node002"]})]


def test_input_stage_gate_defers_to_selected_target_then_clears_when_ready():
    released = []
    escalations = []
    task = _task()

    deferred = apply_launch_input_staging_gate(
        task,
        target="node002",
        stage_state="needs_stage",
        deps=_deps(released, escalations),
    )

    assert deferred.blocked is True
    assert deferred.event["type"] == "launch_input_stage_deferred"
    assert task["stage_input_target"] == "node002"
    assert task["node"] is None

    task["stage_input_target"] = "node002"
    ready = apply_launch_input_staging_gate(
        task,
        target="node002",
        stage_state="ready",
        deps=_deps(released, escalations),
    )
    assert ready.blocked is False
    assert "stage_input_target" not in task


def test_launch_staging_gate_stage_failed_requeues_and_records_failed_node():
    released = []
    escalations = []
    task = _task(launch_fail_count=1)

    result = apply_launch_staging_gate(
        task,
        target="node003",
        cwd_for_stage="/work",
        stage_state="stage_failed",
        deps=_deps(released, escalations, max_retry=3),
    )

    assert result.blocked is True
    assert result.event["type"] == "launch_failed_retry"
    assert result.event["error"] == "failed node003 /work"
    assert task["status"] == "queued"
    assert task["launch_fail_count"] == 2
    assert task["launch_failed_nodes"]["node003"] == {
        "ts": 123.0,
        "attempt": 2,
        "error": "stage_cwd: failed node003 /work",
    }
    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert released == [("t1", {"extra_nodes": ["node003"]})]
    assert escalations == []


def test_launch_staging_gate_stage_failed_escalates_at_retry_cap():
    released = []
    escalations = []
    task = _task(launch_fail_count=2)

    result = apply_launch_staging_gate(
        task,
        target="node004",
        cwd_for_stage="/work",
        stage_state="stage_failed",
        deps=_deps(released, escalations, max_retry=3),
    )

    assert result.blocked is True
    assert result.event["type"] == "launch_failed_terminal"
    assert task["status"] == "failed"
    assert task["launch_fail_count"] == 3
    assert escalations == [
        (
            "t1",
            "LAUNCH_FAIL_CAP",
            {"reason": "failed node004 /work", "tail": "failed node004 /work"},
        )
    ]
