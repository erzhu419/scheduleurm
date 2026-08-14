from __future__ import annotations

from skill.scheduler_dispatch_task_gates import (
    DispatchTaskGateDeps,
    apply_dispatch_task_gates,
)


def _deps(**overrides):
    values = {
        "clear_disallowed_cpu_fallback_selection": lambda task: False,
        "reconcile_queued_launch_artifacts_before_dispatch": lambda task, state: None,
        "task_run_identity": lambda task: task.get("identity"),
        "same_run_identity_live_artifact_reason": lambda task, state, run_key: "",
        "eviction_cooldown_block_reason": lambda task: "",
        "queued_wait_for_file_block_reason": lambda task: "",
        "queued_cpu_training_block_reason": lambda task: "",
    }
    values.update(overrides)
    return DispatchTaskGateDeps(**values)


def test_dispatch_task_gates_clears_stale_cpu_fallback_without_skipping():
    task = {"id": "t1", "identity": "run1"}

    result = apply_dispatch_task_gates(
        task,
        state={"tasks": [task]},
        running_keys=set(),
        deps=_deps(clear_disallowed_cpu_fallback_selection=lambda task: True),
    )

    assert result.skip_task is False
    assert result.events == []
    assert "cleared stale CPU fallback placement" in task["last_block_reason"]


def test_dispatch_task_gates_artifact_event_skips_and_adds_running_key():
    task = {"id": "t2", "status": "running", "identity": ("sig", "cmd")}
    running_keys = set()
    artifact_event = {"type": "launched", "task_id": "t2", "task": task}

    result = apply_dispatch_task_gates(
        task,
        state={"tasks": [task]},
        running_keys=running_keys,
        deps=_deps(
            reconcile_queued_launch_artifacts_before_dispatch=lambda task, state: artifact_event
        ),
    )

    assert result.skip_task is True
    assert result.events == [artifact_event]
    assert running_keys == {("sig", "cmd")}


def test_dispatch_task_gates_blocks_duplicate_run_identity():
    task = {"id": "t3", "signature": "demo/sig", "identity": "same"}

    result = apply_dispatch_task_gates(
        task,
        state={"tasks": [task]},
        running_keys={"same"},
        deps=_deps(),
    )

    assert result.skip_task is True
    assert result.events[0]["type"] == "blocked"
    assert result.events[0]["task_id"] == "t3"
    assert "already has a running/launching task" in result.events[0]["reason"]
    assert task["last_block_reason"] == result.events[0]["reason"]


def test_dispatch_task_gates_blocks_terminal_artifact_before_other_reasons():
    task = {"id": "t4", "identity": "run4"}

    result = apply_dispatch_task_gates(
        task,
        state={"tasks": [task]},
        running_keys=set(),
        deps=_deps(
            same_run_identity_live_artifact_reason=lambda task, state, run_key: "terminal artifact",
            eviction_cooldown_block_reason=lambda task: "cooldown",
        ),
    )

    assert result.skip_task is True
    assert result.events[0]["reason"] == "terminal artifact"
    assert task["last_block_reason"] == "terminal artifact"


def test_dispatch_task_gates_blocks_first_ordered_policy_reason():
    task = {"id": "t5", "identity": "run5"}

    result = apply_dispatch_task_gates(
        task,
        state={"tasks": [task]},
        running_keys=set(),
        deps=_deps(
            eviction_cooldown_block_reason=lambda task: "",
            queued_wait_for_file_block_reason=lambda task: "wait for file",
            queued_cpu_training_block_reason=lambda task: "cpu training",
        ),
    )

    assert result.skip_task is True
    assert result.events == [
        {
            "type": "blocked",
            "task_id": "t5",
            "task": task,
            "reason": "wait for file",
        }
    ]
    assert task["last_block_reason"] == "wait for file"
