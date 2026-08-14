from __future__ import annotations

from skill.scheduler_launch_result import LaunchResultDeps, apply_launch_result_to_task


def _deps(calls, *, max_launch_retry=3):
    return LaunchResultDeps(
        max_launch_retry=max_launch_retry,
        remember_claim_intent=lambda task, node: calls.append(("remember_claim", task["id"], node)),
        release_task_claims_and_intents=lambda task, **kwargs: calls.append(("release_claims", task["id"], kwargs)),
        write_escalation=lambda task, code, payload: calls.append(("escalate", task["id"], code, payload)),
        clear_claim_intent_markers=lambda task: calls.append(("clear_claim", task["id"])),
        now=lambda: 500.0,
    )


def test_claim_race_requeues_without_releasing_claims():
    calls = []
    events = []
    task = {
        "id": "t1",
        "status": "launching",
        "node": "node001",
        "gpu_idx": 0,
        "launching_started_at": 1,
        "launch_token": "tok",
    }

    ok = apply_launch_result_to_task(
        task,
        False,
        "CLAIM_RACE: taken",
        events,
        deps=_deps(calls),
    )

    assert ok is False
    assert task["status"] == "queued"
    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert "launch_token" not in task
    assert calls == [("remember_claim", "t1", "node001")]
    assert events[0]["type"] == "claim_race"


def test_launch_failure_records_failed_node_and_retries_before_cap():
    calls = []
    events = []
    task = {"id": "t1", "status": "launching", "node": "node002", "gpu_idx": 1}

    ok = apply_launch_result_to_task(task, False, "env missing", events, deps=_deps(calls))

    assert ok is False
    assert task["status"] == "queued"
    assert task["launch_fail_count"] == 1
    assert task["launch_failed_nodes"]["node002"] == {
        "ts": 500.0,
        "attempt": 1,
        "error": "env missing",
    }
    assert task["last_block_reason"] == "launch attempt 1/3: env missing"
    assert ("release_claims", "t1", {"extra_nodes": ["node002"], "clear_markers": False}) in calls
    assert events[0]["type"] == "launch_failed_retry"


def test_launch_failure_marks_terminal_at_cap_and_escalates():
    calls = []
    events = []
    task = {
        "id": "t1",
        "status": "launching",
        "node": "node003",
        "gpu_idx": 0,
        "launch_fail_count": 1,
    }

    ok = apply_launch_result_to_task(
        task,
        False,
        "still broken",
        events,
        deps=_deps(calls, max_launch_retry=2),
    )

    assert ok is False
    assert task["status"] == "failed"
    assert ("escalate", "t1", "LAUNCH_FAIL_CAP", {"reason": "still broken", "tail": "still broken"}) in calls
    assert events[0]["type"] == "launch_failed_terminal"


def test_success_clears_retry_and_cooldown_fields_but_preserves_warnings():
    calls = []
    events = []
    task = {
        "id": "t1",
        "status": "launching",
        "launch_fail_count": 2,
        "launch_failed_nodes": {"node001": {}},
        "evict_cooldown_until": 10,
        "evict_node_cooldowns": {"local": 20},
        "launch_token": "tok",
        "last_block_reason": "warn: high cpu",
    }

    ok = apply_launch_result_to_task(task, True, "pid=42", events, deps=_deps(calls))

    assert ok is True
    assert "launch_fail_count" not in task
    assert "launch_failed_nodes" not in task
    assert "evict_cooldown_until" not in task
    assert "evict_node_cooldowns" not in task
    assert "launch_token" not in task
    assert task["last_block_reason"] == "warn: high cpu"
    assert calls == [("clear_claim", "t1")]
    assert events == [{"type": "launched", "task_id": "t1", "task": task, "msg": "pid=42"}]
