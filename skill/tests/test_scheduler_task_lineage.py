from __future__ import annotations

from skill import scheduler_task_lineage as lineage


def _identity(task):
    return task.get("identity")


def test_task_is_descendant_of_follows_parent_chain_and_stops_cycles():
    by_id = {
        "a": {"id": "a", "parent_id": "root"},
        "b": {"id": "b", "parent_id": "a"},
        "cycle1": {"id": "cycle1", "parent_id": "cycle2"},
        "cycle2": {"id": "cycle2", "parent_id": "cycle1"},
    }

    assert lineage.task_is_descendant_of(by_id["b"], "root", by_id) is True
    assert lineage.task_is_descendant_of(by_id["b"], "missing", by_id) is False
    assert lineage.task_is_descendant_of(by_id["cycle1"], "missing", by_id) is False


def test_mark_user_cancelled_sets_terminal_audit_fields():
    task = {"id": "t1", "status": "launching", "launch_token": "tok", "launching_started_at": 1}

    lineage.mark_user_cancelled(
        task,
        "stop",
        actor_info=lambda action, reason: {"label": "tester", "action": action, "reason": reason},
        now=lambda: 123.0,
    )

    assert task["status"] == "cancelled"
    assert task["finished_at"] == 123.0
    assert task["cancelled_at"] == 123.0
    assert task["cancelled_by_user"] is True
    assert task["cancelled_by"] == "tester"
    assert task["cancel_actor"]["action"] == "cancel"
    assert task["cancel_reason"] == "stop"
    assert task["last_block_reason"] == "stop"
    assert "launch_token" not in task
    assert "launching_started_at" not in task


def test_remember_last_placement_records_node_and_gpu_without_overwriting_missing_values():
    task = {"node": "node001", "gpu_idx": 0}
    lineage.remember_last_placement(task)
    assert task["last_node"] == "node001"
    assert task["last_gpu_idx"] == 0

    task.pop("node")
    task["gpu_idx"] = None
    lineage.remember_last_placement(task)
    assert task["last_node"] == "node001"
    assert task["last_gpu_idx"] == 0


def test_cancel_related_queued_retries_only_cancels_same_identity_queued_or_launching():
    calls = []
    state = {
        "tasks": [
            {"id": "target", "status": "queued", "identity": "same"},
            {"id": "q", "status": "queued", "identity": "same", "node": "n1", "gpu_idx": 1, "remote_pids": [10]},
            {"id": "l", "status": "launching", "identity": "same", "node": "n2", "gpu_idx": 2, "remote_pids": [20]},
            {"id": "r", "status": "running", "identity": "same", "node": "n3"},
            {"id": "other", "status": "queued", "identity": "other", "node": "n4"},
        ]
    }
    target = state["tasks"][0]

    count = lineage.cancel_related_queued_retries(
        state,
        target,
        "user cancel",
        task_run_identity=_identity,
        release_task_claims_and_intents=lambda task: calls.append(("release", task["id"])) or 1,
        mark_user_cancelled=lambda task, reason: calls.append(("cancel", task["id"], reason)) or task.update(
            status="cancelled",
            last_block_reason=reason,
        ),
    )

    assert count == 2
    assert calls[0] == ("release", "q")
    assert calls[1][0:2] == ("cancel", "q")
    assert calls[2] == ("release", "l")
    assert calls[3][0:2] == ("cancel", "l")
    assert state["tasks"][1]["node"] is None
    assert state["tasks"][1]["gpu_idx"] is None
    assert state["tasks"][1]["remote_pids"] == []
    assert state["tasks"][2]["node"] is None
    assert state["tasks"][3]["status"] == "running"
    assert state["tasks"][4]["status"] == "queued"


def test_has_user_cancelled_retry_descendant_detects_nested_cancelled_retry():
    parent = {"id": "p", "status": "failed", "identity": "same"}
    child = {"id": "c", "status": "failed", "parent_id": "p", "identity": "same"}
    grandchild = {"id": "g", "status": "cancelled", "parent_id": "c", "identity": "same"}

    assert lineage.has_user_cancelled_retry_descendant(
        parent,
        {"tasks": [parent, child, grandchild]},
        task_run_identity=_identity,
    ) is True
    assert lineage.has_user_cancelled_retry_descendant(
        parent,
        {"tasks": [parent, dict(grandchild, identity="other")]},
        task_run_identity=_identity,
    ) is False


def test_reconcile_requeue_lineage_repairs_dispatchable_parents():
    calls = []
    parent = {
        "id": "p",
        "status": "queued",
        "requeued_as": "c",
        "node": "n1",
        "gpu_idx": 0,
        "remote_pids": [123],
        "launching_started_at": 1,
        "launch_token": "tok",
    }
    child = {"id": "c", "status": "queued"}
    cancelled_parent = {"id": "pc", "status": "launching", "requeued_as": "cc", "node": "n2"}
    cancelled_child = {"id": "cc", "status": "cancelled"}
    state = {"tasks": [parent, child, cancelled_parent, cancelled_child]}

    changed = lineage.reconcile_requeue_lineage_invariants(
        state,
        release_task_claims_and_intents=lambda task: calls.append(("release", task["id"])) or 1,
        mark_user_cancelled=lambda task, reason: calls.append(("cancel", task["id"], reason)) or task.update(
            status="cancelled",
            last_block_reason=reason,
        ),
        now=lambda: 500.0,
    )

    assert changed == 2
    assert calls[0] == ("release", "p")
    assert calls[1] == ("release", "pc")
    assert calls[2][0:2] == ("cancel", "pc")
    assert parent["status"] == "failed"
    assert parent["finished_at"] == 500.0
    assert parent["node"] is None
    assert parent["gpu_idx"] is None
    assert parent["remote_pids"] == []
    assert "launch_token" not in parent
    assert "superseded by retry c" in parent["last_block_reason"]
    assert cancelled_parent["status"] == "cancelled"
