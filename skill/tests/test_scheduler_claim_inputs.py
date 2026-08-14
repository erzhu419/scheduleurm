from __future__ import annotations

from skill.scheduler_claim_inputs import (
    claim_pid_updates,
    collect_claim_renewal_inputs,
    stale_claim_task_ids,
)


def test_collect_claim_renewal_inputs_groups_active_running_and_pid_records():
    record_calls = []

    def record_for(task):
        record_calls.append(task["id"])
        return {"cpu_cores": task.get("cpu_cores", 1)}

    out = collect_claim_renewal_inputs(
        [
            {
                "id": "t0001",
                "status": "running",
                "node": "node001",
                "remote_pids": ["123"],
                "cpu_cores": 4,
            },
            {
                "id": "t0002",
                "status": "launching",
                "node": "node001",
                "remote_pids": [456],
            },
            {
                "id": "t0003",
                "status": "running",
                "node": "node002",
                "remote_pids": [],
            },
            {
                "id": "t0004",
                "status": "done",
                "node": "node001",
                "remote_pids": [789],
            },
            {
                "id": "t0005",
                "status": "running",
                "node": "node-disabled",
                "remote_pids": [111],
            },
            {
                "id": "t0006",
                "status": "running",
                "node": None,
                "remote_pids": [222],
            },
        ],
        enabled_for_node=lambda node: node in {"node001", "node002"},
        claim_record_for_task=record_for,
    )

    assert out.active_claim_ids_by_node == {
        "node001": {"t0001", "t0002"},
        "node002": {"t0003"},
    }
    assert out.running_by_node == {
        "node001": ["t0001"],
        "node002": ["t0003"],
    }
    assert out.claim_records_by_node == {
        "node001": {"t0001": {"cpu_cores": 4}},
        "node002": {"t0003": {"cpu_cores": 1}},
    }
    assert out.live_pid_by_node == {
        "node001": {"t0001": 123},
    }
    assert record_calls == ["t0001", "t0003"]


def test_collect_claim_renewal_inputs_preserves_invalid_pid_failure():
    try:
        collect_claim_renewal_inputs(
            [{"id": "t0007", "status": "running", "node": "node001", "remote_pids": ["bad"]}],
            enabled_for_node=lambda node: True,
            claim_record_for_task=lambda task: {},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid remote_pids[0] should still surface like legacy code")


def test_stale_claim_task_ids_selects_only_own_inactive_claims():
    stale = stale_claim_task_ids(
        [
            {"scheduler_id": "sid", "task_id": "t0001"},
            {"scheduler_id": "sid", "task_id": "t0002"},
            {"scheduler_id": "other", "task_id": "t0003"},
            {"scheduler_id": "sid", "task_id": None},
            {"scheduler_id": "sid"},
            {"scheduler_id": "sid", "task_id": "t0002"},
        ],
        own_scheduler_id="sid",
        active_task_ids={"t0001"},
    )

    assert stale == ["t0002", "t0002"]


def test_claim_pid_updates_selects_only_missing_or_different_pids():
    updates = claim_pid_updates(
        [
            {"task_id": "t0001", "pid": 123},
            {"task_id": "t0002", "pid": None},
            {"task_id": "t0003", "pid": 999},
            {"task_id": "t0004", "pid": 456},
            {"task_id": None, "pid": 111},
        ],
        {"t0001": 123, "t0002": 222, "t0003": 333},
    )

    assert updates == [("t0002", 222), ("t0003", 333)]
