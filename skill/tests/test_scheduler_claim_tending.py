from __future__ import annotations

import pytest

from skill.scheduler_claim_inputs import (
    claim_pid_updates,
    collect_claim_renewal_inputs,
    stale_claim_task_ids,
)
from skill.scheduler_claim_tending import ClaimTendingDeps, tend_scheduler_claims


def _deps(calls: list, notifications: list, *, enumerate_by_node=None, enabled=None):
    enabled = enabled or {"node001", "node002"}
    enumerate_by_node = enumerate_by_node or {}

    def enumerate_claims(node):
        calls.append(("enumerate", node))
        values = enumerate_by_node.get(node, [])
        if values and isinstance(values[0], list):
            return values.pop(0)
        return list(values)

    def release_claim(node, task_id):
        calls.append(("release", node, task_id))
        return True

    def renew_many(node, task_ids, records):
        calls.append(("renew_many", node, list(task_ids), dict(records)))
        return len(task_ids)

    def update_pid(node, task_id, pid):
        calls.append(("update_pid", node, task_id, pid))
        return True

    def gc_stale(node):
        calls.append(("gc_stale", node))
        return 0

    def notify(event, payload, **kwargs):
        notifications.append((event, payload, kwargs))

    return ClaimTendingDeps(
        enabled_for_node=lambda node: node in enabled,
        collect_claim_renewal_inputs=collect_claim_renewal_inputs,
        claim_record_for_task=lambda task: {"cpu_cores": task.get("cpu_cores", 1)},
        scheduler_id=lambda: "sid",
        enumerate_claims=enumerate_claims,
        stale_claim_task_ids=stale_claim_task_ids,
        release_claim=release_claim,
        renew_many=renew_many,
        claim_pid_updates=claim_pid_updates,
        update_pid=update_pid,
        gc_stale=gc_stale,
        notify=notify,
    )


def test_tend_scheduler_claims_releases_renews_and_reconciles_pids():
    calls = []
    notifications = []
    enumerate_by_node = {
        "node001": [
            [
                {"scheduler_id": "sid", "task_id": "t-run", "pid": None},
                {"scheduler_id": "sid", "task_id": "t-launch", "pid": None},
                {"scheduler_id": "sid", "task_id": "t-stale", "pid": 999},
                {"scheduler_id": "other", "task_id": "t-other", "pid": 111},
            ],
            [
                {"scheduler_id": "sid", "task_id": "t-run", "pid": None},
            ],
        ],
    }
    tasks = [
        {
            "id": "t-run",
            "status": "running",
            "node": "node001",
            "remote_pids": [123],
            "cpu_cores": 4,
        },
        {
            "id": "t-launch",
            "status": "launching",
            "node": "node001",
            "remote_pids": [],
            "cpu_cores": 2,
        },
    ]

    result = tend_scheduler_claims(
        tasks,
        {"node001": {}, "node-disabled": {}},
        deps=_deps(calls, notifications, enumerate_by_node=enumerate_by_node),
    )

    assert result.released_by_node == {"node001": ["t-stale"]}
    assert result.renewed_by_node == {"node001": ["t-run"]}
    assert result.pid_updates_by_node == {"node001": [("t-run", 123)]}
    assert ("release", "node001", "t-stale") in calls
    assert ("renew_many", "node001", ["t-run"], {"t-run": {"cpu_cores": 4}}) in calls
    assert ("update_pid", "node001", "t-run", 123) in calls
    assert notifications == [
        (
            "claims_released_inactive",
            {"node": "node001", "task_ids": ["t-stale"], "count": 1},
            {"feishu_enabled": False},
        )
    ]


def test_tend_scheduler_claims_gc_when_enabled_node_has_no_running_tasks():
    calls = []
    notifications = []

    result = tend_scheduler_claims(
        [{"id": "t-launch", "status": "launching", "node": "node001"}],
        ["node001", "node002"],
        deps=_deps(calls, notifications),
    )

    assert result.gc_nodes == ["node001", "node002"]
    assert ("gc_stale", "node001") in calls
    assert ("gc_stale", "node002") in calls
    assert not any(call[0] == "renew_many" for call in calls)


def test_tend_scheduler_claims_node_error_is_logged_and_next_node_continues():
    calls = []
    notifications = []
    deps = _deps(calls, notifications)

    def renew_many(node, task_ids, records):
        calls.append(("renew_many", node))
        if node == "node001":
            raise RuntimeError("boom")
        return 1

    deps.renew_many = renew_many
    tasks = [
        {"id": "t1", "status": "running", "node": "node001", "remote_pids": []},
        {"id": "t2", "status": "running", "node": "node002", "remote_pids": []},
    ]

    result = tend_scheduler_claims(tasks, ["node001", "node002"], deps=deps)

    assert result.error_nodes == {"node001": "boom"}
    assert result.renewed_by_node == {"node002": ["t2"]}
    assert ("renew_many", "node002") in calls
    assert notifications == [
        (
            "claims_tend_error",
            {"node": "node001", "error": "boom"},
            {"feishu_enabled": False},
        )
    ]


def test_tend_scheduler_claims_preserves_outer_collect_failure():
    calls = []
    notifications = []
    deps = _deps(calls, notifications)

    def fail_collect(*args, **kwargs):
        raise ValueError("bad pid")

    deps.collect_claim_renewal_inputs = fail_collect

    with pytest.raises(ValueError, match="bad pid"):
        tend_scheduler_claims([], ["node001"], deps=deps)
