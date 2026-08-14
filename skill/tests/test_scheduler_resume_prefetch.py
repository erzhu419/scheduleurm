from __future__ import annotations

from dataclasses import dataclass

import skill.scheduler_migration.resume_prefetch as resume_prefetch
from skill.scheduler_migration.resume_prefetch import (
    ResumePrefetchDeps,
    prefetch_resume_scans_outside_lock,
)


@dataclass
class _LockState:
    depth: int = 0


class _Lock:
    def __init__(self, lock_state: _LockState, calls: list, purpose: str):
        self.lock_state = lock_state
        self.calls = calls
        self.purpose = purpose

    def __enter__(self):
        self.lock_state.depth += 1
        self.calls.append(("lock_enter", self.purpose))

    def __exit__(self, *args):
        self.calls.append(("lock_exit", self.purpose))
        self.lock_state.depth -= 1


def _task(task_id: str, priority: str, submitted_at: int):
    return {
        "id": task_id,
        "status": "queued",
        "priority": priority,
        "submitted_at": submitted_at,
        "ckpt_dir": f"/ckpt/{task_id}",
    }


def _deps(state: dict, calls: list, *, max_tasks=8, scan_hook=None, now=100.0):
    lock_state = _LockState()

    def state_lock(*, purpose):
        return _Lock(lock_state, calls, purpose)

    def cached(task, nodes):
        key = [task["ckpt_dir"], [node["name"] for node in nodes]]
        fresh = task.get("resume_scan_key") == key and task.get("resume_scan_completed_at") == now
        return fresh, list(task.get("resume_locations") or []), {}

    def scan(task, *, nodes, cache):
        assert lock_state.depth == 0
        calls.append(("scan", task["id"]))
        if scan_hook is not None:
            return scan_hook(task, state)
        return [{"node": nodes[0]["name"], "path": f"/ckpt/{task['id']}/checkpoint.pt"}], {}

    def apply(task, *, scan_key, locations, errors, completed_at, source):
        assert lock_state.depth == 1
        calls.append(("apply", task["id"], source))
        task["resume_scan_key"] = scan_key
        task["resume_scan_completed_at"] = completed_at
        task["resume_locations"] = list(locations)
        task["resume_scan_errors"] = dict(errors)

    return ResumePrefetchDeps(
        state_lock=state_lock,
        load_state=lambda: state,
        save_state=lambda saved: calls.append(("save", lock_state.depth)),
        task_requires_resume_scan=lambda task: bool(task.get("ckpt_dir")),
        resume_scan_key=lambda task, nodes: [
            task["ckpt_dir"],
            [node["name"] for node in nodes],
        ],
        cached_resume_locations_for_task=cached,
        scan_resume_locations=scan,
        apply_resume_scan_result=apply,
        notify=lambda event, payload, **kwargs: calls.append(("notify", event, payload)),
        max_tasks_per_pass=max_tasks,
        task_workers=1,
        inflight_ttl_s=60,
        now=lambda: now,
        token_factory=lambda task: f"claim-{task['id']}",
    )


def test_prefetch_scans_outside_lock_and_commits_weighted_fair_batch():
    calls = []
    tasks = [
        *[_task(f"h{i}", "high", i) for i in range(1, 7)],
        _task("n1", "normal", 1),
        _task("n2", "normal", 2),
        _task("l1", "low", 1),
    ]
    state = {"tasks": tasks}

    summary = prefetch_resume_scans_outside_lock(
        [{"name": "node001"}],
        deps=_deps(state, calls, max_tasks=7),
    )

    assert summary == {"claimed": 7, "completed": 7, "committed": 7, "seconds": 0.0}
    assert [call[1] for call in calls if call[0] == "scan"] == [
        "h1", "h2", "h3", "h4", "n1", "n2", "l1",
    ]
    assert all(task.get("resume_scan_inflight") is None for task in tasks)
    assert [call for call in calls if call[0] == "save"] == [("save", 1), ("save", 1)]


def test_prefetch_skips_fresh_and_live_inflight_but_reclaims_stale_claim():
    calls = []
    fresh = _task("fresh", "high", 1)
    fresh["resume_scan_key"] = [fresh["ckpt_dir"], ["node001"]]
    fresh["resume_scan_completed_at"] = 100.0
    live = _task("live", "normal", 1)
    live["resume_scan_inflight"] = {
        "token": "other",
        "started_at": 90.0,
        "key": [live["ckpt_dir"], ["node001"]],
    }
    stale = _task("stale", "low", 1)
    stale["resume_scan_inflight"] = {
        "token": "old",
        "started_at": 1.0,
        "key": [stale["ckpt_dir"], ["node001"]],
    }
    state = {"tasks": [fresh, live, stale]}

    summary = prefetch_resume_scans_outside_lock(
        [{"name": "node001"}],
        deps=_deps(state, calls),
    )

    assert summary["claimed"] == summary["committed"] == 1
    assert [call[1] for call in calls if call[0] == "scan"] == ["stale"]
    assert live["resume_scan_inflight"]["token"] == "other"
    assert "resume_scan_inflight" not in stale


def test_prefetch_reclaims_fresh_claim_from_dead_process(monkeypatch):
    calls = []
    task = _task("dead-owner", "high", 1)
    task["resume_scan_inflight"] = {
        "token": "12345:dead-owner:claim",
        "started_at": 99.0,
        "key": [task["ckpt_dir"], ["node001"]],
    }
    state = {"tasks": [task]}
    monkeypatch.setattr(resume_prefetch, "_owner_process_is_alive", lambda pid: False)

    summary = prefetch_resume_scans_outside_lock(
        [{"name": "node001"}],
        deps=_deps(state, calls),
    )

    assert summary["claimed"] == summary["committed"] == 1
    assert [call[1] for call in calls if call[0] == "scan"] == ["dead-owner"]
    assert "resume_scan_inflight" not in task


def test_prefetch_does_not_commit_result_after_claim_is_replaced():
    calls = []
    task = _task("t1", "normal", 1)
    state = {"tasks": [task]}

    def replace_claim(task_snapshot, live_state):
        live_state["tasks"][0]["resume_scan_inflight"]["token"] = "new-owner"
        return [{"node": "node001", "path": "/old-result.pt"}], {}

    summary = prefetch_resume_scans_outside_lock(
        [{"name": "node001"}],
        deps=_deps(state, calls, scan_hook=replace_claim),
    )

    assert summary["claimed"] == summary["completed"] == 1
    assert summary["committed"] == 0
    assert task["resume_scan_inflight"]["token"] == "new-owner"
    assert "resume_locations" not in task
