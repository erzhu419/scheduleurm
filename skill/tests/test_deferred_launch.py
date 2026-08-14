from __future__ import annotations

import threading
import time
from contextlib import contextmanager


def _launching_task(tid: str, node: str) -> dict:
    return {
        "id": tid,
        "status": "launching",
        "description": tid,
        "project": "test",
        "signature": f"test/{tid}",
        "cmd": "echo ok",
        "cwd": "/tmp",
        "node": node,
        "gpu_idx": None,
        "remote_pids": [],
        "log_path": None,
        "launch_token": f"token-{tid}",
        "launching_started_at": time.time(),
        "est_vram_mb": 0,
        "ram_mb": 128,
        "cpu_cores": 1,
    }


def test_deferred_launches_run_concurrently_with_per_node_cap(monkeypatch, sch):
    monkeypatch.setattr(sch, "LAUNCH_EXEC_MAX_WORKERS", 6, raising=False)
    monkeypatch.setattr(sch, "LAUNCH_EXEC_MAX_PER_NODE", 2, raising=False)

    tasks = [
        _launching_task("t0001", "node001"),
        _launching_task("t0002", "node001"),
        _launching_task("t0003", "node001"),
        _launching_task("t0004", "node002"),
        _launching_task("t0005", "node002"),
        _launching_task("t0006", "node002"),
    ]
    sch.save_state({"tasks": [dict(t) for t in tasks]})

    counters_lock = threading.Lock()
    active_total = 0
    active_by_node: dict[str, int] = {}
    max_total = 0
    max_by_node: dict[str, int] = {}

    def fake_launch(task, node_state=None):
        nonlocal active_total, max_total
        node = task["node"]
        with counters_lock:
            active_total += 1
            active_by_node[node] = active_by_node.get(node, 0) + 1
            max_total = max(max_total, active_total)
            max_by_node[node] = max(max_by_node.get(node, 0), active_by_node[node])
        try:
            time.sleep(0.03)
            task["status"] = "running"
            task["remote_pids"] = [int(task["id"][1:])]
            task["log_path"] = f"/tmp/sched_{task['id']}.log"
            task["started_at"] = time.time()
            return True, "ok"
        finally:
            with counters_lock:
                active_total -= 1
                active_by_node[node] -= 1

    monkeypatch.setattr(sch, "launch", fake_launch)

    events = [
        {
            "type": "launch_intent",
            "task_id": t["id"],
            "task": dict(t),
            "node_state": {"name": t["node"]},
        }
        for t in tasks
    ]

    final_events = sch._execute_deferred_launches(events)
    state = sch.load_state()

    assert max_total > 1
    assert max(max_by_node.values()) <= 2
    assert [ev["type"] for ev in final_events] == ["launched"] * len(tasks)
    assert {t["status"] for t in state["tasks"]} == {"running"}
    assert all("launch_token" not in t for t in state["tasks"])
    assert all(t["remote_pids"] for t in state["tasks"])


def test_deferred_launches_commit_batch_with_one_state_lock(monkeypatch, sch):
    monkeypatch.setattr(sch, "LAUNCH_EXEC_MAX_WORKERS", 3, raising=False)
    monkeypatch.setattr(sch, "LAUNCH_EXEC_MAX_PER_NODE", 3, raising=False)

    tasks = [
        _launching_task("t0201", "node001"),
        _launching_task("t0202", "node002"),
        _launching_task("t0203", "node003"),
    ]
    sch.save_state({"tasks": [dict(t) for t in tasks]})

    def fake_launch(task, node_state=None):
        task["status"] = "running"
        task["remote_pids"] = [int(task["id"][1:])]
        task["log_path"] = f"/tmp/sched_{task['id']}.log"
        task["started_at"] = time.time()
        return True, "ok"

    real_state_lock = sch.state_lock
    commit_lock_purposes = []

    @contextmanager
    def counting_state_lock(*args, **kwargs):
        purpose = kwargs.get("purpose") or (args[2] if len(args) >= 3 else "")
        if str(purpose).startswith("launch:commit"):
            commit_lock_purposes.append(str(purpose))
        with real_state_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(sch, "launch", fake_launch)
    monkeypatch.setattr(sch, "state_lock", counting_state_lock)

    events = [
        {
            "type": "launch_intent",
            "task_id": t["id"],
            "task": dict(t),
            "node_state": {"name": t["node"]},
        }
        for t in tasks
    ]

    final_events = sch._execute_deferred_launches(events)
    state = sch.load_state()

    assert [ev["type"] for ev in final_events] == ["launched"] * len(tasks)
    assert commit_lock_purposes == ["launch:commit-batch"]
    assert {t["status"] for t in state["tasks"]} == {"running"}


def test_deferred_launches_use_node_slots_not_global_launch_lock(monkeypatch, sch):
    monkeypatch.setattr(sch, "LAUNCH_EXEC_MAX_WORKERS", 2, raising=False)
    monkeypatch.setattr(sch, "LAUNCH_EXEC_MAX_PER_NODE", 1, raising=False)

    tasks = [
        _launching_task("t0101", "node001"),
        _launching_task("t0102", "node002"),
    ]
    sch.save_state({"tasks": [dict(t) for t in tasks]})

    def fake_launch(task, node_state=None):
        task["status"] = "running"
        task["remote_pids"] = [int(task["id"][1:])]
        task["log_path"] = f"/tmp/sched_{task['id']}.log"
        task["started_at"] = time.time()
        return True, "ok"

    legacy_global_lock = sch.launch_exec_lock

    def forbidden_global_lock(*args, **kwargs):
        raise AssertionError("deferred launch must not acquire the legacy global launch lock")

    monkeypatch.setattr(sch, "launch", fake_launch)
    monkeypatch.setattr(sch, "launch_exec_lock", forbidden_global_lock)

    events = [
        {
            "type": "launch_intent",
            "task_id": t["id"],
            "task": dict(t),
            "node_state": {"name": t["node"]},
        }
        for t in tasks
    ]

    with legacy_global_lock(timeout_s=1, purpose="held-legacy-global"):
        final_events = sch._execute_deferred_launches(events)

    state = sch.load_state()
    assert [ev["type"] for ev in final_events] == ["launched", "launched"]
    assert {t["status"] for t in state["tasks"]} == {"running"}
