from __future__ import annotations

import time
import threading
from contextlib import contextmanager

from skill.scheduler_launch_executor import execute_launch_intents


@contextmanager
def _null_lock():
    yield


def _intent(tid: str, node: str) -> dict:
    return {
        "type": "launch_intent",
        "task_id": tid,
        "task": {"id": tid, "node": node},
        "node_state": {"name": node},
    }


def test_execute_launch_intents_returns_same_events_when_no_intents():
    events = [{"type": "blocked", "task_id": "t1"}]

    out = execute_launch_intents(
        events,
        max_workers=4,
        launch_fn=lambda task, node_state=None: (True, "ok"),
        commit_fn=lambda ev, task, ok, msg: [],
        slot_lock_factory=lambda node: _null_lock(),
        notify_fn=lambda *args, **kwargs: None,
    )

    assert out is events


def test_execute_launch_intents_preserves_event_order_after_parallel_launches():
    events = [
        {"type": "blocked", "task_id": "before"},
        _intent("t0001", "node001"),
        {"type": "git_warn", "task_id": "middle"},
        _intent("t0002", "node002"),
    ]
    locked_nodes = []

    @contextmanager
    def slot_lock(node):
        locked_nodes.append(node)
        yield

    def launch(task, node_state=None):
        if task["id"] == "t0001":
            time.sleep(0.03)
        task["status"] = "running"
        return True, f"launched {task['id']}"

    def commit(ev, task, ok, msg):
        assert ok is True
        return [{
            "type": "launched",
            "task_id": task["id"],
            "status": task["status"],
            "msg": msg,
        }]

    out = execute_launch_intents(
        events,
        max_workers=2,
        launch_fn=launch,
        commit_fn=commit,
        slot_lock_factory=slot_lock,
        notify_fn=lambda *args, **kwargs: None,
    )

    assert [ev["task_id"] for ev in out] == ["before", "t0001", "middle", "t0002"]
    assert [ev["type"] for ev in out] == ["blocked", "launched", "git_warn", "launched"]
    assert set(locked_nodes) == {"node001", "node002"}
    assert "status" not in events[1]["task"]


def test_execute_launch_intents_uses_batch_commit_when_provided():
    events = [
        {"type": "blocked", "task_id": "before"},
        _intent("t0001", "node001"),
        _intent("t0002", "node002"),
    ]
    batch_calls = []

    def launch(task, node_state=None):
        task["status"] = "running"
        return True, f"ok {task['id']}"

    def batch_commit(records):
        batch_calls.append([(r["idx"], r["task"]["id"], r["ok"], r["msg"]) for r in records])
        return {
            r["idx"]: [{
                "type": "launched",
                "task_id": r["task"]["id"],
                "msg": r["msg"],
            }]
            for r in records
        }

    out = execute_launch_intents(
        events,
        max_workers=2,
        launch_fn=launch,
        commit_fn=lambda ev, task, ok, msg: (_ for _ in ()).throw(
            AssertionError("per-task commit should not run")
        ),
        batch_commit_fn=batch_commit,
        slot_lock_factory=lambda node: _null_lock(),
        notify_fn=lambda *args, **kwargs: None,
    )

    assert batch_calls == [[(1, "t0001", True, "ok t0001"), (2, "t0002", True, "ok t0002")]]
    assert [ev["task_id"] for ev in out] == ["before", "t0001", "t0002"]


def test_execute_launch_intents_converts_commit_exception_to_error_event():
    notified = []

    def launch(task, node_state=None):
        task["status"] = "running"
        return True, "ok"

    def commit(ev, task, ok, msg):
        raise RuntimeError("state file busy")

    def notify(*args, **kwargs):
        notified.append((args, kwargs))

    out = execute_launch_intents(
        [_intent("t0003", "node003")],
        max_workers=1,
        launch_fn=launch,
        commit_fn=commit,
        slot_lock_factory=lambda node: _null_lock(),
        notify_fn=notify,
    )

    assert out[0]["type"] == "launch_commit_error"
    assert out[0]["task_id"] == "t0003"
    assert "state file busy" in out[0]["error"]
    assert notified[0][0][0] == "deferred_launch_commit_error"
    assert notified[0][1] == {"feishu_enabled": False}


def test_execute_launch_intents_commits_completed_work_before_slow_tail_finishes():
    release_tail = threading.Event()
    first_commit = threading.Event()
    committed = []

    def launch(task, node_state=None):
        if task["id"] == "slow":
            assert release_tail.wait(timeout=3)
        task["status"] = "running"
        return True, "ok"

    def batch_commit(records):
        committed.extend(record["task"]["id"] for record in records)
        first_commit.set()
        return {
            record["idx"]: [{"type": "launched", "task_id": record["task"]["id"]}]
            for record in records
        }

    events = [_intent(f"fast-{idx}", f"node{idx}") for idx in range(4)]
    events.append(_intent("slow", "slow-node"))
    result = {}

    def run():
        result["events"] = execute_launch_intents(
            events,
            max_workers=5,
            launch_fn=launch,
            commit_fn=lambda *args: [],
            batch_commit_fn=batch_commit,
            slot_lock_factory=lambda node: _null_lock(),
            notify_fn=lambda *args, **kwargs: None,
            commit_batch_size=2,
            commit_max_delay_s=0.05,
        )

    worker = threading.Thread(target=run)
    worker.start()
    assert first_commit.wait(timeout=1)
    assert "slow" not in committed
    assert any(task_id.startswith("fast-") for task_id in committed)
    release_tail.set()
    worker.join(timeout=3)

    assert worker.is_alive() is False
    assert [event["task_id"] for event in result["events"]] == [
        "fast-0",
        "fast-1",
        "fast-2",
        "fast-3",
        "slow",
    ]
