from __future__ import annotations

from skill.scheduler_result.batch_completion import detect_batch_completions, signature_batch_key


def _task(task_id: str, status: str, signature: str, **overrides):
    task = {
        "id": task_id,
        "status": status,
        "signature": signature,
        "submitted_at": 1.0,
        "finished_at": 10.0,
    }
    task.update(overrides)
    return task


def test_signature_batch_key_uses_top_two_components():
    assert signature_batch_key("") == ""
    assert signature_batch_key(None) == ""
    assert signature_batch_key("single") == "single"
    assert signature_batch_key("project/run/seed1") == "project/run"


def test_detects_newly_completed_batch_and_records_notify_timestamp():
    state = {
        "tasks": [
            _task("t1", "done", "proj/a/s1", submitted_at=1.0, finished_at=5.0),
            _task("t2", "failed", "proj/a/s2", submitted_at=2.0, finished_at=8.0),
            _task("t3", "cancelled", "proj/a/s3", submitted_at=3.0, finished_at=7.0),
        ]
    }

    completions = detect_batch_completions(state, ["t2"])

    assert completions == [
        {
            "prefix": "proj/a",
            "total": 3,
            "done": 1,
            "failed": 1,
            "cancelled": 1,
            "task_ids": ["t1", "t2", "t3"],
            "earliest_submit": 1.0,
            "latest_finish": 8.0,
        }
    ]
    assert state["_batch_notify_ts"]["proj/a"] == 8.0


def test_active_sibling_prevents_completion():
    state = {
        "tasks": [
            _task("t1", "done", "proj/a/s1", finished_at=5.0),
            _task("t2", "running", "proj/a/s2", finished_at=0.0),
        ]
    }

    assert detect_batch_completions(state, ["t1"]) == []
    assert state["_batch_notify_ts"] == {}


def test_completion_is_not_repeated_until_newer_finish_arrives():
    state = {
        "_batch_notify_ts": {"proj/a": 8.0},
        "tasks": [
            _task("t1", "done", "proj/a/s1", finished_at=5.0),
            _task("t2", "failed", "proj/a/s2", finished_at=8.0),
        ],
    }

    assert detect_batch_completions(state, ["t2"]) == []

    state["tasks"].append(_task("t3", "done", "proj/a/s3", finished_at=12.0))
    completions = detect_batch_completions(state, ["t3"])

    assert len(completions) == 1
    assert completions[0]["latest_finish"] == 12.0
    assert completions[0]["total"] == 3
    assert state["_batch_notify_ts"]["proj/a"] == 12.0


def test_auto_adopted_siblings_are_ignored():
    state = {
        "tasks": [
            _task("t1", "done", "proj/a/s1", finished_at=5.0),
            _task("ext", "running", "proj/a/external", auto_adopted=True, finished_at=0.0),
        ]
    }

    completions = detect_batch_completions(state, ["t1"])

    assert len(completions) == 1
    assert completions[0]["task_ids"] == ["t1"]
