from __future__ import annotations

from contextlib import contextmanager

from skill.scheduler_launch_commit import (
    LaunchCommitDeps,
    commit_deferred_launch_result,
    commit_deferred_launch_results_batch,
)


@contextmanager
def _null_lock(*args, **kwargs):
    yield


def _deps(state, calls):
    def apply_result(task, ok, msg, events):
        calls.append(("apply", task.get("id"), ok, msg))
        if ok:
            task["status"] = "running"
            task.pop("launch_token", None)
            events.append({"type": "launched", "task_id": task["id"], "msg": msg})
            return True
        task["status"] = "queued"
        events.append({"type": "launch_failed_retry", "task_id": task["id"], "msg": msg})
        return False

    return LaunchCommitDeps(
        state_lock=_null_lock,
        load_state=lambda: state,
        save_state=lambda saved: calls.append(("save", [t["id"] for t in saved["tasks"]])),
        apply_launch_result_to_task=apply_result,
        kill_task=lambda task: calls.append(("kill", task.get("id"))),
        release_task_claims_and_intents=lambda task: calls.append(("release", task.get("id"))),
    )


def _intent(task_id="t1", token="tok", snapshot=None):
    out = {
        "type": "launch_intent",
        "task_id": task_id,
        "task": {"id": task_id, "launch_token": token},
    }
    if snapshot is not None:
        out["post_launch_snapshot"] = snapshot
    return out


def test_commit_deferred_launch_result_applies_current_lease():
    state = {"tasks": [{"id": "t1", "status": "launching", "launch_token": "tok"}]}
    calls = []

    events = commit_deferred_launch_result(
        _intent(snapshot={"label": "post"}),
        {"id": "t1", "status": "launching", "launch_token": "tok", "remote_pids": [123]},
        True,
        "ok",
        mark_notified_launch=True,
        deps=_deps(state, calls),
    )

    task = state["tasks"][0]
    assert [ev["type"] for ev in events] == ["launched", "post_launch_snapshot"]
    assert task["status"] == "running"
    assert task["remote_pids"] == [123]
    assert task["notified_launch"] is True
    assert task["last_resource_snapshot"] == {"label": "post"}
    assert ("save", ["t1"]) in calls
    assert not any(call[0] == "kill" for call in calls)


def test_commit_deferred_launch_result_aborts_stale_successful_launch():
    state = {"tasks": [{"id": "t1", "status": "queued", "launch_token": "different"}]}
    calls = []

    events = commit_deferred_launch_result(
        _intent(),
        {"id": "t1", "status": "running", "remote_pids": [123]},
        True,
        "ok",
        deps=_deps(state, calls),
    )

    assert events == [{
        "type": "launch_commit_skipped",
        "task_id": "t1",
        "reason": "launch lease no longer current",
    }]
    assert state["tasks"][0]["status"] == "queued"
    assert ("kill", "t1") in calls
    assert ("release", "t1") in calls
    assert not any(call[0] == "save" for call in calls)


def test_batch_commit_uses_one_save_and_keeps_per_idx_events():
    state = {
        "tasks": [
            {"id": "t1", "status": "launching", "launch_token": "tok1"},
            {"id": "t2", "status": "launching", "launch_token": "tok2"},
        ]
    }
    calls = []
    records = [
        {
            "idx": 2,
            "intent": _intent("t2", "tok2"),
            "task": {"id": "t2", "status": "launching", "launch_token": "tok2"},
            "ok": False,
            "msg": "boom",
        },
        {
            "idx": 1,
            "intent": _intent("t1", "tok1"),
            "task": {"id": "t1", "status": "launching", "launch_token": "tok1"},
            "ok": True,
            "msg": "ok",
        },
    ]

    out = commit_deferred_launch_results_batch(records, deps=_deps(state, calls))

    assert [event["type"] for event in out[1]] == ["launched"]
    assert [event["type"] for event in out[2]] == ["launch_failed_retry"]
    assert [task["status"] for task in state["tasks"]] == ["running", "queued"]
    assert [call[0] for call in calls].count("save") == 1
