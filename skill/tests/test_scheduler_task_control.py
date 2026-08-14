from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager

import pytest

from skill.scheduler_task_control import (
    TaskControlDeps,
    cmd_cancel,
    cmd_clear_queue,
    cmd_forget,
    parse_cancel_task_ids,
)


def _deps(state, *, intent=None, events=None, kill_result=(True, "ok")):
    events = events if events is not None else []

    @contextmanager
    def state_lock(**kwargs):
        events.append(("lock", kwargs))
        try:
            yield
        finally:
            events.append(("unlock", kwargs))

    def mark_user_cancelled(task, reason):
        task["status"] = "cancelled"
        task["cancel_reason"] = reason
        task["cancelled_by"] = "tester"
        task["cancel_actor"] = {"label": "tester"}

    def kill_task(task, timeout=15):
        events.append(("kill", task["id"], task.get("node"), timeout))
        if callable(kill_result):
            return kill_result(task)
        return kill_result

    def kill_tasks_batch(tasks, timeout=15):
        events.append((
            "kill_batch",
            [task["id"] for task in tasks],
            [task.get("node") for task in tasks],
            timeout,
        ))
        return {
            task["id"]: kill_result(task) if callable(kill_result) else kill_result
            for task in tasks
        }

    return TaskControlDeps(
        read_dispatch_intent=lambda: intent,
        cancel_defers_to_dispatch_intent=True,
        dispatch_intent_message=lambda item: f"busy:{item.get('label')}",
        state_lock=state_lock,
        load_state=lambda: state,
        save_state=lambda saved: events.append(("save", [t["status"] for t in saved["tasks"]])),
        recover_stale_launching_tasks=lambda saved: events.append(("recover", len(saved["tasks"]))),
        release_task_claims_and_intents=lambda task: events.append(("release", task["id"])),
        mark_user_cancelled=mark_user_cancelled,
        cancel_related_queued_retries=lambda saved, task, reason: 1 if task.get("related") else 0,
        notify=lambda event_type, payload, **kwargs: events.append(("notify", event_type, payload)),
        task_pids=lambda task: list(task.get("remote_pids") or []),
        record_task_kill_actor=lambda task, action, reason: {"label": "tester", "action": action, "reason": reason},
        kill_task_processes=kill_task,
        actor_info=lambda action, reason: {"label": "tester", "action": action, "reason": reason},
        now=lambda: 123.0,
        write_dispatch_intent=lambda **kwargs: events.append(("write_intent", kwargs)) or {"pid": 1},
        clear_dispatch_intent=lambda: events.append(("clear_intent",)),
        kill_task_processes_batch=kill_tasks_batch,
        cancel_kill_max_workers=4,
    )


def _cancel_args(task_id="t1", **overrides):
    data = {
        "id": task_id,
        "force": False,
        "ignore_dispatch_intent": False,
        "force_lock_wait": False,
        "lock_timeout": None,
    }
    data.update(overrides)
    return Namespace(**data)


def _batch_args(ids=None, **overrides):
    data = {
        "cmd": "cancel-batch",
        "ids": list(ids or []),
        "project": None,
        "signature": None,
        "statuses": None,
        "force": False,
        "dry_run": False,
        "confirm": False,
        "ignore_missing": False,
        "ignore_dispatch_intent": False,
        "force_lock_wait": False,
        "lock_timeout": None,
    }
    data.update(overrides)
    return Namespace(**data)


def test_parse_cancel_task_ids_accepts_groups_ranges_and_deduplicates():
    assert parse_cancel_task_ids([
        "t0001-t0003,t0005",
        "t0003",
        "t8-t6",
    ]) == ["t0001", "t0002", "t0003", "t0005", "t8", "t7", "t6"]


def test_cancel_defers_when_dispatch_intent_is_active():
    state = {"tasks": [{"id": "t1", "status": "queued"}]}

    with pytest.raises(SystemExit, match="cancel deferred: busy:bulk-submit"):
        cmd_cancel(_cancel_args(), deps=_deps(state, intent={"label": "bulk-submit"}))

    assert state["tasks"][0]["status"] == "queued"


def test_cancel_queued_task_releases_claims_marks_cancelled_and_notifies(capsys):
    state = {"tasks": [{"id": "t1", "status": "launching", "launch_token": "abc", "related": True}]}
    events = []

    cmd_cancel(_cancel_args(), deps=_deps(state, events=events))

    out = capsys.readouterr().out
    task = state["tasks"][0]
    assert task["status"] == "cancelled"
    assert task["cancel_reason"] == "user cancel"
    assert "launch_token" not in task
    assert "cancelled launching task t1 (+1 duplicate queued retry) by tester" in out
    assert ("release", "t1") in events
    assert any(event[0] == "notify" and event[1] == "task_cancelled" for event in events)


def test_cancel_running_requires_force():
    state = {"tasks": [{"id": "t1", "status": "running", "node": "node001"}]}

    with pytest.raises(SystemExit, match="pass --force"):
        cmd_cancel(_cancel_args(), deps=_deps(state))

    assert state["tasks"][0]["status"] == "running"


def test_force_cancel_running_kills_processes_and_marks_cancelled(capsys):
    state = {"tasks": [{
        "id": "t1",
        "status": "running",
        "node": "node001",
        "remote_pids": [11, 12],
    }]}
    events = []

    cmd_cancel(_cancel_args(force=True), deps=_deps(state, events=events, kill_result=(True, "verified dead")))

    out = capsys.readouterr().out
    task = state["tasks"][0]
    assert task["status"] == "cancelled"
    assert task["last_killed_by"] == "tester"
    assert task["last_kill_ok"] is True
    assert "verified dead" in out
    assert any(event[0] == "notify" and event[1] == "task_killed" for event in events)


def test_force_cancel_running_keeps_task_tracked_when_kill_is_unconfirmed():
    state = {"tasks": [{
        "id": "t1",
        "status": "running",
        "node": "node001",
        "remote_pids": [11, 12],
    }]}
    events = []

    with pytest.raises(SystemExit, match="remains RUNNING and tracked"):
        cmd_cancel(
            _cancel_args(force=True),
            deps=_deps(
                state,
                events=events,
                kill_result=(False, "connection refused"),
            ),
        )

    task = state["tasks"][0]
    assert task["status"] == "running"
    assert task["last_kill_ok"] is False
    assert task["cancel_request_token"]
    assert not any(event[0] == "release" for event in events)


def test_batch_cancel_queued_and_launching_uses_one_state_save(capsys):
    state = {"tasks": [
        {"id": "q1", "status": "queued"},
        {"id": "q2", "status": "launching", "launch_token": "abc"},
        {"id": "r1", "status": "running", "node": "node001"},
    ]}
    events = []

    cmd_cancel(_batch_args(["q1,q2"]), deps=_deps(state, events=events))

    assert [task["status"] for task in state["tasks"]] == ["cancelled", "cancelled", "running"]
    assert sum(event[0] == "save" for event in events) == 1
    assert not any(event[0] == "kill" for event in events)
    assert any(event[0] == "notify" and event[1] == "tasks_cancelled_batch" for event in events)
    assert "cancelled 2 task(s) in batch" in capsys.readouterr().out


def test_batch_cancel_with_running_task_requires_force_atomically():
    state = {"tasks": [
        {"id": "q1", "status": "queued"},
        {"id": "r1", "status": "running", "node": "node001", "remote_pids": [11]},
    ]}
    events = []

    with pytest.raises(SystemExit, match="no tasks were cancelled"):
        cmd_cancel(_batch_args(["q1", "r1"]), deps=_deps(state, events=events))

    assert [task["status"] for task in state["tasks"]] == ["queued", "running"]
    assert not any(event[0] == "save" for event in events)
    assert not any(event[0] == "kill" for event in events)


def test_batch_force_cancel_kills_outside_lock_then_finalizes_once(capsys):
    state = {"tasks": [
        {"id": "t1", "status": "running", "node": "node001", "remote_pids": [11]},
        {"id": "t2", "status": "running", "node": "node001", "remote_pids": [22]},
        {"id": "q1", "status": "queued"},
    ]}
    events = []

    def kill_result(task):
        return (task["id"] == "t1", "ok" if task["id"] == "t1" else "already dead")

    with pytest.raises(SystemExit, match="remain RUNNING and tracked"):
        cmd_cancel(
            _batch_args(["t1-t2", "q1"], force=True),
            deps=_deps(state, events=events, kill_result=kill_result),
        )

    assert [task["status"] for task in state["tasks"]] == ["cancelled", "running", "cancelled"]
    assert sum(event[0] == "save" for event in events) == 2
    first_unlock = next(i for i, event in enumerate(events) if event[0] == "unlock")
    finalize_lock = next(
        i for i, event in enumerate(events)
        if event[0] == "lock" and event[1].get("purpose") == "cancel-batch:finalize"
    )
    kill_indexes = [i for i, event in enumerate(events) if event[0] == "kill_batch"]
    assert len(kill_indexes) == 1
    assert events[kill_indexes[0]][1] == ["t1", "t2"]
    assert all(first_unlock < index < finalize_lock for index in kill_indexes)
    assert state["tasks"][1]["last_kill_ok"] is False
    assert state["tasks"][1]["cancel_request_token"]
    assert "kill_failures=1" in capsys.readouterr().out


def test_selector_batch_cancel_previews_then_requires_confirm(capsys):
    state = {"tasks": [
        {"id": "q1", "status": "queued", "project": "Laplace-SMDP", "signature": "Laplace/a"},
        {"id": "q2", "status": "queued", "project": "Other", "signature": "Other/a"},
        {"id": "d1", "status": "done", "project": "Laplace-SMDP", "signature": "Laplace/done"},
    ]}
    events = []
    deps = _deps(state, events=events)

    cmd_cancel(_batch_args(project="Laplace-*"), deps=deps)

    assert [task["status"] for task in state["tasks"]] == ["queued", "queued", "done"]
    assert "preview-only without --confirm" in capsys.readouterr().out
    assert not any(event[0] == "save" for event in events)

    cmd_cancel(_batch_args(project="Laplace-*", confirm=True), deps=deps)

    assert [task["status"] for task in state["tasks"]] == ["cancelled", "queued", "done"]
    assert "cancelled 1 task(s) in batch" in capsys.readouterr().out


def test_batch_cancel_missing_id_is_atomic_without_ignore_missing():
    state = {"tasks": [{"id": "q1", "status": "queued"}]}
    events = []

    with pytest.raises(SystemExit, match="no tasks were cancelled"):
        cmd_cancel(_batch_args(["q1", "missing"]), deps=_deps(state, events=events))

    assert state["tasks"][0]["status"] == "queued"
    assert not any(event[0] == "save" for event in events)


def test_forget_marks_task_without_killing_processes(capsys):
    state = {"tasks": [{"id": "t1", "status": "running"}]}

    cmd_forget(Namespace(id="t1"), deps=_deps(state))

    assert state["tasks"][0]["status"] == "forgotten"
    assert state["tasks"][0]["finished_at"] == 123.0
    assert "forgot t1 (was running)" in capsys.readouterr().out


def test_clear_queue_dry_run_and_confirm(capsys):
    state = {"tasks": [
        {"id": "q1", "status": "queued"},
        {"id": "q2", "status": "queued"},
        {"id": "r1", "status": "running"},
    ]}
    events = []
    deps = _deps(state, events=events)

    cmd_clear_queue(Namespace(confirm=False), deps=deps)
    assert [task["status"] for task in state["tasks"]] == ["queued", "queued", "running"]
    assert "would cancel 2 queued tasks" in capsys.readouterr().out

    cmd_clear_queue(Namespace(confirm=True), deps=deps)
    assert [task["status"] for task in state["tasks"]] == ["cancelled", "cancelled", "running"]
    assert "cancelled 2 queued tasks" in capsys.readouterr().out
    assert any(event[0] == "notify" and event[1] == "clear_queue_cancelled" for event in events)
