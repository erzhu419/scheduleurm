from __future__ import annotations

from skill.scheduler_dispatch_launch_execution import (
    DispatchLaunchExecutionDeps,
    apply_dispatch_launch_execution,
)


def _deps(calls: list, *, launch_ok: bool = True, apply_ok: bool = True):
    def snapshot(state, nodes, task, node, gpu_idx, label):
        calls.append(("snapshot", label, node, gpu_idx))
        if label == "pre_launch":
            return {
                "label": label,
                "gpu": {"over_one_third": False},
                "node_ram": {"below_headroom": False},
            }
        return {
            "label": label,
            "gpu": {"over_one_third": True},
            "node_ram": {"below_headroom": True},
        }

    def debit(nodes, task):
        calls.append(("debit", task["id"]))
        nodes[0]["free_cpu"] -= int(task.get("cpu_cores") or 0)

    def save_state(state):
        calls.append(("save", len(state.get("tasks", []))))

    def launch(task, node_state=None):
        calls.append(("launch", task["id"], (node_state or {}).get("name")))
        return launch_ok, "pid=123" if launch_ok else "boom"

    def apply_launch_result(task, ok, msg, events):
        calls.append(("apply_result", ok, msg))
        if not apply_ok:
            events.append({"type": "launch_failed_retry", "task_id": task["id"]})
            return False
        task["status"] = "running"
        task["remote_pids"] = [123]
        events.append({"type": "launched", "task_id": task["id"], "msg": msg})
        return True

    return DispatchLaunchExecutionDeps(
        resource_snapshot_for_placement=snapshot,
        new_launch_token=lambda: "tok-1",
        debit_node_resources_for_launch=debit,
        task_run_identity=lambda task: task.get("identity"),
        save_state=save_state,
        launch=launch,
        apply_launch_result_to_task=apply_launch_result,
        notify=lambda *args, **kwargs: calls.append(("notify", args, kwargs)),
        now=lambda: 123.0,
    )


def test_apply_dispatch_launch_execution_defers_launch_intent_without_launching():
    calls = []
    task = {
        "id": "t1",
        "node": "node001",
        "gpu_idx": 0,
        "cpu_cores": 2,
        "identity": "run-1",
    }
    state = {"tasks": [task]}
    nodes = [{"name": "node001", "free_cpu": 8}]
    events = []
    running_keys = set()

    ok = apply_dispatch_launch_execution(
        task,
        state=state,
        nodes=nodes,
        picked_state={"name": "node001"},
        running_keys=running_keys,
        events=events,
        defer_launches=True,
        deps=_deps(calls),
    )

    assert ok is True
    assert task["status"] == "launching"
    assert task["launch_token"] == "tok-1"
    assert task["launching_started_at"] == 123.0
    assert nodes[0]["free_cpu"] == 6
    assert running_keys == {"run-1"}
    assert [ev["type"] for ev in events] == ["pre_launch_snapshot", "launch_intent"]
    intent = events[1]
    assert intent["task"] is not task
    assert intent["task"]["launch_token"] == "tok-1"
    assert intent["post_launch_snapshot"]["crossed_one_third_from_pre"] is True
    assert intent["post_launch_snapshot"]["crossed_ram_headroom_from_pre"] is True
    assert "last_launch_post_snapshot" not in task
    assert ("launch", "t1", "node001") not in calls
    assert not any(call[0] == "save" for call in calls)


def test_apply_dispatch_launch_execution_success_launches_saves_and_debits():
    calls = []
    task = {
        "id": "t2",
        "node": "node002",
        "gpu_idx": 1,
        "cpu_cores": 3,
        "identity": "run-2",
    }
    state = {"tasks": [task]}
    nodes = [{"name": "node002", "free_cpu": 10}]
    events = []
    running_keys = set()

    ok = apply_dispatch_launch_execution(
        task,
        state=state,
        nodes=nodes,
        picked_state={"name": "node002"},
        running_keys=running_keys,
        events=events,
        defer_launches=False,
        deps=_deps(calls),
    )

    assert ok is True
    assert task["status"] == "running"
    assert task["remote_pids"] == [123]
    assert task["last_launch_pre_snapshot"]["label"] == "pre_launch"
    assert task["last_launch_post_snapshot"]["label"] == "post_launch_estimated"
    assert task["last_resource_snapshot"] == task["last_launch_post_snapshot"]
    assert task["last_launch_post_snapshot"]["crossed_one_third_from_pre"] is True
    assert task["last_launch_post_snapshot"]["crossed_ram_headroom_from_pre"] is True
    assert running_keys == {"run-2"}
    assert nodes[0]["free_cpu"] == 7
    assert [ev["type"] for ev in events] == [
        "pre_launch_snapshot",
        "launched",
        "post_launch_snapshot",
    ]
    assert [call[0] for call in calls].count("save") == 2
    assert ("launch", "t2", "node002") in calls
    assert ("debit", "t2") in calls


def test_apply_dispatch_launch_execution_launch_failure_does_not_debit():
    calls = []
    task = {
        "id": "t3",
        "node": "node003",
        "gpu_idx": None,
        "cpu_cores": 4,
        "identity": "run-3",
    }
    state = {"tasks": [task]}
    nodes = [{"name": "node003", "free_cpu": 10}]
    events = []
    running_keys = set()

    ok = apply_dispatch_launch_execution(
        task,
        state=state,
        nodes=nodes,
        picked_state={"name": "node003"},
        running_keys=running_keys,
        events=events,
        defer_launches=False,
        deps=_deps(calls, launch_ok=False, apply_ok=False),
    )

    assert ok is False
    assert nodes[0]["free_cpu"] == 10
    assert running_keys == set()
    assert [ev["type"] for ev in events] == [
        "pre_launch_snapshot",
        "launch_failed_retry",
    ]
    assert [call[0] for call in calls].count("save") == 1
    assert ("debit", "t3") not in calls
