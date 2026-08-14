from __future__ import annotations

from skill.scheduler_watch_state_phase import (
    WatchStatePhaseDeps,
    run_watch_state_phase,
)


def test_run_watch_state_phase_preserves_watcher_state_side_effects():
    calls = []
    phases = []
    notifications = []
    state = {
        "tasks": [
            {"id": "done", "status": "running", "description": "finished"},
            {"id": "crash", "status": "failed", "description": "crashed"},
            {
                "id": "evicted",
                "status": "running",
                "last_resource_eviction": {"id": "evicted", "why": "gpu"},
            },
        ]
    }
    nodes = [{"name": "node001"}]

    def record_phase(name, _started_at, **payload):
        phases.append((name, payload))

    def update_running_tasks(s, **_kwargs):
        calls.append("update_running_tasks")
        s["tasks"][0]["status"] = "done"

    def classify(tasks, pre_status):
        assert pre_status["done"] == "running"
        return ([tasks[0]], [tasks[1]])

    def build_crash_payload(task, _state, _nodes):
        return {"id": task["id"], "kind": "oom" if task["id"] == "crash" else "other"}

    def do_dispatch(_state, _nodes, defer_launches=False, max_queued=None):
        assert defer_launches is True
        assert max_queued == 80
        return (
            [
                {"type": "launched", "task_id": "new", "task": {"id": "new"}},
                {"type": "blocked", "task_id": "blocked"},
            ],
            2,
        )

    deps = WatchStatePhaseDeps(
        recover_queued_live_local_tasks=lambda _state: 1,
        update_running_tasks=update_running_tasks,
        seed_pending_eta_from_history=lambda _state: calls.append("seed_eta"),
        detect_oom_kills_local=lambda _state: [],
        requeue_after_crash=lambda _task, _state: None,
        notify=lambda *args, **kwargs: notifications.append((args, kwargs)),
        classify_running_terminal_transitions=classify,
        remember_running_resource_snapshots=lambda _state, _nodes: calls.append("remember"),
        reconcile_aggregate_only_vram=lambda _state, _nodes: calls.append("reconcile_vram"),
        build_crash_forensics_payload=build_crash_payload,
        build_oom_forensics_payload=lambda payload, _state: {**payload, "oom": True},
        is_oom_like_forensics=lambda payload: payload.get("kind") == "oom",
        enforce_post_dispatch_thresholds=lambda _state, _nodes: ["evicted"],
        reserve_inflight_vram=lambda _state, _nodes: calls.append("reserve"),
        do_dispatch=do_dispatch,
        reconcile_external_tasks=lambda _state, probe_data=None: [{"id": "adopted"}],
        archive_terminal_tasks=lambda _state: 3,
        detect_batch_completions=lambda _state, ids: [{"ids": list(ids)}],
        save_state=lambda _state: calls.append("save_state"),
        watch_dispatch_max_queued_per_cycle=80,
        now=lambda: 100.0,
    )

    result = run_watch_state_phase(
        state,
        nodes,
        recovered_launching_count=5,
        running_probe_results={"done": {"state": "dead"}},
        running_probe_ids={"done"},
        eta_tail_outputs={},
        eta_tail_ids=set(),
        terminal_diagnostics={},
        should_auto_adopt=True,
        auto_adopt_probe_data={"node001": []},
        defer_launches=True,
        record_phase=record_phase,
        deps=deps,
    )

    assert result.recovered_launching_count == 5
    assert result.recovered_queued_live_count == 1
    assert result.qcount == 2
    assert result.events[0]["task"]["notified_launch"] is True
    assert result.auto_adopted == [{"id": "adopted"}]
    assert result.auto_adopt_completed is True
    assert result.archived_count == 3
    assert result.batch_completions == [{"ids": ["done", "crash"]}]
    assert result.resource_evictions == [{"id": "evicted", "why": "gpu"}]
    assert state["tasks"][1]["last_crash_forensics"] == {"id": "crash", "kind": "oom"}
    assert state["tasks"][1]["last_oom_forensics"] == {
        "id": "crash",
        "kind": "oom",
        "oom": True,
    }
    assert notifications == []
    assert calls[-1] == "save_state"
    assert calls.count("save_state") == 2
    assert [name for name, _payload in phases] == [
        "recover_stale",
        "update_running_tasks",
        "save_after_running_update",
        "seed_pending_eta",
        "detect_oom_kills",
        "classify_transitions",
        "resource_and_forensics",
        "evict_and_reserve",
        "do_dispatch",
        "mark_notified_auto_adopt",
        "archive_terminal",
        "batch_completions",
        "save_state",
    ]


def test_run_watch_state_phase_requeues_oom_flipped_tasks_and_notifies():
    notifications = []
    oom_task = {
        "id": "oom",
        "status": "failed",
        "started_at": 10,
        "finished_at": 40,
        "description": "oom task",
    }
    state = {"tasks": [oom_task]}

    deps = WatchStatePhaseDeps(
        recover_queued_live_local_tasks=lambda _state: 0,
        update_running_tasks=lambda _state, **_kwargs: None,
        seed_pending_eta_from_history=lambda _state: None,
        detect_oom_kills_local=lambda _state: [oom_task],
        requeue_after_crash=lambda _task, _state: "t-retry",
        notify=lambda *args, **kwargs: notifications.append((args, kwargs)),
        classify_running_terminal_transitions=lambda _tasks, _pre: ([], []),
        remember_running_resource_snapshots=lambda _state, _nodes: None,
        reconcile_aggregate_only_vram=lambda _state, _nodes: None,
        build_crash_forensics_payload=lambda _task, _state, _nodes: {},
        build_oom_forensics_payload=lambda _payload, _state: {},
        is_oom_like_forensics=lambda _payload: False,
        enforce_post_dispatch_thresholds=lambda _state, _nodes: [],
        reserve_inflight_vram=lambda _state, _nodes: None,
        do_dispatch=lambda _state, _nodes, defer_launches=False, max_queued=None: ([], 0),
        reconcile_external_tasks=lambda _state, probe_data=None: [],
        archive_terminal_tasks=lambda _state: 0,
        detect_batch_completions=lambda _state, _ids: [],
        save_state=lambda _state: None,
        now=lambda: 1.0,
    )

    run_watch_state_phase(
        state,
        [],
        recovered_launching_count=0,
        running_probe_results=None,
        running_probe_ids=None,
        eta_tail_outputs=None,
        eta_tail_ids=None,
        terminal_diagnostics=None,
        should_auto_adopt=False,
        auto_adopt_probe_data=None,
        defer_launches=False,
        record_phase=lambda *_args, **_kwargs: None,
        deps=deps,
    )

    assert oom_task["requeued_as"] == "t-retry"
    assert notifications[0][0][0] == "task_oom_requeue"
    assert notifications[0][0][1]["lifetime_s"] == 30
    assert notifications[0][1] == {"feishu_enabled": False}


def test_run_watch_state_phase_ignores_deferred_crash_requeue_as_terminal():
    task = {
        "id": "deferred",
        "status": "running",
        "node": "node001",
        "remote_pids": [123],
    }
    state = {"tasks": [task]}

    def requeue_after_crash(task_arg, _state):
        task_arg["status"] = "running"
        task_arg["node_probe_state"] = "unknown"
        task_arg["last_block_reason"] = "auto-requeue deferred"
        return None

    def update_running_tasks(s, **_kwargs):
        task_arg = s["tasks"][0]
        task_arg["status"] = "failed"
        task_arg["last_block_reason"] = "crash before requeue"
        requeue_after_crash(task_arg, s)

    deps = WatchStatePhaseDeps(
        recover_queued_live_local_tasks=lambda _state: 0,
        update_running_tasks=update_running_tasks,
        seed_pending_eta_from_history=lambda _state: None,
        detect_oom_kills_local=lambda _state: [],
        requeue_after_crash=requeue_after_crash,
        notify=lambda *args, **kwargs: None,
        classify_running_terminal_transitions=lambda tasks, pre: (
            [],
            [
                task_arg
                for task_arg in tasks
                if pre.get(task_arg.get("id")) == "running" and task_arg.get("status") == "failed"
            ],
        ),
        remember_running_resource_snapshots=lambda _state, _nodes: None,
        reconcile_aggregate_only_vram=lambda _state, _nodes: None,
        build_crash_forensics_payload=lambda _task, _state, _nodes: {},
        build_oom_forensics_payload=lambda _payload, _state: {},
        is_oom_like_forensics=lambda _payload: False,
        enforce_post_dispatch_thresholds=lambda _state, _nodes: [],
        reserve_inflight_vram=lambda _state, _nodes: None,
        do_dispatch=lambda _state, _nodes, defer_launches=False, max_queued=None: ([], 0),
        reconcile_external_tasks=lambda _state, probe_data=None: [],
        archive_terminal_tasks=lambda _state: 0,
        detect_batch_completions=lambda _state, _ids: [],
        save_state=lambda _state: None,
        now=lambda: 1.0,
    )

    result = run_watch_state_phase(
        state,
        [],
        recovered_launching_count=0,
        running_probe_results={"deferred": {"state": "dead"}},
        running_probe_ids={"deferred"},
        eta_tail_outputs=None,
        eta_tail_ids=None,
        terminal_diagnostics=None,
        should_auto_adopt=False,
        auto_adopt_probe_data=None,
        defer_launches=True,
        record_phase=lambda *args, **kwargs: None,
        deps=deps,
    )

    assert task["status"] == "running"
    assert task["node_probe_state"] == "unknown"
    assert result.newly_crashed == []
