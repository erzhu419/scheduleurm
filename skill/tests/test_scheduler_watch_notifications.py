from __future__ import annotations

from skill.scheduler_watch_notifications import (
    crashed_task_payload,
    notify_dispatch_events,
    notify_terminal_transitions,
)


def test_notify_terminal_transitions_emits_done_and_crashed_payloads():
    calls = []

    def notify(event_type, payload, **kwargs):
        calls.append((event_type, payload, kwargs))

    done = {"id": "t1001", "project": "done-project"}
    crashed = {
        "id": "t1002",
        "project": "crash-project",
        "description": "failed run",
        "node": "node001",
        "gpu_idx": 2,
        "_diagnosis": {
            "lifetime_s": 42,
            "log_size": 2048,
            "log_path": "/tmp/sched_t1002.log",
            "reason": "RuntimeError",
            "tail": "traceback tail",
        },
    }

    notify_terminal_transitions(
        [done],
        [crashed],
        notify_fn=notify,
        compact_task_payload_fn=lambda task: {"compact_id": task["id"]},
    )

    assert calls == [
        ("task_done", {"compact_id": "t1001"}, {}),
        (
            "task_crashed",
            {
                "id": "t1002",
                "project": "crash-project",
                "description": "failed run",
                "node": "node001",
                "gpu_idx": 2,
                "lifetime_s": 42,
                "log_size": 2048,
                "log_path": "/tmp/sched_t1002.log",
                "reason": "RuntimeError",
                "tail": "traceback tail",
            },
            {},
        ),
    ]


def test_crashed_task_payload_tolerates_missing_or_none_diagnosis():
    payload = crashed_task_payload({
        "id": "t1003",
        "project": "p",
        "_diagnosis": None,
    })

    assert payload == {
        "id": "t1003",
        "project": "p",
        "description": "",
        "node": None,
        "gpu_idx": None,
        "lifetime_s": 0,
        "log_size": 0,
        "log_path": None,
        "reason": "?",
        "tail": "",
    }


def test_notify_dispatch_events_emits_existing_watcher_mappings():
    calls = []

    def notify(event_type, payload, **kwargs):
        calls.append((event_type, payload, kwargs))

    def compact(task):
        return {"compact_id": task["id"]}

    notify_dispatch_events(
        [
            {"type": "launched", "task": {"id": "t0001"}},
            {"type": "pre_launch_snapshot", "snapshot": {"phase": "pre"}},
            {"type": "post_launch_snapshot", "snapshot": None},
            {"type": "blocked", "task_id": "t0002", "reason": "no fit"},
            {
                "type": "launch_failed_retry",
                "task_id": "t0003",
                "error": "ssh failed",
                "task": {"launch_fail_count": 2},
            },
            {"type": "launch_failed_terminal", "task_id": "t0004", "error": "gave up"},
            {
                "type": "migrated",
                "task_id": "t0005",
                "from_node": "node001",
                "to_node": "node002",
                "eta_seconds": 30,
                "reason": "faster",
            },
            {
                "type": "preempted",
                "task_id": "t0006",
                "freed_node": "node003",
                "cpu_freed": 4,
                "ram_freed": 1024,
            },
            {"type": "claim_race", "task_id": "t0007", "reason": "x" * 350},
            {"type": "launch_commit_skipped", "task_id": "t0008", "reason": "y" * 350},
        ],
        notify_fn=notify,
        compact_task_payload_fn=compact,
    )

    assert calls == [
        ("task_launched", {"compact_id": "t0001"}, {}),
        ("pre_launch_snapshot", {"phase": "pre"}, {"feishu_enabled": False}),
        ("post_launch_snapshot", {}, {"feishu_enabled": False}),
        ("task_blocked", {"task_id": "t0002", "reason": "no fit"}, {}),
        (
            "task_launch_retry",
            {"task_id": "t0003", "error": "ssh failed", "attempt": 2},
            {},
        ),
        ("task_failed", {"task_id": "t0004", "error": "gave up"}, {}),
        (
            "task_migrated",
            {
                "task_id": "t0005",
                "from_node": "node001",
                "to_node": "node002",
                "eta_seconds": 30,
                "reason": "faster",
            },
            {},
        ),
        (
            "task_preempted",
            {
                "task_id": "t0006",
                "freed_node": "node003",
                "cpu_freed": 4,
                "ram_freed": 1024,
            },
            {},
        ),
        (
            "task_claim_race",
            {"task_id": "t0007", "reason": "x" * 300},
            {"feishu_enabled": False},
        ),
        (
            "task_launch_commit_skipped",
            {"task_id": "t0008", "reason": "y" * 300},
            {"feishu_enabled": False},
        ),
    ]


def test_notify_dispatch_events_ignores_noisy_internal_events():
    calls = []

    notify_dispatch_events(
        [
            {"type": "no_fit", "task_id": "t0009"},
            {"type": "resume_found", "task_id": "t0010"},
        ],
        notify_fn=lambda *args, **kwargs: calls.append((args, kwargs)),
        compact_task_payload_fn=lambda task: task,
    )

    assert calls == []
