from __future__ import annotations

from contextlib import contextmanager
import time

from skill.scheduler_failure.terminal_snapshot import (
    TerminalDiagnosticsSnapshotDeps,
    terminal_diagnostics_snapshot_outside_lock,
)


def test_terminal_diagnostics_snapshot_precomputes_after_releasing_state_lock():
    events = []
    lock_depth = {"value": 0}
    state = {
        "tasks": [
            {
                "id": "t1",
                "status": "running",
                "node": "node001",
                "log_path": "/tmp/t1.log",
                "started_at": 10.0,
                "cmd": "python train.py",
            }
        ]
    }

    @contextmanager
    def state_lock(**kwargs):
        events.append(("lock_enter", kwargs))
        lock_depth["value"] += 1
        try:
            yield
        finally:
            lock_depth["value"] -= 1
            events.append(("lock_exit", None))

    def diagnose(task):
        events.append(("diagnose", lock_depth["value"], task["id"]))
        return {"is_crash": False, "reason": "done"}

    out = terminal_diagnostics_snapshot_outside_lock(
        {"t1": {"state": "dead"}},
        {"t1"},
        "unit",
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: state,
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=diagnose,
            discover_result_artifacts=lambda task, include_log=True: ["log"] if include_log else [],
            notify=lambda *args, **kwargs: None,
            now=lambda: 100.0,
        ),
    )

    assert out["t1"]["diagnosis"] == {"is_crash": False, "reason": "done"}
    assert out["t1"]["result_artifacts"] == ["log"]
    assert events[0] == ("lock_enter", {"shared": True, "purpose": "unit:snapshot"})
    assert events[1] == ("lock_exit", None)
    assert events[2] == ("diagnose", 0, "t1")


def test_terminal_diagnostics_snapshot_limits_dead_tasks_and_can_skip_log_artifacts():
    state = {
        "tasks": [
            {
                "id": "t1",
                "status": "running",
                "node": "node001",
                "log_path": "/tmp/t1.log",
                "started_at": 10.0,
                "cmd": "python train.py",
            },
            {
                "id": "t2",
                "status": "running",
                "node": "node001",
                "log_path": "/tmp/t2.log",
                "started_at": 11.0,
                "cmd": "python train.py",
            },
        ]
    }
    artifact_calls = []

    @contextmanager
    def state_lock(**kwargs):
        yield

    out = terminal_diagnostics_snapshot_outside_lock(
        {"t1": {"state": "dead"}, "t2": {"state": "dead"}},
        {"t1", "t2"},
        "unit",
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: state,
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=lambda task: {"is_crash": False, "reason": task["id"]},
            discover_result_artifacts=lambda task, include_log=True: (
                artifact_calls.append((task["id"], include_log)) or []
            ),
            notify=lambda *args, **kwargs: None,
            now=lambda: 100.0,
            max_tasks_per_cycle=1,
            include_log_result_artifacts=False,
        ),
    )

    assert set(out) == {"t1"}
    assert out["t1"]["diagnosis"]["reason"] == "t1"
    assert artifact_calls == [("t1", False)]


def test_terminal_diagnostics_snapshot_prioritizes_already_deferred_tasks():
    state = {
        "tasks": [
            {"id": "fresh", "status": "running", "started_at": 10.0},
            {
                "id": "deferred",
                "status": "running",
                "started_at": 11.0,
                "terminal_transition_deferred": True,
                "terminal_transition_deferred_at": 90.0,
            },
        ]
    }

    @contextmanager
    def state_lock(**_kwargs):
        yield

    out = terminal_diagnostics_snapshot_outside_lock(
        {"fresh": {"state": "dead"}, "deferred": {"state": "dead"}},
        {"fresh", "deferred"},
        "unit",
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: state,
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=lambda task: {"is_crash": False, "reason": task["id"]},
            discover_result_artifacts=lambda task, include_log=True: [],
            notify=lambda *args, **kwargs: None,
            now=lambda: 100.0,
            max_tasks_per_cycle=1,
        ),
    )

    assert set(out) == {"deferred"}


def test_terminal_diagnostics_snapshot_finalizes_all_decisive_batch_evidence():
    tasks = [
        {
            "id": f"t{index}",
            "status": "running",
            "node": "node001",
            "log_path": f"/tmp/t{index}.log",
            "started_at": 10.0,
            "cmd": "python work.py && echo DONE",
        }
        for index in range(3)
    ]

    @contextmanager
    def state_lock(**_kwargs):
        yield

    out = terminal_diagnostics_snapshot_outside_lock(
        {task["id"]: {"state": "dead"} for task in tasks},
        {task["id"] for task in tasks},
        "unit",
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: {"tasks": tasks},
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=lambda task: {
                "is_crash": False,
                "reason": "done",
                "success_marker": "DONE",
            },
            discover_result_artifacts=lambda task, include_log=True: [],
            notify=lambda *args, **kwargs: None,
            now=lambda: 100.0,
            max_tasks_per_cycle=1,
            collect_terminal_evidence=lambda selected, **_kwargs: {
                task["id"]: {
                    "log_path": task["log_path"],
                    "tail": "DONE\n",
                    "log_size": 5,
                    "artifact_checked": True,
                }
                for task in selected
            },
        ),
    )

    assert set(out) == {"t0", "t1", "t2"}


def test_terminal_diagnostics_snapshot_batches_backend_completed_tasks():
    tasks = [
        {
            "id": f"completed-{index}",
            "status": "running",
            "node": "node001",
            "log_path": f"/tmp/completed-{index}.log",
            "started_at": 10.0,
            "cmd": "python work.py && echo DONE",
        }
        for index in range(3)
    ]

    @contextmanager
    def state_lock(**_kwargs):
        yield

    out = terminal_diagnostics_snapshot_outside_lock(
        {
            task["id"]: {"state": "dead", "terminal_ok": True}
            for task in tasks
        },
        {task["id"] for task in tasks},
        "unit",
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: {"tasks": tasks},
            scan_completed_log_for_crash=lambda task: (_ for _ in ()).throw(
                AssertionError("decisive batch evidence must avoid per-task scans")
            ),
            diagnose_terminal=lambda task: {
                "is_crash": False,
                "reason": "normal exit (success marker found)",
                "success_marker": "DONE",
            },
            discover_result_artifacts=lambda task, include_log=True: (
                _ for _ in ()
            ).throw(
                AssertionError("batch evidence already carries result artifacts")
            ),
            notify=lambda *args, **kwargs: None,
            now=lambda: 100.0,
            max_tasks_per_cycle=1,
            collect_terminal_evidence=lambda selected, **_kwargs: {
                task["id"]: {
                    "log_path": task["log_path"],
                    "tail": "DONE\n",
                    "log_size": 5,
                    "artifact_checked": True,
                    "result_artifacts": [{"path": f"/results/{task['id']}"}],
                }
                for task in selected
            },
        ),
    )

    assert set(out) == {task["id"] for task in tasks}
    assert all(payload["completed_log_crash"] == (False, "") for payload in out.values())
    assert all(payload["result_artifacts"] for payload in out.values())


def test_terminal_diagnostics_snapshot_skips_unknown_without_precomputed_safety():
    tasks = [
        {
            "id": f"unknown-{index}",
            "status": "running",
            "node": "node001",
            "log_path": f"/tmp/unknown-{index}.log",
            "started_at": 10.0,
            "cmd": "python work.py",
            "reroute_on_node_down": True,
            "probe_unknown_since": 1.0,
        }
        for index in range(4)
    ]
    selected_ids = []

    @contextmanager
    def state_lock(**_kwargs):
        yield

    terminal_diagnostics_snapshot_outside_lock(
        {task["id"]: {"state": "unknown"} for task in tasks},
        {task["id"] for task in tasks},
        "unit",
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: {"tasks": tasks},
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=lambda task: {
                "is_crash": False,
                "reason": "done",
            },
            discover_result_artifacts=lambda task, include_log=True: [],
            notify=lambda *args, **kwargs: None,
            now=lambda: 100.0,
            max_tasks_per_cycle=2,
            node_down_requeue_s=30,
            collect_terminal_evidence=lambda selected, **_kwargs: (
                selected_ids.extend(task["id"] for task in selected) or {}
            ),
        ),
    )

    assert selected_ids == []


def test_terminal_diagnostics_snapshot_limits_prevalidated_unknown_before_remote_io():
    tasks = [
        {
            "id": f"unknown-{index}",
            "status": "running",
            "node": "node001",
            "log_path": f"/tmp/unknown-{index}.log",
            "started_at": 10.0,
            "cmd": "python work.py",
            "reroute_on_node_down": True,
            "probe_unknown_since": 1.0,
        }
        for index in range(4)
    ]
    selected_ids = []

    @contextmanager
    def state_lock(**_kwargs):
        yield

    terminal_diagnostics_snapshot_outside_lock(
        {
            task["id"]: {
                "state": "unknown",
                "launch_safety_state": "dead",
            }
            for task in tasks
        },
        {task["id"] for task in tasks},
        "unit",
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: {"tasks": tasks},
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=lambda task: {
                "is_crash": False,
                "reason": "done",
            },
            discover_result_artifacts=lambda task, include_log=True: [],
            notify=lambda *args, **kwargs: None,
            now=lambda: 100.0,
            max_tasks_per_cycle=2,
            node_down_requeue_s=30,
            collect_terminal_evidence=lambda selected, **_kwargs: (
                selected_ids.extend(task["id"] for task in selected) or {}
            ),
        ),
    )

    assert selected_ids == ["unknown-0", "unknown-1"]


def test_terminal_diagnostics_snapshot_passes_remaining_wall_budget_to_batch():
    seen_timeouts = []

    @contextmanager
    def state_lock(**_kwargs):
        yield

    terminal_diagnostics_snapshot_outside_lock(
        {"t1": {"state": "dead"}},
        {"t1"},
        "unit",
        deps=TerminalDiagnosticsSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: {
                "tasks": [{"id": "t1", "status": "running", "node": "node001"}]
            },
            scan_completed_log_for_crash=lambda task: (False, ""),
            diagnose_terminal=lambda task: {"is_crash": False},
            discover_result_artifacts=lambda task, include_log=True: [],
            notify=lambda *args, **kwargs: None,
            now=time.monotonic,
            max_wall_s=0.25,
            collect_terminal_evidence=lambda tasks, *, timeout_s: (
                seen_timeouts.append(timeout_s) or {}
            ),
        ),
    )

    assert len(seen_timeouts) == 1
    assert 0.0 < seen_timeouts[0] <= 0.25
