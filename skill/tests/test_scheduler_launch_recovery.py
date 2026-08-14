from contextlib import contextmanager

from skill.scheduler_launch_recovery import (
    LaunchRecoveryDeps,
    StaleLaunchRecoveryDeps,
    recover_stale_launching_tasks,
    recover_stale_launching_tasks_outside_lock,
)


def _deps(*, orphan=False, terminal=False, claim_enabled=True, calls=None, now=1000.0):
    calls = calls if calls is not None else []

    return LaunchRecoveryDeps(
        try_recover_orphan_local_task=lambda task, node: calls.append(("orphan", task["id"], node)) or orphan,
        try_finalize_terminal_local_task=lambda task, node, state: calls.append(("terminal", task["id"], node)) or terminal,
        claim_enabled_for=lambda node: claim_enabled,
        release_task_claims_and_intents=lambda task, **kwargs: calls.append(("release", task["id"], kwargs)),
        clear_live_eta_fields=lambda task, **kwargs: calls.append(("clear_eta", task["id"], kwargs)),
        now=lambda: now,
    )


def test_recent_launching_task_is_left_alone():
    calls = []
    task = {"id": "t1", "status": "launching", "node": "node001", "launching_started_at": 990.0}
    state = {"tasks": [task]}

    reverted = recover_stale_launching_tasks(
        state,
        reset_s=30,
        deps=_deps(calls=calls, now=1000.0),
    )

    assert reverted == 0
    assert task["status"] == "launching"
    assert calls == []


def test_stale_launching_orphan_adoption_wins_before_revert():
    calls = []
    task = {"id": "t1", "status": "launching", "node": "node001", "launching_started_at": 900.0}

    reverted = recover_stale_launching_tasks(
        {"tasks": [task]},
        reset_s=30,
        deps=_deps(orphan=True, calls=calls, now=1000.0),
    )

    assert reverted == 0
    assert task["status"] == "launching"
    assert calls == [("orphan", "t1", "node001")]


def test_stale_launching_terminal_finalize_wins_before_revert():
    calls = []
    task = {"id": "t1", "status": "launching", "node": "node001", "launching_started_at": 900.0}

    reverted = recover_stale_launching_tasks(
        {"tasks": [task]},
        reset_s=30,
        deps=_deps(terminal=True, calls=calls, now=1000.0),
    )

    assert reverted == 0
    assert task["status"] == "launching"
    assert calls == [
        ("orphan", "t1", "node001"),
        ("terminal", "t1", "node001"),
    ]


def test_stale_launching_reverts_and_releases_claim():
    calls = []
    task = {
        "id": "t1",
        "status": "launching",
        "node": "node001",
        "launching_started_at": 900.0,
        "launch_token": "abc",
        "eta_seconds": 123,
    }

    reverted = recover_stale_launching_tasks(
        {"tasks": [task]},
        reset_s=30,
        deps=_deps(calls=calls, now=1000.0),
    )

    assert reverted == 1
    assert task["status"] == "queued"
    assert "100s" in task["last_block_reason"]
    assert "launching_started_at" not in task
    assert "launch_token" not in task
    assert calls == [
        ("orphan", "t1", "node001"),
        ("terminal", "t1", "node001"),
        ("release", "t1", {"extra_nodes": ["node001"]}),
        ("clear_eta", "t1", {"clear_runtime_projection": True}),
    ]


def test_revert_skips_claim_release_when_claims_disabled():
    calls = []
    task = {"id": "t1", "status": "launching", "node": "node001", "launching_started_at": 900.0}

    reverted = recover_stale_launching_tasks(
        {"tasks": [task]},
        reset_s=30,
        deps=_deps(claim_enabled=False, calls=calls, now=1000.0),
    )

    assert reverted == 1
    assert ("release", "t1", {"extra_nodes": ["node001"]}) not in calls


def _outside_deps(state, evidence_by_node, *, calls, scan_hook=None):
    held = {"depth": 0}

    @contextmanager
    def state_lock(**kwargs):
        calls.append(("lock_enter", kwargs))
        held["depth"] += 1
        try:
            yield
        finally:
            held["depth"] -= 1
            calls.append(("lock_exit", kwargs))

    def scan_node(node, tasks):
        calls.append(("scan", node, [task["id"] for task in tasks], held["depth"]))
        assert held["depth"] == 0
        if scan_hook is not None:
            scan_hook(node, tasks)
        result = evidence_by_node[node]
        if isinstance(result, BaseException):
            raise result
        return result

    def adopt(task, node, rows, prior_status):
        calls.append(("adopt", task["id"], node, rows, prior_status))
        task["status"] = "running"
        task["remote_pids"] = [int(rows[0]["pid"])]
        task.pop("launching_started_at", None)
        task.pop("launch_token", None)
        return True

    def set_usage(task, vram, ram, pcpu):
        calls.append(("usage", task["id"], vram, ram, pcpu))

    return StaleLaunchRecoveryDeps(
        state_lock=state_lock,
        load_state=lambda: calls.append(("load", held["depth"])) or state,
        save_state=lambda saved: calls.append(("save", saved)),
        scan_node_evidence=scan_node,
        adopt_live_pid_rows=adopt,
        node_is_windows=lambda node: node.startswith("win"),
        claim_enabled_for=lambda _node: False,
        release_task_claims_and_intents=lambda *args, **kwargs: calls.append(
            ("release", args, kwargs)
        ),
        clear_live_eta_fields=lambda task, **kwargs: calls.append(
            ("clear_eta", task["id"], kwargs)
        ),
        set_current_usage=set_usage,
        notify=lambda event, payload, **kwargs: calls.append(
            ("notify", event, payload, kwargs)
        ),
        now=lambda: 1000.0,
        max_workers=8,
    )


def test_outside_lock_recovery_scans_one_time_per_node_for_large_batch():
    calls = []
    tasks = [
        {
            "id": f"t{index}",
            "status": "launching",
            "node": "node001",
            "launch_token": f"token-{index}",
            "launching_started_at": 900.0,
        }
        for index in range(100)
    ]
    state = {"tasks": tasks}
    evidence = {
        "node001": {
            "ok": True,
            "supported": True,
            "rows_by_task": {},
            "exit_statuses": {},
        }
    }

    reverted = recover_stale_launching_tasks_outside_lock(
        reset_s=30,
        deps=_outside_deps(state, evidence, calls=calls),
        purpose="test-recovery",
    )

    assert reverted == 100
    scans = [call for call in calls if call[0] == "scan"]
    assert len(scans) == 1
    assert scans[0][1] == "node001"
    assert scans[0][3] == 0
    assert len(scans[0][2]) == 100
    assert all(task["status"] == "queued" for task in tasks)
    assert [call[1]["purpose"] for call in calls if call[0] == "lock_enter"] == [
        "test-recovery:snapshot",
        "test-recovery:commit",
    ]
    assert len([call for call in calls if call[0] == "save"]) == 1


def test_outside_lock_recovery_merges_live_exit_missing_and_probe_error_safely():
    calls = []
    tasks = [
        {"id": "live", "status": "launching", "node": "node001", "launch_token": "live-token", "launching_started_at": 900.0},
        {"id": "done", "status": "launching", "node": "node001", "launch_token": "done-token", "launching_started_at": 900.0},
        {"id": "missing", "status": "launching", "node": "node001", "launch_token": "missing-token", "launching_started_at": 900.0},
        {"id": "error", "status": "launching", "node": "node002", "launch_token": "error-token", "launching_started_at": 900.0},
        {"id": "windows", "status": "launching", "node": "win001", "launch_token": "win-token", "launching_started_at": 900.0},
    ]
    state = {"tasks": tasks}
    evidence = {
        "node001": {
            "ok": True,
            "supported": True,
            "rows_by_task": {"live": [{"pid": "101"}]},
            "exit_statuses": {
                "done": {
                    "exit_code": 0,
                    "finished_at": 950.0,
                    "token": "done-token",
                }
            },
        },
        "node002": RuntimeError("probe unavailable"),
    }

    reverted = recover_stale_launching_tasks_outside_lock(
        reset_s=30,
        deps=_outside_deps(state, evidence, calls=calls),
    )

    by_id = {task["id"]: task for task in tasks}
    assert reverted == 1
    assert by_id["live"]["status"] == "running"
    assert by_id["done"]["status"] == "running"
    assert by_id["done"]["exit_code"] == 0
    assert by_id["missing"]["status"] == "queued"
    assert by_id["error"]["status"] == "launching"
    assert by_id["windows"]["status"] == "launching"
    payload = next(
        call[2]
        for call in calls
        if call[0] == "notify" and call[1] == "launching_recovery_batch"
    )
    assert payload == {
        "candidates": 5,
        "nodes": 3,
        "adopted": 1,
        "terminal_sentinel": 1,
        "reverted": 1,
        "deferred": 2,
        "identity_changed": 0,
        "errors": {"node002": "RuntimeError: probe unavailable"},
    }


def test_outside_lock_recovery_skips_commit_when_launch_identity_changed():
    calls = []
    task = {
        "id": "race",
        "status": "launching",
        "node": "node001",
        "launch_token": "old-token",
        "launching_started_at": 900.0,
    }
    state = {"tasks": [task]}
    evidence = {
        "node001": {
            "ok": True,
            "supported": True,
            "rows_by_task": {},
            "exit_statuses": {},
        }
    }

    def replace_launch(_node, _tasks):
        task["launch_token"] = "new-token"

    reverted = recover_stale_launching_tasks_outside_lock(
        reset_s=30,
        deps=_outside_deps(
            state,
            evidence,
            calls=calls,
            scan_hook=replace_launch,
        ),
    )

    assert reverted == 0
    assert task["status"] == "launching"
    assert task["launch_token"] == "new-token"
    assert not any(call[0] == "save" for call in calls)
    payload = next(call[2] for call in calls if call[:2] == ("notify", "launching_recovery_batch"))
    assert payload["identity_changed"] == 1


def test_outside_lock_recovery_commits_sentinel_when_process_absence_is_unknown():
    calls = []
    tasks = [
        {"id": "done", "status": "launching", "node": "node001", "launch_token": "done-token", "launching_started_at": 900.0},
        {"id": "unknown", "status": "launching", "node": "node001", "launch_token": "unknown-token", "launching_started_at": 900.0},
    ]
    state = {"tasks": tasks}
    evidence = {
        "node001": {
            "ok": True,
            "supported": True,
            "absence_proven": False,
            "rows_by_task": {},
            "exit_statuses": {
                "done": {
                    "exit_code": 0,
                    "finished_at": 950.0,
                    "token": "done-token",
                }
            },
            "error": "process scan timed out",
        }
    }

    reverted = recover_stale_launching_tasks_outside_lock(
        reset_s=30,
        deps=_outside_deps(state, evidence, calls=calls),
    )

    assert reverted == 0
    assert tasks[0]["status"] == "running"
    assert tasks[0]["exit_code"] == 0
    assert tasks[1]["status"] == "launching"
    payload = next(call[2] for call in calls if call[:2] == ("notify", "launching_recovery_batch"))
    assert payload["terminal_sentinel"] == 1
    assert payload["deferred"] == 1
    assert payload["errors"] == {"node001": "process scan timed out"}
