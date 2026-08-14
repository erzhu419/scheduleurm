from __future__ import annotations

from skill.scheduler_result_sync import ResultSyncDeps, sync_completed_results_outside_lock


class _Lock:
    def __init__(self, calls, label):
        self.calls = calls
        self.label = label

    def __enter__(self):
        self.calls.append(("lock_enter", self.label))
        return self

    def __exit__(self, *args):
        self.calls.append(("lock_exit", self.label))
        return False


def _task(task_id, **overrides):
    task = {
        "id": task_id,
        "status": "done",
        "node": "remote",
        "result_dir": f"/remote/{task_id}",
        "local_result_dir": f"/local/{task_id}",
        "result_synced_at": None,
        "result_sync_attempts": 0,
        "result_syncing_at": None,
        "cmd": "python train.py",
        "cwd": "/work",
    }
    task.update(overrides)
    return task


def _deps(
    *,
    state,
    calls=None,
    sync_results=None,
    now=1000.0,
    fail_snapshot=False,
    fail_commit=False,
    infer_dirs=None,
    getpid=lambda: 0,
    pid_alive=lambda _pid: False,
    dependency_max=8,
):
    calls = calls if calls is not None else []
    sync_results = list(sync_results or [])
    load_count = {"n": 0}

    def state_lock(**kwargs):
        calls.append(("state_lock", kwargs))
        if fail_snapshot and kwargs:
            raise RuntimeError("snapshot failed")
        if fail_commit and not kwargs:
            raise RuntimeError("commit failed")
        return _Lock(calls, kwargs.get("purpose", "commit"))

    def load_state():
        calls.append(("load_state",))
        load_count["n"] += 1
        return state

    def save_state(saved):
        calls.append(("save_state", saved))

    def notify(event, payload, **kwargs):
        calls.append(("notify", event, payload, kwargs))

    def sync_one(candidate):
        calls.append(("sync_one", dict(candidate)))
        if sync_results:
            result = sync_results.pop(0)
            if callable(result):
                return result(candidate)
            return result
        return True, "ok"

    return ResultSyncDeps(
        state_lock=state_lock,
        load_state=load_state,
        save_state=save_state,
        notify=notify,
        node_configs={
            "remote": {"host": "user@host"},
            "local": {"host": None},
        },
        infer_bapr_result_dirs_from_cmd=lambda cmd, cwd: list(infer_dirs or []),
        sync_one_result=sync_one,
        result_sync_timeout_s=1800,
        result_sync_stale_grace_s=600,
        result_sync_max_per_cycle=1,
        result_sync_dependency_max_per_cycle=dependency_max,
        result_sync_max_attempts=3,
        now=lambda: now,
        getpid=getpid,
        pid_alive=pid_alive,
    )


def test_success_claims_syncs_and_commits_result_synced_at():
    calls = []
    state = {"tasks": [_task("t1")]}

    sync_completed_results_outside_lock(deps=_deps(state=state, calls=calls, now=1234.0))

    task = state["tasks"][0]
    assert task.get("result_syncing_at") is None
    assert task["result_synced_at"] == 1234.0
    assert task["result_sync_error"] is None
    assert ("sync_one", {
        "id": "t1",
        "node": "remote",
        "host": "user@host",
        "result_dir": "/remote/t1",
        "local_result_dir": "/local/t1",
    }) in calls
    assert any(call[0] == "notify" and call[1] == "result_sync_done" for call in calls)


def test_retired_node_result_is_preserved_without_remote_sync_attempt():
    calls = []
    state = {"tasks": [_task("old", node="retired")]}
    deps = _deps(state=state, calls=calls)
    deps.node_configs["retired"] = {"host": "old-host", "retired": True}

    sync_completed_results_outside_lock(deps=deps)

    assert not any(call[0] == "sync_one" for call in calls)
    assert state["tasks"][0]["result_synced_at"] is None


def test_failure_clears_claim_and_increments_attempts():
    state = {"tasks": [_task("t1")]}

    sync_completed_results_outside_lock(
        deps=_deps(state=state, sync_results=[(False, "rsync rc=23")]),
    )

    task = state["tasks"][0]
    assert task.get("result_syncing_at") is None
    assert task["result_synced_at"] is None
    assert task["result_sync_attempts"] == 1
    assert task["result_sync_error"] == "rsync rc=23"


def test_skip_rules_exclude_non_remote_or_ineligible_tasks():
    calls = []
    state = {
        "tasks": [
            _task("queued", status="queued"),
            _task("localdone", node="local"),
            _task("synced", result_synced_at=10.0),
            _task("attemptcap", result_sync_attempts=3),
            _task("missing", result_dir=None, result_dirs=[]),
            _task("eligible"),
        ]
    }

    sync_completed_results_outside_lock(deps=_deps(state=state, calls=calls, now=2000.0))

    sync_calls = [call for call in calls if call[0] == "sync_one"]
    assert [call[1]["id"] for call in sync_calls] == ["eligible"]


def test_dependency_producer_preempts_older_result_sync_backlog():
    calls = []
    state = {
        "tasks": [
            _task("old-result"),
            _task(
                "archive",
                result_dir="/remote/archive",
                local_result_dir="/local/archive",
            ),
            _task(
                "consumer",
                status="queued",
                result_dir=None,
                wait_for_files=["/local/archive/heldout.json"],
            ),
        ]
    }

    sync_completed_results_outside_lock(deps=_deps(state=state, calls=calls))

    sync_calls = [call for call in calls if call[0] == "sync_one"]
    assert [call[1]["id"] for call in sync_calls] == ["archive"]


def test_dependency_fan_in_uses_bounded_burst_without_draining_backlog():
    calls = []
    producers = [
        _task(
            f"producer-{index}",
            result_dir=f"/remote/fan-in/{index}",
            local_result_dir=f"/local/fan-in/{index}",
        )
        for index in range(6)
    ]
    state = {
        "tasks": [
            _task("old-result"),
            *producers,
            _task(
                "consumer",
                status="queued",
                result_dir=None,
                wait_for_files=[
                    f"/local/fan-in/{index}/manifest.json"
                    for index in range(6)
                ],
            ),
        ]
    }

    sync_completed_results_outside_lock(
        deps=_deps(state=state, calls=calls, dependency_max=4),
    )

    sync_calls = [call for call in calls if call[0] == "sync_one"]
    assert [call[1]["id"] for call in sync_calls] == [
        "producer-0",
        "producer-1",
        "producer-2",
        "producer-3",
    ]


def test_fresh_claim_is_skipped_but_stale_claim_is_reclaimed():
    calls = []
    state = {
        "tasks": [
            _task("fresh", result_syncing_at=900.0),
            _task("stale", result_syncing_at=1.0),
        ]
    }

    sync_completed_results_outside_lock(
        max_candidates=2,
        deps=_deps(state=state, calls=calls, now=3000.0),
    )

    sync_calls = [call for call in calls if call[0] == "sync_one"]
    assert [call[1]["id"] for call in sync_calls] == ["stale"]
    assert any(call[0] == "notify" and call[1] == "result_sync_claim_reclaimed" for call in calls)


def test_claim_owned_by_dead_worker_is_reclaimed_immediately():
    calls = []
    state = {
        "tasks": [
            _task("dead-owner", result_syncing_at=999.0, result_syncing_pid=321),
        ]
    }

    sync_completed_results_outside_lock(
        deps=_deps(
            state=state,
            calls=calls,
            now=1000.0,
            getpid=lambda: 654,
            pid_alive=lambda pid: pid != 321,
        ),
    )

    assert any(call[0] == "sync_one" for call in calls)
    reclaimed = next(
        call for call in calls if call[0] == "notify" and call[1] == "result_sync_claim_reclaimed"
    )
    assert reclaimed[2]["owner_pid"] == 321
    assert reclaimed[2]["owner_dead"] is True


def test_result_dirs_candidate_preserves_local_dir_mapping():
    calls = []
    state = {
        "tasks": [
            _task(
                "multi",
                result_dir=None,
                result_dirs=["/r/a", "/r/b"],
                local_result_dirs=["/l/a", "/l/b"],
            )
        ]
    }

    sync_completed_results_outside_lock(deps=_deps(state=state, calls=calls))

    candidate = next(call[1] for call in calls if call[0] == "sync_one")
    assert candidate["result_dirs"] == ["/r/a", "/r/b"]
    assert candidate["local_result_dirs"] == ["/l/a", "/l/b"]
    assert candidate["result_dir"] == "/r/a"
    assert candidate["local_result_dir"] == "/local/multi"


def test_inferred_bapr_result_dirs_are_used_when_result_fields_missing():
    calls = []
    state = {"tasks": [_task("infer", result_dir=None, result_dirs=[])]}

    sync_completed_results_outside_lock(
        deps=_deps(state=state, calls=calls, infer_dirs=["/inferred/a", "/inferred/b"]),
    )

    candidate = next(call[1] for call in calls if call[0] == "sync_one")
    assert candidate["result_dirs"] == ["/inferred/a", "/inferred/b"]


def test_status_changed_during_sync_only_clears_claim():
    state = {"tasks": [_task("t1")]}

    def sync_and_cancel(candidate):
        state["tasks"][0]["status"] = "cancelled"
        return True, "ok"

    sync_completed_results_outside_lock(
        deps=_deps(state=state, sync_results=[sync_and_cancel]),
    )

    task = state["tasks"][0]
    assert task["status"] == "cancelled"
    assert task.get("result_syncing_at") is None
    assert task.get("result_synced_at") is None


def test_snapshot_and_commit_errors_notify_without_raising():
    calls_snapshot = []
    sync_completed_results_outside_lock(
        deps=_deps(state={"tasks": []}, calls=calls_snapshot, fail_snapshot=True),
    )
    assert any(call[0] == "notify" and call[1] == "result_sync_snapshot_error" for call in calls_snapshot)

    calls_commit = []
    sync_completed_results_outside_lock(
        deps=_deps(state={"tasks": [_task("t1")]}, calls=calls_commit, fail_commit=True),
    )
    assert any(call[0] == "notify" and call[1] == "result_sync_commit_error" for call in calls_commit)
