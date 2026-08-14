from __future__ import annotations

from contextlib import contextmanager

from skill.scheduler_migration.staging import (
    MigrationCandidateStagingDeps,
    stage_migration_candidates_outside_lock,
)


def _deps(
    calls,
    *,
    state=None,
    nodes=None,
    snapshot=None,
    stage_results=None,
    staged_tasks=None,
    staged_tasks_max=100,
    probe_error=None,
):
    state = state if state is not None else {
        "tasks": [{
            "id": "candidate",
            "status": "queued",
            "preferred_node": "loaded",
            "eta_seconds": 600,
        }]
    }
    nodes = nodes or [{"name": "free", "alive": True}]
    snapshot = snapshot if snapshot is not None else []
    stage_results = stage_results or {}
    staged_tasks = staged_tasks if staged_tasks is not None else {}

    @contextmanager
    def state_lock(**kwargs):
        calls.append(("lock_enter", kwargs))
        try:
            yield
        finally:
            calls.append(("lock_exit", kwargs))

    def probe_all():
        calls.append(("probe_all", None))
        if probe_error is not None:
            raise probe_error
        return nodes

    def identify(state_arg, nodes_arg, **kwargs):
        calls.append(("identify", state_arg, nodes_arg, kwargs))
        return snapshot

    def stage(task, target):
        calls.append(("stage", task["id"], target))
        result = stage_results.get(task["id"], (True, "ok"))
        if isinstance(result, BaseException):
            raise result
        return result

    return MigrationCandidateStagingDeps(
        state_lock=state_lock,
        load_state=lambda: calls.append(("load_state", None)) or state,
        probe_all=probe_all,
        identify_migration_candidates=identify,
        stage_for_migration=stage,
        record_staging_failure=lambda task_id, target: calls.append(("record_fail", task_id, target)),
        notify=lambda event, payload, **kwargs: calls.append(("notify", event, payload, kwargs)),
        staged_tasks=staged_tasks,
        staged_tasks_max=staged_tasks_max,
        now=lambda: 123.0,
    )


def test_stage_migration_candidates_stages_successes_after_snapshot():
    calls = []
    staged = {}
    deps = _deps(
        calls,
        snapshot=[({"id": "t1"}, "node001"), ({"id": "t2"}, "node002")],
        staged_tasks=staged,
    )

    stage_migration_candidates_outside_lock(deps=deps, max_candidates=3)

    assert calls[:4] == [
        ("lock_enter", {"shared": True, "purpose": "migration-staging:snapshot"}),
        ("load_state", None),
        ("lock_exit", {"shared": True, "purpose": "migration-staging:snapshot"}),
        ("probe_all", None),
    ]
    assert any(call[0] == "identify" and call[3] == {"max_candidates": 3} for call in calls)
    assert staged == {("t1", "node001"): 123.0, ("t2", "node002"): 123.0}
    assert [call for call in calls if call[0] == "record_fail"] == []


def test_stage_migration_candidates_records_failures_and_exceptions():
    calls = []
    deps = _deps(
        calls,
        snapshot=[({"id": "bad"}, "node001"), ({"id": "boom"}, "node002")],
        stage_results={
            "bad": (False, "missing env"),
            "boom": RuntimeError("ssh died"),
        },
    )

    stage_migration_candidates_outside_lock(deps=deps)

    assert ("record_fail", "bad", "node001") in calls
    assert ("record_fail", "boom", "node002") in calls
    assert ("notify", "migration_staging_skip", {
        "task_id": "bad",
        "target": "node001",
        "reason": "missing env",
    }, {"feishu_enabled": False}) in calls
    assert ("notify", "migration_staging_error", {
        "task_id": "boom",
        "target": "node002",
        "error": "ssh died",
    }, {"feishu_enabled": False}) in calls


def test_stage_migration_candidates_snapshot_error_is_not_fatal():
    calls = []
    deps = _deps(calls, probe_error=RuntimeError("probe failed"))

    stage_migration_candidates_outside_lock(deps=deps)

    assert ("notify", "migration_snapshot_error", {"error": "probe failed"}, {"feishu_enabled": False}) in calls
    assert all(call[0] != "stage" for call in calls)


def test_stage_migration_candidates_skips_probe_without_possible_candidate():
    calls = []
    deps = _deps(calls, state={"tasks": []})

    stage_migration_candidates_outside_lock(deps=deps)

    assert all(call[0] != "probe_all" for call in calls)
    assert all(call[0] != "identify" for call in calls)


def test_stage_migration_candidates_evicts_old_cache_entries_when_full():
    calls = []
    staged = {("old1", "n"): 1.0, ("old2", "n"): 2.0, ("old3", "n"): 3.0, ("old4", "n"): 4.0}
    deps = _deps(
        calls,
        snapshot=[({"id": "new"}, "node001")],
        staged_tasks=staged,
        staged_tasks_max=4,
    )

    stage_migration_candidates_outside_lock(deps=deps)

    assert ("old1", "n") not in staged
    assert staged[("new", "node001")] == 123.0
