from __future__ import annotations

from skill.scheduler_migration.policy import (
    MigrationPolicyDeps,
    compute_node_load_seconds,
    consider_migration,
    identify_migration_candidates,
)


def _task(task_id, **overrides):
    task = {
        "id": task_id,
        "status": "queued",
        "preferred_node": "loaded",
        "eta_seconds": 900,
        "cwd": "/work",
        "ckpt_dir": "/ckpt",
        "cmd": "python train.py",
        "signature": "proj/run",
    }
    task.update(overrides)
    return task


def _nodes():
    return [
        {"name": "free", "alive": True},
        {"name": "loaded", "alive": True},
        {"name": "down", "alive": False},
    ]


def _deps(
    *,
    loads=None,
    blocked=None,
    failed=None,
    staging_failed=None,
    can_migrate=None,
    now=1000.0,
    max_per_dispatch=1,
    calls=None,
):
    calls = calls if calls is not None else []
    blocked = blocked or {}
    failed = failed or {}
    staging_failed = staging_failed or set()
    can_migrate = can_migrate or set()
    loads = loads or {"free": 0, "loaded": 5000, "down": 0}

    def compute(state):
        calls.append(("compute", state))
        return dict(loads)

    return MigrationPolicyDeps(
        compute_node_load_seconds=compute,
        blocked_nodes_for_task=lambda task: set(blocked.get(task["id"], set())),
        launch_failed_nodes_for_task=lambda task: set(failed.get(task["id"], set())),
        staging_recently_failed=lambda task_id, target: (task_id, target) in staging_failed,
        can_migrate_to=lambda task, target: (task["id"], target) in can_migrate,
        migration_free_threshold_s=600,
        migration_load_ratio=2.0,
        migration_min_source_load_s=600,
        migration_min_task_eta_s=300,
        migration_cooldown_s=1800,
        migration_max_per_dispatch=max_per_dispatch,
        now=lambda: now,
    )


def test_compute_node_load_seconds_counts_active_and_pinned_known_nodes_only():
    state = {
        "tasks": [
            {"id": "run", "status": "running", "node": "loaded", "eta_seconds": 100},
            {"id": "launch", "status": "launching", "node": "free", "eta_seconds": 200},
            {"id": "hard", "status": "queued", "require_node": "loaded", "preferred_node": "free", "eta_seconds": 300},
            {"id": "soft", "status": "queued", "preferred_node": "free", "eta_seconds": 400},
            {"id": "unknown", "status": "queued", "preferred_node": "free", "eta_seconds": 0},
            {"id": "external", "status": "queued", "preferred_node": "free", "eta_seconds": 500, "auto_adopted": True},
            {"id": "stale", "status": "running", "node": "stale", "eta_seconds": 600},
            {"id": "done", "status": "done", "node": "loaded", "eta_seconds": 700},
        ]
    }

    assert compute_node_load_seconds(
        state,
        node_configs={"free": {}, "loaded": {}, "down": {}},
    ) == {"free": 600, "loaded": 400, "down": 0}


def test_identify_returns_snapshots_sorted_by_eta_and_skips_unsafe_targets():
    state = {
        "tasks": [
            _task("blocked", eta_seconds=400),
            _task("failed", eta_seconds=500),
            _task("staging_failed", eta_seconds=600),
            _task("ok2", eta_seconds=800),
            _task("ok1", eta_seconds=700),
            _task("hard", eta_seconds=650, require_node="loaded"),
            _task("unknown_eta", eta_seconds=0),
        ]
    }

    result = identify_migration_candidates(
        state,
        _nodes(),
        max_candidates=10,
        deps=_deps(
            blocked={"blocked": {"free"}},
            failed={"failed": {"free"}},
            staging_failed={("staging_failed", "free")},
        ),
    )

    assert [(task["id"], target) for task, target in result] == [
        ("ok1", "free"),
        ("ok2", "free"),
    ]
    snapshot = result[0][0]
    assert snapshot == {
        "id": "ok1",
        "cwd": "/work",
        "ckpt_dir": "/ckpt",
        "cmd": "python train.py",
        "preferred_node": "loaded",
        "signature": "proj/run",
        "eta_seconds": 700,
    }


def test_identify_respects_load_gates_and_alive_nodes():
    state = {"tasks": [_task("ok")]}

    assert identify_migration_candidates(
        state,
        _nodes(),
        deps=_deps(loads={"free": 700, "loaded": 5000}),
    ) == []
    assert identify_migration_candidates(
        state,
        [{"name": "free", "alive": False}, {"name": "loaded", "alive": True}],
        deps=_deps(),
    ) == []
    assert identify_migration_candidates(
        state,
        _nodes(),
        deps=_deps(loads={"free": 500, "loaded": 700}),
    ) == []


def test_consider_uses_supplied_loads_without_recompute_and_sets_staged_node():
    calls = []
    state = {"tasks": [_task("move", eta_seconds=600)]}

    migrated = consider_migration(
        state,
        _nodes(),
        loads={"free": 0, "loaded": 5000},
        deps=_deps(can_migrate={("move", "free")}, calls=calls, now=1234.0),
    )

    assert migrated == ["move"]
    assert calls == []
    task = state["tasks"][0]
    assert task["preferred_node"] == "free"
    assert task["staged_node"] == "free"
    assert task["migrated_from"] == "loaded"
    assert task["migrated_at"] == 1234.0
    assert "migrated: preferred loaded(load=5000s)" in task["last_block_reason"]


def test_consider_rechecks_cooldown_blocked_failed_and_staging_cache():
    state = {
        "tasks": [
            _task("cool", eta_seconds=400, migrated_at=500.0),
            _task("blocked", eta_seconds=500),
            _task("failed", eta_seconds=600),
            _task("not_staged", eta_seconds=700),
            _task("move", eta_seconds=800),
        ]
    }

    migrated = consider_migration(
        state,
        _nodes(),
        deps=_deps(
            blocked={"blocked": {"free"}},
            failed={"failed": {"free"}},
            can_migrate={("move", "free")},
            now=1000.0,
        ),
    )

    assert migrated == ["move"]
    assert state["tasks"][0]["preferred_node"] == "loaded"
    assert state["tasks"][3]["preferred_node"] == "loaded"
    assert state["tasks"][4]["preferred_node"] == "free"


def test_consider_respects_max_per_dispatch():
    state = {"tasks": [_task("a", eta_seconds=400), _task("b", eta_seconds=500)]}

    migrated = consider_migration(
        state,
        _nodes(),
        deps=_deps(can_migrate={("a", "free"), ("b", "free")}, max_per_dispatch=2),
    )

    assert migrated == ["a", "b"]


def test_migration_excludes_measurement_reserved_source_and_target():
    state = {"tasks": [_task("move", eta_seconds=900)]}
    reserved_target = [
        {"name": "free", "alive": True, "measurement_reservation_active": True},
        {"name": "loaded", "alive": True},
    ]
    assert identify_migration_candidates(
        state,
        reserved_target,
        deps=_deps(),
    ) == []

    reserved_source = [
        {"name": "free", "alive": True},
        {"name": "loaded", "alive": True, "measurement_reservation_active": True},
    ]
    assert consider_migration(
        state,
        reserved_source,
        deps=_deps(can_migrate={("move", "free")}),
    ) == []
    assert state["tasks"][0]["preferred_node"] == "loaded"
