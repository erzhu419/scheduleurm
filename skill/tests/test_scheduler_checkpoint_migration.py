from __future__ import annotations

from skill.scheduler_migration.checkpoint import (
    CheckpointMigrationDeps,
    checkpoint_migration_extra_allowed_nodes,
)


def _deps(
    *,
    requires=True,
    waits=None,
    blocked=None,
    failed=None,
    fits=None,
    stage=None,
    stage_seconds=None,
    min_wait_s=300,
    safety_s=30,
):
    waits = waits or {}
    blocked = blocked or set()
    failed = failed or set()
    fits = fits if fits is not None else set()
    stage = stage or {}
    stage_seconds = stage_seconds or {}

    return CheckpointMigrationDeps(
        task_requires_resume_scan=lambda task: requires,
        estimate_node_fit_wait_seconds=lambda task, node_state, state: waits.get(node_state.get("name") if node_state else None),
        blocked_nodes_for_task=lambda task: set(blocked),
        launch_failed_nodes_for_task=lambda task: set(failed),
        pick_placement=lambda task, nodes, **kwargs: (
            (nodes[0]["name"], 0) if nodes and nodes[0]["name"] in fits else None
        ),
        resume_checkpoint_stage_check=lambda task, node_name, locations: stage.get(
            node_name,
            ("ready", {"node": "src"}, "ready"),
        ),
        resume_ckpt_stage_estimate_seconds=lambda stage_state, source_loc: stage_seconds.get(
            stage_state,
            0 if stage_state == "ready" else 100,
        ),
        min_wait_s=min_wait_s,
        safety_s=safety_s,
    )


def test_checkpoint_migration_adds_faster_staged_target():
    task = {"id": "t1"}
    nodes = [
        {"name": "source", "alive": True},
        {"name": "local", "alive": True},
        {"name": "node002", "alive": True},
    ]
    locations = [{"node": "source"}]

    extra, plan = checkpoint_migration_extra_allowed_nodes(
        task,
        nodes,
        locations,
        {"tasks": []},
        deps=_deps(
            waits={"source": 7200},
            fits={"local", "node002"},
            stage={
                "local": ("needs_stage", {"node": "source"}, "copy ckpt"),
                "node002": ("ready", {"node": "source"}, "already there"),
            },
            stage_seconds={"needs_stage": 100, "ready": 0},
        ),
    )

    assert extra == ["local", "node002"]
    assert plan == {
        "checkpoint_wait_s": 7200,
        "safety_s": 30,
        "targets": [
            {
                "node": "node002",
                "source": "source",
                "stage_state": "ready",
                "stage_estimate_s": 0,
                "stage_msg": "already there",
            },
            {
                "node": "local",
                "source": "source",
                "stage_state": "needs_stage",
                "stage_estimate_s": 100,
                "stage_msg": "copy ckpt",
            },
        ],
    }


def test_checkpoint_migration_respects_hard_and_soft_node_filters():
    nodes = [
        {"name": "source", "alive": True},
        {"name": "local", "alive": True},
        {"name": "node002", "alive": True},
    ]
    locations = [{"node": "source"}]

    extra, _plan = checkpoint_migration_extra_allowed_nodes(
        {"id": "t1", "require_node": "source"},
        nodes,
        locations,
        {},
        deps=_deps(waits={"source": 7200}, fits={"local", "node002"}),
    )
    assert extra == []

    extra, plan = checkpoint_migration_extra_allowed_nodes(
        {"id": "t1", "allowed_nodes": ["node002"]},
        nodes,
        locations,
        {},
        deps=_deps(
            waits={"source": 7200},
            fits={"local", "node002"},
            blocked={"node002"},
        ),
    )
    assert extra == ["local"]
    assert plan["targets"][0]["node"] == "local"


def test_checkpoint_migration_skips_when_wait_is_short_or_stage_too_slow():
    task = {"id": "t1"}
    nodes = [{"name": "source", "alive": True}, {"name": "local", "alive": True}]
    locations = [{"node": "source"}]

    extra, plan = checkpoint_migration_extra_allowed_nodes(
        task,
        nodes,
        locations,
        {},
        deps=_deps(waits={"source": 100}, fits={"local"}, min_wait_s=300),
    )
    assert (extra, plan) == ([], None)

    extra, plan = checkpoint_migration_extra_allowed_nodes(
        task,
        nodes,
        locations,
        {},
        deps=_deps(
            waits={"source": 7200},
            fits={"local"},
            stage={"local": ("needs_stage", {"node": "source"}, "copy ckpt")},
            stage_seconds={"needs_stage": 7190},
            safety_s=30,
        ),
    )
    assert (extra, plan) == ([], None)


def test_checkpoint_migration_uses_alternative_when_locality_wait_is_unknown():
    task = {"id": "t1"}
    nodes = [
        {"name": "source", "alive": True},
        {"name": "node002", "alive": True},
    ]
    locations = [{"node": "source"}]

    extra, plan = checkpoint_migration_extra_allowed_nodes(
        task,
        nodes,
        locations,
        {},
        deps=_deps(
            waits={},
            fits={"node002"},
            stage={
                "node002": (
                    "needs_stage",
                    {"node": "source", "size": 150 * 1024 * 1024},
                    "relay checkpoint",
                ),
            },
            stage_seconds={"needs_stage": 64},
        ),
    )

    assert extra == ["node002"]
    assert plan["checkpoint_wait_s"] == 10 ** 9
    assert plan["targets"][0]["source"] == "source"
