from __future__ import annotations

from skill.scheduler_migration.dispatch_resume_placement import (
    DispatchResumePlacementDeps,
    resolve_resume_aware_placement,
    validate_selected_resume_checkpoint,
)


def _deps(
    calls: list,
    *,
    requires_scan: bool = True,
    resume_locations: list | None = None,
    resume_errors: dict | None = None,
    scan_ready: bool = True,
    allow_scan_error: bool = False,
    stage_state: str = "ready",
    stage_states: dict[str, str] | None = None,
    migration_extra: list | None = None,
    migration_plan: dict | None = None,
    soft_require: list | None = None,
):
    def cached(task, nodes):
        calls.append(("cached", task["id"], tuple(n["name"] for n in nodes)))
        return scan_ready, list(resume_locations or []), dict(resume_errors or {})

    def stage_check(task, node, locations):
        calls.append(("stage_check", task["id"], node))
        selected_state = (stage_states or {}).get(node, stage_state)
        source = (
            {"node": "source", "path": "/ckpt"}
            if selected_state == "ready" else None
        )
        return selected_state, source, "stage message"

    def record_staged(task, node, source_loc):
        calls.append(("record_staged", task["id"], node, source_loc["node"]))
        locs = list(task.get("resume_locations") or [])
        locs.append({"node": node, "path": source_loc["path"]})
        task["resume_locations"] = locs

    def checkpoint_extra(task, nodes, locations, state):
        calls.append(("checkpoint_extra", task["id"]))
        return list(migration_extra or []), migration_plan

    def release(task, *args, **kwargs):
        calls.append(("release", task["id"]))

    return DispatchResumePlacementDeps(
        task_requires_resume_scan=lambda task: requires_scan,
        cached_resume_locations_for_task=cached,
        allow_initial_resume_scan_error=lambda task: allow_scan_error,
        resume_checkpoint_stage_check=stage_check,
        record_staged_resume_location=record_staged,
        checkpoint_migration_extra_allowed_nodes=checkpoint_extra,
        soft_require_nodes=lambda task: list(soft_require or []),
        release_task_claims_and_intents=release,
        now=lambda: 456.0,
    )


def test_resolve_resume_aware_placement_without_resume_scan_clears_plan_and_picks():
    calls = []
    task = {"id": "t1", "resume_checkpoint_migration_plan": {"old": True}}
    nodes = [{"name": "node001"}]

    def pick(task_arg, nodes_arg, extra_allowed_nodes=None):
        calls.append(("pick", extra_allowed_nodes))
        return ("node001", None)

    result = resolve_resume_aware_placement(
        task,
        state={"tasks": [task]},
        nodes=nodes,
        resume_scan_cache={},
        events=[],
        pick_placement=pick,
        deps=_deps(calls, requires_scan=False),
    )

    assert result.placement == ("node001", None)
    assert result.skip_task is False
    assert "resume_checkpoint_migration_plan" not in task
    assert calls == [("pick", None)]


def test_resolve_resume_scan_pending_defers_without_remote_work():
    calls = []
    task = {
        "id": "t-pending",
        "resume_scan_inflight": {"token": "claim-1"},
    }
    events = []

    result = resolve_resume_aware_placement(
        task,
        state={"tasks": [task]},
        nodes=[{"name": "node001"}],
        resume_scan_cache={},
        events=events,
        pick_placement=lambda task, nodes, extra_allowed_nodes=None: ("node001", None),
        deps=_deps(calls, scan_ready=False),
    )

    assert result.skip_task is True
    assert events[0]["type"] == "blocked"
    assert "outside the scheduler state lock" in events[0]["reason"]
    assert calls == [("cached", "t-pending", ("node001",))]


def test_resolve_resume_scan_failure_blocks_fail_closed():
    calls = []
    task = {"id": "t2", "resume_locations": [{"node": "node001", "path": "/old"}]}
    events = []

    result = resolve_resume_aware_placement(
        task,
        state={"tasks": [task]},
        nodes=[{"name": "node001"}],
        resume_scan_cache={},
        events=events,
        pick_placement=lambda task, nodes, extra_allowed_nodes=None: ("node001", None),
        deps=_deps(calls, resume_locations=[], resume_errors={"node001": "ssh failed"}),
    )

    assert result.skip_task is True
    assert events[0]["type"] == "blocked"
    assert "scan failed on all checked nodes" in events[0]["reason"]
    assert task["last_block_reason"] == events[0]["reason"]
    assert ("checkpoint_extra", "t2") not in calls


def test_resolve_resume_scan_error_can_be_ignored_then_places_task():
    calls = []
    task = {"id": "t3", "resume_locations": [{"node": "node001", "path": "/old"}]}
    events = []

    result = resolve_resume_aware_placement(
        task,
        state={"tasks": [task]},
        nodes=[{"name": "node001"}],
        resume_scan_cache={},
        events=events,
        pick_placement=lambda task, nodes, extra_allowed_nodes=None: ("node001", None),
        deps=_deps(
            calls,
            resume_locations=[],
            resume_errors={"node001": "ssh failed"},
            allow_scan_error=True,
        ),
    )

    assert result.skip_task is False
    assert result.placement == ("node001", None)
    assert task["resume_scan_errors_ignored_at"] == 456.0
    assert task["resume_scan_errors_ignored"] == {"node001": "ssh failed"}
    assert events == []


def test_required_node_missing_checkpoint_blocks_when_stage_needed():
    calls = []
    task = {"id": "t4", "require_node": "node002", "resume_locations": [{"node": "node001", "path": "/ckpt"}]}
    events = []

    result = resolve_resume_aware_placement(
        task,
        state={"tasks": [task]},
        nodes=[{"name": "node001"}, {"name": "node002"}],
        resume_scan_cache={},
        events=events,
        pick_placement=lambda task, nodes, extra_allowed_nodes=None: ("node002", None),
        deps=_deps(
            calls,
            resume_locations=[{"node": "node001", "path": "/ckpt"}],
            stage_state="needs_stage",
        ),
    )

    assert result.skip_task is True
    assert events[0]["type"] == "blocked"
    assert "require_node=node002 has no matching checkpoint" in events[0]["reason"]
    assert ("stage_check", "t4", "node002") in calls
    assert ("checkpoint_extra", "t4") not in calls


def test_required_node_ready_checkpoint_records_and_uses_migration_extra_allowed():
    calls = []
    task = {"id": "t5", "require_node": "node002", "resume_locations": [{"node": "node001", "path": "/ckpt"}]}
    events = []
    pick_calls = []

    def pick(task_arg, nodes, extra_allowed_nodes=None):
        pick_calls.append(list(extra_allowed_nodes or []))
        return ("node002", None)

    result = resolve_resume_aware_placement(
        task,
        state={"tasks": [task]},
        nodes=[{"name": "node001"}, {"name": "node002"}],
        resume_scan_cache={},
        events=events,
        pick_placement=pick,
        deps=_deps(
            calls,
            resume_locations=[{"node": "node001", "path": "/ckpt"}],
            stage_state="ready",
            migration_extra=["node002"],
            migration_plan={"from": "node001", "to": "node002"},
        ),
    )

    assert result.skip_task is False
    assert result.placement == ("node002", None)
    assert {"node": "node002", "path": "/ckpt"} in task["resume_locations"]
    assert task["resume_checkpoint_migration_plan"] == {"from": "node001", "to": "node002"}
    assert pick_calls[-1] == ["node002"]
    assert events == []


def test_soft_require_pool_skips_sentinel_checkpoint_and_prefers_ready_member():
    calls = []
    task = {
        "id": "t-soft-pool",
        "require_node": "zhengliang-hpc",
        "resume_locations": [
            {"node": "node001", "path": "/complete"},
            {"node": "node002", "path": "/partial"},
        ],
    }
    nodes = [{"name": "node001"}, {"name": "node002"}]
    pick_calls = []

    def pick(task_arg, candidate_nodes, extra_allowed_nodes=None):
        names = [node["name"] for node in candidate_nodes]
        pick_calls.append(names)
        return (names[0], None) if names else None

    result = resolve_resume_aware_placement(
        task,
        state={"tasks": [task]},
        nodes=nodes,
        resume_scan_cache={},
        events=[],
        pick_placement=pick,
        deps=_deps(
            calls,
            resume_locations=task["resume_locations"],
            stage_states={"node001": "ready", "node002": "needs_stage"},
            soft_require=["node001", "node002"],
        ),
    )

    assert result.skip_task is False
    assert result.placement == ("node001", None)
    assert pick_calls[-1] == ["node001"]
    assert ("stage_check", "t-soft-pool", "zhengliang-hpc") not in calls


def test_resume_placement_prefers_generation_verified_ready_node():
    calls = []
    task = {
        "id": "t-ready-first",
        "resume_locations": [
            {"node": "node001", "path": "/partial"},
            {"node": "node002", "path": "/complete"},
        ],
    }
    nodes = [{"name": "node001"}, {"name": "node002"}]
    pick_calls = []

    def pick(task_arg, candidate_nodes, extra_allowed_nodes=None):
        names = [node["name"] for node in candidate_nodes]
        pick_calls.append(names)
        return (names[0], None) if names else None

    result = resolve_resume_aware_placement(
        task,
        state={"tasks": [task]},
        nodes=nodes,
        resume_scan_cache={},
        events=[],
        pick_placement=pick,
        deps=_deps(
            calls,
            resume_locations=task["resume_locations"],
            stage_states={"node001": "needs_stage", "node002": "ready"},
        ),
    )

    assert result.placement == ("node002", None)
    assert pick_calls[0] == ["node001", "node002"]  # pre-scan capacity gate
    assert pick_calls[-1] == ["node002"]


def test_validate_selected_resume_checkpoint_blocks_and_clears_selected_node():
    calls = []
    task = {"id": "t6", "node": "node002", "gpu_idx": 0}
    events = []

    ok = validate_selected_resume_checkpoint(
        task,
        resume_locations=[{"node": "node001", "path": "/ckpt"}],
        resume_nodes={"node001"},
        events=events,
        deps=_deps(calls, stage_state="cap_exceeded"),
    )

    assert ok is False
    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert events[0]["type"] == "blocked"
    assert "selected node node002 lacks it" in events[0]["reason"]
    assert ("release", "t6") in calls


def test_validate_selected_resume_checkpoint_ready_records_and_allows_launch():
    calls = []
    task = {"id": "t7", "node": "node002", "gpu_idx": 0}
    events = []

    ok = validate_selected_resume_checkpoint(
        task,
        resume_locations=[{"node": "node001", "path": "/ckpt"}],
        resume_nodes={"node001"},
        events=events,
        deps=_deps(calls, stage_state="ready"),
    )

    assert ok is True
    assert task["node"] == "node002"
    assert task["gpu_idx"] == 0
    assert {"node": "node002", "path": "/ckpt"} in task["resume_locations"]
    assert events == []
    assert ("release", "t7") not in calls


def test_validate_selected_stale_checkpoint_still_requires_staging():
    calls = []
    task = {"id": "t8", "node": "node002", "gpu_idx": 0}
    events = []

    ok = validate_selected_resume_checkpoint(
        task,
        resume_locations=[
            {"node": "node001", "path": "/ckpt/new", "mtime": 20},
            {"node": "node002", "path": "/ckpt/old", "mtime": 10},
        ],
        resume_nodes={"node001", "node002"},
        events=events,
        deps=_deps(calls, stage_state="needs_stage"),
    )

    assert ok is False
    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert ("stage_check", "t8", "node002") in calls
    assert "no matching checkpoint yet" in events[0]["reason"]
