from __future__ import annotations

from types import SimpleNamespace

from skill.scheduler_dispatch_loop import (
    DispatchLoopDeps,
    do_dispatch,
    select_dispatch_cycle_tasks,
)


def _task(task_id: str, **overrides):
    task = {
        "id": task_id,
        "status": "queued",
        "priority": "normal",
        "submitted_at": 1,
        "cwd": "/work",
        "identity": f"run-{task_id}",
    }
    task.update(overrides)
    return task


def _deps(
    calls: list,
    *,
    repaired: int = 0,
    migrated: list | None = None,
    preempted: list | None = None,
    global_batch_plan: dict | None = None,
    global_batch_event: dict | None = None,
    gate_skips: set[str] | None = None,
    placement_by_task: dict[str, object] | None = None,
    validate_ok: bool = True,
    precheck_by_task: dict[str, tuple[bool, str]] | None = None,
    resume_by_task: dict[str, str] | None = None,
    node_configs: dict | None = None,
    stage_state_by_target: dict[str, str] | None = None,
    stage_blocked: bool = False,
    input_stage_state=None,
):
    migrated = migrated or []
    preempted = preempted or []
    global_batch_plan = global_batch_plan or {}
    gate_skips = gate_skips or set()
    placement_by_task = placement_by_task or {}
    precheck_by_task = precheck_by_task or {}
    resume_by_task = resume_by_task or {}
    node_configs = node_configs or {"n1": {"host": None}, "n2": {"host": None}}
    stage_state_by_target = stage_state_by_target or {}

    def prepare_node_accounting(state, nodes, preempted_events, *, deps):
        calls.append(("prepare", [ev.get("id") for ev in preempted_events]))
        return [
            {
                "type": "preempted",
                "task_id": ev["id"],
                "freed_node": ev["node"],
                "cpu_freed": ev.get("cpu_freed", 0),
                "ram_freed": ev.get("ram_freed", 0),
            }
            for ev in preempted_events
        ]

    def build_queue_plan(state, nodes, target_ids, prio, *, task_run_identity, algorithm_global_batch_plan):
        calls.append(("queue_plan", sorted(target_ids)))
        queued = [
            t for t in state["tasks"]
            if t.get("status") == "queued"
            and (not target_ids or str(t.get("id")) in target_ids)
        ]
        return SimpleNamespace(
            running_keys={
                task_run_identity(t)
                for t in state["tasks"]
                if t.get("status") in ("running", "launching")
            },
            queued=queued,
            global_batch_plan=global_batch_plan,
            global_batch_event=global_batch_event,
        )

    def apply_gate(task, *, state, running_keys, deps):
        calls.append(("gate", task["id"]))
        return SimpleNamespace(
            events=[{"type": "gate_seen", "task_id": task["id"]}],
            skip_task=task["id"] in gate_skips,
        )

    def pick_placement(task, search_nodes, extra_allowed_nodes=None):
        calls.append((
            "pick",
            task["id"],
            task.get("preferred_node"),
            task.get("global_batch_hint"),
            task.get("require_node"),
            task.get("require_gpu_idx"),
            extra_allowed_nodes,
        ))
        return placement_by_task.get(
            task["id"],
            (
                task.get("require_node") or "n1",
                task.get("require_gpu_idx"),
            ),
        )

    def resolve_resume(task, *, state, nodes, resume_scan_cache, events, pick_placement, deps):
        placement = placement_by_task.get(task["id"], "__call_pick__")
        if placement == "__call_pick__":
            placement = pick_placement(task, nodes, extra_allowed_nodes=["n1", "n2"])
        return SimpleNamespace(
            skip_task=False,
            placement=placement,
            resume_locations=[],
            resume_nodes=[],
            extra_allowed_nodes=["n1", "n2"],
        )

    def apply_placement(task, *, placement, nodes, state, batch_hint, deps):
        calls.append(("apply_placement", task["id"], placement, batch_hint))
        if placement:
            task["node"], task["gpu_idx"] = placement
        return {"name": task.get("node"), "picked": True}

    def release(task, *args, **kwargs):
        calls.append(("release", task["id"], kwargs))

    def stage_gate(task, *, target, cwd_for_stage, stage_state, deps):
        calls.append(("stage_gate", task["id"], target, cwd_for_stage, stage_state))
        return SimpleNamespace(
            blocked=stage_blocked,
            event={"type": "launch_stage_deferred", "task_id": task["id"]} if stage_blocked else None,
        )

    def input_stage_gate(task, *, target, stage_state, deps):
        calls.append(("input_stage_gate", task["id"], target, stage_state))
        return SimpleNamespace(blocked=False, event=None)

    def launch_execution(task, *, state, nodes, picked_state, running_keys, events, defer_launches, deps):
        calls.append(("launch", task["id"], picked_state, defer_launches))
        running_keys.add(task.get("identity"))
        task["status"] = "launching" if defer_launches else "running"
        events.append({"type": "launched", "task_id": task["id"]})
        return True

    return DispatchLoopDeps(
        reconcile_requeue_lineage_invariants=lambda state: repaired,
        consider_migration=lambda state, nodes: calls.append(("migrate", None)) or list(migrated),
        preempt_for_high_priority=lambda state, nodes: calls.append(("preempt", None)) or list(preempted),
        prepare_dispatch_node_accounting=prepare_node_accounting,
        dispatch_accounting_deps=lambda: object(),
        refresh_queued_resource_estimates=lambda state, history, *, deps: calls.append(("refresh", history)),
        load_history=lambda: {"history": True},
        resource_estimate_deps=lambda: object(),
        build_dispatch_queue_plan=build_queue_plan,
        task_run_identity=lambda task: task.get("identity"),
        algorithm_global_batch_plan=lambda queued, nodes: ({}, None),
        apply_dispatch_task_gates=apply_gate,
        dispatch_task_gate_deps=lambda: object(),
        resolve_resume_aware_placement=resolve_resume,
        dispatch_resume_placement_deps=lambda: object(),
        build_no_fit_reason=lambda task, nodes, extra_allowed_nodes=None: (
            f"no fit for {task['id']} in {extra_allowed_nodes}"
        ),
        apply_dispatch_placement=apply_placement,
        dispatch_placement_apply_deps=lambda: object(),
        validate_selected_resume_checkpoint=lambda *args, **kwargs: validate_ok,
        release_task_claims_and_intents=release,
        precheck_git=lambda task: precheck_by_task.get(task["id"], (True, "")),
        resume_location_for_node=lambda task, node: (
            calls.append(("resume_location", task["id"], node))
            or ({"path": resume_by_task[task["id"]]} if task["id"] in resume_by_task else None)
        ),
        node_configs=node_configs,
        stage_cwd_check=lambda target, cwd: (
            calls.append(("stage_check", target, cwd))
            or stage_state_by_target.get(target, "ready")
        ),
        launch_input_stage_state=input_stage_state or (
            lambda task, target: "ready"
        ),
        apply_launch_staging_gate=stage_gate,
        apply_launch_input_staging_gate=input_stage_gate,
        dispatch_launch_staging_deps=lambda: object(),
        apply_dispatch_launch_execution=launch_execution,
        dispatch_launch_execution_deps=lambda: object(),
        pick_placement=pick_placement,
    )


def test_do_dispatch_happy_path_records_migration_preempt_global_batch_resume_and_launch():
    calls = []
    task = _task("t1", migrated_from="old", preferred_node=None, eta_seconds=30)
    state = {"tasks": [task]}
    nodes = [{"name": "n1"}, {"name": "n2"}]
    events, qcount = do_dispatch(
        state,
        nodes,
        deps=_deps(
            calls,
            repaired=1,
            migrated=["t1"],
            preempted=[{"id": "r1", "node": "old", "cpu_freed": 2, "ram_freed": 100}],
            global_batch_plan={"t1": {"node": "n2", "batch": 1}},
            global_batch_event={"type": "global_batch_plan"},
            precheck_by_task={"t1": (True, "warn: dirty tree")},
            resume_by_task={"t1": "/ckpt/model.pt"},
        ),
    )

    assert qcount == 1
    assert task["status"] == "running"
    assert task["node"] == "n1"
    assert task["resume_from"] == "/ckpt/model.pt"
    assert [ev["type"] for ev in events] == [
        "lineage_repaired",
        "migrated",
        "preempted",
        "global_batch_plan",
        "gate_seen",
        "git_warn",
        "resume_found",
        "launched",
    ]
    assert (
        "pick", "t1", "n2", {"node": "n2", "batch": 1},
        None, None, ["n1", "n2"],
    ) in calls
    assert any(call[0] == "launch" and call[1] == "t1" for call in calls)


def test_enforced_global_batch_launches_only_selected_tasks_at_exact_actions():
    calls = []
    tasks = [_task("t1", est_vram_mb=512), _task("t2", est_vram_mb=512)]
    state = {"tasks": tasks}
    contract = "enforced_exact_placement_v1"

    events, qcount = do_dispatch(
        state,
        [{"name": "n1"}, {"name": "n2"}],
        max_queued=1,
        deps=_deps(
            calls,
            global_batch_plan={
                "t1": {
                    "node": "n2",
                    "gpu_idx": 1,
                    "execution_contract": contract,
                },
            },
            global_batch_event={
                "type": "algorithm_global_batch_plan",
                "execution_contract": contract,
                "fail_closed": True,
            },
        ),
    )

    assert qcount == 2
    assert tasks[0]["status"] == "running"
    assert (tasks[0]["node"], tasks[0]["gpu_idx"]) == ("n2", 1)
    assert tasks[1]["status"] == "queued"
    assert not any(call[0] == "gate" and call[1] == "t2" for call in calls)
    assert (
        "pick", "t1", None,
        {
            "node": "n2",
            "gpu_idx": 1,
            "execution_contract": contract,
        },
        "n2", 1, ["n1", "n2"],
    ) in calls
    execution = next(
        event for event in events
        if event["type"] == "algorithm_global_batch_execution"
    )
    assert execution["executed_task_ids"] == ["t1"]
    assert execution["unresolved_task_ids"] == []
    assert execution["exact_execution_ready"] is True
    assert execution["oracle_gap_applies_to_execution"] is True


def test_enforced_global_batch_replans_instead_of_accepting_mismatched_placement():
    calls = []
    task = _task("t1", est_vram_mb=512)
    contract = "enforced_exact_placement_v1"

    events, _ = do_dispatch(
        {"tasks": [task]},
        [{"name": "n1"}, {"name": "n2"}],
        deps=_deps(
            calls,
            global_batch_plan={
                "t1": {
                    "node": "n2",
                    "gpu_idx": 1,
                    "execution_contract": contract,
                },
            },
            global_batch_event={
                "type": "algorithm_global_batch_plan",
                "execution_contract": contract,
                "fail_closed": True,
            },
            placement_by_task={"t1": ("n1", 0)},
        ),
    )

    assert task["status"] == "queued"
    assert not any(call[0] == "apply_placement" for call in calls)
    assert not any(call[0] == "launch" for call in calls)
    assert any(
        event["type"] == "algorithm_global_batch_replan_required"
        for event in events
    )
    execution = events[-1]
    assert execution["type"] == "algorithm_global_batch_execution"
    assert execution["executed_task_ids"] == []
    assert execution["unresolved_task_ids"] == ["t1"]
    assert execution["oracle_gap_applies_to_execution"] is False


def test_enforced_global_batch_plan_error_fails_closed_without_legacy_dispatch():
    calls = []
    task = _task("t1")
    contract = "enforced_exact_placement_v1"

    events, qcount = do_dispatch(
        {"tasks": [task]},
        [{"name": "n1"}],
        deps=_deps(
            calls,
            global_batch_event={
                "type": "algorithm_global_batch_plan_error",
                "execution_contract": contract,
                "fail_closed": True,
                "reason": "unit failure",
            },
        ),
    )

    assert qcount == 1
    assert task["status"] == "queued"
    assert not any(call[0] in ("gate", "pick", "launch") for call in calls)
    assert events[0]["type"] == "algorithm_global_batch_plan_error"
    assert events[-1]["type"] == "algorithm_global_batch_execution"
    assert events[-1]["exact_execution_ready"] is False


def test_skip_resume_scan_never_calls_resume_location_lookup():
    calls = []
    task = _task("t1", skip_resume_scan=True, ckpt_dir="/ckpt")
    state = {"tasks": [task]}

    events, qcount = do_dispatch(
        state,
        [{"name": "n1"}],
        deps=_deps(calls, resume_by_task={"t1": "/ckpt/model.pt"}),
    )

    assert qcount == 1
    assert task["status"] == "running"
    assert "resume_from" not in task
    assert not any(call[0] == "resume_location" for call in calls)


def test_skip_launch_staging_bypasses_dispatch_stage_gate():
    calls = []
    task = _task("t1", skip_launch_staging=True)
    state = {"tasks": [task]}

    events, qcount = do_dispatch(
        state,
        [{"name": "n1"}],
        deps=_deps(
            calls,
            node_configs={"n1": {"host": "remote"}},
            stage_state_by_target={"n1": "needs_stage"},
            stage_blocked=True,
        ),
    )

    assert qcount == 1
    assert task["status"] == "running"
    assert not any(call[0] == "stage_gate" for call in calls)
    assert events[-1]["type"] == "launched"
    assert any(event["type"] == "launched" for event in events)


def test_target_ids_filter_queue_and_skip_migration_and_preemption():
    calls = []
    state = {"tasks": [_task("t1"), _task("t2")]}
    events, qcount = do_dispatch(
        state,
        [{"name": "n1"}],
        target_task_ids={"t2"},
        defer_launches=True,
        deps=_deps(calls),
    )

    assert qcount == 1
    assert state["tasks"][0]["status"] == "queued"
    assert state["tasks"][1]["status"] == "launching"
    assert ("migrate", None) not in calls
    assert ("preempt", None) not in calls
    assert ("queue_plan", ["t2"]) in calls
    assert [ev["task_id"] for ev in events if ev["type"] == "launched"] == ["t2"]


def test_no_fit_sets_reason_and_skips_launch():
    calls = []
    task = _task("t1")
    state = {"tasks": [task]}
    events, qcount = do_dispatch(
        state,
        [{"name": "n1"}],
        deps=_deps(calls, placement_by_task={"t1": None}),
    )

    assert qcount == 1
    assert task["status"] == "queued"
    assert "no fit for t1" in task["last_block_reason"]
    assert [ev["type"] for ev in events] == ["gate_seen", "no_fit"]
    assert not any(call[0] == "launch" for call in calls)


def test_git_block_releases_claims_clears_placement_and_skips_launch():
    calls = []
    task = _task("t1")
    state = {"tasks": [task]}
    events, _ = do_dispatch(
        state,
        [{"name": "n1"}],
        deps=_deps(calls, precheck_by_task={"t1": (False, "git dirty")}),
    )

    assert task["node"] is None
    assert task["gpu_idx"] is None
    assert task["last_block_reason"] == "git dirty"
    assert [ev["type"] for ev in events] == ["gate_seen", "blocked"]
    assert [call[0] for call in calls].count("release") == 2
    assert not any(call[0] == "launch" for call in calls)


def test_remote_stage_block_defers_before_launch():
    calls = []
    task = _task("t1")
    state = {"tasks": [task]}
    events, _ = do_dispatch(
        state,
        [{"name": "remote"}],
        deps=_deps(
            calls,
            node_configs={"remote": {"host": "remote-host"}},
            placement_by_task={"t1": ("remote", None)},
            stage_state_by_target={"remote": "needs_stage"},
            stage_blocked=True,
        ),
    )

    assert task["node"] == "remote"
    assert [ev["type"] for ev in events] == ["gate_seen", "launch_stage_deferred"]
    assert ("stage_gate", "t1", "remote", "/work", "needs_stage") in calls
    assert not any(call[0] == "launch" for call in calls)


def test_dispatch_deduplicates_shared_launch_cwd_state_checks():
    calls = []
    tasks = [_task("t1"), _task("t2")]

    do_dispatch(
        {"tasks": tasks},
        [{"name": "remote"}],
        deps=_deps(
            calls,
            node_configs={"remote": {"host": "remote-host"}},
            placement_by_task={
                "t1": ("remote", None),
                "t2": ("remote", None),
            },
        ),
    )

    assert [
        call for call in calls if call[0] == "stage_check"
    ] == [("stage_check", "remote", "/work")]
    assert [task["status"] for task in tasks] == ["running", "running"]


def test_dispatch_deduplicates_shared_launch_input_state_checks():
    calls = []
    input_checks = []
    tasks = [
        _task(
            "t1",
            wait_for_files=["/work/shared/input.json"],
        ),
        _task(
            "t2",
            wait_for_files=["/work/shared/input.json"],
        ),
    ]

    do_dispatch(
        {"tasks": tasks},
        [{"name": "n1"}],
        deps=_deps(
            calls,
            input_stage_state=lambda task, target: (
                input_checks.append((task["id"], target)) or "ready"
            ),
        ),
    )

    assert input_checks == [("t1", "n1")]
    assert [task["status"] for task in tasks] == ["running", "running"]


def test_do_dispatch_max_queued_caps_work_without_changing_reported_queue_count():
    calls = []
    tasks = [_task("t1"), _task("t2"), _task("t3")]
    state = {"tasks": tasks}

    events, qcount = do_dispatch(
        state,
        [{"name": "n1"}],
        max_queued=2,
        deps=_deps(calls),
    )

    assert qcount == 3
    assert [task["status"] for task in tasks] == ["running", "running", "queued"]
    assert [ev["task_id"] for ev in events if ev["type"] == "launched"] == ["t1", "t2"]


def test_bounded_dispatch_rotates_past_recently_attempted_queue_head():
    tasks = [
        _task("t1", submitted_at=1, last_dispatch_attempt_at=20),
        _task("t2", submitted_at=2, last_dispatch_attempt_at=20),
        _task("t3", submitted_at=3),
        _task("t4", submitted_at=4),
    ]

    selected = select_dispatch_cycle_tasks(tasks, 2)

    assert [task["id"] for task in selected] == ["t3", "t4"]


def test_bounded_dispatch_interleaves_gpu_and_cpu_resource_lanes():
    tasks = [
        *[
            _task(f"gpu-{index}", est_vram_mb=4096, cpu_cores=8)
            for index in range(20)
        ],
        *[
            _task(f"cpu-{index}", est_vram_mb=0, cpu_cores=12)
            for index in range(20)
        ],
    ]

    selected = select_dispatch_cycle_tasks(tasks, 10)
    selected_ids = [task["id"] for task in selected]

    assert sum(task_id.startswith("gpu-") for task_id in selected_ids) == 5
    assert sum(task_id.startswith("cpu-") for task_id in selected_ids) == 5


def test_bounded_dispatch_retries_staged_input_before_fresh_same_lane():
    tasks = [
        _task(
            "staged",
            cpu_cores=12,
            last_dispatch_attempt_at=100,
            stage_input_target="node001",
            last_block_reason=(
                "launch input staging: inputs not yet staged to node001; "
                "will retry next dispatch cycle"
            ),
        ),
        _task("fresh", cpu_cores=12, last_dispatch_attempt_at=0),
    ]

    selected = select_dispatch_cycle_tasks(tasks, 1)

    assert [task["id"] for task in selected] == ["staged"]


def test_dispatch_records_attempt_metadata_even_when_gate_blocks_task():
    calls = []
    task = _task("t1")

    do_dispatch(
        {"tasks": [task]},
        [{"name": "n1"}],
        max_queued=1,
        deps=_deps(calls, gate_skips={"t1"}),
    )

    assert task["dispatch_attempt_count"] == 1
    assert task["last_dispatch_attempt_at"] > 0


def run(check, _sch):
    for fn in (
        test_do_dispatch_happy_path_records_migration_preempt_global_batch_resume_and_launch,
        test_target_ids_filter_queue_and_skip_migration_and_preemption,
        test_no_fit_sets_reason_and_skips_launch,
        test_git_block_releases_claims_clears_placement_and_skips_launch,
        test_remote_stage_block_defers_before_launch,
        test_do_dispatch_max_queued_caps_work_without_changing_reported_queue_count,
        test_bounded_dispatch_rotates_past_recently_attempted_queue_head,
        test_dispatch_records_attempt_metadata_even_when_gate_blocks_task,
    ):
        try:
            fn()
            check(f"dispatch loop module: {fn.__name__}", True)
        except Exception as exc:
            check(f"dispatch loop module: {fn.__name__}", False, diag=repr(exc))
