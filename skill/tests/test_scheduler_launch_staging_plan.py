from __future__ import annotations

from skill.scheduler_launch_staging_plan import (
    LaunchStagingPlanDeps,
    build_protected_under_cwd,
    collect_launch_staging_plan,
    launch_stage_candidate_key,
)


def _deps(
    *,
    nodes: dict | None = None,
    wait_ids: set[str] | None = None,
    soft_pool: dict[str, list[str]] | None = None,
    resume_ids: set[str] | None = None,
    cache_hits: set[tuple] | None = None,
    cap_hits: set[tuple] | None = None,
    stage_state: str = "ready",
    fallback_blocked: set[tuple[str, str]] | None = None,
):
    nodes = nodes or {
        "local": {"host": None},
        "node001": {"host": "n1", "capabilities": ["cuda"]},
        "node002": {"host": "n2"},
        "node003": {"host": "n3", "stage_only_when_targeted": True},
        "win001": {"host": "w1", "windows": True},
    }
    wait_ids = wait_ids or set()
    soft_pool = soft_pool or {}
    resume_ids = resume_ids or set()
    cache_hits = cache_hits or set()
    cap_hits = cap_hits or set()
    fallback_blocked = fallback_blocked or set()

    return LaunchStagingPlanDeps(
        node_configs=nodes,
        queued_wait_for_file_block_reason=lambda task: (
            "waiting" if task.get("id") in wait_ids else ""
        ),
        hpc_cpu_pool_soft_require_nodes=lambda task: list(soft_pool.get(task.get("id"), [])),
        task_requires_resume_scan=lambda task: task.get("id") in resume_ids,
        node_is_windows=lambda node: bool(nodes.get(node, {}).get("windows")),
        node_cpu_fallback_block_reason=lambda task, node, info: (
            "blocked" if (task.get("id"), node) in fallback_blocked else ""
        ),
        staging_cache_hit=lambda key: key in cache_hits,
        staging_cap_recent=lambda key: key in cap_hits,
        resume_checkpoint_stage_check=lambda task, node, locs: (
            stage_state,
            {"node": locs[0]["node"], "path": locs[0]["path"]} if stage_state == "needs_stage" else None,
            "",
        ),
    )


def test_build_protected_under_cwd_includes_stage_excludes_and_output_dirs():
    tasks = [
        {
            "cwd": "/work/proj",
            "stage_excludes": ["custom", "/nested/cache/"],
            "ckpt_dir": "/work/proj/checkpoints/run1",
            "result_dir": "/work/proj/results/run1",
            "local_result_dir": "/other/place",
        },
        {
            "cwd": "/work/proj",
            "local_result_dir": "/work/proj/collected/run1",
        },
    ]

    protected = build_protected_under_cwd(tasks)

    assert protected["/work/proj"] == {
        "custom",
        "nested/cache",
        "checkpoints/run1/",
        "results/run1/",
        "collected/run1/",
    }


def test_collect_launch_staging_plan_stages_preferred_to_all_fallback_nodes():
    state = {
        "tasks": [{
            "id": "t1",
            "status": "queued",
            "cwd": "/work/a",
            "preferred_node": "node001",
            "est_vram_mb": 0,
        }]
    }

    plan = collect_launch_staging_plan(state, None, deps=_deps())

    assert {
        ("node001", "/work/a"),
        ("node002", "/work/a"),
    }.issubset(plan.candidates)
    assert ("node003", "/work/a") not in plan.candidates
    assert ("win001", "/work/a") in plan.candidates


def test_collect_launch_staging_plan_hard_require_allowed_cache_and_cap_filters():
    state = {
        "tasks": [{
            "id": "t2",
            "status": "queued",
            "cwd": "/work/b",
            "require_node": "node001",
            "allowed_nodes": ["node001", "node002"],
            "est_vram_mb": 0,
        }]
    }

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(cache_hits={("local", "node001", "/work/b")}),
    )
    assert plan.candidates == set()

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(cap_hits={("local", "node001", "/work/b")}),
    )
    assert plan.candidates == set()


def test_launch_input_candidates_dedupe_shared_workspace_paths():
    nodes = {
        "node001": {
            "host": "head",
            "shared_workspace_group": "hpc-home",
        },
        "node002": {
            "host": "head",
            "shared_workspace_group": "hpc-home",
        },
    }
    state = {
        "tasks": [
            {
                "id": "t1",
                "status": "queued",
                "cwd": "/work/repo",
                "allowed_nodes": ["node001", "node002"],
                "stage_input_target": "node001",
                "stage_input_paths": ["/work/repo/bundle"],
                "est_vram_mb": 0,
            },
            {
                "id": "t2",
                "status": "queued",
                "cwd": "/work/repo",
                "allowed_nodes": ["node001", "node002"],
                "stage_input_target": "node002",
                "stage_input_paths": ["/work/repo/bundle"],
                "est_vram_mb": 0,
            },
        ]
    }

    plan = collect_launch_staging_plan(
        state, None, deps=_deps(nodes=nodes))

    assert len(plan.input_candidates) == 1


def test_collect_launch_staging_plan_honors_task_level_skip_launch_staging():
    state = {
        "tasks": [{
            "id": "t2-skip",
            "status": "queued",
            "cwd": "/work/provisioned",
            "allowed_nodes": ["node001", "node002"],
            "skip_launch_staging": True,
            "est_vram_mb": 1024,
        }]
    }

    plan = collect_launch_staging_plan(state, None, deps=_deps())

    assert plan.candidates == set()


def test_collect_launch_staging_plan_stages_code_while_waiting_for_prerequisites():
    state = {
        "tasks": [{
            "id": "wait-source-refresh",
            "status": "queued",
            "cwd": "/work/proj",
            "allowed_nodes": ["node001"],
            "est_vram_mb": 0,
            "resume_locations": [{"node": "node002", "path": "/work/ckpt"}],
        }]
    }

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(wait_ids={"wait-source-refresh"}, resume_ids={"wait-source-refresh"}, stage_state="needs_stage"),
    )

    assert plan.candidates == {("node001", "/work/proj")}
    assert plan.ckpt_candidates == []


def test_collect_launch_staging_plan_uses_soft_cpu_pool_targets():
    state = {
        "tasks": [{
            "id": "t3",
            "status": "queued",
            "cwd": "/work/c",
            "require_node": "node999",
            "est_vram_mb": 0,
        }]
    }

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(soft_pool={"t3": ["node001", "node002"]}),
    )

    assert plan.candidates == {
        ("node001", "/work/c"),
        ("node002", "/work/c"),
    }


def test_collect_launch_staging_plan_prioritizes_high_gpu_and_stages_waiting_cpu_source():
    state = {
        "tasks": [
            {
                "id": "gpu",
                "status": "queued",
                "cwd": "/work/gpu",
                "priority": "high",
                "est_vram_mb": 1024,
            },
            {
                "id": "cpu-wait",
                "status": "queued",
                "cwd": "/work/cpu-wait",
                "est_vram_mb": 0,
            },
            {
                "id": "cpu-ready",
                "status": "queued",
                "cwd": "/work/cpu-ready",
                "est_vram_mb": 0,
            },
        ]
    }

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(wait_ids={"cpu-wait"}),
    )

    assert plan.gpu_staging_pending is True
    assert ("win001", "/work/gpu") not in plan.candidates
    assert ("node001", "/work/gpu") in plan.candidates
    assert ("node001", "/work/cpu-ready") in plan.candidates
    assert ("node001", "/work/cpu-wait") in plan.candidates


def test_blocked_high_gpu_does_not_starve_normal_gpu_input_staging():
    state = {
        "tasks": [
            {
                "id": "blocked-high",
                "status": "queued",
                "cwd": "/work/high",
                "priority": "high",
                "allowed_nodes": ["node001"],
                "est_vram_mb": 1024,
            },
            {
                "id": "normal-ready",
                "status": "queued",
                "cwd": "/work/normal",
                "priority": "normal",
                "allowed_nodes": ["node001"],
                "stage_input_target": "node001",
                "stage_input_paths": ["/work/normal/inputs"],
                "est_vram_mb": 1024,
            },
        ]
    }

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(wait_ids={"blocked-high"}),
    )

    assert plan.gpu_staging_pending is True
    assert [(task["id"], target) for task, target in plan.input_candidates] == [
        ("normal-ready", "node001")
    ]


def test_collect_launch_staging_plan_stages_each_checkpoint_to_one_target():
    state = {
        "tasks": [{
            "id": "t4",
            "status": "queued",
            "cwd": "/work/d",
            "est_vram_mb": 0,
            "resume_locations": [{"node": "node001", "path": "/ckpt"}],
        }]
    }

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(resume_ids={"t4"}, stage_state="needs_stage"),
    )

    assert ("node001", "/work/d") in plan.candidates
    assert ("node002", "/work/d") in plan.candidates
    assert [target for _, target, _ in plan.ckpt_candidates] == ["node001"]
    assert all(snapshot["id"] == "t4" for snapshot, _, _ in plan.ckpt_candidates)


def test_collect_launch_staging_plan_balances_checkpoint_targets():
    state = {
        "tasks": [
            {
                "id": f"t{index}",
                "status": "queued",
                "cwd": f"/work/{index}",
                "est_vram_mb": 1024,
                "priority": "high",
                "resume_locations": [
                    {"node": "local", "path": f"/ckpt/{index}"}],
            }
            for index in range(4)
        ]
    }
    resume_ids = {task["id"] for task in state["tasks"]}

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(resume_ids=resume_ids, stage_state="needs_stage"),
    )

    targets = [target for _, target, _ in plan.ckpt_candidates]
    assert targets == ["node001", "node002", "node001", "node002"]
    assert len(plan.ckpt_candidates) == len(state["tasks"])


def test_collect_launch_staging_plan_keeps_checkpoint_on_sticky_input_target():
    state = {
        "tasks": [{
            "id": "sticky",
            "status": "queued",
            "cwd": "/work/sticky",
            "allowed_nodes": ["node001", "node002"],
            "stage_input_target": "node002",
            "stage_input_paths": ["/inputs/sticky"],
            "est_vram_mb": 1024,
            "resume_locations": [{"node": "local", "path": "/ckpt/sticky"}],
        }]
    }

    plan = collect_launch_staging_plan(
        state,
        None,
        deps=_deps(resume_ids={"sticky"}, stage_state="needs_stage"),
    )

    assert [(snapshot["id"], target) for snapshot, target in plan.input_candidates] == [
        ("sticky", "node002")
    ]
    assert [(snapshot["id"], target) for snapshot, target, _ in plan.ckpt_candidates] == [
        ("sticky", "node002")
    ]


def test_collect_launch_staging_plan_deduplicates_shared_inputs_per_target():
    state = {
        "tasks": [
            {
                "id": f"shared-{index}",
                "status": "queued",
                "cwd": "/work/shared",
                "allowed_nodes": ["node001"],
                "stage_input_target": "node001",
                "stage_input_paths": ["/inputs/archive"],
                "est_vram_mb": 0,
            }
            for index in range(50)
        ]
    }

    plan = collect_launch_staging_plan(state, None, deps=_deps())

    assert len(plan.input_candidates) == 1
    assert plan.input_candidates[0][0]["id"] == "shared-0"
    assert plan.input_candidates[0][1] == "node001"


def test_launch_stage_candidate_key_orders_gpu_then_plain_nodes_then_windows():
    nodes = {
        "gpu": {"capabilities": ["cuda"]},
        "plain": {},
        "plain2": {},
        "win": {"windows": True},
    }
    items = [("win", "/w"), ("plain", "/p"), ("gpu", "/g"), ("plain2", "/s")]

    ordered = sorted(
        items,
        key=lambda item: launch_stage_candidate_key(
            item,
            node_configs=nodes,
            node_is_windows=lambda node: bool(nodes.get(node, {}).get("windows")),
        ),
    )

    assert [node for node, _ in ordered] == ["gpu", "plain", "plain2", "win"]
