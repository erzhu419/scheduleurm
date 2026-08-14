from skill.scheduler_migration.resume_fit import (
    ResumeFitDeps,
    estimate_gpu_fit_wait_seconds,
    estimate_node_fit_wait_seconds,
    record_staged_resume_location,
    resume_checkpoint_stage_check,
    resume_ckpt_stage_key,
    resume_ckpt_stage_estimate_seconds,
    resume_location_for_target,
    resume_stage_source_for_target,
    running_tasks_for_node,
    task_vram_pressure_mb,
    task_wait_eta_seconds,
)


def _deps(
    *,
    node_configs=None,
    cap=None,
    fails=None,
    cache_hit=None,
    gpu_fits=None,
    pick_placement=None,
    required_gpu=None,
    loads=None,
    now=1000.0,
):
    return ResumeFitDeps(
        node_configs=node_configs or {
            "local": {"host": None},
            "node001": {"host": "node001"},
            "node002": {"host": "node002"},
        },
        staging_cap_exceeded=cap if cap is not None else {},
        staging_fails=fails if fails is not None else {},
        staging_ttl_s=300,
        staging_fail_cooldown_s=120,
        resume_ckpt_migrate_overhead_s=10,
        resume_ckpt_migrate_est_mbps=100.0,
        remote_path_for_node=lambda node, path: f"/remote/{node}{path}" if node != "local" else path,
        staging_cache_hit=cache_hit or (lambda key: False),
        gpu_fits=gpu_fits or (lambda task, gpu, node_info: gpu.get("free_mb", 0) >= task.get("est_vram_mb", 0)),
        pick_placement=pick_placement or (lambda *a, **k: None),
        task_required_gpu_idx=lambda task: required_gpu,
        compute_node_load_seconds=lambda state: loads or {},
        now=lambda: now,
    )


def test_resume_source_requires_marker_for_ambiguous_target_copy():
    locations = [
        {"node": "node001", "path": "/ckpt/r1"},
        {"node": "local", "path": "/ckpt/local"},
    ]
    task = {"ckpt_dir": "/ckpt", "resume_locations": locations}

    assert resume_stage_source_for_target(task, "node001", deps=_deps()) == locations[1]
    key = resume_ckpt_stage_key("local", "node001", "/ckpt", locations[1])
    assert resume_stage_source_for_target(
        task,
        "node001",
        deps=_deps(cache_hit=lambda got: got == key),
    ) == locations[0]
    assert resume_stage_source_for_target(task, "node002", deps=_deps()) == locations[1]
    assert resume_stage_source_for_target(task, "local", deps=_deps()) == locations[0]


def test_same_generation_partial_target_requires_full_directory_staging():
    local = {
        "node": "local", "path": "/run/checkpoints/train_state.pkl",
        "mtime": 200.0, "size": 453189,
    }
    remote = {
        "node": "node001", "path": "/run/checkpoints/train_state.pkl",
        "mtime": 200.0, "size": 453189,
    }
    task = {
        "ckpt_dir": "/run",
        "resume_locations": [remote, local],
    }
    key = resume_ckpt_stage_key("local", "node001", "/run", local)

    state, source, reason = resume_checkpoint_stage_check(
        task, "node001", deps=_deps())
    assert state == "needs_stage"
    assert source == local
    assert reason == "checkpoint not yet staged to target"

    state, source, reason = resume_checkpoint_stage_check(
        task,
        "node001",
        deps=_deps(cache_hit=lambda got: got == key),
    )
    assert state == "ready"
    assert source["node"] == "node001"
    assert reason == "checkpoint already on target"


def test_resume_checkpoint_stage_check_cache_cap_fail_and_expiry():
    task = {
        "ckpt_dir": "/ckpt",
        "resume_locations": [{"node": "local", "path": "/ckpt", "size": 100}],
    }
    key = resume_ckpt_stage_key(
        "local", "node001", "/ckpt", task["resume_locations"][0])

    state, source, reason = resume_checkpoint_stage_check(
        task,
        "node001",
        deps=_deps(cache_hit=lambda got: got == key),
    )
    assert state == "ready"
    assert source["node"] == "node001"
    assert source["path"] == "/remote/node001/ckpt"
    assert reason == "checkpoint staged to target"

    cap = {key: 950.0}
    state, source, reason = resume_checkpoint_stage_check(
        task,
        "node001",
        deps=_deps(cap=cap, now=1000.0),
    )
    assert state == "cap_exceeded"
    assert cap[key] == 950.0

    cap = {key: 600.0}
    state, source, reason = resume_checkpoint_stage_check(
        task,
        "node001",
        deps=_deps(cap=cap, now=1000.0),
    )
    assert state == "needs_stage"
    assert key not in cap

    fails = {key: (980.0, "rsync failed")}
    state, source, reason = resume_checkpoint_stage_check(
        task,
        "node001",
        deps=_deps(fails=fails, now=1000.0),
    )
    assert state == "stage_failed"
    assert reason == "rsync failed"


def test_stale_target_checkpoint_is_replaced_by_newest_source_generation():
    locations = [
        {"node": "local", "path": "/ckpt/train_state.pkl",
         "mtime": 200.0, "size": 2000},
        {"node": "node001", "path": "/ckpt/train_state.pkl",
         "mtime": 100.0, "size": 1000},
    ]
    task = {"ckpt_dir": "/ckpt", "resume_locations": locations}
    expected_key = resume_ckpt_stage_key(
        "local", "node001", "/ckpt", locations[0])

    state, source, reason = resume_checkpoint_stage_check(
        task,
        "node001",
        deps=_deps(cache_hit=lambda key: key == expected_key),
    )

    assert state == "ready"
    assert source["node"] == "node001"
    assert source["mtime"] == 200.0
    assert reason == "checkpoint staged to target"
    assert expected_key != (
        "resume_ckpt", "local", "node001", "/ckpt")


def test_remote_target_relays_newest_remote_generation_instead_of_stale_local():
    locations = [
        {"node": "node002", "path": "/ckpt/train_state.pkl",
         "mtime": 200.0, "size": 2000},
        {"node": "local", "path": "/ckpt/train_state.pkl",
         "mtime": 100.0, "size": 1000},
        {"node": "node001", "path": "/ckpt/train_state.pkl",
         "mtime": 100.0, "size": 1000},
    ]
    task = {"ckpt_dir": "/ckpt", "resume_locations": locations}

    state, source, reason = resume_checkpoint_stage_check(
        task, "node001", deps=_deps())

    assert state == "needs_stage"
    assert source == locations[0]
    assert reason == "checkpoint not yet staged to target"


def test_local_target_stages_from_newest_remote_generation():
    locations = [
        {"node": "node001", "path": "/ckpt/train_state.pkl",
         "mtime": 200.0, "size": 2000},
        {"node": "local", "path": "/ckpt/train_state.pkl",
         "mtime": 100.0, "size": 1000},
    ]
    source = resume_stage_source_for_target(
        {"resume_locations": locations}, "local", deps=_deps())

    assert source == locations[0]


def test_resume_ckpt_stage_estimate_uses_size_and_ready_short_circuit():
    assert resume_ckpt_stage_estimate_seconds("ready", {"size": 999}, deps=_deps()) == 0
    assert resume_ckpt_stage_estimate_seconds("needs_stage", None, deps=_deps()) == 10 ** 9
    assert resume_ckpt_stage_estimate_seconds(
        "needs_stage",
        {"size": 250 * 1024 * 1024},
        deps=_deps(),
    ) == 13


def test_running_task_and_resource_pressure_helpers():
    state = {
        "tasks": [
            {"id": "t1", "status": "running", "node": "node001", "gpu_idx": 0},
            {"id": "t2", "status": "running", "node": "node001", "gpu_idx": 1},
            {"id": "t3", "status": "queued", "node": "node001", "gpu_idx": 0},
        ]
    }
    assert [task["id"] for task in running_tasks_for_node(state, "node001")] == ["t1", "t2"]
    assert [task["id"] for task in running_tasks_for_node(state, "node001", 0)] == ["t1"]
    assert task_wait_eta_seconds({"eta_seconds": "bad"}) == 0
    assert task_wait_eta_seconds({"eta_seconds": "42"}) == 42
    assert task_vram_pressure_mb({"current_vram_mb": 100, "peak_vram_mb": 200, "est_vram_mb": 150}) == 200


def test_estimate_gpu_fit_waits_until_enough_vram_is_freed():
    task = {"est_vram_mb": 5000}
    gpu = {"idx": 0, "used_mb": 9000, "total_mb": 12000, "free_mb": 3000, "running_task_count": 2}
    running = [
        {"id": "slow", "eta_seconds": 100, "current_vram_mb": 2000},
        {"id": "fast", "eta_seconds": 20, "current_vram_mb": 1500},
    ]

    wait = estimate_gpu_fit_wait_seconds(task, gpu, {}, running, deps=_deps())

    assert wait == 100
    assert gpu["used_mb"] == 9000


def test_estimate_node_fit_prefers_immediate_fit_then_gpu_wait_then_load():
    task = {"est_vram_mb": 5000}
    node = {"name": "node001", "alive": True, "gpus": [{"idx": 0, "used_mb": 9000, "total_mb": 12000, "free_mb": 3000}]}
    state = {"tasks": [{"id": "r1", "status": "running", "node": "node001", "gpu_idx": 0, "eta_seconds": 33, "current_vram_mb": 3000}]}

    assert estimate_node_fit_wait_seconds(
        task,
        node,
        state,
        deps=_deps(pick_placement=lambda *a, **k: ("node001", 0)),
    ) == 0
    assert estimate_node_fit_wait_seconds(task, node, state, deps=_deps()) == 33
    cpu_node = {"name": "node001", "alive": True, "gpus": []}
    assert estimate_node_fit_wait_seconds(
        {"est_vram_mb": 0},
        cpu_node,
        {"tasks": []},
        deps=_deps(loads={"node001": 77}),
    ) == 77


def test_record_staged_resume_location_orders_locations_and_sets_resume_from():
    task = {
        "resume_locations": [
            {"node": "local", "path": "/ckpt/local", "mtime": 10},
            {"node": "node002", "path": "/ckpt/old", "mtime": 5},
        ],
        "resume_from": "/old",
    }

    source = {"node": "local", "path": "/ckpt/local", "mtime": 20, "size": 100}
    record_staged_resume_location(task, "node001", source, deps=_deps())

    assert task["resume_locations"][0]["node"] == "node001"
    assert task["resume_locations"][0]["path"] == "/remote/node001/ckpt/local"
    assert task["resume_preferred_nodes"] == ["node001", "local", "node002"]
    assert task["resume_checkpoint_node"] == "node001"
    assert task["resume_from"] == "/remote/node001/ckpt/local"


def test_resume_location_for_target_rewrites_path():
    out = resume_location_for_target({"node": "local", "path": "/ckpt"}, "node001", deps=_deps())
    assert out == {"node": "node001", "path": "/remote/node001/ckpt"}
