def test_measurement_reservation_blocks_placement_without_mutating_telemetry(sch, monkeypatch):
    monkeypatch.setattr(sch, "NODES", {
        "gpu-node": {"host": "gpu-node", "max_vram_per_task": None}
    }, raising=False)
    node = {
        "name": "gpu-node",
        "alive": True,
        "measurement_reservation_active": True,
        "measurement_reservation": {"purpose": "theorem calibration"},
        "free_cpu": 8,
        "total_cpu": 8,
        "free_ram_mb": 64000,
        "gpus": [{
            "idx": 0,
            "used_mb": 0,
            "free_mb": 12000,
            "total_mb": 12000,
            "util_pct": 0,
            "running_task_count": 0,
        }],
    }
    task = {
        "id": "reserved",
        "status": "queued",
        "priority": "normal",
        "project": "unit",
        "est_vram_mb": 1000,
        "ram_mb": 1000,
        "cpu_cores": 1,
    }

    assert sch.pick_placement(task, [node]) is None
    assert "measurement-reserved(theorem calibration)" in sch._build_no_fit_reason(task, [node])
    assert node["free_cpu"] == 8
    assert node["gpus"][0]["free_mb"] == 12000


def test_no_fit_reason_ignores_removed_slurm_pending_fields(sch, monkeypatch):
    monkeypatch.setattr(sch, "NODES", {
        "nodeA": {
            "host": "nodeA",
            "slurm_backend": "slurm",
            "max_vram_per_task": None,
        }
    }, raising=False)

    reason = sch._build_no_fit_reason(
        {
            "id": "tlocal",
            "status": "queued",
            "priority": "normal",
            "project": "unit",
            "est_vram_mb": 1000,
            "ram_mb": 1000,
            "cpu_cores": 1,
        },
        [{
            "name": "nodeA",
            "alive": True,
            "free_cpu": 0,
            "free_ram_mb": 64000,
            "gpus": [{"idx": 0, "used_mb": 0, "free_mb": 24000, "total_mb": 24000, "util_pct": 0}],
            "slurm_pending_split": {"cpu": 99, "gpu": 99},
        }],
    )

    assert "no fit (prio=normal)" in reason
    assert "slurm" not in reason.lower()
    assert "nodeA=" in reason


def test_cpu_pool_no_fit_reason_reports_actual_hpc_candidates(sch, monkeypatch):
    node_names = [
        "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007",
        *[f"node{i:03d}" for i in range(1, 7)],
    ]
    configs = {
        name: {
            "host": name,
            "cpu_cores": 192,
            "max_vram_per_task": 0 if name.startswith("node00") else None,
            "capabilities": ["cpu"],
            "live_cpu_backfill": True,
            "live_cpu_backfill_max_task_cores": 64,
        }
        for name in node_names
    }
    monkeypatch.setattr(sch, "NODES", configs, raising=False)
    nodes = [
        {
            "name": name,
            "alive": True,
            "free_cpu": 0,
            "observed_free_cpu": 12,
            "cpu_hard_free": 7,
            "cpu_hard_reserved": 5,
            "cpu_startup_hold_remaining_s": 42,
            "cpu_slot_accounting": True,
            "total_cpu": 192,
            "free_ram_mb": 160000,
            "total_ram_mb": 192793,
            "gpus": [],
        }
        for name in node_names
    ]

    reason = sch._build_no_fit_reason(
        {
            "id": "tcpu",
            "status": "queued",
            "priority": "normal",
            "project": "KG-SYNTH",
            "est_vram_mb": 0,
            "ram_mb": 512,
            "cpu_cores": 12,
        },
        nodes,
    )

    for index in range(1, 7):
        assert f"node{index:03d}=cpu: need 12, free 7/192" in reason
    assert "observed_free=12 slot_free=0 startup_hold=5 hold_release<=42s" in reason
    assert "outside-cpu-pool" not in reason


def test_gpu_fit_treats_driver_noise_as_empty_but_tracked_task_as_occupied(sch):
    task = {"est_vram_mb": 10500}
    idle_with_driver_context = {
        "idx": 0,
        "total_mb": 12288,
        "used_mb": 169,
        "free_mb": 11872,
        "util_pct": 0,
        "running_task_count": 0,
    }
    occupied_with_low_early_use = dict(
        idle_with_driver_context,
        running_task_count=1,
    )

    assert sch._gpu_fits(task, idle_with_driver_context, {})
    assert not sch._gpu_fits(task, occupied_with_low_early_use, {})


def test_no_fit_reason_omits_disabled_one_third_rule(sch, monkeypatch):
    monkeypatch.setattr(sch, "NODES", {
        "gpu-node": {
            "host": "gpu-node",
            "max_vram_per_task": None,
            "allow_gpu_over_one_third": True,
        }
    }, raising=False)

    reason = sch._build_no_fit_reason(
        {
            "id": "tgpu",
            "status": "queued",
            "priority": "normal",
            "project": "unit",
            "est_vram_mb": 8192,
            "ram_mb": 1000,
            "cpu_cores": 1,
            "allowed_nodes": ["gpu-node"],
        },
        [{
            "name": "gpu-node",
            "alive": True,
            "free_cpu": 8,
            "total_cpu": 8,
            "free_ram_mb": 64000,
            "running_count": 1,
            "gpus": [{
                "idx": 0,
                "used_mb": 9000,
                "free_mb": 3000,
                "total_mb": 12288,
                "util_pct": 0,
                "running_task_count": 1,
            }],
        }],
    )

    assert "free<est+margin" in reason
    assert "1/3+grace" not in reason
