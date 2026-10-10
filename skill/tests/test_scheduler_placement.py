import pytest
from collections import Counter


def _cpu_node(name, total, **overrides):
    node = {"name": name, "alive": True, "total_cpu": total,
            "free_cpu": total, "observed_free_cpu": total, "cpu_hard_free": total,
            "cpu_slot_accounting": True, "cpu_slot_reserved": 0,
            "free_ram_mb": 180000, "total_ram_mb": 192000,
            "running_count": 0, "gpus": []}
    node.update(overrides)
    return node


def _cpu_task(**overrides):
    task = {"id": "cpu-balance", "status": "queued", "project": "unit",
            "cpu_cores": 4, "ram_mb": 512, "est_vram_mb": 0}
    task.update(overrides)
    return task


def _cpu_configs(nodes):
    return {n['name']: {'host': n['name'], 'cpu_cores': n['total_cpu'],
                        'max_vram_per_task': 0, 'live_cpu_backfill': True,
                        'live_cpu_backfill_max_task_cores': 64} for n in nodes}


def _gpu_node(name, **overrides):
    node = _cpu_node(name, 12, free_ram_mb=64000, total_ram_mb=64000,
                     gpus=[{"idx": 0, "used_mb": 0, "free_mb": 12000,
                            "total_mb": 12288, "util_pct": 0,
                            "running_task_count": 0}])
    node.update(overrides)
    return node


def _gpu_configs(nodes):
    return {n["name"]: {"host": n["name"], "cpu_cores": n["total_cpu"],
                        "max_vram_per_task": None, "max_tasks_per_gpu": 12,
                        "allow_gpu_over_one_third": True}
            for n in nodes}


def test_gpu_wave_spreads_tiny_jobs_before_faster_history(sch, monkeypatch):
    nodes = [_gpu_node("gpu-fast"), _gpu_node("gpu-other")]
    monkeypatch.setattr(sch, "NODES", _gpu_configs(nodes))
    monkeypatch.setattr(sch, "_candidate_runtime_seconds",
                        lambda task, node, gpu: 100 if node == "gpu-fast" else 0)
    placements = Counter()
    for _ in range(8):
        task = _cpu_task(cpu_cores=1, est_vram_mb=32)
        task["node"], task["gpu_idx"] = sch.pick_placement(task, nodes)
        placements[task["node"]] += 1
        sch._debit_node_resources_for_launch(nodes, task)
    assert placements == {"gpu-fast": 4, "gpu-other": 4}


def test_gpu_placement_accounts_for_host_cpu_pressure(sch, monkeypatch):
    nodes = [_gpu_node("gpu-busy", free_cpu=1, observed_free_cpu=1,
                       cpu_hard_free=1), _gpu_node("gpu-idle")]
    monkeypatch.setattr(sch, "NODES", _gpu_configs(nodes))
    monkeypatch.setattr(sch, "_candidate_runtime_seconds",
                        lambda task, node, gpu: 100 if node == "gpu-busy" else 0)
    assert sch.pick_placement(_cpu_task(cpu_cores=2, est_vram_mb=401), nodes) == ("gpu-idle", 0)


def test_gpu_placement_spreads_tracked_tiny_jobs_across_cards(sch, monkeypatch):
    node = _gpu_node("gpu-node")
    node["gpus"].append(dict(node["gpus"][0], idx=1))
    node["gpus"][0].update(used_mb=32, free_mb=11968, running_task_count=1)
    monkeypatch.setattr(sch, "NODES", _gpu_configs([node]))
    monkeypatch.setattr(sch, "_candidate_runtime_seconds",
                        lambda task, node, gpu: 100 if gpu == 0 else 0)
    assert sch.pick_placement(_cpu_task(cpu_cores=1, est_vram_mb=32), [node]) == ("gpu-node", 1)


def test_input_staging_affinity_only_breaks_equal_cpu_pressure(sch, monkeypatch):
    nodes = [_cpu_node("node001", 96), _cpu_node("node006", 96)]
    monkeypatch.setattr(sch, "NODES", _cpu_configs(nodes))
    task = _cpu_task(stage_input_target="node006")
    assert sch.pick_placement(task, nodes) == ("node006", None)
    nodes[1].update(free_cpu=16, observed_free_cpu=16, cpu_hard_free=16,
                    cpu_slot_reserved=80)
    assert sch.pick_placement(task, nodes) == ("node001", None)


def test_cpu_placement_prefers_post_launch_load_over_faster_history(sch, monkeypatch):
    nodes = [_cpu_node('node001', 96, observed_free_cpu=8, cpu_hard_free=8),
             _cpu_node('node006', 96)]
    monkeypatch.setattr(sch, 'NODES', _cpu_configs(nodes))
    monkeypatch.setattr(sch, '_candidate_runtime_seconds',
                        lambda task, node, gpu: 100 if node == 'node001' else 0)
    assert sch.pick_placement(_cpu_task(), nodes) == ('node006', None)


def test_cpu_wave_spreads_after_each_debit_despite_one_nodes_history(sch, monkeypatch):
    nodes = [_cpu_node(f'node{i:03d}', 96) for i in range(1, 7)]
    monkeypatch.setattr(sch, 'NODES', _cpu_configs(nodes))
    monkeypatch.setattr(sch, '_candidate_runtime_seconds',
                        lambda task, node, gpu: 100 if node == 'node001' else 0)
    placements = Counter()
    for _ in range(48):
        task = _cpu_task()
        task['node'], task['gpu_idx'] = sch.pick_placement(task, nodes)
        placements[task['node']] += 1
        sch._debit_node_resources_for_launch(nodes, task)
    assert placements == {n['name']: 8 for n in nodes}
    assert all(n['cpu_slot_reserved'] == 32 for n in nodes)
    assert all(n['observed_free_cpu'] == 96 for n in nodes)


def test_cpu_wave_balances_pressure_instead_of_equal_counts_across_sizes(sch, monkeypatch):
    nodes = [_cpu_node('large', 96), _cpu_node('small', 64)]
    monkeypatch.setattr(sch, 'NODES', _cpu_configs(nodes))
    monkeypatch.setattr(sch, '_candidate_runtime_seconds', lambda *args: 0)
    placements = Counter()
    for _ in range(20):
        task = _cpu_task()
        task['node'], task['gpu_idx'] = sch.pick_placement(task, nodes)
        placements[task['node']] += 1
        sch._debit_node_resources_for_launch(nodes, task)
    assert placements == {'large': 12, 'small': 8}


def test_cpu_placement_accounts_for_reserved_work_still_starting(sch, monkeypatch):
    nodes = [_cpu_node('node001', 96, cpu_slot_reserved=80, free_cpu=16),
             _cpu_node('node006', 96, cpu_slot_reserved=20, free_cpu=76,
                       observed_free_cpu=60, cpu_hard_free=60)]
    monkeypatch.setattr(sch, 'NODES', _cpu_configs(nodes))
    monkeypatch.setattr(sch, '_candidate_runtime_seconds', lambda *args: 0)
    assert sch.pick_placement(_cpu_task(), nodes) == ('node006', None)


@pytest.mark.parametrize('pin', ['require_node', 'staged_node', 'preferred_node'])
def test_cpu_balancing_preserves_explicit_pins_and_checkpoint_locality(sch, monkeypatch, pin):
    nodes = [_cpu_node('node001', 96, observed_free_cpu=8, cpu_hard_free=8),
             _cpu_node('node006', 96)]
    monkeypatch.setattr(sch, 'NODES', _cpu_configs(nodes))
    assert sch.pick_placement(_cpu_task(**{pin: 'node001'}), nodes) == ('node001', None)


def test_cpu_balancing_does_not_bypass_a_full_or_blocked_node(sch, monkeypatch):
    nodes = [_cpu_node('node001', 96, observed_free_cpu=0, cpu_hard_free=0),
             _cpu_node('node005', 96), _cpu_node('node006', 96)]
    monkeypatch.setattr(sch, 'NODES', _cpu_configs(nodes))
    monkeypatch.setattr(sch, '_blocked_nodes_for_task', lambda task: {'node006'})
    task = _cpu_task()
    assert sch.pick_placement(task, nodes) == ('node005', None)
    assert sch.pick_placement(dict(task, require_node='node001'), nodes) is None


@pytest.mark.parametrize("free_cpu", [0, 1, 2])
def test_311_gpu_placement_requires_cpu_even_with_free_vram(sch, free_cpu):
    node = {
        "name": "cpu_a", "alive": True, "total_cpu": 8,
        "free_cpu": free_cpu, "free_ram_mb": 32000,
        "total_ram_mb": 64000, "running_count": 0,
        "gpus": [{"idx": 0, "used_mb": 0, "free_mb": 7800,
                  "total_mb": 8192, "util_pct": 0, "running_task_count": 0}],
    }
    task = _cpu_task(cpu_cores=2, est_vram_mb=401,
                     allowed_nodes=["cpu_a"])

    ok, reason = sch._node_resources_ok(task, node, sch.NODES["cpu_a"])

    assert ok is (free_cpu >= 2)
    if free_cpu < 2:
        assert f"cpu: need 2, free {free_cpu}/8" in reason
        assert sch.pick_placement(task, [node]) is None
        assert f"cpu_a=cpu: need 2, free {free_cpu}/8" in sch._build_no_fit_reason(task, [node])
    else:
        assert sch.pick_placement(task, [node]) == ("cpu_a", 0)


def test_311_cpu_pressure_routes_gpu_jobs_to_another_available_node(sch):
    busy = {
        "name": "cpu_a", "alive": True, "total_cpu": 8,
        "free_cpu": 0, "free_ram_mb": 32000, "total_ram_mb": 64000,
        "running_count": 0,
        "gpus": [{"idx": 0, "used_mb": 0, "free_mb": 7800,
                  "total_mb": 8192, "util_pct": 0, "running_task_count": 0}],
    }
    available = dict(busy, name="gpu_a", total_cpu=12, free_cpu=12,
                     gpus=[dict(busy["gpus"][0])])
    task = _cpu_task(cpu_cores=2, est_vram_mb=401,
                     allowed_nodes=["cpu_a", "gpu_a"])

    assert sch.pick_placement(task, [busy, available]) == ("gpu_a", 0)


def test_311_gpu_wave_debits_cpu_before_placing_next_task(sch):
    nodes = [{
        "name": "cpu_a", "alive": True, "total_cpu": 8,
        "free_cpu": 2, "free_ram_mb": 32000, "total_ram_mb": 64000,
        "running_count": 0,
        "gpus": [{"idx": 0, "used_mb": 0, "free_mb": 7800,
                  "total_mb": 8192, "util_pct": 0, "running_task_count": 0}],
    }]
    task = _cpu_task(cpu_cores=2, est_vram_mb=401,
                     allowed_nodes=["cpu_a"])
    task["node"], task["gpu_idx"] = sch.pick_placement(task, nodes)
    sch._debit_node_resources_for_launch(nodes, task)

    assert nodes[0]["free_cpu"] == 0
    assert nodes[0]["gpus"][0]["free_mb"] > 7000
    assert sch.pick_placement(_cpu_task(cpu_cores=2, est_vram_mb=401,
                                      allowed_nodes=["cpu_a"]), nodes) is None


@pytest.mark.parametrize("est_vram", [0, 1000])
@pytest.mark.parametrize("pin", [None, "preferred_node", "require_node", "staged_node"])
def test_windows_nodes_never_receive_tasks_even_when_pinned_or_forced(sch, monkeypatch, est_vram, pin):
    configs = {
        "win": {"os": "windows", "cpu_cores": 128, "max_vram_per_task": 0},
        "linux": {"cpu_cores": 8, "max_vram_per_task": None},
    }
    monkeypatch.setattr(sch, "NODES", configs)
    monkeypatch.setattr(sch, "_hard_rule_bypassed", lambda *args, **kwargs: True)
    nodes = [{"name": name, "alive": True, "free_cpu": cfg["cpu_cores"],
              "free_ram_mb": 500000, "running_count": 0,
              "gpus": [{"idx": 0, "used_mb": 0, "free_mb": 24000,
                        "total_mb": 24000, "util_pct": 0}]}
             for name, cfg in configs.items()]
    task = {"id": "blocked-win", "status": "queued", "project": "unit",
            "cpu_cores": 1, "ram_mb": 512, "est_vram_mb": est_vram,
            "allowed_nodes": ["win", "linux"]}
    if pin:
        task[pin] = "win"

    expected = None if pin in ("require_node", "staged_node") else (
        "linux", None if est_vram == 0 else 0,
    )
    assert sch.pick_placement(task, nodes) == expected
    ok, reason = sch._node_resources_ok(task, nodes[0], configs["win"])
    assert not ok
    assert "windows-node-disabled" in reason
    assert "windows-node-disabled" in sch._explain_node_fit(task, nodes[0])


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
        "local", "gpu_a", "gpu_b", "cpu_a", "node007",
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


def test_explicit_cpu_whitelist_no_fit_reason_includes_every_admitted_node(sch, monkeypatch):
    names = ["local", "gpu_a", "gpu_b", "cpu_a", "node007",
             *[f"node{i:03d}" for i in range(1, 7)]]
    allowed = names[5:]
    monkeypatch.setattr(sch, "NODES", {
        name: {"host": name, "cpu_cores": 192, "capabilities": ["cpu"],
               "ram_headroom_frac": 0.10, "max_vram_per_task": 0}
        for name in names
    })
    task = {"id": "whitelist", "status": "queued", "priority": "normal",
            "project": "Thesis", "cpu_cores": 1, "ram_mb": 9000, "est_vram_mb": 0,
            "allowed_nodes": allowed, "allowed_nodes_user_explicit": True}
    nodes = [{"name": name, "alive": True, "free_cpu": 100, "total_cpu": 192,
              "free_ram_mb": 20000, "total_ram_mb": 192000, "gpus": []}
             for name in names]
    reason = sch._build_no_fit_reason(task, nodes)
    for name in allowed:
        assert name + "=ram:" in reason
    assert "local=outside-allowed-nodes" not in reason


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
