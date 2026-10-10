from __future__ import annotations

import sys

from skill.scheduler_node import inventory as scheduler_nodes
from skill.scheduler_node.gates import gpu_task_cap_for_task, node_concurrency_cap_for_task


def test_gpu2_small_jobs_use_cpu_bounded_capacity_without_relaxing_large_jobs():
    node = scheduler_nodes.NODES["gpu_b"]
    small = {"est_vram_mb": 768}
    assert gpu_task_cap_for_task(small, node) == 6
    assert node_concurrency_cap_for_task(small, node) == node["cpu_cores"] == 12
    assert gpu_task_cap_for_task({"est_vram_mb": 2200}, node) == 4
    assert gpu_task_cap_for_task(small, scheduler_nodes.NODES["gpu_a"]) == 4


def test_311_gpu_jobs_require_available_host_cpu():
    assert scheduler_nodes.NODES["cpu_a"]["ignore_cpu_for_gpu_tasks"] is False


def test_node_inventory_keeps_temporarily_unreachable_hpc_nodes_visible():
    assert scheduler_nodes.NODE_NAME_ALIASES["node007-direct"] == "node007"
    assert "node007" in scheduler_nodes.NODES
    assert "node007-direct" not in scheduler_nodes.NODES
    assert scheduler_nodes.NODES["node007"].get("gpu_launch_user_thread_reserve", 0) == 0
    assert scheduler_nodes.NODES["node007"]["min_free_user_threads_for_gpu_task"] == 256
    assert not scheduler_nodes.NODES["node007"].get("retired", False)
    assert scheduler_nodes.NODES["node007"]["max_concurrent_running"] == 24
    assert scheduler_nodes.NODES["node007"]["max_tasks_per_gpu"] == 4
    assert scheduler_nodes.NODES["node007"]["small_vram_max_tasks_per_gpu"] == 6
    for name in ("gpu_a", "gpu_b"):
        node = scheduler_nodes.NODES[name]
        for key, expected in scheduler_nodes.GPU_A_CAPACITY_POLICY.items():
            assert node[key] == expected
    assert (
        "jax_experiments/analysis"
        in scheduler_nodes.NODES["node007"]["launch_stage_include_paths"]
    )
    assert scheduler_nodes.NODES["login_hpc"]["monitor_only"] is True
    for name in ("gpu_a", "gpu_b", "cpu_a"):
        assert (
            scheduler_nodes.NODES[name]["launch_stage_include_paths"]
            == scheduler_nodes.BAPR_JAX_LAUNCH_INCLUDE_PATHS
        )
    assert "BAPR" not in scheduler_nodes.NODES["cpu_a"].get(
        "blocked_gpu_projects", [])

    for idx in range(1, 7):
        name = f"node{idx:03d}"
        node = scheduler_nodes.NODES[name]
        assert not node.get("retired", False)
        assert node["cpu_cores"] == node["physical_cores"] == 96
        assert node["max_concurrent_running"] is None
        assert node["sudo_ssh_host"] == name
        assert node["ssh_proxy_jumps"] == ["gpu_a", "gpu_b"]
        assert node["max_vram_per_task"] == 0
        assert node["cpu_slot_accounting"] is True
        assert node["reserved_cpu_cores"] == 1
        assert node["live_cpu_backfill_max_task_cores"] == 64
        assert node["live_cpu_backfill_startup_grace_s"] == 90
        assert node["live_cpu_backfill_single_thread_startup_grace_s"] == 10
        assert node["persistent_cpu_live_backfill_max_declared_cores"] == 138
        assert node["min_free_user_threads_for_jax_cpu_task"] == 1024
        assert node["jax_cpu_launch_user_thread_reserve"] == 512
        assert node["capabilities"] == ["cpu", "jax_cpu"]
        assert (
            node["launch_stage_include_paths"]
            == scheduler_nodes.BAPR_JAX_LAUNCH_INCLUDE_PATHS
        )
        assert (
            node["launch_stage_fingerprint_exclude_paths"]
            == scheduler_nodes.LAUNCH_STAGE_FINGERPRINT_EXCLUDE_PATHS
        )
        assert node["shared_workspace_group"] == "login_hpc-home"
        assert node["resume_scan_python"].endswith(
            "/conda_envs/csbapr-gpu-py310/bin/python")


def test_scheduler_imports_shared_node_inventory(sch):
    assert sch.NODES is sys.modules["scheduler_node.inventory"].NODES
    assert sch._canonical_node_name("node007-direct") == "node007"


def test_bus_module_python_rewrites_to_each_nodes_torch_environment(sch):
    command = (
        "/home/researcher/anaconda3/bin/python -u -m "
        "bus_experiments.run_policy_bank_train_v1 --seed 86003"
    )
    jtl = sch._apply_node_cmd_rewrites("gpu_a", command)
    node007 = sch._apply_node_cmd_rewrites("node007", command)

    assert jtl.startswith(
        scheduler_nodes.OFFLINE_SUMO_PYTHON + " -u -m bus_experiments")
    assert node007.startswith(
        scheduler_nodes.NODE007_RESAC_JAX_ENV
        + "/bin/python -u -m bus_experiments")


def test_pinned_jax_gpu_python_uses_isolated_node007_environment(sch):
    command = scheduler_nodes.GPU_A_RE_SAC_JAX_ENV + "/bin/python -u -m jax_experiments.analysis.run_regularized_comparison_v39 train"
    assert sch._apply_node_cmd_rewrites("node007", command) == (
        scheduler_nodes.NODE007_RESAC_JAX_0438_ENV
        + "/bin/python -u -m jax_experiments.analysis.run_regularized_comparison_v39 train"
    )


def test_gpu_launch_debit_does_not_invent_thread_usage_without_configured_reserve(sch):
    nodes = [{
        "name": "node007",
        "free_cpu": 64,
        "observed_free_cpu": 64,
        "free_ram_mb": 100_000,
        "running_count": 0,
        "user_threads": 2_000,
        "free_user_threads": 2_096,
        "gpus": [{
            "idx": 0,
            "used_mb": 0,
            "free_mb": 12_000,
            "running_task_count": 0,
        }],
    }]
    task = {
        "node": "node007",
        "gpu_idx": 0,
        "cpu_cores": 2,
        "ram_mb": 4_096,
        "est_vram_mb": 2_800,
    }

    sch._debit_node_resources_for_launch(nodes, task)

    assert nodes[0]["user_threads"] == 2_000
    assert nodes[0]["free_user_threads"] == 2_096


def test_resac_lstm_environment_maps_only_on_node007(sch):
    command = (
        "/home/researcher/anaconda3/envs/LSTM-RL/bin/python -u -m "
        "jax_experiments.experiments.bus_head_alignment_confirmation --day 2"
    )
    rewritten = sch._apply_node_cmd_rewrites("node007", command)
    assert rewritten == command.replace(
        "/home/researcher/anaconda3/envs/LSTM-RL/bin/python",
        "/home/hpcuser/scheduleurm_work/conda_envs/sumo-lstm-py310/bin/python",
    )
    for node in ("gpu_a", "gpu_b", "cpu_a", "local"):
        assert sch._apply_node_cmd_rewrites(node, command) == command


def test_jax_cpu_launch_debit_reserves_transient_runtime_threads(sch):
    nodes = [{
        "name": "node006",
        "free_cpu": 96,
        "observed_free_cpu": 96,
        "free_ram_mb": 100_000,
        "running_count": 0,
        "user_threads": 1_000,
        "free_user_threads": 3_096,
        "gpus": [],
    }]
    task = {
        "node": "node006",
        "gpu_idx": None,
        "cpu_cores": 8,
        "ram_mb": 8_192,
        "est_vram_mb": 0,
        "cmd": "JAX_PLATFORMS=cpu python -m eval",
    }

    sch._debit_node_resources_for_launch(nodes, task)

    assert nodes[0]["user_threads"] == 1_512
    assert nodes[0]["free_user_threads"] == 2_584
    assert nodes[0]["observed_free_cpu"] == 96


def test_hpc_cpu_launch_debit_records_startup_hold_release_window(sch):
    nodes = [{
        "name": "node001",
        "free_cpu": 0,
        "cpu_slot_reserved": 240,
        "cpu_slot_free": 0,
        "cpu_hard_reserved": 0,
        "cpu_hard_free": 48,
        "cpu_startup_hold_remaining_s": 0,
        "total_cpu": 96,
        "free_ram_mb": 160_000,
        "running_count": 20,
        "gpus": [],
    }]
    task = {
        "node": "node001",
        "gpu_idx": None,
        "cpu_cores": 12,
        "ram_mb": 8_192,
        "est_vram_mb": 0,
    }

    sch._debit_node_resources_for_launch(nodes, task)

    assert nodes[0]["cpu_hard_reserved"] == 12
    assert nodes[0]["cpu_hard_free"] == 36
    assert nodes[0]["cpu_startup_hold_remaining_s"] == 90


def test_hpc_cpu_wave_fills_to_one_core_headroom(sch):
    nodes = [{
        "name": "node001",
        "free_cpu": 96,
        "observed_free_cpu": 96,
        "cpu_slot_accounting": True,
        "cpu_slot_free": 96,
        "cpu_hard_free": 96,
        "total_cpu": 96,
        "free_ram_mb": 160_000,
        "running_count": 0,
        "gpus": [],
    }]
    task = {
        "node": "node001",
        "gpu_idx": None,
        "cpu_cores": 12,
        "ram_mb": 1_024,
        "est_vram_mb": 0,
    }
    node_info = sch.NODES["node001"]

    assert sch._node_schedulable_cpu_free(task, nodes[0], node_info)[0] == 95
    for _ in range(7):
        free, _, _, _ = sch._node_schedulable_cpu_free(task, nodes[0], node_info)
        assert free >= 12
        sch._debit_node_resources_for_launch(nodes, task)

    free, source, slot_free, observed_free = sch._node_schedulable_cpu_free(
        task, nodes[0], node_info
    )
    assert (free, source, slot_free, observed_free) == (
        11,
        "live_guarded",
        12,
        96,
    )
    task["cpu_cores"] = 11
    sch._debit_node_resources_for_launch(nodes, task)
    assert sch._node_schedulable_cpu_free(task, nodes[0], node_info) == (
        0, "live_guarded", 1, 96,
    )
