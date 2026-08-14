from __future__ import annotations

from skill.scheduler_placement_engine import helpers


def test_ram_headroom_and_snapshot_use_probe_total_and_eviction_grace():
    assert helpers.node_ram_headroom_mb(
        {"total_ram_mb": 10000},
        {"ram_mb": 20000},
        default_frac=0.1,
    ) == 1000
    assert helpers.node_ram_headroom_mb(
        {"total_ram_mb": 10000},
        {"ram_headroom_mb": 2048},
        default_frac=0.1,
    ) == 2048

    snap = helpers.node_ram_snapshot(
        {"name": "node001", "total_ram_mb": 10000, "free_ram_mb": 900},
        node_configs={"node001": {"ram_mb": 20000, "ram_headroom_frac": 0.1}},
        node_ram_headroom_mb=lambda node_state, node_info: helpers.node_ram_headroom_mb(
            node_state, node_info, default_frac=0.1
        ),
        ram_headroom_eviction_grace_mb=200,
    )

    assert snap["headroom_mb"] == 1000
    assert snap["eviction_headroom_mb"] == 800
    assert snap["below_headroom"] is True
    assert snap["below_eviction_headroom"] is False
    assert snap["headroom_gap_mb"] == 100


def test_gpu_freeze_line_and_threshold_snapshot():
    assert helpers.gpu_freeze_line_mb(0, one_third_pack_grace_mb=512) == 0
    assert helpers.gpu_freeze_line_mb(12288, one_third_pack_grace_mb=512) == 4608

    snap = helpers.gpu_threshold_snapshot(
        {"idx": 2, "total_mb": 12000, "used_mb": 5000, "util_pct": 91},
        gpu_freeze_line_mb=lambda total: helpers.gpu_freeze_line_mb(total, one_third_pack_grace_mb=500),
    )

    assert snap["idx"] == 2
    assert snap["one_third_mb"] == 4000
    assert snap["freeze_line_mb"] == 4500
    assert snap["over_one_third"] is True
    assert snap["over_freeze_line"] is True
    assert snap["used_pct"] == 42
    assert helpers.gpu_threshold_snapshot(None, gpu_freeze_line_mb=lambda total: 0) is None


def test_assign_windows_pin_plan_finds_free_contiguous_slot_and_clears_non_windows():
    task = {"id": "target", "node": "win", "cpu_cores": 2}
    state = {
        "tasks": [
            {"id": "a", "status": "running", "node": "win", "cpu_cores": 2, "windows_pin_base": 0, "started_at": 1},
            {"id": "b", "status": "launching", "node": "win", "cpu_cores": 1, "windows_pin_base": 4, "started_at": 2},
            task,
        ]
    }

    helpers.assign_windows_pin_plan(
        task,
        state,
        {"name": "win"},
        node_is_windows=lambda node: node == "win",
        node_physical_cores=lambda node, node_state: 5,
        node_configs={"win": {"cpu_cores": 5}},
        default_cpu_cores=1,
    )

    assert task["windows_pin_base"] == 2
    assert task["windows_pin_cores"] == 2

    task.update({"node": "linux", "windows_pin_base": 1, "windows_pin_cores": 2})
    helpers.assign_windows_pin_plan(
        task,
        state,
        None,
        node_is_windows=lambda node: False,
        node_physical_cores=lambda node, node_state: 4,
        node_configs={},
        default_cpu_cores=1,
    )
    assert "windows_pin_base" not in task
    assert "windows_pin_cores" not in task


def test_gpu_capacity_and_remote_gpu_cpu_ignore_policy():
    assert helpers.task_is_gpu_capacity_task({"est_vram_mb": 0}, default_vram_mb=512) is False
    assert helpers.task_is_gpu_capacity_task({"est_vram_mb": "bad"}, default_vram_mb=512) is True
    assert helpers.task_is_gpu_capacity_task(
        {"est_vram_mb": 1024, "cpu_fallback_selected": True},
        default_vram_mb=512,
    ) is False

    task = {"est_vram_mb": 1024, "node": "remote", "gpu_idx": 0}
    assert helpers.ignore_cpu_for_server_gpu_task(
        task,
        task_is_gpu_capacity_task=lambda task: True,
        node_is_windows=lambda node: False,
        node_configs={"remote": {"host": "host"}},
    ) is True
    assert helpers.ignore_cpu_for_server_gpu_task(
        task,
        node_name="local",
        task_is_gpu_capacity_task=lambda task: True,
        node_is_windows=lambda node: False,
        node_configs={"local": {"host": None}},
    ) is False
    assert helpers.ignore_cpu_for_server_gpu_task(
        task,
        node_state={"name": "remote", "gpus": []},
        task_is_gpu_capacity_task=lambda task: True,
        node_is_windows=lambda node: False,
        node_configs={"remote": {"host": "host"}},
    ) is False
    assert helpers.ignore_cpu_for_server_gpu_task(
        task,
        node_info={"host": "host", "ignore_cpu_for_gpu_tasks": "off"},
        gpu_idx=0,
        task_is_gpu_capacity_task=lambda task: True,
        node_is_windows=lambda node: False,
        node_configs={},
    ) is False


def test_one_third_override_gpu_util_limit_and_cpu_fallback_helpers():
    calls = []

    assert helpers.task_ignores_one_third_pack_rule(
        {"signature": "BAPR/bus_v2/bapr/s2"},
        {},
        hard_rule_bypassed=lambda rule, *args, **kwargs: calls.append(rule) or False,
    ) is True
    assert calls == ["one_third_pack"]
    assert helpers.task_ignores_one_third_pack_rule(
        {"allow_gpu_over_one_third": "1"},
        {},
        hard_rule_bypassed=lambda *args, **kwargs: False,
    ) is True
    assert helpers.task_ignores_one_third_pack_rule(
        {},
        {"allow_gpu_over_one_third": True},
        hard_rule_bypassed=lambda *args, **kwargs: False,
    ) is True
    assert helpers.task_ignores_one_third_pack_rule(
        {},
        {},
        hard_rule_bypassed=lambda *args, **kwargs: True,
    ) is True

    assert helpers.node_gpu_util_limit({}, default_limit=85) == 85
    assert helpers.node_gpu_util_limit({"gpu_util_saturation_pct": "ignore"}, default_limit=85) is None
    assert helpers.node_gpu_util_limit({"gpu_util_saturation_pct": "75"}, default_limit=85) == 75

    task = {
        "est_vram_mb": 1000,
        "cpu_fallback_selected": True,
        "cpu_fallback_original_vram_mb": 1000,
        "cpu_fallback_capability": "x",
    }
    assert helpers.clear_disallowed_cpu_fallback_selection(task) is True
    assert "cpu_fallback_selected" not in task

    allowed = {"est_vram_mb": 1000, "cpu_fallback_selected": True, "allow_cpu_training": True}
    assert helpers.clear_disallowed_cpu_fallback_selection(allowed) is False
    assert helpers.task_launch_cpu_mode({"est_vram_mb": 0}) is True
    assert helpers.task_launch_cpu_mode(allowed) is True
    assert helpers.task_launch_cpu_mode({"est_vram_mb": 1000}) is False
