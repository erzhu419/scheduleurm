from __future__ import annotations

from skill.scheduler_resource.accounting import (
    AggregateVramReconcileDeps,
    InflightVramReservationDeps,
    ResourceAccountingDeps,
    cpu_ownership_snapshot,
    dispatch_cycle_log_payload,
    reconcile_aggregate_only_vram,
    resource_accounting_payload,
    reserve_inflight_vram,
)


def _deps():
    return ResourceAccountingDeps(
        scheduler_id=lambda: "sid",
        reserved_cpu_slots_for_task=lambda task: int(task.get("cpu_cores") or 0),
        node_physical_cores=lambda name, node=None: int((node or {}).get("total_cpu") or 0),
        algorithm_name=lambda: "legacy",
        algorithm_config_snapshot=lambda: {"mode": "test"},
    )


def test_cpu_ownership_snapshot_separates_ours_external_and_other():
    state = {"tasks": [
        {"id": "ours", "status": "running", "node": "nodeA",
         "origin": "scheduleurm", "scheduler_id": "sid", "cpu_cores": 4},
        {"id": "external", "status": "running", "node": "nodeA",
         "origin": "scheduleurm", "scheduler_id": "other", "cpu_cores": 3},
        {"id": "queued", "status": "queued", "node": "nodeA",
         "origin": "scheduleurm", "scheduler_id": "sid", "cpu_cores": 8},
    ]}
    nodes = [{"name": "nodeA", "alive": True, "total_cpu": 16, "free_cpu": 6}]

    row = cpu_ownership_snapshot(state, nodes, deps=_deps())[0]

    assert row["used_cpu_est"] == 10
    assert row["scheduleurm_cpu_reserved"] == 4
    assert row["external_tracked_cpu_reserved"] == 3
    assert row["untracked_or_other_user_cpu_est"] == 3
    assert row["scheduleurm_task_ids"] == ["ours"]
    assert row["external_tracked_task_ids"] == ["external"]


def test_resource_accounting_payload_counts_statuses_and_nodes():
    state = {"tasks": [
        {"id": "r", "status": "running", "node": "nodeA", "origin": "external", "cpu_cores": 1},
        {"id": "l", "status": "launching", "node": "nodeA", "origin": "scheduleurm", "scheduler_id": "sid", "cpu_cores": 1},
        {"id": "q", "status": "queued"},
    ]}
    nodes = [{
        "name": "nodeA",
        "alive": True,
        "total_cpu": 4,
        "free_cpu": 2,
        "cpu_startup_hold_remaining_s": 37,
    }]

    payload = resource_accounting_payload(state, nodes, deps=_deps())

    assert payload["running"] == 1
    assert payload["launching"] == 1
    assert payload["queued"] == 1
    assert payload["nodes"][0]["total_cpu"] == 4
    assert payload["nodes"][0]["cpu_startup_hold_remaining_s"] == 37


def test_dispatch_cycle_log_payload_adds_event_summaries_and_algorithm():
    task = {"id": "tq", "cpu_cores": 2, "ram_mb": 100, "est_vram_mb": 0, "last_block_reason": "waiting"}
    payload = dispatch_cycle_log_payload(
        {"tasks": [task]},
        [{"name": "nodeA", "alive": True, "total_cpu": 8, "free_cpu": 8}],
        [
            {"type": "blocked", "task_id": "tb", "reason": "gate"},
            {"type": "no_fit", "task_id": "tq", "task": task},
            {"type": "launched", "task_id": "tr", "task": {"id": "tr", "node": "nodeA", "cpu_cores": 1}},
        ],
        3,
        deps=_deps(),
    )

    assert payload["placement_algorithm"] == "legacy"
    assert payload["placement_algorithm_config"] == {"mode": "test"}
    assert payload["queued_seen_by_dispatch"] == 3
    assert payload["event_counts"] == {"blocked": 1, "no_fit": 1, "launched": 1}
    assert payload["blocked"] == [{"task_id": "tb", "reason": "gate"}]
    assert payload["no_fit"][0]["reason"] == "waiting"
    assert payload["launched"][0]["task_id"] == "tr"


def test_reconcile_aggregate_only_vram_distributes_residual_by_estimate_weight():
    state = {"tasks": [
        {
            "id": "known", "status": "running", "node": "nodeA", "gpu_idx": 0,
            "current_vram_mb": 300,
        },
        {
            "id": "small", "status": "running", "node": "nodeA", "gpu_idx": 0,
            "est_vram_mb": 100,
            "last_launch_pre_snapshot": {
                "target_node": "nodeA",
                "target_gpu_idx": 0,
                "gpu": {"used_mb": 200},
            },
        },
        {
            "id": "large", "status": "running", "node": "nodeA", "gpu_idx": 0,
            "est_vram_mb": 300,
            "last_launch_pre_snapshot": {
                "target_node": "nodeA",
                "target_gpu_idx": 0,
                "gpu": {"used_mb": 200},
            },
        },
    ]}
    nodes = [{"name": "nodeA", "alive": True, "gpus": [{"idx": 0, "observed_used_mb": 1300}]}]

    changed = reconcile_aggregate_only_vram(
        state,
        nodes,
        deps=AggregateVramReconcileDeps(is_slurm_managed=lambda task: task.get("slurm", False)),
    )

    assert changed == 2
    small = state["tasks"][1]
    large = state["tasks"][2]
    assert small["current_vram_mb"] == 200
    assert large["current_vram_mb"] == 600
    assert small["vram_estimation_source"] == "aggregate_residual"
    assert large["peak_vram_mb"] == 600


def test_reconcile_aggregate_only_vram_marks_unknown_zero_when_no_residual():
    state = {"tasks": [
        {"id": "t1", "status": "running", "node": "nodeA", "gpu_idx": 0},
    ]}
    nodes = [{"name": "nodeA", "alive": True, "gpus": [{"idx": 0, "observed_used_mb": 50}]}]

    changed = reconcile_aggregate_only_vram(
        state,
        nodes,
        deps=AggregateVramReconcileDeps(is_slurm_managed=lambda task: False),
    )

    assert changed == 1
    assert state["tasks"][0]["current_vram_mb"] == 0
    assert state["tasks"][0]["vram_estimation_source"] == "aggregate_observed_zero"


def test_reserve_inflight_vram_adds_startup_floor_until_progress_or_peak():
    state = {"tasks": [
        {"id": "startup", "status": "launching", "node": "nodeA", "gpu_idx": 0, "est_vram_mb": 3000},
        {"id": "progress", "status": "running", "node": "nodeA", "gpu_idx": 0, "est_vram_mb": 3000, "last_progress_line": "1/10"},
        {"id": "peaked", "status": "running", "node": "nodeA", "gpu_idx": 0, "est_vram_mb": 3000, "peak_vram_mb": 500},
        {"id": "cpu", "status": "running", "node": "nodeA", "gpu_idx": None, "est_vram_mb": 3000},
    ]}
    nodes = [{"name": "nodeA", "alive": True, "gpus": [{"idx": 0, "used_mb": 100, "free_mb": 7900, "total_mb": 8000}]}]

    reserve_inflight_vram(
        state,
        nodes,
        deps=InflightVramReservationDeps(
            node_configs={"nodeA": {}},
            startup_floor_mb=500,
            is_slurm_managed=lambda task: False,
            hard_rule_bypassed=lambda *args, **kwargs: False,
        ),
    )

    assert nodes[0]["gpus"][0]["used_mb"] == 600
    assert nodes[0]["gpus"][0]["free_mb"] == 7400


def test_reserve_inflight_vram_respects_hard_rule_and_slurm_skip():
    state = {"tasks": [
        {"id": "skip_rule", "status": "launching", "node": "nodeA", "gpu_idx": 0, "est_vram_mb": 3000},
        {"id": "skip_slurm", "status": "launching", "node": "nodeA", "gpu_idx": 0, "est_vram_mb": 3000, "slurm": True},
    ]}
    nodes = [{"name": "nodeA", "alive": True, "gpus": [{"idx": 0, "used_mb": 100, "free_mb": 7900, "total_mb": 8000}]}]

    reserve_inflight_vram(
        state,
        nodes,
        deps=InflightVramReservationDeps(
            node_configs={"nodeA": {}},
            startup_floor_mb=500,
            is_slurm_managed=lambda task: bool(task.get("slurm")),
            hard_rule_bypassed=lambda rule, task, **kwargs: task["id"] == "skip_rule",
        ),
    )

    assert nodes[0]["gpus"][0]["used_mb"] == 100


def test_reserve_inflight_vram_keeps_explicit_peak_after_progress():
    state = {"tasks": [{
        "id": "late-peak",
        "status": "running",
        "node": "nodeA",
        "gpu_idx": 0,
        "est_vram_mb": 10500,
        "est_vram_mb_explicit": True,
        "current_vram_mb": 400,
        "peak_vram_mb": 400,
        "last_progress_line": "10/397",
        "runtime_current_unit": 10,
    }]}
    nodes = [{
        "name": "nodeA",
        "alive": True,
        "gpus": [{
            "idx": 0,
            "used_mb": 569,
            "free_mb": 11719,
            "total_mb": 12288,
        }],
    }]

    reserve_inflight_vram(
        state,
        nodes,
        deps=InflightVramReservationDeps(
            node_configs={"nodeA": {}},
            startup_floor_mb=500,
            is_slurm_managed=lambda task: False,
            hard_rule_bypassed=lambda *args, **kwargs: False,
        ),
    )

    gpu = nodes[0]["gpus"][0]
    assert gpu["observed_used_mb"] == 569
    assert gpu["reserved_vram_mb"] == 10100
    assert gpu["used_mb"] == 10669
    assert gpu["free_mb"] == 1619
