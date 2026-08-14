from __future__ import annotations

from skill.scheduler_dispatch_placement_apply import (
    DispatchPlacementApplyDeps,
    apply_dispatch_placement,
)


def _deps(assign_calls: list, *, cpu_block_reason: str = ""):
    return DispatchPlacementApplyDeps(
        node_configs={
            "node001": {"allow_cpu_fallback": True},
            "node002": {"allow_cpu_fallback": False},
        },
        algorithm_name=lambda: "test_algo",
        algorithm_config_snapshot=lambda: {"mode": "test"},
        node_cpu_fallback_block_reason=lambda task, node, info: cpu_block_reason,
        task_cpu_fallback_capability=lambda task: "cpu-cap",
        algorithm_selected_gpu_audit=lambda task, node, gpu: {
            "task": task["id"],
            "node": node["name"],
            "gpu": gpu["idx"],
        },
        assign_windows_pin_plan=lambda task, state, picked_state: assign_calls.append(
            (task["id"], picked_state["name"] if picked_state else None)
        ),
    )


def test_apply_dispatch_placement_sets_algorithm_hint_cpu_fallback_and_pin_plan():
    assign_calls = []
    task = {"id": "t1", "est_vram_mb": 1024}
    nodes = [{"name": "node001", "gpus": []}]
    state = {"tasks": [task]}

    picked = apply_dispatch_placement(
        task,
        placement=("node001", None),
        nodes=nodes,
        state=state,
        batch_hint={"node": "node001", "gpu_idx": None},
        deps=_deps(assign_calls),
    )

    assert picked == nodes[0]
    assert task["node"] == "node001"
    assert task["gpu_idx"] is None
    assert task["placement_algorithm"] == "test_algo"
    assert task["placement_algorithm_config"] == {"mode": "test"}
    assert task["placement_algorithm_global_batch_hint"] == {
        "node": "node001",
        "gpu_idx": None,
        "execution_contract": None,
        "matched": True,
    }
    assert task["cpu_fallback_selected"] is True
    assert task["cpu_fallback_original_vram_mb"] == 1024
    assert task["cpu_fallback_capability"] == "cpu-cap"
    assert "placement_algorithm_audit" not in task
    assert assign_calls == [("t1", "node001")]


def test_apply_dispatch_placement_sets_gpu_audit_and_clears_cpu_fallback_fields():
    assign_calls = []
    task = {
        "id": "t2",
        "est_vram_mb": 2048,
        "cpu_fallback_selected": True,
        "cpu_fallback_original_vram_mb": 2048,
        "cpu_fallback_capability": "old",
    }
    nodes = [{"name": "node002", "gpus": [{"idx": 1}, {"idx": "2"}]}]
    state = {"tasks": [task]}

    picked = apply_dispatch_placement(
        task,
        placement=("node002", 2),
        nodes=nodes,
        state=state,
        batch_hint={"node": "node002", "gpu_idx": 1},
        deps=_deps(assign_calls),
    )

    assert picked == nodes[0]
    assert task["gpu_idx"] == 2
    assert task["placement_algorithm_global_batch_hint"]["matched"] is False
    assert task["placement_algorithm_audit"] == {
        "task": "t2",
        "node": "node002",
        "gpu": "2",
    }
    assert "cpu_fallback_selected" not in task
    assert "cpu_fallback_original_vram_mb" not in task
    assert "cpu_fallback_capability" not in task
    assert assign_calls == [("t2", "node002")]


def test_apply_dispatch_placement_clears_stale_hint_and_audit_when_not_applicable():
    assign_calls = []
    task = {
        "id": "t3",
        "est_vram_mb": 0,
        "placement_algorithm_global_batch_hint": {"old": True},
        "placement_algorithm_audit": {"old": True},
    }
    nodes = [{"name": "node001", "gpus": [{"idx": 0}]}]
    state = {"tasks": [task]}

    apply_dispatch_placement(
        task,
        placement=("node001", None),
        nodes=nodes,
        state=state,
        batch_hint=None,
        deps=_deps(assign_calls),
    )

    assert "placement_algorithm_global_batch_hint" not in task
    assert "placement_algorithm_audit" not in task
    assert "cpu_fallback_selected" not in task
    assert assign_calls == [("t3", "node001")]
