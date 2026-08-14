from skill.scheduler_dispatch_core import (
    DispatchAccountingDeps,
    build_dispatch_queue_plan,
    prepare_dispatch_node_accounting,
)


def test_build_dispatch_queue_plan_orders_targets_and_running_keys():
    state = {
        "tasks": [
            {"id": "r1", "status": "running", "identity": "same"},
            {"id": "l1", "status": "launching", "identity": "launching-key"},
            {"id": "q-low", "status": "queued", "priority": "low", "submitted_at": 1},
            {"id": "q-high-late", "status": "queued", "priority": "high", "submitted_at": 5},
            {"id": "q-high-early", "status": "queued", "priority": "high", "submitted_at": 2},
        ]
    }
    seen_queued = []

    def global_batch_plan(queued, nodes):
        seen_queued.extend(t["id"] for t in queued)
        return {"q-high-early": {"node": "nodeA"}}, {"type": "algorithm_global_batch_plan"}

    plan = build_dispatch_queue_plan(
        state,
        [{"name": "nodeA"}],
        {"q-low", "q-high-early"},
        {"high": 0, "normal": 1, "low": 2},
        task_run_identity=lambda t: t.get("identity"),
        algorithm_global_batch_plan=global_batch_plan,
    )

    assert plan.running_keys == {"same", "launching-key"}
    assert [t["id"] for t in plan.queued] == ["q-high-early", "q-low"]
    assert seen_queued == ["q-high-early", "q-low"]
    assert plan.global_batch_plan == {"q-high-early": {"node": "nodeA"}}
    assert plan.global_batch_event == {"type": "algorithm_global_batch_plan"}


def test_prepare_dispatch_node_accounting_seeds_counts_and_preemption():
    state = {
        "tasks": [
            {"id": "r1", "status": "running", "node": "nodeA", "gpu_idx": 0},
            {"id": "r2", "status": "running", "node": "nodeA", "gpu_idx": None},
            {"id": "q1", "status": "queued", "node": None, "gpu_idx": None},
        ]
    }
    nodes = [
        {
            "name": "nodeA",
            "free_cpu": 2,
            "free_ram_mb": 1000,
            "gpus": [{"idx": 0}, {"idx": 1}],
        },
        {
            "name": "nodeB",
            "free_cpu": 8,
            "free_ram_mb": 4000,
            "gpus": [],
        },
    ]
    applied = []

    def apply_cpu_accounting(seen_state, seen_nodes):
        applied.append((seen_state, seen_nodes))

    deps = DispatchAccountingDeps(
        counts_against_node_concurrency=lambda t: t.get("status") == "running",
        apply_cpu_slot_accounting_to_nodes=apply_cpu_accounting,
    )
    events = prepare_dispatch_node_accounting(
        state,
        nodes,
        [{
            "id": "r1",
            "node": "nodeA",
            "cpu_freed": 4,
            "ram_freed": 512,
            "target_id": "hi",
            "cpu_deficit": 3,
            "ram_deficit": 256,
            "protected_skipped": ["keep"],
        }],
        deps=deps,
    )

    assert applied == [(state, nodes)]
    assert nodes[0]["running_count"] == 1
    assert nodes[0]["gpus"][0]["running_task_count"] == 1
    assert nodes[0]["gpus"][1]["running_task_count"] == 0
    assert nodes[0]["free_cpu"] == 6
    assert nodes[0]["free_ram_mb"] == 1512
    assert nodes[1]["running_count"] == 0
    assert events == [{
        "type": "preempted",
        "task_id": "r1",
        "freed_node": "nodeA",
        "cpu_freed": 4,
        "ram_freed": 512,
        "target_id": "hi",
        "cpu_deficit": 3,
        "ram_deficit": 256,
        "protected_skipped": ["keep"],
    }]
