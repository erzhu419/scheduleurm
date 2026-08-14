from __future__ import annotations

from skill.scheduler_claim import records as claim_records


def test_claim_intent_nodes_deduplicates_legacy_single_and_list_markers():
    task = {
        "claim_intent_node": "node001",
        "claim_intent_nodes": ["node001", "node002", "", "node003"],
    }

    assert claim_records.claim_intent_nodes(task) == ["node001", "node002", "node003"]
    assert claim_records.claim_intent_nodes({"claim_intent_nodes": "node004"}) == ["node004"]


def test_remember_and_clear_claim_intent_markers():
    task = {"claim_intent_nodes": ["node001"]}

    claim_records.remember_claim_intent(
        task,
        "node002",
        claim_enabled_for=lambda node: node != "disabled",
        now=lambda: 123.0,
    )
    claim_records.remember_claim_intent(
        task,
        "disabled",
        claim_enabled_for=lambda node: node != "disabled",
        now=lambda: 999.0,
    )

    assert task["claim_intent_nodes"] == ["node001", "node002"]
    assert task["claim_intent_at"] == 123.0

    claim_records.clear_claim_intent_markers(task)
    assert "claim_intent_node" not in task
    assert "claim_intent_nodes" not in task
    assert "claim_intent_at" not in task


def test_claim_resource_record_uses_observed_vram_and_can_ignore_cpu():
    calls = []

    record = claim_records.claim_resource_record_for_task(
        {
            "id": "t1",
            "node": "node001",
            "gpu_idx": 2,
            "remote_pids": ["4321"],
            "current_vram_mb": 80,
            "peak_vram_mb": 120,
            "est_vram_mb": 600,
            "current_ram_mb": 8192,
            "ram_mb": 4096,
            "cpu_cores": 8,
        },
        node_configs={"node001": {"kind": "gpu"}},
        ignore_cpu_for_server_gpu_task=lambda task, **kwargs: calls.append(kwargs) or True,
        task_ignores_one_third_pack_rule=lambda task, node_info: node_info.get("kind") == "gpu",
        default_ram_mb=1024,
        default_cpu_cores=1,
        startup_floor_mb=500,
    )

    assert record == {
        "gpu_idx": 2,
        "vram_mb": 120,
        "cpu_cores": 0,
        "ram_mb": 8192,
        "ignore_cpu_capacity": True,
        "ignore_one_third_pack_rule": True,
        "pid": 4321,
    }
    assert calls[0]["node_name"] == "node001"
    assert calls[0]["gpu_idx"] == 2


def test_claim_resource_record_caps_startup_estimate_before_observation():
    record = claim_records.claim_resource_record_for_task(
        {
            "node": "node001",
            "est_vram_mb": 2000,
            "cpu_cores": 0,
        },
        node_configs={"node001": {}},
        ignore_cpu_for_server_gpu_task=lambda task, **kwargs: False,
        task_ignores_one_third_pack_rule=lambda task, node_info: False,
        default_ram_mb=4096,
        default_cpu_cores=1,
        startup_floor_mb=500,
    )

    assert record["vram_mb"] == 500
    assert record["cpu_cores"] == 1
    assert record["ram_mb"] == 4096
    assert record["pid"] is None


def test_release_task_claims_and_intents_deduplicates_filters_and_preserves_unreleased_markers():
    released = []

    def claim_release(node, task_id):
        released.append((node, task_id))
        return node != "node002"

    task = {
        "id": "t1",
        "node": "node001",
        "last_node": "node001",
        "claim_intent_nodes": ["node002", "node003"],
    }

    count = claim_records.release_task_claims_and_intents(
        task,
        claim_enabled_for=lambda node: node != "disabled",
        claim_release=claim_release,
        extra_nodes=["node003", "disabled"],
        exclude_nodes={"node001"},
        clear_markers=False,
    )

    assert count == 1
    assert released == [("node002", "t1"), ("node003", "t1")]
    assert task["claim_intent_nodes"] == ["node002"]


def test_release_task_claims_and_intents_clears_markers_even_when_release_raises():
    task = {"id": "t1", "claim_intent_node": "node001", "claim_intent_nodes": ["node001"]}

    count = claim_records.release_task_claims_and_intents(
        task,
        claim_enabled_for=lambda node: True,
        claim_release=lambda node, task_id: (_ for _ in ()).throw(RuntimeError("ssh down")),
    )

    assert count == 0
    assert "claim_intent_node" not in task
    assert "claim_intent_nodes" not in task
