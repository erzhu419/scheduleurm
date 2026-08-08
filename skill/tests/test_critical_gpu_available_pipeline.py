from __future__ import annotations

from algorithm.experiments.critical_gpu_available_lcb_cache_merge import (
    AVAILABLE_NODES,
    EFFECTIVE_NODE_BUCKETS,
    _validate_nonpooling_equivalence,
)
from algorithm.experiments.critical_gpu_completion_campaign import (
    TRAINING_WAVES,
    campaign_cells,
)
from algorithm.experiments.critical_gpu_statewise_sota_replay_gate import (
    WORKLOAD_RESOURCE_KINDS,
    _dominates,
    _taskset,
    _weakly_noninferior,
)
from simulation.fast_forward import WorkloadSpec
from simulation.sota_baselines import SotaCandidateUnionPolicy
from simulation.sota_baselines import scheduleurm_tail_bridge_policies
from simulation.service_cache import ProfileRecord, ServiceRateCache


def test_nominally_identical_3080ti_nodes_are_distinct_execution_classes():
    assert len(set(EFFECTIVE_NODE_BUCKETS.values())) == len(AVAILABLE_NODES)
    assert EFFECTIVE_NODE_BUCKETS["jtl110gpu"] != EFFECTIVE_NODE_BUCKETS["jtl110gpu2"]


def test_nonpooling_gate_requires_constructed_nine_cell_failure(tmp_path):
    payload = {
        "gate": "critical_gpu_homogeneous_equivalence_gate",
        "status": "FAIL_EQUIVALENCE",
        "pass": False,
        "measurements_ready": True,
        "hardware_bucket_pooling_ready": False,
        "same_nine_workload_env_profile_cells": True,
        "representative_node": "jtl110gpu",
        "equivalence_node": "jtl110gpu2",
        "validation_errors": [],
        "wait_reasons": [],
        "comparison": {"constructed": True, "matched_cell_count": 9},
    }
    errors = []
    _validate_nonpooling_equivalence(payload, path=tmp_path / "gate.json", errors=errors)
    assert errors == []

    payload["hardware_bucket_pooling_ready"] = True
    _validate_nonpooling_equivalence(payload, path=tmp_path / "gate.json", errors=errors)
    assert errors[-1]["code"] == "NONPOOLING_EQUIVALENCE_CONTRACT_INVALID"


def test_gpu_replay_scopes_preserve_registered_action_profiles():
    cells = campaign_cells(node="node007", wave=TRAINING_WAVES[0])
    q01 = _taskset(node="node007", cells=cells, scope="q01", task_count_per_workload=8)
    q11 = _taskset(node="node007", cells=cells, scope="q11", task_count_per_workload=8)
    portfolio = _taskset(node="node007", cells=cells, scope="portfolio", task_count_per_workload=8)

    assert len(q01.members) == 3
    assert len(q11.members) == 1
    assert len(portfolio.members) == 4
    llm = next(member for member in q01.members if member.workload_key == "gpu_llm_distilgpt2")
    assert llm.required_profiles == (1, 3, 10)
    assert {member.resource_kind for member in portfolio.members} == set(
        WORKLOAD_RESOURCE_KINDS.values()
    )
    assert all(member.resource_count == 4 for member in portfolio.members)


def test_pareto_predicate_requires_noninferiority_and_real_improvement():
    assert _dominates(9.0, 10.0, 10.0, 10.0, tolerance=0.0)
    assert not _dominates(10.0, 10.0, 10.0, 10.0, tolerance=0.0)
    assert _dominates(10.04, 9.0, 10.0, 10.0, tolerance=0.005)
    assert not _dominates(10.06, 9.0, 10.0, 10.0, tolerance=0.005)
    assert _weakly_noninferior(10.0, 10.0, 10.0, 10.0, tolerance=0.005)
    assert _weakly_noninferior(10.04, 10.04, 10.0, 10.0, tolerance=0.005)
    assert not _weakly_noninferior(10.06, 10.0, 10.0, 10.0, tolerance=0.005)


def test_hybrid_packing_guard_uses_only_bounded_lower_service_tradeoff():
    selector = SotaCandidateUnionPolicy(
        name="test-union",
        selection_objective="online_pareto_slack",
    )
    spec = WorkloadSpec(
        workload_key="hybrid_rl_resac_ant",
        resource_kind="hybrid_rl",
        task_count=24,
        total_units=40.0,
        resource_count=2,
    )
    delay = {
        "family_name": "scheduleurm_bridge_hybrid_rl_p3_to_p2_tail",
        "action_id": "delay",
        "makespan_s": 100.0,
        "mean_flow_s": 80.0,
    }
    packing = {
        "family_name": "scheduleurm_bridge_hybrid_rl_statewise_packing_srf",
        "action_id": "packing",
        "makespan_s": 95.0,
        "mean_flow_s": 92.0,
    }
    assert selector._hybrid_statewise_packing_action_row(
        spec, [delay, packing]
    ) is packing

    excessive_delay = {**packing, "mean_flow_s": 97.0}
    assert selector._hybrid_statewise_packing_action_row(
        spec, [delay, excessive_delay]
    ) is None


def test_hybrid_packing_action_is_not_admitted_for_jax():
    action = next(
        policy
        for policy in scheduleurm_tail_bridge_policies()
        if "hybrid_rl_statewise_packing_srf" in policy.name
    )
    cache = ServiceRateCache(
        (
            ProfileRecord(
                workload_key="gpu_heavy_jax_matmul",
                command_fingerprint="test",
                resource_kind="gpu",
                node_bucket="",
                profile=1,
                unit="step",
                total_units=40.0,
                aggregate_rate=1.0,
                per_task_rates=(1.0,),
                source="test",
            ),
        )
    )
    jax = WorkloadSpec(
        workload_key="gpu_heavy_jax_matmul",
        resource_kind="gpu_heavy",
        task_count=24,
        total_units=40.0,
        resource_count=2,
    )
    import pytest

    with pytest.raises(KeyError, match="not admitted"):
        action.select_profile(cache, jax)


def test_hybrid_packing_action_rejects_uncalibrated_completion_row():
    action = next(
        policy
        for policy in scheduleurm_tail_bridge_policies()
        if "hybrid_rl_statewise_packing_srf" in policy.name
    )
    cache = ServiceRateCache(
        (
            ProfileRecord(
                workload_key="hybrid_rl_resac_ant",
                command_fingerprint="test",
                resource_kind="hybrid",
                node_bucket="",
                profile=5,
                unit="iter",
                total_units=40.0,
                aggregate_rate=1.0,
                per_task_rates=(0.2,) * 5,
                source="synthetic-default-cache",
            ),
        )
    )
    spec = WorkloadSpec(
        workload_key="hybrid_rl_resac_ant",
        resource_kind="hybrid_rl",
        task_count=24,
        total_units=40.0,
        resource_count=2,
    )

    import pytest

    with pytest.raises(KeyError, match="natural-completion"):
        action.select_profile(cache, spec)
