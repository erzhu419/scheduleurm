from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments.port_scheduling_benchmark import (
    ALL_POLICIES,
    BASELINE_POLICIES,
    PortSchedulingSimulator,
    THEOREM_POLICY,
    build_port_scheduling_benchmark,
    run_port_instance,
)
from algorithm.experiments.port_scheduling_instances import (
    built_in_port_instances,
    load_port_instance,
    port_instance_from_mapping,
)


DATA = Path(__file__).resolve().parent / "data" / "port_small_instance.json"


@pytest.fixture(scope="module")
def small_instance():
    return load_port_instance(DATA)


@pytest.fixture(scope="module")
def full_report():
    return build_port_scheduling_benchmark()


def test_registered_instances_are_non_toy_and_heterogeneous():
    instances = built_in_port_instances()

    assert len(instances) == 3
    assert sum(len(instance.vessels) for instance in instances) >= 24
    for instance in instances:
        assert len(instance.vessels) >= 8
        assert len(instance.berths) >= 3
        assert len(instance.quay_cranes) >= 4
        assert len(instance.yard_blocks) >= 3
        assert len(instance.gate_lanes) >= 2
        assert len({row.rate_multiplier for row in instance.berths}) > 1
        assert len({row.base_rate for row in instance.quay_cranes}) > 1
        assert len({row.rate("general") for row in instance.yard_blocks}) > 1
        instance.validate()


def test_json_instance_round_trip_is_deterministic(small_instance):
    second = load_port_instance(DATA)

    assert small_instance.snapshot() == second.snapshot()
    assert small_instance.digest == second.digest
    assert small_instance.name == "test_port_heterogeneous_4"


def test_instance_validation_rejects_unknown_preferred_resource(small_instance):
    raw = small_instance.snapshot()
    raw["vessels"][0]["preferred_berth"] = "does_not_exist"

    with pytest.raises(ValueError, match="unknown preferred resource"):
        port_instance_from_mapping(raw).validate()


def test_initial_candidate_family_enforces_berth_and_crane_compatibility(small_instance):
    simulator = PortSchedulingSimulator(small_instance, THEOREM_POLICY)
    simulator._release_arrivals()
    actions = simulator.feasible_atomic_actions()
    large_actions = [row for row in actions if row.vessel_id == "large"]

    assert large_actions
    assert all(row.action_type == "start_quay" for row in large_actions)
    assert all(row.metadata["berth_id"] == "b_deep" for row in large_actions)
    assert all("qc_small" not in row.metadata["crane_ids"] for row in large_actions)
    assert all(row.theorem_ready for row in actions)


def test_event_driven_run_completes_all_stages_and_resources(small_instance):
    result = run_port_instance(small_instance, THEOREM_POLICY)

    assert result["pass"] is True
    assert result["status"] == "PORT_SIMULATION_PASS"
    assert result["metrics"]["completed_vessels"] == len(small_instance.vessels)
    assert result["metrics"]["feasibility_violation_count"] == 0
    assert {row["event"] for row in result["event_trace"]} >= {
        "arrival",
        "dispatch",
        "stage_complete",
        "vessel_complete",
    }
    utilization = result["resource_utilization"]
    assert any(key.startswith("berth:") and value > 0.0 for key, value in utilization.items())
    assert any(key.startswith("crane:") and value > 0.0 for key, value in utilization.items())
    assert any(key.startswith("yard:") and value > 0.0 for key, value in utilization.items())
    assert any(key.startswith("gate:") and value > 0.0 for key, value in utilization.items())


def test_selected_global_configurations_are_resource_and_task_disjoint(small_instance):
    result = run_port_instance(small_instance, THEOREM_POLICY)

    for audit in result["decision_audits"]:
        resources = [
            resource
            for action in audit["selected_actions"]
            for resource in action["resource_ids"]
        ]
        task_ids = [action["task_id"] for action in audit["selected_actions"]]
        assert len(resources) == len(set(resources))
        assert len(task_ids) == len(set(task_ids))
        assert audit["candidate_configuration_count"] >= 1
        assert audit["statewise_feasible_family"] is True


def test_theorem_policy_has_exact_finite_family_oracle_certificate(small_instance):
    result = run_port_instance(small_instance, THEOREM_POLICY)

    assert result["theorem_contract"]["exact_finite_family_oracle"] is True
    assert result["theorem_contract"]["bounded_reconfiguration_penalty"] is True
    assert result["theorem_contract"]["lower_service_source"] == "registered_instance_conservative_rate"
    assert result["decision_audits"]
    assert all(row["score_semantics"] == "robust_maxweight_lower_service" for row in result["decision_audits"])
    assert all(row["oracle_gap_alpha0"] == 0.0 for row in result["decision_audits"])
    assert all(row["oracle_gap_alpha1"] == 0.0 for row in result["decision_audits"])
    assert all(row["theorem_ready"] is True for row in result["decision_audits"])


def test_repeated_simulation_is_byte_stable_at_hash_boundary(small_instance):
    first = run_port_instance(small_instance, THEOREM_POLICY)
    second = run_port_instance(small_instance, THEOREM_POLICY)

    assert first["result_hash"] == second["result_hash"]
    assert first["metrics"] == second["metrics"]
    assert first["decision_audits"] == second["decision_audits"]
    assert first["event_trace"] == second["event_trace"]


def test_all_policies_emit_identical_metric_schema(small_instance):
    rows = {policy: run_port_instance(small_instance, policy) for policy in ALL_POLICIES}
    schemas = {tuple(sorted(row["metrics"])) for row in rows.values()}

    assert len(BASELINE_POLICIES) >= 4
    assert len(schemas) == 1
    assert all(row["pass"] for row in rows.values())
    assert all(row["metrics"]["completed_vessels"] == 4 for row in rows.values())


def test_full_gate_exercises_all_migration_analogues_and_keeps_claim_boundary(full_report):
    assert full_report["pass"] is True
    assert full_report["status"] == "PORT_BENCHMARK_PASS"
    assert full_report["registered_instance_count"] == 3
    assert full_report["registered_vessel_count"] >= 24
    assert full_report["baseline_count"] >= 4
    assert full_report["migration_analogues_exercised"] == [
        "crane_reassignment",
        "reberth",
        "yard_rehandle",
    ]
    assert full_report["all_three_migration_analogues_exercised"] is True
    assert full_report["physical_port_claim_ready"] is False
    assert full_report["global_port_optimality_claim_ready"] is False
    assert full_report["frame_bridge_ready"] is False
    assert full_report["claim_boundary"]["does_not_support"]


def test_full_gate_reports_performance_without_conflating_it_with_validity(full_report):
    assert full_report["deterministic_reproduction_ready"] is True
    assert full_report["same_metric_schema_ready"] is True
    assert full_report["statewise_feasibility_ready"] is True
    assert full_report["exact_finite_family_oracle_ready"] is True
    assert full_report["ours_pareto_nondominated"] is True
    assert isinstance(full_report["ours_strictly_dominates_all_baselines"], bool)
    assert THEOREM_POLICY in full_report["aggregate_metrics"]
    assert set(BASELINE_POLICIES) <= set(full_report["aggregate_metrics"])
    assert full_report["aggregate_metrics"]["fcfs_static"] != full_report["aggregate_metrics"]["edd_static"]
