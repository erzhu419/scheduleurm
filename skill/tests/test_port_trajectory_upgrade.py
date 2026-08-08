from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments.port_public_benchmark_adapter import DEFAULT_FIXTURE, DEFAULT_MANIFEST
from algorithm.experiments.port_scheduling_benchmark import (
    ALL_POLICIES,
    RECONFIGURATION_ACTIONS,
    THEOREM_POLICY,
)
from algorithm.experiments.port_scheduling_instances import load_port_instance
from algorithm.experiments.port_trajectory_upgrade import (
    GLOBAL_TRAJECTORY_CONFIG,
    TRAJECTORY_POLICY,
    TrajectoryPlan,
    TrajectorySearchConfig,
    _SyntheticTrajectorySimulator,
    compact_certificate,
    compact_markdown,
    _trajectory_frame_spec,
    _without_runtime,
    build_port_trajectory_upgrade,
    initial_trajectory_family,
    local_improvement_neighbors,
    trajectory_objective,
)


SMALL_INSTANCE = Path(__file__).resolve().parent / "data" / "port_small_instance.json"


@pytest.fixture(scope="module")
def report():
    return build_port_trajectory_upgrade()


def test_global_configuration_and_initial_family_are_locked_and_deterministic():
    GLOBAL_TRAJECTORY_CONFIG.validate()
    first = initial_trajectory_family()
    second = initial_trajectory_family()

    assert [row.plan_id for row in first] == [row.plan_id for row in second]
    assert len(first) == len(ALL_POLICIES) + len(GLOBAL_TRAJECTORY_CONFIG.rolling_seeds)
    assert {row.policies for row in first if len(row.policies) == 1} == {
        (policy,) for policy in ALL_POLICIES
    }
    assert len(GLOBAL_TRAJECTORY_CONFIG.digest) == 64


def test_configuration_and_plan_validation_fail_closed():
    with pytest.raises(ValueError, match="registered atomic policies"):
        TrajectoryPlan(("unregistered",))
    with pytest.raises(ValueError, match="locked period"):
        TrajectorySearchConfig(rolling_seeds=((THEOREM_POLICY, "spt_static"),)).validate()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        TrajectorySearchConfig(risk_weight=float("nan")).validate()


def test_local_improvement_is_finite_unique_and_fixed_period():
    seed = TrajectoryPlan((THEOREM_POLICY, "spt_static", "edd_static"))
    rows = local_improvement_neighbors(seed)

    assert rows
    assert len(rows) == len(set(row.plan_id for row in rows))
    assert all(len(row.policies) == GLOBAL_TRAJECTORY_CONFIG.period for row in rows)
    assert all(set(row.policies) <= set(ALL_POLICIES) for row in rows)


def test_trajectory_objective_uses_lower_service_and_bounded_penalties():
    instance = load_port_instance(SMALL_INSTANCE)
    plan = TrajectoryPlan((THEOREM_POLICY, "spt_static", "reconfiguration_greedy"))
    first = _SyntheticTrajectorySimulator(instance, plan).run()
    second = _SyntheticTrajectorySimulator(instance, plan).run()
    frame = _trajectory_frame_spec(instance, None)
    objective = trajectory_objective(first, frame)

    assert first["result_hash"] == second["result_hash"]
    assert 0.0 <= objective["terminal_q_dot_lower_service"] <= 1.0
    assert objective["total_bounded_penalty"] >= 0.0
    assert objective["total_bounded_penalty"] <= objective["uniform_penalty_bound"]
    assert objective["risk_exposure_bounded"] <= objective["risk_exposure_bound"]
    assert objective["robust_terminal_objective"] == pytest.approx(
        objective["terminal_q_dot_lower_service"]
        - objective["total_bounded_penalty"],
        abs=2e-8,
    )
    assert objective["selected_action_count"] > 0


def test_full_gate_selects_one_global_plan_by_exact_trajectory_oracle(report):
    assert report["pass"] is True
    assert report["status"] == "PORT_TRAJECTORY_UPGRADE_PASS"
    assert report["selection_scope"] == (
        "one_plan_over_public_and_all_registered_synthetic_instances"
    )
    assert report["selection_uses_registered_terminal_service_coordinates"] is True
    assert report["selection_uses_external_optima_or_bks"] is False
    assert report["per_instance_cherry_pick"] is False
    assert report["registered_instance_pareto_ready"] is True
    assert report["trajectory_oracle_gap_alpha0"] == pytest.approx(0.0)
    assert report["trajectory_oracle_gap_alpha1"] == 0.0
    assert report["trajectory_family_size"] >= report["full_rollout_seed_count"]
    assert report["rolling_seed_count"] > 0
    assert report["local_improvement_candidate_count"] > 0
    assert report["evaluated_trajectory_instance_pairs"] == (
        report["trajectory_family_size"] * report["evaluated_instance_count"]
    )
    selected_id = report["selected_global_plan_id"]
    assert report["public_bacasp_s"]["selected_global_plan_id"] == selected_id
    assert report["synthetic_four_resource"]["selected_global_plan_id"] == selected_id
    assert all(
        row["selected_global_plan_id"] == selected_id
        for row in report["synthetic_four_resource"]["instances"]
    )
    gaps = [
        float(row["trajectory_oracle_gap"])
        for row in report["trajectory_family"].values()
    ]
    assert min(gaps) == pytest.approx(0.0)
    assert all(gap >= -1e-8 for gap in gaps)


def test_public_source_core_boundary_and_baseline_comparison_are_preserved(report):
    public = report["public_bacasp_s"]

    assert public["source_authenticity"]["ready"] is True
    assert public["source_core_feasibility_ready"] is True
    assert public["time_invariant_assignment_ready"] is True
    assert public["mid_service_migration_admitted"] is False
    assert public["yard_gate_metrics_are_source_core"] is False
    assert set(public["baseline_results"]) == set(ALL_POLICIES) - {THEOREM_POLICY}
    assert set(public["source_core_metrics"]) == {
        TRAJECTORY_POLICY,
        *[policy for policy in ALL_POLICIES if policy != THEOREM_POLICY],
    }
    assert all(
        int(public["selected_result"]["action_counts"].get(action_type, 0)) == 0
        for action_type in RECONFIGURATION_ACTIONS
    )
    assert public["ours_source_core_pareto_nondominated"] is True
    assert TRAJECTORY_POLICY in public["source_core_pareto"]["nondominated_policies"]


def test_synthetic_four_resource_migration_semantics_remain_separate(report):
    synthetic = report["synthetic_four_resource"]

    assert synthetic["resource_semantics"] == [
        "berth",
        "quay_crane",
        "yard_block",
        "gate_lane",
    ]
    assert synthetic["all_three_migration_analogues_in_candidate_family"] is True
    assert set(synthetic["migration_semantics_preserved"]) == RECONFIGURATION_ACTIONS
    assert all(row["selected_feasible"] for row in synthetic["instances"])
    for row in synthetic["instances"]:
        assert set(row["baseline_results"]) == set(ALL_POLICIES) - {THEOREM_POLICY}
        assert TRAJECTORY_POLICY in row["comparison_metrics"]
        assert TRAJECTORY_POLICY in row["pareto"]["nondominated_policies"]
    assert synthetic["ours_pareto_nondominated_on_every_instance"] is True


def test_penalty_and_slack_mapping_do_not_overclaim_stability(report):
    certificate = report["bounded_penalty_certificate"]
    theorem = report["theorem_mapping"]

    assert certificate["ready"] is True
    assert certificate["queue_independent"] is True
    assert certificate["queue_scaled_penalty_slope_beta"] == 0.0
    assert certificate["approximate_oracle_slope_alpha1"] == 0.0
    assert certificate["finite_family_P0_bound"] >= max(
        certificate["selected_instance_penalties"]
    )
    assert certificate["finite_family_P0_bound"] <= certificate["pre_registered_uniform_P0_bound"]
    assert "additive drift constant P0" in theorem["penalty_role_in_drift"]
    assert "separately requires positive eta" in theorem["slack_condition"]
    assert "variable-duration frame drift theorem" in theorem["frame_boundary"]
    assert report["performance_superiority_claim_ready"] is False
    assert "online nonanticipative stochastic optimality" in report["claim_boundary"]["does_not_support"]


def test_report_hash_excludes_observed_runtime_but_not_scientific_fields(report):
    stripped = _without_runtime(report)
    changed_runtime = json.loads(json.dumps(report))
    changed_runtime["runtime_seconds_observed"] += 123.0
    assert _without_runtime(changed_runtime) == stripped

    changed_science = json.loads(json.dumps(report))
    changed_science["selected_global_plan_id"] = "tampered"
    assert _without_runtime(changed_science) != stripped


def test_compact_certificate_binds_full_ledger_without_event_payload(report):
    compact = compact_certificate(report)
    payload = json.dumps(compact, sort_keys=True).encode("ascii")

    assert compact["pass"] is True
    assert compact["full_artifact_hash"] == report["artifact_hash"]
    assert compact["selected_global_plan_id"] == report["selected_global_plan_id"]
    assert "trajectory_family" not in compact
    assert "selected_result" not in compact["public_bacasp_s"]
    assert all("selected_result" not in row for row in compact["synthetic_four_resource"]["instances"])
    assert len(payload) < 100_000
    markdown = compact_markdown(compact)
    assert report["artifact_hash"] in markdown
    assert report["selected_global_plan_id"] in markdown
    assert "Public BACASP-S source-core comparison" in markdown
    assert "Synthetic four-resource comparison" in markdown


def test_tampered_public_fixture_is_rejected_before_search(tmp_path):
    changed = tmp_path / "changed.dat"
    changed.write_bytes(DEFAULT_FIXTURE.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="authenticity gate"):
        build_port_trajectory_upgrade(changed, DEFAULT_MANIFEST)
