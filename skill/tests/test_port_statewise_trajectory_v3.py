from __future__ import annotations

from dataclasses import replace

import pytest

from algorithm.experiments.port_statewise_trajectory_v3 import (
    GeneralizedTransitionCost,
    PersistentTransition,
    PortDecisionStateV3,
    PortJobState,
    PortStatewiseV3Error,
    RECONFIGURATION_SAFETY_POLICY,
    ROBUST_POLICY,
    SPT_SAFETY_POLICY,
    TrajectoryCandidateV3,
    _demo_state_and_candidates,
    build_port_statewise_trajectory_v3,
    build_source_core_bacasp_certificate,
    build_synthetic_migration_certificate,
    select_statewise_trajectory_v3,
)


def _source_report() -> dict:
    return {
        "pass": True,
        "artifact_hash": "a" * 64,
        "source_authenticity_ready": True,
        "resource_feasibility_ready": True,
        "source_time_invariant_assignment_ready": True,
        "mid_service_reconfiguration_admitted": False,
        "source_core_metrics": {
            "robust_maxweight": {
                "quay_makespan": 100.0,
                "mean_quay_flow_time": 50.0,
                "source_objective": 25.0,
            },
            "spt_static": {
                "quay_makespan": 101.0,
                "mean_quay_flow_time": 48.0,
                "source_objective": 26.0,
            },
        },
        "source_core_pareto": {
            "nondominated_policies": ["robust_maxweight", "spt_static"]
        },
    }


def test_generalized_cost_records_delay_lost_work_and_risk_exactly():
    cost = GeneralizedTransitionCost(
        fixed_cost=2.0,
        operation_delay=4.0,
        delay_unit_cost=0.5,
        lost_work=3.0,
        lost_work_unit_cost=2.0,
        risk_probability=0.1,
        risk_impact=20.0,
        resource_cost=1.0,
    )

    assert cost.delay_penalty == pytest.approx(2.0)
    assert cost.lost_work_penalty == pytest.approx(6.0)
    assert cost.risk_penalty == pytest.approx(2.0)
    assert cost.total_penalty == pytest.approx(13.0)
    assert cost.snapshot()["total_penalty"] == pytest.approx(13.0)
    with pytest.raises(ValueError, match="must not exceed one"):
        replace(cost, risk_probability=1.01).validate()


def test_persistent_tug_and_channel_conflict_fails_closed_until_completion():
    state, candidates = _demo_state_and_candidates()
    selection = select_statewise_trajectory_v3(state, candidates)
    rejected = selection.snapshot()["rejected_candidates"]

    assert "conflicting_immediate_reberth" in rejected
    assert any("persistent_resource_busy" in reason for reason in rejected["conflicting_immediate_reberth"])
    assert "robust_delayed_reberth" in selection.snapshot()["feasible_candidate_ids"]


def test_transfer_team_is_persistent_across_overlapping_crane_transitions():
    state, candidates = _demo_state_and_candidates()
    first = PersistentTransition(
        transition_id="crane_a",
        kind="crane_reassignment",
        vessel_id="v_short",
        start_time=0.0,
        complete_at=3.0,
        resource_tokens=("transfer:team2",),
        cost=GeneralizedTransitionCost(operation_delay=3.0),
    )
    second = replace(
        first,
        transition_id="crane_b",
        vessel_id="v_long",
        start_time=2.0,
        complete_at=4.0,
    )
    invalid = TrajectoryCandidateV3(
        candidate_id="overlapping_transfer",
        policy_family=ROBUST_POLICY,
        duration=6.0,
        cumulative_lower_service=(("v_short", 1.0),),
        terminal_costs=(("makespan", 90.0), ("mean_flow_time", 45.0), ("weighted_tardiness", 30.0)),
        transitions=(first, second),
    )

    selection = select_statewise_trajectory_v3(state, (*candidates, invalid))
    reasons = selection.snapshot()["rejected_candidates"]["overlapping_transfer"]
    assert any("overlapping transitions share persistent resources" in reason for reason in reasons)


def test_transition_kind_requires_explicit_resource_classes():
    bad = PersistentTransition(
        transition_id="bad_reberth",
        kind="reberth",
        vessel_id="v_short",
        start_time=0.0,
        complete_at=1.0,
        resource_tokens=("tug:t2",),
    )
    with pytest.raises(ValueError, match="lacks persistent resources"):
        bad.validate()


def test_duration_normalization_changes_the_oracle_ordering():
    state, candidates = _demo_state_and_candidates()
    spt, reconfiguration, _, conflict = candidates
    fast = TrajectoryCandidateV3(
        candidate_id="fast_normalized",
        policy_family=ROBUST_POLICY,
        duration=2.0,
        cumulative_lower_service=(("v_short", 5.0),),
        terminal_costs=(("makespan", 70.0), ("mean_flow_time", 30.0), ("weighted_tardiness", 18.0)),
    )
    slow = TrajectoryCandidateV3(
        candidate_id="slow_raw_numerator",
        policy_family=ROBUST_POLICY,
        duration=20.0,
        cumulative_lower_service=(("v_short", 15.0),),
        terminal_costs=(("makespan", 69.0), ("mean_flow_time", 31.0), ("weighted_tardiness", 18.0)),
    )
    selection = select_statewise_trajectory_v3(
        state, (spt, reconfiguration, fast, slow, conflict)
    )

    assert slow.duration_normalized_score(state) < fast.duration_normalized_score(state)
    assert slow.snapshot(state)["unnormalized_robust_numerator"] > fast.snapshot(state)["unnormalized_robust_numerator"]
    assert selection.selected.candidate_id == "fast_normalized"


def test_terminal_cost_pareto_does_not_override_the_robust_oracle():
    state, candidates = _demo_state_and_candidates()
    spt, reconfiguration, robust, conflict = candidates
    dominated = TrajectoryCandidateV3(
        candidate_id="high_score_but_dominated",
        policy_family=ROBUST_POLICY,
        duration=1.0,
        cumulative_lower_service=(("v_short", 20.0),),
        terminal_costs=(("makespan", 81.0), ("mean_flow_time", 36.0), ("weighted_tardiness", 23.0)),
    )
    selection = select_statewise_trajectory_v3(
        state, (spt, reconfiguration, robust, conflict, dominated)
    )

    assert selection.selected.candidate_id == dominated.candidate_id
    assert dominated.candidate_id in selection.score_semantic_frontier_ids
    assert dominated.candidate_id not in selection.terminal_pareto_frontier_ids
    snapshot = selection.snapshot()
    assert snapshot["score_semantic_pruning_exact"] is True
    assert snapshot["selected_terminal_pareto"] is False
    assert snapshot["finite_family_penalty_bound_P0"] >= 0.0


def test_both_safety_families_are_mandatory_and_candidate_bound_is_finite():
    state, candidates = _demo_state_and_candidates()
    without_spt = tuple(row for row in candidates if row.policy_family != SPT_SAFETY_POLICY)

    with pytest.raises(PortStatewiseV3Error, match="lacks feasible safety candidates"):
        select_statewise_trajectory_v3(state, without_spt)
    with pytest.raises(PortStatewiseV3Error, match="finite, nonempty, and bounded"):
        select_statewise_trajectory_v3(state, candidates, max_candidates=2)


def test_unknown_resource_and_excess_service_are_rejected_not_silently_admitted():
    state, candidates = _demo_state_and_candidates()
    bad_resource = replace(
        candidates[2],
        candidate_id="unknown_tug",
        transitions=(
            replace(
                candidates[2].transitions[0],
                transition_id="unknown_tug_transition",
                resource_tokens=("tug:unregistered", "channel:main"),
            ),
        ),
    )
    excess = replace(
        candidates[2],
        candidate_id="excess_service",
        cumulative_lower_service=(("v_long", 61.0),),
        transitions=(),
    )
    selection = select_statewise_trajectory_v3(
        state, (*candidates, bad_resource, excess)
    )
    rejected = selection.snapshot()["rejected_candidates"]

    assert any("unknown_resources" in reason for reason in rejected["unknown_tug"])
    assert rejected["excess_service"] == ["service_exceeds_remaining_work:v_long"]


def test_source_core_and_synthetic_migration_certificates_do_not_substitute():
    source = build_source_core_bacasp_certificate(_source_report())
    state, candidates = _demo_state_and_candidates()
    synthetic = build_synthetic_migration_certificate(
        select_statewise_trajectory_v3(state, candidates)
    )

    assert source["certificate_type"] == "bacasp_source_core_only"
    assert source["mid_service_migration_evidence"] is False
    assert source["synthetic_migration_evidence_used"] is False
    assert synthetic["certificate_type"] == "synthetic_statewise_migration_only"
    assert synthetic["bacasp_source_core_evidence_used"] is False
    assert synthetic["physical_terminal_observations_used"] is False
    assert synthetic["persistent_resource_kinds"] == ["channel", "transfer", "tug"]


def test_source_core_rejects_non_sha256_artifact_identity():
    report = _source_report()
    report["artifact_hash"] = "not-a-source-hash"

    with pytest.raises(PortStatewiseV3Error, match="not SHA-256"):
        build_source_core_bacasp_certificate(report)


def test_public_certificate_fails_closed_if_midservice_migration_is_implied():
    report = _source_report()
    report["mid_service_reconfiguration_admitted"] = True

    with pytest.raises(PortStatewiseV3Error, match="failed closed"):
        build_source_core_bacasp_certificate(report)


def test_full_v3_report_is_deterministic_and_keeps_industrial_boundary():
    first = build_port_statewise_trajectory_v3(_source_report())
    second = build_port_statewise_trajectory_v3(_source_report())

    assert first == second
    assert first["pass"] is True
    assert first["status"] == "PORT_STATEWISE_TRAJECTORY_V3_PASS"
    assert first["finite_statewise_candidate_family_ready"] is True
    assert first["duration_normalized_oracle_ready"] is True
    assert first["score_semantic_pruning_ready"] is True
    assert first["terminal_pareto_audit_ready"] is True
    assert first["persistent_tug_channel_transfer_constraints_ready"] is True
    assert first["certificate_non_substitution_ready"] is True
    assert first["industrial_deployment_claim_ready"] is False
    assert first["global_port_optimality_claim_ready"] is False
    assert "deployment in an operating industrial terminal" in first["claim_boundary"]["does_not_support"]
