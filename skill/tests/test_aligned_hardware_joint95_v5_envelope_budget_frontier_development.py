import copy

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_budget_frontier_development as frontier,
)


def test_registration_freezes_post_scout_grid_without_authorizing_trajectories():
    protocol = frontier.build_registration()

    assert protocol["post_scout_development"] is True
    assert protocol["independent_confirmation"] is False
    assert protocol["lane_budget_grid"] == {
        "8": [36, 43, 44, 48],
        "16": [9, 18, 36, 67, 68, 72],
    }
    assert protocol["exact_endpoint_budget"] == 4096
    assert protocol["source_contract"]["scientific_decision"] == (
        frontier.SOURCE_DECISION
    )
    assert protocol["registration_scope"]["selector_invoked_during_registration"] is False
    assert protocol["registration_scope"]["trajectory_built"] is False
    assert "not independent confirmation" in protocol["claim_boundary"]


def test_protocol_refuses_budget_grid_drift():
    protocol = frontier.build_registration()
    changed = copy.deepcopy(protocol)
    changed["lane_budget_grid"]["8"][-1] = 47

    with pytest.raises(
        frontier.BudgetFrontierRefusal,
        match="registered budget-frontier protocol changed",
    ):
        frontier._validate_protocol(changed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("trace_rule", "independent reruns"),
        ("expected_work", {"cumulative_traces": 1}),
        (
            "budget_selection_rule",
            {
                **frontier.BUDGET_SELECTION_RULE,
                "score_tolerance": 1.0,
            },
        ),
        (
            "registration_scope",
            {
                **frontier.REGISTRATION_SCOPE,
                "paper_edited": True,
            },
        ),
    ],
)
def test_protocol_refuses_scientific_contract_drift(field, replacement):
    protocol = frontier.build_registration()
    protocol[field] = replacement

    with pytest.raises(
        frontier.BudgetFrontierRefusal,
        match="registered budget-frontier protocol changed",
    ):
        frontier._validate_protocol(protocol)


def test_decision_requires_support_compute_then_development_roster_action_identity():
    assert frontier._decision(False, []) == frontier.INVALID_DECISION
    assert frontier._decision(
        True,
        [
            {
                "minimum_support_compute_budget": None,
                "minimum_development_roster_action_identity_budget": None,
            }
        ],
    ) == frontier.NO_PAIR_DECISION
    assert frontier._decision(
        True,
        [
            {
                "minimum_support_compute_budget": 44,
                "minimum_development_roster_action_identity_budget": None,
            }
        ],
    ) == frontier.SUPPORT_ONLY_DECISION
    assert frontier._decision(
        True,
        [
            {
                "minimum_support_compute_budget": 44,
                "minimum_development_roster_action_identity_budget": 48,
            },
            {
                "minimum_support_compute_budget": 9,
                "minimum_development_roster_action_identity_budget": 9,
            },
        ],
    ) == frontier.PASS_DECISION


def test_work_gate_is_componentwise_strict():
    candidate = {
        "lp_calls": 9,
        "lp_matrix_coefficients": 90,
        "candidate_evaluations": 10,
    }
    reference = {
        "lp_calls": 10,
        "lp_matrix_coefficients": 100,
        "candidate_evaluations": 10,
    }

    comparison = frontier._work_comparison(candidate, reference)

    assert comparison["lp_calls"]["strictly_lower"] is True
    assert comparison["lp_matrix_coefficients"]["strictly_lower"] is True
    assert comparison["candidate_evaluations"]["strictly_lower"] is False
