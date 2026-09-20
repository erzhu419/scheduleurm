import json

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_budget_pair_confirmation as confirmation,
)


def test_registration_materializes_fresh_roster_without_selector_calls(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("registration invoked a selector")

    monkeypatch.setattr(
        confirmation, "select_canonical_count_space_bnb_frame", forbidden
    )
    protocol = confirmation.build_registration()
    roster = protocol["queue_roster_contract"]

    assert protocol["independent_confirmation"] is True
    assert protocol["selector_contract"]["fixed_budget_by_lane"] == {
        "8": 48,
        "16": 9,
    }
    assert protocol["registration_scope"]["selector_invoked"] is False
    assert protocol["registration_scope"]["trajectory_built"] is False
    assert roster["seed"] == 260919
    assert roster["candidate_draws"] == 550
    assert roster["rng_integer_draws"] == 1650
    assert len(roster["queues"]) == len({tuple(q) for q in roster["queues"]}) == 512
    assert roster["old_1000_grid_intersection_count"] == 0
    assert set(roster["prior_roster_intersection_count"].values()) == {0}


def test_roster_generation_is_deterministic_and_disjoint():
    protocol = confirmation.build_registration()
    exclusions = confirmation._load_exclusion_rosters(protocol["source_paths"])

    first = confirmation.generate_roster(exclusions)
    second = confirmation.generate_roster(exclusions)

    assert first == second == protocol["queue_roster_contract"]
    assert first["prior_roster_union_size"] == 1536


def test_prior_roster_rejects_wrong_cardinality():
    payload = confirmation._read_object(confirmation.DEFAULT_CANONICAL_ROSTER)
    historical = confirmation._read_object(
        confirmation.DEFAULT_CANONICAL_PROTOCOL
    )["queue_roster_contract"]["base_queues"]
    payload["queues"] = payload["queues"][:-1]
    payload["requested_queue_count"] = 511

    with pytest.raises(
        confirmation.BudgetPairConfirmationRefusal,
        match="prior roster is invalid",
    ):
        confirmation._validated_prior_roster(
            payload,
            experiment=confirmation.canonical.EXPERIMENT,
            seed=confirmation.canonical.ROSTER_SEED,
            expected_queues=historical,
        )


def test_prior_roster_rejects_valid_but_nonhistorical_queue():
    payload = confirmation._read_object(confirmation.DEFAULT_CANONICAL_ROSTER)
    historical = confirmation._read_object(
        confirmation.DEFAULT_CANONICAL_PROTOCOL
    )["queue_roster_contract"]["base_queues"]
    replacement = [0, 0, 6]
    assert replacement not in payload["queues"]
    payload["queues"][0] = replacement

    with pytest.raises(
        confirmation.BudgetPairConfirmationRefusal,
        match="prior roster is invalid",
    ):
        confirmation._validated_prior_roster(
            payload,
            experiment=confirmation.canonical.EXPERIMENT,
            seed=confirmation.canonical.ROSTER_SEED,
            expected_queues=historical,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "selector_contract",
            {
                **confirmation.SELECTOR_CONTRACT,
                "fixed_budget_by_lane": {"8": 47, "16": 9},
            },
        ),
        (
            "registration_scope",
            {
                **confirmation.REGISTRATION_SCOPE,
                "roster_rng_integer_draws": 1650,
                "paper_edited": True,
            },
        ),
    ],
)
def test_protocol_refuses_scientific_contract_drift(field, replacement):
    protocol = confirmation.build_registration()
    protocol[field] = replacement

    with pytest.raises(
        confirmation.BudgetPairConfirmationRefusal,
        match="registered confirmation protocol changed",
    ):
        confirmation._validate_protocol(protocol)


def test_decision_branches_are_ordered_and_frozen():
    passing = {
        "support_gate_passed": True,
        "independent_roster_action_identity_gate_passed": True,
        "actual_compute_gate_passed": True,
    }
    assert confirmation._decision(False, [passing], True) == confirmation.INVALID_DECISION
    assert confirmation._decision(
        True, [{**passing, "support_gate_passed": False}], True
    ) == confirmation.SUPPORT_FAIL_DECISION
    assert confirmation._decision(
        True,
        [{**passing, "independent_roster_action_identity_gate_passed": False}],
        True,
    ) == confirmation.ACTION_FAIL_DECISION
    assert confirmation._decision(True, [passing], False) == confirmation.COMPUTE_FAIL_DECISION
    assert confirmation._decision(True, [passing], True) == confirmation.PASS_DECISION


def test_mocked_pass_authorizes_only_separate_envelope_trajectory(
    tmp_path, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **confirmation.build_registration(),
    }
    registration = tmp_path / "protocol.json"
    roster = tmp_path / "roster.json"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "result.json"
    registration.write_text(json.dumps(protocol), encoding="utf-8")
    roster.write_text(
        json.dumps(protocol["queue_roster_contract"]), encoding="utf-8"
    )
    fixed.write_text(
        json.dumps(confirmation._current_fixed_sources()), encoding="utf-8"
    )

    def fake_screen(_model, queues, lanes, capacity_rows):
        assert len(queues) == 512
        assert len(capacity_rows) == 3
        candidate = {field: 1 for field in confirmation.canonical.WORK_FIELDS}
        reference = {field: 2 for field in confirmation.canonical.WORK_FIELDS}
        return (
            {
                "lanes": lanes,
                "state_count": 512,
                "candidate_actual_work": candidate,
                "cold_canonical_reference_actual_work": reference,
                "support_gate_passed": True,
                "independent_roster_action_identity_gate_passed": True,
                "actual_compute_gate_passed": True,
            },
            {"synthetic_execution_check": True},
        )

    monkeypatch.setattr(confirmation, "_screen_lane", fake_screen)
    report = confirmation.run_registered(registration, roster, fixed, output)

    assert report["status"] == "PASS"
    assert report["scientific_decision"] == confirmation.PASS_DECISION
    assert report["independent_static_confirmation_passed"] is True
    assert report["trajectory_registration_prerequisite_passed"] is True
    assert report["capacity_closure_reclassified"] is False
    assert report["retained_v5_capacity_established"] is False
    assert report["trajectory_runs"] == 0
