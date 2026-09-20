import copy
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_r16_bulk_tie_confirmation as confirmation,
)


@pytest.fixture(scope="module")
def frozen_protocol():
    return confirmation.build_registration()


@pytest.fixture(scope="module")
def v5_model(frozen_protocol):
    snapshot = frozen_protocol["development_contract"][
        "physical_complete_frame_envelope"
    ]["model"]
    return (
        confirmation.previous.development.source.capacity.base
        ._model_from_snapshot(snapshot)
    )


def test_registration_materializes_fresh_roster_without_selector_calls(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("registration invoked a selector")

    for owner, name in (
        (confirmation, "select_canonical_count_space_bnb_frame"),
        (confirmation, "select_canonical_count_space_bnb_single_exchange_frame"),
        (confirmation, "select_canonical_score_tie_bulk_substitution_frame"),
        (confirmation.development, "select_canonical_count_space_bnb_frame"),
        (
            confirmation.development,
            "select_canonical_score_tie_bulk_substitution_frame",
        ),
        (
            confirmation.single_confirmation,
            "select_canonical_count_space_bnb_frame",
        ),
        (
            confirmation.single_confirmation,
            "select_canonical_count_space_bnb_single_exchange_frame",
        ),
    ):
        monkeypatch.setattr(owner, name, forbidden)

    protocol = confirmation.build_registration()
    roster = protocol["queue_roster_contract"]

    assert protocol["independent_confirmation"] is True
    assert protocol["selector_contract"]["fixed_budget_by_lane"] == {
        "8": 48,
        "16": 9,
    }
    assert protocol["registration_scope"]["selector_invoked"] is False
    assert protocol["registration_scope"]["reference_invoked"] is False
    assert protocol["registration_scope"]["trajectory_built"] is False
    assert roster["seed"] == confirmation.ROSTER_SEED == 280919
    assert roster["candidate_draws"] == confirmation.ROSTER_CANDIDATE_DRAWS == 567
    assert roster["rng_integer_draws"] == confirmation.ROSTER_RNG_INTEGER_DRAWS
    assert len(roster["queues"]) == len({tuple(q) for q in roster["queues"]}) == 512
    assert roster["old_1000_grid_intersection_count"] == 0
    assert set(roster["prior_roster_intersection_count"].values()) == {0}


def test_roster_is_deterministic_and_excludes_five_roster_union_2560(
    frozen_protocol,
):
    exclusions = confirmation._load_exclusion_rosters(
        frozen_protocol["source_paths"]
    )
    first = confirmation.generate_roster(exclusions)
    second = confirmation.generate_roster(exclusions)
    exclusion_sets = {
        name: {tuple(row) for row in rows} for name, rows in exclusions.items()
    }
    prior_union = set().union(*exclusion_sets.values())
    new_roster = {tuple(row) for row in first["queues"]}

    assert set(exclusion_sets) == {
        "count_space_confirmation",
        "joint95_static",
        "canonical_development",
        "budget_pair_confirmation",
        "single_exchange_confirmation",
    }
    assert all(len(rows) == 512 for rows in exclusion_sets.values())
    assert len(prior_union) == 2560
    assert first == second == frozen_protocol["queue_roster_contract"]
    assert first["prior_roster_union_size"] == 2560
    assert first["all_five_prior_512_rosters_excluded"] is True
    assert new_roster.isdisjoint(prior_union)
    assert new_roster.isdisjoint(confirmation._old_grid())


def test_fifth_prior_roster_tamper_is_refused(tmp_path, frozen_protocol):
    paths = copy.deepcopy(frozen_protocol["source_paths"])
    payload = confirmation._read_object(
        confirmation.DEFAULT_SINGLE_CONFIRMATION_ROSTER
    )
    replacement = list(frozen_protocol["queue_roster_contract"]["queues"][0])
    assert replacement not in payload["queues"]
    payload["queues"][0] = replacement
    changed = tmp_path / "changed_seed270919_roster.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    paths["single_confirmation_roster"] = str(changed)

    with pytest.raises(confirmation.R16BulkTieConfirmationRefusal):
        confirmation._load_exclusion_rosters(paths)


def test_registration_rejects_changed_development_fixed_sources(tmp_path):
    fixed = confirmation._read_object(
        confirmation.DEFAULT_DEVELOPMENT_FIXED_SOURCES
    )
    key = next(iter(fixed))
    fixed[key] += "\n# changed"
    changed = tmp_path / "changed_development_fixed_sources.json"
    changed.write_text(json.dumps(fixed), encoding="utf-8")

    with pytest.raises(
        confirmation.R16BulkTieConfirmationRefusal,
        match="frozen development sources changed",
    ):
        confirmation.build_registration(development_fixed_sources=changed)


def test_development_result_rejects_mutated_authoritative_source_contract():
    protocol = confirmation._read_object(
        confirmation.DEFAULT_DEVELOPMENT_REGISTRATION
    )
    result = confirmation._read_object(confirmation.DEFAULT_DEVELOPMENT_RESULT)
    protocol["source_contract"]["physical_complete_frame_envelope"]["model"][
        "options"
    ][0]["duration"] += 1.0
    result["protocol"] = copy.deepcopy(protocol)

    with pytest.raises(
        confirmation.R16BulkTieConfirmationRefusal,
        match="frozen R16 bulk-tie development result differs",
    ):
        confirmation._validated_development_contract(result, protocol)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "selector_contract",
            {
                **confirmation.SELECTOR_CONTRACT,
                "fixed_budget_by_lane": {"8": 48, "16": 10},
            },
        ),
        (
            "expected_work",
            {
                **confirmation.EXPECTED_WORK,
                "r16_composite_candidate_calls": 511,
            },
        ),
    ],
)
def test_protocol_rejects_method_or_work_contract_drift(
    frozen_protocol, field, replacement
):
    protocol = copy.deepcopy(frozen_protocol)
    protocol[field] = replacement

    with pytest.raises(
        confirmation.R16BulkTieConfirmationRefusal,
        match="registered R16 bulk-tie confirmation protocol changed",
    ):
        confirmation._validate_protocol(protocol)


def test_r8_screen_rejects_undercharged_exchange_attempts(
    frozen_protocol, v5_model, monkeypatch
):
    capacity_rows = frozen_protocol["development_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["8"]
    real_total_work_row = confirmation.single_exchange_total_work_row

    def undercharged(result):
        row = real_total_work_row(result)
        row["candidate_evaluations"] -= result.exchange_attempts
        row["residual_fill_option_probes"] -= result.exchange_attempts
        return row

    monkeypatch.setattr(confirmation, "ROSTER_SIZE", 1)
    monkeypatch.setattr(confirmation, "single_exchange_total_work_row", undercharged)
    row, checks = confirmation._screen_lane(
        v5_model, [(3, 1, 11)], confirmation.R8_LANES, capacity_rows
    )

    assert row["exchange_attempt_count"] > 0
    assert checks[
        "candidate_work_is_exactly_base_plus_registered_polish_charge"
    ] is False
    assert row["lane_confirmation_gate_passed"] is False


def test_r16_screen_rejects_undercharged_structural_pair_checks(
    frozen_protocol, v5_model, monkeypatch
):
    capacity_rows = frozen_protocol["development_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    real_total_work_row = confirmation.bulk_tie_total_work_row

    def undercharged(result):
        row = real_total_work_row(result)
        row["candidate_evaluations"] -= result.pair_checks
        row["residual_fill_option_probes"] -= result.pair_checks
        return row

    monkeypatch.setattr(confirmation, "ROSTER_SIZE", 1)
    monkeypatch.setattr(confirmation, "bulk_tie_total_work_row", undercharged)
    row, checks = confirmation._screen_lane(
        v5_model, [(11, 11, 12)], confirmation.R16_LANES, capacity_rows
    )

    assert row["polish_trigger_state_count"] == 1
    assert row["structural_pair_check_count"] == 7
    assert checks[
        "candidate_work_is_exactly_base_plus_registered_polish_charge"
    ] is False
    assert row["lane_confirmation_gate_passed"] is False


def test_decision_branches_are_ordered_and_frozen():
    passing = {
        "support_gate_passed": True,
        "independent_roster_action_identity_gate_passed": True,
        "actual_compute_gate_passed": True,
    }

    assert (
        confirmation._decision(False, [passing], True)
        == confirmation.INVALID_DECISION
    )
    assert confirmation._decision(
        True, [{**passing, "support_gate_passed": False}], True
    ) == confirmation.SUPPORT_FAIL_DECISION
    assert confirmation._decision(
        True,
        [{**passing, "independent_roster_action_identity_gate_passed": False}],
        True,
    ) == confirmation.ACTION_FAIL_DECISION
    assert (
        confirmation._decision(True, [passing], False)
        == confirmation.COMPUTE_FAIL_DECISION
    )
    assert confirmation._decision(True, [passing], True) == confirmation.PASS_DECISION


def test_mocked_pass_only_authorizes_trajectory_registration(
    tmp_path, frozen_protocol, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(frozen_protocol),
    }
    registration = tmp_path / "protocol.json"
    roster = tmp_path / "roster.json"
    fixed = tmp_path / "fixed_sources.json"
    output = tmp_path / "result.json"
    registration.write_text(json.dumps(protocol), encoding="utf-8")
    roster.write_text(
        json.dumps(protocol["queue_roster_contract"]), encoding="utf-8"
    )
    fixed.write_text(
        json.dumps(confirmation._current_fixed_sources()), encoding="utf-8"
    )

    def fake_screen(_model, queues, lanes, capacity_rows):
        assert len(queues) == confirmation.ROSTER_SIZE
        assert len(capacity_rows) == 3
        candidate = {field: 1 for field in confirmation.canonical.WORK_FIELDS}
        reference = {field: 2 for field in confirmation.canonical.WORK_FIELDS}
        return (
            {
                "lanes": lanes,
                "state_count": confirmation.ROSTER_SIZE,
                "candidate_charged_work": candidate,
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
    assert report["envelope_normalized_trajectory_registration_authorized"] is True
    assert report["trajectory_registration_prerequisite_passed"] is True
    assert report["capacity_closure_reclassified"] is False
    assert report["retained_v5_capacity_established"] is False
    assert report["retained_v5_closure_passed"] is False
    assert report["trajectory_runs"] == 0
    assert report["arrival_generations"] == 0
