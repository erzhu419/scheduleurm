import copy
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_r16_bulk_tie_development as development,
)


@pytest.fixture(scope="module")
def frozen_protocol():
    return development.build_registration()


@pytest.fixture(scope="module")
def v5_model(frozen_protocol):
    snapshot = frozen_protocol["source_contract"][
        "physical_complete_frame_envelope"
    ]["model"]
    return (
        development.source.previous.development.source.capacity.base
        ._model_from_snapshot(snapshot)
    )


def test_registration_binds_three_consumed_rosters_without_selector_calls(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("registration invoked a selector")

    monkeypatch.setattr(
        development, "select_canonical_count_space_bnb_frame", forbidden
    )
    monkeypatch.setattr(
        development,
        "select_canonical_score_tie_bulk_substitution_frame",
        forbidden,
    )
    monkeypatch.setattr(
        development.source,
        "select_canonical_count_space_bnb_frame",
        forbidden,
    )
    monkeypatch.setattr(
        development.source,
        "select_canonical_count_space_bnb_single_exchange_frame",
        forbidden,
    )

    protocol = development.build_registration()

    assert protocol["post_result_development"] is True
    assert protocol["independent_confirmation"] is False
    assert set(protocol["consumed_rosters"]) == set(development.ROSTER_NAMES)
    assert all(
        len(rows) == len({tuple(row) for row in rows}) == development.ROSTER_SIZE
        for rows in protocol["consumed_rosters"].values()
    )
    assert {
        name: protocol["source_contract"][f"{name}_roster_contract"]["seed"]
        for name in development.ROSTER_NAMES
    } == {"seed13041": 13041, "seed260919": 260919, "seed270919": 270919}
    assert protocol["registration_scope"]["selector_invoked"] is False
    assert protocol["registration_scope"]["reference_invoked"] is False
    assert protocol["registration_scope"]["new_roster_generated"] is False
    assert protocol["registration_scope"]["trajectory_built"] is False


@pytest.mark.parametrize("roster_name", development.ROSTER_NAMES)
def test_each_consumed_roster_is_bound_to_its_frozen_payload(
    frozen_protocol, roster_name
):
    contract = copy.deepcopy(frozen_protocol["source_contract"])
    roster = contract[f"{roster_name}_roster_contract"]
    roster["queues"][0][0] += 1

    with pytest.raises(
        development.R16BulkTieDevelopmentRefusal,
        match=f"consumed roster changed: {roster_name}",
    ):
        development._load_consumed_rosters(contract)


def test_registration_rejects_changed_source_fixed_sources(tmp_path):
    fixed = development._read_object(development.DEFAULT_SOURCE_FIXED_SOURCES)
    key = next(iter(fixed))
    fixed[key] += "\n# changed"
    changed = tmp_path / "changed_source_fixed_sources.json"
    changed.write_text(json.dumps(fixed), encoding="utf-8")

    with pytest.raises(
        development.R16BulkTieDevelopmentRefusal,
        match="frozen source confirmation changed",
    ):
        development.build_registration(source_fixed_sources=changed)


def test_source_contract_rejects_mutated_embedded_envelope():
    protocol = development._read_object(development.DEFAULT_SOURCE_REGISTRATION)
    result = development._read_object(development.DEFAULT_SOURCE_RESULT)
    protocol["development_contract"]["physical_complete_frame_envelope"]["model"][
        "options"
    ][0]["duration"] += 1.0
    result["protocol"] = copy.deepcopy(protocol)

    with pytest.raises(
        development.R16BulkTieDevelopmentRefusal,
        match="embedded development contract differs",
    ):
        development._validated_source_contract(
            result, protocol, development.DEFAULT_SOURCE_ROSTER
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "method_contract",
            {**development.METHOD_CONTRACT, "base_lp_budget": 10},
        ),
        (
            "expected_work",
            {**development.EXPECTED_WORK, "r16_candidate_calls": 1535},
        ),
    ],
)
def test_protocol_rejects_method_or_work_contract_drift(
    frozen_protocol, field, replacement
):
    protocol = copy.deepcopy(frozen_protocol)
    protocol[field] = replacement

    with pytest.raises(
        development.R16BulkTieDevelopmentRefusal,
        match="registered R16 bulk-tie protocol changed",
    ):
        development._validate_protocol(protocol)


def test_screen_rejects_undercharged_structural_pair_checks(
    frozen_protocol, v5_model, monkeypatch
):
    capacity_rows = frozen_protocol["source_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    real_total_work_row = development.total_work_row

    def undercharged(result):
        row = real_total_work_row(result)
        row["candidate_evaluations"] -= result.pair_checks
        row["residual_fill_option_probes"] -= result.pair_checks
        return row

    monkeypatch.setattr(development, "ROSTER_SIZE", 1)
    monkeypatch.setattr(development, "total_work_row", undercharged)
    row, checks = development._screen_roster(
        v5_model, [(11, 11, 12)], capacity_rows, "seed270919"
    )

    assert row["structural_pair_check_count"] == 7
    assert row["active_bulk_candidate_attempt_count"] == 1
    assert checks[
        "all_seven_structural_pair_checks_are_charged_per_triggered_state"
    ] is False
    assert row["development_gate_passed"] is False


def test_nontriggered_state_adds_no_structural_pair_work(
    frozen_protocol, v5_model, monkeypatch
):
    capacity_rows = frozen_protocol["source_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    monkeypatch.setattr(development, "ROSTER_SIZE", 1)

    row, checks = development._screen_roster(
        v5_model, [(1, 1, 11)], capacity_rows, "seed13041"
    )

    assert row["polish_trigger_state_count"] == 0
    assert row["structural_pair_check_count"] == 0
    assert row["active_bulk_candidate_attempt_count"] == 0
    assert row["candidate_charged_work"] == row["candidate_base_work"]
    assert checks[
        "every_candidate_method_relation_and_certificate_is_valid"
    ] is True
    assert checks[
        "all_seven_structural_pair_checks_are_charged_per_triggered_state"
    ] is True
    assert row["development_gate_passed"] is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("independent_roster_seed", 270920),
        ("independent_roster_state_count", 511),
    ],
)
def test_source_result_rejects_seed_or_state_count_drift(field, replacement):
    protocol = development._read_object(development.DEFAULT_SOURCE_REGISTRATION)
    result = development._read_object(development.DEFAULT_SOURCE_RESULT)
    result[field] = replacement

    with pytest.raises(
        development.R16BulkTieDevelopmentRefusal,
        match="frozen failed bulk-tie source confirmation differs",
    ):
        development._validated_source_contract(
            result, protocol, development.DEFAULT_SOURCE_ROSTER
        )


@pytest.mark.parametrize("mode", ["seed", "state_count"])
def test_source_roster_contract_reuses_authoritative_roster_validation(
    tmp_path, mode
):
    protocol = development._read_object(development.DEFAULT_SOURCE_REGISTRATION)
    result = development._read_object(development.DEFAULT_SOURCE_RESULT)
    roster = development._read_object(development.DEFAULT_SOURCE_ROSTER)
    if mode == "seed":
        roster["seed"] += 1
    else:
        roster["queues"] = roster["queues"][:-1]
        roster["requested_queue_count"] = len(roster["queues"])
    protocol["queue_roster_contract"] = copy.deepcopy(roster)
    result["protocol"] = copy.deepcopy(protocol)
    changed = tmp_path / f"changed_{mode}_roster.json"
    changed.write_text(json.dumps(roster), encoding="utf-8")

    with pytest.raises(
        development.R16BulkTieDevelopmentRefusal,
        match="registered independent roster changed or is invalid",
    ):
        development._validated_source_contract(result, protocol, changed)


def test_source_failure_and_decision_order_are_frozen():
    protocol = development._read_object(development.DEFAULT_SOURCE_REGISTRATION)
    result = development._read_object(development.DEFAULT_SOURCE_RESULT)
    contract = development._validated_source_contract(
        result, protocol, development.DEFAULT_SOURCE_ROSTER
    )
    passing = {
        "support_gate_passed": True,
        "action_identity_gate_passed": True,
        "charged_compute_gate_passed": True,
    }

    assert result["scientific_decision"] == development.SOURCE_DECISION
    assert contract["scientific_decision"] == development.SOURCE_DECISION
    assert contract["source_r16_failure"] == {
        "candidate_score_exact_state_count": 512,
        "candidate_canonical_exact_state_count": 0,
        "candidate_action_matches_reference_count": 509,
        "score_tie_action_mismatch_count": 3,
        "positive_regret_action_mismatch_count": 0,
    }
    assert (
        development._decision(False, [passing], True, True)
        == development.INVALID_DECISION
    )
    assert (
        development._decision(True, [passing], True, False)
        == development.INVALID_DECISION
    )
    assert development._decision(
        True, [{**passing, "support_gate_passed": False}], True, True
    ) == development.SUPPORT_FAIL_DECISION
    assert development._decision(
        True, [{**passing, "action_identity_gate_passed": False}], True, True
    ) == development.ACTION_FAIL_DECISION
    assert (
        development._decision(True, [passing], False, True)
        == development.COMPUTE_FAIL_DECISION
    )
    assert (
        development._decision(True, [passing], True, True)
        == development.PASS_DECISION
    )


def test_mocked_pass_authorizes_only_new_static_confirmation(
    tmp_path, frozen_protocol, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(frozen_protocol),
    }
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed_sources.json"
    output = tmp_path / "result.json"
    registration.write_text(json.dumps(protocol), encoding="utf-8")
    fixed.write_text(
        json.dumps(development._current_fixed_sources()), encoding="utf-8"
    )

    def fake_screen(_model, queues, _capacity_rows, roster_name):
        assert len(queues) == development.ROSTER_SIZE
        candidate = {field: 1 for field in development.canonical.WORK_FIELDS}
        reference = {field: 2 for field in development.canonical.WORK_FIELDS}
        return (
            {
                "roster": roster_name,
                "state_count": development.ROSTER_SIZE,
                "candidate_charged_work": candidate,
                "cold_reference_actual_work": reference,
                "support_gate_passed": True,
                "action_identity_gate_passed": True,
                "charged_compute_gate_passed": True,
            },
            {"synthetic_execution_check": True},
        )

    monkeypatch.setattr(development, "_screen_roster", fake_screen)
    report = development.run_registered(registration, fixed, output)

    assert report["status"] == "PASS"
    assert report["scientific_decision"] == development.PASS_DECISION
    assert report["new_independent_static_confirmation_registration_authorized"] is True
    assert report["trajectory_registration_prerequisite_passed"] is False
    assert report["capacity_closure_reclassified"] is False
    assert report["retained_v5_capacity_established"] is False
    assert report["retained_v5_closure_passed"] is False
    assert report["trajectory_runs"] == 0
    assert report["arrival_generations"] == 0
