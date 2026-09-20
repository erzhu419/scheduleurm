import copy
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_single_exchange_development as development,
)


def test_registration_consumes_two_rosters_without_selector_calls(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("registration invoked a selector")

    monkeypatch.setattr(
        development, "select_canonical_count_space_bnb_frame", forbidden
    )
    monkeypatch.setattr(
        development,
        "select_canonical_count_space_bnb_single_exchange_frame",
        forbidden,
    )

    protocol = development.build_registration()

    assert protocol["post_result_development"] is True
    assert protocol["independent_confirmation"] is False
    assert protocol["method_contract"] == development.METHOD_CONTRACT
    assert set(protocol["consumed_rosters"]) == set(development.ROSTER_NAMES)
    assert all(
        len(rows) == len({tuple(row) for row in rows}) == development.ROSTER_SIZE
        for rows in protocol["consumed_rosters"].values()
    )
    seed260919 = protocol["source_contract"]["seed260919_roster_contract"]
    assert seed260919["path"] == str(development.DEFAULT_SOURCE_ROSTER.resolve())
    assert seed260919["payload"]["queues"] == seed260919["queues"]
    assert protocol["registration_scope"]["trajectory_built"] is False


@pytest.mark.parametrize("mode", ["short", "duplicate"])
def test_consumed_roster_validation_rejects_bad_cardinality_or_duplicates(
    tmp_path, mode
):
    contract = copy.deepcopy(development.build_registration()["source_contract"])
    roster_contract = contract["seed260919_roster_contract"]
    payload = copy.deepcopy(roster_contract["payload"])
    if mode == "short":
        payload["queues"] = payload["queues"][:-1]
        payload["requested_queue_count"] = len(payload["queues"])
    else:
        payload["queues"][-1] = list(payload["queues"][0])
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps(payload), encoding="utf-8")
    roster_contract["path"] = str(roster)
    roster_contract["queues"] = copy.deepcopy(payload["queues"])
    roster_contract["size"] = len(payload["queues"])
    roster_contract["payload"] = copy.deepcopy(payload)

    with pytest.raises(
        development.SingleExchangeDevelopmentRefusal,
        match="consumed roster changed: seed260919",
    ):
        development._load_consumed_rosters(contract)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "method_contract",
            {**development.METHOD_CONTRACT, "base_lp_budget": 47},
        ),
        (
            "registration_scope",
            {**development.REGISTRATION_SCOPE, "paper_edited": True},
        ),
    ],
)
def test_protocol_refuses_scientific_contract_drift(field, replacement):
    protocol = development.build_registration()
    protocol[field] = replacement

    with pytest.raises(
        development.SingleExchangeDevelopmentRefusal,
        match="registered single-exchange protocol changed",
    ):
        development._validate_protocol(protocol)


def test_source_contract_rejects_mutated_embedded_development_model():
    protocol = development._read_object(development.DEFAULT_SOURCE_REGISTRATION)
    result = development._read_object(development.DEFAULT_SOURCE_RESULT)
    protocol["development_contract"]["physical_complete_frame_envelope"]["model"][
        "options"
    ][0]["duration"] += 1.0
    result["protocol"] = copy.deepcopy(protocol)

    with pytest.raises(
        development.SingleExchangeDevelopmentRefusal,
        match="embedded development contract differs",
    ):
        development._validated_source_contract(
            result, protocol, development.DEFAULT_SOURCE_ROSTER
        )


def test_decision_branches_are_ordered_and_frozen():
    passing = {
        "support_gate_passed": True,
        "action_identity_gate_passed": True,
        "charged_compute_gate_passed": True,
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


def test_screen_detects_undercharged_exchange_attempts(monkeypatch):
    protocol = development.build_registration()
    contract = protocol["source_contract"]
    model = development.source.development.source.capacity.base._model_from_snapshot(
        contract["physical_complete_frame_envelope"]["model"]
    )
    capacity_rows = contract["physical_envelope_capacity_normalized_loads"]["8"]
    real_total_work_row = development.total_work_row

    def undercharged(result):
        row = real_total_work_row(result)
        row["candidate_evaluations"] -= result.exchange_attempts
        row["residual_fill_option_probes"] -= result.exchange_attempts
        return row

    monkeypatch.setattr(development, "ROSTER_SIZE", 1)
    monkeypatch.setattr(development, "total_work_row", undercharged)
    row, checks = development._screen_roster(
        model, [(3, 1, 11)], capacity_rows, "seed260919"
    )

    assert row["exchange_attempt_count"] > 0
    assert checks["polish_attempts_are_fully_charged"] is False
    assert row["development_gate_passed"] is False


def test_mocked_pass_authorizes_only_new_static_confirmation(tmp_path, monkeypatch):
    protocol = {
        "registered_at_utc": "test",
        **development.build_registration(),
    }
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
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
    assert report["trajectory_runs"] == 0
