import copy
import json
from pathlib import Path
import random

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_r16_support_rescue_development as development,
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


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_registration_reads_four_consumed_rosters_without_selector_or_rng_calls(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("registration invoked a selector or RNG")

    for module, name in (
        (development, "select_canonical_support_rescue_bulk_tie_frame"),
        (development, "select_canonical_count_space_bnb_frame"),
        (development.source, "select_canonical_count_space_bnb_frame"),
        (
            development.source,
            "select_canonical_count_space_bnb_single_exchange_frame",
        ),
        (
            development.source,
            "select_canonical_score_tie_bulk_substitution_frame",
        ),
    ):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(random, "Random", forbidden)

    protocol = development.build_registration()
    rosters = protocol["consumed_rosters"]
    union = {
        tuple(queue)
        for rows in rosters.values()
        for queue in rows
    }

    assert protocol["post_result_development"] is True
    assert protocol["independent_confirmation"] is False
    assert set(rosters) == set(development.ROSTER_NAMES)
    assert all(
        len(rows) == len({tuple(row) for row in rows}) == development.ROSTER_SIZE
        for rows in rosters.values()
    )
    assert len(union) == 4 * development.ROSTER_SIZE
    assert protocol["registration_scope"]["selector_invoked"] is False
    assert protocol["registration_scope"]["reference_invoked"] is False
    assert protocol["registration_scope"]["rng_draws"] == 0


@pytest.mark.parametrize("roster_name", development.ROSTER_NAMES)
def test_each_roster_contract_freezes_absolute_path_and_complete_payload(
    frozen_protocol, roster_name
):
    frozen = frozen_protocol["source_contract"][f"{roster_name}_roster_contract"]
    path = Path(frozen["path"])
    payload = development._read_object(path)

    assert path.is_absolute()
    assert frozen["seed"] == int(roster_name.removeprefix("seed"))
    assert frozen["size"] == development.ROSTER_SIZE
    assert len(frozen["queues"]) == development.ROSTER_SIZE
    assert frozen.get("payload") == payload


def test_registration_rejects_changed_source_fixed_sources(tmp_path):
    fixed = development._read_object(development.DEFAULT_SOURCE_FIXED_SOURCES)
    key = next(iter(fixed))
    fixed[key] += "\n# changed"
    changed = _write_json(tmp_path / "changed_fixed_sources.json", fixed)

    with pytest.raises(
        development.R16SupportRescueDevelopmentRefusal,
        match="frozen source confirmation changed",
    ):
        development.build_registration(source_fixed_sources=changed)


def test_registration_rejects_changed_source_protocol(tmp_path):
    protocol = development._read_object(development.DEFAULT_SOURCE_REGISTRATION)
    result = development._read_object(development.DEFAULT_SOURCE_RESULT)
    protocol["selector_contract"]["fixed_budget_by_lane"]["16"] = 10
    result["protocol"] = copy.deepcopy(protocol)
    changed_protocol = _write_json(tmp_path / "changed_protocol.json", protocol)
    changed_result = _write_json(tmp_path / "changed_protocol_result.json", result)

    with pytest.raises(
        development.R16SupportRescueDevelopmentRefusal,
        match="registered R16 bulk-tie confirmation protocol changed",
    ):
        development.build_registration(
            source_registration=changed_protocol,
            source_result=changed_result,
        )


def test_registration_rejects_changed_unique_positive_regret_witness(tmp_path):
    result = development._read_object(development.DEFAULT_SOURCE_RESULT)
    r16 = next(row for row in result["lane_summaries"] if row["lanes"] == 16)
    r16["first_action_mismatch_witness"]["value"] += 1e-3
    changed = _write_json(tmp_path / "changed_witness_result.json", result)

    with pytest.raises(
        development.R16SupportRescueDevelopmentRefusal,
        match="frozen failed support confirmation differs",
    ):
        development.build_registration(source_result=changed)


def test_source_contract_binds_the_only_positive_regret_witness(frozen_protocol):
    contract = frozen_protocol["source_contract"]
    r16 = contract["source_lane_outcomes"]["16"]
    witness = contract["source_unique_r16_positive_regret_witness"]

    assert r16["candidate_action_mismatch_count"] == 1
    assert r16["score_tie_action_mismatch_count"] == 0
    assert r16["positive_regret_action_mismatch_count"] == 1
    assert r16["first_action_mismatch_witness"] == witness
    assert witness == development.SOURCE_R16_WITNESS
    assert witness["mismatch_kind"] == "positive_regret"
    assert witness["base_queue"] == [0, 0, 10]
    assert witness["scaled_queue"] == [0, 0, 80]


def test_method_contract_freezes_option_count_budget_rule_and_score_tolerance(
    frozen_protocol,
):
    method = frozen_protocol["method_contract"]
    option_count = len(
        frozen_protocol["source_contract"]["physical_complete_frame_envelope"][
            "model"
        ]["options"]
    )

    assert method["lane"] == development.R16_LANES
    assert method["primary_lp_budget"] == development.PRIMARY_BUDGET
    assert method["rescue_lp_budget"] == development.R16_LANES * option_count
    assert "option" in method["rescue_budget_rule"].lower()
    assert method["score_tolerance"] == development.RESCUE_SCORE_TOLERANCE


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "method_contract",
            {
                **development.METHOD_CONTRACT,
                "score_tolerance": 2 * development.RESCUE_SCORE_TOLERANCE,
            },
        ),
        (
            "method_contract",
            {
                **development.METHOD_CONTRACT,
                "rescue_budget_rule": "changed",
            },
        ),
        (
            "expected_work",
            {
                **development.EXPECTED_WORK,
                "r16_primary_composite_calls": 4 * development.ROSTER_SIZE - 1,
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
        development.R16SupportRescueDevelopmentRefusal,
        match="registered R16 support-rescue protocol changed",
    ):
        development._validate_protocol(protocol)


def test_failure_witness_triggers_exact_canonical_rescue(
    frozen_protocol, v5_model, monkeypatch
):
    capacity_rows = frozen_protocol["source_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    monkeypatch.setattr(development, "ROSTER_SIZE", 1)

    row, checks = development._screen_roster(
        v5_model, [(0, 0, 10)], capacity_rows, "seed280919"
    )

    assert row["rescue_trigger_state_count"] == 1
    assert row["exact_rescue_state_count"] == 1
    assert row["candidate_score_exact_state_count"] == 1
    assert row["candidate_canonical_exact_state_count"] == 1
    assert row["candidate_action_matches_reference_count"] == 1
    assert row["first_rescue_witness"]["base_queue"] == [0, 0, 10]
    assert row["first_rescue_witness"]["scaled_queue"] == [0, 0, 80]
    assert row["first_rescue_witness"]["rescue_score_exact"] is True
    assert row["first_rescue_witness"]["rescue_canonical_exact"] is True
    assert checks["every_triggered_rescue_is_score_exact_and_canonical_exact"]
    assert checks["candidate_work_is_exactly_primary_plus_executed_rescue"]


def test_safe_nonexact_state_does_not_trigger_rescue(
    frozen_protocol, v5_model, monkeypatch
):
    capacity_rows = frozen_protocol["source_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    monkeypatch.setattr(development, "ROSTER_SIZE", 1)

    row, checks = development._screen_roster(
        v5_model, [(1, 1, 11)], capacity_rows, "seed13041"
    )

    assert row["rescue_trigger_state_count"] == 0
    assert row["exact_rescue_state_count"] == 0
    assert row["candidate_score_exact_state_count"] == 0
    assert row["candidate_canonical_exact_state_count"] == 0
    assert all(value == 0 for value in row["rescue_charged_work"].values())
    assert row["candidate_total_charged_work"] == row["primary_charged_work"]
    assert all(checks.values())


def test_screen_rejects_undercharged_rescue_work(
    frozen_protocol, v5_model, monkeypatch
):
    capacity_rows = frozen_protocol["source_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    real_total_work_row = development.rescue_total_work_row

    def undercharged(result):
        row = real_total_work_row(result)
        row["candidate_evaluations"] -= 1
        return row

    monkeypatch.setattr(development, "ROSTER_SIZE", 1)
    monkeypatch.setattr(development, "rescue_total_work_row", undercharged)
    row, checks = development._screen_roster(
        v5_model, [(0, 0, 10)], capacity_rows, "seed280919"
    )

    assert row["rescue_trigger_state_count"] == 1
    assert checks[
        "every_candidate_method_relation_and_certificate_is_valid"
    ] is False
    assert checks[
        "candidate_work_is_exactly_primary_plus_executed_rescue"
    ] is False
    assert row["development_gate_passed"] is False


def test_decision_gate_order_is_frozen():
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
    assert development._decision(
        True, [{**passing, "charged_compute_gate_passed": False}], True, True
    ) == development.COMPUTE_FAIL_DECISION
    assert (
        development._decision(True, [passing], False, True)
        == development.COMPUTE_FAIL_DECISION
    )
    assert (
        development._decision(True, [passing], True, True)
        == development.PASS_DECISION
    )


def test_mocked_pass_authorizes_only_new_independent_static_confirmation(
    tmp_path, frozen_protocol, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(frozen_protocol),
    }
    registration = _write_json(tmp_path / "protocol.json", protocol)
    fixed = _write_json(
        tmp_path / "fixed_sources.json", development._current_fixed_sources()
    )
    output = tmp_path / "result.json"

    def fake_screen(_model, queues, _capacity_rows, roster_name):
        assert len(queues) == development.ROSTER_SIZE
        primary = {field: 1 for field in development.canonical.WORK_FIELDS}
        rescue = {field: 1 for field in development.canonical.WORK_FIELDS}
        candidate = {field: 2 for field in development.canonical.WORK_FIELDS}
        reference = {field: 3 for field in development.canonical.WORK_FIELDS}
        return (
            {
                "roster": roster_name,
                "state_count": development.ROSTER_SIZE,
                "primary_charged_work": primary,
                "rescue_charged_work": rescue,
                "candidate_total_charged_work": candidate,
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
    assert report[
        "new_independent_static_confirmation_registration_authorized"
    ] is True
    assert report["trajectory_registration_prerequisite_passed"] is False
    assert report["capacity_closure_reclassified"] is False
    assert report["retained_v5_capacity_established"] is False
    assert report["retained_v5_closure_passed"] is False
    assert report["trajectory_runs"] == 0
    assert report["arrival_generations"] == 0
    assert report["rng_draws"] == 0
