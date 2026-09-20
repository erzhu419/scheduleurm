import copy
import json
from pathlib import Path

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_r16_support_rescue_confirmation as confirmation,
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
        confirmation.development.source.previous.development.source.capacity.base
        ._model_from_snapshot(snapshot)
    )


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_registration_uses_only_seed_290919_rng_and_no_selector(monkeypatch):
    selector_calls = []

    def forbidden_selector(*args, **kwargs):
        selector_calls.append((args, kwargs))
        raise AssertionError("registration invoked a selector")

    for owner, name in (
        (confirmation, "select_canonical_count_space_bnb_frame"),
        (confirmation, "select_canonical_count_space_bnb_single_exchange_frame"),
        (confirmation, "select_canonical_support_rescue_bulk_tie_frame"),
        (confirmation.source, "select_canonical_count_space_bnb_frame"),
        (
            confirmation.source,
            "select_canonical_count_space_bnb_single_exchange_frame",
        ),
        (confirmation.development, "select_canonical_count_space_bnb_frame"),
        (
            confirmation.development,
            "select_canonical_support_rescue_bulk_tie_frame",
        ),
    ):
        monkeypatch.setattr(owner, name, forbidden_selector, raising=False)

    original_random = confirmation.random.Random
    rng_seeds = []

    def recorded_random(seed):
        rng_seeds.append(seed)
        return original_random(seed)

    monkeypatch.setattr(confirmation.random, "Random", recorded_random)
    protocol = confirmation.build_registration()
    roster = protocol["queue_roster_contract"]

    assert selector_calls == []
    assert rng_seeds == [confirmation.ROSTER_SEED] == [290919]
    assert protocol["independent_confirmation"] is True
    assert protocol["registration_scope"]["roster_rng_initializations"] == 1
    assert protocol["registration_scope"]["selector_invoked"] is False
    assert protocol["registration_scope"]["reference_invoked"] is False
    assert protocol["registration_scope"]["trajectory_built"] is False
    assert protocol["registration_scope"]["paper_edited"] is False
    assert roster["seed"] == 290919
    assert len(roster["queues"]) == len({tuple(row) for row in roster["queues"]}) == 512


def test_source_development_pass_fixed_sources_and_protocol_are_bound(frozen_protocol):
    paths = frozen_protocol["source_paths"]
    source_protocol = confirmation._read_object(
        Path(paths["development_registration"])
    )
    source_result = confirmation._read_object(Path(paths["development_result"]))
    source_fixed = confirmation._read_object(
        Path(paths["development_fixed_sources"])
    )

    assert source_fixed == confirmation.development._current_fixed_sources()
    assert source_result["protocol"] == source_protocol
    assert source_result["status"] == "PASS"
    assert source_result["scientific_decision"] == confirmation.development.PASS_DECISION
    assert source_result[
        "new_independent_static_confirmation_registration_authorized"
    ] is True
    assert confirmation._validated_development_contract(
        source_result, source_protocol
    ) == frozen_protocol["development_contract"]
    contract = frozen_protocol["development_contract"]
    assert contract["method_contract"] == confirmation.development.METHOD_CONTRACT
    assert contract["development_result_payload"] == source_result
    assert contract["development_result_payload"]["protocol"] == source_protocol


def test_registration_rejects_changed_development_sources_and_protocol(tmp_path):
    fixed = confirmation._read_object(
        confirmation.DEFAULT_DEVELOPMENT_FIXED_SOURCES
    )
    fixed[next(iter(fixed))] += "\n# changed"
    with pytest.raises(confirmation.R16SupportRescueConfirmationRefusal):
        confirmation.build_registration(
            development_fixed_sources=_write_json(tmp_path / "fixed.json", fixed)
        )

    protocol = confirmation._read_object(
        confirmation.DEFAULT_DEVELOPMENT_REGISTRATION
    )
    result = confirmation._read_object(confirmation.DEFAULT_DEVELOPMENT_RESULT)
    protocol["method_contract"]["rescue_lp_budget"] += 1
    result["protocol"] = copy.deepcopy(protocol)
    with pytest.raises(confirmation.R16SupportRescueConfirmationRefusal):
        confirmation.build_registration(
            development_registration=_write_json(
                tmp_path / "protocol.json", protocol
            ),
            development_result=_write_json(tmp_path / "result.json", result),
        )



@pytest.mark.parametrize("mutation", ("decision", "outcome", "work"))
def test_registration_rejects_three_development_result_mutations(
    tmp_path, mutation
):
    result = confirmation._read_object(confirmation.DEFAULT_DEVELOPMENT_RESULT)
    if mutation == "decision":
        result["scientific_decision"] = "CHANGED"
    elif mutation == "outcome":
        result["r16_consumed_roster_summaries"][0][
            "maximum_candidate_epsilon"
        ] += 1e-6
    else:
        result["r16_consumed_roster_summaries"][0][
            "candidate_total_charged_work"
        ]["lp_calls"] += 1

    with pytest.raises(confirmation.R16SupportRescueConfirmationRefusal):
        confirmation.build_registration(
            development_result=_write_json(
                tmp_path / f"changed_{mutation}_result.json", result
            )
        )


def test_roster_is_complete_unique_and_disjoint_from_six_prior_rosters(
    frozen_protocol,
):
    exclusions = confirmation._load_exclusion_rosters(
        frozen_protocol["source_paths"]
    )
    first = confirmation.generate_roster(exclusions)
    second = confirmation.generate_roster(exclusions)
    sets = {
        name: {tuple(row) for row in rows} for name, rows in exclusions.items()
    }
    union = set().union(*sets.values())
    queues = {tuple(row) for row in first["queues"]}

    assert set(sets) == {
        "count_space_confirmation",
        "joint95_static",
        "canonical_development",
        "budget_pair_confirmation",
        "single_exchange_confirmation",
        "bulk_tie_confirmation",
    }
    assert all(len(rows) == 512 for rows in sets.values())
    assert len(union) == first["prior_roster_union_size"] == 3072
    assert first == second == frozen_protocol["queue_roster_contract"]
    assert first["all_six_prior_512_rosters_excluded"] is True
    assert first["schema_version"] == 1
    assert first["experiment"] == confirmation.EXPERIMENT
    assert first["seed"] == confirmation.ROSTER_SEED == 290919
    assert first["requested_queue_count"] == confirmation.ROSTER_SIZE == 512
    assert first["candidate_draws"] == confirmation.ROSTER_CANDIDATE_DRAWS == 558
    assert first["rng_integer_draws"] == confirmation.ROSTER_RNG_INTEGER_DRAWS
    assert first["rng_integer_draws"] == 1674
    assert first["coordinate_ranges_inclusive"] == [
        list(bounds) for bounds in confirmation.ROSTER_RANGES
    ]
    assert first["generation_rule"] == confirmation.ROSTER_GENERATION_RULE
    assert first["zero_queue_excluded"] is True
    assert first["old_1000_grid_excluded"] is True
    assert first["old_1000_grid_intersection_count"] == 0
    assert set(first["prior_roster_intersection_count"].values()) == {0}
    assert len(queues) == 512
    assert (0, 0, 0) not in queues
    assert queues.isdisjoint(union)
    assert queues.isdisjoint(confirmation.source._old_grid())

    contract = frozen_protocol["exclusion_roster_contract"]
    assert set(contract) == set(sets)
    assert all(
        row["size"] == 512
        and len(row["queues"]) == 512
        and {tuple(queue) for queue in row["queues"]} == sets[name]
        for name, row in contract.items()
    )


def test_sixth_prior_roster_tamper_is_refused(tmp_path, frozen_protocol):
    paths = copy.deepcopy(frozen_protocol["source_paths"])
    payload = confirmation._read_object(
        confirmation.DEFAULT_BULK_CONFIRMATION_ROSTER
    )
    replacement = list(frozen_protocol["queue_roster_contract"]["queues"][0])
    assert replacement not in payload["queues"]
    payload["queues"][0] = replacement
    changed = _write_json(tmp_path / "changed_seed280919.json", payload)
    paths["bulk_confirmation_roster"] = str(changed)

    with pytest.raises(confirmation.R16SupportRescueConfirmationRefusal):
        confirmation._load_exclusion_rosters(paths)


def test_selector_contract_freezes_budgets_eta_tolerance_and_cardinality(
    frozen_protocol,
):
    selector = frozen_protocol["selector_contract"]
    method = selector["r16_method_contract"]
    model = frozen_protocol["development_contract"][
        "physical_complete_frame_envelope"
    ]["model"]
    option_count = len(model["options"])

    assert selector["fixed_budget_by_lane"] == {"8": 48, "16": 9}
    assert selector["r16_rescue_lp_budget"] == 144
    assert selector["r16_support_epsilon_limit"] == (
        confirmation.development.SUPPORT_EPSILON_LIMIT
    )
    assert selector["score_tolerance"] == confirmation.canonical.SCORE_TOLERANCE
    assert selector["r8_method_contract"] == (
        confirmation.source.single_confirmation.development.METHOD_CONTRACT
    )
    assert method == confirmation.development.METHOD_CONTRACT
    assert method["primary_lp_budget"] == 9
    assert method["rescue_lp_budget"] == 144 == 16 * option_count
    assert method["support_epsilon_limit"] == (
        confirmation.development.SUPPORT_EPSILON_LIMIT
    )
    assert method["score_tolerance"] == (
        confirmation.development.RESCUE_SCORE_TOLERANCE
    )
    assert method["score_tolerance"] == confirmation.canonical.SCORE_TOLERANCE


@pytest.mark.parametrize("field", ("selector_contract", "expected_work"))
def test_protocol_rejects_selector_or_work_contract_drift(
    frozen_protocol, field,
):
    protocol = copy.deepcopy(frozen_protocol)
    if field == "selector_contract":
        protocol[field]["r16_method_contract"]["rescue_lp_budget"] = 143
    else:
        protocol[field]["r16_primary_composite_calls"] -= 1
    with pytest.raises(confirmation.R16SupportRescueConfirmationRefusal):
        confirmation._validate_protocol(protocol)


def test_r8_k48_single_exchange_contract_is_retained(
    frozen_protocol, v5_model, monkeypatch,
):
    capacity = frozen_protocol["development_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["8"]
    monkeypatch.setattr(confirmation, "ROSTER_SIZE", 1)
    monkeypatch.setattr(confirmation.source, "ROSTER_SIZE", 1)

    row, checks = confirmation._screen_lane(
        v5_model, [(3, 1, 11)], confirmation.R8_LANES, capacity
    )

    assert row["lanes"] == 8
    assert row["fixed_lp_budget"] == 48
    assert row["exchange_attempt_count"] > 0
    assert checks[
        "candidate_work_is_exactly_base_plus_registered_polish_charge"
    ]


def test_triggered_r16_rescue_is_exact_canonical_and_fully_charged(
    frozen_protocol, v5_model, monkeypatch,
):
    capacity = frozen_protocol["development_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    monkeypatch.setattr(confirmation, "ROSTER_SIZE", 1)
    monkeypatch.setattr(confirmation.development, "ROSTER_SIZE", 1)

    row, checks = confirmation._screen_lane(
        v5_model, [(0, 0, 10)], confirmation.R16_LANES, capacity
    )

    assert row["lanes"] == 16
    assert row["primary_lp_budget"] == 9
    assert row["rescue_lp_budget"] == 144
    assert row["rescue_trigger_state_count"] == 1
    assert row["exact_rescue_state_count"] == 1
    assert row["candidate_score_exact_state_count"] == 1
    assert row["candidate_canonical_exact_state_count"] == 1
    assert row["candidate_action_matches_reference_count"] == 1
    assert row["first_rescue_witness"]["rescue_score_exact"] is True
    assert row["first_rescue_witness"]["rescue_canonical_exact"] is True
    assert checks["every_triggered_rescue_is_score_exact_and_canonical_exact"]
    assert checks["candidate_work_is_exactly_primary_plus_executed_rescue"]
    assert row["candidate_charged_work"] == {
        field: row["primary_charged_work"][field]
        + row["rescue_charged_work"][field]
        for field in confirmation.canonical.WORK_FIELDS
    }


def test_safe_r16_witness_skips_rescue_and_charges_only_primary(
    frozen_protocol, v5_model, monkeypatch,
):
    capacity = frozen_protocol["development_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    monkeypatch.setattr(confirmation, "ROSTER_SIZE", 1)
    monkeypatch.setattr(confirmation.development, "ROSTER_SIZE", 1)

    row, checks = confirmation._screen_lane(
        v5_model, [(1, 1, 11)], confirmation.R16_LANES, capacity
    )

    assert row["rescue_trigger_state_count"] == 0
    assert row["exact_rescue_state_count"] == 0
    assert row["first_rescue_witness"] is None
    assert row["rescue_charged_work"] == {
        field: 0 for field in confirmation.canonical.WORK_FIELDS
    }
    assert row["candidate_charged_work"] == row["primary_charged_work"]
    assert checks["every_triggered_rescue_is_score_exact_and_canonical_exact"]
    assert checks["candidate_work_is_exactly_primary_plus_executed_rescue"]


def test_r16_screen_rejects_undercharged_rescue_work(
    frozen_protocol, v5_model, monkeypatch,
):
    capacity = frozen_protocol["development_contract"][
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    real_total = confirmation.development.rescue_total_work_row

    def undercharged(result):
        row = real_total(result)
        row["candidate_evaluations"] -= 1
        return row

    monkeypatch.setattr(confirmation, "ROSTER_SIZE", 1)
    monkeypatch.setattr(confirmation.development, "ROSTER_SIZE", 1)
    monkeypatch.setattr(
        confirmation.development, "rescue_total_work_row", undercharged
    )
    row, checks = confirmation._screen_lane(
        v5_model, [(0, 0, 10)], confirmation.R16_LANES, capacity
    )

    assert row["rescue_trigger_state_count"] == 1
    assert checks["candidate_work_is_exactly_primary_plus_executed_rescue"] is False
    assert row["lane_confirmation_gate_passed"] is False


def test_decision_branches_are_ordered_and_include_aggregate_gate():
    passing = {
        "support_gate_passed": True,
        "independent_roster_action_identity_gate_passed": True,
        "actual_compute_gate_passed": True,
    }
    assert confirmation._decision(False, [passing], True) == (
        confirmation.INVALID_DECISION
    )
    assert confirmation._decision(
        True, [{**passing, "support_gate_passed": False}], True
    ) == confirmation.SUPPORT_FAIL_DECISION
    assert confirmation._decision(
        True,
        [{**passing, "independent_roster_action_identity_gate_passed": False}],
        True,
    ) == confirmation.ACTION_FAIL_DECISION
    assert confirmation._decision(True, [passing], False) == (
        confirmation.COMPUTE_FAIL_DECISION
    )
    assert confirmation._decision(
        True, [{**passing, "actual_compute_gate_passed": False}], True
    ) == confirmation.COMPUTE_FAIL_DECISION
    assert confirmation._decision(True, [passing], True) == confirmation.PASS_DECISION


def test_run_rejects_roster_tamper_before_selector_or_rng(
    tmp_path, frozen_protocol, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(frozen_protocol),
    }
    changed_roster = copy.deepcopy(protocol["queue_roster_contract"])
    changed_roster["queues"][0] = list(changed_roster["queues"][1])
    registration = _write_json(tmp_path / "protocol.json", protocol)
    roster = _write_json(tmp_path / "roster.json", changed_roster)
    fixed = _write_json(
        tmp_path / "fixed_sources.json", confirmation._current_fixed_sources()
    )
    output = tmp_path / "result.json"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tampered roster reached selector or RNG")

    monkeypatch.setattr(confirmation.random, "Random", forbidden)
    monkeypatch.setattr(confirmation, "generate_roster", forbidden)
    monkeypatch.setattr(confirmation, "_screen_lane", forbidden)

    with pytest.raises(
        confirmation.R16SupportRescueConfirmationRefusal,
        match="registered independent roster changed",
    ):
        confirmation.run_registered(registration, roster, fixed, output)

    assert not output.exists()


def test_run_uses_frozen_roster_without_rng_and_pass_only_authorizes_trajectory_registration(
    tmp_path, frozen_protocol, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(frozen_protocol),
    }
    registration = _write_json(tmp_path / "protocol.json", protocol)
    roster = _write_json(
        tmp_path / "roster.json", protocol["queue_roster_contract"]
    )
    fixed = _write_json(
        tmp_path / "fixed_sources.json", confirmation._current_fixed_sources()
    )
    output = tmp_path / "result.json"
    paper_path = confirmation.ROOT / "paper/main.tex"
    paper_before = paper_path.read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("execution regenerated a roster or initialized RNG")

    def fake_screen(_model, queues, lanes, capacity_rows):
        assert len(queues) == confirmation.ROSTER_SIZE
        assert lanes in (confirmation.R8_LANES, confirmation.R16_LANES)
        assert len(capacity_rows) == len(confirmation.LOADS)
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

    monkeypatch.setattr(confirmation.random, "Random", forbidden)
    monkeypatch.setattr(confirmation, "generate_roster", forbidden)
    monkeypatch.setattr(confirmation, "_screen_lane", fake_screen)
    report = confirmation.run_registered(
        registration, roster, fixed, output
    )

    assert report["status"] == "PASS"
    assert report["scientific_decision"] == confirmation.PASS_DECISION
    assert report["independent_static_confirmation_passed"] is True
    assert report["envelope_normalized_trajectory_registration_authorized"] is True
    assert report["trajectory_registration_prerequisite_passed"] is True
    assert report["trajectory_runs"] == 0
    assert report["arrival_generations"] == 0
    assert report["rng_draws_during_execution"] == 0
    assert report["server_pulls"] == 0
    assert report["capacity_closure_reclassified"] is False
    assert report["retained_v5_capacity_established"] is False
    assert report["retained_v5_closure_passed"] is False
    assert protocol["registration_scope"]["paper_edited"] is False
    assert paper_path.read_bytes() == paper_before
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_write_new_refuses_to_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    confirmation._write_new(path, {"version": 1})

    with pytest.raises(FileExistsError):
        confirmation._write_new(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
