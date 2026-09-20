from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_trajectory_failure_budget_frontier_development
    as diagnostic,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _work(value: int) -> dict[str, int]:
    return {field: value for field in diagnostic.WORK_FIELDS}


def _selection(
    *,
    indices=(0,),
    score=1.0,
    upper=1.0,
    epsilon=0.0,
    score_exact=True,
    canonical_exact=True,
    service=(1,),
    duration=1.0,
):
    return SimpleNamespace(
        option_indices=tuple(indices),
        score=float(score),
        upper_bound=float(upper),
        gap=max(0.0, float(upper) - float(score)),
        epsilon=float(epsilon),
        score_exact=bool(score_exact),
        canonical_exact=bool(canonical_exact),
        service=tuple(service),
        duration=float(duration),
    )


def _one_class_model() -> FrameModel:
    return FrameModel(
        node="fixture",
        scenario_id="fixture",
        lanes=8,
        workloads=("a",),
        scales=(1.0,),
        options=(
            FrameOption(
                action_id="a1",
                requirements=(1,),
                duration=1.0,
                completion_offsets=((1.0,),),
            ),
        ),
    )


@pytest.fixture(scope="module")
def registration():
    return diagnostic.build_registration()


def test_registration_is_explicit_post_result_development(registration):
    assert registration["status"] == diagnostic.REGISTERED_STATUS
    assert registration["registration_scope"] == {
        "post_result_development": True,
        "source_result_observed_before_registration": True,
        "independent_confirmation": False,
        "trajectory_or_flow_evidence": False,
        "rng_initializations": 0,
        "arrival_generations": 0,
        "trajectory_replays": 0,
        "selector_calls": 0,
        "server_pulls": 0,
    }
    assert registration["independent_confirmation_eligible"] is False
    assert registration["paper_claim_authorized"] is False
    frozen = registration["diagnostic_contract"]
    assert frozen["tape_count"] == frozen["trajectory_replays"] == 120
    assert frozen["counterfactual_trajectory_runs"] == 0
    assert frozen["arrival_seeds"] == list(range(12041, 12061))
    assert frozen["rescue_budgets_by_lane"] == {
        "8": [96, 144, 288, 576, 1024],
        "16": [96, 144, 288, 576, 1024],
    }
    assert frozen["trigger_rules"] == [
        "epsilon_near_eta_or_canonical_inexact"
    ]
    assert frozen["trigger_diagnostic_components"] == [
        "epsilon_near_lane_min_eta",
        "canonical_inexact",
    ]
    assert "not counted as a third method" in frozen["trigger_relation"]
    assert frozen["reference_is_not_a_trigger_input"] is True
    assert frozen["each_budget_is_an_independent_fresh_cold_call"] is True
    assert registration["source_transition_contract"]["scientific_decision"] == (
        diagnostic.source.INVALID_DECISION
    )
    assert len(registration["source_transition_contract"]["tape_rows"]) == 120


def test_lane_minimum_eta_is_derived_from_all_three_frozen_cases(registration):
    for lanes in diagnostic.LANES:
        expected = min(
            row["effective_slack_eta"]
            for row in registration[
                "physical_envelope_capacity_normalized_loads"
            ][str(lanes)]
        )
        assert registration["lane_minimum_effective_slack_eta"][str(lanes)] == expected


def test_registration_performs_no_rng_replay_or_selector_work(tmp_path, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("registration performed diagnostic work")

    monkeypatch.setattr(diagnostic.source, "poisson_arrivals", forbidden)
    monkeypatch.setattr(diagnostic.source, "_select_candidate", forbidden)
    monkeypatch.setattr(diagnostic.source, "_reference_call", forbidden)
    monkeypatch.setattr(diagnostic, "select_canonical_count_space_bnb_frame", forbidden)
    registration_path = tmp_path / "protocol.json"
    fixed_path = tmp_path / "fixed.json"
    result_path = tmp_path / "result.json"

    payload = diagnostic.register(registration_path, fixed_path, result_path)

    assert calls == []
    assert registration_path.is_file() and fixed_path.is_file()
    assert not result_path.exists()
    assert payload["registration_scope"]["selector_calls"] == 0


def test_changed_source_result_is_refused(tmp_path):
    payload = diagnostic._read_object(diagnostic.DEFAULT_SOURCE_RESULT)
    payload["development_advance"] = True
    changed = _write(tmp_path / "changed.json", payload)

    with pytest.raises(
        diagnostic.PostResultDevelopmentRefusal,
        match="frozen failed envelope-normalized transition",
    ):
        diagnostic.build_registration(source_result=changed)


def test_trigger_definitions_and_subset_relation_are_exact():
    exact = _selection(epsilon=0.0, canonical_exact=True)
    below = _selection(epsilon=0.09, score_exact=False, canonical_exact=False)
    boundary = _selection(epsilon=0.1, score_exact=False, canonical_exact=False)

    assert diagnostic._trigger_flags(exact, 0.1) == {
        "epsilon_near_lane_min_eta": False,
        "canonical_inexact": False,
        "epsilon_near_eta_or_canonical_inexact": False,
    }
    assert diagnostic._trigger_flags(below, 0.1) == {
        "epsilon_near_lane_min_eta": False,
        "canonical_inexact": True,
        "epsilon_near_eta_or_canonical_inexact": True,
    }
    assert diagnostic._trigger_flags(boundary, 0.1) == {
        "epsilon_near_lane_min_eta": True,
        "canonical_inexact": True,
        "epsilon_near_eta_or_canonical_inexact": True,
    }
    impossible = _selection(epsilon=0.1, canonical_exact=True)
    with pytest.raises(
        diagnostic.PostResultDevelopmentRefusal,
        match="must imply canonical inexactness",
    ):
        diagnostic._trigger_flags(impossible, 0.1)


def test_cold_rescue_never_degrades_base_incumbent():
    base = _selection(indices=(1,), score=10.0, duration=2.0)
    lower = _selection(indices=(0,), score=9.0, duration=1.0)
    microscopically_lower = _selection(
        indices=(0,),
        score=10.0 - diagnostic.SCORE_TOLERANCE / 2,
        duration=1.0,
    )
    equal_better_key = _selection(indices=(0,), score=10.0, duration=1.0)

    assert diagnostic._adopted_selection(base, lower, 2) == (base, "base")
    assert diagnostic._adopted_selection(base, microscopically_lower, 2) == (
        base,
        "base",
    )
    assert diagnostic._adopted_selection(base, equal_better_key, 2) == (
        equal_better_key,
        "rescue",
    )


def test_composite_bound_is_conservative_and_charges_both_calls(monkeypatch):
    model = _one_class_model()
    base = _selection(score=10.0, upper=12.0, epsilon=2.0)
    rescue_selection = _selection(
        score=9.0,
        upper=10.0,
        epsilon=1.0,
        score_exact=False,
        canonical_exact=False,
    )
    shadow = _selection(score=10.0, upper=10.0)
    monkeypatch.setattr(diagnostic, "_selection_certificate_valid", lambda *args: True)

    row = diagnostic._variant(
        model,
        (1,),
        {"selection": base, "work": _work(2), "valid": True},
        {"selection": rescue_selection, "work": _work(3), "valid": True},
        shadow,
        _work(9),
    )

    assert row["adopted"] == "base"
    assert row["upper_bound"] == 10.0
    assert row["epsilon"] == 0.0
    assert row["score_exact"] is True
    assert row["canonical_exact"] is True
    assert row["work"] == _work(5)
    assert row["shadow_work"] == _work(9)


def test_within_tolerance_bound_closure_does_not_invent_score_exact(monkeypatch):
    model = _one_class_model()
    base = _selection(
        score=10.0,
        upper=10.0 + diagnostic.SCORE_TOLERANCE / 2,
        epsilon=diagnostic.SCORE_TOLERANCE / 2,
        score_exact=False,
        canonical_exact=False,
    )
    shadow = _selection(score=10.0, upper=10.0)
    monkeypatch.setattr(diagnostic, "_selection_certificate_valid", lambda *args: True)

    row = diagnostic._variant(
        model,
        (1,),
        {"selection": base, "work": _work(1), "valid": True},
        None,
        shadow,
        _work(3),
    )

    assert row["combined_bound_closed_within_tolerance"] is True
    assert row["combined_bound_closed_by_exact_float_equality"] is False
    assert row["score_exact_witness_present"] is False
    assert row["score_exact"] is False
    assert row["score_exact_provenance"] is None


def test_float_equal_bound_and_different_action_exact_witness_do_not_transfer_exact(
    monkeypatch,
):
    model = _one_class_model()
    base = _selection(
        indices=(0,),
        score=10.0,
        upper=10.0,
        score_exact=False,
        canonical_exact=False,
    )
    other_action_exact = _selection(
        indices=(1,),
        score=10.0,
        upper=10.0,
        score_exact=True,
        canonical_exact=True,
    )
    shadow = _selection(indices=(0,), score=10.0, upper=10.0)
    monkeypatch.setattr(diagnostic, "_selection_certificate_valid", lambda *args: True)

    row = diagnostic._variant(
        model,
        (1,),
        {"selection": base, "work": _work(1), "valid": True},
        {
            "selection": other_action_exact,
            "work": _work(1),
            "valid": True,
        },
        shadow,
        _work(3),
    )

    assert row["adopted"] == "base"
    assert row["combined_bound_closed_by_exact_float_equality"] is True
    assert row["score_exact_witness_present"] is False
    assert row["score_exact"] is False
    assert row["canonical_exact"] is False


def test_replay_is_fifo_online_and_deduplicates_repeated_states(monkeypatch):
    model = _one_class_model()
    selection = _selection(service=(1,), duration=1.0)
    call = {
        "selection": selection,
        "work": _work(1),
        "valid": True,
        "method_relation": True,
    }
    observed = []

    def select(_model, queue):
        observed.append(tuple(queue))
        return call

    monkeypatch.setattr(diagnostic.source, "_select_candidate", select)
    states = {}
    frames = diagnostic._replay_candidate_tape(
        model,
        ((0.0, 0, 0), (0.5, 0, 1)),
        0.5,
        lanes=8,
        load=0.5,
        seed=12041,
        states=states,
    )

    assert frames == 2
    assert observed == [(1,), (1,)]
    assert set(states) == {(8, (1,))}
    record = states[(8, (1,))]
    assert record["occurrences_by_case"] == {0.5: 2}
    assert record["occurrences_by_tape"] == {(0.5, 12041): 2}


def test_every_unique_state_gets_shadow_and_each_budget_one_independent_call(
    monkeypatch,
):
    model = _one_class_model()
    base = _selection(
        indices=(0,),
        score=1.0,
        upper=1.1,
        epsilon=0.1,
        score_exact=False,
        canonical_exact=False,
    )
    shadow = _selection(indices=(0,), score=1.1, upper=1.1)
    record = {
        "lanes": 8,
        "queue": (1,),
        "observed_candidate_call": {
            "selection": base,
            "work": _work(1),
            "valid": True,
            "method_relation": True,
        },
        "occurrences_by_case": {0.5: 3},
        "occurrences_by_tape": {(0.5, 12041): 3},
        "first_occurrence": {"load": 0.5, "seed": 12041, "frame": 0},
    }
    calls = []

    monkeypatch.setattr(
        diagnostic.source,
        "_reference_call",
        lambda _model, _queue: (shadow, _work(10), True),
    )
    monkeypatch.setattr(
        diagnostic,
        "_diagnostic_base_call",
        lambda _model, _queue: {
            "selection": base,
            "work": _work(1),
            "valid": True,
            "method_relation": True,
            "method": "fixture_primary",
            "lp_node_budget": 48,
        },
    )
    monkeypatch.setattr(diagnostic, "_selection_certificate_valid", lambda *args: True)

    def rescue(_model, _queue, budget):
        calls.append(budget)
        return {
            "selection": shadow,
            "work": _work(2),
            "valid": True,
        }

    monkeypatch.setattr(diagnostic, "_rescue_call", rescue)
    row = diagnostic._evaluate_unique_state(model, record, lane_eta=0.1)

    assert calls == list(diagnostic.RESCUE_BUDGETS)
    assert row["trigger_flags"] == {
        "epsilon_near_lane_min_eta": True,
        "canonical_inexact": True,
        "epsilon_near_eta_or_canonical_inexact": True,
    }
    for rule in diagnostic.TRIGGER_RULES:
        for budget in diagnostic.RESCUE_BUDGETS:
            variant = row["variants"][(rule, budget)]
            assert variant["triggered"] is True
            assert variant["work"] == _work(3)
            assert variant["shadow_work"] == _work(10)


def test_r16_frontier_excludes_observed_old_k144_rescue_work(monkeypatch):
    model = replace(_one_class_model(), lanes=16)
    observed = _selection(
        indices=(0,) * 16,
        score=1.0,
        upper=1.0,
        epsilon=0.0,
        score_exact=True,
        canonical_exact=True,
        service=(16,),
    )
    primary = _selection(
        indices=(0,) * 16,
        score=0.9,
        upper=1.0,
        epsilon=0.1,
        score_exact=False,
        canonical_exact=False,
        service=(16,),
    )
    shadow = _selection(
        indices=(0,) * 16,
        score=1.0,
        upper=1.0,
        service=(16,),
    )
    record = {
        "lanes": 16,
        "queue": (16,),
        "observed_candidate_call": {
            "selection": observed,
            "work": _work(100),
            "valid": True,
            "method_relation": True,
        },
        "occurrences_by_case": {0.95: 2},
        "occurrences_by_tape": {(0.95, 12041): 2},
        "first_occurrence": {"load": 0.95, "seed": 12041, "frame": 0},
    }
    monkeypatch.setattr(
        diagnostic,
        "_diagnostic_base_call",
        lambda _model, _queue: {
            "selection": primary,
            "work": _work(1),
            "valid": True,
            "method_relation": True,
            "method": "r16_k9_bulk_primary_without_old_k144_rescue",
            "lp_node_budget": 9,
        },
    )
    monkeypatch.setattr(
        diagnostic.source,
        "_reference_call",
        lambda _model, _queue: (shadow, _work(10), True),
    )
    monkeypatch.setattr(diagnostic, "_selection_certificate_valid", lambda *args: True)
    rescue_calls = []

    def rescue(_model, _queue, budget):
        rescue_calls.append(budget)
        return {"selection": shadow, "work": _work(2), "valid": True}

    monkeypatch.setattr(diagnostic, "_rescue_call", rescue)
    row = diagnostic._evaluate_unique_state(model, record, lane_eta=0.1)

    assert row["observed_candidate"]["work"] == _work(100)
    assert row["frontier_base"]["work"] == _work(1)
    assert row["frontier_base"]["lp_node_budget"] == 9
    assert rescue_calls == list(diagnostic.RESCUE_BUDGETS)
    for budget in diagnostic.RESCUE_BUDGETS:
        variant = row["variants"][(diagnostic.TRIGGER_RULES[0], budget)]
        assert variant["work"] == _work(3)


def test_r16_diagnostic_base_calls_only_k9_bulk_primary(monkeypatch):
    model = replace(_one_class_model(), lanes=16)
    selection = _selection(
        indices=(0,) * 16,
        service=(16,),
        score_exact=False,
        canonical_exact=False,
    )
    composite = SimpleNamespace(base_selection=selection, selection=selection)
    observed = []

    def bulk(options, queue, lanes, budget):
        observed.append((options, tuple(queue), lanes, budget))
        return composite

    def forbidden(*_args, **_kwargs):
        raise AssertionError("R16 frontier inherited the old conditional K144 call")

    monkeypatch.setattr(
        diagnostic, "select_canonical_score_tie_bulk_substitution_frame", bulk
    )
    monkeypatch.setattr(diagnostic, "bulk_total_work_row", lambda _row: _work(1))
    monkeypatch.setattr(diagnostic.source, "_bulk_relation", lambda *args: True)
    monkeypatch.setattr(diagnostic.source, "_certificate_valid", lambda *args: True)
    monkeypatch.setattr(diagnostic, "_selection_certificate_valid", lambda *args: True)
    monkeypatch.setattr(diagnostic.source, "_candidate_call", forbidden)

    row = diagnostic._diagnostic_base_call(model, (16,))

    assert [(item[1], item[2], item[3]) for item in observed] == [((16,), 16, 9)]
    assert row["lp_node_budget"] == 9
    assert row["work"] == _work(1)
    assert row["method"] == "r16_k9_bulk_primary_without_old_k144_rescue"


def test_failure_labels_do_not_gate_state_evaluation(monkeypatch):
    model = _one_class_model()
    exact = _selection()
    record = {
        "lanes": 8,
        "queue": (1,),
        "observed_candidate_call": {
            "selection": exact,
            "work": _work(1),
            "valid": True,
            "method_relation": True,
        },
        "occurrences_by_case": {0.5: 1},
        "occurrences_by_tape": {(0.5, 12041): 1},
        "first_occurrence": {"load": 0.5, "seed": 12041, "frame": 0},
    }
    shadow_calls = []
    rescue_calls = []

    def shadow(_model, _queue):
        shadow_calls.append(tuple(_queue))
        return exact, _work(3), True

    monkeypatch.setattr(diagnostic.source, "_reference_call", shadow)
    monkeypatch.setattr(
        diagnostic,
        "_diagnostic_base_call",
        lambda _model, _queue: {
            "selection": exact,
            "work": _work(1),
            "valid": True,
            "method_relation": True,
            "method": "fixture_primary",
            "lp_node_budget": 48,
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "_rescue_call",
        lambda *_args: rescue_calls.append(_args),
    )
    monkeypatch.setattr(diagnostic, "_selection_certificate_valid", lambda *args: True)
    row = diagnostic._evaluate_unique_state(model, record, lane_eta=0.1)

    assert shadow_calls == [(1,)]
    assert rescue_calls == []
    assert row["failure_reasons"] == []
    assert len(row["variants"]) == len(diagnostic.TRIGGER_RULES) * len(
        diagnostic.RESCUE_BUDGETS
    )


def test_occurrence_weighted_work_and_strict_gates_are_separate_from_dedup_calls():
    variant = {
        "triggered": True,
        "adopted": "rescue",
        "certificate_valid": True,
        "same_state_shadow_action_match": True,
        "score_exact": True,
        "canonical_exact": True,
        "within_lane_min_eta": True,
        "regret_bounded_by_epsilon": True,
        "epsilon": 0.0,
        "actual_normalized_regret": 0.0,
        "work": _work(2),
        "shadow_work": _work(3),
    }
    row = {
        "lanes": 8,
        "total_occurrences": 4,
        "occurrences_by_case": {0.5: 4},
        "variants": {
            (rule, budget): dict(variant)
            for rule in diagnostic.TRIGGER_RULES
            for budget in diagnostic.RESCUE_BUDGETS
        },
    }
    summary = diagnostic._scope_summary(
        [row],
        "epsilon_near_eta_or_canonical_inexact",
        96,
        lambda item: item["occurrences_by_case"].get(0.5, 0),
        scope={"scope": "case", "lanes": 8, "load": 0.5},
    )

    assert summary["unique_state_count"] == 1
    assert summary["state_occurrence_count"] == 4
    assert summary["candidate_occurrence_weighted_work"] == _work(8)
    assert summary["same_state_k4096_occurrence_weighted_work"] == _work(12)
    comparison = summary["work_comparison"]
    assert comparison["work_gate_fields_strictly_lower"] is True
    assert comparison["all_positive_shadow_fields_strictly_lower"] is True


def test_frontier_certificate_check_covers_base_and_every_budget_variant():
    row = {
        "frontier_base": {"current_contract_valid": True},
        "variants": {
            (rule, budget): {"certificate_valid": True}
            for rule in diagnostic.TRIGGER_RULES
            for budget in diagnostic.RESCUE_BUDGETS
        },
    }
    assert diagnostic._all_frontier_certificates_valid([row]) is True

    row["variants"][(diagnostic.TRIGGER_RULES[0], 576)][
        "certificate_valid"
    ] = False
    assert diagnostic._all_frontier_certificates_valid([row]) is False
    row["variants"][(diagnostic.TRIGGER_RULES[0], 576)][
        "certificate_valid"
    ] = True
    row["frontier_base"]["current_contract_valid"] = False
    assert diagnostic._all_frontier_certificates_valid([row]) is False


def test_protocol_tamper_is_rejected_before_model_rng_or_selector_work(
    tmp_path, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(diagnostic.build_registration()),
    }
    protocol["diagnostic_contract"]["rescue_budgets_by_lane"]["8"][0] = 95
    registration_path = _write(tmp_path / "protocol.json", protocol)
    fixed_path = _write(tmp_path / "fixed.json", diagnostic._current_fixed_sources())
    output = tmp_path / "result.json"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tampered protocol reached execution work")

    monkeypatch.setattr(diagnostic.source.baseline, "_model_from_snapshot", forbidden)
    monkeypatch.setattr(diagnostic.source, "poisson_arrivals", forbidden)
    monkeypatch.setattr(diagnostic.source, "_select_candidate", forbidden)
    monkeypatch.setattr(diagnostic.source, "_reference_call", forbidden)
    monkeypatch.setattr(diagnostic, "_rescue_call", forbidden)

    with pytest.raises(
        diagnostic.PostResultDevelopmentRefusal,
        match="protocol changed",
    ):
        diagnostic.run_registered(registration_path, fixed_path, output)
    assert not output.exists()


def test_postflight_rereads_and_rejects_registration_changed_during_run(tmp_path):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(diagnostic.build_registration()),
    }
    registration_path = _write(tmp_path / "protocol.json", protocol)
    fixed_path = _write(tmp_path / "fixed.json", diagnostic._current_fixed_sources())
    paths = protocol["source_paths"]

    diagnostic._postflight_revalidate(
        registration_path, fixed_path, protocol, paths
    )
    changed = copy.deepcopy(protocol)
    changed["diagnostic_contract"]["arrival_horizon_s"] = 601.0
    _write(registration_path, changed)

    with pytest.raises(
        diagnostic.PostResultDevelopmentRefusal,
        match="registration changed during execution",
    ):
        diagnostic._postflight_revalidate(
            registration_path, fixed_path, protocol, paths
        )


def test_run_cli_prints_only_small_result_summary(tmp_path, monkeypatch, capsys):
    output = tmp_path / "large-result.json"
    fake = {
        "status": "COMPLETE",
        "scientific_decision": diagnostic.COMPLETE_DECISION,
        "unique_state_count": 12,
        "failure_state_count": 3,
        "state_occurrence_count": 40,
        "timing": {"wall_s": 1.25},
        "unique_states": ["large-payload-marker"],
        "frontier": {"large": "large-payload-marker"},
    }
    monkeypatch.setattr(diagnostic, "run_registered", lambda *args: fake)

    diagnostic.main(
        [
            "--run",
            "--registration",
            str(tmp_path / "protocol.json"),
            "--fixed-sources",
            str(tmp_path / "fixed.json"),
            "--output",
            str(output),
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert printed == {
        "status": "COMPLETE",
        "scientific_decision": diagnostic.COMPLETE_DECISION,
        "output": str(output.resolve()),
        "unique_state_count": 12,
        "failure_state_count": 3,
        "state_occurrence_count": 40,
        "timing": {"wall_s": 1.25},
    }
    assert "large-payload-marker" not in json.dumps(printed)


def test_no_formal_development_artifact_was_registered_or_run():
    assert not diagnostic.DEFAULT_REGISTRATION.exists()
    assert not diagnostic.DEFAULT_FIXED_SOURCES.exists()
    assert not diagnostic.DEFAULT_RESULT.exists()
