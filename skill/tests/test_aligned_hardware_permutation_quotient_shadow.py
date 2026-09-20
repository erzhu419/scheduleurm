"""Focused tests for the registered lane-permutation quotient shadow."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_permutation_quotient_shadow as shadow


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    """Register once while proving registration cannot execute the selector."""
    directory = tmp_path_factory.mktemp("permutation_quotient_registration")
    output = directory / "shadow.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("registration must not execute the quotient selector")

    patcher = pytest.MonkeyPatch()
    patcher.setattr(shadow, "select_permutation_quotient_frame", forbidden)
    try:
        protocol = shadow.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def test_registration_reuses_the_existing_roster_without_copying_or_running(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = shadow._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")
    roster = json.loads(shadow.REGISTERED_STATES.read_text())

    assert result_path == output and not output.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert protocol["registered_state_roster"] == shadow.REGISTERED_STATES.name
    assert protocol["state_count"] == roster["state_count"] == shadow.STATE_COUNT
    assert len(roster["states"]) == shadow.STATE_COUNT == 5003
    assert len({tuple(row["queue"]) for row in roster["states"]}) == 5003
    assert len(protocol["source_population_cells"]) == 18
    assert protocol["rule"] == shadow.RULE
    assert protocol["advance_gate"] == shadow.ADVANCE_GATE


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    for source, target in zip(
        shadow._paths(source_output)[1:],
        shadow._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


@pytest.mark.parametrize("tamper", ("protocol", "rule", "roster", "source"))
def test_tampering_fails_before_any_selector_call(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, sources_path = shadow._paths(output)

    if tamper == "protocol":
        payload = json.loads(protocol_path.read_text())
        payload["state_count"] += 1
        protocol_path.write_text(json.dumps(payload))
        message = "registered permutation-quotient roster or constants changed"
    elif tamper == "rule":
        payload = json.loads(protocol_path.read_text())
        payload["rule"]["equivalence_key"] = "tampered"
        protocol_path.write_text(json.dumps(payload))
        message = "registered permutation-quotient protocol rule changed"
    elif tamper == "roster":
        roster_path = tmp_path / shadow.REGISTERED_STATES.name
        payload = json.loads(shadow.REGISTERED_STATES.read_text())
        payload["states"][0]["queue"][0] += 1
        roster_path.write_text(json.dumps(payload))
        monkeypatch.setattr(shadow, "REGISTERED_STATES", roster_path)
        message = "registered permutation-quotient states changed"
    else:
        payload = json.loads(sources_path.read_text())
        name = "algorithm/permutation_quotient_frame_candidates.py"
        payload[name] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "registered permutation-quotient sources changed"

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("invalid registered input reached selector execution")

    monkeypatch.setattr(shadow, "select_permutation_quotient_frame", forbidden)
    with pytest.raises(ValueError, match=message):
        shadow.run_registered(output)
    assert calls == []
    assert not output.exists()


@dataclass(frozen=True)
class _Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _option(requirements, duration, completion_offsets=None):
    return _Option(
        requirements=requirements,
        duration=duration,
        completion_offsets=(
            completion_offsets
            if completion_offsets is not None
            else tuple((duration,) * count for count in requirements)
        ),
    )


def test_crafted_state_executes_a_realizable_changed_physical_action(
    monkeypatch,
):
    model = SimpleNamespace(
        lanes=2,
        workloads=("a", "b"),
        options=(
            _option((2, 1), 5.0, ((1.0, 5.0), (4.0,))),
            _option((0, 1), 5.0, ((), (2.0,))),
            _option((3, 3), 3.0),
        ),
    )
    monkeypatch.setattr(shadow, "SMALL_BUDGET", 7)
    monkeypatch.setattr(shadow, "EXACT_QUOTIENT_BUDGET", 10)
    monkeypatch.setattr(shadow, "ORDERED_ACTION_COUNT", 16)
    monkeypatch.setattr(shadow, "QUOTIENT_ORBIT_COUNT", 10)
    state = {
        "queue": [2, 5],
        "population_visits": [],
        "legacy_k26": {
            "option_indices": [1, 1],
            "service": [0, 2],
            "duration": 5.0,
            "score": 2.0,
            "upper_bound": 19.0 / 3.0,
            "gap": 13.0 / 3.0,
            "epsilon": 13.0 / 21.0,
            "attempts": 7,
            "exact": False,
        },
        "support_reference_score": 2.8,
        "legacy_first_exact_score_attempt": 13,
    }

    evaluated = shadow._evaluate_state(model, 0, state)
    selected = evaluated["variant_k26"]
    exact = evaluated["variant_k55"]

    assert selected["option_indices"] == (0, 1)
    assert selected["service"] == (2, 2)
    assert selected["score"] == pytest.approx(2.8)
    assert selected["cache_hits"] > 0
    assert evaluated["support_outcome"] == "improved"
    assert evaluated["selected_physical_signature_changed_from_legacy_k26"]
    assert evaluated["selection_bound_and_work_checks_pass"]
    assert exact["exact"] and exact["score"] == pytest.approx(2.8)
    assert evaluated["variant_k55_support_matches_registered_reference"]

    forward, _ = shadow._physical_signature(
        model, (0, 1), selected["service"], selected["duration"]
    )
    reversed_lanes, _ = shadow._physical_signature(
        model, (1, 0), selected["service"], selected["duration"]
    )
    legacy, _ = shadow._physical_signature(
        model,
        state["legacy_k26"]["option_indices"],
        state["legacy_k26"]["service"],
        state["legacy_k26"]["duration"],
    )
    assert forward == reversed_lanes
    assert forward != legacy


def _population_rows(*, support_improvement=True, work_improvement=True):
    rows = []
    for load in shadow.LOADS:
        rows.append({
            "load": load,
            "legacy": {
                "mean_finite_budget_support_regret": 1.0,
                "strict_support_loss_frame_count": 10,
                "mean_certified_gap": 0.4,
                "mean_candidate_attempts": 26.0,
                "exact_support_score_frame_fraction": 0.8,
                "exact_certified_frame_fraction": 0.7,
            },
            "variant": {
                "mean_finite_budget_support_regret": (
                    0.5 if support_improvement else 1.0
                ),
                "strict_support_loss_frame_count": (
                    8 if support_improvement else 10
                ),
                "mean_certified_gap": 0.3 if work_improvement else 0.4,
                "mean_candidate_attempts": 25.0 if work_improvement else 26.0,
                "exact_support_score_frame_fraction": (
                    0.85 if support_improvement else 0.8
                ),
                "exact_certified_frame_fraction": (
                    0.75 if work_improvement else 0.7
                ),
            },
        })
    return rows


def _evaluation(*, changed=False, support_delta=0.1):
    return {
        "support_score_delta_over_legacy_k26": support_delta,
        "upper_bound_delta_over_legacy_k26": -0.1,
        "gap_delta_over_legacy_k26": -0.1,
        "raw_lane_witness_changed_from_legacy_k26": False,
        "selected_physical_signature_changed_from_legacy_k26": changed,
    }


def _visited_state():
    return {"population_visits": [[0.5, 41, shadow.SMALL_BUDGET, 1, 1.0]]}


def test_primary_and_sensitivity_gates_are_both_vetoes():
    primary = _population_rows()
    sensitivity = _population_rows()
    evaluations = [_evaluation(changed=True)]
    checks = shadow._advance_checks(primary, sensitivity, evaluations, True)
    assert all(checks.values())

    primary[1]["variant"]["mean_certified_gap"] = 0.41
    checks = shadow._advance_checks(primary, sensitivity, evaluations, True)
    assert not checks["primary_mean_certified_gap_nonworse_at_every_load"]

    primary = _population_rows()
    sensitivity[2]["variant"]["exact_support_score_frame_fraction"] = 0.79
    checks = shadow._advance_checks(primary, sensitivity, evaluations, True)
    assert not checks["sensitivity_exact_support_fraction_nonworse_at_every_load"]


def test_decision_has_invalid_advance_efficiency_only_and_closed_branches():
    primary = _population_rows()
    sensitivity = _population_rows()
    evaluations = [_evaluation(changed=True)]
    states = [_visited_state()]
    passing = shadow._advance_checks(primary, sensitivity, evaluations, True)
    assert shadow._decision(passing, primary, evaluations, states) == (
        "ADVANCE_TO_AUTHORED_TRANSITION_SCREEN"
    )

    invalid = dict(passing)
    invalid["execution_passed"] = False
    assert shadow._decision(invalid, primary, evaluations, states) == (
        "INVALID_EXECUTION"
    )

    efficiency_primary = _population_rows(support_improvement=False)
    unchanged = [_evaluation(changed=False, support_delta=0.0)]
    efficiency = shadow._advance_checks(
        efficiency_primary,
        _population_rows(support_improvement=False),
        unchanged,
        True,
    )
    assert not efficiency[
        "at_least_one_primary_load_strictly_improves_support"
    ]
    assert efficiency[
        "at_least_one_primary_load_strictly_improves_certificate_or_attempts"
    ]
    assert shadow._decision(
        efficiency, efficiency_primary, unchanged, states
    ) == "EFFICIENCY_ONLY_STOP"

    closed = dict(passing)
    closed["primary_mean_certified_gap_nonworse_at_every_load"] = False
    assert shadow._decision(closed, primary, evaluations, states) == (
        "CLOSE_WITHOUT_TRANSITION_SCREEN"
    )
