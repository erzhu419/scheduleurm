"""Focused tests for the registered two-lane anchor-star state shadow."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_anchor_star_shadow as shadow
from algorithm.experiments.aligned_hardware_frame_model import load_model


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    """Register once while proving registration cannot execute the selector."""
    directory = tmp_path_factory.mktemp("anchor_star_registration")
    output = directory / "shadow.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("registration must not execute the anchor-star selector")

    patcher = pytest.MonkeyPatch()
    patcher.setattr(shadow, "select_anchor_star_frame", forbidden)
    try:
        protocol = shadow.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def test_registration_reuses_roster_and_freezes_100_action_k82_structure(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = shadow._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")
    structure = protocol["anchor_star_structure"]

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert "states" not in protocol
    assert protocol["registered_state_roster"] == shadow.REGISTERED_STATES.name
    assert protocol["state_count"] == 5003
    assert len(protocol["known_quotient_action_pair_roster"]) == 16
    assert shadow._structure_is_expected(structure)
    assert structure == {
        "option_count": 9,
        "lanes": 2,
        "local_choice_count_with_idle": 10,
        "represented_ordered_action_count": 100,
        "homogeneous_seed_attempts": 10,
        "directed_heterogeneous_pair_attempts": 72,
        "legacy_registered_exact_budget": 120,
        "anchor_star_exhaustive_support_candidate_cap": 82,
    }
    assert protocol["small_checkpoint_budget"] == 26
    assert protocol["anchor_star_exact_budget"] == 82


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    for source, target in zip(
        shadow._paths(source_output)[1:],
        shadow._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


@pytest.mark.parametrize(
    "tamper",
    ("protocol", "source", "roster", "closed_quotient"),
)
def test_tampering_fails_before_the_first_selector_call(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, sources_path = shadow._paths(output)

    if tamper == "protocol":
        payload = json.loads(protocol_path.read_text())
        payload["small_checkpoint_budget"] += 1
        protocol_path.write_text(json.dumps(payload))
        message = "registered anchor-star roster or constants changed"
    elif tamper == "source":
        payload = json.loads(sources_path.read_text())
        payload[shadow.NEW_SOURCES[0]] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "registered anchor-star sources changed"
    elif tamper == "roster":
        roster_path = tmp_path / shadow.REGISTERED_STATES.name
        payload = json.loads(shadow.REGISTERED_STATES.read_text())
        payload["states"][0]["queue"][0] += 1
        roster_path.write_text(json.dumps(payload))
        monkeypatch.setattr(shadow, "REGISTERED_STATES", roster_path)
        monkeypatch.setattr(shadow.base, "REGISTERED_STATES", roster_path)
        message = "registered permutation-quotient states changed"
    else:
        transition_path = tmp_path / shadow.QUOTIENT_TRANSITION.name
        payload = json.loads(shadow.QUOTIENT_TRANSITION.read_text())
        payload["scientific_decision"] = "TAMPERED"
        transition_path.write_text(json.dumps(payload))
        monkeypatch.setattr(shadow, "QUOTIENT_TRANSITION", transition_path)
        message = "completed quotient transition is unavailable or changed"

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("tampered inputs reached selector execution")

    monkeypatch.setattr(shadow, "select_anchor_star_frame", forbidden)
    with pytest.raises(ValueError, match=message):
        shadow.run_registered(output)
    assert calls == []
    assert not output.exists()


@dataclass(frozen=True)
class _Option:
    action_id: str
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def test_evaluate_state_requires_exact_support_without_exact_raw_tie_identity(
    monkeypatch,
):
    model = SimpleNamespace(
        lanes=2,
        workloads=("a", "b"),
        options=(
            _Option("fast-a", (1, 0), 2.0, ((2.0,), ())),
            _Option("slow-a", (1, 0), 4.0, ((4.0,), ())),
            _Option("b", (0, 1), 2.0, ((), (2.0,))),
        ),
    )
    state = {
        "queue": [1, 1],
        "population_visits": [],
        "legacy_k26": {
            "option_indices": [2, 0],
            "service": [1, 1],
            "duration": 2.0,
            "score": 1.0,
            "upper_bound": 1.0,
            "gap": 0.0,
            "epsilon": 0.0,
            "attempts": 7,
            "exact": True,
        },
        "support_reference_score": 1.0,
    }
    source_exact = {
        "option_indices": [2, 0],
        "service": [1, 1],
        "duration": 2.0,
        "score": 1.0,
        "upper_bound": 1.0,
        "gap": 0.0,
        "epsilon": 0.0,
        "exact": True,
    }

    signature_calls = []
    original_action_signatures = shadow._action_signatures

    def counted_action_signatures(local_model, option_indices):
        signature_calls.append(tuple(option_indices))
        return original_action_signatures(local_model, option_indices)

    monkeypatch.setattr(
        shadow, "_action_signatures", counted_action_signatures
    )

    evaluated = shadow._evaluate_state(
        model, [], 9, state, source_exact
    )

    assert signature_calls == [(2, 0), (0, 2), (0, 2)]
    assert evaluated["variant_k82"]["option_indices"] == (0, 2)
    assert evaluated["variant_k82"]["exact"]
    assert evaluated["variant_k82_support_matches_registered_reference"]
    assert not evaluated["variant_k82_selection_matches_legacy_k120"]
    assert evaluated["selection_bound_mapping_and_work_checks_pass"]
    assert evaluated["raw_lane_witness_changed_from_legacy_k26"]
    assert not evaluated[
        "selected_physical_signature_changed_from_legacy_k26"
    ]
    assert not evaluated[
        "fifo_execution_signature_changed_from_legacy_k26"
    ]


def _population_rows(*, support_gain=True, work_gain=True):
    rows = []
    for load in shadow.LOADS:
        legacy = {
            "mean_finite_budget_support_regret": 1.0,
            "strict_support_loss_frame_count": 10,
            "mean_certified_gap": 0.4,
            "mean_candidate_attempts": 26.0,
            "exact_support_score_frame_fraction": 0.8,
            "exact_certified_frame_fraction": 0.7,
        }
        variant = {
            "mean_finite_budget_support_regret": 0.5 if support_gain else 1.0,
            "strict_support_loss_frame_count": 8 if support_gain else 10,
            "mean_certified_gap": 0.3 if work_gain else 0.4,
            "mean_candidate_attempts": 25.0 if work_gain else 26.0,
            "exact_support_score_frame_fraction": 0.85 if support_gain else 0.8,
            "exact_certified_frame_fraction": 0.75 if work_gain else 0.7,
        }
        rows.append({"load": load, "legacy": legacy, "variant": variant})
    return rows


def _evaluation(*, physical=True, raw=True, fifo=True, known=False):
    return {
        "support_score_delta_over_legacy_k26": 0.1,
        "upper_bound_delta_over_legacy_k26": -0.1,
        "gap_delta_over_legacy_k26": -0.1,
        "raw_lane_witness_changed_from_legacy_k26": raw,
        "selected_physical_signature_changed_from_legacy_k26": physical,
        "fifo_execution_signature_changed_from_legacy_k26": fifo,
        "known_quotient_physical_pair": known,
        "known_quotient_fifo_pair": known,
        "known_quotient_physical_and_raw_pair": known and raw,
    }


@pytest.mark.parametrize("population", ("primary", "sensitivity"))
@pytest.mark.parametrize(
    ("metric", "regressed", "check_suffix"),
    (
        (
            "mean_finite_budget_support_regret",
            1.1,
            "mean_support_regret_nonworse_at_every_load",
        ),
        (
            "strict_support_loss_frame_count",
            11,
            "strict_support_loss_count_nonworse_at_every_load",
        ),
        (
            "mean_certified_gap",
            0.41,
            "mean_certified_gap_nonworse_at_every_load",
        ),
        (
            "mean_candidate_attempts",
            26.1,
            "mean_candidate_attempts_nonworse_at_every_load",
        ),
        (
            "exact_support_score_frame_fraction",
            0.79,
            "exact_support_fraction_nonworse_at_every_load",
        ),
        (
            "exact_certified_frame_fraction",
            0.69,
            "exact_certified_fraction_nonworse_at_every_load",
        ),
    ),
)
def test_each_population_metric_is_an_individual_regression_veto(
    population, metric, regressed, check_suffix,
):
    primary = _population_rows()
    sensitivity = _population_rows()
    selected = primary if population == "primary" else sensitivity
    selected[1]["variant"][metric] = regressed
    evaluations = [_evaluation()]

    checks = shadow._advance_checks(
        primary, sensitivity, evaluations, execution_passed=True
    )

    assert not checks[f"{population}_{check_suffix}"]
    assert shadow._decision(checks, evaluations) == "REGRESSION_STOP"


def test_decision_uses_fifo_execution_to_stop_or_advance_action_sets():
    primary = _population_rows()
    sensitivity = _population_rows()

    novel = [_evaluation(known=False)]
    passing = shadow._advance_checks(primary, sensitivity, novel, True)
    assert all(passing.values())
    assert shadow._decision(passing, novel) == "ADVANCE_AUTHORED_TRANSITION"

    invalid = dict(passing)
    invalid["execution_passed"] = False
    assert shadow._decision(invalid, novel) == "INVALID_EXECUTION"

    regressed = dict(passing)
    regressed["no_registered_state_has_lower_k26_support"] = False
    assert shadow._decision(regressed, novel) == "REGRESSION_STOP"

    no_work_gain = shadow._advance_checks(
        _population_rows(work_gain=False),
        _population_rows(work_gain=False),
        novel,
        True,
    )
    assert shadow._decision(no_work_gain, novel) == "NO_GAIN_STOP"

    raw_fifo_equivalent = [
        _evaluation(physical=False, raw=True, fifo=False, known=False)
    ]
    assert (
        shadow._decision(passing, raw_fifo_equivalent)
        == "KNOWN_ACTION_STOP"
    )

    raw_fifo_effective = [
        _evaluation(known=True),
        _evaluation(physical=False, raw=True, fifo=True, known=False),
    ]
    assert (
        shadow._decision(passing, raw_fifo_effective)
        == "ADVANCE_AUTHORED_TRANSITION"
    )

    all_known = [_evaluation(known=True), _evaluation(known=True)]
    assert shadow._decision(passing, all_known) == "KNOWN_ACTION_STOP"

    mixed = [_evaluation(known=True), _evaluation(known=False)]
    assert shadow._decision(passing, mixed) == "ADVANCE_AUTHORED_TRANSITION"

    unchanged = [
        _evaluation(physical=False, raw=False, fifo=False, known=False)
    ]
    assert shadow._decision(passing, unchanged) == "NO_GAIN_STOP"


def test_fifo_pair_identity_distinguishes_tail_effective_raw_lane_order():
    signature_a = {
        "service": [1, 1],
        "duration": 2.0,
        "completion_offset_multisets": [[2.0], [2.0]],
    }
    signature_b = {
        "service": [2, 1],
        "duration": 3.0,
        "completion_offset_multisets": [[1.0, 3.0], [2.0]],
    }
    fifo_a = {
        "service": [1, 1],
        "duration": 2.0,
        "ordered_completion_offsets": [[2.0], [2.0]],
    }
    fifo_b = {
        "service": [2, 1],
        "duration": 3.0,
        "ordered_completion_offsets": [[1.0, 3.0], [2.0]],
    }
    fifo_b_reordered = {
        "service": [2, 1],
        "duration": 3.0,
        "ordered_completion_offsets": [[3.0, 1.0], [2.0]],
    }
    registered = shadow._pair_record(
        12,
        (2, 2),
        (0, 1),
        (2, 3),
        signature_a,
        signature_b,
        fifo_a,
        fifo_b,
    )
    same_physical_new_raw = shadow._pair_record(
        12,
        (2, 2),
        (1, 0),
        (3, 2),
        signature_a,
        signature_b,
        fifo_a,
        fifo_b_reordered,
    )
    same_fifo_new_raw = shadow._pair_record(
        12,
        (2, 2),
        (1, 0),
        (3, 2),
        signature_a,
        signature_b,
        fifo_a,
        fifo_b,
    )

    assert registered != same_physical_new_raw
    assert shadow._physical_pair(registered) == shadow._physical_pair(
        same_physical_new_raw
    )
    assert shadow._fifo_pair(registered) != shadow._fifo_pair(
        same_physical_new_raw
    )
    assert registered != same_fifo_new_raw
    assert shadow._fifo_pair(registered) == shadow._fifo_pair(
        same_fifo_new_raw
    )


def test_real_raw_lane_fixture_has_same_physics_but_different_fifo_execution():
    model = load_model()
    forward_physical, forward_fifo, _ = shadow._action_signatures(
        model, (0, 7)
    )
    reverse_physical, reverse_fifo, _ = shadow._action_signatures(
        model, (7, 0)
    )

    assert forward_physical == reverse_physical
    assert forward_fifo != reverse_fifo
    assert forward_fifo["ordered_completion_offsets"][2] == [
        model.options[0].completion_offsets[2][0],
        *model.options[7].completion_offsets[2],
    ]
    assert reverse_fifo["ordered_completion_offsets"][2] == [
        *model.options[7].completion_offsets[2],
        model.options[0].completion_offsets[2][0],
    ]


def test_disjoint_class_raw_swap_is_fifo_execution_equivalent():
    model = load_model()
    forward_physical, forward_fifo, _ = shadow._action_signatures(
        model, (3, 8)
    )
    reverse_physical, reverse_fifo, _ = shadow._action_signatures(
        model, (8, 3)
    )

    assert forward_physical == reverse_physical
    assert forward_fifo == reverse_fifo
