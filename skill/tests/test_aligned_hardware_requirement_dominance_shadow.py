"""Focused tests for the registered requirement-dominance state shadow."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
import algorithm.experiments.aligned_hardware_requirement_dominance_shadow as shadow


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    """Register once while proving registration cannot execute the selector."""
    directory = tmp_path_factory.mktemp("requirement_dominance_registration")
    output = directory / "shadow.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("registration must not execute the requirement-dominance selector")

    patcher = pytest.MonkeyPatch()
    patcher.setattr(shadow, "select_requirement_dominance_frame", forbidden)
    try:
        protocol = shadow.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def test_registration_reuses_roster_and_freezes_structural_reduction(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = shadow._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")
    reduction = protocol["catalogue_reduction"]

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert "states" not in protocol
    assert protocol["registered_state_roster"] == shadow.REGISTERED_STATES.name
    assert protocol["state_count"] == 5003
    assert len(protocol["known_quotient_physical_pair_roster"]) == 16
    assert shadow._reduction_is_expected(reduction)
    assert reduction["original_option_count"] == 9
    assert reduction["retained_option_count"] == 8
    assert reduction["removed_original_indices"] == [1]
    assert reduction["original_ordered_action_count"] == 100
    assert reduction["reduced_ordered_action_count"] == 81
    assert reduction["legacy_registered_exact_budget"] == 120
    assert reduction["reduced_exhaustive_budget"] == 99


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
        message = "registered requirement-dominance roster or constants changed"
    elif tamper == "source":
        payload = json.loads(sources_path.read_text())
        payload[shadow.NEW_SOURCES[0]] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "registered requirement-dominance sources changed"
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

    monkeypatch.setattr(shadow, "select_requirement_dominance_frame", forbidden)
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


def test_evaluate_state_requires_exact_support_without_exact_raw_tie_identity():
    model = SimpleNamespace(
        lanes=2,
        workloads=("a", "b"),
        options=(
            _Option("fast-a", (1, 0), 2.0, ((2.0,), ())),
            _Option("slow-a", (1, 0), 4.0, ((4.0,), ())),
            _Option("b", (0, 1), 2.0, ((), (2.0,))),
        ),
    )
    reduction = shadow._reduction_snapshot(model)
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

    evaluated = shadow._evaluate_state(
        model, reduction, [], 9, state, source_exact
    )

    assert evaluated["variant_k99"]["option_indices"] == (0, 2)
    assert evaluated["variant_k99_support_matches_registered_reference"]
    assert not evaluated["variant_k99_selection_matches_legacy_k120"]
    assert evaluated["selection_bound_mapping_and_work_checks_pass"]
    assert evaluated["raw_lane_witness_changed_from_legacy_k26"]
    assert not evaluated[
        "selected_physical_signature_changed_from_legacy_k26"
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


def _evaluation(*, changed=True, known=False):
    return {
        "support_score_delta_over_legacy_k26": 0.1 if changed else 0.0,
        "upper_bound_delta_over_legacy_k26": -0.1,
        "gap_delta_over_legacy_k26": -0.1,
        "raw_lane_witness_changed_from_legacy_k26": changed,
        "selected_physical_signature_changed_from_legacy_k26": changed,
        "known_quotient_physical_pair": known,
        "known_quotient_physical_and_raw_pair": known,
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

    checks = shadow._advance_checks(
        primary, sensitivity, [_evaluation()], execution_passed=True
    )

    assert not checks[f"{population}_{check_suffix}"]
    assert shadow._decision(checks, [_evaluation()]) == "REGRESSION_STOP"


def _physical_only_known_evaluation():
    model = load_model()
    queue = (2, 2, 5)
    state_index = 279
    legacy = select_budgeted_frame(
        model.options, queue, model.lanes, shadow.SMALL_BUDGET
    )
    source_exact = select_budgeted_frame(
        model.options, queue, model.lanes, shadow.LEGACY_EXACT_BUDGET
    )
    legacy_signature, _ = shadow._physical_signature(
        model, legacy.option_indices
    )
    known_variant_indices = (6, 7)
    variant_signature, _ = shadow._physical_signature(
        model, known_variant_indices
    )
    known_pairs = [shadow._pair_record(
        state_index,
        queue,
        legacy.option_indices,
        known_variant_indices,
        legacy_signature,
        variant_signature,
    )]
    state = {
        "queue": list(queue),
        "population_visits": [],
        "legacy_k26": asdict(legacy),
        "support_reference_score": source_exact.score,
    }
    return shadow._evaluate_state(
        model,
        shadow._reduction_snapshot(model),
        known_pairs,
        state_index,
        state,
        asdict(source_exact),
    )


def test_all_decision_branches_include_physical_only_known_pair_veto():
    primary = _population_rows()
    sensitivity = _population_rows()
    novel = [_evaluation()]
    passing = shadow._advance_checks(primary, sensitivity, novel, True)
    assert all(passing.values())
    assert shadow._decision(passing, novel) == (
        "ADVANCE_NEW_ACTIONS_TO_AUTHORED_TRANSITION_SCREEN"
    )

    invalid = dict(passing)
    invalid["execution_passed"] = False
    assert shadow._decision(invalid, novel) == "INVALID_EXECUTION"

    regressed = dict(passing)
    regressed["no_registered_state_has_lower_k26_support"] = False
    assert shadow._decision(regressed, novel) == "REGRESSION_STOP"

    efficiency_primary = _population_rows(
        support_gain=False, work_gain=True
    )
    unchanged = [_evaluation(changed=False)]
    efficiency = shadow._advance_checks(
        efficiency_primary,
        _population_rows(support_gain=False, work_gain=True),
        unchanged,
        True,
    )
    assert shadow._decision(efficiency, unchanged) == "EFFICIENCY_ONLY_STOP"

    support_only = shadow._advance_checks(
        _population_rows(support_gain=True, work_gain=False),
        _population_rows(support_gain=True, work_gain=False),
        novel,
        True,
    )
    assert shadow._decision(support_only, novel) == "NO_GAIN_STOP"

    known = _physical_only_known_evaluation()
    assert known["selection_bound_mapping_and_work_checks_pass"]
    assert known["variant_k26"]["option_indices"] == (7, 6)
    assert known["known_quotient_physical_pair"]
    assert not known["known_quotient_physical_and_raw_pair"]
    assert shadow._decision(passing, [known]) == "KNOWN_QUOTIENT_ACTION_STOP"
