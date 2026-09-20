"""Contracts for the canonical B&B requirement-duration quotient wrapper."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import pytest

import algorithm.canonical_count_space_bnb_requirement_dominance_frame_candidates as quotient
from algorithm.canonical_count_space_bnb_frame_candidates import (
    CanonicalCountSpaceBnbSelection,
    select_canonical_count_space_bnb_frame,
    select_canonical_count_space_bnb_score_exact_tie_continuation_frame,
)
from algorithm.canonical_count_space_bnb_requirement_dominance_frame_candidates import (
    prepare_canonical_count_space_bnb_requirement_dominance_catalogue,
    select_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame,
    select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _trigger_catalogue() -> tuple[Option, ...]:
    # Removing option 2 leaves the two-option K2 -> K3 canonical-tie case.
    return (
        Option((2, 1), 2.0),
        Option((1, 0), 2.0),
        Option((1, 0), 3.0),
    )


def _assert_full_selection_reconstructs(selection, options, queue, lanes):
    assert len(selection.option_counts) == len(options)
    assert len(selection.option_indices) == lanes
    counts = tuple(selection.option_indices.count(index) for index in range(len(options)))
    assert selection.option_counts == counts
    service = tuple(
        sum(
            options[index].requirements[coordinate]
            for index in selection.option_indices
            if index is not None
        )
        for coordinate in range(len(queue))
    )
    duration = max((
        1.0,
        *(
            options[index].duration
            for index in selection.option_indices
            if index is not None
        ),
    ))
    assert selection.service == service
    assert selection.duration == duration
    assert selection.score == pytest.approx(
        sum(waiting * done for waiting, done in zip(queue, service, strict=True))
        / duration
    )


def test_registered_catalogue_is_prepared_once_as_nine_to_eight_with_lift_proof(
    monkeypatch,
):
    model = load_model()
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        model.options
    )
    metadata = prepared.reduction
    proof = prepared.certificate_lift_proof

    assert metadata.original_option_count == 9
    assert metadata.retained_option_count == 8
    assert metadata.removed_option_count == 1
    assert metadata.retained_original_indices == (0, 2, 3, 4, 5, 6, 7, 8)
    assert metadata.removed_original_indices == (1,)
    assert prepared.reduced_options == tuple(
        model.options[index] for index in metadata.retained_original_indices
    )
    assert proof.removed_to_dominator_original_indices == ((1, 0),)
    assert all((
        proof.retained_index_map_strictly_increasing,
        proof.replacement_requirements_equal,
        proof.replacement_effective_duration_strictly_smaller,
        proof.replacement_original_index_strictly_smaller,
        proof.full_catalogue_score_upper_bound_valid,
        proof.full_catalogue_canonical_order_valid,
    ))
    assert (
        metadata.reduction_option_probes,
        metadata.reduction_requirement_coordinate_reads,
        metadata.reduction_duration_comparisons,
    ) == (9, 27, 10)
    assert (
        metadata.lift_option_probes,
        metadata.lift_requirement_coordinate_reads,
        metadata.lift_requirement_vector_comparisons,
        metadata.lift_duration_comparisons,
        metadata.lift_index_comparisons,
    ) == (9, 27, 8, 1, 1)

    def reduction_must_not_repeat(*args, **kwargs):
        raise AssertionError("prepared per-frame selection repeated preprocessing")

    monkeypatch.setattr(
        quotient, "reduce_requirement_dominated_options", reduction_must_not_repeat
    )
    first = select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        prepared, (0, 0, 0), 8, 288, 576
    )
    second = select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        prepared, (3, 2, 1), 8, 288, 576
    )
    assert first.reduction is second.reduction is metadata
    assert first.certificate_lift.proof is proof
    assert second.certificate_lift.proof is proof
    assert (
        first.selection.duration_root_count,
        first.selection.lp_calls,
        first.selection.lp_matrix_coefficients,
        first.selection.candidate_evaluations,
    ) == (8, 8, 348, 9)


def test_prepared_trigger_path_maps_both_checkpoints_and_keeps_live_frontier_work():
    options = _trigger_catalogue()
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        options
    )
    selected = select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        prepared, (1, 1), lanes=2, initial_budget=2, continuation_budget=3
    )

    assert selected.continuation_triggered
    assert selected.primary_selection.score_exact
    assert not selected.primary_selection.canonical_exact
    assert selected.selection.score_exact and selected.selection.canonical_exact
    assert selected.primary_selection is not selected.selection
    for checkpoint in (selected.primary_selection, selected.selection):
        _assert_full_selection_reconstructs(checkpoint, options, (1, 1), 2)
        assert len(checkpoint.option_counts) == 3
        assert checkpoint.option_counts[2] == 0
        assert 2 not in checkpoint.option_indices
    assert selected.selection.lp_calls == 2
    assert selected.primary_selection.lp_calls == 2
    assert selected.certificate_lift.primary_score_exact_lifted
    assert not selected.certificate_lift.primary_canonical_exact_lifted
    assert selected.certificate_lift.selection_score_exact_lifted
    assert selected.certificate_lift.selection_canonical_exact_lifted


def test_prepared_ordinary_path_preserves_primary_identity_and_full_idle_action():
    options = _trigger_catalogue()
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        options
    )
    selected = select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        prepared, (0, 0), lanes=2, initial_budget=2, continuation_budget=3
    )

    assert not selected.continuation_triggered
    assert selected.selection is selected.primary_selection
    assert selected.selection.option_counts == (0, 0, 0)
    assert selected.selection.option_indices == (None, None)
    assert selected.selection.score_exact and selected.selection.canonical_exact


def test_all_reduced_bnb_work_fields_remain_the_direct_eight_option_values():
    model = load_model()
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        model.options
    )
    queue = (12, 2, 8)
    wrapped = select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        prepared, queue, lanes=8, initial_budget=288, continuation_budget=576
    )
    direct = select_canonical_count_space_bnb_score_exact_tie_continuation_frame(
        prepared.reduced_options, queue, 8, 288, 576
    )

    for wrapped_checkpoint, direct_checkpoint in (
        (wrapped.primary_selection, direct.primary_selection),
        (wrapped.selection, direct.selection),
    ):
        for field in fields(CanonicalCountSpaceBnbSelection):
            if field.name in {"option_counts", "option_indices"}:
                continue
            assert getattr(wrapped_checkpoint, field.name) == getattr(
                direct_checkpoint, field.name
            )
        assert wrapped_checkpoint.option_counts[1] == 0
        assert 1 not in wrapped_checkpoint.option_indices
        assert wrapped_checkpoint.represented_ordered_configurations == 9**8
        assert wrapped_checkpoint.represented_count_vectors == math.comb(16, 8)


@pytest.mark.parametrize(
    ("lanes", "initial_budget", "continuation_budget", "queue"),
    (
        (8, 288, 576, (0, 0, 0)),
        (8, 288, 576, (12, 2, 8)),
        (8, 288, 576, (11, 6, 10)),
        (16, 576, 1152, (25, 3, 23)),
    ),
)
def test_registered_quotient_matches_full_k4096_action_score_and_certificates(
    lanes, initial_budget, continuation_budget, queue
):
    model = load_model()
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        model.options
    )
    quotient_result = select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        prepared, queue, lanes, initial_budget, continuation_budget
    )
    selected = quotient_result.selection
    shadow = select_canonical_count_space_bnb_frame(
        model.options, queue, lanes, lp_node_budget=4096
    )

    _assert_full_selection_reconstructs(selected, model.options, queue, lanes)
    assert 1 not in selected.option_indices
    assert selected.option_counts[1] == 0
    assert selected.option_indices == shadow.option_indices
    assert selected.option_counts == shadow.option_counts
    assert selected.service == shadow.service
    assert selected.duration == shadow.duration
    assert selected.score == shadow.score
    assert selected.upper_bound == shadow.upper_bound == selected.score
    assert (
        selected.exact,
        selected.score_exact,
        selected.canonical_exact,
    ) == (
        shadow.exact,
        shadow.score_exact,
        shadow.canonical_exact,
    ) == (True, True, True)
    assert quotient_result.certificate_lift.selection_score_exact_lifted
    assert quotient_result.certificate_lift.selection_canonical_exact_lifted


def test_one_shot_wrapper_has_the_same_mapped_contract():
    options = _trigger_catalogue()
    selected = select_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        options, (1, 1), 2, 2, 3
    )

    assert selected.continuation_triggered
    assert selected.reduction.retained_original_indices == (0, 1)
    assert selected.reduction.removed_original_indices == (2,)
    assert selected.certificate_lift.proof.removed_to_dominator_original_indices == (
        (2, 1),
    )
    assert selected.selection.option_counts == (0, 1, 0)


def test_prepare_rejects_a_reduction_that_cannot_lift_the_canonical_order():
    # Option 0 is duration-dominated by option 1, but replacing it can increase
    # the original-index tie key when another lane determines frame duration.
    options = (
        Option((1, 0), 2.0),
        Option((1, 0), 1.0),
        Option((0, 1), 3.0),
    )

    with pytest.raises(ValueError, match="earlier retained strict-duration"):
        prepare_canonical_count_space_bnb_requirement_dominance_catalogue(options)


@pytest.mark.parametrize(
    ("queue", "initial_budget", "continuation_budget", "message"),
    (
        ((1, -1), 2, 3, "queues must contain"),
        ((1,), 2, 3, "requirements must match"),
        ((1, 1), 1.5, 3, "nonnegative integers"),
        ((1, 1), 3, 3, "strictly increasing"),
    ),
)
def test_prepared_selector_retains_core_input_validation(
    queue, initial_budget, continuation_budget, message
):
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        _trigger_catalogue()
    )
    with pytest.raises(ValueError, match=message):
        select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
            prepared, queue, 2, initial_budget, continuation_budget
        )


@pytest.mark.parametrize("idle_duration", (0.0, -1.0, float("inf"), float("nan")))
def test_prepare_rejects_invalid_idle_duration(idle_duration):
    with pytest.raises(ValueError, match="idle_duration"):
        prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
            _trigger_catalogue(), idle_duration
        )
