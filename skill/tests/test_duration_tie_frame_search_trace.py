"""Regression tests for the duration-resolved offline search trace."""
from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from algorithm.duration_tie_frame_candidates import (
    BudgetedFrameSelection,
    select_duration_tie_frame,
)
from algorithm.experiments.duration_tie_frame_search_trace import (
    DurationTieFrameSearchTrace,
    trace_duration_tie_frame_search,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class TraceOption:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _assert_selection_fields_equal(actual, expected):
    for field in fields(BudgetedFrameSelection):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


@pytest.mark.parametrize("queues", ((6, 6, 10), (4, 3, 5)))
def test_trace_checkpoints_match_independent_duration_selector_runs(queues):
    model = load_model()
    traced = trace_duration_tie_frame_search(
        model.options,
        queues,
        model.lanes,
        small_budget=26,
        exact_budget=120,
    )

    assert isinstance(traced, DurationTieFrameSearchTrace)
    _assert_selection_fields_equal(
        traced.small,
        select_duration_tie_frame(model.options, queues, model.lanes, 26),
    )
    _assert_selection_fields_equal(
        traced.exact,
        select_duration_tie_frame(model.options, queues, model.lanes, 120),
    )
    assert traced.exact.exact
    assert traced.small_gap == traced.small.upper_bound - traced.small.score
    assert traced.score_regret == traced.exact.score - traced.small.score
    assert traced.bound_slack == traced.small.upper_bound - traced.exact.score
    assert traced.gap_identity_error == pytest.approx(0.0, abs=1e-15)
    assert (
        traced.physical_signature_evaluations
        == traced.exact.feasible_candidate_count + 1
    )
    assert traced.physical_signature_completion_offset_reads > 0


def test_trace_covers_early_exact_and_nonzero_small_gap():
    model = load_model()

    early = trace_duration_tie_frame_search(
        model.options, (6, 6, 10), model.lanes
    )
    assert early.small.exact
    assert early.small.attempts == 17 < 26
    assert early.small == early.exact
    assert early.small_gap == early.score_regret == early.bound_slack == 0.0

    open_gap = trace_duration_tie_frame_search(
        model.options, (4, 3, 5), model.lanes
    )
    assert not open_gap.small.exact
    assert open_gap.small.attempts == 26
    assert open_gap.small.gap > 0.0
    assert open_gap.exact.exact
    assert open_gap.exact.attempts == 56
    assert open_gap.bound_slack > 0.0


def test_k120_exact_certifies_support_without_a_unique_physical_action():
    options = (
        TraceOption((2,), 2.0, ((1.0, 2.0),)),
        TraceOption((1,), 1.0, ((1.0,),)),
    )
    traced = trace_duration_tie_frame_search(options, (2,), lanes=1)

    # Both physically different actions attain support two.  The exact flag
    # certifies that value; it neither makes the maximizing action unique nor
    # turns the first support witness into a canonical physical reference.
    assert 2 * 2 / 2.0 == 2 * 1 / 1.0 == traced.exact.score
    assert traced.exact.exact
    assert traced.exact.option_indices == (1,)
    assert traced.first_exact_score_attempt == 2
    assert traced.first_k120_selected_physical_signature_attempt == 3
    assert traced.physical_signature_completion_offset_reads == 4
