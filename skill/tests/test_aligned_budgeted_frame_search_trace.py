"""Regression tests for the offline legacy-v4 search trace."""
from __future__ import annotations

from dataclasses import dataclass, fields
import itertools

import pytest

from algorithm.budgeted_frame_candidates import (
    BudgetedFrameSelection,
    select_budgeted_frame,
)
from algorithm.experiments.aligned_budgeted_frame_search_trace import (
    trace_budgeted_frame_search,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _assert_selection_fields_equal(actual, expected):
    for field in fields(BudgetedFrameSelection):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


def _cartesian_oracle(options, queues, lanes, idle_duration=1.0):
    best = None
    for indices in itertools.product((None, *range(len(options))), repeat=lanes):
        service = tuple(
            sum(
                options[index].requirements[coordinate]
                for index in indices
                if index is not None
            )
            for coordinate in range(len(queues))
        )
        if any(count > available for count, available in zip(service, queues)):
            continue
        duration = max((
            idle_duration,
            *(options[index].duration for index in indices if index is not None),
        ))
        score = sum(q * count for q, count in zip(queues, service)) / duration
        key = (
            duration,
            tuple(len(options) if index is None else index for index in indices),
        )
        if best is None or score > best[0] or (score == best[0] and key < best[1]):
            best = score, key, indices, service, duration
    return best


@pytest.mark.parametrize("queues", ((0, 0, 0), (4, 3, 5), (6, 6, 10)))
def test_model_trace_matches_both_independent_selector_runs_field_by_field(queues):
    model = load_model()
    traced = trace_budgeted_frame_search(model.options, queues, model.lanes)

    _assert_selection_fields_equal(
        traced.small,
        select_budgeted_frame(model.options, queues, model.lanes, 26),
    )
    _assert_selection_fields_equal(
        traced.exact,
        select_budgeted_frame(model.options, queues, model.lanes, 120),
    )
    assert traced.small_gap == traced.small.upper_bound - traced.small.score
    assert traced.score_regret == traced.exact.score - traced.small.score
    assert traced.bound_slack == traced.small.upper_bound - traced.exact.score
    assert traced.gap_identity_error == 0.0
    assert (
        traced.physical_signature_evaluations
        == traced.exact.feasible_candidate_count + 1
    )

    if queues == (0, 0, 0):
        assert traced.small.attempts < 26
        assert traced.small == traced.exact
        assert traced.first_exact_score_attempt == 1
        assert traced.first_k120_selected_physical_signature_attempt == 1
    elif queues == (4, 3, 5):
        assert not traced.small.exact
        assert traced.score_regret > 0.0
        assert traced.bound_slack > 0.0
        assert traced.first_exact_score_attempt == 28
        assert traced.first_k120_selected_physical_signature_attempt == 28


def test_crafted_oracle_distinguishes_first_optimal_score_from_final_physical_action():
    # Both options attain support four.  The second has the shorter duration, so
    # the frozen tie break makes it the final action although option zero's seed
    # reaches the final optimal score one paid attempt earlier.
    options = (
        Option((2,), 2.0, ((1.0, 2.0),)),
        Option((1,), 1.0, ((1.0,),)),
    )
    queues = (2,)
    traced = trace_budgeted_frame_search(options, queues, lanes=1)
    oracle = _cartesian_oracle(options, queues, lanes=1)

    _assert_selection_fields_equal(
        traced.small,
        select_budgeted_frame(options, queues, 1, 26),
    )
    _assert_selection_fields_equal(
        traced.exact,
        select_budgeted_frame(options, queues, 1, 120),
    )
    assert traced.exact.score == oracle[0]
    assert traced.exact.option_indices == oracle[2]
    assert traced.exact.service == oracle[3]
    assert traced.exact.duration == oracle[4]
    assert traced.small.attempts == 3 < 26
    assert traced.small == traced.exact
    assert traced.first_exact_score_attempt == 2
    assert traced.first_k120_selected_physical_signature_attempt == 3
    assert traced.physical_signature_evaluations == 4
    assert traced.physical_signature_completion_offset_reads == 4
    assert traced.small_gap == traced.score_regret == traced.bound_slack == 0.0
    assert traced.gap_identity_error == 0.0
