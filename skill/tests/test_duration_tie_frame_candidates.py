"""Focused contract tests for duration-resolved candidate ordering."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.duration_tie_frame_candidates import (
    BudgetedFrameSelection,
    select_duration_tie_frame,
)


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _cartesian_optimum(options, queues, lanes, idle_duration=1.0):
    optimum = 0.0
    for assignment in itertools.product((None, *range(len(options))), repeat=lanes):
        service = tuple(
            sum(
                options[index].requirements[coordinate]
                for index in assignment
                if index is not None
            )
            for coordinate in range(len(queues))
        )
        if any(count > available for count, available in zip(service, queues)):
            continue
        duration = max((
            idle_duration,
            *(options[index].duration for index in assignment if index is not None),
        ))
        optimum = max(
            optimum,
            sum(q * count for q, count in zip(queues, service)) / duration,
        )
    return optimum


def _value_error(function, *args, **kwargs):
    with pytest.raises(ValueError) as caught:
        function(*args, **kwargs)
    return str(caught.value)


def test_input_validation_and_minimum_budget_match_frozen_v4():
    valid = (Option((1,), 1.0),)
    invalid_calls = (
        (valid, (1,), 0, 2, 1.0),
        (valid, (1,), 1, 1, 1.0),
        (valid, (1,), 1, 2, 0.0),
        (valid, (-1,), 1, 2, 1.0),
        ((Option((1, 0), 1.0),), (1,), 1, 2, 1.0),
        ((Option((0,), 1.0),), (1,), 1, 2, 1.0),
        ((Option((1,), float("inf")),), (1,), 1, 2, 1.0),
    )
    for args in invalid_calls:
        assert _value_error(select_duration_tie_frame, *args) == _value_error(
            select_budgeted_frame, *args
        )

    options = (
        Option((1, 0), 2.0),
        Option((3, 0), 3.0),
        Option((0, 2), 1.0),
    )
    selected = select_duration_tie_frame(options, (1, 0), 4, 4)
    legacy = select_budgeted_frame(options, (1, 0), 4, 4)
    assert isinstance(selected, BudgetedFrameSelection)
    assert asdict(selected) == asdict(legacy)
    assert selected.attempts == selected.seed_attempts == len(options) + 1
    assert selected.feasible_candidate_count == 2


def test_bounds_cover_cartesian_oracle_and_k120_is_exact():
    options = (
        Option((1, 0), 1.0),
        Option((2, 0), 2.5),
        Option((0, 1), 1.2),
        Option((1, 1), 3.0),
    )
    for queues in itertools.product(range(5), repeat=2):
        optimum = _cartesian_optimum(options, queues, lanes=2)
        previous_score = -1.0
        previous_upper = float("inf")
        for budget in (5, 8, 26, 120):
            selected = select_duration_tie_frame(options, queues, 2, budget)
            assert selected.score <= optimum + 1e-12
            assert optimum <= selected.upper_bound + 1e-12
            assert selected.score >= previous_score
            assert selected.upper_bound <= previous_upper + 1e-12
            assert selected.attempts <= budget
            assert selected.gap == pytest.approx(
                selected.upper_bound - selected.score
            )
            previous_score = selected.score
            previous_upper = selected.upper_bound
        assert selected.exact
        assert selected.score == pytest.approx(optimum)


def test_equal_weight_shorter_duration_candidate_is_reached_first():
    # At Q=(2, 2), options 0 and 2 both have weight 2.  Frozen v4 orders
    # option 0 first by catalogue index even though it lasts twice as long.
    options = (
        Option((1, 0), 2.0),
        Option((2, 0), 1.0),
        Option((0, 1), 1.0),
    )
    legacy = select_budgeted_frame(options, (2, 2), 2, 8)
    selected = select_duration_tie_frame(options, (2, 2), 2, 8)

    assert legacy.option_indices == (1, None)
    assert legacy.score == 4.0
    assert selected.option_indices == (1, 2)
    assert selected.service == (2, 1)
    assert selected.duration == 1.0
    assert selected.score == 6.0
    assert selected.exact


def test_available_option_precedes_shorter_unavailable_option():
    options = (
        Option((2, 0), 1.0),  # unavailable, weight 2, shorter
        Option((0, 1), 4.0),  # available, weight 1, longer
    )
    seeded = select_duration_tie_frame(options, (1, 1), 2, 3)
    one_search_attempt = select_duration_tie_frame(options, (1, 1), 2, 4)

    assert seeded.feasible_candidate_count == 2
    assert one_search_attempt.attempts == 4
    # The first post-seed attempt extends the available option.  Sorting by
    # duration before availability would spend it on infeasible option zero.
    assert one_search_attempt.feasible_candidate_count == 3
    assert one_search_attempt.option_indices == (1, None)


def test_equal_weight_order_uses_duration_after_idle_floor():
    options = (
        Option((1, 0), 0.8),
        Option((2, 0), 1.0),
        Option((0, 1), 0.2),
    )
    selected = select_duration_tie_frame(
        options, (2, 2), 2, 8, idle_duration=1.0
    )

    # Options zero and two both have weight 2 and effective duration 1.  Their
    # raw durations must not move option two ahead of the lower catalogue index.
    assert selected.option_indices == (0, 0)
    assert selected.duration == 1.0
    assert selected.score == 4.0
    assert not selected.exact
    exact = select_duration_tie_frame(
        options, (2, 2), 2, 120, idle_duration=1.0
    )
    assert exact.option_indices == (1, 2)
    assert exact.score == 6.0
    assert exact.exact


def test_distinct_weight_order_retains_frozen_v4_outputs():
    options = (
        Option((1, 0), 8.0),  # weight 5
        Option((0, 1), 1.0),  # weight 3
        Option((1, 1), 4.0),  # weight 8
    )
    for budget in (4, 5, 6, 8, 12, 120):
        selected = select_duration_tie_frame(options, (5, 3), 2, budget)
        legacy = select_budgeted_frame(options, (5, 3), 2, budget)
        assert asdict(selected) == asdict(legacy)
