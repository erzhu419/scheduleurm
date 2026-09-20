"""Focused contracts for the binary-price residual-capacity bound."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.residual_capacity_frame_candidates import (
    BudgetedFrameSelection,
    select_residual_capacity_frame,
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
                for index in assignment if index is not None
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


def _selection_identity(selection):
    return (
        selection.option_indices,
        selection.service,
        selection.duration,
        selection.score,
        selection.attempts,
        selection.seed_attempts,
        selection.feasible_candidate_count,
        selection.local_score_evaluations,
    )


def test_seed_selection_and_v4_order_are_unchanged_while_bound_tightens():
    options = (
        Option((1, 2, 1), 4.0),
        Option((2, 2, 3), 4.0),
        Option((2, 2, 1), 4.0),
        Option((1, 2, 2), 2.0),
        Option((3, 0, 3), 2.0),
    )
    queues = (6, 7, 5)
    budget = len(options) + 1
    legacy = select_budgeted_frame(options, queues, 3, budget)
    selected = select_residual_capacity_frame(options, queues, 3, budget)

    assert isinstance(selected, BudgetedFrameSelection)
    assert _selection_identity(selected) == _selection_identity(legacy)
    assert selected.ordered_option_indices == (1, 4, 2, 3, 0, None)
    assert legacy.upper_bound == 49.5
    assert selected.upper_bound == 42.5
    assert selected.upper_bound >= _cartesian_optimum(options, queues, 3)
    assert selected.bound_strict_tightening_count == 1
    assert selected.bound_tightening_max == 7.0


def test_bounds_cover_independent_cartesian_oracles_across_budgets():
    rng = random.Random(1709)
    for _ in range(16):
        options = []
        for _ in range(5):
            requirements = tuple(rng.randint(0, 3) for _ in range(3))
            if not any(requirements):
                requirements = (1, 0, 0)
            options.append(Option(requirements, rng.choice((0.5, 1.0, 2.5, 7.0))))
        options = tuple(options)
        queues = tuple(rng.randint(0, 7) for _ in range(3))
        optimum = _cartesian_optimum(options, queues, lanes=3)
        previous_score = -1.0
        previous_upper = float("inf")
        for budget in (6, 8, 14, 26, 120, 500):
            selected = select_residual_capacity_frame(options, queues, 3, budget)
            assert selected.score <= optimum + 1e-12
            assert optimum <= selected.upper_bound + 1e-12
            assert selected.score >= previous_score
            assert selected.upper_bound <= previous_upper + 1e-12
            assert selected.attempts <= budget
            previous_score = selected.score
            previous_upper = selected.upper_bound
        assert selected.exact
        assert selected.score == pytest.approx(optimum)


def test_binary_price_work_counts_close_exactly():
    options = (
        Option((1, 0, 1), 3.0),
        Option((0, 2, 0), 1.0),
        Option((0, 0, 3), 5.0),
    )
    queues = (4, 3, 5)
    selected = select_residual_capacity_frame(options, queues, 2, 12)
    dimensions = len(queues)
    masks = 1 << dimensions
    positions = len(options) + 1
    thresholds = len({1.0, *(max(1.0, option.duration) for option in options)})

    assert selected.bound_price_vector_count == masks
    assert selected.bound_dual_table_option_probes == thresholds * masks * positions
    assert selected.bound_dual_mask_evaluations == (
        selected.bound_threshold_evaluations * masks
    )
    assert selected.bound_dual_mask_sums == selected.upper_bound_calls * masks
    assert selected.bound_dual_coordinate_products == dimensions * (
        masks * positions + selected.upper_bound_calls
    )
    assert 0 <= selected.bound_strict_tightening_count <= selected.upper_bound_calls
    assert 0.0 <= selected.bound_tightening_max <= selected.bound_tightening_sum


def test_non_dominant_threshold_reduction_is_not_reported_as_bound_activation():
    options = (
        Option((3, 2, 1), 7.0),
        Option((0, 2, 0), 1.0),
        Option((0, 3, 2), 3.0),
        Option((3, 2, 3), 7.0),
        Option((1, 1, 3), 3.0),
    )
    queues = (5, 7, 5)
    legacy = select_budgeted_frame(options, queues, 3, 6)
    selected = select_residual_capacity_frame(options, queues, 3, 6)

    # The D=3 cap tightens from 89/3 to 88/3, but D=1 remains the
    # max-over-threshold bound at 42.  Activation therefore remains zero.
    assert selected.ordered_option_indices == (3, 0, 2, 4, 1, None)
    assert legacy.upper_bound == selected.upper_bound == 42.0
    assert selected.bound_strict_tightening_count == 0
    assert selected.bound_tightening_sum == selected.bound_tightening_max == 0.0
    assert selected.upper_bound_calls == 1


def test_empty_queue_and_catalogue_keep_paid_idle_exact():
    selected = select_residual_capacity_frame(
        (Option((1, 0), 1.0), Option((0, 1), 2.0)),
        (0, 0),
        2,
        3,
    )
    assert selected.option_indices == (None, None)
    assert selected.score == selected.upper_bound == selected.gap == 0.0
    assert selected.exact
    empty = select_residual_capacity_frame((), (3, 2), 2, 1)
    assert empty.service == (0, 0)
    assert empty.attempts == 1
    assert empty.exact


def test_large_family_does_not_enumerate_cartesian_actions(monkeypatch):
    options = tuple(
        Option(
            (index % 4 + 1, 0) if index < 8 else (0, index % 4 + 1),
            float(index % 5 + 1),
        )
        for index in range(16)
    )

    def forbidden_product(*args, **kwargs):
        raise AssertionError("selector enumerated the Cartesian action family")

    monkeypatch.setattr(itertools, "product", forbidden_product)
    selected = select_residual_capacity_frame(options, (5, 4), 8, 40)
    assert selected.attempts <= 40
    assert selected.seed_attempts == 17
    assert selected.bound_price_vector_count == 4
    assert selected.bound_dual_table_option_probes <= 17 * 4 * 17
    assert selected.frontier_peak <= 40 - 17 + 1
