"""Independent Cartesian oracles for queue-capped search bounds."""
from dataclasses import dataclass
import itertools
import random

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _cartesian_optimum(options, queues, lanes):
    optimum = 0.0
    for assignment in itertools.product((None, *range(len(options))), repeat=lanes):
        departures = tuple(
            sum(options[index].requirements[i] for index in assignment if index is not None)
            for i in range(len(queues))
        )
        if any(departure > available for departure, available in zip(departures, queues)):
            continue
        duration = max(1.0, *(options[index].duration if index is not None else 1.0
                              for index in assignment))
        support = sum(q * d for q, d in zip(queues, departures)) / duration
        optimum = max(optimum, support)
    return optimum


def _assert_budgets_cover_oracle(options, queues, lanes, budgets):
    optimum = _cartesian_optimum(options, queues, lanes)
    previous_score = -1.0
    previous_upper = float("inf")
    for budget in budgets:
        selected = select_budgeted_frame(options, queues, lanes, budget)
        context = (queues, lanes, budget, selected)
        assert selected.score <= optimum + 1e-12, context
        assert selected.upper_bound + 1e-12 >= optimum, context
        assert selected.score >= previous_score, context
        assert selected.upper_bound <= previous_upper + 1e-12, context
        assert selected.attempts <= budget, context
        assert all(d <= q for d, q in zip(selected.service, queues)), context
        previous_score, previous_upper = selected.score, selected.upper_bound
    assert selected.exact, context
    assert selected.score == pytest.approx(optimum), context


def test_all_225_registered_small_queues_keep_valid_bounds_across_budgets():
    # The Poisson model can reach every one of these integer queues from idle.
    # Shared batches, missing resident classes, and p3/p8 tails all occur here.
    model = load_model()
    for queues in itertools.product(range(5), range(5), range(9)):
        _assert_budgets_cover_oracle(model.options, queues, model.lanes, (10, 14, 26, 120))


@pytest.mark.parametrize("queues", ((1, 0, 1), (0, 1, 3)))
def test_k26_finds_complementary_lanes_missed_by_uncapped_search(queues):
    model = load_model()
    selected = select_budgeted_frame(model.options, queues, model.lanes, 26)
    assert selected.exact
    assert selected.score == pytest.approx(_cartesian_optimum(model.options, queues, model.lanes))


def test_five_llm_jobs_preserve_the_p3_plus_p1_completion_in_the_bound():
    model = load_model()
    queues = (4, 1, 5)
    optimum = _cartesian_optimum(model.options, queues, model.lanes)
    # This queue remains a useful budget-limited case even when a later search
    # improvement eventually makes K=26 exact: its upper bound must cover p3+p1.
    selected = select_budgeted_frame(model.options, queues, model.lanes, 26)
    assert optimum == pytest.approx(2.1136107668950994)
    assert selected.score <= optimum + 1e-12
    assert selected.upper_bound + 1e-12 >= optimum
    _assert_budgets_cover_oracle(model.options, queues, model.lanes, (10, 14, 26, 120))


def test_mixed_requirements_do_not_turn_coordinate_maxima_into_false_pruning():
    # Coordinate maxima may come from different options. Bounds must remain
    # optimistic after a prefix consumes a shared class or skips a sibling.
    rng = random.Random(10713)
    for case in range(12):
        options = [
            Option(tuple(1 if i == coordinate else 0 for i in range(3)),
                   rng.choice((0.5, 1.0, 2.75, 6.0)))
            for coordinate in range(3)
        ]
        for _ in range(3):
            coordinates = rng.sample(range(3), 2)
            requirements = [0, 0, 0]
            for coordinate in coordinates:
                requirements[coordinate] = rng.randint(1, 3)
            options.append(Option(tuple(requirements), rng.choice((0.5, 1.0, 2.75, 6.0))))
        queues = tuple(rng.randint(0, 7) for _ in range(3))
        lanes = 2 + case % 2
        _assert_budgets_cover_oracle(tuple(options), queues, lanes, (7, 11, 26, 128, 500))
