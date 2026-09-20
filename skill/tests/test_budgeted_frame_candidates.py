from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _oracle(options, queues, lanes, idle_duration=1.0):
    best = 0.0
    for indices in itertools.product((None, *range(len(options))), repeat=lanes):
        service = tuple(
            sum(options[index].requirements[coordinate] for index in indices if index is not None)
            for coordinate in range(len(queues))
        )
        if any(count > queue for count, queue in zip(service, queues)):
            continue
        duration = max((idle_duration, *(options[index].duration for index in indices if index is not None)))
        best = max(best, sum(queue * count for queue, count in zip(queues, service)) / duration)
    return best


def _assert_realizable(result, options, queues, lanes, idle_duration=1.0):
    assert len(result.option_indices) == lanes
    service = tuple(
        sum(options[index].requirements[coordinate] for index in result.option_indices if index is not None)
        for coordinate in range(len(queues))
    )
    assert result.service == service
    assert all(count <= queue for count, queue in zip(service, queues))
    assert result.duration == max((idle_duration, *(options[index].duration for index in result.option_indices if index is not None)))
    assert result.score == sum(queue * count for queue, count in zip(queues, service)) / result.duration


def test_bounds_cover_independent_small_cartesian_oracle_and_tighten_with_budget():
    rng = random.Random(194)
    for _ in range(12):
        options = tuple(
            Option((rng.randint(1, 3), 0) if index < 2 else (0, rng.randint(1, 3)),
                   rng.choice((0.5, 1.0, 2.5, 7.0)))
            for index in range(4)
        )
        queues = (rng.randint(0, 5), rng.randint(0, 5))
        lanes = 3
        optimum = _oracle(options, queues, lanes)
        previous_score, previous_upper = -1.0, float("inf")
        for budget in (5, 6, 10, 25, 100, 500):
            result = select_budgeted_frame(options, queues, lanes, budget)
            _assert_realizable(result, options, queues, lanes)
            assert result.score <= optimum + 1e-12
            assert optimum <= result.upper_bound + 1e-12
            assert result.score >= previous_score
            assert result.upper_bound <= previous_upper + 1e-12
            assert result.attempts <= budget
            assert result.feasible_candidate_count <= result.attempts
            assert result.seed_attempts == len(options) + 1
            assert result.local_score_evaluations == len(options)
            assert result.gap == result.upper_bound - result.score
            assert result.epsilon == (result.gap / sum(queues) if sum(queues) else 0.0)
            previous_score, previous_upper = result.score, result.upper_bound
        assert result.exact
        assert result.score == pytest.approx(optimum)


def test_frame_duration_is_maximum_not_sum_of_individual_service_rates():
    options = (Option((1, 0), 1.0), Option((0, 4), 10.0))
    queues = (1, 4)
    result = select_budgeted_frame(options, queues, 2, 100)
    assert result.service == (1, 4)
    assert result.duration == 10.0
    assert result.score == 1.7
    assert result.exact


def test_minimum_budget_keeps_singleton_tail_and_charges_unusable_seeds():
    options = (Option((1, 0), 2.0), Option((3, 0), 3.0), Option((0, 2), 1.0))
    result = select_budgeted_frame(options, (1, 0), 4, 4)
    assert result.option_indices == (0, None, None, None)
    assert result.service == (1, 0)
    assert result.attempts == result.seed_attempts == 4
    assert result.feasible_candidate_count == 2  # idle and the projected p1 seed
    with pytest.raises(ValueError, match="4 seed attempts"):
        select_budgeted_frame(options, (1, 0), 4, 3)


def test_rejected_search_attempts_are_paid_and_do_not_discard_siblings():
    options = (Option((2, 0), 1.0), Option((0, 1), 1.0))
    queues = (1, 1)
    first = select_budgeted_frame(options, queues, 2, 5)
    assert first.attempts == 5
    # Paid seed feasibility puts the usable b first; its second copy fails.
    # Idle, the b seed and the first b prefix are the three feasible attempts.
    assert first.feasible_candidate_count == 3
    result = select_budgeted_frame(options, queues, 2, 100)
    assert result.service == (0, 1)
    assert result.score == _oracle(options, queues, 2)
    assert result.exact


def test_empty_queue_and_empty_catalogue_return_paid_idle_with_zero_gap():
    options = (Option((1, 0), 1.0), Option((0, 1), 2.0))
    result = select_budgeted_frame(options, (0, 0), 2, 3)
    assert result.option_indices == (None, None)
    assert result.attempts == 3
    assert result.feasible_candidate_count == 1
    assert result.score == result.upper_bound == result.gap == result.epsilon == 0.0
    assert result.exact
    empty = select_budgeted_frame((), (3, 2), 2, 1)
    assert empty.service == (0, 0)
    assert empty.attempts == 1
    assert empty.exact


def test_large_cartesian_family_obeys_small_budget_without_product_enumeration(monkeypatch):
    # 17**8 is over six billion full combinations, while only 40 attempts are paid.
    options = tuple(
        Option((index % 4 + 1, 0) if index < 8 else (0, index % 4 + 1),
               float(index % 5 + 1))
        for index in range(16)
    )

    def forbidden_product(*args, **kwargs):
        raise AssertionError("selector enumerated the full Cartesian family")

    monkeypatch.setattr(itertools, "product", forbidden_product)
    result = select_budgeted_frame(options, (5, 4), 8, 40)
    _assert_realizable(result, options, (5, 4), 8)
    assert result.attempts <= 40
    assert result.seed_attempts == 17
    assert result.feasible_candidate_count <= 40
    assert result.frontier_peak <= 40 - 17 + 1
    assert result.local_score_evaluations == 16
    # One quadratic table build; at most two threshold queries per attempt.
    assert result.bound_option_probes <= 17 * 17
    assert result.bound_threshold_evaluations <= (2 * (40 - 17) + 1) * 17
    assert result.bound_option_probes > 0
