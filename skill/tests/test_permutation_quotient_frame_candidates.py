"""Focused contracts for lane-permutation quotient candidate accounting."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import random

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.permutation_quotient_frame_candidates import (
    PermutationQuotientFrameSelection,
    select_permutation_quotient_frame,
)


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _cartesian_optimum(options, queues, lanes, idle_duration=1.0):
    optimum = 0.0
    for assignment in itertools.product(
        (None, *range(len(options))), repeat=lanes
    ):
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
            *(
                options[index].duration
                for index in assignment
                if index is not None
            ),
        ))
        optimum = max(
            optimum,
            sum(q * count for q, count in zip(queues, service)) / duration,
        )
    return optimum


def _assert_realizable(options, queues, lanes, selected, idle_duration=1.0):
    assert len(selected.option_indices) == lanes
    service = tuple(
        sum(
            options[index].requirements[coordinate]
            for index in selected.option_indices
            if index is not None
        )
        for coordinate in range(len(queues))
    )
    duration = max((
        idle_duration,
        *(
            options[index].duration
            for index in selected.option_indices
            if index is not None
        ),
    ))
    assert selected.service == service
    assert all(count <= available for count, available in zip(service, queues))
    assert selected.duration == duration
    assert selected.score == pytest.approx(
        sum(q * count for q, count in zip(queues, service)) / duration
    )


def _random_family(rng):
    dimension = rng.randint(1, 3)
    option_count = rng.randint(2, 5)
    lanes = rng.randint(2, 3)
    options = []
    for _ in range(option_count):
        requirements = tuple(rng.randint(0, 3) for _ in range(dimension))
        if not any(requirements):
            requirements = (1, *requirements[1:])
        options.append(Option(
            requirements,
            rng.choice((0.5, 1.0, 2.5, 5.0, 7.0)),
        ))
    queues = tuple(rng.randint(0, 7) for _ in range(dimension))
    return tuple(options), queues, lanes


def test_random_small_families_are_exact_by_the_multiset_budget():
    rng = random.Random(260911)
    for _ in range(18):
        options, queues, lanes = _random_family(rng)
        exact_budget = math.comb(len(options) + lanes, lanes)
        selected = select_permutation_quotient_frame(
            options, queues, lanes, exact_budget
        )

        assert isinstance(selected, PermutationQuotientFrameSelection)
        _assert_realizable(options, queues, lanes, selected)
        assert selected.exact
        assert selected.score == pytest.approx(
            _cartesian_optimum(options, queues, lanes)
        )
        assert selected.upper_bound == selected.score
        assert selected.attempts <= exact_budget
        assert selected.cache_entries <= selected.represented_lane_multisets


def test_budget_path_is_monotone_and_no_worse_than_legacy_at_the_same_k():
    rng = random.Random(110926)
    for _ in range(12):
        options, queues, lanes = _random_family(rng)
        seed_budget = len(options) + 1
        exact_budget = math.comb(len(options) + lanes, lanes)
        previous_score = -1.0
        previous_upper = float("inf")
        previous_gap = float("inf")

        for budget in range(seed_budget, exact_budget + 1):
            selected = select_permutation_quotient_frame(
                options, queues, lanes, budget
            )
            legacy = select_budgeted_frame(options, queues, lanes, budget)

            _assert_realizable(options, queues, lanes, selected)
            assert selected.score + 1e-12 >= legacy.score
            assert selected.upper_bound <= legacy.upper_bound + 1e-12
            assert selected.gap <= legacy.gap + 1e-12
            assert selected.score + 1e-12 >= previous_score
            assert selected.upper_bound <= previous_upper + 1e-12
            assert selected.gap <= previous_gap + 1e-12
            assert selected.attempts <= budget
            assert selected.gap == pytest.approx(
                selected.upper_bound - selected.score
            )
            assert selected.epsilon == pytest.approx(
                selected.gap / sum(queues) if sum(queues) else 0.0
            )
            previous_score = selected.score
            previous_upper = selected.upper_bound
            previous_gap = selected.gap


def test_two_lane_nine_option_family_reports_100_orderings_and_55_orbits():
    options = tuple(
        Option((index + 1,), float(index % 4 + 1))
        for index in range(9)
    )
    selected = select_permutation_quotient_frame(
        options, (100,), lanes=2, budget=26
    )

    assert selected.seed_attempts == 10
    assert selected.represented_ordered_configurations == 100
    assert selected.represented_lane_multisets == 55
    assert selected.symmetry_duplicate_ordered_configurations == 45


def test_cache_hits_and_work_accounting_identities_close():
    options = (
        Option((1, 0), 1.0),
        Option((1, 0), 1.0),
        Option((1, 2), 1.0),
    )
    selected = select_permutation_quotient_frame(
        options, (2, 1), lanes=2, budget=10
    )

    assert selected.cached_feasible_reuses > 0
    assert selected.cached_infeasible_reuses > 0
    assert selected.cache_hits == selected.free_cache_hit_pops
    assert selected.cache_misses_after_seeds == selected.paid_search_pops
    assert selected.logical_search_pops == (
        selected.paid_search_pops + selected.free_cache_hit_pops
    )
    assert selected.free_cache_hit_pops == (
        selected.cached_feasible_reuses + selected.cached_infeasible_reuses
    )
    assert selected.attempts == (
        selected.seed_attempts + selected.paid_search_pops
    )
    assert selected.cache_entries == selected.attempts
    assert selected.candidate_score_evaluations == (
        selected.feasible_candidate_count
    )
    assert selected.cache_key_evaluations == (
        selected.seed_attempts
        + selected.logical_search_pops
        + selected.budget_blocked_cache_miss_lookups
    )
    assert selected.cache_key_item_reads == (
        2 * selected.cache_key_evaluations
    )
    assert selected.upper_bound_calls == (
        selected.frontier_pushes + selected.bound_rejected_prefixes
    )
    assert selected.frontier_pushes - selected.logical_search_pops == (
        selected.exact_stop_frontier_entries_cleared
        + selected.frontier_entries_remaining
    )
    assert selected.symmetry_duplicate_ordered_configurations == (
        selected.represented_ordered_configurations
        - selected.represented_lane_multisets
    )


def test_cache_key_does_not_replace_the_raw_lane_order_tie_witness():
    options = (
        Option((1, 0), 1.0),
        Option((1, 0), 1.0),
        Option((2, 0), 1.0),
    )
    queues = (3, 1)
    selected = select_permutation_quotient_frame(
        options, queues, lanes=2, budget=7
    )
    legacy_exact = select_budgeted_frame(options, queues, lanes=2, budget=8)

    # The multiset cache key is (0, 2), but the unchanged v4 heap first reaches
    # the physically equivalent raw lane tuple (2, 0). Returning the cache key
    # here would silently change the registered lane-order tie behavior.
    assert selected.cache_hits > 0
    assert selected.option_indices == (2, 0)
    assert selected.option_indices != tuple(sorted(selected.option_indices))
    assert selected.option_indices == legacy_exact.option_indices
    assert selected.score == legacy_exact.score == 9.0
    assert selected.exact


def test_input_validation_matches_the_budgeted_selector_contract():
    options = (Option((1, 0), 1.0), Option((0, 1), 2.0))

    with pytest.raises(ValueError, match="lanes must be a positive integer"):
        select_permutation_quotient_frame(options, (1, 1), 0, 3)
    with pytest.raises(ValueError, match="lanes must be a positive integer"):
        select_permutation_quotient_frame(options, (1, 1), 1.5, 3)
    with pytest.raises(ValueError, match="covering 3 seed attempts"):
        select_permutation_quotient_frame(options, (1, 1), 2, 2)
    with pytest.raises(ValueError, match="budget must be an integer"):
        select_permutation_quotient_frame(options, (1, 1), 2, 3.5)
    with pytest.raises(ValueError, match="idle_duration must be positive"):
        select_permutation_quotient_frame(
            options, (1, 1), 2, 3, idle_duration=0.0
        )
    with pytest.raises(ValueError, match="queues must contain nonnegative"):
        select_permutation_quotient_frame(options, (1, -1), 2, 3)
    with pytest.raises(ValueError, match="queues must contain nonnegative"):
        select_permutation_quotient_frame(options, (1, 0.5), 2, 3)

    invalid_options = (
        Option((1,), 1.0),
        Option((-1, 0), 1.0),
        Option((0.5, 0), 1.0),
        Option((0, 0), 1.0),
    )
    for invalid in invalid_options:
        with pytest.raises(ValueError):
            select_permutation_quotient_frame((invalid,), (1, 1), 2, 2)
    for duration in (0.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="durations must be positive"):
            select_permutation_quotient_frame(
                (Option((1, 0), duration),), (1, 1), 2, 2
            )
