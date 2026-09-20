"""Independent contracts for FIFO-execution-signature candidate reuse."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import itertools
import math
import random

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.fifo_execution_quotient_frame_candidates import (
    FifoExecutionQuotientFrameSelection,
    fifo_execution_signature,
    select_fifo_execution_quotient_frame,
)


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _independent_signature(
    options, option_indices, lanes, dimensions, idle_duration=1.0,
):
    """Build the semantic key without using the selector's helper."""
    padded = tuple(option_indices) + (None,) * (lanes - len(option_indices))
    service = tuple(
        sum(
            options[index].requirements[coordinate]
            for index in padded
            if index is not None
        )
        for coordinate in range(dimensions)
    )
    duration = max((
        idle_duration,
        *(options[index].duration for index in padded if index is not None),
    ))
    ordered_offsets = tuple(
        tuple(
            float(offset)
            for index in padded
            if index is not None
            for offset in options[index].completion_offsets[coordinate]
        )
        for coordinate in range(dimensions)
    )
    return service, duration, ordered_offsets


def _assignments(option_count, lanes):
    return tuple(
        itertools.product((None, *range(option_count)), repeat=lanes)
    )


def _oracle(options, queues, lanes, idle_duration=1.0):
    best_score = -1.0
    for assignment in _assignments(len(options), lanes):
        service, duration, _ = _independent_signature(
            options, assignment, lanes, len(queues), idle_duration
        )
        if any(count > available for count, available in zip(service, queues)):
            continue
        score = sum(q * count for q, count in zip(queues, service)) / duration
        best_score = max(best_score, score)
    return best_score


def _assert_realizable(options, queues, lanes, selected, idle_duration=1.0):
    assert len(selected.option_indices) == lanes
    service, duration, _ = _independent_signature(
        options, selected.option_indices, lanes, len(queues), idle_duration
    )
    assert selected.service == service
    assert all(count <= available for count, available in zip(service, queues))
    assert selected.duration == duration
    assert selected.score == pytest.approx(
        sum(q * count for q, count in zip(queues, service)) / duration
    )


def _random_family(rng):
    dimensions = rng.randint(1, 3)
    option_count = rng.randint(2, 4)
    lanes = rng.randint(2, 3)
    options = []
    for _ in range(option_count):
        requirements = tuple(rng.randint(0, 2) for _ in range(dimensions))
        if not any(requirements):
            requirements = (1, *requirements[1:])
        duration = rng.choice((1.0, 2.0, 3.0, 5.0))
        offsets = tuple(
            tuple(
                duration * rng.choice((0.25, 0.5, 0.75, 1.0))
                for _ in range(count)
            )
            for count in requirements
        )
        options.append(Option(requirements, duration, offsets))
    queues = tuple(rng.randint(0, 6) for _ in range(dimensions))
    return tuple(options), queues, lanes


def _signature_cap(options, queues, lanes):
    signatures = {
        _independent_signature(options, row, lanes, len(queues))
        for row in _assignments(len(options), lanes)
    }
    seeds = select_fifo_execution_quotient_frame(
        options, queues, lanes, len(options) + 1
    )
    # Catalogue signature classes are paid once, while the frozen contract
    # deliberately keeps colliding homogeneous seeds separately paid.
    return len(signatures) + seeds.seed_signature_collisions, signatures, seeds


def test_random_small_families_are_exact_by_the_fifo_signature_cap():
    rng = random.Random(260911)
    collision_free_count = 0
    for _ in range(24):
        options, queues, lanes = _random_family(rng)
        cap, signatures, seeds = _signature_cap(options, queues, lanes)
        selected = select_fifo_execution_quotient_frame(
            options, queues, lanes, cap
        )

        if seeds.seed_signature_collisions == 0:
            collision_free_count += 1
            assert cap == len(signatures)
        assert isinstance(selected, FifoExecutionQuotientFrameSelection)
        _assert_realizable(options, queues, lanes, selected)
        assert selected.exact
        assert selected.score == pytest.approx(_oracle(options, queues, lanes))
        assert selected.upper_bound == selected.score
        assert selected.gap == 0.0
        assert selected.attempts <= cap
        assert selected.cache_entries <= len(signatures)
        assert selected.cache_entries + selected.seed_signature_collisions == (
            selected.attempts
        )
        for assignment in _assignments(len(options), lanes):
            assert fifo_execution_signature(
                options, assignment, lanes, len(queues)
            ) == _independent_signature(
                options, assignment, lanes, len(queues)
            )
    assert collision_free_count >= 20


def test_same_paid_budget_is_nonworse_and_budget_paths_are_monotone():
    rng = random.Random(110926)
    for _ in range(16):
        options, queues, lanes = _random_family(rng)
        cap, _, _ = _signature_cap(options, queues, lanes)
        seed_budget = len(options) + 1
        previous_score = -1.0
        previous_upper = float("inf")
        previous_gap = float("inf")

        for budget in range(seed_budget, cap + 1):
            selected = select_fifo_execution_quotient_frame(
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


def test_real_catalogue_has_100_actions_and_71_fifo_execution_signatures():
    model = load_model()
    assignments = _assignments(len(model.options), model.lanes)
    by_signature = {}
    for assignment in assignments:
        independent = _independent_signature(
            model.options, assignment, model.lanes, len(model.workloads)
        )
        assert fifo_execution_signature(
            model.options, assignment, model.lanes, len(model.workloads)
        ) == independent
        by_signature.setdefault(independent, []).append(assignment)

    assert len(assignments) == 100
    assert len(by_signature) == 71
    assert sum(len(group) - 1 for group in by_signature.values()) == 29
    assert {len(group) for group in by_signature.values()} == {1, 2}

    dangerous_forward = fifo_execution_signature(
        model.options, (0, 7), 2, len(model.workloads)
    )
    dangerous_reverse = fifo_execution_signature(
        model.options, (7, 0), 2, len(model.workloads)
    )
    assert dangerous_forward[:2] == dangerous_reverse[:2]
    assert dangerous_forward != dangerous_reverse

    safe_forward = fifo_execution_signature(
        model.options, (3, 8), 2, len(model.workloads)
    )
    safe_reverse = fifo_execution_signature(
        model.options, (8, 3), 2, len(model.workloads)
    )
    assert safe_forward == safe_reverse
    assert fifo_execution_signature(
        model.options, (3,), 2, len(model.workloads)
    ) == fifo_execution_signature(
        model.options, (None, 3), 2, len(model.workloads)
    )


def test_k71_is_exact_on_representative_real_queues_without_state_roster():
    model = load_model()
    queues = (
        (0, 0, 0),
        (1, 0, 4),
        (2, 2, 5),
        (3, 3, 9),
        (7, 4, 13),
        (10, 10, 10),
    )
    for queue in queues:
        selected = select_fifo_execution_quotient_frame(
            model.options, queue, model.lanes, 71
        )
        legacy = select_budgeted_frame(
            model.options, queue, model.lanes, 120
        )
        assert selected.represented_ordered_configurations == 100
        assert selected.seed_signature_collisions == 0
        assert selected.attempts <= 71
        assert selected.exact
        assert selected.score == legacy.score
        assert selected.option_indices == legacy.option_indices


def test_cache_and_signature_work_accounting_identities_close():
    options = (
        Option((1, 0), 1.0, ((0.5,), ())),
        Option((1, 0), 1.0, ((0.5,), ())),
        Option((1, 2), 1.0, ((0.25,), (0.5, 1.0))),
    )
    selected = select_fifo_execution_quotient_frame(
        options, (2, 1), lanes=2, budget=5
    )

    assert selected.seed_signature_collisions == 1
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
    assert selected.cache_entries + selected.seed_signature_collisions == (
        selected.attempts
    )
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
    assert selected.cache_key_requirement_coordinate_reads == (
        2 * selected.cache_key_duration_reads
    )
    assert selected.cache_key_completion_offset_reads >= (
        selected.cache_key_duration_reads
    )
    assert selected.upper_bound_calls == (
        selected.frontier_pushes + selected.bound_rejected_prefixes
    )
    assert selected.frontier_pushes - selected.logical_search_pops == (
        selected.exact_stop_frontier_entries_cleared
        + selected.frontier_entries_remaining
    )


def test_one_copy_zero_copy_and_two_copy_seed_cache_paths_are_distinct():
    one_copy_options = (
        Option((1, 0), 1.0, ((1.0,), ())),
        Option((0, 1), 1.0, ((), (1.0,))),
    )
    one_copy = select_fifo_execution_quotient_frame(
        one_copy_options, (1, 1), lanes=2, budget=3
    )
    assert one_copy.paid_search_pops == 0
    assert one_copy.cached_feasible_reuses == 1
    assert one_copy.budget_blocked_cache_miss_lookups == 1

    mixed_seed_options = (
        Option((1, 0), 1.0, ((0.5,), ())),
        Option((1, 0), 1.0, ((0.5,), ())),
        Option((1, 2), 1.0, ((0.25,), (0.5, 1.0))),
    )
    zero_copy = select_fifo_execution_quotient_frame(
        mixed_seed_options, (2, 1), lanes=2, budget=5
    )
    assert zero_copy.cached_infeasible_reuses > 0

    two_copy_options = (
        Option((0, 2), 20.0, ((), (20.0, 20.0))),
        Option((1, 3), 20.0, ((20.0,), (20.0, 20.0, 20.0))),
        Option((2, 0), 1.0, ((1.0, 1.0), ())),
    )
    seed_only = select_fifo_execution_quotient_frame(
        two_copy_options, (3, 6), lanes=2, budget=4
    )
    one_more = select_fifo_execution_quotient_frame(
        two_copy_options, (3, 6), lanes=2, budget=5
    )
    assert not seed_only.exact
    assert one_more.paid_search_pops == 1
    assert one_more.free_cache_hit_pops == 0
    assert one_more.attempts == seed_only.attempts + 1
    assert fifo_execution_signature(two_copy_options, (1,), 2, 2) != (
        fifo_execution_signature(two_copy_options, (1, 1), 2, 2)
    )


def test_free_signature_reuse_keeps_the_encountered_raw_lane_tie_witness():
    options = (
        Option((1, 0), 1.0, ((1.0,), ())),
        Option((1, 0), 1.0, ((1.0,), ())),
        Option((2, 0), 1.0, ((1.0, 1.0), ())),
    )
    selected = select_fifo_execution_quotient_frame(
        options, (3, 1), lanes=2, budget=7
    )
    legacy_exact = select_budgeted_frame(options, (3, 1), lanes=2, budget=8)

    assert fifo_execution_signature(options, (0, 2), 2, 2) == (
        fifo_execution_signature(options, (2, 0), 2, 2)
    )
    assert selected.cache_hits > 0
    assert selected.option_indices == (2, 0)
    assert selected.option_indices != (0, 2)
    assert selected.option_indices == legacy_exact.option_indices
    assert selected.score == legacy_exact.score == 9.0
    assert selected.exact


def test_input_and_completion_offset_validation_is_explicit():
    options = (
        Option((1, 0), 1.0, ((1.0,), ())),
        Option((0, 1), 2.0, ((), (2.0,))),
    )
    with pytest.raises(ValueError, match="lanes must be a positive integer"):
        select_fifo_execution_quotient_frame(options, (1, 1), 0, 3)
    with pytest.raises(ValueError, match="covering 3 seed attempts"):
        select_fifo_execution_quotient_frame(options, (1, 1), 2, 2)
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_fifo_execution_quotient_frame(options, (1, -1), 2, 3)
    with pytest.raises(ValueError, match="idle_duration must be positive"):
        select_fifo_execution_quotient_frame(
            options, (1, 1), 2, 3, idle_duration=0.0
        )

    invalid_options = (
        Option((1,), 1.0, ((1.0,),)),
        Option((1, 0), 1.0, ((), ())),
        Option((1, 0), 1.0, ((1.5,), ())),
        Option((0, 0), 1.0, ((), ())),
    )
    for invalid in invalid_options:
        with pytest.raises(ValueError):
            select_fifo_execution_quotient_frame((invalid,), (1, 1), 2, 2)

    with pytest.raises(ValueError, match="cannot exceed the lane count"):
        fifo_execution_signature(options, (0, 1, 0), 2, 2)
    with pytest.raises(ValueError, match="unknown option"):
        fifo_execution_signature(options, (2,), 2, 2)
    with pytest.raises(ValueError, match="dimensions must be"):
        fifo_execution_signature(options, (), 2, -1)


def test_selector_surface_has_no_arrival_randomness_or_cartesian_enumeration():
    signature = inspect.signature(select_fifo_execution_quotient_frame)
    assert tuple(signature.parameters) == (
        "options", "queues", "lanes", "budget", "idle_duration"
    )
    source = inspect.getsource(
        __import__(
            "algorithm.fifo_execution_quotient_frame_candidates",
            fromlist=["dummy"],
        )
    )
    assert "itertools" not in source
    assert "product(" not in source
    assert "arrival" not in source.lower()
    assert "random" not in source.lower()
