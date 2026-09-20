"""Contracts for pruning strictly slower equal-requirement local options."""
from __future__ import annotations

from dataclasses import dataclass
import itertools

import pytest

from algorithm.experiments.aligned_hardware_frame_experiment import (
    exact_search_budget,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.requirement_dominance_frame_candidates import (
    RequirementDominanceFrameSelection,
    reduce_requirement_dominated_options,
    select_requirement_dominance_frame,
)


@dataclass(frozen=True)
class Option:
    action_id: str
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...] = ()


def _cartesian_optimum(options, queues, lanes, idle_duration=1.0):
    optimum = 0.0
    for indices in itertools.product(
        (None, *range(len(options))), repeat=lanes
    ):
        service = tuple(
            sum(
                options[index].requirements[coordinate]
                for index in indices
                if index is not None
            )
            for coordinate in range(len(queues))
        )
        if any(
            count > available
            for count, available in zip(service, queues, strict=True)
        ):
            continue
        duration = max((
            idle_duration,
            *(
                options[index].duration
                for index in indices
                if index is not None
            ),
        ))
        support = sum(
            queue * served
            for queue, served in zip(queues, service, strict=True)
        ) / duration
        optimum = max(optimum, support)
    return optimum


def _assert_realizable(selected, options, queues, lanes, idle_duration=1.0):
    assert len(selected.option_indices) == lanes
    assert all(
        index is None or 0 <= index < len(options)
        for index in selected.option_indices
    )
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
    assert all(
        served <= available
        for served, available in zip(service, queues, strict=True)
    )
    assert selected.duration == duration
    assert selected.score == pytest.approx(
        sum(
            queue * served
            for queue, served in zip(queues, service, strict=True)
        ) / duration
    )


def test_reducer_removes_only_strictly_longer_effective_duration_options():
    options = (
        Option("floor-fast", (1, 0), 0.5),
        Option("floor-tie", (1, 0), 0.8),
        Option("strictly-slow", (1, 0), 2.0),
        Option("equal-minimum-a", (0, 1), 3.0),
        Option("equal-minimum-b", (0, 1), 3.0),
        Option("different-requirements", (1, 1), 8.0),
    )

    reduced = reduce_requirement_dominated_options(options, idle_duration=1.0)

    assert reduced.original_option_count == 6
    assert reduced.retained_option_count == 5
    assert reduced.removed_option_count == 1
    assert reduced.retained_original_indices == (0, 1, 3, 4, 5)
    assert reduced.removed_original_indices == (2,)
    assert reduced.options == tuple(options[index] for index in (0, 1, 3, 4, 5))
    assert reduced.reduction_option_probes == len(options)
    assert reduced.reduction_requirement_coordinate_reads == 2 * len(options)
    assert reduced.reduction_duration_comparisons == 2 * len(options) - 3


def test_small_catalogues_preserve_exhaustive_maximum_support():
    options = (
        Option("a-fast", (1, 0), 2.0),
        Option("a-slow", (1, 0), 6.0),
        Option("a-equal-fast", (1, 0), 2.0),
        Option("b", (0, 1), 1.5),
        Option("mixed", (1, 1), 4.0),
    )
    reduced = reduce_requirement_dominated_options(options)
    assert reduced.removed_original_indices == (1,)

    for lanes in (1, 2, 3):
        exact_budget = exact_search_budget(reduced.retained_option_count, lanes)
        for queues in itertools.product(range(4), repeat=2):
            original_optimum = _cartesian_optimum(options, queues, lanes)
            reduced_optimum = _cartesian_optimum(
                reduced.options, queues, lanes
            )
            selected = select_requirement_dominance_frame(
                options, queues, lanes, exact_budget
            )

            assert reduced_optimum == pytest.approx(original_optimum)
            assert isinstance(selected, RequirementDominanceFrameSelection)
            _assert_realizable(selected, options, queues, lanes)
            assert selected.exact
            assert selected.score == pytest.approx(original_optimum)
            assert selected.upper_bound == pytest.approx(selected.score)
            assert all(
                index not in reduced.removed_original_indices
                for index in selected.option_indices
                if index is not None
            )


def test_selector_maps_every_lane_back_to_the_original_catalogue():
    options = (
        Option("removed-before-retained", (1, 0), 8.0),
        Option("retained-a", (1, 0), 2.0),
        Option("retained-b", (0, 1), 3.0),
    )
    exact_budget = exact_search_budget(2, 2)
    selected = select_requirement_dominance_frame(
        options, (1, 1), lanes=2, budget=exact_budget
    )

    _assert_realizable(selected, options, (1, 1), 2)
    assert selected.exact
    assert set(selected.option_indices) == {1, 2}
    assert 0 not in selected.option_indices
    assert selected.retained_original_indices == (1, 2)
    assert selected.removed_original_indices == (0,)


def test_registered_model_removes_only_option_one_but_not_its_flow_semantics():
    model = load_model()
    reduced = reduce_requirement_dominated_options(model.options)
    faster = model.options[0]
    removed = model.options[1]

    assert len(model.options) == reduced.original_option_count == 9
    assert reduced.retained_option_count == 8
    assert reduced.removed_option_count == 1
    assert reduced.retained_original_indices == (0, 2, 3, 4, 5, 6, 7, 8)
    assert reduced.removed_original_indices == (1,)
    assert faster.requirements == removed.requirements == (1, 0, 1)
    assert faster.duration < removed.duration
    assert faster.completion_offsets != removed.completion_offsets
    assert removed.completion_offsets[2][0] < faster.completion_offsets[2][0]
    assert (len(model.options) + 1) ** model.lanes == 100
    assert (reduced.retained_option_count + 1) ** model.lanes == 81
    assert exact_search_budget(len(model.options), model.lanes) == 120
    assert exact_search_budget(reduced.retained_option_count, model.lanes) == 99


def test_budget_and_work_accounting_exclude_removed_options_without_escape():
    options = (
        Option("slow-a", (1, 0), 7.0),
        Option("fast-a", (1, 0), 2.0),
        Option("b", (0, 1), 3.0),
        Option("mixed", (1, 1), 5.0),
    )
    queues = (2, 2)
    reduced = reduce_requirement_dominated_options(options)
    seed_budget = reduced.retained_option_count + 1
    exact_budget = exact_search_budget(reduced.retained_option_count, 2)
    optimum = _cartesian_optimum(options, queues, lanes=2)
    previous_score = -1.0
    previous_upper = float("inf")

    for budget in (seed_budget, seed_budget + 1, 10, exact_budget):
        selected = select_requirement_dominance_frame(
            options, queues, lanes=2, budget=budget
        )

        _assert_realizable(selected, options, queues, 2)
        assert selected.original_option_count == len(options)
        assert selected.retained_option_count == len(reduced.options)
        assert selected.removed_option_count == 1
        assert selected.retained_original_indices == (1, 2, 3)
        assert selected.removed_original_indices == (0,)
        assert 0 not in selected.option_indices
        assert selected.seed_attempts == seed_budget
        assert selected.local_score_evaluations == len(reduced.options)
        assert selected.attempts <= budget
        assert selected.feasible_candidate_count <= selected.attempts
        assert selected.score <= optimum + 1e-12
        assert selected.upper_bound + 1e-12 >= optimum
        assert selected.gap == pytest.approx(
            selected.upper_bound - selected.score
        )
        assert selected.epsilon == pytest.approx(
            selected.gap / sum(queues)
        )
        assert selected.score + 1e-12 >= previous_score
        assert selected.upper_bound <= previous_upper + 1e-12
        assert selected.reduction_option_probes == len(options)
        assert selected.reduction_requirement_coordinate_reads == 2 * len(options)
        previous_score = selected.score
        previous_upper = selected.upper_bound

    assert selected.exact
    assert selected.score == pytest.approx(optimum)
