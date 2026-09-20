"""Contracts for legacy-priority residual-bound pruning."""
from __future__ import annotations

from dataclasses import dataclass
import heapq as python_heapq
import itertools
import random

import pytest

import algorithm.budgeted_frame_candidates as legacy_module
import algorithm.residual_sidecar_frame_candidates as sidecar_module
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.residual_sidecar_frame_candidates import (
    ResidualSidecarFrameSelection,
    select_residual_sidecar_frame,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


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


def _realizable(options, queues, selected):
    service = tuple(
        sum(
            options[index].requirements[coordinate]
            for index in selected.option_indices if index is not None
        )
        for coordinate in range(len(queues))
    )
    duration = max((
        1.0,
        *(options[index].duration
          for index in selected.option_indices if index is not None),
    ))
    return (
        service == selected.service
        and all(count <= q for count, q in zip(service, queues))
        and duration == selected.duration
        and selected.score
        == sum(q * count for q, count in zip(queues, service)) / duration
    )


def test_random_cartesian_oracles_and_same_budget_legacy_dominance():
    rng = random.Random(260910)
    for _ in range(14):
        options = []
        for _ in range(5):
            requirements = tuple(rng.randint(0, 3) for _ in range(3))
            if not any(requirements):
                requirements = (1, 0, 0)
            options.append(Option(
                requirements, rng.choice((0.5, 1.0, 2.5, 5.0, 7.0))
            ))
        options = tuple(options)
        queues = tuple(rng.randint(0, 7) for _ in range(3))
        optimum = _cartesian_optimum(options, queues, lanes=3)
        previous_score = -1.0
        previous_upper = float("inf")
        for budget in (6, 8, 14, 26, 120, 500):
            legacy = select_budgeted_frame(options, queues, 3, budget)
            selected = select_residual_sidecar_frame(options, queues, 3, budget)
            assert isinstance(selected, ResidualSidecarFrameSelection)
            assert _realizable(options, queues, selected)
            assert selected.score + 1e-12 >= legacy.score
            assert selected.upper_bound <= legacy.upper_bound + 1e-12
            assert selected.gap <= legacy.gap + 1e-12
            assert selected.score <= optimum + 1e-12
            assert selected.upper_bound + 1e-12 >= optimum
            assert selected.score + 1e-12 >= previous_score
            assert selected.upper_bound <= previous_upper + 1e-12
            assert selected.attempts <= budget
            previous_score = selected.score
            previous_upper = selected.upper_bound
        assert selected.exact
        assert selected.score == pytest.approx(optimum)


def test_four_integrated_bound_regressions_restore_legacy_k26_support():
    model = load_model()
    for queue in ((3, 1, 4), (3, 1, 5), (3, 2, 4), (3, 2, 5)):
        legacy = select_budgeted_frame(model.options, queue, model.lanes, 26)
        selected = select_residual_sidecar_frame(
            model.options, queue, model.lanes, 26
        )
        assert selected.score == legacy.score
        assert selected.option_indices == legacy.option_indices == (6, 7)


class _HeapSpy:
    def __init__(self):
        self.pops = []

    def heappush(self, heap, item):
        python_heapq.heappush(heap, item)

    def heappop(self, heap):
        item = python_heapq.heappop(heap)
        self.pops.append(item[-1])
        return item


def _prefix(prefix):
    return (
        prefix.indices,
        prefix.service,
        prefix.numerator,
        prefix.duration,
        prefix.next_index,
    )


def test_heap_priority_matches_v4_where_integrated_bound_reordered(monkeypatch):
    options = (
        Option((0, 3, 2), 7.0),
        Option((3, 1, 2), 5.0),
        Option((1, 0, 3), 2.0),
        Option((1, 0, 1), 2.0),
    )
    legacy_spy = _HeapSpy()
    sidecar_spy = _HeapSpy()
    monkeypatch.setattr(legacy_module, "heapq", legacy_spy)
    select_budgeted_frame(options, (4, 7, 2), 3, 20)
    monkeypatch.setattr(sidecar_module, "heapq", sidecar_spy)
    select_residual_sidecar_frame(options, (4, 7, 2), 3, 20)

    assert tuple(map(_prefix, sidecar_spy.pops[:6])) == tuple(
        map(_prefix, legacy_spy.pops[:6])
    )


def test_equal_bound_is_retained_but_strict_generation_and_lazy_prunes_are_free():
    equality_options = (
        Option((2, 0, 3), 5.0),
        Option((0, 1, 2), 3.0),
        Option((2, 2, 0), 7.0),
    )
    equality = select_residual_sidecar_frame(
        equality_options, (4, 3, 5), 3, 7
    )
    assert equality.safe_pruned_prefixes_on_generation == 0
    assert equality.safe_pruned_prefixes_on_pop == 0

    generation_options = (
        Option((3, 2, 3), 3.0),
        Option((0, 2, 1), 1.0),
        Option((1, 3, 2), 1.0),
    )
    generated = select_residual_sidecar_frame(
        generation_options, (9, 5, 5), 2, 10
    )
    assert generated.safe_pruned_prefixes_on_generation > 0

    lazy_options = (
        Option((0, 2, 3), 7.0),
        Option((3, 0, 2), 2.0),
        Option((0, 3, 3), 3.0),
        Option((3, 3, 0), 2.0),
        Option((2, 3, 2), 3.0),
    )
    lazy = select_residual_sidecar_frame(
        lazy_options, (3, 5, 9), 2, 24
    )
    assert lazy.safe_pruned_prefixes_on_generation > 0
    assert lazy.safe_pruned_prefixes_on_pop > 0
    for selected in (generated, lazy):
        assert selected.attempts == (
            selected.seed_attempts + selected.paid_search_pops
        )


def test_bound_and_frontier_work_identities_close():
    options = (
        Option((0, 2, 3), 7.0),
        Option((3, 0, 2), 2.0),
        Option((0, 3, 3), 3.0),
        Option((3, 3, 0), 2.0),
        Option((2, 3, 2), 3.0),
    )
    queues = (3, 5, 9)
    selected = select_residual_sidecar_frame(options, queues, 2, 24)
    dimensions = len(queues)
    masks = 1 << dimensions
    positions = len(options) + 1
    thresholds = len({1.0, *(max(1.0, option.duration) for option in options)})

    assert selected.bound_price_vector_count == masks
    assert selected.bound_option_probes == thresholds * positions
    assert selected.bound_dual_table_option_probes == thresholds * masks * positions
    assert selected.bound_dual_mask_evaluations == (
        selected.bound_threshold_evaluations * masks
    )
    assert selected.bound_dual_mask_sums == selected.upper_bound_calls * masks
    assert selected.bound_dual_coordinate_products == dimensions * (
        masks * positions + selected.upper_bound_calls
    )
    assert selected.upper_bound_calls == (
        selected.legacy_rejected_prefixes
        + selected.virtual_legacy_serial_allocations
    )
    assert selected.virtual_legacy_serial_allocations == (
        selected.safe_pruned_prefixes_on_generation + selected.frontier_pushes
    )
    assert selected.frontier_pushes == (
        selected.paid_search_pops
        + selected.safe_pruned_prefixes_on_pop
        + selected.exact_stop_frontier_entries_cleared
        + selected.certificate_frontier_entries
    )
    assert selected.attempts == selected.seed_attempts + selected.paid_search_pops
    assert selected.certificate_frontier_scans == 1
