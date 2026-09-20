"""Tests for the paired residual-bound first-divergence trace."""
from __future__ import annotations

from dataclasses import dataclass
import heapq as python_heapq

import algorithm.budgeted_frame_candidates as legacy_module
import algorithm.residual_capacity_frame_candidates as residual_module
from algorithm.experiments.residual_bound_first_divergence_trace import (
    trace_first_bound_divergence,
)


@dataclass(frozen=True)
class _Option:
    requirements: tuple[int, ...]
    duration: float


class _HeapSpy:
    def __init__(self):
        self.pops = []

    def heappush(self, heap, item):
        python_heapq.heappush(heap, item)

    def heappop(self, heap):
        item = python_heapq.heappop(heap)
        self.pops.append(item[2])
        return item


def _prefix(prefix):
    return (
        prefix.indices,
        prefix.service,
        prefix.numerator,
        prefix.duration,
        prefix.next_index,
    )


def _production_pop_sequences(monkeypatch, options, queue, lanes, budget):
    legacy_spy = _HeapSpy()
    residual_spy = _HeapSpy()
    monkeypatch.setattr(legacy_module, "heapq", legacy_spy)
    legacy_module.select_budgeted_frame(options, queue, lanes, budget)
    monkeypatch.setattr(residual_module, "heapq", residual_spy)
    residual_module.select_residual_capacity_frame(options, queue, lanes, budget)
    return (
        tuple(_prefix(prefix) for prefix in legacy_spy.pops),
        tuple(_prefix(prefix) for prefix in residual_spy.pops),
    )


def test_bound_change_without_paid_search_change_is_not_divergence():
    options = (
        _Option((1, 2, 1), 4.0),
        _Option((2, 2, 3), 4.0),
        _Option((2, 2, 1), 4.0),
        _Option((1, 2, 2), 2.0),
        _Option((3, 0, 3), 2.0),
    )
    traced = trace_first_bound_divergence(
        options, (6, 7, 5), lanes=3, budget=6
    )

    assert traced["classification"] == "no_divergence_within_budget"
    assert traced["first_divergence_attempt"] is None
    assert traced["first_strict_bound_change"]["old_upper_bound"] == 49.5
    assert traced["first_strict_bound_change"]["new_upper_bound"] == 42.5
    assert traced["common_paid_attempt_count"] == 0


def test_heap_reprioritization_matches_independent_production_heap_spies(
    monkeypatch,
):
    options = (
        _Option((0, 3, 2), 7.0),
        _Option((3, 1, 2), 5.0),
        _Option((1, 0, 3), 2.0),
        _Option((1, 0, 1), 2.0),
    )
    queue = (4, 7, 2)
    legacy_pops, residual_pops = _production_pop_sequences(
        monkeypatch, options, queue, lanes=3, budget=20
    )
    first_different = next(
        index for index, pair in enumerate(zip(legacy_pops, residual_pops))
        if pair[0] != pair[1]
    )
    traced = trace_first_bound_divergence(
        options, queue, lanes=3, budget=20
    )

    assert first_different == 5
    assert traced["first_divergence_attempt"] == 11
    assert traced["common_paid_attempt_count"] == first_different
    assert traced["classification"] == "heap_reprioritization"
    assert traced["residual_only_prune_count_before_divergence"] == 0
    divergence = traced["divergence"]
    assert divergence["frontier_membership_equal"]
    assert divergence["legacy_top_present_in_residual_frontier"]
    assert divergence["residual_top_present_in_legacy_frontier"]
    assert tuple(divergence["legacy_top"]["prefix"]["indices"]) == ()
    assert divergence["legacy_top"]["prefix"]["next_index"] == 2
    assert tuple(divergence["residual_top"]["prefix"]["indices"]) == (1,)
    assert divergence["residual_top"]["prefix"]["next_index"] == 1
    assert legacy_pops[first_different] == (
        tuple(divergence["legacy_top"]["prefix"]["indices"]),
        tuple(divergence["legacy_top"]["prefix"]["service"]),
        divergence["legacy_top"]["prefix"]["numerator"],
        divergence["legacy_top"]["prefix"]["duration"],
        divergence["legacy_top"]["prefix"]["next_index"],
    )
    assert residual_pops[first_different] == (
        tuple(divergence["residual_top"]["prefix"]["indices"]),
        tuple(divergence["residual_top"]["prefix"]["service"]),
        divergence["residual_top"]["prefix"]["numerator"],
        divergence["residual_top"]["prefix"]["duration"],
        divergence["residual_top"]["prefix"]["next_index"],
    )


def test_early_prune_records_admission_before_later_paid_divergence(monkeypatch):
    options = (
        _Option((2, 0, 3), 5.0),
        _Option((0, 1, 2), 3.0),
        _Option((2, 2, 0), 7.0),
    )
    queue = (4, 3, 5)
    legacy_pops, residual_pops = _production_pop_sequences(
        monkeypatch, options, queue, lanes=3, budget=26
    )
    common = 0
    while (
        common < len(legacy_pops)
        and common < len(residual_pops)
        and legacy_pops[common] == residual_pops[common]
    ):
        common += 1
    traced = trace_first_bound_divergence(
        options, queue, lanes=3, budget=26
    )

    assert common == 14
    assert traced["first_divergence_attempt"] == 19
    assert traced["classification"] == "early_pruning"
    pruned = traced["first_residual_only_prune"]
    assert pruned["parent_attempt"] == 7
    assert pruned["prefix"]["next_index"] == 3
    assert pruned["new_upper_bound"] == pruned["push_incumbent_score"]
    assert pruned["old_upper_bound"] > pruned["push_incumbent_score"]
    assert pruned["legacy_retained"] and not pruned["residual_retained"]
    assert len(legacy_pops) > common


def test_bound_pair_work_identities_close():
    options = (
        _Option((0, 3, 2), 7.0),
        _Option((3, 1, 2), 5.0),
        _Option((1, 0, 3), 2.0),
        _Option((1, 0, 1), 2.0),
    )
    traced = trace_first_bound_divergence(
        options, (4, 7, 2), lanes=3, budget=20
    )
    work = traced["work"]
    masks = 8
    positions = len(options) + 1
    thresholds = 4

    assert work["generated_prefixes"] == work["bound_pair_calls"]
    assert work["bound_option_probes"] == thresholds * positions
    assert work["dual_table_option_probes"] == thresholds * masks * positions
    assert work["dual_mask_evaluations"] == (
        work["bound_threshold_evaluations"] * masks
    )
    assert work["dual_mask_sums"] == work["bound_pair_calls"] * masks
    assert work["dual_coordinate_products"] == 3 * (
        masks * positions + work["bound_pair_calls"]
    )
