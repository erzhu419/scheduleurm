"""Reuse an integer, shared-slot axis bound across paid strict-score boxes."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any, Sequence

from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseIntegerCapacitySearch
from algorithm.frontier_score_proof_certificate_routed_frame_candidates import (
    FrontierScoreProofRoutedSelection, PaidLpScoreFrontier, capture_strict_frontier,
    _prove_frontier_no_strict_improvement,
)
from algorithm.resumable_probe_frame_candidates import _probe
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import (
    StrictScoreProofCanonicalCompletionSelection, StrictScoreProofWork, _complete_canonical,
)


@dataclass(frozen=True)
class AxisCapacityBound:
    threshold_fraction: tuple[int, int]
    eligible: tuple[int, ...]
    requirements: tuple[tuple[int, ...], ...]
    upper_numerator: int
    target_numerator: int
    box_indices: tuple[int, ...]
    axis_service_profiles: tuple[tuple[int, ...], ...]


CAPACITY_COUNTERS = (
    "states_visited", "axis_states", "merge_states", "threshold_tables", "cache_hits",
    "bound_checks", "boxes_pruned", "bitset_shifts", "bitset_ors", "bitset_masks",
    "service_reads", "merge_candidates", "requirement_coordinate_reads",
)


@dataclass(frozen=True)
class AxisCapacityWork:
    status: str = "NOT_ATTEMPTED"
    attempted: bool = False
    states_visited: int = 0
    axis_states: int = 0
    merge_states: int = 0
    threshold_tables: int = 0
    cache_hits: int = 0
    bound_checks: int = 0
    boxes_pruned: int = 0
    bitset_shifts: int = 0
    bitset_ors: int = 0
    bitset_masks: int = 0
    service_reads: int = 0
    merge_candidates: int = 0
    requirement_coordinate_reads: int = 0
    bounds: tuple[AxisCapacityBound, ...] = ()


@dataclass(frozen=True)
class AxisCapacityStrictScoreProofWork(StrictScoreProofWork):
    capacity: AxisCapacityWork = field(default_factory=AxisCapacityWork)
    dfs_states_visited: int = 0


def _axis_capacity_upper_bound(requirements, queue, lanes, budget):
    """Return (upper, work, exact-slot service profiles), or upper=None.

    Each resource's bitsets represent service reachable using exactly k slots.
    Option-by-option unbounded recurrence allows at most the shared lane count
    copies; service beyond its original queue is removed. Max-plus merging
    enforces the common slot capacity. Box bounds and exact-duration attainment
    are relaxed, so the result is only an upper bound, never an action witness.

    Initial seeds and every computed option/lane and merge output cell consume
    a shared-budget state, including unreachable cells. Bitset and candidate
    operations are recorded separately. service_reads counts bit_length reads.
    """
    queue = tuple(queue)
    dimension = len(queue)
    work = dict.fromkeys(CAPACITY_COUNTERS, 0)
    amounts = [[] for _ in queue]
    for required in requirements:
        positive = []
        for axis, amount in enumerate(required):
            work["requirement_coordinate_reads"] += 1
            if amount > 0:
                positive.append((axis, amount))
        if len(positive) > 1:
            return None, AxisCapacityWork(**work), ()
        if positive:
            axis, amount = positive[0]
            amounts[axis].append(amount)

    class StateBudgetExhausted(Exception):
        pass

    def visit(counter):
        if work["states_visited"] == budget:
            raise StateBudgetExhausted
        work["states_visited"] += 1
        work[counter] += 1

    profiles = []
    try:
        for axis in range(dimension):
            visit("axis_states")
            previous = [0] * (lanes + 1)
            previous[0] = 1
            mask = (1 << (queue[axis] + 1)) - 1
            work["bitset_shifts"] += 1
            for amount in amounts[axis]:
                current = [0] * (lanes + 1)
                for slots in range(lanes + 1):
                    visit("axis_states")
                    shifted = 0
                    if slots:
                        shifted = current[slots - 1] << amount
                        work["bitset_shifts"] += 1
                    current[slots] = (previous[slots] | shifted) & mask
                    work["bitset_ors"] += 1
                    work["bitset_masks"] += 1
                previous = current
            profiles.append(tuple(bits.bit_length() - 1 for bits in previous))
            work["service_reads"] += lanes + 1

        visit("merge_states")
        merged = [-1] * (lanes + 1)
        merged[0] = 0
        for axis, profile in enumerate(profiles):
            current = [-1] * (lanes + 1)
            for slots in range(lanes + 1):
                visit("merge_states")
                for used in range(slots + 1):
                    work["merge_candidates"] += 1
                    if merged[slots - used] >= 0 and profile[used] >= 0:
                        value = merged[slots - used] + queue[axis] * profile[used]
                        current[slots] = max(current[slots], value)
            merged = current
    except StateBudgetExhausted:
        return None, AxisCapacityWork(
            "STATE_BUDGET_EXHAUSTED", True, **work), tuple(profiles)
    work["threshold_tables"] = 1
    return max(merged), AxisCapacityWork("COMPLETE", True, **work), tuple(profiles)


def _prove_axis_capacity_frontier(
    options, queue, lanes, evidence: PaidLpScoreFrontier, budget, idle_duration=1.0,
) -> AxisCapacityStrictScoreProofWork:
    """Prepay shared tables, then run the frozen box DFS in the remaining cap."""
    groups = {}
    for index, region in enumerate(evidence.regions):
        key = (region.threshold_fraction, region.eligible, region.requirements)
        groups.setdefault(key, []).append(index)
    total = dict.fromkeys(CAPACITY_COUNTERS, 0)
    bounds, pruned = [], set()
    attempted = False
    incumbent = Fraction(*evidence.incumbent_score_fraction)

    def capacity(status):
        return AxisCapacityWork(status, attempted, **total, bounds=tuple(bounds))

    for (threshold_pair, eligible, requirements), indices in groups.items():
        if len(indices) < 2:
            continue
        upper, work, profiles = _axis_capacity_upper_bound(
            requirements, queue, lanes, budget - total["states_visited"])
        for counter in CAPACITY_COUNTERS:
            total[counter] += getattr(work, counter)
        attempted |= work.attempted
        if work.status == "STATE_BUDGET_EXHAUSTED":
            return AxisCapacityStrictScoreProofWork(
                "STATE_BUDGET_EXHAUSTED", True, False, budget,
                evidence.incumbent_score_fraction, states_visited=total["states_visited"],
                bound_prunes=total["boxes_pruned"], thresholds_checked=total["boxes_pruned"],
                capacity=capacity("STATE_BUDGET_EXHAUSTED"), dfs_states_visited=0)
        if not work.attempted:
            continue
        threshold_target = incumbent * Fraction(*threshold_pair)
        target = threshold_target.numerator // threshold_target.denominator + 1
        total["bound_checks"] += len(indices)
        total["cache_hits"] += len(indices) - 1
        bounds.append(AxisCapacityBound(
            threshold_pair, eligible, requirements, upper, target, tuple(indices), profiles))
        if upper < target:
            pruned.update(indices)
            total["boxes_pruned"] += len(indices)

    remaining = replace(evidence, regions=tuple(region for index, region in
        enumerate(evidence.regions) if index not in pruned))
    dfs = _prove_frontier_no_strict_improvement(
        options, queue, lanes, remaining, budget - total["states_visited"], idle_duration)
    values = {item.name: getattr(dfs, item.name) for item in fields(StrictScoreProofWork)}
    values.update(state_budget=budget, states_visited=total["states_visited"] + dfs.states_visited,
                  bound_prunes=total["boxes_pruned"] + dfs.bound_prunes,
                  thresholds_checked=total["boxes_pruned"] + dfs.thresholds_checked)
    return AxisCapacityStrictScoreProofWork(
        **values, capacity=capacity("COMPLETE" if attempted else "NOT_ATTEMPTED"),
        dfs_states_visited=dfs.states_visited)


def _axis_frontier_probe(options, queue, lanes, stage, budget, idle_duration, evidence):
    baseline = stage.selection
    proof = _prove_axis_capacity_frontier(options, queue, lanes, evidence, budget, idle_duration)
    certified = baseline
    if proof.complete:
        certified = replace(baseline, score_exact=True, upper_bound=baseline.score, gap=0.0, epsilon=0.0)
        selected, completion = _complete_canonical(
            options, queue, lanes, certified, budget - proof.states_visited, idle_duration)
    else:
        selected = baseline
        completion = ScoreFaceCompletionWork(
            "SKIPPED_SCORE_UNPROVEN", False, False, budget - proof.states_visited)
    return StrictScoreProofCanonicalCompletionSelection(
        selected, baseline, certified, stage.preprocessing, stage.capacity_propagation,
        proof, completion, budget)


def select_branch_reserved_axis_capacity_frontier_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> FrontierScoreProofRoutedSelection:
    return _select_axis_capacity_frontier_frame(
        BranchReservedIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)


def select_dual_reuse_axis_capacity_frontier_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> FrontierScoreProofRoutedSelection:
    return _select_axis_capacity_frontier_frame(
        DualReuseIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)


def _select_axis_capacity_frontier_frame(
    search_type, options, queues, lanes, initial_lp_budget, lp_node_budget,
    integer_state_budget, idle_duration,
):
    for value, name in ((initial_lp_budget, "initial_lp_budget"),
                        (lp_node_budget, "lp_node_budget"), (integer_state_budget, "integer_state_budget")):
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    initial_budget, total_budget, probe_budget = map(int, (initial_lp_budget, lp_node_budget, integer_state_budget))
    if total_budget < initial_budget:
        raise ValueError("total LP budget must cover the initial LP budget")
    options, queue = tuple(options), tuple(queues)
    search = search_type(options, queue, lanes, idle_duration=idle_duration)
    initial = search.advance_to(initial_budget)
    evidence = None

    def finish(selected, final_lp, probe, resumed, status, probe_stage):
        return FrontierScoreProofRoutedSelection(
            selected, initial, final_lp, probe, resumed, initial_budget, total_budget,
            probe_budget, status, probe_stage, evidence)

    if initial.selection.canonical_exact:
        return finish(initial.selection, initial, None, False, "LP_INITIAL_COMPLETE", "none")
    probe, probe_stage = None, "none"
    if initial.selection.score_exact:
        probe = _probe(options, queue, lanes, initial, probe_budget, idle_duration)
        probe_stage = "initial"
        if probe.selection.canonical_exact:
            return finish(probe.selection, initial, probe, False, "INTEGER_PROBE_COMPLETE", probe_stage)
    final_lp = search.advance_to(total_budget)
    selected = final_lp.selection
    if not selected.canonical_exact and selected.score_exact and probe is None:
        probe = _probe(options, queue, lanes, final_lp, probe_budget, idle_duration)
        probe_stage = "resumed"
        selected = probe.selection
        if selected.canonical_exact:
            return finish(selected, final_lp, probe, True, "INTEGER_PROBE_COMPLETE", probe_stage)
    if not selected.canonical_exact and not selected.score_exact and probe is None:
        evidence = capture_strict_frontier(search)
        probe = _axis_frontier_probe(options, queue, lanes, final_lp, probe_budget, idle_duration, evidence)
        selected = probe.selection
        return finish(selected, final_lp, probe, True,
                      "FRONTIER_PROBE_COMPLETE" if selected.canonical_exact else "FRONTIER_PROBE_INCOMPLETE",
                      "frontier_late")
    return finish(selected, final_lp, probe, True,
                  "LP_RESUME_COMPLETE" if selected.canonical_exact else "LP_RESUME_INCOMPLETE", probe_stage)
