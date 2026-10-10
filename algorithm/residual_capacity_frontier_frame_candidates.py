"""Reuse paid integer capacity tables in original-index canonical completion.

Strict-score proof retains its final exact-slot max-plus vector. A paid prefix
maximum then bounds service attainable with the remaining slots. It relaxes
remaining resources and suffix eligibility, and is reused only when every
positive original requirement at that duration is represented in the table.
Proof, prefix, and canonical DFS consume one integer-state budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
import math
from typing import Any, Sequence

from algorithm.axis_capacity_frontier_score_proof_frame_candidates import (
    AxisCapacityBound, AxisCapacityWork, AxisCapacityStrictScoreProofWork, CAPACITY_COUNTERS,
)
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
class ResidualAxisCapacityBound(AxisCapacityBound):
    exact_slot_numerators: tuple[int, ...]


@dataclass(frozen=True)
class ResidualAxisCapacityWork(AxisCapacityWork):
    merged_values_copied: int = 0


@dataclass(frozen=True)
class ResidualCapacityStrictScoreProofWork(AxisCapacityStrictScoreProofWork):
    capacity: ResidualAxisCapacityWork = field(default_factory=ResidualAxisCapacityWork)


@dataclass(frozen=True)
class ResidualCapacityCompletionWork(ScoreFaceCompletionWork):
    prefix_states: int = 0
    dfs_states_visited: int = 0
    prefix_reads: int = 0
    capacity_queries: int = 0
    capacity_prunes: int = 0
    tables_reused: int = 0
    coverage_coordinate_reads: int = 0


RESIDUAL_CAPACITY_COUNTERS = CAPACITY_COUNTERS + ("merged_values_copied",)


def _residual_axis_capacity_upper_bound(requirements, queue, lanes, budget):
    """Return (upper, work, exact-slot profiles, retained merged vector).

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
    work = dict.fromkeys(RESIDUAL_CAPACITY_COUNTERS, 0)
    amounts = [[] for _ in queue]
    for required in requirements:
        positive = []
        for axis, amount in enumerate(required):
            work["requirement_coordinate_reads"] += 1
            if amount > 0:
                positive.append((axis, amount))
        if len(positive) > 1:
            return None, ResidualAxisCapacityWork(**work), (), ()
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
        return None, ResidualAxisCapacityWork(
            "STATE_BUDGET_EXHAUSTED", True, **work), tuple(profiles), ()
    work["threshold_tables"] = 1
    work["merged_values_copied"] = lanes + 1
    return max(merged), ResidualAxisCapacityWork("COMPLETE", True, **work), tuple(profiles), tuple(merged)


def _prove_residual_capacity_frontier(
    options, queue, lanes, evidence: PaidLpScoreFrontier, budget, idle_duration=1.0,
) -> ResidualCapacityStrictScoreProofWork:
    """Prepay shared tables, then run the frozen box DFS in the remaining cap."""
    groups = {}
    for index, region in enumerate(evidence.regions):
        key = (region.threshold_fraction, region.eligible, region.requirements)
        groups.setdefault(key, []).append(index)
    total = dict.fromkeys(RESIDUAL_CAPACITY_COUNTERS, 0)
    bounds, pruned = [], set()
    attempted = False
    incumbent = Fraction(*evidence.incumbent_score_fraction)

    def capacity(status):
        return ResidualAxisCapacityWork(status, attempted, **total, bounds=tuple(bounds))

    for (threshold_pair, eligible, requirements), indices in groups.items():
        if len(indices) < 2:
            continue
        upper, work, profiles, merged = _residual_axis_capacity_upper_bound(
            requirements, queue, lanes, budget - total["states_visited"])
        for counter in RESIDUAL_CAPACITY_COUNTERS:
            total[counter] += getattr(work, counter)
        attempted |= work.attempted
        if work.status == "STATE_BUDGET_EXHAUSTED":
            return ResidualCapacityStrictScoreProofWork(
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
        bounds.append(ResidualAxisCapacityBound(
            threshold_pair, eligible, requirements, upper, target, tuple(indices), profiles, merged))
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
    return ResidualCapacityStrictScoreProofWork(
        **values, capacity=capacity("COMPLETE" if attempted else "NOT_ATTEMPTED"),
        dfs_states_visited=dfs.states_visited)


def _complete_residual_canonical(options, queue, lanes, baseline, budget, idle_duration, bounds):
    """Keep frozen canonical order; prune only with a covered paid table.

    Prefix maxima are built lazily at their duration. Each stored-vector read
    consumes a state before it occurs. tables_reused counts selected prefix
    attempts, including a partial prefix or an attempt with zero budget.
    """
    if not bounds:
        return _complete_canonical(options, queue, lanes, baseline, budget, idle_duration)
    queue, lanes = tuple(int(q) for q in queue), int(lanes)
    size, dimension = len(options), len(queue)
    requirements = tuple(tuple(int(r) for r in option.requirements) for option in options)
    idle = Fraction.from_float(float(idle_duration))
    durations = tuple(max(idle, Fraction.from_float(float(option.duration))) for option in options)
    incumbent_duration = max((durations[i] for i, n in enumerate(baseline.option_counts) if n),
                             default=idle)
    proven_score = Fraction(sum(q * s for q, s in zip(queue, baseline.service, strict=True))) / incumbent_duration
    if float(proven_score) != baseline.score or float(incumbent_duration) != baseline.duration:
        raise ArithmeticError("stage-one score certificate disagrees with its action")
    weights = tuple(sum(q * r for q, r in zip(queue, required, strict=True))
                    for required in requirements)
    work = {name: 0 for name in (
        "states_visited", "memo_hits", "count_branches", "bound_prunes", "gcd_prunes",
        "duration_prunes", "failed_states_cached", "thresholds_checked",
        "noninteger_thresholds", "suffix_coordinate_reads", "resource_coordinate_reads",
        "prefix_states", "dfs_states_visited", "prefix_reads", "capacity_queries",
        "capacity_prunes", "tables_reused", "coverage_coordinate_reads")}

    def finish(counts, threshold):
        service = tuple(sum(n * requirements[i][j] for i, n in enumerate(counts))
                        for j in range(dimension))
        actual_duration = max((durations[i] for i, n in enumerate(counts) if n), default=idle)
        actual_score = Fraction(sum(q * s for q, s in zip(queue, service, strict=True))) / actual_duration
        if (actual_duration != threshold or actual_score != proven_score or sum(counts) > lanes
                or any(s > q for s, q in zip(service, queue, strict=True))):
            raise ArithmeticError("integer completion contradicts the certified score")
        indices = tuple(i for i, n in enumerate(counts) for _ in range(n)) + (None,) * (lanes - sum(counts))
        selected = replace(baseline, option_counts=counts, option_indices=indices, service=service,
                           duration=float(threshold), score=float(proven_score), upper_bound=float(proven_score),
                           gap=0.0, epsilon=0.0, exact=True, score_exact=True, canonical_exact=True,
                           canonical_tie_regions_remaining=0, frontier_regions_remaining=0,
                           inherited_unsolved_regions=0)
        return selected, ResidualCapacityCompletionWork(
            "COMPLETE", True, True, budget, **work,
            action_changed=(counts != baseline.option_counts or selected.duration != baseline.duration),
            score_fraction=(proven_score.numerator, proven_score.denominator))

    # Feasibility of a non-idle option implies positive service against Q, so a
    # zero optimum has the unique all-idle feasible action at score zero.
    if proven_score == 0:
        return _complete_canonical(options, queue, lanes, baseline, budget, idle_duration)

    thresholds = sorted({d for d in durations if d <= incumbent_duration})
    candidate_thresholds = {threshold for threshold in thresholds
                            if (proven_score * threshold).denominator == 1}
    candidate_bounds = tuple(bound for bound in bounds
                             if Fraction(*bound.threshold_fraction) in candidate_thresholds)
    if not candidate_bounds:
        return _complete_canonical(options, queue, lanes, baseline, budget, idle_duration)

    class StateBudgetExhausted(Exception):
        pass

    for threshold in thresholds:
        work["thresholds_checked"] += 1
        numerator = proven_score * threshold
        if numerator.denominator != 1:
            work["noninteger_thresholds"] += 1
            continue
        target = numerator.numerator
        # Positions are always original indices; ineligible options have count
        # zero. A state's remaining score is determined by its resource vector,
        # rather than a second score coordinate in its memo key.
        allowed = tuple(d <= threshold for d in durations)
        capacity_prefix = None
        for bound in candidate_bounds:
            if Fraction(*bound.threshold_fraction) != threshold:
                continue
            represented = set()
            for required in bound.requirements:
                represented.add(tuple(value for value in required))
                work["coverage_coordinate_reads"] += len(required)
            covered = True
            for required, eligible in zip(requirements, allowed, strict=True):
                if not eligible:
                    continue
                work["coverage_coordinate_reads"] += dimension
                if any(required) and required not in represented:
                    covered = False
                    break
            if not covered:
                continue
            work["tables_reused"] += 1
            prefix, upper = [], -1
            try:
                for slots in range(lanes + 1):
                    if work["states_visited"] == budget:
                        raise StateBudgetExhausted
                    work["states_visited"] += 1
                    work["prefix_states"] += 1
                    work["prefix_reads"] += 1
                    upper = max(upper, bound.exact_slot_numerators[slots])
                    prefix.append(upper)
            except StateBudgetExhausted:
                return baseline, ResidualCapacityCompletionWork(
                    "STATE_BUDGET_EXHAUSTED", True, False, budget, **work,
                    score_fraction=(proven_score.numerator, proven_score.denominator))
            capacity_prefix = tuple(prefix)
            break
        max_weight = [0] * (size + 1)
        divisors = [0] * (size + 1)
        max_requirement = [(0,) * dimension for _ in range(size + 1)]
        can_attain = [False] * (size + 1)
        for pos in reversed(range(size)):
            max_weight[pos], divisors[pos] = max_weight[pos + 1], divisors[pos + 1]
            max_requirement[pos], can_attain[pos] = max_requirement[pos + 1], can_attain[pos + 1]
            if allowed[pos]:
                max_weight[pos] = max(max_weight[pos], weights[pos])
                divisors[pos] = math.gcd(divisors[pos], weights[pos])
                max_requirement[pos] = tuple(max(a, b) for a, b in
                                             zip(requirements[pos], max_requirement[pos], strict=True))
                work["suffix_coordinate_reads"] += 2 * dimension
                can_attain[pos] |= durations[pos] == threshold
        failed = set()
        choices = {}

        def feasible(pos, slots, remaining, attained):
            state = (pos, slots, remaining, attained)
            if state in failed:
                work["memo_hits"] += 1
                return False
            if work["states_visited"] == budget:
                raise StateBudgetExhausted
            work["states_visited"] += 1
            work["dfs_states_visited"] += 1
            needed = target - sum(q * (q - r) for q, r in zip(queue, remaining, strict=True))
            work["resource_coordinate_reads"] += dimension

            reason = None
            if needed == 0 and attained:
                choices[state] = None
                return True
            if needed < 0 or pos == size or slots == 0:
                reason = "bound_prunes"
            elif not attained and not can_attain[pos]:
                reason = "duration_prunes"
            else:
                resource_limit = sum(q * min(r, slots * m) for q, r, m in
                                     zip(queue, remaining, max_requirement[pos], strict=True))
                work["resource_coordinate_reads"] += 2 * dimension
                if needed > min(slots * max_weight[pos], resource_limit):
                    reason = "bound_prunes"
                else:
                    if capacity_prefix is not None:
                        work["capacity_queries"] += 1
                        if needed > capacity_prefix[slots]:
                            work["capacity_prunes"] += 1
                            reason = "bound_prunes"
                    if reason is None and (not divisors[pos] or needed % divisors[pos]):
                        reason = "gcd_prunes"
            if reason is not None:
                work[reason] += 1
            else:
                count_limit = 0
                if allowed[pos]:
                    count_limit = min(slots, *(r // a for r, a in
                                              zip(remaining, requirements[pos], strict=True) if a))
                    work["resource_coordinate_reads"] += 2 * dimension
                    if weights[pos]:
                        count_limit = min(count_limit, needed // weights[pos])
                for count in range(count_limit, -1, -1):
                    work["count_branches"] += 1
                    residual = tuple(r - count * a for r, a in zip(remaining, requirements[pos], strict=True))
                    work["resource_coordinate_reads"] += 2 * dimension
                    child = (pos + 1, slots - count, residual,
                             attained or (count > 0 and durations[pos] == threshold))
                    if feasible(*child):
                        choices[state] = count
                        return True
            failed.add(state)
            work["failed_states_cached"] += 1
            return False

        try:
            root = (0, lanes, queue, False)
            found = feasible(*root)
        except StateBudgetExhausted:
            return baseline, ResidualCapacityCompletionWork(
                "STATE_BUDGET_EXHAUSTED", True, False, budget, **work,
                score_fraction=(proven_score.numerator, proven_score.denominator))
        if found:
            counts = [0] * size
            state = root
            while choices[state] is not None:
                pos, slots, remaining, attained = state
                count = choices[state]
                counts[pos] = count
                state = (pos + 1, slots - count,
                         tuple(r - count * a for r, a in zip(remaining, requirements[pos], strict=True)),
                         attained or (count > 0 and durations[pos] == threshold))
            return finish(tuple(counts), threshold)
    raise ArithmeticError("no canonical action exists at the stage-one certified score")


def _residual_frontier_probe(options, queue, lanes, stage, budget, idle_duration, evidence):
    baseline = stage.selection
    proof = _prove_residual_capacity_frontier(options, queue, lanes, evidence, budget, idle_duration)
    certified = baseline
    if proof.complete:
        certified = replace(baseline, score_exact=True, upper_bound=baseline.score, gap=0.0, epsilon=0.0)
        selected, completion = _complete_residual_canonical(
            options, queue, lanes, certified, budget - proof.states_visited, idle_duration,
            proof.capacity.bounds)
    else:
        selected = baseline
        completion = ScoreFaceCompletionWork(
            "SKIPPED_SCORE_UNPROVEN", False, False, budget - proof.states_visited)
    return StrictScoreProofCanonicalCompletionSelection(
        selected, baseline, certified, stage.preprocessing, stage.capacity_propagation,
        proof, completion, budget)


def select_branch_reserved_residual_capacity_frontier_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> FrontierScoreProofRoutedSelection:
    return _select_residual_capacity_frontier_frame(
        BranchReservedIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)


def select_dual_reuse_residual_capacity_frontier_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> FrontierScoreProofRoutedSelection:
    return _select_residual_capacity_frontier_frame(
        DualReuseIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)


def _select_residual_capacity_frontier_frame(
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
        probe = _residual_frontier_probe(options, queue, lanes, final_lp, probe_budget, idle_duration, evidence)
        selected = probe.selection
        return finish(selected, final_lp, probe, True,
                      "FRONTIER_PROBE_COMPLETE" if selected.canonical_exact else "FRONTIER_PROBE_INCOMPLETE",
                      "frontier_late")
    return finish(selected, final_lp, probe, True,
                  "LP_RESUME_COMPLETE" if selected.canonical_exact else "LP_RESUME_INCOMPLETE", probe_stage)
