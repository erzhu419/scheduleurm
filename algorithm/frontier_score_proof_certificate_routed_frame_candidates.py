"""Complete an open LP score using only its paid, still-live strict-score boxes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Any, Sequence

from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
from algorithm.certificate_routed_frame_candidates import CertificateRoutedSelection
from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseIntegerCapacitySearch
from algorithm.resumable_probe_frame_candidates import _probe
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import (
    StrictScoreProofCanonicalCompletionSelection, StrictScoreProofWork, _complete_canonical,
)


@dataclass(frozen=True)
class FrontierExportWork:
    heap_entries_read: int
    group_ids_copied: int
    requirement_rows_copied: int
    requirement_coordinates_copied: int
    box_count_values_copied: int
    exact_duration_positions_copied: int
    representative_options_copied: int
    fraction_pairs_copied: int


@dataclass(frozen=True)
class FrontierProofRegion:
    threshold_fraction: tuple[int, int]
    bound_fraction: tuple[int, int]
    eligible: tuple[int, ...]
    requirements: tuple[tuple[int, ...], ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    exact_duration_positions: tuple[int, ...]
    representative_options: tuple[int, ...]


@dataclass(frozen=True)
class PaidLpScoreFrontier:
    incumbent_score_fraction: tuple[int, int]
    root_count: int
    raw_region_count: int
    live_region_count: int
    live_tie_region_count: int
    regions: tuple[FrontierProofRegion, ...]
    export_work: FrontierExportWork


@dataclass(frozen=True)
class FrontierScoreProofRoutedSelection(CertificateRoutedSelection):
    frontier_evidence: PaidLpScoreFrontier | None


def capture_strict_frontier(search) -> PaidLpScoreFrontier:
    """Read the suspended native frontier without advancing or snapshotting it.

    Every root has already been processed. Heap boxes with an exact bound
    above the incumbent cover all remaining strict improvements, including
    unsolved boxes and failed LP calls. Score-equal boxes remain relevant to
    the separate original-option canonical completion, rather than this proof.
    Export order is the existing raw heap order.
    """
    local = search._search.gi_frame.f_locals
    incumbent = local["best"].score
    heap = local["frontier"]
    requirements, representatives = local["requirements"], local["representatives"]
    paid = search._snapshot.selection
    regions = []
    group_copies = coordinate_copies = exact_copies = 0
    for entry in heap:
        region = entry[-1]
        if region.bound <= incumbent:
            continue
        eligible = tuple(index for index in region.eligible)
        local_requirements = tuple(tuple(value for value in requirements[index]) for index in eligible)
        mapping = representatives[region.threshold_index]
        regions.append(FrontierProofRegion(
            (region.threshold.numerator, region.threshold.denominator),
            (region.bound.numerator, region.bound.denominator), eligible, local_requirements,
            tuple(value for value in region.lower), tuple(value for value in region.upper),
            tuple(position for position in region.exact_duration_positions),
            tuple(mapping[index] for index in eligible)))
        group_copies += len(eligible)
        coordinate_copies += sum(len(row) for row in local_requirements)
        exact_copies += len(region.exact_duration_positions)
    return PaidLpScoreFrontier(
        (incumbent.numerator, incumbent.denominator), local["root_count"], len(heap),
        paid.frontier_regions_remaining, paid.canonical_tie_regions_remaining, tuple(regions),
        FrontierExportWork(len(heap), group_copies, group_copies, coordinate_copies,
                           2 * group_copies, exact_copies, group_copies, 1 + 2 * len(regions)))


def _prove_frontier_no_strict_improvement(
    options, queue, lanes, evidence: PaidLpScoreFrontier, budget, idle_duration=1.0,
) -> StrictScoreProofWork:
    """Search each exported integer box under one global visited-state cap.

    Mandatory lower counts are allocated before searching extra counts.
    Failed-state memoization stays within its box, whose upper bounds and
    duration-attainment condition remain fixed. All score and pruning
    arithmetic is integer or rational.
    """
    queue, options = tuple(queue), tuple(options)
    dimension = len(queue)
    incumbent = Fraction(*evidence.incumbent_score_fraction)
    idle = Fraction.from_float(float(idle_duration))
    work = {name: 0 for name in (
        "states_visited", "memo_hits", "count_branches", "bound_prunes", "duration_prunes",
        "gcd_prunes", "failed_states_cached", "thresholds_checked",
        "suffix_coordinate_reads", "resource_coordinate_reads")}

    def finish(status, witness=None):
        return StrictScoreProofWork(status, True, status == "COMPLETE", budget,
                                    evidence.incumbent_score_fraction, **work, witness=witness)

    class StateBudgetExhausted(Exception):
        pass

    def visit():
        if work["states_visited"] == budget:
            raise StateBudgetExhausted
        work["states_visited"] += 1

    try:
        for region in evidence.regions:
            work["thresholds_checked"] += 1
            threshold = Fraction(*region.threshold_fraction)
            numerator = incumbent * threshold
            target = numerator.numerator // numerator.denominator + 1
            numerator_bound = Fraction(*region.bound_fraction) * threshold
            if numerator_bound.numerator // numerator_bound.denominator < target:
                visit()
                work["bound_prunes"] += 1
                continue

            requirements, size = region.requirements, len(region.eligible)
            weights = tuple(sum(q * r for q, r in zip(queue, required, strict=True))
                            for required in requirements)
            work["suffix_coordinate_reads"] += 2 * size * dimension
            caps = tuple(right - left for left, right in zip(region.lower, region.upper, strict=True))
            base_service = tuple(sum(count * required[c]
                                     for count, required in zip(region.lower, requirements, strict=True))
                                 for c in range(dimension))
            remaining = tuple(q - done for q, done in zip(queue, base_service, strict=True))
            work["resource_coordinate_reads"] += size * dimension + 2 * dimension
            exact_positions = set(region.exact_duration_positions)
            attained = threshold == idle or any(region.lower[position] for position in exact_positions)
            max_weight, divisors, box_weight = [0] * (size + 1), [0] * (size + 1), [0] * (size + 1)
            max_requirement = [(0,) * dimension for _ in range(size + 1)]
            can_attain = [False] * (size + 1)
            for pos in reversed(range(size)):
                max_weight[pos], divisors[pos] = max_weight[pos + 1], divisors[pos + 1]
                box_weight[pos] = box_weight[pos + 1] + caps[pos] * weights[pos]
                max_requirement[pos], can_attain[pos] = max_requirement[pos + 1], can_attain[pos + 1]
                if caps[pos]:
                    max_weight[pos] = max(max_weight[pos], weights[pos])
                    divisors[pos] = math.gcd(divisors[pos], weights[pos])
                    max_requirement[pos] = tuple(max(left, right) for left, right in
                                                 zip(requirements[pos], max_requirement[pos], strict=True))
                    work["suffix_coordinate_reads"] += 2 * dimension
                    can_attain[pos] |= pos in exact_positions
            failed, choices = set(), {}

            def feasible(pos, slots, residual, duration_attained):
                state = (pos, slots, residual, duration_attained)
                if state in failed:
                    work["memo_hits"] += 1
                    return False
                visit()
                needed = target - sum(q * (q - r) for q, r in zip(queue, residual, strict=True))
                work["resource_coordinate_reads"] += dimension
                reason = None
                if needed <= 0 and duration_attained:
                    choices[state] = None
                    return True
                if pos == size or slots == 0:
                    reason = "bound_prunes"
                elif not duration_attained and not can_attain[pos]:
                    reason = "duration_prunes"
                else:
                    resource_limit = sum(q * min(r, slots * m) for q, r, m in
                                         zip(queue, residual, max_requirement[pos], strict=True))
                    work["resource_coordinate_reads"] += 2 * dimension
                    upper = min(slots * max_weight[pos], resource_limit, box_weight[pos])
                    if needed > upper:
                        reason = "bound_prunes"
                    elif needed > 0 and divisors[pos] and (
                            (needed + divisors[pos] - 1) // divisors[pos] * divisors[pos] > upper):
                        reason = "gcd_prunes"
                if reason is not None:
                    work[reason] += 1
                else:
                    count_limit = min(caps[pos], slots, *(r // amount for r, amount in
                                      zip(residual, requirements[pos], strict=True) if amount))
                    work["resource_coordinate_reads"] += 2 * dimension
                    for count in range(count_limit, -1, -1):
                        work["count_branches"] += 1
                        next_residual = tuple(r - count * amount for r, amount in
                                              zip(residual, requirements[pos], strict=True))
                        work["resource_coordinate_reads"] += 2 * dimension
                        child = (pos + 1, slots - count, next_residual,
                                 duration_attained or (count > 0 and pos in exact_positions))
                        if feasible(*child):
                            choices[state] = count
                            return True
                failed.add(state)
                work["failed_states_cached"] += 1
                return False

            root = (0, lanes - sum(region.lower), remaining, attained)
            if feasible(*root):
                local_counts, state = list(region.lower), root
                while choices[state] is not None:
                    pos, slots, residual, duration_attained = state
                    count = choices[state]
                    local_counts[pos] += count
                    state = (pos + 1, slots - count,
                             tuple(r - count * amount for r, amount in
                                   zip(residual, requirements[pos], strict=True)),
                             duration_attained or (count > 0 and pos in exact_positions))
                counts = [0] * len(options)
                for index, count in zip(region.representative_options, local_counts, strict=True):
                    counts[index] = count
                service = tuple(sum(count * required[c] for count, required in
                                    zip(local_counts, requirements, strict=True)) for c in range(dimension))
                actual_duration = max((idle, *(Fraction.from_float(max(float(idle_duration),
                    float(options[index].duration))) for index, count in enumerate(counts) if count)))
                score = Fraction(sum(q * s for q, s in zip(queue, service, strict=True))) / actual_duration
                if (actual_duration != threshold or score <= incumbent or sum(counts) > lanes
                        or any(s > q for s, q in zip(service, queue, strict=True))):
                    raise ArithmeticError("frontier strict-score witness is not a feasible improvement")
                expanded = tuple(index for index, count in enumerate(counts) for _ in range(count))
                return finish("BETTER_ACTION_FOUND", {
                    "option_counts": tuple(counts), "option_indices": expanded + (None,) * (lanes - len(expanded)),
                    "service": service, "duration": float(actual_duration),
                    "score_fraction": (score.numerator, score.denominator)})
    except StateBudgetExhausted:
        return finish("STATE_BUDGET_EXHAUSTED")
    return finish("COMPLETE")


def _frontier_probe(options, queue, lanes, stage, budget, idle_duration, evidence):
    baseline = stage.selection
    proof = _prove_frontier_no_strict_improvement(options, queue, lanes, evidence, budget, idle_duration)
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


def select_branch_reserved_frontier_score_proof_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> FrontierScoreProofRoutedSelection:
    return _select_frontier_score_proof_frame(
        BranchReservedIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)


def select_dual_reuse_frontier_score_proof_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> FrontierScoreProofRoutedSelection:
    return _select_frontier_score_proof_frame(
        DualReuseIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)


def _select_frontier_score_proof_frame(
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
        probe = _frontier_probe(options, queue, lanes, final_lp, probe_budget, idle_duration, evidence)
        selected = probe.selection
        return finish(selected, final_lp, probe, True,
                      "FRONTIER_PROBE_COMPLETE" if selected.canonical_exact else "FRONTIER_PROBE_INCOMPLETE",
                      "frontier_late")
    return finish(selected, final_lp, probe, True,
                  "LP_RESUME_COMPLETE" if selected.canonical_exact else "LP_RESUME_INCOMPLETE", probe_stage)
