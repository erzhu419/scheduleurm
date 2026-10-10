"""Complete canonical ties after a separately certified MaxWeight score.

The first stage is the frozen integer-capacity LP search. Only its unresolved
score-optimal face enters the second stage. Original option counts are searched
in canonical order with exact integer resource arithmetic and a separate state
budget. No reference-oracle implementation or reference answer is used here.

Search work counters on ``selection`` describe stage one; its live-region
counts become zero when the separate completion proof discharges them. The
retained ``baseline_selection`` is the unmodified first-stage snapshot.
``completion`` accounts for the integer stage, including incomplete work.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Any, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.integer_capacity_shared_threshold_frame_candidates import (
    CapacityPropagationWork,
    select_integer_capacity_shared_threshold_frame,
)
from algorithm.threshold_canonical_requirement_dominance_frame_candidates import ThresholdCatalogueWork


@dataclass(frozen=True)
class ScoreFaceCompletionWork:
    status: str
    attempted: bool
    complete: bool
    state_budget: int
    states_visited: int = 0
    memo_hits: int = 0
    count_branches: int = 0
    bound_prunes: int = 0
    gcd_prunes: int = 0
    duration_prunes: int = 0
    failed_states_cached: int = 0
    thresholds_checked: int = 0
    noninteger_thresholds: int = 0
    suffix_coordinate_reads: int = 0
    resource_coordinate_reads: int = 0
    action_changed: bool = False
    score_fraction: tuple[int, int] | None = None


@dataclass(frozen=True)
class ScoreFaceCanonicalCompletionSelection:
    selection: CanonicalCountSpaceBnbSelection
    baseline_selection: CanonicalCountSpaceBnbSelection
    preprocessing: ThresholdCatalogueWork
    capacity_propagation: CapacityPropagationWork
    completion: ScoreFaceCompletionWork


def select_score_face_canonical_completion_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    lp_node_budget: int,
    canonical_state_budget: int,
    idle_duration: float = 1.0,
) -> ScoreFaceCanonicalCompletionSelection:
    """Select with separate LP-call and conditional integer-state budgets.

    For a fixed duration, descending original-index counts enumerate the
    sorted option-index key in increasing order. Increasing duration thresholds
    therefore make the first feasible score-face action canonical. Failed
    suffix states are memoized; every newly examined state, including a bound
    rejection, consumes the shared integer budget across all thresholds.
    """
    if int(canonical_state_budget) != canonical_state_budget or canonical_state_budget < 0:
        raise ValueError("canonical_state_budget must be a nonnegative integer")
    budget = int(canonical_state_budget)
    options, queue = tuple(options), tuple(queues)
    stage_one = select_integer_capacity_shared_threshold_frame(
        options, queue, lanes, lp_node_budget, idle_duration)
    baseline = stage_one.selection

    def result(selection, work):
        return ScoreFaceCanonicalCompletionSelection(
            selection, baseline, stage_one.preprocessing, stage_one.capacity_propagation, work)

    if baseline.canonical_exact:
        return result(baseline, ScoreFaceCompletionWork(
            "SKIPPED_CANONICAL_COMPLETE", False, True, budget))
    if not baseline.score_exact:
        return result(baseline, ScoreFaceCompletionWork(
            "SKIPPED_SCORE_UNPROVEN", False, False, budget))

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
        "noninteger_thresholds", "suffix_coordinate_reads", "resource_coordinate_reads")}

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
        return result(selected, ScoreFaceCompletionWork(
            "COMPLETE", True, True, budget, **work,
            action_changed=(counts != baseline.option_counts or selected.duration != baseline.duration),
            score_fraction=(proven_score.numerator, proven_score.denominator)))

    # Feasibility of a non-idle option implies positive service against Q, so a
    # zero optimum has the unique all-idle feasible action at score zero.
    if proven_score == 0:
        return finish((0,) * size, idle)

    class StateBudgetExhausted(Exception):
        pass

    for threshold in sorted({d for d in durations if d <= incumbent_duration}):
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
                elif not divisors[pos] or needed % divisors[pos]:
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
            return result(baseline, ScoreFaceCompletionWork(
                "STATE_BUDGET_EXHAUSTED", True, False, budget, **work,
                score_fraction=(proven_score.numerator, proven_score.denominator)))
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
