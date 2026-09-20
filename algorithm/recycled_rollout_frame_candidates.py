"""Return an unused four-attempt rollout reserve to the same current search.

The first stage is the frozen batch-rollout current search. Distinct analytic
challenger and support-incumbent physical actions retain the original fixed
two-plus-two clearing comparison. Otherwise the preserved frontier advances
with the full current budget and returns the latest support incumbent. A new,
unrolled challenger found during this continuation is diagnostic only.

Seeds, local weights and bound tables are built once. Candidate analytic costs
are cached by object identity across both stages; total cost evaluations may
exceed the final eligible count when the support floor changes. The imported
batch clearing helpers and their terminal approximation remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence


from algorithm.batch_rollout_frame_candidates import (
    BatchRolloutFrameSelection,
    current_frame_budget,
    _FrameCandidate,
    _Prefix,
    _LookaheadCost,
    _build_rollout_profiles,
    _rollout_old_jobs,
    _same_physical_action,
)


@dataclass(frozen=True)
class RecycledRolloutFrameSelection(BatchRolloutFrameSelection):
    initial_root_budget: int
    initial_root_attempts: int
    reclaimed_reserve: bool
    recycled_candidate_attempts: int
    current_search_resumed: bool
    initial_cost_candidate_evaluations: int
    recycled_cost_candidate_evaluations: int


def select_recycled_rollout_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
    *,
    arrival_rates: Sequence[float],
    terminal_service_rates: Sequence[float],
) -> RecycledRolloutFrameSelection:
    """Continue the same current frontier when the rollout reserve is unused.

    The initial stage allocates K0 with the frozen four-attempt reserve rule.
    Distinct physical candidates use exactly the old clearing comparison. If
    they coincide, the reserve becomes current-search budget, including when
    the frontier already finished. current_search_resumed is true only when
    this continuation actually constructs additional candidates.

    The returned support certificate refers to the final current pool. Cost
    evaluation counts are cumulative unique-object scores across both phases;
    eligible_candidate_count refers only to final eligibility. New diagnostic
    challengers after budget recovery never replace the support incumbent.
    """
    arrival_rates = tuple(float(rate) for rate in arrival_rates)
    terminal_service_rates = tuple(float(rate) for rate in terminal_service_rates)
    if any(not math.isfinite(rate) or rate < 0.0 for rate in arrival_rates):
        raise ValueError("arrival_rates must be finite and nonnegative")
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in terminal_service_rates):
        raise ValueError("terminal_service_rates must be finite and positive")
    terminal_weight = 1.0
    max_relative_support_loss = 0.01
    options = tuple(options)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    seed_attempts = len(options) + 1
    if int(budget) != budget or budget < seed_attempts:
        raise ValueError(f"budget must be an integer covering {seed_attempts} seed attempts")
    budget = int(budget)
    root_budget = current_frame_budget(len(options), budget)
    initial_root_budget = root_budget
    idle_duration = float(idle_duration)
    if not math.isfinite(idle_duration) or idle_duration <= 0.0:
        raise ValueError("idle_duration must be positive and finite")
    if any(int(value) != value or value < 0 for value in queues):
        raise ValueError("queues must contain nonnegative integer job counts")
    queue = tuple(int(value) for value in queues)
    if len(arrival_rates) != len(queue) or len(terminal_service_rates) != len(queue):
        raise ValueError("arrival and terminal service rates must match queue coordinates")
    total_arrival_rate = sum(arrival_rates)
    requirements: list[tuple[int, ...]] = []
    durations: list[float] = []
    weights: list[int] = []
    for option in options:
        counts = tuple(option.requirements)
        if len(counts) != len(queue) or any(int(value) != value or value < 0 for value in counts):
            raise ValueError("option requirements must match the nonnegative integer queue coordinates")
        counts = tuple(int(value) for value in counts)
        if not any(counts):
            raise ValueError("a non-idle option must require at least one job")
        duration = float(option.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("option durations must be positive and finite")
        requirements.append(counts)
        durations.append(max(idle_duration, duration))
        weights.append(sum(q * count for q, count in zip(queue, counts)))

    queue_norm = sum(queue)

    # All-idle is the first paid, feasible candidate. None denotes a vacant lane.
    best_indices: tuple[int | None, ...] = (None,) * lanes
    best_service = (0,) * len(queue)
    best_duration = idle_duration
    best_score = 0.0
    attempts = 1
    feasible_count = 1
    candidate_pool = [_FrameCandidate(best_indices, best_service, best_duration, best_score)]
    support_incumbent_candidate = candidate_pool[0]

    def consider(indices: tuple[int | None, ...], service: tuple[int, ...],
                 numerator: int, duration: float) -> None:
        nonlocal best_indices, best_service, best_duration, best_score, feasible_count, support_incumbent_candidate
        feasible_count += 1
        padded = indices + (None,) * (lanes - len(indices))
        score = numerator / duration
        candidate = _FrameCandidate(padded, service, duration, score)
        candidate_pool.append(candidate)
        key = (duration, tuple(len(options) if index is None else index for index in padded))
        best_key = (best_duration, tuple(len(options) if index is None else index for index in best_indices))
        if score > best_score or (score == best_score and key < best_key):
            best_indices, best_service = padded, service
            best_duration, best_score = duration, score
            support_incumbent_candidate = candidate

    seed_copies = []
    for index, counts in enumerate(requirements):
        attempts += 1
        copies = min(lanes, min(q // count for q, count in zip(queue, counts) if count))
        seed_copies.append(copies)
        if copies:
            consider(
                (index,) * copies,
                tuple(copies * count for count in counts),
                copies * weights[index],
                durations[index],
            )

    seed_support = best_score

    # Reuse paid seed feasibility and retain the original weighted-count order
    # among available options. Unavailable large batches move behind useful
    # small-queue actions without changing the order when all options fit.
    # The order is fixed for this queue, independently of the search budget.
    ordered: tuple[int | None, ...] = tuple(
        sorted(range(len(options)), key=lambda index: (
            seed_copies[index] == 0, -weights[index], index,
        ))
    ) + (None,)
    local_weights = tuple(0 if index is None else weights[index] for index in ordered)
    local_durations = tuple(idle_duration if index is None else durations[index] for index in ordered)
    local_requirements = tuple((0,) * len(queue) if index is None else requirements[index] for index in ordered)
    thresholds = tuple(sorted({idle_duration, *durations}))
    bound_probes = 0
    threshold_evaluations = 0

    # For each possible duration, suffix maxima cover the untried siblings.
    # Reuse them for all prefixes instead of rescanning the catalogue per bound.
    tables = []
    for duration in thresholds:
        max_weights = [0] * (len(ordered) + 1)
        max_counts = [(0,) * len(queue)] * (len(ordered) + 1)
        for position in range(len(ordered) - 1, -1, -1):
            bound_probes += 1
            max_weights[position] = max_weights[position + 1]
            max_counts[position] = max_counts[position + 1]
            if local_durations[position] <= duration:
                max_weights[position] = max(max_weights[position], local_weights[position])
                max_counts[position] = tuple(
                    max(old, count)
                    for old, count in zip(max_counts[position], local_requirements[position])
                )
        tables.append((duration, max_weights, max_counts))

    def upper_bound(prefix: _Prefix) -> float:
        nonlocal threshold_evaluations
        remaining = lanes - len(prefix.indices)
        residual = tuple(q - used for q, used in zip(queue, prefix.service))
        best_upper = 0.0
        # Every completed frame has a duration in this finite threshold set.
        # Coordinate maxima may come from different options. Thus even with the
        # residual-queue cap this is a relaxation, not an uncharged frame search.
        for duration, max_weights, max_counts in tables:
            if duration < prefix.duration:
                continue
            threshold_evaluations += 1
            weighted_cap = max_weights[prefix.next_index] + (remaining - 1) * max_weights[0]
            queue_cap = sum(
                q * min(left, first + (remaining - 1) * other)
                for q, left, first, other in zip(
                    queue, residual, max_counts[prefix.next_index], max_counts[0],
                )
            )
            numerator = prefix.numerator + min(weighted_cap, queue_cap)
            best_upper = max(best_upper, numerator / duration)
        return best_upper

    frontier: list[tuple[float, int, _Prefix]] = []
    serial = 0
    frontier_peak = 0

    def push(prefix: _Prefix) -> None:
        nonlocal serial, frontier_peak
        upper = upper_bound(prefix)
        if upper > best_score:
            heapq.heappush(frontier, (-upper, serial, prefix))
            serial += 1
            frontier_peak = max(frontier_peak, len(frontier))

    push(_Prefix((), (0,) * len(queue), 0, idle_duration, 0))
    def advance(limit: int) -> None:
        nonlocal attempts
        while attempts < limit and frontier:
            negative_upper, _, prefix = heapq.heappop(frontier)
            if -negative_upper <= best_score:
                # This is the largest remaining upper bound, so the search is exact.
                frontier.clear()
                break
            attempts += 1
            position = prefix.next_index
            index = ordered[position]
            counts = (0,) * len(queue) if index is None else requirements[index]
            service = tuple(old + count for old, count in zip(prefix.service, counts))
            feasible = all(count <= q for count, q in zip(service, queue))
            if feasible:
                indices = prefix.indices + (index,)
                numerator = prefix.numerator + local_weights[position]
                duration = max(prefix.duration, local_durations[position])
                consider(indices, service, numerator, duration)
                if len(indices) < lanes:
                    push(_Prefix(indices, service, numerator, duration, 0))
            # Keep untried siblings even when this single attempted choice was invalid.
            if position + 1 < len(ordered):
                push(_Prefix(prefix.indices, prefix.service, prefix.numerator,
                             prefix.duration, position + 1))

    advance(initial_root_budget)
    initial_root_attempts = attempts
    completion_offset_reads = 0
    next_queue_coordinate_evaluations = 0
    terminal_value_coordinate_evaluations = 0
    cost_cache: dict[int, _LookaheadCost] = {}

    def lookahead_cost(candidate: _FrameCandidate) -> _LookaheadCost:
        nonlocal completion_offset_reads, next_queue_coordinate_evaluations, terminal_value_coordinate_evaluations
        cached = cost_cache.get(id(candidate))
        if cached is not None:
            return cached
        completion_offsets = 0.0
        for index in candidate.indices:
            if index is not None:
                for offsets in options[index].completion_offsets:
                    for offset in offsets:
                        completion_offsets += offset
                        completion_offset_reads += 1
        duration = candidate.duration
        current_area = ((queue_norm - sum(candidate.service)) * duration
                        + completion_offsets
                        + total_arrival_rate * duration**2 / 2.0)
        expected_next_queue = []
        expected_terminal_area = 0.0
        for q, served, arrival_rate, service_rate in zip(
            queue, candidate.service, arrival_rates, terminal_service_rates,
        ):
            arrival_mean = arrival_rate * duration
            next_mean = q - served + arrival_mean
            next_queue_coordinate_evaluations += 1
            expected_next_queue.append(next_mean)
            expected_terminal_area += (next_mean**2 + arrival_mean) / (2.0 * service_rate)
            terminal_value_coordinate_evaluations += 1
        result = _LookaheadCost(
            current_holding_area=current_area,
            expected_next_queue=tuple(expected_next_queue),
            expected_terminal_area=expected_terminal_area,
            total=current_area + terminal_weight * expected_terminal_area,
        )
        cost_cache[id(candidate)] = result
        return result

    def price_pool():
        seed_eligible = [candidate for candidate in candidate_pool if candidate.score >= seed_support]
        support_floor = max(seed_support, (1.0 - max_relative_support_loss) * best_score)
        eligible = [candidate for candidate in seed_eligible if candidate.score >= support_floor]
        priced = [(candidate, lookahead_cost(candidate)) for candidate in eligible]
        if queue_norm == 0:
            challenger, challenger_cost = priced[0]
        else:
            challenger, challenger_cost = min(priced, key=lambda item: (
                item[1].total, -item[0].score, item[0].duration,
                tuple(len(options) if index is None else index for index in item[0].indices),
            ))
        incumbent_cost = next(cost for candidate, cost in priced if candidate is support_incumbent_candidate)
        return seed_eligible, support_floor, eligible, challenger, challenger_cost, incumbent_cost

    seed_eligible, support_floor, eligible, challenger, challenger_cost, best_lookahead_cost = price_pool()
    initial_cost_evaluations = len(cost_cache)
    incumbent = support_incumbent_candidate
    reclaimed_reserve = current_search_resumed = False
    recycled_candidate_attempts = 0
    rollout_attempts = rollout_class_evaluations = rollout_profile_probes = 0
    rollout_setup_reads = rollout_offset_reads = rollout_terminal_evaluations = 0
    physical_offset_reads = 0
    rollout_used = rollout_vetoed = False
    selected_rollout_cost = incumbent_rollout_cost = challenger_rollout_cost = None
    chosen, chosen_cost = incumbent, best_lookahead_cost
    if root_budget < budget:
        same_action, physical_offset_reads = _same_physical_action(challenger, incumbent, options)
        if not same_action:
            profiles, rollout_setup_reads = _build_rollout_profiles(requirements, durations)
            evaluated = []
            for candidate, analytic_cost in ((challenger, challenger_cost), (incumbent, best_lookahead_cost)):
                residual = tuple(q - served for q, served in zip(queue, candidate.service))
                result = _rollout_old_jobs(
                    options, residual, lanes, idle_duration,
                    terminal_service_rates, profiles, durations,
                )
                rollout_attempts += result.attempts
                rollout_class_evaluations += result.class_evaluations
                rollout_profile_probes += result.profile_probes
                rollout_offset_reads += result.completion_offset_reads
                rollout_terminal_evaluations += result.terminal_value_coordinate_evaluations
                evaluated.append((candidate, analytic_cost, analytic_cost.total + result.correction))
            challenger_rollout_cost = evaluated[0][2]
            incumbent_rollout_cost = evaluated[1][2]
            chosen, chosen_cost, selected_rollout_cost = min(evaluated, key=lambda item: (
                item[2], -item[0].score, item[0].duration,
                tuple(len(options) if index is None else index for index in item[0].indices),
            ))
            rollout_used = True
            rollout_vetoed = chosen is incumbent
        else:
            reclaimed_reserve = True
            root_budget = budget
            advance(root_budget)
            recycled_candidate_attempts = attempts - initial_root_attempts
            current_search_resumed = recycled_candidate_attempts > 0
            seed_eligible, support_floor, eligible, challenger, challenger_cost, best_lookahead_cost = price_pool()
            incumbent = support_incumbent_candidate
            chosen, chosen_cost = incumbent, best_lookahead_cost
    root_attempts = attempts
    upper = max(best_score, -frontier[0][0] if frontier else best_score)
    gap = max(0.0, upper - chosen.score)
    return RecycledRolloutFrameSelection(
        option_indices=chosen.indices,
        service=chosen.service,
        duration=chosen.duration,
        score=chosen.score,
        upper_bound=upper,
        gap=gap,
        epsilon=gap / queue_norm if queue_norm else 0.0,
        attempts=root_attempts + rollout_attempts,
        seed_attempts=seed_attempts,
        feasible_candidate_count=feasible_count + rollout_attempts,
        local_score_evaluations=len(options) + rollout_class_evaluations,
        bound_option_probes=bound_probes,
        bound_threshold_evaluations=threshold_evaluations,
        frontier_peak=frontier_peak,
        exact=gap == 0.0,
        seed_support=seed_support,
        support_floor=support_floor,
        relative_support_loss=(best_score - chosen.score) / best_score if best_score > 0.0 else 0.0,
        support_cap_excluded_candidate_count=len(seed_eligible) - len(eligible),
        eligible_candidate_count=len(eligible),
        support_incumbent_score=best_score,
        support_search_exact=upper == best_score,
        current_holding_area=chosen_cost.current_holding_area,
        expected_next_queue=chosen_cost.expected_next_queue,
        expected_terminal_area=chosen_cost.expected_terminal_area,
        lookahead_cost=chosen_cost.total,
        support_incumbent_lookahead_cost=best_lookahead_cost.total,
        cost_candidate_evaluations=len(cost_cache),
        completion_offset_reads=completion_offset_reads + physical_offset_reads + rollout_offset_reads,
        next_queue_coordinate_evaluations=next_queue_coordinate_evaluations,
        terminal_value_coordinate_evaluations=terminal_value_coordinate_evaluations + rollout_terminal_evaluations,
        root_budget=root_budget,
        root_attempts=root_attempts,
        rollout_attempts=rollout_attempts,
        rollout_used=rollout_used,
        rollout_vetoed=rollout_vetoed,
        analytic_challenger_score=challenger.score,
        root_lookahead_cost=challenger_cost.total,
        selected_rollout_cost=selected_rollout_cost,
        incumbent_rollout_cost=incumbent_rollout_cost,
        analytic_challenger_rollout_cost=challenger_rollout_cost,
        rollout_class_evaluations=rollout_class_evaluations,
        rollout_profile_probes=rollout_profile_probes,
        rollout_setup_requirement_reads=rollout_setup_reads,
        rollout_completion_offset_reads=rollout_offset_reads,
        rollout_terminal_value_coordinate_evaluations=rollout_terminal_evaluations,
        rollout_cost_candidate_evaluations=2 if rollout_used else 0,
        physical_action_completion_offset_reads=physical_offset_reads,
        initial_root_budget=initial_root_budget,
        initial_root_attempts=initial_root_attempts,
        reclaimed_reserve=reclaimed_reserve,
        recycled_candidate_attempts=recycled_candidate_attempts,
        current_search_resumed=current_search_resumed,
        initial_cost_candidate_evaluations=initial_cost_evaluations,
        recycled_cost_candidate_evaluations=len(cost_cache) - initial_cost_evaluations,
    )
