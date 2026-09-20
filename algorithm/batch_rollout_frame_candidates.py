"""Compare two support-capped current actions with paid batch clearing steps.

The current paid search retains frozen v4 generation, ordering and pruning.
When the total budget covers the seed reserve plus four attempts, four attempts
are reserved for two deterministic clearing frames for each of two candidates:
the old analytic lookahead challenger and the current support incumbent.
Otherwise the current search uses the full budget and returns its incumbent.

The two clearing frames serve only the known integer residual backlog, with no
new arrivals. Their batch holding area and remaining fluid value replace only
the known-backlog fluid term in the old analytic cost. The complete old Poisson
arrival correction is retained, so interactions between new arrivals and later
batch choices remain a fluid approximation. This is not a stochastic simulation
of future arrivals or a global flow-time guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence


from algorithm.lookahead_frame_candidates import LookaheadFrameSelection


@dataclass(frozen=True)
class BatchRolloutFrameSelection(LookaheadFrameSelection):
    root_budget: int
    root_attempts: int
    rollout_attempts: int
    rollout_used: bool
    rollout_vetoed: bool
    analytic_challenger_score: float
    root_lookahead_cost: float
    selected_rollout_cost: float | None
    incumbent_rollout_cost: float | None
    analytic_challenger_rollout_cost: float | None
    rollout_class_evaluations: int
    rollout_profile_probes: int
    rollout_setup_requirement_reads: int
    rollout_completion_offset_reads: int
    rollout_terminal_value_coordinate_evaluations: int
    rollout_cost_candidate_evaluations: int
    physical_action_completion_offset_reads: int


@dataclass(frozen=True)
class _FrameCandidate:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    duration: float
    score: float


@dataclass(frozen=True)
class _LookaheadCost:
    current_holding_area: float
    expected_next_queue: tuple[float, ...]
    expected_terminal_area: float
    total: float


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


@dataclass(frozen=True)
class _RolloutEvaluation:
    correction: float
    holding_area: float
    initial_fluid_value: float
    tail_fluid_value: float
    attempts: int
    class_evaluations: int
    profile_probes: int
    completion_offset_reads: int
    terminal_value_coordinate_evaluations: int


def current_frame_budget(option_count: int, total_budget: int) -> int:
    """Reserve four attempts only when the current search can still pay all seeds."""
    return total_budget - 4 if total_budget >= option_count + 1 + 4 else total_budget


def _build_rollout_profiles(requirements, durations):
    """Read each requirement once and index pure profiles by descending batch."""
    by_class = [dict() for _ in range(len(requirements[0]))]
    reads = 0
    for index, counts in enumerate(requirements):
        nonzero = []
        for coordinate, count in enumerate(counts):
            reads += 1
            if count:
                nonzero.append((coordinate, count))
        if len(nonzero) == 1:
            coordinate, batch = nonzero[0]
            old = by_class[coordinate].get(batch)
            if old is None or (durations[index], index) < (durations[old], old):
                by_class[coordinate][batch] = index
    return tuple(tuple(sorted(profiles.items(), reverse=True)) for profiles in by_class), reads


def _rollout_old_jobs(options, residual, lanes, idle_duration, service_rates, profiles, durations):
    """Construct exactly two pure-profile clearing frames, charging idle too."""
    remaining = tuple(residual)
    initial_fluid = sum(q**2 / (2.0 * rate) for q, rate in zip(remaining, service_rates))
    holding_area = 0.0
    class_evaluations = profile_probes = offset_reads = 0
    for _ in range(2):
        chosen_class = None
        best_priority = -1.0
        for coordinate, (count, rate) in enumerate(zip(remaining, service_rates)):
            if count:
                class_evaluations += 1
                priority = count / rate
                if priority > best_priority:
                    chosen_class, best_priority = coordinate, priority
        service = [0] * len(remaining)
        duration = idle_duration
        offsets_sum = 0.0
        if chosen_class is not None:
            chosen_profile = None
            for batch, index in profiles[chosen_class]:
                profile_probes += 1
                if batch <= remaining[chosen_class]:
                    chosen_profile = (batch, index)
                    break
            if chosen_profile is None:
                raise ValueError("rollout requires a fitting pure profile for every nonempty chosen class")
            batch, index = chosen_profile
            copies = min(lanes, remaining[chosen_class] // batch)
            service[chosen_class] = copies * batch
            duration = durations[index]
            for _ in range(copies):
                for offsets in options[index].completion_offsets:
                    for offset in offsets:
                        offsets_sum += offset
                        offset_reads += 1
        # This is the sole constructed complete frame for this paid step.
        if any(served > count for served, count in zip(service, remaining)):
            raise ValueError("constructed rollout frame exceeds its visible residual queue")
        holding_area += (sum(remaining) - sum(service)) * duration + offsets_sum
        remaining = tuple(count - served for count, served in zip(remaining, service))
    tail_fluid = sum(q**2 / (2.0 * rate) for q, rate in zip(remaining, service_rates))
    return _RolloutEvaluation(
        correction=holding_area + tail_fluid - initial_fluid,
        holding_area=holding_area, initial_fluid_value=initial_fluid,
        tail_fluid_value=tail_fluid, attempts=2,
        class_evaluations=class_evaluations, profile_probes=profile_probes,
        completion_offset_reads=offset_reads,
        terminal_value_coordinate_evaluations=2 * len(remaining),
    )


def _same_physical_action(left, right, options):
    if left.indices == right.indices:
        return True, 0
    if left.service != right.service or left.duration != right.duration:
        return False, 0
    reads = 0

    def offsets_by_class(candidate):
        nonlocal reads
        per_class = [[] for _ in candidate.service]
        for index in candidate.indices:
            if index is not None:
                for coordinate, offsets in enumerate(options[index].completion_offsets):
                    for offset in offsets:
                        per_class[coordinate].append(offset)
                        reads += 1
        return tuple(tuple(sorted(offsets)) for offsets in per_class)

    return offsets_by_class(left) == offsets_by_class(right), reads


def select_batch_rollout_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
    *,
    arrival_rates: Sequence[float],
    terminal_service_rates: Sequence[float],
) -> BatchRolloutFrameSelection:
    """Use a single paid current search and at most four future frame attempts.

    The current pool keeps the homogeneous-seed and one-percent support floors.
    Analytic costs are evaluated on that pool exactly as in frozen lookahead.
    Only its challenger and the support incumbent receive two clearing steps;
    equivalent physical actions skip these steps and return the incumbent.

    Current-action support bounds never include future-frame searches. Total
    attempts, feasible candidates and local-score counts include future work;
    bound probes remain current-only. Analytic cost_candidate_evaluations stays
    current-only, with the two additional costs reported separately. Offset and
    terminal-coordinate totals include their separately reported extra reads.
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
    while attempts < root_budget and frontier:
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

    upper = max(best_score, -frontier[0][0] if frontier else best_score)
    seed_eligible = [candidate for candidate in candidate_pool if candidate.score >= seed_support]
    support_floor = max(seed_support, (1.0 - max_relative_support_loss) * best_score)
    eligible = [candidate for candidate in seed_eligible if candidate.score >= support_floor]
    completion_offset_reads = 0
    next_queue_coordinate_evaluations = 0
    terminal_value_coordinate_evaluations = 0

    def lookahead_cost(candidate: _FrameCandidate) -> _LookaheadCost:
        nonlocal completion_offset_reads, next_queue_coordinate_evaluations, terminal_value_coordinate_evaluations
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
        return _LookaheadCost(
            current_holding_area=current_area,
            expected_next_queue=tuple(expected_next_queue),
            expected_terminal_area=expected_terminal_area,
            total=current_area + terminal_weight * expected_terminal_area,
        )

    priced = [(candidate, lookahead_cost(candidate)) for candidate in eligible]
    if queue_norm == 0:
        chosen, chosen_cost = priced[0]
    else:
        chosen, chosen_cost = min(priced, key=lambda item: (
            item[1].total, -item[0].score, item[0].duration,
            tuple(len(options) if index is None else index for index in item[0].indices),
        ))
    best_lookahead_cost = next(cost for candidate, cost in priced if candidate is support_incumbent_candidate)
    challenger, challenger_cost = chosen, chosen_cost
    incumbent = support_incumbent_candidate
    root_attempts = attempts
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
    gap = max(0.0, upper - chosen.score)
    return BatchRolloutFrameSelection(
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
        cost_candidate_evaluations=len(priced),
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
    )
