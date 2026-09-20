"""Choose registered-arrival holding cost within the paid v4 candidate pool.

Candidate generation and max-support pruning retain the frozen v4 semantics.
After all homogeneous seeds are evaluated, their best support is a fixed
eligibility floor. Only the final choice uses complete-job holding-rate cost,
including every dispatch-to-completion offset. The supplied total arrival rate
is a registered policy constant, including during finite-population draining.
An optional relative loss cap raises eligibility against the paid support
incumbent. With a cap below one, increasing the budget can raise this floor;
holding-rate cost is then not guaranteed to decrease across budgets.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence


from algorithm.budgeted_frame_candidates import BudgetedFrameSelection


@dataclass(frozen=True)
class WaitingCostFrameSelection(BudgetedFrameSelection):
    seed_support: float
    support_floor: float
    relative_support_loss: float
    support_cap_excluded_candidate_count: int
    holding_rate_cost: float
    eligible_candidate_count: int
    support_incumbent_score: float
    support_incumbent_holding_rate_cost: float
    cost_candidate_evaluations: int
    completion_offset_reads: int
    support_search_exact: bool


@dataclass(frozen=True)
class _FrameCandidate:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    duration: float
    score: float


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


def select_waiting_cost_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
    *,
    arrival_rate: float,
    max_relative_support_loss: float = 1.0,
) -> WaitingCostFrameSelection:
    """Choose holding cost above the seed floor after the unchanged v4 search.

    ``budget`` must cover ``len(options) + 1`` seed attempts. An option seed uses
    as many identical lanes as the visible queue permits; zero copies is a paid
    failed attempt. Subsequent attempts extend one feasible prefix by one local
    option, and the remaining lanes idle. Heap nodes retain the untried sibling
    choices, so stopping mid-search leaves a bound on the entire unseen family.

    Upper bounds cap each service coordinate by the unassigned queue. Threshold
    tables contain local weighted and coordinatewise maxima; they do not test
    complete frame feasibility. Building these tables and querying thresholds
    are counted separately from candidate construction.

    ``max_relative_support_loss`` caps loss against the paid support incumbent,
    in addition to the homogeneous-seed floor. Its default of one reproduces
    the original seed-only choice; it is not a loss bound against a full oracle.
    """
    arrival_rate = float(arrival_rate)
    if not math.isfinite(arrival_rate) or arrival_rate < 0.0:
        raise ValueError("arrival_rate must be finite and nonnegative")
    max_relative_support_loss = float(max_relative_support_loss)
    if not math.isfinite(max_relative_support_loss) or not 0.0 <= max_relative_support_loss <= 1.0:
        raise ValueError("max_relative_support_loss must be finite and in [0, 1]")
    options = tuple(options)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    seed_attempts = len(options) + 1
    if int(budget) != budget or budget < seed_attempts:
        raise ValueError(f"budget must be an integer covering {seed_attempts} seed attempts")
    budget = int(budget)
    idle_duration = float(idle_duration)
    if not math.isfinite(idle_duration) or idle_duration <= 0.0:
        raise ValueError("idle_duration must be positive and finite")
    if any(int(value) != value or value < 0 for value in queues):
        raise ValueError("queues must contain nonnegative integer job counts")
    queue = tuple(int(value) for value in queues)
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
    while attempts < budget and frontier:
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

    def holding_cost(candidate: _FrameCandidate) -> float:
        nonlocal completion_offset_reads
        completion_offsets = 0.0
        for index in candidate.indices:
            if index is not None:
                for offsets in options[index].completion_offsets:
                    for offset in offsets:
                        completion_offsets += offset
                        completion_offset_reads += 1
        return (queue_norm - sum(candidate.service)
                + completion_offsets / candidate.duration
                + arrival_rate * candidate.duration / 2.0)

    priced = [(candidate, holding_cost(candidate)) for candidate in eligible]
    if queue_norm == 0:
        chosen, chosen_cost = priced[0]
    else:
        chosen, chosen_cost = min(priced, key=lambda item: (
            item[1], -item[0].score, item[0].duration,
            tuple(len(options) if index is None else index for index in item[0].indices),
        ))
    best_holding_cost = next(cost for candidate, cost in priced if candidate is support_incumbent_candidate)
    gap = max(0.0, upper - chosen.score)
    return WaitingCostFrameSelection(
        option_indices=chosen.indices,
        service=chosen.service,
        duration=chosen.duration,
        score=chosen.score,
        upper_bound=upper,
        gap=gap,
        epsilon=gap / queue_norm if queue_norm else 0.0,
        attempts=attempts,
        seed_attempts=seed_attempts,
        feasible_candidate_count=feasible_count,
        local_score_evaluations=len(options),
        bound_option_probes=bound_probes,
        bound_threshold_evaluations=threshold_evaluations,
        frontier_peak=frontier_peak,
        exact=gap == 0.0,
        seed_support=seed_support,
        support_floor=support_floor,
        relative_support_loss=(best_score - chosen.score) / best_score if best_score > 0.0 else 0.0,
        support_cap_excluded_candidate_count=len(seed_eligible) - len(eligible),
        holding_rate_cost=chosen_cost,
        eligible_candidate_count=len(eligible),
        support_incumbent_score=best_score,
        support_incumbent_holding_rate_cost=best_holding_cost,
        cost_candidate_evaluations=len(priced),
        completion_offset_reads=completion_offset_reads,
        support_search_exact=upper == best_score,
    )
