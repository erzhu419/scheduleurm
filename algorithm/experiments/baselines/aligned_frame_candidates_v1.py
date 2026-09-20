"""Historical v1 selector with an added observational threshold counter.

Policy, ordering and pruning match the original v1 implementation. The new
bound_threshold_evaluations field counts effective upper-bound thresholds.

Budgeted Cartesian candidate construction for synchronized completion frames.

Each local option exposes ``requirements`` (job counts) and ``duration`` (seconds).
A frame assigns one option or idle to each identical lane, completes the sum of
the assigned requirements, and lasts the largest assigned duration, subject to
``idle_duration``. The support score is Q dot completed_jobs / frame_seconds.

The budget counts candidate construction/feasibility attempts, including idle,
one maximal feasible homogeneous seed per option, and rejected search attempts.
Local catalogue scoring and upper-bound option probes are reported separately.
The search never materializes the full Cartesian action family.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class BudgetedFrameSelection:
    option_indices: tuple[int | None, ...]
    service: tuple[int, ...]
    duration: float
    score: float
    upper_bound: float
    gap: float
    epsilon: float
    attempts: int
    seed_attempts: int
    feasible_candidate_count: int
    local_score_evaluations: int
    bound_option_probes: int
    bound_threshold_evaluations: int
    frontier_peak: int
    exact: bool


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


def select_budgeted_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
) -> BudgetedFrameSelection:
    """Select a feasible frame and bound the support of every unexplored frame.

    ``budget`` must cover ``len(options) + 1`` seed attempts. An option seed uses
    as many identical lanes as the visible queue permits; zero copies is a paid
    failed attempt. Subsequent attempts extend one feasible prefix by one local
    option, and the remaining lanes idle. Heap nodes retain the untried sibling
    choices, so stopping mid-search leaves a bound on the entire unseen family.

    An upper-bound probe inspects one local option at one possible frame-duration
    threshold. Such probes are not complete candidate evaluations and are not
    included in ``budget``. Their count is exposed as ``bound_option_probes``.
    """
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

    # All-idle is the first paid, feasible candidate. None denotes a vacant lane.
    best_indices: tuple[int | None, ...] = (None,) * lanes
    best_service = (0,) * len(queue)
    best_duration = idle_duration
    best_score = 0.0
    attempts = 1
    feasible_count = 1

    def consider(indices: tuple[int | None, ...], service: tuple[int, ...],
                 numerator: int, duration: float) -> None:
        nonlocal best_indices, best_service, best_duration, best_score, feasible_count
        feasible_count += 1
        padded = indices + (None,) * (lanes - len(indices))
        score = numerator / duration
        key = (duration, tuple(len(options) if index is None else index for index in padded))
        best_key = (best_duration, tuple(len(options) if index is None else index for index in best_indices))
        if score > best_score or (score == best_score and key < best_key):
            best_indices, best_service = padded, service
            best_duration, best_score = duration, score

    for index, counts in enumerate(requirements):
        attempts += 1
        copies = min(lanes, min(q // count for q, count in zip(queue, counts) if count))
        if copies:
            consider(
                (index,) * copies,
                tuple(copies * count for count in counts),
                copies * weights[index],
                durations[index],
            )

    # Local ordering is independent of search budget. Idle remains last on ties.
    ordered: tuple[int | None, ...] = tuple(
        sorted(range(len(options)), key=lambda index: (-weights[index], index))
    ) + (None,)
    local_weights = tuple(0 if index is None else weights[index] for index in ordered)
    local_durations = tuple(idle_duration if index is None else durations[index] for index in ordered)
    thresholds = tuple(sorted({idle_duration, *durations}))
    bound_probes = 0
    threshold_evaluations = 0

    def upper_bound(prefix: _Prefix) -> float:
        nonlocal bound_probes, threshold_evaluations
        remaining = lanes - len(prefix.indices)
        best_upper = 0.0
        # Every completed frame has a duration in this finite threshold set.
        # Ignore shared queue constraints here; relaxing them only raises support.
        for duration in thresholds:
            if duration < prefix.duration:
                continue
            threshold_evaluations += 1
            first_max = 0
            for position in range(prefix.next_index, len(ordered)):
                bound_probes += 1
                if local_durations[position] <= duration:
                    first_max = max(first_max, local_weights[position])
            other_max = 0
            if remaining > 1:
                for position in range(len(ordered)):
                    bound_probes += 1
                    if local_durations[position] <= duration:
                        other_max = max(other_max, local_weights[position])
            numerator = prefix.numerator + first_max + (remaining - 1) * other_max
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
    gap = max(0.0, upper - best_score)
    queue_norm = sum(queue)
    return BudgetedFrameSelection(
        option_indices=best_indices,
        service=best_service,
        duration=best_duration,
        score=best_score,
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
    )
