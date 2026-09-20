"""Offline attempt trace for duration-resolved candidate ordering.

The search below deliberately mirrors ``select_duration_tie_frame`` without
calling or modifying it.  A single frontier is advanced through
``small_budget`` and then ``exact_budget`` so the small-budget certificate and
the discovery times refer to one continuous search trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence

from algorithm.duration_tie_frame_candidates import BudgetedFrameSelection


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


@dataclass(frozen=True)
class DurationTieFrameSearchTrace:
    """Two checkpoints and the exact-score discovery decomposition."""

    small: BudgetedFrameSelection
    exact: BudgetedFrameSelection
    first_exact_score_attempt: int
    first_k120_selected_physical_signature_attempt: int
    small_gap: float
    score_regret: float
    bound_slack: float
    gap_identity_error: float
    physical_signature_evaluations: int
    physical_signature_completion_offset_reads: int


def _physical_signature(
    options: tuple[Any, ...],
    indices: tuple[int | None, ...],
    service: tuple[int, ...],
    duration: float,
) -> tuple[
    tuple[tuple[int, ...], float, tuple[tuple[float, ...], ...]],
    int,
]:
    reads = 0
    grouped_offsets = []
    for coordinate in range(len(service)):
        offsets = []
        for index in indices:
            if index is None:
                continue
            for offset in options[index].completion_offsets[coordinate]:
                reads += 1
                offsets.append(float(offset))
        grouped_offsets.append(tuple(sorted(offsets)))
    return (service, duration, tuple(grouped_offsets)), reads


def trace_duration_tie_frame_search(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    small_budget: int = 26,
    exact_budget: int = 120,
    idle_duration: float = 1.0,
) -> DurationTieFrameSearchTrace:
    """Trace one duration-resolved search through small and exact checkpoints.

    Every attempt, feasibility decision, heap key, bound query, and tie break
    matches :func:`select_duration_tie_frame`.  The extra physical-action reads
    are diagnostic only and do not enter the search or its reported work
    counts. ``exact_budget`` must be large enough to empty or certify the
    frontier.
    """
    options = tuple(options)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    seed_attempts = len(options) + 1
    if int(small_budget) != small_budget or small_budget < seed_attempts:
        raise ValueError(
            f"small_budget must be an integer covering {seed_attempts} seed attempts"
        )
    small_budget = int(small_budget)
    if int(exact_budget) != exact_budget or exact_budget < small_budget:
        raise ValueError("exact_budget must be an integer at least small_budget")
    exact_budget = int(exact_budget)
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
        if len(counts) != len(queue) or any(
            int(value) != value or value < 0 for value in counts
        ):
            raise ValueError(
                "option requirements must match the nonnegative integer queue coordinates"
            )
        counts = tuple(int(value) for value in counts)
        if not any(counts):
            raise ValueError("a non-idle option must require at least one job")
        duration = float(option.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("option durations must be positive and finite")
        if len(option.completion_offsets) != len(queue):
            raise ValueError("option completion offsets must match queue coordinates")
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
    physical_signature_evaluations = 0
    physical_signature_completion_offset_reads = 0

    def physical_signature(
        indices: tuple[int | None, ...],
        service: tuple[int, ...],
        duration: float,
    ) -> tuple[tuple[int, ...], float, tuple[tuple[float, ...], ...]]:
        nonlocal physical_signature_evaluations
        nonlocal physical_signature_completion_offset_reads
        signature, reads = _physical_signature(options, indices, service, duration)
        physical_signature_evaluations += 1
        physical_signature_completion_offset_reads += reads
        return signature

    observations = [(
        attempts,
        best_score,
        physical_signature(best_indices, best_service, best_duration),
    )]

    def consider(
        indices: tuple[int | None, ...],
        service: tuple[int, ...],
        numerator: int,
        duration: float,
    ) -> None:
        nonlocal best_indices, best_service, best_duration, best_score, feasible_count
        feasible_count += 1
        padded = indices + (None,) * (lanes - len(indices))
        score = numerator / duration
        observations.append((
            attempts,
            score,
            physical_signature(padded, service, duration),
        ))
        key = (duration, tuple(
            len(options) if index is None else index for index in padded
        ))
        best_key = (best_duration, tuple(
            len(options) if index is None else index for index in best_indices
        ))
        if score > best_score or (score == best_score and key < best_key):
            best_indices, best_service = padded, service
            best_duration, best_score = duration, score

    seed_copies = []
    for index, counts in enumerate(requirements):
        attempts += 1
        copies = min(
            lanes,
            min(q // count for q, count in zip(queue, counts) if count),
        )
        seed_copies.append(copies)
        if copies:
            consider(
                (index,) * copies,
                tuple(copies * count for count in counts),
                copies * weights[index],
                durations[index],
            )

    # Reuse paid seed feasibility and retain the weighted-count order among
    # available options. Equal weights prefer shorter effective durations;
    # unavailable large batches still move behind useful small-queue actions.
    # The order is fixed for this queue, independently of the search budget.
    ordered: tuple[int | None, ...] = tuple(
        sorted(range(len(options)), key=lambda index: (
            seed_copies[index] == 0,
            -weights[index],
            durations[index],
            index,
        ))
    ) + (None,)
    local_weights = tuple(
        0 if index is None else weights[index] for index in ordered
    )
    local_durations = tuple(
        idle_duration if index is None else durations[index] for index in ordered
    )
    local_requirements = tuple(
        (0,) * len(queue) if index is None else requirements[index]
        for index in ordered
    )
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
                max_weights[position] = max(
                    max_weights[position], local_weights[position]
                )
                max_counts[position] = tuple(
                    max(old, count)
                    for old, count in zip(
                        max_counts[position], local_requirements[position]
                    )
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
            weighted_cap = (
                max_weights[prefix.next_index]
                + (remaining - 1) * max_weights[0]
            )
            queue_cap = sum(
                q * min(left, first + (remaining - 1) * other)
                for q, left, first, other in zip(
                    queue,
                    residual,
                    max_counts[prefix.next_index],
                    max_counts[0],
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

    def snapshot() -> BudgetedFrameSelection:
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

    push(_Prefix((), (0,) * len(queue), 0, idle_duration, 0))
    small: BudgetedFrameSelection | None = (
        snapshot() if attempts == small_budget else None
    )
    while attempts < exact_budget and frontier:
        negative_upper, _, prefix = heapq.heappop(frontier)
        if -negative_upper <= best_score:
            # This is the largest remaining upper bound, so the search is exact.
            frontier.clear()
            break
        attempts += 1
        position = prefix.next_index
        index = ordered[position]
        counts = (0,) * len(queue) if index is None else requirements[index]
        service = tuple(
            old + count for old, count in zip(prefix.service, counts)
        )
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
            push(_Prefix(
                prefix.indices,
                prefix.service,
                prefix.numerator,
                prefix.duration,
                position + 1,
            ))
        if attempts == small_budget:
            small = snapshot()

    final = snapshot()
    if small is None:
        # The frontier became exact before the requested small checkpoint.
        small = final
    if not final.exact:
        raise ValueError("exact_budget did not certify the final optimal support")

    exact_signature = physical_signature(
        final.option_indices,
        final.service,
        final.duration,
    )
    first_exact_score_attempt = min(
        attempt for attempt, score, _ in observations if score == final.score
    )
    first_k120_selected_physical_signature_attempt = min(
        attempt
        for attempt, _, signature in observations
        if signature == exact_signature
    )
    score_regret = final.score - small.score
    bound_slack = small.upper_bound - final.score
    small_gap = small.upper_bound - small.score
    gap_identity_error = small_gap - (score_regret + bound_slack)
    return DurationTieFrameSearchTrace(
        small=small,
        exact=final,
        first_exact_score_attempt=first_exact_score_attempt,
        first_k120_selected_physical_signature_attempt=(
            first_k120_selected_physical_signature_attempt
        ),
        small_gap=small_gap,
        score_regret=score_regret,
        bound_slack=bound_slack,
        gap_identity_error=gap_identity_error,
        physical_signature_evaluations=physical_signature_evaluations,
        physical_signature_completion_offset_reads=(
            physical_signature_completion_offset_reads
        ),
    )


__all__ = ["DurationTieFrameSearchTrace", "trace_duration_tie_frame_search"]
