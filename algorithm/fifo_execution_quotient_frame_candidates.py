"""Budgeted v4 search with exact FIFO-execution candidate reuse.

The logical prefix heap, option order, support objective, upper bound, and raw
lane-order tie break are the same as :mod:`algorithm.budgeted_frame_candidates`.
After the paid seeds, the first action with a given aggregate service, duration,
and per-class FIFO-ordered completion offsets pays for construction and
feasibility. Later actions with exactly that execution signature reuse the
result without consuming candidate budget. Lane permutations with different
FIFO pairing remain distinct paid candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class FifoExecutionQuotientFrameSelection:
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
    represented_ordered_configurations: int
    cache_entries: int
    cache_hits: int
    cache_misses_after_seeds: int
    seed_signature_collisions: int
    logical_search_pops: int
    paid_search_pops: int
    free_cache_hit_pops: int
    cached_feasible_reuses: int
    cached_infeasible_reuses: int
    candidate_score_evaluations: int
    cache_key_evaluations: int
    cache_key_item_reads: int
    cache_key_requirement_coordinate_reads: int
    cache_key_duration_reads: int
    cache_key_completion_offset_reads: int
    budget_blocked_cache_miss_lookups: int
    upper_bound_calls: int
    frontier_pushes: int
    bound_rejected_prefixes: int
    exact_stop_frontier_entries_cleared: int
    frontier_entries_remaining: int


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


@dataclass(frozen=True)
class _Candidate:
    feasible: bool
    service: tuple[int, ...]
    numerator: int
    duration: float
    score: float


def _signature_from_normalized(
    requirements: Sequence[tuple[int, ...]],
    durations: Sequence[float],
    completion_offsets: Sequence[tuple[tuple[float, ...], ...]],
    option_indices: Sequence[int | None],
    lanes: int,
    dimensions: int,
    idle_duration: float,
) -> tuple[tuple, int, int, int, int]:
    indices = tuple(option_indices)
    padded = indices + (None,) * (lanes - len(indices))
    service = [0] * dimensions
    duration = idle_duration
    ordered_offsets = [[] for _ in range(dimensions)]
    requirement_reads = 0
    duration_reads = 0
    offset_reads = 0
    for index in padded:
        if index is None:
            continue
        duration_reads += 1
        duration = max(duration, durations[index])
        for coordinate, count in enumerate(requirements[index]):
            requirement_reads += 1
            service[coordinate] += count
            local = completion_offsets[index][coordinate]
            offset_reads += len(local)
            ordered_offsets[coordinate].extend(local)
    signature = (
        tuple(service),
        duration,
        tuple(tuple(local) for local in ordered_offsets),
    )
    return signature, len(padded), requirement_reads, duration_reads, offset_reads


def fifo_execution_signature(
    options: Sequence[Any],
    option_indices: Sequence[int | None],
    lanes: int,
    dimensions: int,
    idle_duration: float = 1.0,
) -> tuple:
    """Return the exact execution key used by the post-seed cache.

    Completion offsets are concatenated separately for each class in raw lane
    order and in each option's stored local order. ``option_indices`` may be a
    search prefix; trailing lanes are padded with idle.
    """
    options = tuple(options)
    indices = tuple(option_indices)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    if int(dimensions) != dimensions or dimensions < 0:
        raise ValueError("dimensions must be a nonnegative integer")
    dimensions = int(dimensions)
    if len(indices) > lanes:
        raise ValueError("option_indices cannot exceed the lane count")
    if any(
        index is not None
        and (int(index) != index or not 0 <= int(index) < len(options))
        for index in indices
    ):
        raise ValueError("option_indices contains an unknown option")
    indices = tuple(None if index is None else int(index) for index in indices)
    idle_duration = float(idle_duration)
    if not math.isfinite(idle_duration) or idle_duration <= 0.0:
        raise ValueError("idle_duration must be positive and finite")

    requirements: list[tuple[int, ...]] = []
    durations: list[float] = []
    offsets: list[tuple[tuple[float, ...], ...]] = []
    for option in options:
        counts = tuple(option.requirements)
        if len(counts) != dimensions or any(
            int(value) != value or value < 0 for value in counts
        ):
            raise ValueError(
                "option requirements must match the nonnegative integer dimensions"
            )
        counts = tuple(int(value) for value in counts)
        if not any(counts):
            raise ValueError("a non-idle option must require at least one job")
        raw_duration = float(option.duration)
        if not math.isfinite(raw_duration) or raw_duration <= 0.0:
            raise ValueError("option durations must be positive and finite")
        duration = max(idle_duration, raw_duration)
        raw_offsets = tuple(option.completion_offsets)
        if len(raw_offsets) != dimensions:
            raise ValueError("option completion offsets must match the dimensions")
        normalized = tuple(
            tuple(float(offset) for offset in coordinate_offsets)
            for coordinate_offsets in raw_offsets
        )
        if tuple(map(len, normalized)) != counts or any(
            not math.isfinite(offset) or offset < 0.0 or offset > duration
            for coordinate_offsets in normalized
            for offset in coordinate_offsets
        ):
            raise ValueError(
                "completion offsets must match requirements and lie inside the frame"
            )
        requirements.append(counts)
        durations.append(duration)
        offsets.append(normalized)
    signature, *_ = _signature_from_normalized(
        requirements,
        durations,
        offsets,
        indices,
        lanes,
        dimensions,
        idle_duration,
    )
    return signature


def select_fifo_execution_quotient_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
) -> FifoExecutionQuotientFrameSelection:
    """Select a frame while charging each exact FIFO execution once.

    ``budget`` counts the all-idle and homogeneous seed attempts plus cache
    misses reached by the unchanged logical v4 search. Cache hits retain the
    raw ordered tuple for v4's duration/index tie break and expand its original
    prefix, but do not repeat feasibility or score computation. Once the paid
    budget is exhausted, consecutive cache hits are drained until the next
    miss so the returned frontier bound reflects all free quotient work. Every
    seed remains paid even if two catalogue entries share one signature.
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
    completion_offsets: list[tuple[tuple[float, ...], ...]] = []
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
        raw_offsets = tuple(option.completion_offsets)
        if len(raw_offsets) != len(queue):
            raise ValueError("option completion offsets must match queue coordinates")
        normalized_offsets = tuple(
            tuple(float(offset) for offset in coordinate_offsets)
            for coordinate_offsets in raw_offsets
        )
        effective_duration = max(idle_duration, duration)
        if any(
            not math.isfinite(offset)
            or offset < 0.0
            or offset > effective_duration
            for coordinate_offsets in normalized_offsets
            for offset in coordinate_offsets
        ):
            raise ValueError("completion offsets must be finite and inside the frame")
        if tuple(map(len, normalized_offsets)) != counts:
            raise ValueError("completion-offset counts must equal option requirements")
        requirements.append(counts)
        durations.append(effective_duration)
        completion_offsets.append(normalized_offsets)
        weights.append(sum(q * count for q, count in zip(queue, counts)))

    represented_ordered = (len(options) + 1) ** lanes

    def pad(indices: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return indices + (None,) * (lanes - len(indices))

    cache_key_evaluations = 0
    cache_key_item_reads = 0
    cache_key_requirement_coordinate_reads = 0
    cache_key_duration_reads = 0
    cache_key_completion_offset_reads = 0

    def cache_key(indices: tuple[int | None, ...]) -> tuple:
        nonlocal cache_key_evaluations, cache_key_item_reads
        nonlocal cache_key_requirement_coordinate_reads, cache_key_duration_reads
        nonlocal cache_key_completion_offset_reads
        signature, items, requirement_reads, duration_reads, offset_reads = (
            _signature_from_normalized(
                requirements,
                durations,
                completion_offsets,
                indices,
                lanes,
                len(queue),
                idle_duration,
            )
        )
        cache_key_evaluations += 1
        cache_key_item_reads += items
        cache_key_requirement_coordinate_reads += requirement_reads
        cache_key_duration_reads += duration_reads
        cache_key_completion_offset_reads += offset_reads
        return signature

    best_indices: tuple[int | None, ...] = (None,) * lanes
    best_service = (0,) * len(queue)
    best_duration = idle_duration
    best_score = 0.0
    attempts = 1
    feasible_count = 1
    candidate_score_evaluations = 1
    idle_key = cache_key(())
    cache: dict[tuple, _Candidate] = {
        idle_key: _Candidate(True, best_service, 0, idle_duration, 0.0)
    }

    def consider(
        indices: tuple[int | None, ...], candidate: _Candidate, *, evaluated: bool
    ) -> None:
        nonlocal best_indices, best_service, best_duration, best_score
        nonlocal feasible_count, candidate_score_evaluations
        if evaluated:
            feasible_count += 1
            candidate_score_evaluations += 1
        padded = pad(indices)
        key = (
            candidate.duration,
            tuple(len(options) if index is None else index for index in padded),
        )
        best_key = (
            best_duration,
            tuple(len(options) if index is None else index for index in best_indices),
        )
        if candidate.score > best_score or (
            candidate.score == best_score and key < best_key
        ):
            best_indices = padded
            best_service = candidate.service
            best_duration = candidate.duration
            best_score = candidate.score

    seed_copies: list[int] = []
    seed_signature_collisions = 0
    for index, counts in enumerate(requirements):
        attempts += 1
        copies = min(
            lanes,
            min(q // count for q, count in zip(queue, counts) if count),
        )
        seed_copies.append(copies)
        if copies:
            service = tuple(copies * count for count in counts)
            numerator = copies * weights[index]
            candidate = _Candidate(
                True, service, numerator, durations[index], numerator / durations[index]
            )
            key = cache_key((index,) * copies)
            cached_seed = cache.get(key)
            if cached_seed is None:
                cache[key] = candidate
            elif cached_seed != candidate:
                raise RuntimeError("equal seed execution signatures disagree")
            else:
                seed_signature_collisions += 1
            consider((index,) * copies, candidate, evaluated=True)
        else:
            # The paid zero-copy seed proves that even one copy is infeasible.
            # Associate that proof with the padded singleton encountered later.
            service = requirements[index]
            key = cache_key((index,))
            candidate = _Candidate(
                False, service, weights[index], durations[index], 0.0
            )
            cached_seed = cache.get(key)
            if cached_seed is None:
                cache[key] = candidate
            elif cached_seed.feasible:
                raise RuntimeError("equal seed execution signatures disagree")
            else:
                seed_signature_collisions += 1

    ordered: tuple[int | None, ...] = tuple(
        sorted(
            range(len(options)),
            key=lambda index: (
                seed_copies[index] == 0,
                -weights[index],
                index,
            ),
        )
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

    upper_bound_calls = 0

    def upper_bound(prefix: _Prefix) -> float:
        nonlocal threshold_evaluations, upper_bound_calls
        upper_bound_calls += 1
        remaining = lanes - len(prefix.indices)
        residual = tuple(q - used for q, used in zip(queue, prefix.service))
        best_upper = 0.0
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
    frontier_pushes = 0
    bound_rejected_prefixes = 0

    def push(prefix: _Prefix) -> None:
        nonlocal serial, frontier_peak, frontier_pushes, bound_rejected_prefixes
        upper = upper_bound(prefix)
        if upper > best_score:
            heapq.heappush(frontier, (-upper, serial, prefix))
            serial += 1
            frontier_pushes += 1
            frontier_peak = max(frontier_peak, len(frontier))
        else:
            bound_rejected_prefixes += 1

    logical_search_pops = 0
    paid_search_pops = 0
    free_cache_hit_pops = 0
    cached_feasible_reuses = 0
    cached_infeasible_reuses = 0
    budget_blocked_cache_miss_lookups = 0
    exact_stop_cleared = 0

    push(_Prefix((), (0,) * len(queue), 0, idle_duration, 0))
    while frontier:
        negative_upper, _, prefix = frontier[0]
        if -negative_upper <= best_score:
            exact_stop_cleared = len(frontier)
            frontier.clear()
            break

        position = prefix.next_index
        index = ordered[position]
        raw_indices = prefix.indices + (index,)
        key = cache_key(raw_indices)
        cached = cache.get(key)
        if cached is None and attempts >= budget:
            budget_blocked_cache_miss_lookups += 1
            break

        heapq.heappop(frontier)
        logical_search_pops += 1
        if cached is None:
            attempts += 1
            paid_search_pops += 1
            counts = (
                (0,) * len(queue) if index is None else requirements[index]
            )
            service = tuple(
                old + count for old, count in zip(prefix.service, counts)
            )
            feasible = all(count <= q for count, q in zip(service, queue))
            if feasible:
                numerator = prefix.numerator + local_weights[position]
                duration = max(prefix.duration, local_durations[position])
                cached = _Candidate(
                    True, service, numerator, duration, numerator / duration
                )
                consider(raw_indices, cached, evaluated=True)
            else:
                cached = _Candidate(False, service, 0, idle_duration, 0.0)
            cache[key] = cached
        else:
            free_cache_hit_pops += 1
            if cached.feasible:
                cached_feasible_reuses += 1
                consider(raw_indices, cached, evaluated=False)
            else:
                cached_infeasible_reuses += 1

        if cached.feasible and len(raw_indices) < lanes:
            push(_Prefix(
                raw_indices,
                cached.service,
                cached.numerator,
                cached.duration,
                0,
            ))
        if position + 1 < len(ordered):
            push(_Prefix(
                prefix.indices,
                prefix.service,
                prefix.numerator,
                prefix.duration,
                position + 1,
            ))

    upper = max(best_score, -frontier[0][0] if frontier else best_score)
    gap = max(0.0, upper - best_score)
    queue_norm = sum(queue)
    return FifoExecutionQuotientFrameSelection(
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
        represented_ordered_configurations=represented_ordered,
        cache_entries=len(cache),
        cache_hits=free_cache_hit_pops,
        cache_misses_after_seeds=paid_search_pops,
        seed_signature_collisions=seed_signature_collisions,
        logical_search_pops=logical_search_pops,
        paid_search_pops=paid_search_pops,
        free_cache_hit_pops=free_cache_hit_pops,
        cached_feasible_reuses=cached_feasible_reuses,
        cached_infeasible_reuses=cached_infeasible_reuses,
        candidate_score_evaluations=candidate_score_evaluations,
        cache_key_evaluations=cache_key_evaluations,
        cache_key_item_reads=cache_key_item_reads,
        cache_key_requirement_coordinate_reads=(
            cache_key_requirement_coordinate_reads
        ),
        cache_key_duration_reads=cache_key_duration_reads,
        cache_key_completion_offset_reads=cache_key_completion_offset_reads,
        budget_blocked_cache_miss_lookups=budget_blocked_cache_miss_lookups,
        upper_bound_calls=upper_bound_calls,
        frontier_pushes=frontier_pushes,
        bound_rejected_prefixes=bound_rejected_prefixes,
        exact_stop_frontier_entries_cleared=exact_stop_cleared,
        frontier_entries_remaining=len(frontier),
    )


__all__ = [
    "FifoExecutionQuotientFrameSelection",
    "fifo_execution_signature",
    "select_fifo_execution_quotient_frame",
]
