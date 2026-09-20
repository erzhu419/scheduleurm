"""Budgeted frame search with free reuse of lane-permutation candidates.

The logical prefix heap, option order, support objective, upper bound, and raw
lane-order tie break are the same as :mod:`algorithm.budgeted_frame_candidates`.
Only candidate accounting changes: the first ordered representative of an
option multiset pays for construction and feasibility, while later lane
permutations reuse that result without consuming candidate budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class PermutationQuotientFrameSelection:
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
    represented_lane_multisets: int
    symmetry_duplicate_ordered_configurations: int
    cache_entries: int
    cache_hits: int
    cache_misses_after_seeds: int
    logical_search_pops: int
    paid_search_pops: int
    free_cache_hit_pops: int
    cached_feasible_reuses: int
    cached_infeasible_reuses: int
    candidate_score_evaluations: int
    cache_key_evaluations: int
    cache_key_item_reads: int
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


def select_permutation_quotient_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
) -> PermutationQuotientFrameSelection:
    """Select a frame while charging each lane permutation orbit once.

    ``budget`` counts the all-idle and homogeneous seed attempts plus cache
    misses reached by the unchanged logical v4 search. Cache hits retain the
    raw ordered tuple for v4's duration/index tie break and expand its original
    prefix, but do not repeat feasibility or score computation. Once the paid
    budget is exhausted, consecutive cache hits are drained until the next
    miss so the returned frontier bound reflects all free quotient work.
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
        requirements.append(counts)
        durations.append(max(idle_duration, duration))
        weights.append(sum(q * count for q, count in zip(queue, counts)))

    represented_ordered = (len(options) + 1) ** lanes
    represented_multisets = math.comb(len(options) + lanes, lanes)

    def pad(indices: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return indices + (None,) * (lanes - len(indices))

    def cache_key(indices: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return tuple(sorted(
            pad(indices),
            key=lambda index: (
                index is None,
                len(options) if index is None else index,
            ),
        ))

    best_indices: tuple[int | None, ...] = (None,) * lanes
    best_service = (0,) * len(queue)
    best_duration = idle_duration
    best_score = 0.0
    attempts = 1
    feasible_count = 1
    candidate_score_evaluations = 1
    idle_key = cache_key(())
    cache: dict[tuple[int | None, ...], _Candidate] = {
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
            if key in cache:
                raise RuntimeError("homogeneous seed cache keys must be unique")
            cache[key] = candidate
            consider((index,) * copies, candidate, evaluated=True)
        else:
            # The paid zero-copy seed proves that even one copy is infeasible.
            # Associate that proof with the padded singleton encountered later.
            service = requirements[index]
            key = cache_key((index,))
            if key in cache:
                raise RuntimeError("homogeneous seed cache keys must be unique")
            cache[key] = _Candidate(
                False, service, weights[index], durations[index], 0.0
            )

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
    cache_key_evaluations = seed_attempts
    cache_key_item_reads = lanes * seed_attempts
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
        cache_key_evaluations += 1
        cache_key_item_reads += lanes
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
    return PermutationQuotientFrameSelection(
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
        represented_lane_multisets=represented_multisets,
        symmetry_duplicate_ordered_configurations=(
            represented_ordered - represented_multisets
        ),
        cache_entries=len(cache),
        cache_hits=free_cache_hit_pops,
        cache_misses_after_seeds=paid_search_pops,
        logical_search_pops=logical_search_pops,
        paid_search_pops=paid_search_pops,
        free_cache_hit_pops=free_cache_hit_pops,
        cached_feasible_reuses=cached_feasible_reuses,
        cached_infeasible_reuses=cached_infeasible_reuses,
        candidate_score_evaluations=candidate_score_evaluations,
        cache_key_evaluations=cache_key_evaluations,
        cache_key_item_reads=cache_key_item_reads,
        budget_blocked_cache_miss_lookups=budget_blocked_cache_miss_lookups,
        upper_bound_calls=upper_bound_calls,
        frontier_pushes=frontier_pushes,
        bound_rejected_prefixes=bound_rejected_prefixes,
        exact_stop_frontier_entries_cleared=exact_stop_cleared,
        frontier_entries_remaining=len(frontier),
    )
