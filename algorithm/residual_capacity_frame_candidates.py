"""Budgeted frame construction with a residual-capacity support bound.

Each local option exposes ``requirements`` (job counts) and ``duration`` (seconds).
A frame assigns one option or idle to each identical lane, completes the sum of
the assigned requirements, and lasts the largest assigned duration, subject to
``idle_duration``. The support score is Q dot completed_jobs / frame_seconds.

The budget counts candidate construction/feasibility attempts, including idle,
one maximal feasible homogeneous seed per option, and rejected search attempts.
Local catalogue scoring and upper-bound work are reported separately.  Candidate
order, feasibility, tie breaking, and paid attempts match the frozen v4
selector.  The sole search change is a fixed binary-price Lagrangian cap on the
unexplored suffix; it never materializes the full Cartesian action family.
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
    ordered_option_indices: tuple[int | None, ...]
    bound_option_probes: int
    bound_threshold_evaluations: int
    upper_bound_calls: int
    bound_price_vector_count: int
    bound_dual_table_option_probes: int
    bound_dual_mask_evaluations: int
    bound_dual_mask_sums: int
    bound_dual_coordinate_products: int
    bound_strict_tightening_count: int
    bound_tightening_sum: float
    bound_tightening_max: float
    frontier_peak: int
    exact: bool


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


def select_residual_capacity_frame(
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

    Upper bounds retain the v4 coordinate cap and add eight binary resource-price
    masks for the three-class experiment (``2**len(queues)`` in general).  The
    zero mask reproduces the old weighted cap, so taking the minimum cannot
    loosen any prefix bound.  The price tables and their queries are counted
    separately from candidate construction.
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
    upper_bound_calls = 0
    dual_mask_evaluations = 0
    dual_mask_sums = 0
    dual_coordinate_products = 0
    strict_tightening_count = 0
    tightening_sum = 0.0
    tightening_max = 0.0

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

    # For every nonnegative binary price vector lambda, weak duality gives
    #   sum w(a) <= lambda.R + sum (w(a) - lambda.a).
    # The first remaining choice is restricted to the prefix's suffix; later
    # lanes may use any option.  Idle contributes zero, hence reduced weights
    # are clipped below at zero.  These fixed masks couple queue coordinates
    # without enumerating joint lane assignments.
    price_vectors = tuple(
        tuple(queue[coordinate] if mask & (1 << coordinate) else 0
              for coordinate in range(len(queue)))
        for mask in range(1 << len(queue))
    )
    reduced_weights = []
    for prices in price_vectors:
        local = []
        for weight, counts in zip(local_weights, local_requirements):
            dual_coordinate_products += len(queue)
            local.append(max(
                0,
                weight - sum(price * count for price, count in zip(prices, counts)),
            ))
        reduced_weights.append(tuple(local))

    dual_table_option_probes = 0
    dual_tables = []
    for duration in thresholds:
        mask_tables = []
        for local_reduced in reduced_weights:
            suffix_max = [0] * (len(ordered) + 1)
            for position in range(len(ordered) - 1, -1, -1):
                dual_table_option_probes += 1
                suffix_max[position] = suffix_max[position + 1]
                if local_durations[position] <= duration:
                    suffix_max[position] = max(
                        suffix_max[position], local_reduced[position],
                    )
            mask_tables.append(tuple(suffix_max))
        dual_tables.append(tuple(mask_tables))

    def upper_bound(prefix: _Prefix) -> float:
        nonlocal threshold_evaluations, upper_bound_calls
        nonlocal dual_mask_evaluations, dual_mask_sums, dual_coordinate_products
        nonlocal strict_tightening_count, tightening_sum, tightening_max
        upper_bound_calls += 1
        remaining = lanes - len(prefix.indices)
        residual = tuple(q - used for q, used in zip(queue, prefix.service))
        dual_coordinate_products += len(queue)
        residual_coordinate_values = tuple(
            q * left for q, left in zip(queue, residual)
        )
        price_residuals = []
        for mask in range(len(price_vectors)):
            dual_mask_sums += 1
            price_residuals.append(sum(
                value for coordinate, value in enumerate(residual_coordinate_values)
                if mask & (1 << coordinate)
            ))
        old_best_upper = 0.0
        best_upper = 0.0
        # Every completed frame has a duration in this finite threshold set.
        # Coordinate maxima may come from different options. Thus even with the
        # residual-queue cap this is a relaxation, not an uncharged frame search.
        for threshold_index, (duration, max_weights, max_counts) in enumerate(tables):
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
            dual_cap = math.inf
            for priced_residual, suffix_max in zip(
                price_residuals, dual_tables[threshold_index], strict=True,
            ):
                dual_mask_evaluations += 1
                candidate_cap = (
                    priced_residual
                    + suffix_max[prefix.next_index]
                    + (remaining - 1) * suffix_max[0]
                )
                dual_cap = min(dual_cap, candidate_cap)
            old_increment_cap = min(weighted_cap, queue_cap)
            new_increment_cap = min(weighted_cap, queue_cap, dual_cap)
            if new_increment_cap > old_increment_cap:
                raise ValueError("residual-capacity bound became looser than v4")
            old_best_upper = max(
                old_best_upper,
                (prefix.numerator + old_increment_cap) / duration,
            )
            best_upper = max(
                best_upper,
                (prefix.numerator + new_increment_cap) / duration,
            )
        reduction = old_best_upper - best_upper
        if reduction > 0.0:
            strict_tightening_count += 1
            tightening_sum += reduction
            tightening_max = max(tightening_max, reduction)
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
        ordered_option_indices=ordered,
        bound_option_probes=bound_probes,
        bound_threshold_evaluations=threshold_evaluations,
        upper_bound_calls=upper_bound_calls,
        bound_price_vector_count=len(price_vectors),
        bound_dual_table_option_probes=dual_table_option_probes,
        bound_dual_mask_evaluations=dual_mask_evaluations,
        bound_dual_mask_sums=dual_mask_sums,
        bound_dual_coordinate_products=dual_coordinate_products,
        bound_strict_tightening_count=strict_tightening_count,
        bound_tightening_sum=tightening_sum,
        bound_tightening_max=tightening_max,
        frontier_peak=frontier_peak,
        exact=gap == 0.0,
    )
