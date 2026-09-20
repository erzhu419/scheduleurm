"""Corrected continuous K26-to-K120 trace for the residual pruning sidecar.

The search below deliberately mirrors ``select_residual_sidecar_frame`` without
calling or modifying it. A single legacy-priority frontier is advanced through
``small_budget`` and then ``exact_budget`` so the checkpoints share one search
trajectory. Snapshot certificate scans do not mutate or charge search work; trace-level
counters record every snapshot that is actually computed.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence

from algorithm.residual_sidecar_frame_candidates import (
    ResidualSidecarFrameSelection,
)


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


@dataclass(frozen=True)
class ResidualSidecarFrameSearchTraceV2:
    """Two checkpoints and the exact-score discovery decomposition."""

    small: ResidualSidecarFrameSelection
    exact: ResidualSidecarFrameSelection
    first_exact_score_attempt: int
    first_k120_selected_physical_signature_attempt: int
    small_gap: float
    score_regret: float
    bound_slack: float
    gap_identity_error: float
    physical_signature_evaluations: int
    physical_signature_completion_offset_reads: int
    certificate_frontier_scans: int
    certificate_frontier_entries_scanned: int


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


def trace_residual_sidecar_frame_search_v2(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    small_budget: int = 26,
    exact_budget: int = 120,
    idle_duration: float = 1.0,
) -> ResidualSidecarFrameSearchTraceV2:
    """Trace one legacy-priority sidecar search through two checkpoints.

    Every attempt, feasibility decision, heap key, bound query, and tie break
    matches :func:`select_residual_sidecar_frame`. The extra physical-action
    reads are diagnostic only and do not enter the search work counts.
    ``exact_budget`` must be large enough to empty or certify the frontier.
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

    # Reuse paid seed feasibility and retain the original weighted-count order
    # among available options. Unavailable large batches move behind useful
    # small-queue actions without changing the order when all options fit.
    # The order is fixed for this queue, independently of the search budget.
    ordered: tuple[int | None, ...] = tuple(
        sorted(range(len(options)), key=lambda index: (
            seed_copies[index] == 0,
            -weights[index],
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

    def upper_bounds(prefix: _Prefix) -> tuple[float, float]:
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
        return old_best_upper, best_upper

    frontier: list[tuple[float, int, float, _Prefix]] = []
    virtual_legacy_serial = 0
    frontier_peak = 0
    safe_pruned_on_generation = 0
    safe_pruned_on_pop = 0
    legacy_rejected_prefixes = 0
    frontier_pushes = 0
    paid_search_pops = 0
    exact_stop_cleared = 0
    certificate_frontier_scans = 0
    certificate_frontier_entries_scanned = 0

    def push(prefix: _Prefix) -> None:
        nonlocal virtual_legacy_serial, frontier_peak, safe_pruned_on_generation
        nonlocal legacy_rejected_prefixes, frontier_pushes
        old_upper, new_upper = upper_bounds(prefix)
        if old_upper > best_score:
            serial = virtual_legacy_serial
            virtual_legacy_serial += 1
            if new_upper < best_score:
                safe_pruned_on_generation += 1
            else:
                heapq.heappush(
                    frontier, (-old_upper, serial, new_upper, prefix)
                )
                frontier_pushes += 1
                frontier_peak = max(frontier_peak, len(frontier))
        else:
            legacy_rejected_prefixes += 1

    def snapshot() -> ResidualSidecarFrameSelection:
        nonlocal certificate_frontier_scans, certificate_frontier_entries_scanned
        certificate_frontier_scans += 1
        certificate_frontier_entries_scanned += len(frontier)
        legacy_priority_upper = max(
            best_score, -frontier[0][0] if frontier else best_score
        )
        certificate_entries = len(frontier)
        upper = max(
            best_score,
            max((entry[2] for entry in frontier), default=best_score),
        )
        gap = max(0.0, upper - best_score)
        queue_norm = sum(queue)
        return ResidualSidecarFrameSelection(
            option_indices=best_indices,
            service=best_service,
            duration=best_duration,
            score=best_score,
            upper_bound=upper,
            legacy_priority_upper_bound=legacy_priority_upper,
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
            safe_pruned_prefixes_on_generation=safe_pruned_on_generation,
            safe_pruned_prefixes_on_pop=safe_pruned_on_pop,
            legacy_rejected_prefixes=legacy_rejected_prefixes,
            frontier_pushes=frontier_pushes,
            paid_search_pops=paid_search_pops,
            exact_stop_frontier_entries_cleared=exact_stop_cleared,
            virtual_legacy_serial_allocations=virtual_legacy_serial,
            certificate_frontier_scans=1,
            certificate_frontier_entries=certificate_entries,
            frontier_peak=frontier_peak,
            exact=gap == 0.0,
        )

    push(_Prefix((), (0,) * len(queue), 0, idle_duration, 0))
    small: ResidualSidecarFrameSelection | None = (
        snapshot() if attempts == small_budget else None
    )
    while attempts < exact_budget and frontier:
        negative_old_upper, _, new_upper, prefix = heapq.heappop(frontier)
        if -negative_old_upper <= best_score:
            # This is the largest remaining v4 bound, so no subtree can improve.
            exact_stop_cleared += 1 + len(frontier)
            frontier.clear()
            break
        if new_upper < best_score:
            safe_pruned_on_pop += 1
            continue
        attempts += 1
        paid_search_pops += 1
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
    return ResidualSidecarFrameSearchTraceV2(
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
        certificate_frontier_scans=certificate_frontier_scans,
        certificate_frontier_entries_scanned=certificate_frontier_entries_scanned,
    )


__all__ = [
    "ResidualSidecarFrameSearchTraceV2",
    "trace_residual_sidecar_frame_search_v2",
]
