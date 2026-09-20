"""Paired first-attempt trace for the v4 and residual-capacity bounds.

The trace advances the two searches together only while their paid candidate
attempts agree.  Every generated prefix receives both upper bounds, so the first
different paid attempt can be attributed to a prefix removed by the tighter
bound or to a different ordering of prefixes retained in both frontiers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


def _prefix_record(prefix: _Prefix, ordered: tuple[int | None, ...]) -> dict:
    return {
        "indices": prefix.indices,
        "service": prefix.service,
        "numerator": prefix.numerator,
        "duration": prefix.duration,
        "next_index": prefix.next_index,
        "next_option_index": ordered[prefix.next_index],
    }


def trace_first_bound_divergence(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int = 26,
    idle_duration: float = 1.0,
) -> dict:
    """Return the first paid search difference under the old and new bounds."""
    options = tuple(options)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    seed_attempts = len(options) + 1
    if int(budget) != budget or budget < seed_attempts:
        raise ValueError(f"budget must be an integer covering {seed_attempts} seeds")
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
            raise ValueError("option requirements must match the queue coordinates")
        counts = tuple(int(value) for value in counts)
        if not any(counts):
            raise ValueError("a non-idle option must require at least one job")
        duration = float(option.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("option durations must be positive and finite")
        requirements.append(counts)
        durations.append(max(idle_duration, duration))
        weights.append(sum(q * count for q, count in zip(queue, counts)))

    best_score = 0.0
    attempts = 1
    seed_copies = []
    for index, counts in enumerate(requirements):
        attempts += 1
        copies = min(
            lanes,
            min(q // count for q, count in zip(queue, counts) if count),
        )
        seed_copies.append(copies)
        if copies:
            best_score = max(
                best_score,
                copies * weights[index] / durations[index],
            )
    if attempts != seed_attempts:
        raise ValueError("seed attempt accounting did not close")

    ordered: tuple[int | None, ...] = tuple(
        sorted(
            range(len(options)),
            key=lambda index: (seed_copies[index] == 0, -weights[index], index),
        )
    ) + (None,)
    local_weights = tuple(0 if index is None else weights[index] for index in ordered)
    local_durations = tuple(
        idle_duration if index is None else durations[index] for index in ordered
    )
    local_requirements = tuple(
        (0,) * len(queue) if index is None else requirements[index]
        for index in ordered
    )
    thresholds = tuple(sorted({idle_duration, *durations}))

    bound_option_probes = 0
    tables = []
    for duration in thresholds:
        max_weights = [0] * (len(ordered) + 1)
        max_counts = [(0,) * len(queue)] * (len(ordered) + 1)
        for position in range(len(ordered) - 1, -1, -1):
            bound_option_probes += 1
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
        tables.append((duration, tuple(max_weights), tuple(max_counts)))

    price_vectors = tuple(
        tuple(
            queue[coordinate] if mask & (1 << coordinate) else 0
            for coordinate in range(len(queue))
        )
        for mask in range(1 << len(queue))
    )
    dual_coordinate_products = 0
    reduced_weights = []
    for prices in price_vectors:
        local = []
        for weight, counts in zip(local_weights, local_requirements):
            dual_coordinate_products += len(queue)
            local.append(max(
                0,
                weight - sum(
                    price * count for price, count in zip(prices, counts)
                ),
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
                        suffix_max[position], local_reduced[position]
                    )
            mask_tables.append(tuple(suffix_max))
        dual_tables.append(tuple(mask_tables))

    bound_pair_calls = 0
    bound_threshold_evaluations = 0
    dual_mask_evaluations = 0
    dual_mask_sums = 0
    all_new_bounds_nonlooser = True

    def bound_pair(prefix: _Prefix) -> tuple[float, float]:
        nonlocal bound_pair_calls, bound_threshold_evaluations
        nonlocal dual_mask_evaluations, dual_mask_sums
        nonlocal dual_coordinate_products, all_new_bounds_nonlooser
        bound_pair_calls += 1
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
                value
                for coordinate, value in enumerate(residual_coordinate_values)
                if mask & (1 << coordinate)
            ))

        old_upper = 0.0
        new_upper = 0.0
        for threshold_index, (duration, max_weights, max_counts) in enumerate(tables):
            if duration < prefix.duration:
                continue
            bound_threshold_evaluations += 1
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
                price_residuals, dual_tables[threshold_index], strict=True
            ):
                dual_mask_evaluations += 1
                dual_cap = min(
                    dual_cap,
                    priced_residual
                    + suffix_max[prefix.next_index]
                    + (remaining - 1) * suffix_max[0],
                )
            old_increment = min(weighted_cap, queue_cap)
            new_increment = min(weighted_cap, queue_cap, dual_cap)
            old_upper = max(
                old_upper, (prefix.numerator + old_increment) / duration
            )
            new_upper = max(
                new_upper, (prefix.numerator + new_increment) / duration
            )
        if new_upper > old_upper:
            all_new_bounds_nonlooser = False
            raise ValueError("residual-capacity bound became looser than v4")
        return old_upper, new_upper

    generations: dict[int, dict] = {}
    legacy_frontier: list[tuple[float, int, int]] = []
    residual_frontier: list[tuple[float, int, int]] = []
    next_generation = 0
    legacy_serial = 0
    residual_serial = 0
    first_strict_bound_change = None
    first_residual_only_prune = None

    def generate(
        prefix: _Prefix,
        parent_attempt: int,
        relation: str,
        parent_generation: int | None,
    ) -> int:
        nonlocal next_generation, legacy_serial, residual_serial
        nonlocal first_strict_bound_change, first_residual_only_prune
        generation = next_generation
        next_generation += 1
        old_upper, new_upper = bound_pair(prefix)
        legacy_retained = old_upper > best_score
        residual_retained = new_upper > best_score
        if residual_retained and not legacy_retained:
            raise ValueError("new-bound frontier is not a subset at common incumbent")
        record = {
            "generation": generation,
            "parent_attempt": parent_attempt,
            "relation": relation,
            "parent_generation": parent_generation,
            "prefix": _prefix_record(prefix, ordered),
            "old_upper_bound": old_upper,
            "new_upper_bound": new_upper,
            "tightening": old_upper - new_upper,
            "push_incumbent_score": best_score,
            "legacy_retained": legacy_retained,
            "residual_retained": residual_retained,
            "legacy_serial": legacy_serial if legacy_retained else None,
            "residual_serial": residual_serial if residual_retained else None,
        }
        generations[generation] = record
        if old_upper > new_upper and first_strict_bound_change is None:
            first_strict_bound_change = generation
        if legacy_retained and not residual_retained and first_residual_only_prune is None:
            first_residual_only_prune = generation
        if legacy_retained:
            heapq.heappush(
                legacy_frontier, (-old_upper, legacy_serial, generation)
            )
            legacy_serial += 1
        if residual_retained:
            heapq.heappush(
                residual_frontier, (-new_upper, residual_serial, generation)
            )
            residual_serial += 1
        return generation

    def node_record(generation: int | None) -> dict | None:
        return None if generation is None else dict(generations[generation])

    def frontier_ids(frontier) -> Counter:
        return Counter(entry[2] for entry in frontier)

    common_attempts = []
    generate(
        _Prefix((), (0,) * len(queue), 0, idle_duration, 0),
        seed_attempts,
        "root",
        None,
    )
    classification = "no_divergence_within_budget"
    first_divergence_attempt = None
    divergence = None

    while attempts < budget:
        legacy_top = legacy_frontier[0][2] if legacy_frontier else None
        residual_top = residual_frontier[0][2] if residual_frontier else None
        legacy_can_continue = (
            legacy_top is not None
            and generations[legacy_top]["old_upper_bound"] > best_score
        )
        residual_can_continue = (
            residual_top is not None
            and generations[residual_top]["new_upper_bound"] > best_score
        )
        if not legacy_can_continue and not residual_can_continue:
            break

        if legacy_top != residual_top or legacy_can_continue != residual_can_continue:
            first_divergence_attempt = attempts + 1
            legacy_ids = frontier_ids(legacy_frontier)
            residual_ids = frontier_ids(residual_frontier)
            both_tops_cross_present = (
                legacy_top is not None
                and residual_top is not None
                and residual_ids[legacy_top] > 0
                and legacy_ids[residual_top] > 0
                and legacy_can_continue
                and residual_can_continue
            )
            classification = (
                "heap_reprioritization"
                if both_tops_cross_present else "early_pruning"
            )
            legacy_order = [entry[2] for entry in sorted(legacy_frontier)]
            residual_order = [entry[2] for entry in sorted(residual_frontier)]
            divergence = {
                "incumbent_score": best_score,
                "legacy_top": node_record(legacy_top),
                "residual_top": node_record(residual_top),
                "legacy_top_present_in_residual_frontier": (
                    legacy_top is not None and residual_ids[legacy_top] > 0
                ),
                "residual_top_present_in_legacy_frontier": (
                    residual_top is not None and legacy_ids[residual_top] > 0
                ),
                "legacy_top_rank_under_new_key": (
                    None if legacy_top not in residual_order
                    else residual_order.index(legacy_top) + 1
                ),
                "residual_top_rank_under_old_key": (
                    None if residual_top not in legacy_order
                    else legacy_order.index(residual_top) + 1
                ),
                "legacy_frontier_size": len(legacy_frontier),
                "residual_frontier_size": len(residual_frontier),
                "common_frontier_generation_count": sum(
                    (legacy_ids & residual_ids).values()
                ),
                "frontier_membership_equal": legacy_ids == residual_ids,
                "legacy_can_continue": legacy_can_continue,
                "residual_can_continue": residual_can_continue,
            }
            break

        generation = legacy_top
        heapq.heappop(legacy_frontier)
        heapq.heappop(residual_frontier)
        prefix_data = generations[generation]["prefix"]
        prefix = _Prefix(
            tuple(prefix_data["indices"]),
            tuple(prefix_data["service"]),
            prefix_data["numerator"],
            prefix_data["duration"],
            prefix_data["next_index"],
        )
        attempts += 1
        incumbent_before = best_score
        position = prefix.next_index
        index = ordered[position]
        counts = (0,) * len(queue) if index is None else requirements[index]
        service = tuple(
            old + count for old, count in zip(prefix.service, counts)
        )
        feasible = all(count <= q for count, q in zip(service, queue))
        candidate_score = None
        generated = []
        if feasible:
            indices = prefix.indices + (index,)
            numerator = prefix.numerator + local_weights[position]
            duration = max(prefix.duration, local_durations[position])
            candidate_score = numerator / duration
            best_score = max(best_score, candidate_score)
            if len(indices) < lanes:
                generated.append(generate(
                    _Prefix(indices, service, numerator, duration, 0),
                    attempts,
                    "child",
                    generation,
                ))
        if position + 1 < len(ordered):
            generated.append(generate(
                _Prefix(
                    prefix.indices,
                    prefix.service,
                    prefix.numerator,
                    prefix.duration,
                    position + 1,
                ),
                attempts,
                "sibling",
                generation,
            ))
        common_attempts.append({
            "attempt": attempts,
            "generation": generation,
            "prefix": _prefix_record(prefix, ordered),
            "choice_option_index": index,
            "feasible": feasible,
            "candidate_score": candidate_score,
            "incumbent_before": incumbent_before,
            "incumbent_after": best_score,
            "generated": generated,
        })

    residual_only_prunes = [
        generation
        for generation, row in generations.items()
        if row["legacy_retained"] and not row["residual_retained"]
    ]
    return {
        "queue": queue,
        "budget": budget,
        "seed_attempt_count": seed_attempts,
        "seed_incumbent_score": common_attempts[0]["incumbent_before"]
        if common_attempts else best_score,
        "ordered_option_indices": ordered,
        "common_paid_attempts": common_attempts,
        "common_paid_attempt_count": len(common_attempts),
        "first_divergence_attempt": first_divergence_attempt,
        "classification": classification,
        "divergence": divergence,
        "first_strict_bound_change": node_record(first_strict_bound_change),
        "first_residual_only_prune": node_record(first_residual_only_prune),
        "residual_only_prune_count_before_divergence": len(residual_only_prunes),
        "all_new_bounds_nonlooser": all_new_bounds_nonlooser,
        "work": {
            "generated_prefixes": len(generations),
            "bound_pair_calls": bound_pair_calls,
            "bound_option_probes": bound_option_probes,
            "bound_threshold_evaluations": bound_threshold_evaluations,
            "price_masks": len(price_vectors),
            "dual_table_option_probes": dual_table_option_probes,
            "dual_mask_evaluations": dual_mask_evaluations,
            "dual_mask_sums": dual_mask_sums,
            "dual_coordinate_products": dual_coordinate_products,
        },
    }


__all__ = ["trace_first_bound_divergence"]
