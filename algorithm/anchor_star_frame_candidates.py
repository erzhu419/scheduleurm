"""Two-lane budgeted search over homogeneous seeds and directed anchor stars.

The all-idle frame and one maximal feasible homogeneous frame per local option
form the paid seed catalogue.  Every individually feasible option then anchors
one star whose second lane follows the frozen v4 queue-local option order.
Self choices are already covered by the homogeneous seed, and options proved
individually infeasible by their seed cannot become feasible beside an anchor;
those cursor positions are skipped without consuming candidate budget.  Every
other directed heterogeneous pair pays once for construction and feasibility.

Each anchor cursor carries an upper bound for its unvisited suffix.  The bound
uses global duration-threshold suffix maxima and caps the second lane by the
queue remaining after the anchor.  Consequently the maximum live cursor bound
certifies all untried pairs when the paid budget is exhausted.  The selector
reads only the static catalogue and the currently visible queue; it has no
arrival, rate, RNG, or future-state input.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence

from algorithm.budgeted_frame_candidates import BudgetedFrameSelection


@dataclass(frozen=True)
class AnchorStarFrameSelection(BudgetedFrameSelection):
    """A two-lane anchor-star selection with explicit search-work accounting."""

    ordered_option_indices: tuple[int, ...]
    pair_attempts: int
    feasible_pair_count: int
    active_anchor_count: int
    free_self_skips: int
    free_locally_infeasible_skips: int
    anchor_node_pops: int
    candidate_score_evaluations: int
    upper_bound_calls: int
    bound_queue_coordinate_evaluations: int
    frontier_pushes: int
    bound_rejected_anchor_nodes: int
    exhausted_anchor_cursors: int
    exact_stop_frontier_entries_cleared: int
    frontier_remaining: int
    budget_blocked_pair_lookups: int
    represented_ordered_action_count: int
    exhaustive_support_candidate_cap: int


@dataclass(frozen=True)
class _AnchorCursor:
    anchor_index: int
    next_position: int


def select_anchor_star_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
) -> AnchorStarFrameSelection:
    """Select a two-lane frame from a paid best-first anchor-star search.

    ``budget`` must cover the all-idle attempt and one maximal homogeneous seed
    attempt per option.  After those seeds, every paid attempt is one ordered
    pair of distinct, individually feasible options.  At most
    ``len(options) + 1 + len(options) * (len(options) - 1)`` paid attempts are
    needed to exhaust the support-relevant two-lane family.  ``exact`` certifies
    the maximum support value.  An equal-support action may remain unevaluated
    after that certificate closes, including one with a shorter duration or a
    different physical signature or raw lane tuple.
    """
    options = tuple(options)
    queues = tuple(queues)
    if int(lanes) != lanes or int(lanes) != 2:
        raise ValueError("anchor-star search requires exactly two lanes")
    lanes = 2
    seed_attempts = len(options) + 1
    if int(budget) != budget or budget < seed_attempts:
        raise ValueError(
            f"budget must be an integer covering {seed_attempts} seed attempts"
        )
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
        weights.append(sum(q * count for q, count in zip(queue, counts, strict=True)))

    option_count = len(options)
    best_indices: tuple[int | None, ...] = (None, None)
    best_service = (0,) * len(queue)
    best_duration = idle_duration
    best_score = 0.0
    attempts = 1
    feasible_count = 1
    candidate_score_evaluations = 1

    def consider(
        indices: tuple[int | None, ...],
        service: tuple[int, ...],
        numerator: int,
        duration: float,
    ) -> None:
        nonlocal best_indices, best_service, best_duration, best_score
        nonlocal feasible_count, candidate_score_evaluations
        feasible_count += 1
        candidate_score_evaluations += 1
        padded = indices + (None,) * (lanes - len(indices))
        score = numerator / duration
        key = (
            duration,
            tuple(option_count if index is None else index for index in padded),
        )
        best_key = (
            best_duration,
            tuple(option_count if index is None else index for index in best_indices),
        )
        if score > best_score or (score == best_score and key < best_key):
            best_indices = padded
            best_service = service
            best_duration = duration
            best_score = score

    seed_copies: list[int] = []
    for index, counts in enumerate(requirements):
        attempts += 1
        copies = min(
            lanes,
            min(q // count for q, count in zip(queue, counts, strict=True) if count),
        )
        seed_copies.append(copies)
        if copies:
            consider(
                (index,) * copies,
                tuple(copies * count for count in counts),
                copies * weights[index],
                durations[index],
            )

    ordered = tuple(
        sorted(
            range(option_count),
            key=lambda index: (
                seed_copies[index] == 0,
                -weights[index],
                index,
            ),
        )
    )
    thresholds = tuple(sorted({idle_duration, *durations}))
    bound_option_probes = 0
    bound_threshold_evaluations = 0
    bound_queue_coordinate_evaluations = 0

    tables: list[tuple[float, list[int], list[tuple[int, ...]]]] = []
    for duration in thresholds:
        max_weights = [0] * (option_count + 1)
        max_counts = [(0,) * len(queue)] * (option_count + 1)
        for position in range(option_count - 1, -1, -1):
            bound_option_probes += 1
            max_weights[position] = max_weights[position + 1]
            max_counts[position] = max_counts[position + 1]
            index = ordered[position]
            if durations[index] <= duration:
                max_weights[position] = max(max_weights[position], weights[index])
                max_counts[position] = tuple(
                    max(old, count)
                    for old, count in zip(
                        max_counts[position], requirements[index], strict=True
                    )
                )
        tables.append((duration, max_weights, max_counts))

    upper_bound_calls = 0

    def upper_bound(cursor: _AnchorCursor) -> float:
        nonlocal upper_bound_calls, bound_threshold_evaluations
        nonlocal bound_queue_coordinate_evaluations
        upper_bound_calls += 1
        anchor = cursor.anchor_index
        residual = tuple(
            q - count
            for q, count in zip(queue, requirements[anchor], strict=True)
        )
        best_upper = 0.0
        for duration, max_weights, max_counts in tables:
            if duration < durations[anchor]:
                continue
            bound_threshold_evaluations += 1
            bound_queue_coordinate_evaluations += len(queue)
            weighted_cap = max_weights[cursor.next_position]
            queue_cap = sum(
                q * min(left, count)
                for q, left, count in zip(
                    queue,
                    residual,
                    max_counts[cursor.next_position],
                    strict=True,
                )
            )
            numerator = weights[anchor] + min(weighted_cap, queue_cap)
            best_upper = max(best_upper, numerator / duration)
        return best_upper

    frontier: list[tuple[float, int, int, _AnchorCursor]] = []
    serial = 0
    frontier_pushes = 0
    bound_rejected_anchor_nodes = 0
    frontier_peak = 0

    def push(cursor: _AnchorCursor) -> None:
        nonlocal serial, frontier_pushes, bound_rejected_anchor_nodes
        nonlocal frontier_peak
        upper = upper_bound(cursor)
        if upper > best_score:
            # Equal bounds prefer the smaller original first-lane index.  This
            # keeps the anchor-star tie deterministic even when advancing past
            # a free self slot allocates a later heap serial.
            heapq.heappush(
                frontier, (-upper, cursor.anchor_index, serial, cursor)
            )
            serial += 1
            frontier_pushes += 1
            frontier_peak = max(frontier_peak, len(frontier))
        else:
            bound_rejected_anchor_nodes += 1

    active_anchors = tuple(
        index for index, copies in enumerate(seed_copies) if copies > 0
    )
    if option_count:
        for anchor in active_anchors:
            push(_AnchorCursor(anchor, 0))

    pair_attempts = 0
    feasible_pair_count = 0
    free_self_skips = 0
    free_locally_infeasible_skips = 0
    anchor_node_pops = 0
    exhausted_anchor_cursors = 0
    exact_stop_frontier_entries_cleared = 0
    budget_blocked_pair_lookups = 0

    def advance(cursor: _AnchorCursor) -> None:
        nonlocal exhausted_anchor_cursors
        position = cursor.next_position + 1
        if position < option_count:
            push(_AnchorCursor(cursor.anchor_index, position))
        else:
            exhausted_anchor_cursors += 1

    while frontier:
        negative_upper, _, _, cursor = frontier[0]
        if -negative_upper <= best_score:
            exact_stop_frontier_entries_cleared += len(frontier)
            frontier.clear()
            break

        second = ordered[cursor.next_position]
        if second == cursor.anchor_index:
            heapq.heappop(frontier)
            anchor_node_pops += 1
            free_self_skips += 1
            advance(cursor)
            continue
        if seed_copies[second] == 0:
            heapq.heappop(frontier)
            anchor_node_pops += 1
            free_locally_infeasible_skips += 1
            advance(cursor)
            continue
        if attempts >= budget:
            budget_blocked_pair_lookups += 1
            break

        heapq.heappop(frontier)
        anchor_node_pops += 1
        attempts += 1
        pair_attempts += 1
        service = tuple(
            left + right
            for left, right in zip(
                requirements[cursor.anchor_index],
                requirements[second],
                strict=True,
            )
        )
        if all(count <= q for count, q in zip(service, queue, strict=True)):
            feasible_pair_count += 1
            consider(
                (cursor.anchor_index, second),
                service,
                weights[cursor.anchor_index] + weights[second],
                max(durations[cursor.anchor_index], durations[second]),
            )
        advance(cursor)

    upper = max(best_score, -frontier[0][0] if frontier else best_score)
    gap = max(0.0, upper - best_score)
    queue_norm = sum(queue)
    return AnchorStarFrameSelection(
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
        local_score_evaluations=option_count,
        bound_option_probes=bound_option_probes,
        bound_threshold_evaluations=bound_threshold_evaluations,
        frontier_peak=frontier_peak,
        exact=gap == 0.0,
        ordered_option_indices=ordered,
        pair_attempts=pair_attempts,
        feasible_pair_count=feasible_pair_count,
        active_anchor_count=len(active_anchors),
        free_self_skips=free_self_skips,
        free_locally_infeasible_skips=free_locally_infeasible_skips,
        anchor_node_pops=anchor_node_pops,
        candidate_score_evaluations=candidate_score_evaluations,
        upper_bound_calls=upper_bound_calls,
        bound_queue_coordinate_evaluations=bound_queue_coordinate_evaluations,
        frontier_pushes=frontier_pushes,
        bound_rejected_anchor_nodes=bound_rejected_anchor_nodes,
        exhausted_anchor_cursors=exhausted_anchor_cursors,
        exact_stop_frontier_entries_cleared=(
            exact_stop_frontier_entries_cleared
        ),
        frontier_remaining=len(frontier),
        budget_blocked_pair_lookups=budget_blocked_pair_lookups,
        represented_ordered_action_count=(option_count + 1) ** lanes,
        exhaustive_support_candidate_cap=(
            seed_attempts + option_count * (option_count - 1)
        ),
    )
