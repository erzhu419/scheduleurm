"""One-search budget curve for the frozen legacy-v4 frame selector.

The implementation mirrors :func:`select_budgeted_frame` while retaining one
frontier across all requested budget checkpoints.  For the aligned hardware
catalogue the default checkpoints are every integer budget from K10 through
K120.  Once the support certificate closes, later checkpoints carry the same
selection and cumulative selector work forward without further search.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence

from algorithm.budgeted_frame_candidates import BudgetedFrameSelection


DEFAULT_CHECKPOINT_BUDGETS = tuple(range(10, 121))


@dataclass(frozen=True)
class _Prefix:
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    numerator: int
    duration: float
    next_index: int


@dataclass(frozen=True)
class BudgetedFrameSearchCheckpoint:
    """The frozen selector result at one allowed paid-attempt budget."""

    budget: int
    selection: BudgetedFrameSelection
    carried_after_exact: bool


@dataclass(frozen=True)
class BudgetedFrameSearchCurveWork:
    """Work performed by the single continuous search.

    ``frontier_entries_discarded_on_exact`` counts entries removed by the final
    bulk clear.  The entry used to detect closure is counted in
    ``certificate_closure_pops`` instead.
    """

    trace_calls: int
    checkpoint_reads: int
    carried_exact_checkpoint_reads: int
    paid_candidate_attempts: int
    seed_attempts: int
    feasible_candidate_evaluations: int
    infeasible_candidate_attempts: int
    local_score_evaluations: int
    bound_option_probes: int
    bound_threshold_evaluations: int
    upper_bound_calls: int
    frontier_push_attempts: int
    frontier_pushes: int
    bound_rejected_pushes: int
    frontier_pops: int
    expanded_prefixes: int
    certificate_closure_pops: int
    frontier_entries_discarded_on_exact: int
    frontier_peak: int
    frontier_remaining: int


@dataclass(frozen=True)
class BudgetedFrameSearchCurve:
    """Selections at ordered budget checkpoints from one search frontier."""

    checkpoint_budgets: tuple[int, ...]
    checkpoints: tuple[BudgetedFrameSearchCheckpoint, ...]
    first_exact_budget: int | None
    first_exact_paid_attempt: int | None
    work: BudgetedFrameSearchCurveWork

    def selection_at(self, budget: int) -> BudgetedFrameSelection:
        """Return the selection recorded at ``budget``."""

        for checkpoint in self.checkpoints:
            if checkpoint.budget == budget:
                return checkpoint.selection
        raise KeyError(budget)


def _checkpoint_budgets(
    values: Sequence[int],
    seed_attempts: int,
) -> tuple[int, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError("checkpoint_budgets must not be empty")
    budgets: list[int] = []
    for value in raw:
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("checkpoint budgets must be integers") from error
        if integer != value:
            raise ValueError("checkpoint budgets must be integers")
        budgets.append(integer)
    if any(right <= left for left, right in zip(budgets, budgets[1:])):
        raise ValueError("checkpoint budgets must be strictly increasing")
    if budgets[0] < seed_attempts:
        raise ValueError(
            f"checkpoint budgets must cover {seed_attempts} seed attempts"
        )
    return tuple(budgets)


def trace_budgeted_frame_search_curve(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    checkpoint_budgets: Sequence[int] = DEFAULT_CHECKPOINT_BUDGETS,
    idle_duration: float = 1.0,
) -> BudgetedFrameSearchCurve:
    """Run one legacy-v4 search and snapshot every requested budget.

    Candidate attempts, feasibility decisions, ordering, bounds and tie breaks
    match :func:`algorithm.budgeted_frame_candidates.select_budgeted_frame`.
    Checkpoints must be strictly increasing and cover the seed catalogue.  The
    endpoint may remain inexact; callers that need an exact reference must gate
    the final checkpoint explicitly.
    """

    options = tuple(options)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    seed_attempts = len(options) + 1
    budgets = _checkpoint_budgets(checkpoint_budgets, seed_attempts)
    final_budget = budgets[-1]
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

    best_indices: tuple[int | None, ...] = (None,) * lanes
    best_service = (0,) * len(queue)
    best_duration = idle_duration
    best_score = 0.0
    attempts = 1
    feasible_count = 1

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
        key = (
            duration,
            tuple(len(options) if index is None else index for index in padded),
        )
        best_key = (
            best_duration,
            tuple(
                len(options) if index is None else index
                for index in best_indices
            ),
        )
        if score > best_score or (score == best_score and key < best_key):
            best_indices = padded
            best_service = service
            best_duration = duration
            best_score = score

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
        idle_duration if index is None else durations[index]
        for index in ordered
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
    frontier_push_attempts = 0
    frontier_pushes = 0
    bound_rejected_pushes = 0
    frontier_pops = 0
    expanded_prefixes = 0
    certificate_closure_pops = 0
    frontier_entries_discarded_on_exact = 0

    def push(prefix: _Prefix) -> None:
        nonlocal serial, frontier_peak
        nonlocal frontier_push_attempts, frontier_pushes, bound_rejected_pushes
        frontier_push_attempts += 1
        upper = upper_bound(prefix)
        if upper > best_score:
            heapq.heappush(frontier, (-upper, serial, prefix))
            serial += 1
            frontier_pushes += 1
            frontier_peak = max(frontier_peak, len(frontier))
        else:
            bound_rejected_pushes += 1

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

    requested = set(budgets)
    selections: dict[int, BudgetedFrameSelection] = {}
    first_exact_paid_attempt: int | None = None

    def observe_paid_attempt() -> None:
        nonlocal first_exact_paid_attempt
        current = snapshot()
        if current.exact and first_exact_paid_attempt is None:
            first_exact_paid_attempt = attempts
        if attempts in requested:
            selections[attempts] = current

    observe_paid_attempt()
    while attempts < final_budget and frontier:
        negative_upper, _, prefix = heapq.heappop(frontier)
        frontier_pops += 1
        if -negative_upper <= best_score:
            certificate_closure_pops += 1
            frontier_entries_discarded_on_exact += len(frontier)
            frontier.clear()
            if first_exact_paid_attempt is None:
                first_exact_paid_attempt = attempts
            break
        attempts += 1
        expanded_prefixes += 1
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
        if position + 1 < len(ordered):
            push(
                _Prefix(
                    prefix.indices,
                    prefix.service,
                    prefix.numerator,
                    prefix.duration,
                    position + 1,
                )
            )
        observe_paid_attempt()

    terminal = snapshot()
    if terminal.exact and first_exact_paid_attempt is None:
        first_exact_paid_attempt = attempts
    for budget in budgets:
        if budget not in selections:
            if not terminal.exact:
                raise RuntimeError(
                    "search stopped before a requested checkpoint without an exact certificate"
                )
            selections[budget] = terminal

    checkpoints = tuple(
        BudgetedFrameSearchCheckpoint(
            budget=budget,
            selection=selections[budget],
            carried_after_exact=(
                first_exact_paid_attempt is not None
                and budget > first_exact_paid_attempt
            ),
        )
        for budget in budgets
    )
    first_exact_budget = next(
        (
            checkpoint.budget
            for checkpoint in checkpoints
            if checkpoint.selection.exact
        ),
        None,
    )
    last_selection = checkpoints[-1].selection
    work = BudgetedFrameSearchCurveWork(
        trace_calls=1,
        checkpoint_reads=len(checkpoints),
        carried_exact_checkpoint_reads=sum(
            checkpoint.carried_after_exact for checkpoint in checkpoints
        ),
        paid_candidate_attempts=attempts,
        seed_attempts=seed_attempts,
        feasible_candidate_evaluations=feasible_count,
        infeasible_candidate_attempts=attempts - feasible_count,
        local_score_evaluations=len(options),
        bound_option_probes=bound_probes,
        bound_threshold_evaluations=threshold_evaluations,
        upper_bound_calls=upper_bound_calls,
        frontier_push_attempts=frontier_push_attempts,
        frontier_pushes=frontier_pushes,
        bound_rejected_pushes=bound_rejected_pushes,
        frontier_pops=frontier_pops,
        expanded_prefixes=expanded_prefixes,
        certificate_closure_pops=certificate_closure_pops,
        frontier_entries_discarded_on_exact=(
            frontier_entries_discarded_on_exact
        ),
        frontier_peak=frontier_peak,
        frontier_remaining=len(frontier),
    )
    if (
        last_selection.attempts != work.paid_candidate_attempts
        or last_selection.feasible_candidate_count
        != work.feasible_candidate_evaluations
        or last_selection.local_score_evaluations
        != work.local_score_evaluations
        or last_selection.bound_option_probes != work.bound_option_probes
        or last_selection.bound_threshold_evaluations
        != work.bound_threshold_evaluations
        or last_selection.frontier_peak != work.frontier_peak
    ):
        raise RuntimeError("curve work does not match the final selector checkpoint")
    return BudgetedFrameSearchCurve(
        checkpoint_budgets=budgets,
        checkpoints=checkpoints,
        first_exact_budget=first_exact_budget,
        first_exact_paid_attempt=first_exact_paid_attempt,
        work=work,
    )


__all__ = [
    "DEFAULT_CHECKPOINT_BUDGETS",
    "BudgetedFrameSearchCheckpoint",
    "BudgetedFrameSearchCurve",
    "BudgetedFrameSearchCurveWork",
    "trace_budgeted_frame_search_curve",
]
