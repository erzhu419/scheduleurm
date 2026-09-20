"""Canonical count-space B&B followed by a bounded one-lane exchange polish.

The base selector and its certified upper bound are unchanged.  Only when the
base score is not certified exact, the polish checks each distinct move of one
lane from an occupied option (or idle) to another option (or idle).  The best
feasible score, with the established canonical key as the tie-break, replaces
the incumbent.  Every attempted exchange is charged explicitly; no LP or
frontier work is hidden or reused outside the base call.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    CanonicalCountSpaceBnbSelection,
    select_canonical_count_space_bnb_frame,
)


@dataclass(frozen=True)
class CanonicalSingleExchangeSelection:
    base_selection: CanonicalCountSpaceBnbSelection
    selection: CanonicalCountSpaceBnbSelection
    polish_triggered: bool
    exchange_attempts: int
    feasible_exchange_count: int
    score_improving_exchange_count: int
    canonical_tie_improving_exchange_count: int
    incumbent_changed: bool


def _candidate(
    options: Sequence[Any],
    queue: Sequence[int],
    lanes: int,
    counts_with_idle: Sequence[int],
    idle_duration: float,
) -> tuple[
    tuple[int, ...],
    tuple[int | None, ...],
    tuple[int, ...],
    Fraction,
    Fraction,
] | None:
    option_count = len(options)
    counts = tuple(int(value) for value in counts_with_idle)
    if (
        len(counts) != option_count + 1
        or any(value < 0 for value in counts)
        or sum(counts) != lanes
    ):
        raise ValueError("exchange counts must cover every lane exactly once")
    service = tuple(
        sum(
            counts[index] * int(options[index].requirements[coordinate])
            for index in range(option_count)
        )
        for coordinate in range(len(queue))
    )
    if any(done > waiting for done, waiting in zip(service, queue, strict=True)):
        return None
    active = [index for index in range(option_count) if counts[index]]
    duration = max(
        Fraction.from_float(float(idle_duration)),
        *(
            Fraction.from_float(max(float(idle_duration), float(options[index].duration)))
            for index in active
        ),
    )
    numerator = sum(
        int(waiting) * int(done)
        for waiting, done in zip(queue, service, strict=True)
    )
    score = Fraction(numerator) / duration
    indices = tuple(
        index
        for index in range(option_count)
        for _ in range(counts[index])
    ) + (None,) * counts[-1]
    return counts[:-1], indices, service, duration, score


def _key(
    duration: Fraction, indices: Sequence[int | None], option_count: int
) -> tuple[Fraction, tuple[int, ...]]:
    return (
        duration,
        tuple(option_count if index is None else index for index in indices),
    )


def select_canonical_count_space_bnb_single_exchange_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    lp_node_budget: int,
    idle_duration: float = 1.0,
) -> CanonicalSingleExchangeSelection:
    """Run the certified base selector and polish a nonexact incumbent."""

    options = tuple(options)
    queue = tuple(int(value) for value in queues)
    base = select_canonical_count_space_bnb_frame(
        options, queue, lanes, lp_node_budget, idle_duration
    )
    if base.score_exact:
        return CanonicalSingleExchangeSelection(
            base_selection=base,
            selection=base,
            polish_triggered=False,
            exchange_attempts=0,
            feasible_exchange_count=0,
            score_improving_exchange_count=0,
            canonical_tie_improving_exchange_count=0,
            incumbent_changed=False,
        )

    option_count = len(options)
    counts = tuple(base.option_counts) + (lanes - sum(base.option_counts),)
    incumbent = _candidate(options, queue, lanes, counts, idle_duration)
    if incumbent is None:
        raise ArithmeticError("base selector returned an infeasible incumbent")
    best = incumbent
    best_key = _key(best[3], best[1], option_count)
    attempts = 0
    feasible = 0
    score_improvements = 0
    tie_improvements = 0

    for source in range(option_count + 1):
        if not counts[source]:
            continue
        for destination in range(option_count + 1):
            if destination == source:
                continue
            attempts += 1
            moved = list(counts)
            moved[source] -= 1
            moved[destination] += 1
            candidate = _candidate(options, queue, lanes, moved, idle_duration)
            if candidate is None:
                continue
            feasible += 1
            candidate_key = _key(candidate[3], candidate[1], option_count)
            if candidate[4] > incumbent[4]:
                score_improvements += 1
            elif candidate[4] == incumbent[4] and candidate_key < _key(
                incumbent[3], incumbent[1], option_count
            ):
                tie_improvements += 1
            if candidate[4] > best[4] or (
                candidate[4] == best[4] and candidate_key < best_key
            ):
                best = candidate
                best_key = candidate_key

    changed = best[1] != base.option_indices
    if not changed:
        polished = base
    else:
        score = float(best[4])
        if score > base.upper_bound + 1e-10:
            raise ArithmeticError("exchange incumbent exceeds the certified upper bound")
        gap = max(0.0, base.upper_bound - score)
        polished = replace(
            base,
            option_counts=best[0],
            option_indices=best[1],
            service=best[2],
            duration=float(best[3]),
            score=score,
            gap=gap,
            epsilon=gap / sum(queue) if sum(queue) else 0.0,
        )
    return CanonicalSingleExchangeSelection(
        base_selection=base,
        selection=polished,
        polish_triggered=True,
        exchange_attempts=attempts,
        feasible_exchange_count=feasible,
        score_improving_exchange_count=score_improvements,
        canonical_tie_improving_exchange_count=tie_improvements,
        incumbent_changed=changed,
    )


def total_work_row(result: CanonicalSingleExchangeSelection) -> dict[str, int]:
    """Return base native work plus conservatively charged exchange attempts."""

    selection = result.base_selection
    return {
        "lp_calls": selection.lp_calls,
        "lp_matrix_coefficients": selection.lp_matrix_coefficients,
        "candidate_evaluations": (
            selection.candidate_evaluations + result.exchange_attempts
        ),
        "residual_fill_option_probes": (
            selection.residual_fill_option_probes + result.exchange_attempts
        ),
        "dual_column_checks": selection.dual_column_checks,
        "branch_splits": selection.branch_splits,
        "canonical_key_bound_evaluations": (
            selection.canonical_key_bound_evaluations
        ),
        "canonical_tie_pruned_regions": selection.canonical_tie_pruned_regions,
        "canonical_tie_incumbent_updates": (
            selection.canonical_tie_incumbent_updates
        ),
        "canonical_tie_singletons_resolved": (
            selection.canonical_tie_singletons_resolved
        ),
    }


__all__ = [
    "CanonicalSingleExchangeSelection",
    "select_canonical_count_space_bnb_single_exchange_frame",
    "total_work_row",
]
