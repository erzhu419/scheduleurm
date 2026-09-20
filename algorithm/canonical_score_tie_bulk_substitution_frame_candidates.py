"""Reference-independent canonical tie polish by exact bulk substitutions.

The base canonical count-space selector is unchanged.  When it has certified
the optimum score but has not closed every canonical-tie region, this polish
checks one fixed, model-derived pass of lower-index substitutions between
options that serve the same total number of jobs.  Each candidate replaces all
copies of one occupied source option.  The best feasible replacement is
accepted only after exact score equality and a strict canonical-key
improvement.

Every structural pair check, including an inactive or infeasible source, is
charged conservatively to candidate evaluations and residual option probes.
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
class CanonicalScoreTieBulkSelection:
    base_selection: CanonicalCountSpaceBnbSelection
    selection: CanonicalCountSpaceBnbSelection
    polish_triggered: bool
    pair_checks: int
    bulk_candidate_attempts: int
    feasible_bulk_candidate_count: int
    accepted_bulk_substitution_count: int
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
        raise ValueError("bulk-substitution counts must cover every lane exactly once")
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


def select_canonical_score_tie_bulk_substitution_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    lp_node_budget: int,
    idle_duration: float = 1.0,
) -> CanonicalScoreTieBulkSelection:
    """Run the base selector and descend through exact-score bulk tie moves."""

    options = tuple(options)
    queue = tuple(int(value) for value in queues)
    base = select_canonical_count_space_bnb_frame(
        options, queue, lanes, lp_node_budget, idle_duration
    )
    triggered = base.score_exact and not base.canonical_exact
    if not triggered:
        return CanonicalScoreTieBulkSelection(
            base_selection=base,
            selection=base,
            polish_triggered=False,
            pair_checks=0,
            bulk_candidate_attempts=0,
            feasible_bulk_candidate_count=0,
            accepted_bulk_substitution_count=0,
            incumbent_changed=False,
        )

    option_count = len(options)
    counts = tuple(base.option_counts) + (lanes - sum(base.option_counts),)
    incumbent = _candidate(options, queue, lanes, counts, idle_duration)
    if incumbent is None:
        raise ArithmeticError("base selector returned an infeasible incumbent")
    structural_pairs = tuple(
        (source, destination)
        for source in range(option_count)
        for destination in range(source)
        if sum(int(value) for value in options[source].requirements)
        == sum(int(value) for value in options[destination].requirements)
    )
    pair_checks = len(structural_pairs)
    attempts = 0
    feasible = 0
    current_counts = incumbent[0] + (lanes - sum(incumbent[0]),)
    best = incumbent
    best_key = _key(best[3], best[1], option_count)
    for source, destination in structural_pairs:
        copies = current_counts[source]
        if not copies:
            continue
        attempts += 1
        moved = list(current_counts)
        moved[source] = 0
        moved[destination] += copies
        candidate = _candidate(options, queue, lanes, moved, idle_duration)
        if candidate is None:
            continue
        feasible += 1
        candidate_key = _key(candidate[3], candidate[1], option_count)
        if candidate[4] == incumbent[4] and candidate_key < best_key:
            best = candidate
            best_key = candidate_key
    incumbent = best
    accepted = int(incumbent[1] != base.option_indices)

    changed = incumbent[1] != base.option_indices
    if not changed:
        polished = base
    else:
        score = float(incumbent[4])
        if score > base.upper_bound + 1e-10:
            raise ArithmeticError("bulk tie incumbent exceeds the certified upper bound")
        gap = max(0.0, base.upper_bound - score)
        polished = replace(
            base,
            option_counts=incumbent[0],
            option_indices=incumbent[1],
            service=incumbent[2],
            duration=float(incumbent[3]),
            score=score,
            gap=gap,
            epsilon=gap / sum(queue) if sum(queue) else 0.0,
        )
    return CanonicalScoreTieBulkSelection(
        base_selection=base,
        selection=polished,
        polish_triggered=True,
        pair_checks=pair_checks,
        bulk_candidate_attempts=attempts,
        feasible_bulk_candidate_count=feasible,
        accepted_bulk_substitution_count=accepted,
        incumbent_changed=changed,
    )


def total_work_row(result: CanonicalScoreTieBulkSelection) -> dict[str, int]:
    """Return base native work plus every structural bulk-pair check."""

    base = result.base_selection
    return {
        "lp_calls": base.lp_calls,
        "lp_matrix_coefficients": base.lp_matrix_coefficients,
        "candidate_evaluations": (
            base.candidate_evaluations + result.pair_checks
        ),
        "residual_fill_option_probes": (
            base.residual_fill_option_probes + result.pair_checks
        ),
        "dual_column_checks": base.dual_column_checks,
        "branch_splits": base.branch_splits,
        "canonical_key_bound_evaluations": base.canonical_key_bound_evaluations,
        "canonical_tie_pruned_regions": base.canonical_tie_pruned_regions,
        "canonical_tie_incumbent_updates": base.canonical_tie_incumbent_updates,
        "canonical_tie_singletons_resolved": base.canonical_tie_singletons_resolved,
    }


__all__ = [
    "CanonicalScoreTieBulkSelection",
    "select_canonical_score_tie_bulk_substitution_frame",
    "total_work_row",
]
