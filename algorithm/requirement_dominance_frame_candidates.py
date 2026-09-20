"""Budgeted frame search after exact-requirement duration dominance reduction.

For two local options with identical job requirements, the option with the
strictly longer effective duration ``max(idle_duration, duration)`` cannot
improve MaxWeight support.  Replacing it in any frame preserves feasibility and
service while weakly reducing the frame duration.  This module removes only
that proved redundancy, runs the frozen v4 selector on the retained catalogue,
and maps its lane choices back to indices in the original catalogue.

The paid candidate budget belongs entirely to the reduced v4 search.  Catalogue
validation and reduction are deterministic local work and are reported in
separate counters.  Neither operation uses arrivals or any state beyond the
currently visible queue passed to the selector.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from algorithm.budgeted_frame_candidates import (
    BudgetedFrameSelection,
    select_budgeted_frame,
)


@dataclass(frozen=True)
class RequirementDominanceReduction:
    """A validated catalogue with strictly slower requirement duplicates removed."""

    options: tuple[Any, ...]
    original_option_count: int
    retained_option_count: int
    removed_option_count: int
    retained_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]
    reduction_option_probes: int
    reduction_requirement_coordinate_reads: int
    reduction_duration_comparisons: int


@dataclass(frozen=True)
class RequirementDominanceFrameSelection(BudgetedFrameSelection):
    """A v4 selection whose option indices refer to the original catalogue."""

    original_option_count: int
    retained_option_count: int
    removed_option_count: int
    retained_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]
    reduction_option_probes: int
    reduction_requirement_coordinate_reads: int
    reduction_duration_comparisons: int


def reduce_requirement_dominated_options(
    options: Sequence[Any],
    idle_duration: float = 1.0,
) -> RequirementDominanceReduction:
    """Validate ``options`` and retain every shortest option per requirement tuple.

    Durations are compared after applying the same idle-duration floor as the
    frame selector.  Equality is exact: all options tied at the minimum
    effective duration remain in their original order.  Validation precedes
    reduction, so an invalid option raises even when a valid option would have
    dominated it.
    """
    catalogue = tuple(options)
    idle_duration = float(idle_duration)
    if not math.isfinite(idle_duration) or idle_duration <= 0.0:
        raise ValueError("idle_duration must be positive and finite")

    requirements: list[tuple[int, ...]] = []
    effective_durations: list[float] = []
    coordinate_reads = 0
    dimension: int | None = None
    for option in catalogue:
        raw_counts = tuple(option.requirements)
        if dimension is None:
            dimension = len(raw_counts)
        if len(raw_counts) != dimension:
            raise ValueError("all option requirements must have the same dimension")
        counts: list[int] = []
        for value in raw_counts:
            coordinate_reads += 1
            if int(value) != value or value < 0:
                raise ValueError("option requirements must contain nonnegative integers")
            counts.append(int(value))
        requirement = tuple(counts)
        if not any(requirement):
            raise ValueError("a non-idle option must require at least one job")
        duration = float(option.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("option durations must be positive and finite")
        requirements.append(requirement)
        effective_durations.append(max(idle_duration, duration))

    minimum_duration_by_requirement: dict[tuple[int, ...], float] = {}
    duration_comparisons = 0
    for requirement, duration in zip(requirements, effective_durations, strict=True):
        if requirement not in minimum_duration_by_requirement:
            minimum_duration_by_requirement[requirement] = duration
        else:
            duration_comparisons += 1
            if duration < minimum_duration_by_requirement[requirement]:
                minimum_duration_by_requirement[requirement] = duration

    retained: list[int] = []
    removed: list[int] = []
    for index, (requirement, duration) in enumerate(
        zip(requirements, effective_durations, strict=True)
    ):
        duration_comparisons += 1
        target = retained if duration == minimum_duration_by_requirement[requirement] else removed
        target.append(index)

    retained_indices = tuple(retained)
    removed_indices = tuple(removed)
    return RequirementDominanceReduction(
        options=tuple(catalogue[index] for index in retained_indices),
        original_option_count=len(catalogue),
        retained_option_count=len(retained_indices),
        removed_option_count=len(removed_indices),
        retained_original_indices=retained_indices,
        removed_original_indices=removed_indices,
        reduction_option_probes=len(catalogue),
        reduction_requirement_coordinate_reads=coordinate_reads,
        reduction_duration_comparisons=duration_comparisons,
    )


def select_requirement_dominance_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
) -> RequirementDominanceFrameSelection:
    """Run v4 on the support-equivalent reduced catalogue under ``budget``.

    ``budget`` counts only paid attempts made by the reduced v4 selector.  The
    returned ``option_indices`` are mapped back to the caller's original option
    indices so the unchanged frame simulator executes the correct completion
    offsets and action identifiers.
    """
    catalogue = tuple(options)
    queue_values = tuple(queues)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    if int(budget) != budget:
        raise ValueError("budget must be an integer")
    if any(int(value) != value or value < 0 for value in queue_values):
        raise ValueError("queues must contain nonnegative integer job counts")
    queue = tuple(int(value) for value in queue_values)

    reduction = reduce_requirement_dominated_options(catalogue, idle_duration)
    # The public reducer validates every original option before filtering.  This
    # second catalogue-wide check ties every original requirement vector to the
    # visible queue dimension; calling v4 only on survivors cannot bypass it.
    for option in catalogue:
        if len(tuple(option.requirements)) != len(queue):
            raise ValueError(
                "option requirements must match the nonnegative integer queue coordinates"
            )

    selected = select_budgeted_frame(
        reduction.options,
        queue,
        int(lanes),
        int(budget),
        idle_duration,
    )
    mapped_indices = tuple(
        None if index is None else reduction.retained_original_indices[index]
        for index in selected.option_indices
    )
    return RequirementDominanceFrameSelection(
        option_indices=mapped_indices,
        service=selected.service,
        duration=selected.duration,
        score=selected.score,
        upper_bound=selected.upper_bound,
        gap=selected.gap,
        epsilon=selected.epsilon,
        attempts=selected.attempts,
        seed_attempts=selected.seed_attempts,
        feasible_candidate_count=selected.feasible_candidate_count,
        local_score_evaluations=selected.local_score_evaluations,
        bound_option_probes=selected.bound_option_probes,
        bound_threshold_evaluations=selected.bound_threshold_evaluations,
        frontier_peak=selected.frontier_peak,
        exact=selected.exact,
        original_option_count=reduction.original_option_count,
        retained_option_count=reduction.retained_option_count,
        removed_option_count=reduction.removed_option_count,
        retained_original_indices=reduction.retained_original_indices,
        removed_original_indices=reduction.removed_original_indices,
        reduction_option_probes=reduction.reduction_option_probes,
        reduction_requirement_coordinate_reads=(
            reduction.reduction_requirement_coordinate_reads
        ),
        reduction_duration_comparisons=reduction.reduction_duration_comparisons,
    )
