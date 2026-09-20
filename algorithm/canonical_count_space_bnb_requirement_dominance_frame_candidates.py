"""Canonical count-space B&B on a requirement-duration quotient catalogue.

The public requirement-dominance reducer removes a strictly slower option when
another local option has the same job requirements.  This module runs the
score-exact tie continuation on that reduced catalogue, then lifts both
checkpoints back to the caller's original option indices.

The score certificate lifts because every full-catalogue action containing a
removed option has a reduced-catalogue replacement with identical service and
strictly smaller effective duration.  The canonical certificate additionally
requires that a replacement have a smaller original option index.  The latter
condition makes replacement improve either duration or the established
``(duration, sorted option indices with idle last)`` tie-break.  Catalogues
without that proof are rejected rather than being given a false full-catalogue
canonical certificate.

All B&B work counters remain those of the reduced search.  Reduction and lift
validation are deterministic catalogue preprocessing and have separate
counters.  The wrapper reads no state other than the supplied catalogue,
visible queue, lane count, budgets, and idle-duration floor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    CanonicalCountSpaceBnbSelection,
    select_canonical_count_space_bnb_score_exact_tie_continuation_frame,
)
from algorithm.requirement_dominance_frame_candidates import (
    RequirementDominanceReduction,
    reduce_requirement_dominated_options,
)


@dataclass(frozen=True)
class RequirementDominanceQuotientMetadata:
    """Catalogue reduction and separately paid preprocessing work."""

    original_option_count: int
    retained_option_count: int
    removed_option_count: int
    retained_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]
    idle_duration: float
    reduction_option_probes: int
    reduction_requirement_coordinate_reads: int
    reduction_duration_comparisons: int
    lift_option_probes: int
    lift_requirement_coordinate_reads: int
    lift_requirement_vector_comparisons: int
    lift_duration_comparisons: int
    lift_index_comparisons: int


@dataclass(frozen=True)
class FullCatalogueDominanceLiftProof:
    """Static proof that reduced certificates apply to the full catalogue."""

    removed_to_dominator_original_indices: tuple[tuple[int, int], ...]
    retained_index_map_strictly_increasing: bool
    replacement_requirements_equal: bool
    replacement_effective_duration_strictly_smaller: bool
    replacement_original_index_strictly_smaller: bool
    full_catalogue_score_upper_bound_valid: bool
    full_catalogue_canonical_order_valid: bool


@dataclass(frozen=True)
class FullCatalogueDominanceCertificateLift:
    """Static lift proof plus the certificates reached on one visible queue."""

    proof: FullCatalogueDominanceLiftProof
    primary_score_exact_lifted: bool
    primary_canonical_exact_lifted: bool
    selection_score_exact_lifted: bool
    selection_canonical_exact_lifted: bool


@dataclass(frozen=True)
class CanonicalRequirementDominanceTieContinuationSelection:
    """Mapped conditional-continuation result with quotient proof metadata."""

    primary_selection: CanonicalCountSpaceBnbSelection
    selection: CanonicalCountSpaceBnbSelection
    continuation_triggered: bool
    initial_budget: int
    continuation_budget: int
    reduction: RequirementDominanceQuotientMetadata
    certificate_lift: FullCatalogueDominanceCertificateLift


@dataclass(frozen=True)
class PreparedCanonicalRequirementDominanceCatalogue:
    """Queue-independent quotient and proof, prepared once for frame reuse."""

    reduced_options: tuple[Any, ...]
    reduction: RequirementDominanceQuotientMetadata
    certificate_lift_proof: FullCatalogueDominanceLiftProof


@dataclass(frozen=True)
class _LiftValidation:
    pairs: tuple[tuple[int, int], ...]
    option_probes: int
    requirement_coordinate_reads: int
    requirement_vector_comparisons: int
    duration_comparisons: int
    index_comparisons: int


def _validate_full_catalogue_canonical_lift(
    catalogue: tuple[Any, ...],
    reduction: RequirementDominanceReduction,
    idle_duration: float,
) -> _LiftValidation:
    """Find an earlier strict-duration dominator for every removed option."""

    requirements: list[tuple[int, ...]] = []
    durations: list[float] = []
    coordinate_reads = 0
    for option in catalogue:
        requirement = tuple(int(value) for value in option.requirements)
        coordinate_reads += len(requirement)
        requirements.append(requirement)
        durations.append(max(idle_duration, float(option.duration)))

    retained = reduction.retained_original_indices
    pairs: list[tuple[int, int]] = []
    requirement_comparisons = 0
    duration_comparisons = 0
    index_comparisons = 0
    for removed_index in reduction.removed_original_indices:
        eligible: list[int] = []
        for retained_index in retained:
            requirement_comparisons += 1
            if requirements[retained_index] != requirements[removed_index]:
                continue
            duration_comparisons += 1
            if durations[retained_index] >= durations[removed_index]:
                continue
            index_comparisons += 1
            if retained_index < removed_index:
                eligible.append(retained_index)
        if not eligible:
            raise ValueError(
                "full-catalogue canonical lifting requires every removed option "
                "to have an earlier retained strict-duration dominator"
            )
        pairs.append((removed_index, min(eligible)))

    if any(left >= right for left, right in zip(retained, retained[1:])):
        raise AssertionError("the requirement-dominance reducer changed option order")
    return _LiftValidation(
        pairs=tuple(pairs),
        option_probes=len(catalogue),
        requirement_coordinate_reads=coordinate_reads,
        requirement_vector_comparisons=requirement_comparisons,
        duration_comparisons=duration_comparisons,
        index_comparisons=index_comparisons,
    )


def _map_selection_to_full_catalogue(
    selection: CanonicalCountSpaceBnbSelection,
    reduction: RequirementDominanceQuotientMetadata,
) -> CanonicalCountSpaceBnbSelection:
    """Expand counts and lane indices without changing reduced-search work."""

    retained = reduction.retained_original_indices
    if len(selection.option_counts) != len(retained):
        raise AssertionError("reduced B&B returned an unexpected count-vector width")
    full_counts = [0] * reduction.original_option_count
    for reduced_index, count in enumerate(selection.option_counts):
        full_counts[retained[reduced_index]] = count
    try:
        full_indices = tuple(
            None if index is None else retained[index]
            for index in selection.option_indices
        )
    except IndexError as error:
        raise AssertionError("reduced B&B returned an invalid option index") from error
    return replace(
        selection,
        option_counts=tuple(full_counts),
        option_indices=full_indices,
    )


def _metadata(
    reduction: RequirementDominanceReduction,
    lift: _LiftValidation,
    idle_duration: float,
) -> RequirementDominanceQuotientMetadata:
    return RequirementDominanceQuotientMetadata(
        original_option_count=reduction.original_option_count,
        retained_option_count=reduction.retained_option_count,
        removed_option_count=reduction.removed_option_count,
        retained_original_indices=reduction.retained_original_indices,
        removed_original_indices=reduction.removed_original_indices,
        idle_duration=idle_duration,
        reduction_option_probes=reduction.reduction_option_probes,
        reduction_requirement_coordinate_reads=(
            reduction.reduction_requirement_coordinate_reads
        ),
        reduction_duration_comparisons=reduction.reduction_duration_comparisons,
        lift_option_probes=lift.option_probes,
        lift_requirement_coordinate_reads=lift.requirement_coordinate_reads,
        lift_requirement_vector_comparisons=lift.requirement_vector_comparisons,
        lift_duration_comparisons=lift.duration_comparisons,
        lift_index_comparisons=lift.index_comparisons,
    )


def prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
    options: Sequence[Any],
    idle_duration: float = 1.0,
) -> PreparedCanonicalRequirementDominanceCatalogue:
    """Prepare the queue-independent quotient and full-catalogue lift once."""

    catalogue = tuple(options)
    idle_duration = float(idle_duration)
    if not math.isfinite(idle_duration) or idle_duration <= 0.0:
        raise ValueError("idle_duration must be positive and finite")
    reduction = reduce_requirement_dominated_options(catalogue, idle_duration)
    lift = _validate_full_catalogue_canonical_lift(
        catalogue, reduction, idle_duration
    )
    metadata = _metadata(reduction, lift, idle_duration)
    proof = FullCatalogueDominanceLiftProof(
        removed_to_dominator_original_indices=lift.pairs,
        retained_index_map_strictly_increasing=True,
        replacement_requirements_equal=True,
        replacement_effective_duration_strictly_smaller=True,
        replacement_original_index_strictly_smaller=True,
        full_catalogue_score_upper_bound_valid=True,
        full_catalogue_canonical_order_valid=True,
    )
    return PreparedCanonicalRequirementDominanceCatalogue(
        reduced_options=reduction.options,
        reduction=metadata,
        certificate_lift_proof=proof,
    )


def select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
    prepared: PreparedCanonicalRequirementDominanceCatalogue,
    queues: Sequence[int],
    lanes: int,
    initial_budget: int,
    continuation_budget: int,
) -> CanonicalRequirementDominanceTieContinuationSelection:
    """Run one visible-queue selection on a previously prepared quotient.

    ``initial_budget`` and ``continuation_budget`` count LP calls in the reduced
    catalogue only.  The final and primary selections use original catalogue
    indices and original-width count vectors; every other field is the actual
    reduced-search value.  No catalogue reduction or lift scan occurs here.
    """

    reduced = select_canonical_count_space_bnb_score_exact_tie_continuation_frame(
        prepared.reduced_options,
        queues,
        lanes,
        initial_budget,
        continuation_budget,
        prepared.reduction.idle_duration,
    )
    mapped_primary = _map_selection_to_full_catalogue(
        reduced.primary_selection, prepared.reduction
    )
    mapped_selection = (
        mapped_primary
        if reduced.selection is reduced.primary_selection
        else _map_selection_to_full_catalogue(
            reduced.selection, prepared.reduction
        )
    )
    proof = FullCatalogueDominanceCertificateLift(
        proof=prepared.certificate_lift_proof,
        primary_score_exact_lifted=mapped_primary.score_exact,
        primary_canonical_exact_lifted=mapped_primary.canonical_exact,
        selection_score_exact_lifted=mapped_selection.score_exact,
        selection_canonical_exact_lifted=mapped_selection.canonical_exact,
    )
    return CanonicalRequirementDominanceTieContinuationSelection(
        primary_selection=mapped_primary,
        selection=mapped_selection,
        continuation_triggered=reduced.continuation_triggered,
        initial_budget=reduced.initial_budget,
        continuation_budget=reduced.continuation_budget,
        reduction=prepared.reduction,
        certificate_lift=proof,
    )


def select_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    initial_budget: int,
    continuation_budget: int,
    idle_duration: float = 1.0,
) -> CanonicalRequirementDominanceTieContinuationSelection:
    """Convenience call that prepares once and selects one visible queue."""

    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        options, idle_duration
    )
    return select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        prepared,
        queues,
        lanes,
        initial_budget,
        continuation_budget,
    )


__all__ = [
    "CanonicalRequirementDominanceTieContinuationSelection",
    "FullCatalogueDominanceCertificateLift",
    "FullCatalogueDominanceLiftProof",
    "PreparedCanonicalRequirementDominanceCatalogue",
    "RequirementDominanceQuotientMetadata",
    "prepare_canonical_count_space_bnb_requirement_dominance_catalogue",
    "select_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame",
    "select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame",
]
