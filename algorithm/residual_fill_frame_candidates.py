"""Budgeted residual-fill seeds with a certified fractional frame bound.

The selector constructs a small family containing at most two non-idle option
types.  For each directed ``(anchor, fill)`` pair and feasible anchor count, it
uses the largest fill count allowed by the residual lanes and visible queue.
Equivalent lane multisets are deduplicated before they consume candidate
budget.  The remaining candidate budget is passed to the existing Cartesian
selector.

A duration-threshold fractional relaxation supplies a second global support
upper bound.  HiGHS is used only to propose dual prices.  Those prices are
converted to exact binary rationals, repaired to dual feasibility, and their
dual objective is rounded toward positive infinity.  The returned certificate
therefore does not rely on the floating-point primal objective being an upper
bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Sequence

from scipy.optimize import linprog

from algorithm.budgeted_frame_candidates import (
    BudgetedFrameSelection,
    select_budgeted_frame,
)


@dataclass(frozen=True)
class FractionalFrameUpperCertificate:
    """A repaired dual certificate for the duration-threshold relaxation."""

    upper_bound: float
    raw_primal_max_diagnostic: float
    threshold_lp_count: int
    total_variables: int
    total_constraint_rows: int
    variable_constraint_cells: int
    objective_terms: int
    repaired_threshold_count: int
    exact_dual_feasible: bool


@dataclass(frozen=True)
class ResidualFillFrameSelection(BudgetedFrameSelection):
    """Selection and explicit work ledger for residual-fill plus legacy search."""

    legacy_budget: int
    legacy_attempts: int
    legacy_score: float
    legacy_upper_bound: float
    pair_score: float
    pair_candidate_attempts: int
    pair_candidate_score_evaluations: int
    pair_local_score_evaluations: int
    pair_generation_probes: int
    zero_fill_probes: int
    duplicate_multiset_probes: int
    pair_family_exhausted: bool
    pair_budget_blocked_unique_candidate: bool
    anchor_capacity_divisions: int
    fill_capacity_divisions: int
    residual_coordinate_evaluations: int
    dedup_key_item_reads: int
    fractional_upper_bound: float
    raw_fractional_primal_max_diagnostic: float
    threshold_lp_count: int
    lp_total_variables: int
    lp_total_constraint_rows: int
    lp_variable_constraint_cells: int
    lp_objective_terms: int
    lp_repaired_threshold_count: int
    lp_exact_dual_feasible: bool
    charged_logical_work: int


def _upward_float(value: Fraction) -> float:
    """Convert a nonnegative rational to a float without rounding downward."""

    result = float(value)
    if math.isinf(result):
        return result
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _catalogue(
    options: Sequence[Any], queues: Sequence[int], idle_duration: float
) -> tuple[
    tuple[Any, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[float, ...],
    tuple[int, ...],
]:
    options = tuple(options)
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
    return options, queue, tuple(requirements), tuple(durations), tuple(weights)


def certified_fractional_frame_upper(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    idle_duration: float = 1.0,
) -> FractionalFrameUpperCertificate:
    """Bound every integer frame through repaired fractional dual certificates.

    For each possible non-idle frame duration ``tau``, the relaxation permits
    only options no slower than ``tau`` and maximizes weighted service subject
    to the lane and queue constraints.  Every feasible integer frame appears in
    the relaxation indexed by its duration.
    """

    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    options, queue, requirements, durations, weights = _catalogue(
        options, queues, idle_duration
    )
    if not options:
        return FractionalFrameUpperCertificate(
            upper_bound=0.0,
            raw_primal_max_diagnostic=0.0,
            threshold_lp_count=0,
            total_variables=0,
            total_constraint_rows=0,
            variable_constraint_cells=0,
            objective_terms=0,
            repaired_threshold_count=0,
            exact_dual_feasible=True,
        )

    thresholds = tuple(sorted(set(durations)))
    exact_global_upper = Fraction(0)
    raw_primal_max = 0.0
    total_variables = 0
    total_constraint_rows = 0
    variable_constraint_cells = 0
    objective_terms = 0
    repaired_threshold_count = 0
    exact_dual_feasible = True
    rhs = (lanes, *queue)

    for threshold in thresholds:
        allowed = tuple(
            index
            for index, duration in enumerate(durations)
            if duration <= threshold
        )
        variable_count = len(allowed)
        constraint_count = len(queue) + 1
        total_variables += variable_count
        total_constraint_rows += constraint_count
        variable_constraint_cells += variable_count * constraint_count
        objective_terms += variable_count

        columns = tuple(
            (1, *requirements[index]) for index in allowed
        )
        exact_objective = tuple(
            Fraction(weights[index], 1) / Fraction.from_float(threshold)
            for index in allowed
        )
        matrix = [
            [float(column[row]) for column in columns]
            for row in range(constraint_count)
        ]
        result = linprog(
            [-float(value) for value in exact_objective],
            A_ub=matrix,
            b_ub=[float(value) for value in rhs],
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(
                f"fractional frame relaxation failed at duration {threshold}: "
                f"{result.message}"
            )
        raw_primal_max = max(raw_primal_max, -float(result.fun))

        proposed = tuple(
            Fraction.from_float(max(0.0, -float(value)))
            for value in result.ineqlin.marginals
        )
        deficits = tuple(
            objective
            - sum(
                Fraction(coefficient, 1) * proposed[row]
                for row, coefficient in enumerate(column)
            )
            for column, objective in zip(columns, exact_objective, strict=True)
        )
        repair = max((Fraction(0), *deficits))
        dual = list(proposed)
        if repair > 0:
            # Every option consumes one lane, so adding the largest exact column
            # deficit to the lane price repairs all dual inequalities at once.
            dual[0] += repair
            repaired_threshold_count += 1

        threshold_feasible = all(
            sum(
                Fraction(coefficient, 1) * dual[row]
                for row, coefficient in enumerate(column)
            )
            >= objective
            for column, objective in zip(columns, exact_objective, strict=True)
        ) and all(value >= 0 for value in dual)
        exact_dual_feasible &= threshold_feasible
        if not threshold_feasible:
            raise RuntimeError("exact dual repair failed")
        exact_upper = sum(
            Fraction(value, 1) * price
            for value, price in zip(rhs, dual, strict=True)
        )
        exact_global_upper = max(exact_global_upper, exact_upper)

    return FractionalFrameUpperCertificate(
        upper_bound=_upward_float(exact_global_upper),
        raw_primal_max_diagnostic=raw_primal_max,
        threshold_lp_count=len(thresholds),
        total_variables=total_variables,
        total_constraint_rows=total_constraint_rows,
        variable_constraint_cells=variable_constraint_cells,
        objective_terms=objective_terms,
        repaired_threshold_count=repaired_threshold_count,
        exact_dual_feasible=exact_dual_feasible,
    )


def select_residual_fill_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
) -> ResidualFillFrameSelection:
    """Select from residual-fill seeds and the remaining-budget legacy search.

    The fixed generation order is anchor option index, fill option index, then
    ascending anchor copy count.  A pair candidate pays only after its lane
    multiset is known to be new.  Generation, zero-fill, duplicate, coordinate,
    and LP work remain explicit in ``charged_logical_work`` even though they do
    not consume the candidate-attempt budget.
    """

    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    options, queue, requirements, durations, weights = _catalogue(
        options, queues, idle_duration
    )
    seed_attempts = len(options) + 1
    if int(budget) != budget or budget < seed_attempts:
        raise ValueError(
            f"budget must be an integer covering {seed_attempts} legacy seed attempts"
        )
    budget = int(budget)
    option_count = len(options)
    pair_limit = budget - seed_attempts

    def copies_allowed(
        visible: tuple[int, ...], counts: tuple[int, ...], limit: int
    ) -> tuple[int, int]:
        divisions = sum(count > 0 for count in counts)
        copies = min(
            limit,
            min(
                (waiting // count for waiting, count in zip(visible, counts) if count),
                default=limit,
            ),
        )
        return copies, divisions

    pair_best_indices: tuple[int | None, ...] = (None,) * lanes
    pair_best_service = (0,) * len(queue)
    pair_best_duration = float(idle_duration)
    pair_best_score = 0.0
    seen: set[tuple[int, ...]] = set()
    pair_candidate_attempts = 0
    pair_generation_probes = 0
    zero_fill_probes = 0
    duplicate_multiset_probes = 0
    anchor_capacity_divisions = 0
    fill_capacity_divisions = 0
    residual_coordinate_evaluations = 0
    dedup_key_item_reads = 0
    pair_budget_blocked = False

    anchor_limits: list[int] = []
    for counts in requirements:
        copies, divisions = copies_allowed(queue, counts, max(0, lanes - 1))
        anchor_limits.append(copies)
        anchor_capacity_divisions += divisions

    def action_key(
        score: float,
        duration: float,
        indices: tuple[int | None, ...],
    ) -> tuple[float, float, tuple[int, ...]]:
        return (
            -score,
            duration,
            tuple(
                option_count if index is None else index for index in indices
            ),
        )

    stop = False
    for anchor in range(option_count):
        if stop:
            break
        for fill in range(option_count):
            if anchor == fill:
                continue
            for anchor_copies in range(1, anchor_limits[anchor] + 1):
                pair_generation_probes += 1
                residual_coordinate_evaluations += len(queue)
                residual = tuple(
                    waiting - anchor_copies * count
                    for waiting, count in zip(queue, requirements[anchor], strict=True)
                )
                fill_copies, divisions = copies_allowed(
                    residual, requirements[fill], lanes - anchor_copies
                )
                fill_capacity_divisions += divisions
                if fill_copies < 1:
                    zero_fill_probes += 1
                    continue

                counts_by_option = [0] * option_count
                counts_by_option[anchor] = anchor_copies
                counts_by_option[fill] = fill_copies
                multiset = tuple(counts_by_option)
                dedup_key_item_reads += option_count
                if multiset in seen:
                    duplicate_multiset_probes += 1
                    continue
                if pair_candidate_attempts >= pair_limit:
                    pair_budget_blocked = True
                    stop = True
                    break
                seen.add(multiset)
                pair_candidate_attempts += 1
                indices = tuple(
                    index
                    for index, copies in enumerate(multiset)
                    for _ in range(copies)
                ) + (None,) * (lanes - sum(multiset))
                service = tuple(
                    anchor_copies * left + fill_copies * right
                    for left, right in zip(
                        requirements[anchor], requirements[fill], strict=True
                    )
                )
                duration = max(durations[anchor], durations[fill])
                numerator = (
                    anchor_copies * weights[anchor]
                    + fill_copies * weights[fill]
                )
                score = numerator / duration
                if action_key(score, duration, indices) < action_key(
                    pair_best_score,
                    pair_best_duration,
                    pair_best_indices,
                ):
                    pair_best_indices = indices
                    pair_best_service = service
                    pair_best_duration = duration
                    pair_best_score = score

    legacy_budget = budget - pair_candidate_attempts
    legacy = select_budgeted_frame(
        options, queue, lanes, legacy_budget, idle_duration=idle_duration
    )
    if action_key(
        pair_best_score, pair_best_duration, pair_best_indices
    ) < action_key(legacy.score, legacy.duration, legacy.option_indices):
        best_indices = pair_best_indices
        best_service = pair_best_service
        best_duration = pair_best_duration
        best_score = pair_best_score
    else:
        best_indices = legacy.option_indices
        best_service = legacy.service
        best_duration = legacy.duration
        best_score = legacy.score

    fractional = certified_fractional_frame_upper(
        options, queue, lanes, idle_duration=idle_duration
    )
    upper = max(
        best_score,
        min(legacy.upper_bound, fractional.upper_bound),
    )
    gap = max(0.0, upper - best_score)
    queue_norm = sum(queue)
    charged_logical_work = sum((
        legacy.attempts,
        legacy.local_score_evaluations,
        legacy.bound_option_probes,
        legacy.bound_threshold_evaluations,
        option_count,
        pair_candidate_attempts,
        pair_generation_probes,
        anchor_capacity_divisions,
        fill_capacity_divisions,
        residual_coordinate_evaluations,
        dedup_key_item_reads,
        fractional.threshold_lp_count,
        fractional.total_variables,
        fractional.total_constraint_rows,
        fractional.variable_constraint_cells,
        fractional.objective_terms,
        fractional.repaired_threshold_count,
    ))
    return ResidualFillFrameSelection(
        option_indices=best_indices,
        service=best_service,
        duration=best_duration,
        score=best_score,
        upper_bound=upper,
        gap=gap,
        epsilon=gap / queue_norm if queue_norm else 0.0,
        attempts=legacy.attempts + pair_candidate_attempts,
        seed_attempts=legacy.seed_attempts,
        feasible_candidate_count=(
            legacy.feasible_candidate_count + pair_candidate_attempts
        ),
        local_score_evaluations=legacy.local_score_evaluations,
        bound_option_probes=legacy.bound_option_probes,
        bound_threshold_evaluations=legacy.bound_threshold_evaluations,
        frontier_peak=legacy.frontier_peak,
        exact=gap == 0.0,
        legacy_budget=legacy_budget,
        legacy_attempts=legacy.attempts,
        legacy_score=legacy.score,
        legacy_upper_bound=legacy.upper_bound,
        pair_score=pair_best_score,
        pair_candidate_attempts=pair_candidate_attempts,
        pair_candidate_score_evaluations=pair_candidate_attempts,
        pair_local_score_evaluations=option_count,
        pair_generation_probes=pair_generation_probes,
        zero_fill_probes=zero_fill_probes,
        duplicate_multiset_probes=duplicate_multiset_probes,
        pair_family_exhausted=not pair_budget_blocked,
        pair_budget_blocked_unique_candidate=pair_budget_blocked,
        anchor_capacity_divisions=anchor_capacity_divisions,
        fill_capacity_divisions=fill_capacity_divisions,
        residual_coordinate_evaluations=residual_coordinate_evaluations,
        dedup_key_item_reads=dedup_key_item_reads,
        fractional_upper_bound=fractional.upper_bound,
        raw_fractional_primal_max_diagnostic=(
            fractional.raw_primal_max_diagnostic
        ),
        threshold_lp_count=fractional.threshold_lp_count,
        lp_total_variables=fractional.total_variables,
        lp_total_constraint_rows=fractional.total_constraint_rows,
        lp_variable_constraint_cells=fractional.variable_constraint_cells,
        lp_objective_terms=fractional.objective_terms,
        lp_repaired_threshold_count=fractional.repaired_threshold_count,
        lp_exact_dual_feasible=fractional.exact_dual_feasible,
        charged_logical_work=charged_logical_work,
    )
