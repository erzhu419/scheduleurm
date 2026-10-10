"""Resumable integer-capacity LP search with cumulative work accounting.

The search body retains the frozen integer-capacity algorithm's ordering and
certificate rules. Suspending at a budget takes a read-only snapshot; resuming
continues its live frontier without repeating roots or catalogue preprocessing.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import heapq
import math
from typing import Any, Generator, Sequence

from scipy.optimize import linprog

from algorithm.canonical_count_space_bnb_frame_candidates import (
    CanonicalCountSpaceBnbSelection, _Candidate, _Region, _dual_upper, _float_up,
)
from algorithm.requirement_dominance_frame_candidates import reduce_requirement_dominated_options
from algorithm.threshold_canonical_requirement_dominance_frame_candidates import ThresholdCatalogueWork
from algorithm.fractional_shared_threshold_canonical_requirement_dominance_frame_candidates import (
    _branch_position,
)


from algorithm.integer_capacity_shared_threshold_frame_candidates import (
    CapacityPropagationWork, IntegerCapacityThresholdSelection,
    _integer_capacity_upper_bounds,
)


class IntegerCapacitySearch:
    """Advance one search through nondecreasing total LP budgets.

    Construction records inputs. The first ``advance_to`` performs catalogue
    preprocessing and paid duration roots; subsequent calls continue the same
    frontier. Returned immutable dataclasses remain valid after continuation.
    """

    def __init__(
        self, options: Sequence[Any], queues: Sequence[int], lanes: int,
        idle_duration: float = 1.0,
    ) -> None:
        self._options = tuple(options)
        self._queues = tuple(queues)
        self._lanes = lanes
        self._idle_duration = idle_duration
        self._search: Generator[IntegerCapacityThresholdSelection, int, None] | None = None
        self._budget: int | None = None
        self._snapshot: IntegerCapacityThresholdSelection | None = None

    def advance_to(self, total_lp_budget: int) -> IntegerCapacityThresholdSelection:
        """Return the frozen selector's exact snapshot at this cumulative cap."""
        if int(total_lp_budget) != total_lp_budget or total_lp_budget < 0:
            raise ValueError("LP budget must be a nonnegative integer")
        budget = int(total_lp_budget)
        if self._budget is not None and budget < self._budget:
            raise ValueError("LP budgets must be nondecreasing")
        if budget == self._budget:
            assert self._snapshot is not None
            return self._snapshot
        if self._search is None:
            search = _search_snapshots(
                self._options, self._queues, self._lanes, budget, self._idle_duration)
            snapshot = next(search)
            self._search = search
        else:
            snapshot = self._search.send(budget)
        self._budget = budget
        self._snapshot = snapshot
        return snapshot


def _search_snapshots(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    lp_node_budget: int,
    idle_duration: float = 1.0,
) -> Generator[IntegerCapacityThresholdSelection, int, None]:
    """Return an original action and conservative full-catalogue certificates.

    The total LP budget covers every group-minimum duration root, including
    failed solver calls. Unsolved children inherit their parent's valid bound;
    score-equal regions remain live until their original-index canonical key
    cannot improve the shared incumbent.
    """
    options, queue = tuple(options), tuple(queues)
    reduction = reduce_requirement_dominated_options(options, idle_duration)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    if int(lp_node_budget) != lp_node_budget or lp_node_budget < 0:
        raise ValueError("lp_node_budget must be a nonnegative integer")
    budget = int(lp_node_budget)
    if any(int(value) != value or value < 0 for value in queue):
        raise ValueError("queues must contain nonnegative integer job counts")
    queue = tuple(int(value) for value in queue)
    if options and len(options[0].requirements) != len(queue):
        raise ValueError("option requirements must match queue coordinates")
    option_count, dimension = len(options), len(queue)
    idle_fraction = Fraction.from_float(float(idle_duration))
    original_durations = tuple(Fraction.from_float(max(float(idle_duration), float(option.duration)))
                               for option in options)
    grouped: dict[tuple[int, ...], list[int]] = {}
    for index, option in enumerate(options):
        grouped.setdefault(tuple(int(value) for value in option.requirements), []).append(index)
    groups = tuple(tuple(members) for members in grouped.values())
    requirements = tuple(grouped)
    durations = tuple(min(original_durations[index] for index in members) for members in groups)
    weights = tuple(sum(waiting * amount for waiting, amount in zip(queue, requirement, strict=True))
                    for requirement in requirements)
    thresholds = tuple(sorted(set(durations)))
    root_count = len(thresholds)
    if budget < root_count:
        raise ValueError(f"LP budget must cover {root_count} duration thresholds")
    threshold_positions = {threshold: index for index, threshold in enumerate(thresholds)}
    representative_probes = 0
    representatives: list[dict[int, int]] = []
    for threshold in thresholds:
        current = {}
        for group, members in enumerate(groups):
            if durations[group] > threshold:
                continue
            for index in members:
                representative_probes += 1
                if original_durations[index] <= threshold:
                    current[group] = index
                    break
        representatives.append(current)
    lift_probes = 0
    best = _Candidate((0,) * option_count, (None,) * lanes, (0,) * dimension,
                      idle_fraction, Fraction(0))
    candidate_evaluations = 1
    residual_fill_option_probes = 0
    canonical_tie_incumbent_updates = 0

    def candidate_key(candidate: _Candidate) -> tuple[Fraction, tuple[int, ...]]:
        return candidate.duration, tuple(option_count if index is None else index
                                         for index in candidate.indices)

    def consider_counts(eligible: tuple[int, ...], local_counts: Sequence[int]) -> None:
        nonlocal best, candidate_evaluations, canonical_tie_incumbent_updates, lift_probes
        candidate_evaluations += 1
        if any(value < 0 for value in local_counts) or sum(local_counts) > lanes:
            raise ArithmeticError("candidate count construction is invalid")
        service = tuple(sum(local_counts[position] * requirements[group][coordinate]
                            for position, group in enumerate(eligible))
                        for coordinate in range(dimension))
        if any(done > waiting for done, waiting in zip(service, queue, strict=True)):
            raise ArithmeticError("candidate exceeds the visible queue")
        used = tuple((group, int(count)) for group, count in zip(eligible, local_counts, strict=True)
                     if count)
        duration = max((idle_fraction, *(durations[group] for group, _ in used)))
        counts = [0] * option_count
        if used:
            mapping = representatives[threshold_positions[duration]]
            for group, count in used:
                lift_probes += 1
                counts[mapping[group]] = count
        numerator = sum(waiting * done for waiting, done in zip(queue, service, strict=True))
        score = Fraction(numerator) / duration
        expanded = tuple(index for index, count in enumerate(counts) for _ in range(count))
        candidate = _Candidate(tuple(counts), expanded + (None,) * (lanes - len(expanded)),
                               service, duration, score)
        if candidate.score > best.score or (
            candidate.score == best.score and candidate_key(candidate) < candidate_key(best)
        ):
            canonical_tie_incumbent_updates += int(candidate.score == best.score)
            best = candidate

    successful_lp_calls = 0
    dual_certified_lp_calls = 0
    safe_fallback_lp_calls = 0
    dual_repair_count = 0
    maximum_dual_repair = Fraction(0)
    lp_matrix_coefficients = 0
    dual_column_checks = 0
    lp_calls = 0

    def lower_feasible(region: _Region) -> bool:
        if any(left > right for left, right in zip(region.lower, region.upper, strict=True)):
            return False
        if sum(region.lower) > lanes:
            return False
        return all(
            sum(
                region.lower[position] * requirements[index][coordinate]
                for position, index in enumerate(region.eligible)
            ) <= queue[coordinate]
            for coordinate in range(dimension)
        )

    capacity_region_visits = 0
    capacity_requirement_coordinate_reads = 0
    capacity_tightened_bounds = 0
    capacity_removed_count_values = 0

    def tighten_capacity(region: _Region) -> _Region:
        nonlocal capacity_region_visits, capacity_requirement_coordinate_reads
        nonlocal capacity_tightened_bounds, capacity_removed_count_values
        capacity_region_visits += 1
        local_requirements = tuple(requirements[index] for index in region.eligible)
        # One pass reserves lower counts; another reads each coordinate's cap.
        capacity_requirement_coordinate_reads += 2 * len(local_requirements) * dimension
        upper = _integer_capacity_upper_bounds(
            region.lower, region.upper, local_requirements, queue, lanes)
        capacity_tightened_bounds += sum(new < old for new, old in zip(upper, region.upper, strict=True))
        capacity_removed_count_values += sum(old - new for new, old in zip(upper, region.upper, strict=True))
        return replace(region, upper=upper) if upper != region.upper else region

    def may_attain_threshold(region: _Region) -> bool:
        return any(
            region.upper[position] >= 1
            for position in region.exact_duration_positions
        )

    def analytic_root_bound(
        eligible: tuple[int, ...], threshold: Fraction
    ) -> Fraction:
        weighted_lane_cap = lanes * max(
            (weights[index] for index in eligible), default=0
        )
        visible_queue_cap = sum(value * value for value in queue)
        return Fraction(min(weighted_lane_cap, visible_queue_cap)) / threshold

    def residual_candidate(region: _Region, primal: tuple[float, ...] | None) -> None:
        nonlocal residual_fill_option_probes
        if primal is None:
            counts = list(region.lower)
        else:
            counts = []
            for value, left, right in zip(
                primal, region.lower, region.upper, strict=True
            ):
                bounded = min(max(float(value), float(left)), float(right))
                counts.append(max(left, min(right, math.floor(bounded))))
        if not lower_feasible(_Region(
            region.threshold_index,
            region.threshold,
            region.exact_duration_positions,
            region.eligible,
            tuple(counts),
            region.upper,
            region.bound,
            region.solved,
            region.primal_counts,
        )):
            counts = list(region.lower)

        lane_left = lanes - sum(counts)
        residual = [
            queue[coordinate] - sum(
                counts[position] * requirements[index][coordinate]
                for position, index in enumerate(region.eligible)
            )
            for coordinate in range(dimension)
        ]
        for position, index in enumerate(region.eligible):
            residual_fill_option_probes += 1
            cap = min(region.upper[position] - counts[position], lane_left)
            if cap <= 0:
                continue
            requirement = requirements[index]
            queue_cap = min(
                (residual[coordinate] // amount
                 for coordinate, amount in enumerate(requirement) if amount),
                default=cap,
            )
            copies = min(cap, queue_cap)
            if copies <= 0:
                continue
            counts[position] += copies
            lane_left -= copies
            for coordinate, amount in enumerate(requirement):
                residual[coordinate] -= copies * amount
        consider_counts(region.eligible, counts)

    def solve_region(region: _Region) -> _Region:
        nonlocal lp_calls, successful_lp_calls, dual_certified_lp_calls
        nonlocal safe_fallback_lp_calls, dual_repair_count
        nonlocal maximum_dual_repair, lp_matrix_coefficients
        nonlocal dual_column_checks
        lp_calls += 1
        eligible = region.eligible
        variable_count = len(eligible)
        base_objective = sum(
            region.lower[position] * weights[index]
            for position, index in enumerate(eligible)
        )
        lane_rhs = lanes - sum(region.lower)
        resource_rhs = [
            queue[coordinate] - sum(
                region.lower[position] * requirements[index][coordinate]
                for position, index in enumerate(eligible)
            )
            for coordinate in range(dimension)
        ]
        upper_rhs = [
            right - left
            for left, right in zip(region.lower, region.upper, strict=True)
        ]
        matrix: list[list[int]] = [[1] * variable_count]
        matrix.extend([
            [requirements[index][coordinate] for index in eligible]
            for coordinate in range(dimension)
        ])
        matrix.extend([
            [int(column == row) for column in range(variable_count)]
            for row in range(variable_count)
        ])
        rhs = [lane_rhs, *resource_rhs, *upper_rhs]
        objective = [weights[index] for index in eligible]
        coefficients = len(matrix) * variable_count
        lp_matrix_coefficients += coefficients

        result = None
        try:
            result = linprog(
                [-float(value) for value in objective],
                A_ub=matrix,
                b_ub=rhs,
                bounds=[(0.0, None)] * variable_count,
                method="highs",
            )
        except Exception:
            result = None

        primal = None
        bound = region.bound
        solver_success = bool(result is not None and result.success)
        dual_certified = False
        repair = Fraction(0)
        if solver_success:
            successful_lp_calls += 1
            try:
                primal = tuple(
                    float(value) + region.lower[position]
                    for position, value in enumerate(result.x)
                )
                marginals = tuple(result.ineqlin.marginals)
                dual_bound, repair = _dual_upper(
                    matrix,
                    rhs,
                    objective,
                    base_objective,
                    region.threshold,
                    marginals,
                )
                bound = min(bound, dual_bound)
                dual_certified = True
                dual_certified_lp_calls += 1
                dual_column_checks += variable_count
                if repair > 0:
                    dual_repair_count += 1
                    maximum_dual_repair = max(maximum_dual_repair, repair)
            except (ArithmeticError, AttributeError, TypeError, ValueError):
                primal = None
        if not dual_certified:
            safe_fallback_lp_calls += 1

        residual_candidate(region, primal)
        return _Region(
            region.threshold_index,
            region.threshold,
            region.exact_duration_positions,
            region.eligible,
            region.lower,
            region.upper,
            bound,
            True,
            primal,
        )

    frontier: list[tuple[Fraction, int, tuple[int, ...], tuple[int, ...], int, _Region]] = []
    serial = 0
    frontier_peak = 0
    bound_pruned_regions = 0
    canonical_key_bound_evaluations = 0
    canonical_tie_pruned_regions = 0
    canonical_tie_singletons_resolved = 0
    analytic_infeasible_regions = 0
    threshold_empty_regions = 0

    def region_canonical_key_lower_bound(
        region: _Region,
    ) -> tuple[Fraction, tuple[int, ...]]:
        """Return an optimistic lower bound on every canonical key in a box.

        Starting with all mandatory lower counts reserves those copies for
        later indices.  Remaining lanes are then assigned greedily in option-
        index order up to each box upper bound.  This is the lexicographically
        smallest fixed-length sorted index tuple admitted by only the box and
        lane constraints.  Ignoring queue resources and the requirement to
        attain the duration stratum can only make the key smaller, which is
        exactly the safe direction for equality pruning.
        """
        counts = list(region.lower)
        remaining = lanes - sum(counts)
        for position in range(len(counts)):
            if remaining <= 0:
                break
            copies = min(region.upper[position] - counts[position], remaining)
            counts[position] += copies
            remaining -= copies
        indices = tuple(
            representatives[region.threshold_index][index]
            for position, index in enumerate(region.eligible)
            for _ in range(counts[position])
        )
        encoded = indices + (option_count,) * (lanes - len(indices))
        return region.threshold, encoded

    def region_is_prunable(region: _Region, *, account: bool) -> bool:
        nonlocal canonical_key_bound_evaluations, canonical_tie_pruned_regions
        if region.bound < best.score:
            return True
        if region.bound > best.score:
            return False
        if account:
            canonical_key_bound_evaluations += 1
        prunable = region_canonical_key_lower_bound(region) >= candidate_key(best)
        if prunable and account:
            canonical_tie_pruned_regions += 1
        return prunable

    def push(region: _Region) -> None:
        nonlocal serial, frontier_peak, bound_pruned_regions
        nonlocal analytic_infeasible_regions, threshold_empty_regions
        if not lower_feasible(region):
            analytic_infeasible_regions += 1
            return
        if not region.solved:
            region = tighten_capacity(region)
        if not may_attain_threshold(region):
            threshold_empty_regions += 1
            return
        if region_is_prunable(region, account=True):
            bound_pruned_regions += 1
            return
        heapq.heappush(
            frontier,
            (
                -region.bound,
                region.threshold_index,
                region.lower,
                region.upper,
                serial,
                region,
            ),
        )
        serial += 1
        frontier_peak = max(frontier_peak, len(frontier))

    # Every duration root is an actual paid LP call.
    for threshold_index, threshold in enumerate(thresholds):
        eligible = tuple(sorted(
            representatives[threshold_index],
            key=representatives[threshold_index].__getitem__,
        ))
        exact_positions = tuple(
            position for position, index in enumerate(eligible)
            if durations[index] == threshold
        )
        root_bound = analytic_root_bound(eligible, threshold)
        root = _Region(
            threshold_index,
            threshold,
            exact_positions,
            eligible,
            (0,) * len(eligible),
            (lanes,) * len(eligible),
            root_bound,
            False,
            None,
        )
        push(solve_region(tighten_capacity(root)))

    branch_splits = 0
    singleton_regions_closed = 0
    singleton_candidate_evaluations = 0

    def branch(region: _Region) -> tuple[_Region, _Region] | None:
        position = _branch_position(region)
        if position is None:
            return None
        left_limit = region.lower[position]
        right_limit = region.upper[position]
        if region.primal_counts is None:
            cut = (left_limit + right_limit) // 2
        else:
            proposed = math.floor(region.primal_counts[position])
            cut = proposed if left_limit <= proposed < right_limit else (
                left_limit + right_limit
            ) // 2
        lower_left = list(region.lower)
        upper_left = list(region.upper)
        upper_left[position] = cut
        lower_right = list(region.lower)
        upper_right = list(region.upper)
        lower_right[position] = cut + 1
        common = (
            region.threshold_index,
            region.threshold,
            region.exact_duration_positions,
            region.eligible,
        )
        return (
            _Region(
                *common,
                tuple(lower_left),
                tuple(upper_left),
                region.bound,
                False,
                None,
            ),
            _Region(
                *common,
                tuple(lower_right),
                tuple(upper_right),
                region.bound,
                False,
                None,
            ),
        )

    def snapshot(lp_node_budget: int) -> CanonicalCountSpaceBnbSelection:
        # Discard entries made obsolete by the final global incumbent.
        retained = [
            entry
            for entry in frontier
            if not region_is_prunable(entry[-1], account=False)
        ]
        snapshot_bound_pruned = (
            bound_pruned_regions + len(frontier) - len(retained)
        )
        upper_exact = max(
            (best.score, *(entry[-1].bound for entry in frontier)),
        )
        score = float(best.score)
        upper = score if upper_exact == best.score else _float_up(upper_exact)
        gap = max(0.0, upper - score)
        queue_norm = sum(queue)
        score_exact = upper_exact == best.score
        canonical_ties_remaining = sum(
            entry[-1].bound == best.score for entry in retained
        )
        canonical_exact = score_exact and not retained
        return CanonicalCountSpaceBnbSelection(
            option_counts=best.counts,
            option_indices=best.indices,
            service=best.service,
            duration=float(best.duration),
            score=score,
            upper_bound=upper,
            gap=gap,
            epsilon=gap / queue_norm if queue_norm else 0.0,
            exact=canonical_exact,
            score_exact=score_exact,
            canonical_exact=canonical_exact,
            lp_node_budget=lp_node_budget,
            lp_calls=lp_calls,
            duration_root_count=root_count,
            successful_lp_calls=successful_lp_calls,
            dual_certified_lp_calls=dual_certified_lp_calls,
            safe_fallback_lp_calls=safe_fallback_lp_calls,
            dual_repair_count=dual_repair_count,
            maximum_dual_repair=float(maximum_dual_repair),
            candidate_evaluations=candidate_evaluations,
            singleton_candidate_evaluations=singleton_candidate_evaluations,
            residual_fill_option_probes=residual_fill_option_probes,
            branch_splits=branch_splits,
            analytic_infeasible_regions=analytic_infeasible_regions,
            threshold_empty_regions=threshold_empty_regions,
            singleton_regions_closed=singleton_regions_closed,
            bound_pruned_regions=snapshot_bound_pruned,
            canonical_key_bound_evaluations=canonical_key_bound_evaluations,
            canonical_tie_pruned_regions=canonical_tie_pruned_regions,
            canonical_tie_regions_remaining=canonical_ties_remaining,
            canonical_tie_incumbent_updates=canonical_tie_incumbent_updates,
            canonical_tie_singletons_resolved=canonical_tie_singletons_resolved,
            frontier_regions_remaining=len(retained),
            inherited_unsolved_regions=sum(
                not entry[-1].solved for entry in retained
            ),
            frontier_peak=frontier_peak,
            lp_matrix_coefficients=lp_matrix_coefficients,
            dual_column_checks=dual_column_checks,
            represented_ordered_configurations=(option_count + 1) ** lanes,
            represented_count_vectors=math.comb(option_count + lanes, lanes),
        )

    while True:
        while frontier and lp_calls < budget:
            _, _, _, _, _, region = heapq.heappop(frontier)
            if region_is_prunable(region, account=True):
                bound_pruned_regions += 1
                continue
            if not region.solved:
                push(solve_region(region))
                continue
            children = branch(region)
            if children is None:
                singleton_regions_closed += 1
                actual_threshold = max((idle_fraction, *(durations[index]
                    for position, index in enumerate(region.eligible)
                    if region.lower[position])))
                if actual_threshold == region.threshold:
                    singleton_candidate_evaluations += 1
                    incumbent_before = best
                    consider_counts(region.eligible, region.lower)
                    if best is not incumbent_before and best.score == incumbent_before.score:
                        canonical_tie_singletons_resolved += 1
                continue
            branch_splits += 1
            for child in children:
                push(child)

        budget = yield IntegerCapacityThresholdSelection(
            selection=snapshot(budget),
            preprocessing=ThresholdCatalogueWork(
                original_option_count=option_count,
                requirement_group_count=len(groups),
                duration_threshold_count=root_count,
                reduction_option_probes=reduction.reduction_option_probes,
                reduction_requirement_coordinate_reads=reduction.reduction_requirement_coordinate_reads,
                reduction_duration_comparisons=reduction.reduction_duration_comparisons,
                grouping_option_probes=option_count,
                representative_option_probes=representative_probes,
                proxy_option_count=sum(len(indices) for indices in representatives),
                lift_option_probes=lift_probes,
            ),
            thresholds=tuple(float(threshold) for threshold in thresholds),
            capacity_propagation=CapacityPropagationWork(
                region_visits=capacity_region_visits,
                requirement_coordinate_reads=capacity_requirement_coordinate_reads,
                tightened_bounds=capacity_tightened_bounds,
                removed_count_values=capacity_removed_count_values,
            ),
        )
