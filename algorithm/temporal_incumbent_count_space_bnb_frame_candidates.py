"""Exact count-space B&B seeded by a feasible temporal incumbent.

The search and certificate are the count-space method's, but ``best`` may start
from a feasible projection of the previous frame's exact option counts.  The
projection only supplies a lower bound.  It neither constrains the frontier nor
changes any LP upper bound, so a closed frontier still proves the same exact
optimum and deterministic tie break as a cold search.

An LP node budget counts every call to ``scipy.optimize.linprog``, including a
failed call.  A child that has not yet received an LP call remains on the
frontier with its parent's valid bound.  Candidate construction and matrix work
are reported separately; they are bounded by the number and size of solved
nodes and are never presented as free Cartesian search.  Multiple checkpoints
share one cumulative search rather than paying for the same roots repeatedly.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
from typing import Any, Sequence

from scipy.optimize import linprog


@dataclass(frozen=True)
class CountSpaceBnbSelection:
    option_counts: tuple[int, ...]
    option_indices: tuple[int | None, ...]
    service: tuple[int, ...]
    duration: float
    score: float
    upper_bound: float
    gap: float
    epsilon: float
    exact: bool
    lp_node_budget: int
    lp_calls: int
    duration_root_count: int
    successful_lp_calls: int
    dual_certified_lp_calls: int
    safe_fallback_lp_calls: int
    dual_repair_count: int
    maximum_dual_repair: float
    candidate_evaluations: int
    initial_incumbent_candidate_evaluations: int
    singleton_candidate_evaluations: int
    residual_fill_option_probes: int
    branch_splits: int
    analytic_infeasible_regions: int
    threshold_empty_regions: int
    singleton_regions_closed: int
    bound_pruned_regions: int
    frontier_regions_remaining: int
    inherited_unsolved_regions: int
    frontier_peak: int
    lp_matrix_coefficients: int
    dual_column_checks: int
    represented_ordered_configurations: int
    represented_count_vectors: int


@dataclass(frozen=True)
class TemporalIncumbentProjection:
    """A deterministic feasible projection and its explicit non-LP work."""

    option_counts: tuple[int, ...]
    service: tuple[int, ...]
    duration: float
    score: float
    retained_copies: int
    greedy_added_copies: int
    option_probes: int


@dataclass(frozen=True)
class _Region:
    threshold_index: int
    threshold: Fraction
    exact_duration_positions: tuple[int, ...]
    eligible: tuple[int, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    bound: Fraction
    solved: bool
    primal_counts: tuple[float, ...] | None


@dataclass(frozen=True)
class _Candidate:
    counts: tuple[int, ...]
    indices: tuple[int | None, ...]
    service: tuple[int, ...]
    duration: Fraction
    score: Fraction


def _float_up(value: Fraction) -> float:
    """Convert a nonnegative exact value to a float that does not round down."""
    converted = float(value)
    if Fraction.from_float(converted) < value:
        converted = math.nextafter(converted, math.inf)
    return converted


def _dual_upper(
    matrix: Sequence[Sequence[int]],
    rhs: Sequence[int],
    objective: Sequence[int],
    base_objective: int,
    threshold: Fraction,
    marginals: Sequence[float],
) -> tuple[Fraction, Fraction]:
    """Repair HiGHS marginals into an exactly checked feasible max-LP dual.

    The first row is the lane constraint and has coefficient one in every
    column.  Clamping candidate dual prices to nonnegative values and adding
    the largest remaining column deficit to that lane price therefore yields a
    feasible dual.  All checks below use exact rationals of the binary floats.
    """
    if len(marginals) != len(matrix):
        raise ValueError("dual marginal count does not match the LP rows")
    prices: list[Fraction] = []
    for marginal in marginals:
        value = -float(marginal)
        if not math.isfinite(value):
            raise ValueError("dual marginals must be finite")
        prices.append(Fraction.from_float(max(0.0, value)))

    deficits = []
    for column, coefficient in enumerate(objective):
        lhs = sum(
            Fraction(row[column]) * price
            for row, price in zip(matrix, prices, strict=True)
        )
        deficits.append(max(Fraction(0), Fraction(coefficient) - lhs))
    repair = max(deficits, default=Fraction(0))
    if prices:
        prices[0] += repair

    for column, coefficient in enumerate(objective):
        lhs = sum(
            Fraction(row[column]) * price
            for row, price in zip(matrix, prices, strict=True)
        )
        if lhs < coefficient:
            raise ArithmeticError("repaired dual is not column-feasible")
    if any(price < 0 for price in prices):
        raise ArithmeticError("repaired dual has a negative price")

    numerator_upper = Fraction(base_objective) + sum(
        Fraction(limit) * price
        for limit, price in zip(rhs, prices, strict=True)
    )
    return numerator_upper / threshold, repair


def project_temporal_incumbent(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    previous_option_counts: Sequence[int],
    idle_duration: float = 1.0,
) -> TemporalIncumbentProjection:
    """Project previous exact counts to the current visible queue.

    Copies are retained in catalogue order while capacity permits.  Remaining
    lanes are filled one copy at a time with the feasible option producing the
    best resulting frame score; a copy is added only when it improves the
    candidate under the selector's score/duration/index ordering.  All choices
    use only the current queue and the previous count vector.
    """
    options = tuple(options)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    idle_duration = float(idle_duration)
    if not math.isfinite(idle_duration) or idle_duration <= 0.0:
        raise ValueError("idle_duration must be positive and finite")
    if any(int(value) != value or value < 0 for value in queues):
        raise ValueError("queues must contain nonnegative integer job counts")
    queue = tuple(int(value) for value in queues)
    if len(previous_option_counts) != len(options) or any(
        int(value) != value or value < 0 for value in previous_option_counts
    ):
        raise ValueError(
            "previous_option_counts must match the nonnegative integer catalogue"
        )
    previous = tuple(int(value) for value in previous_option_counts)

    requirements: list[tuple[int, ...]] = []
    durations: list[Fraction] = []
    for option in options:
        requirement = tuple(option.requirements)
        if len(requirement) != len(queue) or any(
            int(value) != value or value < 0 for value in requirement
        ):
            raise ValueError(
                "option requirements must match the nonnegative integer queue coordinates"
            )
        requirement = tuple(int(value) for value in requirement)
        if not any(requirement):
            raise ValueError("a non-idle option must require at least one job")
        duration = float(option.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("option durations must be positive and finite")
        requirements.append(requirement)
        durations.append(Fraction.from_float(max(idle_duration, duration)))

    counts = [0] * len(options)
    residual = list(queue)
    lanes_left = lanes
    retained = 0
    probes = 0
    for index, requested in enumerate(previous):
        if lanes_left <= 0:
            break
        requirement = requirements[index]
        queue_cap = min(
            (
                residual[coordinate] // amount
                for coordinate, amount in enumerate(requirement)
                if amount
            ),
            default=lanes_left,
        )
        copies = min(requested, lanes_left, queue_cap)
        counts[index] += copies
        lanes_left -= copies
        retained += copies
        for coordinate, amount in enumerate(requirement):
            residual[coordinate] -= copies * amount

    idle_fraction = Fraction.from_float(idle_duration)

    def candidate_key(candidate_counts: Sequence[int]):
        used = [
            index for index, count in enumerate(candidate_counts) if count
        ]
        duration = max((idle_fraction, *(durations[index] for index in used)))
        service = tuple(
            sum(
                candidate_counts[index] * requirements[index][coordinate]
                for index in range(len(options))
            )
            for coordinate in range(len(queue))
        )
        score = Fraction(
            sum(
                waiting * done
                for waiting, done in zip(queue, service, strict=True)
            )
        ) / duration
        indices = tuple(
            index
            for index, count in enumerate(candidate_counts)
            for _ in range(count)
        ) + (len(options),) * (lanes - sum(candidate_counts))
        # Maximize score, then match the selector's minimum duration/index tie.
        return score, -duration, tuple(-index for index in indices), service, duration

    greedy_added = 0
    current_key = candidate_key(counts)
    while lanes_left > 0:
        winner = None
        winner_key = current_key
        for index, requirement in enumerate(requirements):
            probes += 1
            if any(
                amount > residual[coordinate]
                for coordinate, amount in enumerate(requirement)
            ):
                continue
            proposed = list(counts)
            proposed[index] += 1
            key = candidate_key(proposed)
            if key[:3] > winner_key[:3]:
                winner = index
                winner_key = key
        if winner is None:
            break
        counts[winner] += 1
        lanes_left -= 1
        greedy_added += 1
        for coordinate, amount in enumerate(requirements[winner]):
            residual[coordinate] -= amount
        current_key = winner_key

    score, _, _, service, duration = candidate_key(counts)
    return TemporalIncumbentProjection(
        option_counts=tuple(counts),
        service=service,
        duration=float(duration),
        score=float(score),
        retained_copies=retained,
        greedy_added_copies=greedy_added,
        option_probes=probes,
    )


def trace_count_space_bnb_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    lp_node_budgets: Sequence[int],
    idle_duration: float = 1.0,
    *,
    initial_option_counts: Sequence[int] | None = None,
) -> tuple[CountSpaceBnbSelection, ...]:
    """Return cumulative anytime snapshots at the requested LP budgets.

    One root is solved for every distinct effective non-idle duration.  Thus
    every requested budget must cover all duration roots.  Roots and subsequent
    branch nodes are solved once through the largest checkpoint; smaller
    checkpoints are cumulative snapshots of that same search.  Every action
    belongs to the stratum matching its actual maximum duration.  A root LP
    relaxes the stratum's requirement to use an option at that exact duration;
    omitting that requirement can only increase its upper bound.

    Each LP is a fractional packing problem over option counts with lane,
    visible-queue, and node-box constraints.  HiGHS' dual candidate is repaired
    and checked with :class:`fractions.Fraction`.  If no certifiable dual is
    available, the node retains its inherited analytic upper bound.
    """
    options = tuple(options)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    budgets = tuple(lp_node_budgets)
    if not budgets:
        raise ValueError("lp_node_budgets must contain at least one checkpoint")
    if any(int(value) != value or value < 0 for value in budgets):
        raise ValueError("lp_node_budgets must contain nonnegative integers")
    budgets = tuple(int(value) for value in budgets)
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("lp_node_budgets must be strictly increasing")
    idle_duration = float(idle_duration)
    if not math.isfinite(idle_duration) or idle_duration <= 0.0:
        raise ValueError("idle_duration must be positive and finite")
    if any(int(value) != value or value < 0 for value in queues):
        raise ValueError("queues must contain nonnegative integer job counts")
    queue = tuple(int(value) for value in queues)

    requirements: list[tuple[int, ...]] = []
    durations: list[Fraction] = []
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
        effective = max(idle_duration, duration)
        requirements.append(counts)
        durations.append(Fraction.from_float(effective))
        weights.append(sum(
            waiting * count
            for waiting, count in zip(queue, counts, strict=True)
        ))

    thresholds = tuple(sorted(set(durations)))
    root_count = len(thresholds)
    if budgets[0] < root_count:
        raise ValueError(
            f"every LP checkpoint must cover {root_count} duration roots"
        )

    option_count = len(options)
    dimension = len(queue)
    idle_fraction = Fraction.from_float(idle_duration)
    idle_indices: tuple[int | None, ...] = (None,) * lanes
    best = _Candidate(
        counts=(0,) * option_count,
        indices=idle_indices,
        service=(0,) * dimension,
        duration=idle_fraction,
        score=Fraction(0),
    )
    candidate_evaluations = 1
    residual_fill_option_probes = 0

    def make_candidate(
        eligible: tuple[int, ...],
        local_counts: Sequence[int],
    ) -> _Candidate:
        counts = [0] * option_count
        for position, index in enumerate(eligible):
            counts[index] = int(local_counts[position])
        if any(value < 0 for value in counts) or sum(counts) > lanes:
            raise ArithmeticError("candidate count construction is invalid")
        service = tuple(
            sum(counts[index] * requirements[index][coordinate]
                for index in range(option_count))
            for coordinate in range(dimension)
        )
        if any(done > waiting for done, waiting in zip(service, queue, strict=True)):
            raise ArithmeticError("candidate exceeds the visible queue")
        used = [index for index, count in enumerate(counts) if count]
        duration = max(
            (idle_fraction, *(durations[index] for index in used))
        )
        numerator = sum(
            waiting * done
            for waiting, done in zip(queue, service, strict=True)
        )
        score = Fraction(numerator) / duration
        expanded = tuple(
            index for index, count in enumerate(counts) for _ in range(count)
        )
        indices = expanded + (None,) * (lanes - len(expanded))
        return _Candidate(
            tuple(counts), indices, service, duration, score
        )

    def candidate_key(candidate: _Candidate):
        return (
            candidate.duration,
            tuple(
                option_count if index is None else index
                for index in candidate.indices
            ),
        )

    def consider_counts(
        eligible: tuple[int, ...],
        local_counts: Sequence[int],
    ) -> None:
        nonlocal best, candidate_evaluations
        candidate_evaluations += 1
        candidate = make_candidate(eligible, local_counts)
        if candidate.score > best.score or (
            candidate.score == best.score
            and candidate_key(candidate) < candidate_key(best)
        ):
            best = candidate

    warm_best = None
    if initial_option_counts is not None:
        if len(initial_option_counts) != option_count or any(
            int(value) != value or value < 0 for value in initial_option_counts
        ):
            raise ValueError(
                "initial_option_counts must match the nonnegative integer catalogue"
            )
        initial_counts = tuple(int(value) for value in initial_option_counts)
        warm_best = make_candidate(tuple(range(option_count)), initial_counts)
        candidate_evaluations += 1

    def active_best() -> _Candidate:
        # Once normal search reaches the warm score, its incumbent has exactly
        # the cold search's tie authority.  Before then, warm contributes score
        # only; returning it is safe only after equal-score regions are closed.
        if warm_best is not None and warm_best.score > best.score:
            return warm_best
        return best

    def score_prunable(bound: Fraction) -> bool:
        # Cold-discovered incumbents retain the original <= pruning semantics.
        # A foreign warm incumbent uses strict pruning so it cannot hide an
        # equal-score action with the cold reference's preferred key.
        return bound <= best.score or (
            warm_best is not None and bound < warm_best.score
        )

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
    analytic_infeasible_regions = 0
    threshold_empty_regions = 0

    def push(region: _Region) -> None:
        nonlocal serial, frontier_peak, bound_pruned_regions
        nonlocal analytic_infeasible_regions, threshold_empty_regions
        if not lower_feasible(region):
            analytic_infeasible_regions += 1
            return
        if not may_attain_threshold(region):
            threshold_empty_regions += 1
            return
        if score_prunable(region.bound):
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
        eligible = tuple(
            index for index, duration in enumerate(durations)
            if duration <= threshold
        )
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
        push(solve_region(root))

    branch_splits = 0
    singleton_regions_closed = 0
    singleton_candidate_evaluations = 0

    def branch(region: _Region) -> tuple[_Region, _Region] | None:
        choices = [
            position for position, (left, right) in enumerate(
                zip(region.lower, region.upper, strict=True)
            ) if left < right
        ]
        if not choices:
            return None
        position = choices[0]
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

    def snapshot(lp_node_budget: int) -> CountSpaceBnbSelection:
        # Report the same terminal cleanup as a stopped one-budget search, but
        # leave the live heap intact.  Later checkpoints then see exactly the
        # frontier (and frontier peak) that an uninterrupted larger-budget run
        # would have seen; dominated entries cost no LP and are removed when
        # encountered.
        retained = [
            entry for entry in frontier if not score_prunable(entry[-1].bound)
        ]
        snapshot_bound_pruned = (
            bound_pruned_regions + len(frontier) - len(retained)
        )
        selected_best = active_best()
        upper_exact = max(
            (selected_best.score, *(entry[-1].bound for entry in retained)),
        )
        tie_unresolved = (
            warm_best is not None
            and warm_best.score > best.score
            and any(entry[-1].bound >= warm_best.score for entry in retained)
        )
        score = float(selected_best.score)
        upper = (
            score
            if upper_exact == selected_best.score
            else _float_up(upper_exact)
        )
        gap = max(0.0, upper - score)
        queue_norm = sum(queue)
        return CountSpaceBnbSelection(
            option_counts=selected_best.counts,
            option_indices=selected_best.indices,
            service=selected_best.service,
            duration=float(selected_best.duration),
            score=score,
            upper_bound=upper,
            gap=gap,
            epsilon=gap / queue_norm if queue_norm else 0.0,
            exact=upper_exact == selected_best.score and not tie_unresolved,
            lp_node_budget=lp_node_budget,
            lp_calls=lp_calls,
            duration_root_count=root_count,
            successful_lp_calls=successful_lp_calls,
            dual_certified_lp_calls=dual_certified_lp_calls,
            safe_fallback_lp_calls=safe_fallback_lp_calls,
            dual_repair_count=dual_repair_count,
            maximum_dual_repair=float(maximum_dual_repair),
            candidate_evaluations=candidate_evaluations,
            initial_incumbent_candidate_evaluations=(
                int(warm_best is not None)
            ),
            singleton_candidate_evaluations=singleton_candidate_evaluations,
            residual_fill_option_probes=residual_fill_option_probes,
            branch_splits=branch_splits,
            analytic_infeasible_regions=analytic_infeasible_regions,
            threshold_empty_regions=threshold_empty_regions,
            singleton_regions_closed=singleton_regions_closed,
            bound_pruned_regions=snapshot_bound_pruned,
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

    snapshots = []
    for checkpoint in budgets:
        while frontier and lp_calls < checkpoint:
            # A solved singleton needs no further LP call, but its direct
            # candidate evaluation remains explicit work.
            _, _, _, _, _, region = heapq.heappop(frontier)
            if score_prunable(region.bound):
                bound_pruned_regions += 1
                continue
            if not region.solved:
                push(solve_region(region))
                continue
            children = branch(region)
            if children is None:
                singleton_regions_closed += 1
                local_counts = region.lower
                actual_threshold = max(
                    (
                        idle_fraction,
                        *(
                            durations[index]
                            for position, index in enumerate(region.eligible)
                            if local_counts[position]
                        ),
                    )
                )
                if actual_threshold == region.threshold:
                    singleton_candidate_evaluations += 1
                    consider_counts(region.eligible, local_counts)
                continue
            branch_splits += 1
            for child in children:
                push(child)
        snapshots.append(snapshot(checkpoint))
    return tuple(snapshots)


def select_count_space_bnb_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    lp_node_budget: int,
    idle_duration: float = 1.0,
    *,
    initial_option_counts: Sequence[int] | None = None,
) -> CountSpaceBnbSelection:
    """Return one feasible frame and its anytime support-loss certificate."""
    return trace_count_space_bnb_frame(
        options,
        queues,
        lanes,
        (lp_node_budget,),
        idle_duration,
        initial_option_counts=initial_option_counts,
    )[0]


__all__ = [
    "CountSpaceBnbSelection",
    "TemporalIncumbentProjection",
    "project_temporal_incumbent",
    "select_count_space_bnb_frame",
    "trace_count_space_bnb_frame",
]
