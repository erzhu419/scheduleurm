"""Keep paid root faces intact for the single canonical integer probe.

The genuine raw-M snapshot, own-tight fixing and ordinary LP continuation are
unchanged. Root processing retains cheap pruning and singleton updates, then
returns the first still-open solved box to the same heap without splitting.
A failed integer probe continues this frontier at the original LP budgets.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import heapq
import math
from typing import Any, Generator, Sequence

from scipy.optimize import linprog

from algorithm.canonical_count_space_bnb_frame_candidates import (
    CanonicalCountSpaceBnbSelection, _Candidate, _float_up,
)
from algorithm.dual_reuse_integer_capacity_frame_candidates import (
    DualReuseIntegerCapacitySearch, DualReuseIntegerCapacityThresholdSelection,
    DualReuseWork, _DualRegion,
)
from algorithm.fractional_shared_threshold_canonical_requirement_dominance_frame_candidates import (
    _branch_position,
)
from algorithm.integer_capacity_shared_threshold_frame_candidates import (
    CapacityPropagationWork, _integer_capacity_upper_bounds,
)
from algorithm.requirement_dominance_frame_candidates import reduce_requirement_dominated_options
from algorithm.root_pruned_integer_capacity_frame_candidates import RootPruningWork
from algorithm.threshold_canonical_requirement_dominance_frame_candidates import ThresholdCatalogueWork


@dataclass(frozen=True)
class NativeOwnTightWork:
    """Explicit reads, cache assignments, copies, calls and diagnostics.

    Cache reads exclude the separate face-test own_bound_reads. Endpoint
    writes exclude values copied into working/final tuples. The last five
    counters diagnose reductions; native arithmetic is charged on its
    ordinary affine, guard, capacity and canonical cumulative counters.
    """
    checks: int = 0
    bound_comparisons: int = 0
    certificate_reads: int = 0
    own_bound_reads: int = 0
    own_bound_comparisons: int = 0
    variable_visits: int = 0
    lower_flag_reads: int = 0
    upper_price_reads: int = 0
    upper_price_comparisons: int = 0
    box_coordinate_reads: int = 0
    box_coordinate_comparisons: int = 0
    bound_values_copied: int = 0
    bound_values_written: int = 0
    own_bound_cache_reads: int = 0
    own_bound_cache_writes: int = 0
    lower_feasibility_rechecks: int = 0
    capacity_rechecks: int = 0
    threshold_rechecks: int = 0
    canonical_rechecks: int = 0
    event_records_written: int = 0
    event_group_reads: int = 0
    event_requirement_reads: int = 0
    event_price_reads: int = 0
    event_bound_values_copied: int = 0
    event_exact_position_reads: int = 0
    event_fraction_pairs_written: int = 0
    event_rational_operations: int = 0
    event_snapshot_entries_copied: int = 0
    tight_regions: int = 0
    changed_regions: int = 0
    changed_lower_coordinates: int = 0
    changed_upper_coordinates: int = 0
    empty_regions: int = 0


NATIVE_OWN_TIGHT_COUNTERS = tuple(NativeOwnTightWork.__dataclass_fields__)
NATIVE_OWN_TIGHT_FEE_COUNTERS = NATIVE_OWN_TIGHT_COUNTERS[:-5]


@dataclass(frozen=True)
class NativeOwnTightEvent:
    threshold_fraction: tuple[int, int]
    bound_fraction: tuple[int, int]
    incumbent_score_fraction: tuple[int, int]
    incumbent_duration_fraction: tuple[int, int]
    incumbent_counts: tuple[int, ...]
    incumbent_indices: tuple[int | None, ...]
    eligible: tuple[int, ...]
    requirements: tuple[tuple[int, ...], ...]
    exact_duration_positions: tuple[int, ...]
    certificate_prices: tuple[tuple[int, int], ...]
    certificate_lower: tuple[int, ...]
    certificate_upper: tuple[int, ...]
    certificate_numerator_fraction: tuple[int, int]
    anchor_kind: str
    anchor_lower: tuple[int, ...]
    anchor_upper: tuple[int, ...]
    anchor_numerator_fraction: tuple[int, int]
    current_own_numerator_fraction: tuple[int, int]
    target_numerator_fraction: tuple[int, int]
    own_bound_fraction: tuple[int, int]
    before_lower: tuple[int, ...]
    before_upper: tuple[int, ...]
    face_lower: tuple[int, ...]
    face_upper: tuple[int, ...]
    after_lower: tuple[int, ...]
    after_upper: tuple[int, ...]
    disposition: str


def _own_tight_endpoints(lower, upper, lower_zero_flags, upper_prices, price_offset):
    """Return the simultaneous exact face, including contradictory endpoints."""
    face_lower, face_upper = list(lower), list(upper)
    changed_lower = changed_upper = 0
    empty = False
    for position, (left, right, lower_zero) in enumerate(zip(
            lower, upper, lower_zero_flags, strict=True)):
        cap = upper_prices[price_offset + position]
        cap_positive = cap > 0
        if left < right:
            if not lower_zero:
                face_upper[position] = left
                changed_upper += 1
            if cap_positive:
                face_lower[position] = right
                changed_lower += 1
            empty |= not lower_zero and cap_positive
    return tuple(face_lower), tuple(face_upper), changed_lower, changed_upper, empty


@dataclass(frozen=True)
class RootSettleWork:
    """Extra phase events plus native-work diagnostics, never duplicated fees.

    The unsolved-head event includes the native read-only pruning/key check.
    Native branch, capacity, dual and guard work remains on the cumulative LP
    snapshot. Frontier before/after values are raw heap lengths. A checkpoint
    visits every raw heap entry once and creates no retained-list scan.
    A branch-stop pop is returned to the heap and recorded separately from
    closed pops; root splitting is absent. Its head-check event includes the
    branch-position test and intact-box return, all inside the whole call.
    """
    calls: int
    head_checks: int
    unsolved_prune_checks: int
    popped_regions: int
    pruned_regions: int
    split_regions: int
    singleton_regions: int
    stopped_for_lp: int
    snapshot_calls: int
    snapshot_heap_entries_read: int
    snapshot_retained_entries_read: int
    lp_calls_before: int
    lp_calls_after: int
    frontier_regions_before: int
    frontier_regions_after: int
    stopped_for_branch: int


ROOT_SETTLE_COUNTERS = tuple(RootSettleWork.__dataclass_fields__)
ROOT_SETTLE_FEE_COUNTERS = (
    "calls", "head_checks", "unsolved_prune_checks", "snapshot_calls",
    "snapshot_heap_entries_read", "snapshot_retained_entries_read",
)


@dataclass(frozen=True)
class ZeroDeltaWork:
    lower_flags_saved: int
    lower_flag_comparisons: int
    guard_calls: int
    zero_delta_skips: int
    variable_visits: int
    box_coordinate_reads: int
    box_coordinate_comparisons: int
    lower_flag_reads: int
    upper_price_reads: int
    upper_price_zero_comparisons: int


@dataclass(frozen=True)
class AffineDualWork:
    certificates_prepared: int
    certificate_numerator_products: int
    price_reads: int
    denominator_lcm_steps: int
    integer_scalings: int
    coefficient_requirement_reads: int
    coefficient_weight_reads: int
    coefficient_integer_operations: int
    bound_variable_visits: int
    box_coordinate_reads: int
    box_coordinate_comparisons: int
    changed_lower_coordinates: int
    changed_upper_coordinates: int
    delta_coefficient_reads: int
    delta_integer_operations: int
    bound_integer_products: int
    bound_rational_conversions: int


@dataclass(frozen=True)
class DirectProbeOwnTightNativeThresholdSelection(DualReuseIntegerCapacityThresholdSelection):
    affine_dual: AffineDualWork
    zero_delta: ZeroDeltaWork
    native_own_tight_work: NativeOwnTightWork
    native_own_tight_evidence: tuple[NativeOwnTightEvent, ...]


@dataclass(frozen=True)
class _AffineCoefficients:
    denominator: int
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    anchor_numerator: int


@dataclass
class _AffineDualCertificate:
    prices: tuple[Fraction, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    numerator: Fraction
    prepared: _AffineCoefficients | None = None


@dataclass
class _ZeroDeltaCertificate(_AffineDualCertificate):
    lower_zero_flags: tuple[bool, ...] = ()


@dataclass(frozen=True)
class _AffineRegion(_DualRegion):
    certificate: _AffineDualCertificate | None = None
    dual_lower: tuple[int, ...] | None = None
    dual_upper: tuple[int, ...] | None = None
    dual_numerator: int | None = None
    own_bound: Fraction | None = None


def _prepare_affine_certificate(
    certificate: _AffineDualCertificate, requirements: Sequence[Sequence[int]],
    weights: Sequence[int],
) -> _AffineCoefficients:
    """Scale this certificate's own numerator and all coefficients once."""
    denominator = 1
    for price in certificate.prices:
        denominator = math.lcm(denominator, price.denominator)
    denominator = math.lcm(denominator, certificate.numerator.denominator)
    prices = tuple(price.numerator * (denominator // price.denominator)
                   for price in certificate.prices)
    dimension = len(requirements[0]) if requirements else 0
    lower = []
    for position, (requirement, weight) in enumerate(zip(requirements, weights, strict=True)):
        resource = 0
        for coordinate, amount in enumerate(requirement):
            resource += amount * prices[1 + coordinate]
        lower.append(denominator * weight - prices[0] - resource - prices[1 + dimension + position])
    numerator = certificate.numerator.numerator * (denominator // certificate.numerator.denominator)
    return _AffineCoefficients(denominator, tuple(lower), prices[1 + dimension:], numerator)


def _update_affine_numerator(
    coefficients: _AffineCoefficients, numerator: int,
    source_lower: Sequence[int], source_upper: Sequence[int],
    lower: Sequence[int], upper: Sequence[int],
) -> tuple[int, int, int]:
    """Scan both box bounds and update only their changed coordinates."""
    changed_lower = changed_upper = 0
    for position, (new_lower, old_lower, new_upper, old_upper) in enumerate(zip(
        lower, source_lower, upper, source_upper, strict=True,
    )):
        if new_lower != old_lower:
            numerator += (new_lower - old_lower) * coefficients.lower[position]
            changed_lower += 1
        if new_upper != old_upper:
            numerator += (new_upper - old_upper) * coefficients.upper[position]
            changed_upper += 1
    return numerator, changed_lower, changed_upper


def _zero_delta_dual_upper_with_prices(
    matrix: Sequence[Sequence[int]],
    rhs: Sequence[int],
    objective: Sequence[int],
    base_objective: int,
    threshold: Fraction,
    marginals: Sequence[float],
) -> tuple[Fraction, Fraction, tuple[Fraction, ...], tuple[bool, ...]]:
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

    lower_zero_flags = []
    for column, coefficient in enumerate(objective):
        lhs = sum(
            Fraction(row[column]) * price
            for row, price in zip(matrix, prices, strict=True)
        )
        if lhs < coefficient:
            raise ArithmeticError("repaired dual is not column-feasible")
        lower_zero_flags.append(lhs == coefficient)
    if any(price < 0 for price in prices):
        raise ArithmeticError("repaired dual has a negative price")

    numerator_upper = Fraction(base_objective) + sum(
        Fraction(limit) * price
        for limit, price in zip(rhs, prices, strict=True)
    )
    return numerator_upper / threshold, repair, tuple(prices), tuple(lower_zero_flags)


def _zero_delta_guard(
    certificate: _ZeroDeltaCertificate,
    source_lower: Sequence[int], source_upper: Sequence[int],
    lower: Sequence[int], upper: Sequence[int],
) -> tuple[bool, int, int, int]:
    """Check zero support, stopping at the first nonzero changed coefficient."""
    visits = lower_reads = upper_reads = 0
    price_offset = len(certificate.prices) - len(certificate.lower)
    for position, (new_lower, old_lower, new_upper, old_upper) in enumerate(zip(
        lower, source_lower, upper, source_upper, strict=True,
    )):
        visits += 1
        lower_changed = new_lower != old_lower
        upper_changed = new_upper != old_upper
        if lower_changed:
            lower_reads += 1
            if not certificate.lower_zero_flags[position]:
                return False, visits, lower_reads, upper_reads
        if upper_changed:
            upper_reads += 1
            if certificate.prices[price_offset + position] != 0:
                return False, visits, lower_reads, upper_reads
    return True, visits, lower_reads, upper_reads


class DirectProbeOwnTightNativeSearch(DualReuseIntegerCapacitySearch):
    """Build raw roots and conditionally settle them in one generator call."""

    def advance_roots_and_settle(self, initial_lp_budget: int) -> tuple[
        DirectProbeOwnTightNativeThresholdSelection,
        DirectProbeOwnTightNativeThresholdSelection | None,
        RootSettleWork | None,
    ]:
        """Return the genuine raw-M record before any zero-LP root progress.

        Every returned record is fully materialized in this call. Settlement
        runs only for a score-exact, canonical-open root and leaves this same
        search at its settled checkpoint for ordinary cumulative continuation.
        """
        if int(initial_lp_budget) != initial_lp_budget or initial_lp_budget < 0:
            raise ValueError("initial LP budget must be a nonnegative integer")
        if self._search is not None:
            raise ValueError("duration roots are already initialized")
        search = _direct_probe_own_tight_native_search_snapshots(
            self._options, self._queues, self._lanes, int(initial_lp_budget), self._idle_duration)
        root, settled, work = next(search)
        self._search = search
        self._budget = root.selection.lp_node_budget
        self._snapshot = settled if settled is not None else root
        return root, settled, work

    def advance_to(self, total_lp_budget: int) -> DirectProbeOwnTightNativeThresholdSelection:
        if int(total_lp_budget) != total_lp_budget or total_lp_budget < 0:
            raise ValueError("LP budget must be a nonnegative integer")
        budget = int(total_lp_budget)
        if self._budget is not None and budget < self._budget:
            raise ValueError("LP budgets must be nondecreasing")
        if budget == self._budget:
            assert self._snapshot is not None
            return self._snapshot
        if self._search is None:
            raise ValueError("initialize duration roots with advance_roots_and_settle before continuing")
        snapshot = self._search.send(budget)
        self._budget = budget
        self._snapshot = snapshot
        return snapshot


def _direct_probe_own_tight_native_search_snapshots(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, idle_duration: float = 1.0,
) -> Generator[
    DirectProbeOwnTightNativeThresholdSelection
    | tuple[DirectProbeOwnTightNativeThresholdSelection,
            DirectProbeOwnTightNativeThresholdSelection | None,
            RootSettleWork | None],
    int, None,
]:
    """Retain the frozen prune-first search with lazy exact affine dual reuse."""
    options, queue = tuple(options), tuple(queues)
    reduction = reduce_requirement_dominated_options(options, idle_duration)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    if int(initial_lp_budget) != initial_lp_budget or initial_lp_budget < 0:
        raise ValueError("initial_lp_budget must be a nonnegative integer")
    budget = int(initial_lp_budget)
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
        raise ValueError(f"initial LP budget must cover {root_count} duration thresholds")
    budget = root_count
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
    certificates_saved = 0
    child_bound_evaluations = 0
    child_bound_improvements = 0
    child_bound_prunes = 0
    child_rhs_entries = 0
    child_requirement_reads = 0
    child_weight_reads = 0
    certificates_prepared = 0
    certificate_numerator_products = 0
    price_reads = 0
    denominator_lcm_steps = 0
    integer_scalings = 0
    coefficient_requirement_reads = 0
    coefficient_weight_reads = 0
    coefficient_integer_operations = 0
    bound_variable_visits = 0
    box_coordinate_reads = 0
    box_coordinate_comparisons = 0
    changed_lower_coordinates = 0
    changed_upper_coordinates = 0
    delta_coefficient_reads = 0
    delta_integer_operations = 0
    bound_integer_products = 0
    bound_rational_conversions = 0
    lower_flags_saved = 0
    lower_flag_comparisons = 0
    guard_calls = 0
    zero_delta_skips = 0
    variable_visits = 0
    guard_box_coordinate_reads = 0
    own_fees = {name: 0 for name in NATIVE_OWN_TIGHT_COUNTERS}
    own_events = []
    roots_initialized = False
    guard_box_coordinate_comparisons = 0
    lower_flag_reads = 0
    upper_price_reads = 0
    upper_price_zero_comparisons = 0

    def lower_feasible(region: _AffineRegion) -> bool:
        if any(left > right for left, right in zip(region.lower, region.upper, strict=True)):
            return False
        if sum(region.lower) > lanes:
            return False
        return all(
            sum(region.lower[position] * requirements[index][coordinate]
                for position, index in enumerate(region.eligible)) <= queue[coordinate]
            for coordinate in range(dimension)
        )

    capacity_region_visits = 0
    capacity_requirement_coordinate_reads = 0
    capacity_tightened_bounds = 0
    capacity_removed_count_values = 0

    def tighten_capacity(region: _AffineRegion) -> _AffineRegion:
        nonlocal capacity_region_visits, capacity_requirement_coordinate_reads
        nonlocal capacity_tightened_bounds, capacity_removed_count_values
        capacity_region_visits += 1
        local_requirements = tuple(requirements[index] for index in region.eligible)
        capacity_requirement_coordinate_reads += 2 * len(local_requirements) * dimension
        upper = _integer_capacity_upper_bounds(
            region.lower, region.upper, local_requirements, queue, lanes)
        capacity_tightened_bounds += sum(new < old for new, old in zip(upper, region.upper, strict=True))
        capacity_removed_count_values += sum(old - new for new, old in zip(upper, region.upper, strict=True))
        if upper != region.upper:
            own_fees["own_bound_cache_reads"] += 1
            own_fees["own_bound_cache_writes"] += 1
            return replace(region, upper=upper)
        return region

    def may_attain_threshold(region: _AffineRegion) -> bool:
        return any(region.upper[position] >= 1 for position in region.exact_duration_positions)

    def analytic_root_bound(eligible: tuple[int, ...], threshold: Fraction) -> Fraction:
        weighted_lane_cap = lanes * max((weights[index] for index in eligible), default=0)
        visible_queue_cap = sum(value * value for value in queue)
        return Fraction(min(weighted_lane_cap, visible_queue_cap)) / threshold

    def residual_candidate(region: _AffineRegion, primal: tuple[float, ...] | None) -> None:
        nonlocal residual_fill_option_probes
        if primal is None:
            counts = list(region.lower)
        else:
            counts = []
            for value, left, right in zip(primal, region.lower, region.upper, strict=True):
                bounded = min(max(float(value), float(left)), float(right))
                counts.append(max(left, min(right, math.floor(bounded))))
        if not lower_feasible(_AffineRegion(
            region.threshold_index, region.threshold, region.exact_duration_positions,
            region.eligible, tuple(counts), region.upper, region.bound, region.solved,
            region.primal_counts,
        )):
            counts = list(region.lower)
        lane_left = lanes - sum(counts)
        residual = [queue[coordinate] - sum(
            counts[position] * requirements[index][coordinate]
            for position, index in enumerate(region.eligible)) for coordinate in range(dimension)]
        for position, index in enumerate(region.eligible):
            residual_fill_option_probes += 1
            cap = min(region.upper[position] - counts[position], lane_left)
            if cap <= 0:
                continue
            requirement = requirements[index]
            queue_cap = min((residual[coordinate] // amount
                             for coordinate, amount in enumerate(requirement) if amount), default=cap)
            copies = min(cap, queue_cap)
            if copies <= 0:
                continue
            counts[position] += copies
            lane_left -= copies
            for coordinate, amount in enumerate(requirement):
                residual[coordinate] -= copies * amount
        consider_counts(region.eligible, counts)

    def solve_region(region: _AffineRegion) -> _AffineRegion:
        nonlocal lp_calls, successful_lp_calls, dual_certified_lp_calls
        nonlocal safe_fallback_lp_calls, dual_repair_count
        nonlocal maximum_dual_repair, lp_matrix_coefficients
        nonlocal dual_column_checks, certificates_saved, certificate_numerator_products
        nonlocal lower_flags_saved, lower_flag_comparisons
        lp_calls += 1
        eligible = region.eligible
        variable_count = len(eligible)
        base_objective = sum(region.lower[position] * weights[index]
                             for position, index in enumerate(eligible))
        lane_rhs = lanes - sum(region.lower)
        resource_rhs = [queue[coordinate] - sum(
            region.lower[position] * requirements[index][coordinate]
            for position, index in enumerate(eligible)) for coordinate in range(dimension)]
        upper_rhs = [right - left for left, right in zip(region.lower, region.upper, strict=True)]
        matrix: list[list[int]] = [[1] * variable_count]
        matrix.extend([[requirements[index][coordinate] for index in eligible]
                       for coordinate in range(dimension)])
        matrix.extend([[int(column == row) for column in range(variable_count)]
                       for row in range(variable_count)])
        rhs = [lane_rhs, *resource_rhs, *upper_rhs]
        objective = [weights[index] for index in eligible]
        coefficients = len(matrix) * variable_count
        lp_matrix_coefficients += coefficients
        result = None
        try:
            result = linprog([-float(value) for value in objective], A_ub=matrix, b_ub=rhs,
                             bounds=[(0.0, None)] * variable_count, method="highs")
        except Exception:
            result = None
        primal = None
        bound = region.bound
        own_bound = region.own_bound
        own_fees["own_bound_cache_reads"] += 1
        dual_prices = region.dual_prices
        certificate = region.certificate
        dual_lower, dual_upper, dual_numerator = region.dual_lower, region.dual_upper, region.dual_numerator
        solver_success = bool(result is not None and result.success)
        dual_certified = False
        repair = Fraction(0)
        if solver_success:
            successful_lp_calls += 1
            try:
                primal = tuple(float(value) + region.lower[position]
                               for position, value in enumerate(result.x))
                marginals = tuple(result.ineqlin.marginals)
                dual_bound, repair, prices, lower_zero_flags = _zero_delta_dual_upper_with_prices(
                    matrix, rhs, objective, base_objective, region.threshold, marginals)
                bound = min(bound, dual_bound)
                own_bound = dual_bound
                dual_prices = prices
                # The new dual may be weaker than region.bound; preserve its
                # own numerator instead of scaling the inherited minimum.
                certificate = _ZeroDeltaCertificate(
                    prices, region.lower, region.upper, dual_bound * region.threshold,
                    lower_zero_flags=lower_zero_flags)
                lower_flags_saved += variable_count
                lower_flag_comparisons += variable_count
                dual_lower = dual_upper = dual_numerator = None
                certificate_numerator_products += 1
                certificates_saved += 1
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
        own_fees["own_bound_cache_writes"] += 1
        return _AffineRegion(region.threshold_index, region.threshold,
                           region.exact_duration_positions, region.eligible,
                           region.lower, region.upper, bound, True, primal, dual_prices,
                           certificate, dual_lower, dual_upper, dual_numerator, own_bound)

    frontier: list[tuple[Fraction, int, tuple[int, ...], tuple[int, ...], int, _AffineRegion]] = []
    serial = 0
    frontier_peak = 0
    bound_pruned_regions = 0
    canonical_key_bound_evaluations = 0
    canonical_tie_pruned_regions = 0
    canonical_tie_singletons_resolved = 0
    analytic_infeasible_regions = 0
    threshold_empty_regions = 0

    def region_canonical_key_lower_bound(region: _AffineRegion) -> tuple[Fraction, tuple[int, ...]]:
        counts = list(region.lower)
        remaining = lanes - sum(counts)
        for position in range(len(counts)):
            if remaining <= 0:
                break
            copies = min(region.upper[position] - counts[position], remaining)
            counts[position] += copies
            remaining -= copies
        indices = tuple(representatives[region.threshold_index][index]
                        for position, index in enumerate(region.eligible)
                        for _ in range(counts[position]))
        encoded = indices + (option_count,) * (lanes - len(indices))
        return region.threshold, encoded

    def region_is_prunable(region: _AffineRegion, *, account: bool) -> bool:
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

    def shrink_own_tight(region: _AffineRegion) -> _AffineRegion | None:
        nonlocal analytic_infeasible_regions, threshold_empty_regions, bound_pruned_regions
        own_fees["checks"] += 1
        own_fees["bound_comparisons"] += 1
        if region.bound != best.score:
            return region
        own_fees["certificate_reads"] += 1
        certificate = region.certificate
        if certificate is None:
            return region
        own_fees["own_bound_reads"] += 1
        own_fees["own_bound_comparisons"] += 1
        own_bound = region.own_bound
        if own_bound != best.score:
            return region
        size = len(region.eligible)
        own_fees["tight_regions"] += 1
        own_fees["variable_visits"] += size
        own_fees["lower_flag_reads"] += size
        own_fees["upper_price_reads"] += size
        own_fees["upper_price_comparisons"] += size
        own_fees["box_coordinate_reads"] += 2 * size
        own_fees["box_coordinate_comparisons"] += size
        own_fees["bound_values_copied"] += 4 * size
        face_lower, face_upper, changed_lower, changed_upper, empty = _own_tight_endpoints(
            region.lower, region.upper, certificate.lower_zero_flags,
            certificate.prices, 1 + dimension)
        own_fees["bound_values_written"] += changed_lower + changed_upper
        if not changed_lower and not changed_upper:
            return region
        own_fees["changed_regions"] += 1
        own_fees["changed_lower_coordinates"] += changed_lower
        own_fees["changed_upper_coordinates"] += changed_upper
        before = region
        own_fees["own_bound_cache_reads"] += 1
        own_fees["own_bound_cache_writes"] += 1
        region = replace(region, lower=face_lower, upper=face_upper)
        disposition = "kept"
        if empty:
            analytic_infeasible_regions += 1
            own_fees["empty_regions"] += 1
            disposition = "direct_empty"
        else:
            if changed_lower:
                own_fees["lower_feasibility_rechecks"] += 1
                if not lower_feasible(region):
                    analytic_infeasible_regions += 1
                    own_fees["empty_regions"] += 1
                    disposition = "infeasible_lower"
                else:
                    own_fees["capacity_rechecks"] += 1
                    region = tighten_capacity(region)
        if disposition == "kept":
            own_fees["threshold_rechecks"] += 1
            if not may_attain_threshold(region):
                threshold_empty_regions += 1
                own_fees["empty_regions"] += 1
                disposition = "threshold_empty"
            else:
                own_fees["canonical_rechecks"] += 1
                if region_is_prunable(region, account=True):
                    bound_pruned_regions += 1
                    disposition = "canonical_pruned"
        # Nonempty fixing only changes zero-coefficient endpoints. Every
        # positive cap has already become a fixed count, so a feasible lower
        # vector makes any subsequent capacity reductions have cap price zero.
        # The cached current own objective therefore remains exact.
        eligible = tuple(index for index in before.eligible)
        local_requirements = tuple(tuple(value for value in requirements[index]) for index in eligible)
        exact = tuple(position for position in before.exact_duration_positions)
        prices = tuple((value.numerator, value.denominator) for value in certificate.prices)
        source_lower = tuple(value for value in certificate.lower)
        source_upper = tuple(value for value in certificate.upper)
        before_lower = tuple(value for value in before.lower)
        before_upper = tuple(value for value in before.upper)
        if before.dual_numerator is None:
            anchor_kind = "certificate"
            anchor_lower, anchor_upper = source_lower, source_upper
            anchor_value = certificate.numerator
        else:
            anchor_kind = "affine"
            anchor_lower = tuple(value for value in before.dual_lower)
            anchor_upper = tuple(value for value in before.dual_upper)
            anchor_value = Fraction(before.dual_numerator, certificate.prepared.denominator)
            own_fees["event_rational_operations"] += 1
            own_fees["event_bound_values_copied"] += 2 * size
        own_value = own_bound * before.threshold
        target_value = best.score * before.threshold
        own_fees["event_records_written"] += 1
        own_fees["event_group_reads"] += 2 * size
        own_fees["event_requirement_reads"] += size * dimension
        own_fees["event_price_reads"] += len(prices)
        own_fees["event_bound_values_copied"] += 4 * size
        own_fees["event_exact_position_reads"] += len(exact)
        own_fees["event_fraction_pairs_written"] += len(prices) + 9
        own_fees["event_rational_operations"] += 2
        own_events.append(NativeOwnTightEvent(
            (before.threshold.numerator, before.threshold.denominator),
            (before.bound.numerator, before.bound.denominator),
            (best.score.numerator, best.score.denominator),
            (best.duration.numerator, best.duration.denominator), best.counts, best.indices,
            eligible, local_requirements, exact, prices, source_lower, source_upper,
            (certificate.numerator.numerator, certificate.numerator.denominator), anchor_kind,
            anchor_lower, anchor_upper, (anchor_value.numerator, anchor_value.denominator),
            (own_value.numerator, own_value.denominator), (target_value.numerator, target_value.denominator),
            (own_bound.numerator, own_bound.denominator), before_lower, before_upper,
            face_lower, face_upper, region.lower, region.upper, disposition))
        return region if disposition == "kept" else None

    def push(region: _AffineRegion) -> None:
        nonlocal serial, frontier_peak, bound_pruned_regions
        nonlocal analytic_infeasible_regions, threshold_empty_regions
        nonlocal child_bound_evaluations, child_bound_improvements, child_bound_prunes
        nonlocal certificates_prepared, price_reads, denominator_lcm_steps, integer_scalings
        nonlocal coefficient_requirement_reads, coefficient_weight_reads, coefficient_integer_operations
        nonlocal bound_variable_visits, box_coordinate_reads, box_coordinate_comparisons
        nonlocal changed_lower_coordinates, changed_upper_coordinates
        nonlocal delta_coefficient_reads, delta_integer_operations
        nonlocal bound_integer_products, bound_rational_conversions
        nonlocal guard_calls, zero_delta_skips, variable_visits
        nonlocal guard_box_coordinate_reads, guard_box_coordinate_comparisons
        nonlocal lower_flag_reads, upper_price_reads, upper_price_zero_comparisons
        if not lower_feasible(region):
            analytic_infeasible_regions += 1
            return
        if not region.solved:
            region = tighten_capacity(region)
        if not may_attain_threshold(region):
            threshold_empty_regions += 1
            return
        # A parent's certified bound already closes some child boxes. Their
        # dual RHS arithmetic cannot change whether they enter the frontier.
        if region_is_prunable(region, account=True):
            bound_pruned_regions += 1
            return
        if not region.solved and region.certificate is not None:
            certificate = region.certificate
            if region.dual_numerator is None:
                source_lower, source_upper = certificate.lower, certificate.upper
            else:
                source_lower, source_upper = region.dual_lower, region.dual_upper
            zero_delta, visits, lower_reads, upper_reads = _zero_delta_guard(
                certificate, source_lower, source_upper, region.lower, region.upper)
            guard_calls += 1
            variable_visits += visits
            guard_box_coordinate_reads += 4 * visits
            guard_box_coordinate_comparisons += 2 * visits
            lower_flag_reads += lower_reads
            upper_price_reads += upper_reads
            upper_price_zero_comparisons += upper_reads
            if zero_delta:
                # Its own numerator is unchanged and its historical certified
                # bound is already at least region.bound. Keep the last real
                # evaluation box so failed LP descendants retain an exact anchor.
                zero_delta_skips += 1
            else:
                variable_count = len(region.eligible)
                child_bound_evaluations += 1
                certificate = region.certificate
                coefficients = certificate.prepared
                if coefficients is None:
                    certificates_prepared += 1
                    price_count = 1 + dimension + variable_count
                    price_reads += 2 * price_count
                    denominator_lcm_steps += price_count + 1
                    integer_scalings += price_count + variable_count + 1
                    coefficient_requirement_reads += variable_count * dimension
                    coefficient_weight_reads += variable_count
                    coefficient_integer_operations += variable_count * (2 * dimension + 3)
                    coefficients = _prepare_affine_certificate(
                        certificate, tuple(requirements[index] for index in region.eligible),
                        tuple(weights[index] for index in region.eligible))
                    certificate.prepared = coefficients
                if region.dual_numerator is None:
                    numerator = coefficients.anchor_numerator
                    source_lower, source_upper = certificate.lower, certificate.upper
                else:
                    numerator = region.dual_numerator
                    source_lower, source_upper = region.dual_lower, region.dual_upper
                numerator, changed_lower, changed_upper = _update_affine_numerator(
                    coefficients, numerator, source_lower, source_upper, region.lower, region.upper)
                bound_variable_visits += variable_count
                box_coordinate_reads += 4 * variable_count
                box_coordinate_comparisons += 2 * variable_count
                changed_lower_coordinates += changed_lower
                changed_upper_coordinates += changed_upper
                changed = changed_lower + changed_upper
                delta_coefficient_reads += changed
                delta_integer_operations += 3 * changed
                bound_integer_products += 2
                bound_rational_conversions += 1
                dual_bound = Fraction(numerator * region.threshold.denominator,
                                      coefficients.denominator * region.threshold.numerator)
                inherited_bound = region.bound
                own_fees["own_bound_cache_writes"] += 1
                region = replace(region, bound=min(inherited_bound, dual_bound),
                                 dual_lower=region.lower, dual_upper=region.upper, dual_numerator=numerator,
                                 own_bound=dual_bound)
                if dual_bound < inherited_bound:
                    child_bound_improvements += 1
                    if region_is_prunable(region, account=True):
                        bound_pruned_regions += 1
                        child_bound_prunes += 1
                        return
        if roots_initialized:
            region = shrink_own_tight(region)
            if region is None:
                return
        heapq.heappush(frontier, (-region.bound, region.threshold_index, region.lower,
                                 region.upper, serial, region))
        serial += 1
        frontier_peak = max(frontier_peak, len(frontier))

    roots_seen = 0
    roots_solved = 0
    roots_skipped_empty = 0
    roots_skipped_bound = 0
    roots_skipped_canonical_tie = 0
    for threshold_index, threshold in enumerate(thresholds):
        eligible = tuple(sorted(representatives[threshold_index],
                                key=representatives[threshold_index].__getitem__))
        exact_positions = tuple(position for position, index in enumerate(eligible)
                                if durations[index] == threshold)
        root_bound = analytic_root_bound(eligible, threshold)
        root = _AffineRegion(threshold_index, threshold, exact_positions, eligible,
                           (0,) * len(eligible), (lanes,) * len(eligible), root_bound, False, None)
        roots_seen += 1
        root = tighten_capacity(root)
        if not may_attain_threshold(root):
            roots_skipped_empty += 1
            threshold_empty_regions += 1
            continue
        if region_is_prunable(root, account=True):
            roots_skipped_bound += 1
            roots_skipped_canonical_tie += int(root.bound == best.score)
            bound_pruned_regions += 1
            continue
        roots_solved += 1
        push(solve_region(root))

    branch_splits = 0
    singleton_regions_closed = 0
    singleton_candidate_evaluations = 0

    def branch(region: _AffineRegion) -> tuple[_AffineRegion, _AffineRegion] | None:
        position = _branch_position(region)
        if position is None:
            return None
        left_limit, right_limit = region.lower[position], region.upper[position]
        if region.primal_counts is None:
            cut = (left_limit + right_limit) // 2
        else:
            proposed = math.floor(region.primal_counts[position])
            cut = proposed if left_limit <= proposed < right_limit else (left_limit + right_limit) // 2
        lower_left = list(region.lower)
        upper_left = list(region.upper)
        upper_left[position] = cut
        lower_right = list(region.lower)
        upper_right = list(region.upper)
        lower_right[position] = cut + 1
        common = (region.threshold_index, region.threshold,
                  region.exact_duration_positions, region.eligible)
        own_bound = region.own_bound
        own_fees["own_bound_cache_reads"] += 1
        own_fees["own_bound_cache_writes"] += 2
        return (_AffineRegion(*common, tuple(lower_left), tuple(upper_left), region.bound,
                            False, None, region.dual_prices, region.certificate,
                            region.dual_lower, region.dual_upper, region.dual_numerator, own_bound),
                _AffineRegion(*common, tuple(lower_right), tuple(upper_right), region.bound,
                            False, None, region.dual_prices, region.certificate,
                            region.dual_lower, region.dual_upper, region.dual_numerator, own_bound))

    # These values cease changing after root initialization. Their actual
    # construction stays inside native initialization and the whole call.
    threshold_values = tuple(float(threshold) for threshold in thresholds)
    root_work = RootPruningWork(roots_seen, roots_solved, roots_skipped_empty,
        roots_skipped_bound, roots_skipped_canonical_tie)
    proxy_option_count = sum(len(indices) for indices in representatives)
    queue_norm = sum(queue)
    represented_ordered = (option_count + 1) ** lanes
    represented_counts = math.comb(option_count + lanes, lanes)

    def snapshot(lp_node_budget: int) -> CanonicalCountSpaceBnbSelection:
        upper_exact = best.score
        retained_count = canonical_ties_remaining = inherited_unsolved = 0
        for entry in frontier:
            region = entry[-1]
            if region.bound > upper_exact:
                upper_exact = region.bound
            if not region_is_prunable(region, account=False):
                retained_count += 1
                canonical_ties_remaining += region.bound == best.score
                inherited_unsolved += not region.solved
        snapshot_bound_pruned = bound_pruned_regions + len(frontier) - retained_count
        score = float(best.score)
        upper = score if upper_exact == best.score else _float_up(upper_exact)
        gap = max(0.0, upper - score)
        score_exact = upper_exact == best.score
        canonical_exact = score_exact and not retained_count
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
            frontier_regions_remaining=retained_count,
            inherited_unsolved_regions=inherited_unsolved,
            frontier_peak=frontier_peak,
            lp_matrix_coefficients=lp_matrix_coefficients,
            dual_column_checks=dual_column_checks,
            represented_ordered_configurations=represented_ordered,
            represented_count_vectors=represented_counts,
        )

    def checkpoint(lp_budget):
        event_values = tuple(own_events)
        own_fees["event_snapshot_entries_copied"] += len(event_values)
        return DirectProbeOwnTightNativeThresholdSelection(
            selection=snapshot(lp_budget),
            preprocessing=ThresholdCatalogueWork(
                original_option_count=option_count,
                requirement_group_count=len(groups),
                duration_threshold_count=root_count,
                reduction_option_probes=reduction.reduction_option_probes,
                reduction_requirement_coordinate_reads=reduction.reduction_requirement_coordinate_reads,
                reduction_duration_comparisons=reduction.reduction_duration_comparisons,
                grouping_option_probes=option_count,
                representative_option_probes=representative_probes,
                proxy_option_count=proxy_option_count,
                lift_option_probes=lift_probes,
            ),
            thresholds=threshold_values,
            capacity_propagation=CapacityPropagationWork(
                region_visits=capacity_region_visits,
                requirement_coordinate_reads=capacity_requirement_coordinate_reads,
                tightened_bounds=capacity_tightened_bounds,
                removed_count_values=capacity_removed_count_values,
            ),
            root_pruning=root_work,
            dual_reuse=DualReuseWork(
                certificates_saved=certificates_saved,
                child_bound_evaluations=child_bound_evaluations,
                child_bound_improvements=child_bound_improvements,
                child_bound_prunes=child_bound_prunes,
                child_rhs_entries=child_rhs_entries,
                child_requirement_reads=child_requirement_reads,
                child_weight_reads=child_weight_reads,
            ),
            affine_dual=AffineDualWork(
                certificates_prepared=certificates_prepared,
                certificate_numerator_products=certificate_numerator_products,
                price_reads=price_reads,
                denominator_lcm_steps=denominator_lcm_steps,
                integer_scalings=integer_scalings,
                coefficient_requirement_reads=coefficient_requirement_reads,
                coefficient_weight_reads=coefficient_weight_reads,
                coefficient_integer_operations=coefficient_integer_operations,
                bound_variable_visits=bound_variable_visits,
                box_coordinate_reads=box_coordinate_reads,
                box_coordinate_comparisons=box_coordinate_comparisons,
                changed_lower_coordinates=changed_lower_coordinates,
                changed_upper_coordinates=changed_upper_coordinates,
                delta_coefficient_reads=delta_coefficient_reads,
                delta_integer_operations=delta_integer_operations,
                bound_integer_products=bound_integer_products,
                bound_rational_conversions=bound_rational_conversions,
            ),
            zero_delta=ZeroDeltaWork(
                lower_flags_saved=lower_flags_saved,
                lower_flag_comparisons=lower_flag_comparisons,
                guard_calls=guard_calls,
                zero_delta_skips=zero_delta_skips,
                variable_visits=variable_visits,
                box_coordinate_reads=guard_box_coordinate_reads,
                box_coordinate_comparisons=guard_box_coordinate_comparisons,
                lower_flag_reads=lower_flag_reads,
                upper_price_reads=upper_price_reads,
                upper_price_zero_comparisons=upper_price_zero_comparisons,
            ),
            native_own_tight_work=NativeOwnTightWork(**own_fees),
            native_own_tight_evidence=event_values,
        )

    # Freeze raw-M evidence before the phase mutates incumbent, heap or work.
    root = checkpoint(root_count)
    roots_initialized = True
    settled = work = None
    if root.selection.score_exact and not root.selection.canonical_exact:
        head_checks = unsolved_prune_checks = popped = pruned = splits = singletons = 0
        stopped_for_lp = stopped_for_branch = 0
        lp_before = lp_calls
        frontier_before = len(frontier)
        while frontier:
            head_checks += 1
            region = frontier[0][-1]
            if not region.solved:
                # This extra peek never increments native canonical work;
                # normal popping/resuming still performs account=True once.
                unsolved_prune_checks += 1
                if not region_is_prunable(region, account=False):
                    stopped_for_lp = 1
                    break
            entry = heapq.heappop(frontier)
            region = entry[-1]
            popped += 1
            if region_is_prunable(region, account=True):
                bound_pruned_regions += 1
                pruned += 1
                continue
            region = shrink_own_tight(region)
            if region is None:
                pruned += 1
                continue
            if _branch_position(region) is not None:
                # Keep the exact remaining domain, with its original serial.
                # Fixing may have changed box keys, so use the current bounds.
                stopped_for_branch = 1
                if region is entry[-1]:
                    heapq.heappush(frontier, entry)
                else:
                    heapq.heappush(frontier, (-region.bound, region.threshold_index,
                        region.lower, region.upper, entry[4], region))
                break
            else:
                singleton_regions_closed += 1
                singletons += 1
                actual_threshold = max((idle_fraction, *(durations[index]
                    for position, index in enumerate(region.eligible) if region.lower[position])))
                if actual_threshold == region.threshold:
                    singleton_candidate_evaluations += 1
                    incumbent_before = best
                    consider_counts(region.eligible, region.lower)
                    if best is not incumbent_before and best.score == incumbent_before.score:
                        canonical_tie_singletons_resolved += 1
                continue
        raw_heap_count = len(frontier)
        settled = checkpoint(root_count)
        work = RootSettleWork(1, head_checks, unsolved_prune_checks, popped, pruned, splits,
            singletons, stopped_for_lp, 1, raw_heap_count,
            0, lp_before, lp_calls,
            frontier_before, raw_heap_count, stopped_for_branch)
    previous_snapshot = settled if settled is not None else root
    budget = yield root, settled, work
    while True:
        while frontier and lp_calls + (roots_seen - roots_solved) < budget:
            _, _, _, _, _, region = heapq.heappop(frontier)
            if region_is_prunable(region, account=True):
                bound_pruned_regions += 1
                continue
            region = shrink_own_tight(region)
            if region is None:
                continue
            if not region.solved:
                push(solve_region(region))
                continue
            children = branch(region)
            if children is None:
                singleton_regions_closed += 1
                actual_threshold = max((idle_fraction, *(durations[index]
                    for position, index in enumerate(region.eligible) if region.lower[position])))
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
        previous_snapshot = checkpoint(budget)
        budget = yield previous_snapshot
