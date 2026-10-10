"""Complete root ties inside the paid boxes and their own exact dual faces.

The root frontier covers every action that can improve the incumbent's
canonical key. At a fixed group-minimum duration, the earliest eligible
original option in each requirement group gives its canonical representative.
Thus boxed group counts retain the original catalogue's canonical optimum.

An inherited minimum is not a dual objective. Each exported certificate is
evaluated from its own source box. Only equality of that objective with the
certified target permits complementary-slackness count fixing. The extra
work counters describe explicit algorithm events, reads and arithmetic, not
the number of Python instructions; the ordinary DFS work stays on completion.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Any, Sequence

from algorithm.early_root_canonical_frame_candidates import (
    EarlyRootCanonicalSelection, _root_budget,
)
from algorithm.frontier_score_proof_certificate_routed_frame_candidates import capture_strict_frontier
from algorithm.residual_capacity_frontier_frame_candidates import _residual_frontier_probe
from algorithm.resumable_probe_frame_candidates import _probe
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import (
    StrictScoreProofCanonicalCompletionSelection, StrictScoreProofWork,
)
from algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates import (
    ZeroDeltaAffineDualReuseIntegerCapacitySearch,
)


@dataclass(frozen=True)
class DualFaceWork:
    capture_calls: int = 0
    heap_entries_read: int = 0
    canonical_box_checks: int = 0
    canonical_key_variable_visits: int = 0
    canonical_key_count_reads: int = 0
    canonical_key_integer_operations: int = 0
    canonical_key_indices_written: int = 0
    canonical_key_comparisons: int = 0
    boxes_exported: int = 0
    group_ids_copied: int = 0
    requirement_coordinates_copied: int = 0
    representative_options_copied: int = 0
    box_bound_values_copied: int = 0
    exact_positions_copied: int = 0
    fraction_pairs_copied: int = 0
    certificate_price_reads: int = 0
    certificate_bound_reads: int = 0
    coefficient_requirement_reads: int = 0
    exact_integer_operations: int = 0
    exact_rational_operations: int = 0
    own_bound_comparisons: int = 0
    coefficient_sign_comparisons: int = 0
    fixed_lower_coordinates: int = 0
    fixed_upper_coordinates: int = 0
    face_bound_values_written: int = 0
    domain_box_visits: int = 0
    domain_bound_reads: int = 0
    domain_sort_calls: int = 0
    domain_sort_entries: int = 0
    candidate_option_visits: int = 0
    candidate_requirement_reads: int = 0
    candidate_key_comparisons: int = 0


DUAL_FACE_COUNTERS = tuple(DualFaceWork.__dataclass_fields__)


@dataclass(frozen=True)
class DualFaceRootBox:
    threshold_fraction: tuple[int, int]
    bound_fraction: tuple[int, int]
    eligible: tuple[int, ...]
    requirements: tuple[tuple[int, ...], ...]
    representative_options: tuple[int, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    exact_duration_positions: tuple[int, ...]
    certificate_prices: tuple[tuple[int, int], ...] | None
    certificate_lower: tuple[int, ...] | None
    certificate_upper: tuple[int, ...] | None
    certificate_numerator_fraction: tuple[int, int] | None
    own_numerator_fraction: tuple[int, int] | None
    target_numerator_fraction: tuple[int, int]
    lower_coefficients: tuple[tuple[int, int], ...]
    upper_coefficients: tuple[tuple[int, int], ...]
    tight: bool
    face_lower: tuple[int, ...]
    face_upper: tuple[int, ...]


@dataclass(frozen=True)
class DualFaceRootEvidence:
    incumbent_score_fraction: tuple[int, int]
    incumbent_duration_fraction: tuple[int, int]
    root_count: int
    raw_region_count: int
    live_region_count: int
    regions: tuple[DualFaceRootBox, ...]


@dataclass(frozen=True)
class DualFaceRootCanonicalSelection(EarlyRootCanonicalSelection):
    dual_face_evidence: DualFaceRootEvidence | None
    dual_face_work: DualFaceWork


def _pair(value):
    return value.numerator, value.denominator


def _capture_root_dual_faces(search, fees):
    """Read the actual heap; never mutate its certificates or LP counters."""
    local = search._search.gi_frame.f_locals
    best, heap = local["best"], local["frontier"]
    requirements, representatives = local["requirements"], local["representatives"]
    weights, lanes, dimension = local["weights"], local["lanes"], local["dimension"]
    option_count = local["option_count"]
    fees["capture_calls"] += 1
    fees["fraction_pairs_copied"] += 2
    incumbent_key = None
    regions = []
    for entry in heap:
        fees["heap_entries_read"] += 1
        fees["canonical_box_checks"] += 1
        region = entry[-1]
        if region.bound < best.score:
            continue
        if region.bound == best.score:
            # This is the frozen native optimistic-key calculation. Its work
            # is additional to the paid root snapshot and is charged here.
            counts = list(region.lower)
            size = len(counts)
            fees["canonical_key_count_reads"] += 2 * size
            fees["canonical_key_integer_operations"] += size + 1
            remaining = lanes - sum(counts)
            for pos in range(size):
                if remaining <= 0:
                    break
                fees["canonical_key_variable_visits"] += 1
                fees["canonical_key_count_reads"] += 2
                fees["canonical_key_integer_operations"] += 3
                copies = min(region.upper[pos] - counts[pos], remaining)
                counts[pos] += copies
                remaining -= copies
            mapping = representatives[region.threshold_index]
            fees["canonical_key_count_reads"] += size
            indices = tuple(mapping[index] for pos, index in enumerate(region.eligible)
                            for _ in range(counts[pos]))
            key = region.threshold, indices + (option_count,) * (lanes - len(indices))
            fees["canonical_key_integer_operations"] += 1
            fees["canonical_key_indices_written"] += lanes
            if incumbent_key is None:
                incumbent_key = best.duration, tuple(option_count if i is None else i for i in best.indices)
                fees["canonical_key_indices_written"] += lanes
            fees["canonical_key_comparisons"] += 1
            if key >= incumbent_key:
                continue

        eligible = tuple(index for index in region.eligible)
        required = tuple(tuple(value for value in requirements[group]) for group in eligible)
        mapping = representatives[region.threshold_index]
        represented = tuple(mapping[group] for group in eligible)
        lower, upper = tuple(value for value in region.lower), tuple(value for value in region.upper)
        exact = tuple(pos for pos in region.exact_duration_positions)
        size = len(eligible)
        fees["boxes_exported"] += 1
        fees["group_ids_copied"] += size
        fees["requirement_coordinates_copied"] += size * dimension
        fees["representative_options_copied"] += size
        fees["box_bound_values_copied"] += 2 * size
        fees["exact_positions_copied"] += len(exact)
        # The mutable working lists and the final evidence tuples each write
        # both bounds. Fixing adds its individual endpoint writes below.
        fees["face_bound_values_written"] += 4 * size
        fees["fraction_pairs_copied"] += 3
        target = best.score * region.threshold
        fees["exact_rational_operations"] += 1
        face_lower, face_upper = list(lower), list(upper)
        certificate = region.certificate
        prices = source_lower = source_upper = source_numerator = own = None
        lower_coefficients, upper_coefficients = (), ()
        tight = False
        if certificate is not None:
            source_lower = tuple(value for value in certificate.lower)
            source_upper = tuple(value for value in certificate.upper)
            prices = tuple(_pair(price) for price in certificate.prices)
            source_numerator = _pair(certificate.numerator)
            fees["certificate_price_reads"] += len(prices)
            fees["certificate_bound_reads"] += 2 * size
            fees["fraction_pairs_copied"] += len(prices) + 2 + 2 * size
            alphas, caps = [], []
            own_value = certificate.numerator
            for pos, group in enumerate(eligible):
                lhs = certificate.prices[0]
                cap = certificate.prices[1 + dimension + pos]
                fees["certificate_price_reads"] += dimension + 2
                for coordinate, amount in enumerate(required[pos]):
                    fees["coefficient_requirement_reads"] += 1
                    lhs += certificate.prices[1 + coordinate] * amount
                    fees["exact_rational_operations"] += 2
                lhs += cap
                alpha = Fraction(weights[group]) - lhs
                fees["exact_rational_operations"] += 3
                alphas.append(alpha)
                caps.append(cap)
                delta_lower = lower[pos] - source_lower[pos]
                delta_upper = upper[pos] - source_upper[pos]
                fees["certificate_bound_reads"] += 2
                fees["exact_integer_operations"] += 2
                own_value += alpha * delta_lower + cap * delta_upper
                fees["exact_rational_operations"] += 4
            own = _pair(own_value)
            lower_coefficients, upper_coefficients = tuple(map(_pair, alphas)), tuple(map(_pair, caps))
            fees["own_bound_comparisons"] += 1
            tight = own_value == target
            if tight:
                for pos, (alpha, cap) in enumerate(zip(alphas, caps, strict=True)):
                    fees["coefficient_sign_comparisons"] += 2
                    if alpha < 0:
                        face_upper[pos] = lower[pos]
                        fees["fixed_lower_coordinates"] += 1
                        fees["face_bound_values_written"] += 1
                    if cap > 0:
                        face_lower[pos] = upper[pos]
                        fees["fixed_upper_coordinates"] += 1
                        fees["face_bound_values_written"] += 1
        regions.append(DualFaceRootBox(
            _pair(region.threshold), _pair(region.bound), eligible, required, represented,
            lower, upper, exact, prices, source_lower, source_upper, source_numerator, own,
            _pair(target), lower_coefficients, upper_coefficients, tight,
            tuple(face_lower), tuple(face_upper)))
    return DualFaceRootEvidence(_pair(best.score), _pair(best.duration), local["root_count"],
                               len(heap), len(regions), tuple(regions))


def _complete_root_dual_faces(options, queue, lanes, baseline, budget, idle_duration, evidence, fees):
    """Search the union of live boxes, including every box at a tied duration."""
    queue, lanes, options = tuple(int(q) for q in queue), int(lanes), tuple(options)
    dimension, option_count = len(queue), len(options)
    score = Fraction(*evidence.incumbent_score_fraction)
    incumbent_duration = Fraction(*evidence.incumbent_duration_fraction)
    idle = Fraction.from_float(float(idle_duration))
    fees["exact_rational_operations"] += 3
    work = {name: 0 for name in (
        "states_visited", "memo_hits", "count_branches", "bound_prunes", "gcd_prunes",
        "duration_prunes", "failed_states_cached", "thresholds_checked", "noninteger_thresholds",
        "suffix_coordinate_reads", "resource_coordinate_reads")}
    selected = baseline

    def key(action):
        return Fraction.from_float(action.duration), tuple(option_count if i is None else i for i in action.option_indices)

    def completed():
        action = replace(selected, upper_bound=selected.score, gap=0.0, epsilon=0.0,
            exact=True, score_exact=True, canonical_exact=True, canonical_tie_regions_remaining=0,
            frontier_regions_remaining=0, inherited_unsolved_regions=0)
        return action, ScoreFaceCompletionWork("COMPLETE", True, True, budget, **work,
            action_changed=action.option_counts != baseline.option_counts or action.duration != baseline.duration,
            score_fraction=_pair(score))

    class StateBudgetExhausted(Exception):
        pass

    def visit():
        if work["states_visited"] == budget:
            raise StateBudgetExhausted
        work["states_visited"] += 1

    fees["domain_sort_calls"] += 1
    fees["domain_sort_entries"] += len(evidence.regions)
    fees["exact_rational_operations"] += len(evidence.regions)
    ordered = sorted(evidence.regions, key=lambda box: Fraction(*box.threshold_fraction))
    best_key = key(selected)
    fees["exact_rational_operations"] += 1
    current_threshold = None
    found_at_threshold = False
    try:
        for box in ordered:
            threshold = Fraction(*box.threshold_fraction)
            fees["exact_rational_operations"] += 1
            if threshold > incumbent_duration:
                break
            if current_threshold != threshold:
                if found_at_threshold:
                    return completed()
                current_threshold = threshold
                work["thresholds_checked"] += 1
            fees["domain_box_visits"] += 1
            numerator = Fraction(*box.target_numerator_fraction)
            fees["exact_rational_operations"] += 1
            if numerator.denominator != 1:
                work["noninteger_thresholds"] += 1
                continue
            target = numerator.numerator
            size, requirements = len(box.eligible), box.requirements
            lower, upper = box.face_lower, box.face_upper
            caps = tuple(right - left for left, right in zip(lower, upper, strict=True))
            fees["domain_bound_reads"] += 2 * size
            fees["exact_integer_operations"] += size
            weights = tuple(sum(q * amount for q, amount in zip(queue, required, strict=True))
                            for required in requirements)
            work["suffix_coordinate_reads"] += size * dimension
            fees["exact_integer_operations"] += 2 * size * dimension
            base_service = tuple(sum(count * required[c] for count, required in zip(lower, requirements, strict=True))
                                 for c in range(dimension))
            fees["domain_bound_reads"] += size * dimension
            fees["exact_integer_operations"] += 2 * size * dimension
            remaining = tuple(q - done for q, done in zip(queue, base_service, strict=True))
            work["resource_coordinate_reads"] += size * dimension + 2 * dimension
            slots = lanes - sum(lower)
            fees["domain_bound_reads"] += size
            fees["exact_integer_operations"] += size + 1 + dimension
            if any(cap < 0 for cap in caps) or slots < 0 or any(r < 0 for r in remaining):
                visit()
                work["bound_prunes"] += 1
                continue
            exact_positions = set(box.exact_duration_positions)
            attained = threshold == idle
            for pos in exact_positions:
                fees["domain_bound_reads"] += 1
                attained |= lower[pos] > 0
            max_weight, divisors, box_weight = [0] * (size + 1), [0] * (size + 1), [0] * (size + 1)
            max_requirement = [(0,) * dimension for _ in range(size + 1)]
            can_attain = [False] * (size + 1)
            for pos in reversed(range(size)):
                cap = caps[pos]
                fees["domain_bound_reads"] += 1
                box_weight[pos] = box_weight[pos + 1] + cap * weights[pos]
                fees["exact_integer_operations"] += 2
                max_weight[pos], divisors[pos] = max_weight[pos + 1], divisors[pos + 1]
                max_requirement[pos], can_attain[pos] = max_requirement[pos + 1], can_attain[pos + 1]
                if cap:
                    max_weight[pos] = max(max_weight[pos], weights[pos])
                    divisors[pos] = math.gcd(divisors[pos], weights[pos])
                    max_requirement[pos] = tuple(max(a, b) for a, b in zip(requirements[pos], max_requirement[pos], strict=True))
                    work["suffix_coordinate_reads"] += 2 * dimension
                    can_attain[pos] |= pos in exact_positions
            failed, choices = set(), {}

            def feasible(pos, free_slots, residual, duration_attained):
                state = pos, free_slots, residual, duration_attained
                if state in failed:
                    work["memo_hits"] += 1
                    return False
                visit()
                needed = target - sum(q * (q - r) for q, r in zip(queue, residual, strict=True))
                work["resource_coordinate_reads"] += dimension
                reason = None
                if needed == 0 and duration_attained:
                    choices[state] = None
                    return True
                if needed < 0 or pos == size or free_slots == 0:
                    reason = "bound_prunes"
                elif not duration_attained and not can_attain[pos]:
                    reason = "duration_prunes"
                else:
                    resource_limit = sum(q * min(r, free_slots * m) for q, r, m in
                                         zip(queue, residual, max_requirement[pos], strict=True))
                    work["resource_coordinate_reads"] += 2 * dimension
                    if needed > min(free_slots * max_weight[pos], resource_limit, box_weight[pos]):
                        reason = "bound_prunes"
                    elif not divisors[pos] or needed % divisors[pos]:
                        reason = "gcd_prunes"
                if reason is not None:
                    work[reason] += 1
                else:
                    fees["domain_bound_reads"] += 1
                    count_limit = min(caps[pos], free_slots, *(r // a for r, a in
                                      zip(residual, requirements[pos], strict=True) if a))
                    work["resource_coordinate_reads"] += 2 * dimension
                    if weights[pos]:
                        count_limit = min(count_limit, needed // weights[pos])
                    for count in range(count_limit, -1, -1):
                        work["count_branches"] += 1
                        child_residual = tuple(r - count * a for r, a in zip(residual, requirements[pos], strict=True))
                        work["resource_coordinate_reads"] += 2 * dimension
                        child = pos + 1, free_slots - count, child_residual, duration_attained or (count > 0 and pos in exact_positions)
                        if feasible(*child):
                            choices[state] = count
                            return True
                failed.add(state)
                work["failed_states_cached"] += 1
                return False

            root = 0, slots, remaining, attained
            if not feasible(*root):
                continue
            local_counts = list(lower)
            fees["domain_bound_reads"] += size
            state = root
            while choices[state] is not None:
                pos, free_slots, residual, duration_attained = state
                count = choices[state]
                local_counts[pos] += count
                state = pos + 1, free_slots - count, tuple(r - count * a for r, a in
                    zip(residual, requirements[pos], strict=True)), duration_attained or (count > 0 and pos in exact_positions)
                work["resource_coordinate_reads"] += 2 * dimension
            counts = [0] * option_count
            for index, count in zip(box.representative_options, local_counts, strict=True):
                counts[index] = count
            service = [0] * dimension
            duration = idle
            for index, option in enumerate(options):
                fees["candidate_option_visits"] += 1
                for coordinate, amount in enumerate(option.requirements):
                    fees["candidate_requirement_reads"] += 1
                    fees["exact_integer_operations"] += 2
                    service[coordinate] += counts[index] * amount
                if counts[index]:
                    duration = max(duration, Fraction.from_float(max(float(idle_duration), float(option.duration))))
                    fees["exact_rational_operations"] += 1
            actual_numerator = sum(q * done for q, done in zip(queue, service, strict=True))
            fees["exact_integer_operations"] += 2 * dimension
            actual_score = Fraction(actual_numerator) / duration
            fees["exact_rational_operations"] += 2
            if (duration != threshold or actual_score != score or sum(counts) > lanes
                    or any(done > q for done, q in zip(service, queue, strict=True))):
                raise ArithmeticError("root face completion contradicts the certified original action")
            indices = tuple(i for i, count in enumerate(counts) for _ in range(count))
            action = replace(baseline, option_counts=tuple(counts), option_indices=indices + (None,) * (lanes - len(indices)),
                             service=tuple(service), duration=float(duration), score=float(actual_score))
            candidate_key = key(action)
            fees["exact_rational_operations"] += 1
            fees["candidate_key_comparisons"] += 1
            if candidate_key < best_key:
                selected, best_key = action, candidate_key
            found_at_threshold = True
        return completed()
    except StateBudgetExhausted:
        return baseline, ScoreFaceCompletionWork("STATE_BUDGET_EXHAUSTED", True, False, budget, **work,
                                                  score_fraction=_pair(score))


def _dual_face_root_probe(options, queue, lanes, stage, budget, idle_duration, evidence, fees):
    selected, completion = _complete_root_dual_faces(options, queue, lanes, stage.selection,
                                                    budget, idle_duration, evidence, fees)
    return StrictScoreProofCanonicalCompletionSelection(
        selected, stage.selection, stage.selection, stage.preprocessing, stage.capacity_propagation,
        StrictScoreProofWork("SKIPPED_SCORE_CERTIFIED", False, True, budget), completion, budget)


def select_dual_face_root_canonical_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> DualFaceRootCanonicalSelection:
    """Use one search, one integer cap, and the frozen 144/256 continuation."""
    for value, name in ((initial_lp_budget, "initial_lp_budget"), (lp_node_budget, "lp_node_budget"),
                        (integer_state_budget, "integer_state_budget")):
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    initial_budget, total_budget, probe_budget = map(int, (initial_lp_budget, lp_node_budget, integer_state_budget))
    if total_budget < initial_budget:
        raise ValueError("total LP budget must cover the initial LP budget")
    options, queue = tuple(options), tuple(queues)
    root_budget, preparation = _root_budget(options, idle_duration)
    if root_budget > initial_budget:
        raise ValueError("initial LP budget must cover all catalogue duration roots")
    search = ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes, idle_duration=idle_duration)
    root = search.advance_to(root_budget)
    raw_heap_count = len(search._search.gi_frame.f_locals["frontier"])
    preparation = replace(preparation, root_snapshot_calls=1,
        root_snapshot_heap_entries_read=2 * raw_heap_count,
        root_snapshot_retained_entries_read=2 * root.selection.frontier_regions_remaining)
    initial, probe, probe_stage, evidence, dual_evidence = root, None, "none", None, None
    fees = {name: 0 for name in DUAL_FACE_COUNTERS}

    def finish(selected, final_lp, resumed, status):
        return DualFaceRootCanonicalSelection(
            selected, root, root_budget, initial, final_lp, probe, resumed, initial_budget, total_budget,
            probe_budget, status, probe_stage, evidence, preparation, dual_evidence, DualFaceWork(**fees))

    if root.selection.canonical_exact:
        return finish(root.selection, root, False, "LP_ROOT_COMPLETE")
    if root.selection.score_exact:
        dual_evidence = _capture_root_dual_faces(search, fees)
        probe = _dual_face_root_probe(options, queue, lanes, root, probe_budget, idle_duration, dual_evidence, fees)
        probe_stage = "root"
        if probe.selection.canonical_exact:
            return finish(probe.selection, root, False, "INTEGER_PROBE_COMPLETE")
    initial = search.advance_to(initial_budget)
    if initial.selection.canonical_exact:
        return finish(initial.selection, initial, False, "LP_INITIAL_COMPLETE")
    if initial.selection.score_exact and probe is None:
        probe = _probe(options, queue, lanes, initial, probe_budget, idle_duration)
        probe_stage = "initial"
        if probe.selection.canonical_exact:
            return finish(probe.selection, initial, False, "INTEGER_PROBE_COMPLETE")
    final_lp = search.advance_to(total_budget)
    selected = final_lp.selection
    if selected.canonical_exact:
        return finish(selected, final_lp, True, "LP_RESUME_COMPLETE")
    if selected.score_exact and probe is None:
        probe = _probe(options, queue, lanes, final_lp, probe_budget, idle_duration)
        probe_stage = "resumed"
        selected = probe.selection
        if selected.canonical_exact:
            return finish(selected, final_lp, True, "INTEGER_PROBE_COMPLETE")
    if not selected.score_exact and probe is None:
        evidence = capture_strict_frontier(search)
        probe = _residual_frontier_probe(options, queue, lanes, final_lp, probe_budget, idle_duration, evidence)
        probe_stage = "frontier_late"
        return finish(probe.selection, final_lp, True,
                      "FRONTIER_PROBE_COMPLETE" if probe.selection.canonical_exact else "FRONTIER_PROBE_INCOMPLETE")
    return finish(selected, final_lp, True, "LP_RESUME_INCOMPLETE")
