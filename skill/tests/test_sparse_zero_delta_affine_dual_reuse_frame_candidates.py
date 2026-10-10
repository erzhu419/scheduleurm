"""Check accumulated sparse deltas, actual fees and unchanged exact paths."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.sparse_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as engine
import algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as previous
from algorithm.dual_reuse_integer_capacity_frame_candidates import _reuse_dual_upper
from algorithm.frontier_score_proof_certificate_routed_frame_candidates import capture_strict_frontier
from algorithm.sparse_zero_delta_affine_dual_residual_capacity_frame_candidates import select_sparse_zero_delta_affine_dual_residual_capacity_frontier_frame
from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import select_zero_delta_affine_dual_residual_capacity_frontier_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (Option((3, 2), 1.0), Option((2, 0), 1.0))
QUEUE, LANES = (4, 3), 3


def behavior(record):
    if isinstance(record, dict):
        return {name: behavior(value) for name, value in record.items()
                if name not in ("affine_dual", "zero_delta", "sparse_delta")}
    if isinstance(record, (tuple, list)):
        return [behavior(value) for value in record]
    return record


def assert_fees(stage, dimension):
    sparse, guard, affine, dual = stage.sparse_delta, stage.zero_delta, stage.affine_dual, stage.dual_reuse
    assert all(type(value) is int and value >= 0 for vector in (sparse, guard, affine, dual) for value in asdict(vector).values())
    assert guard.lower_flags_saved == guard.lower_flag_comparisons == stage.selection.dual_column_checks
    assert guard.guard_calls == guard.zero_delta_skips + dual.child_bound_evaluations
    assert guard.box_coordinate_reads == guard.box_coordinate_comparisons == 0
    assert guard.lower_flag_reads <= guard.variable_visits and guard.upper_price_reads <= guard.variable_visits
    assert guard.upper_price_reads == guard.upper_price_zero_comparisons
    assert sparse.guard_delta_reads == 3 * guard.variable_visits
    assert sparse.guard_delta_comparisons == 2 * guard.variable_visits
    assert affine.box_coordinate_reads == affine.box_coordinate_comparisons == 0
    assert sparse.update_delta_reads == 3 * affine.bound_variable_visits
    assert sparse.update_delta_comparisons == 2 * affine.bound_variable_visits
    changed = affine.changed_lower_coordinates + affine.changed_upper_coordinates
    assert affine.delta_coefficient_reads == changed
    assert affine.delta_integer_operations == 2 * changed
    assert affine.bound_integer_products == 2 * dual.child_bound_evaluations
    assert affine.bound_rational_conversions == dual.child_bound_evaluations
    assert sparse.anchor_resets == dual.certificates_saved + dual.child_bound_evaluations
    assert sparse.anchor_entries_discarded >= affine.bound_variable_visits
    collected = sparse.branch_changes + sparse.capacity_changes
    assert sparse.pending_merge_writes == collected
    assert sparse.pending_merge_reads == collected + sparse.pending_sort_entries
    assert sparse.pending_integer_operations == 3 * collected
    assert sparse.pending_sort_calls == sparse.pending_maps_created
    assert sparse.pending_entries_copied >= sparse.pending_sort_entries
    assert sparse.pending_maps_created <= sparse.pending_sort_entries
    assert sparse.branch_changes <= 2 * stage.selection.branch_splits
    assert sparse.capacity_changes <= stage.capacity_propagation.tightened_bounds
    prepared, variables = affine.certificates_prepared, affine.coefficient_weight_reads
    assert affine.certificate_numerator_products == dual.certificates_saved
    assert affine.price_reads == 2 * ((1 + dimension) * prepared + variables)
    assert affine.denominator_lcm_steps == (2 + dimension) * prepared + variables
    assert affine.integer_scalings == (2 + dimension) * prepared + 2 * variables
    assert affine.coefficient_requirement_reads == dimension * variables
    assert affine.coefficient_integer_operations == (2 * dimension + 3) * variables


def oracle_key(options, queue, lanes):
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        active = [options[index] for index in indices if index < len(options)]
        service = tuple(sum(option.requirements[c] for option in active) for c in range(len(queue)))
        if any(done > waiting for done, waiting in zip(service, queue)):
            continue
        duration = max((Fraction(1), *(Fraction.from_float(option.duration) for option in active)))
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        keys.append((-score, duration, indices))
    return min(keys)


def selected_key(selection, queue, option_count):
    duration = Fraction.from_float(selection.duration)
    return (-Fraction(sum(q * s for q, s in zip(queue, selection.service))) / duration, duration,
            tuple(option_count if index is None else index for index in selection.option_indices))


def fractional_root(upper_price=6.0):
    return SimpleNamespace(success=True, x=(1.0, 0.5),
                           ineqlin=SimpleNamespace(marginals=(0.0, -4.0, 0.0, -upper_price, 0.0)))


def test_merging_repeated_branch_and_capacity_changes_retains_the_exact_anchor():
    pending = engine._merge_pending_deltas((), ((1, 0, -2),))
    pending = engine._merge_pending_deltas(pending, ((0, 1, 0), (1, 0, -1), (2, 0, -2)))
    pending = engine._merge_pending_deltas(pending, ((0, 1, -1),))
    assert pending == ((0, 2, -1), (1, 0, -3), (2, 0, -2))
    assert engine._merge_pending_deltas((), ((1, 0, -2),)) == ((1, 0, -2),)


def test_sparse_guard_and_complete_update_match_full_box_arithmetic():
    requirements, queue, lanes, threshold = ((3, 1), (1, 2), (2, 3)), (8, 8), 4, Fraction(5, 4)
    weights = tuple(sum(q * r for q, r in zip(queue, row)) for row in requirements)
    prices = (Fraction(99, 8), Fraction(15, 4), Fraction(25, 8),
              Fraction(25, 2), Fraction(35, 4), Fraction(85, 4))
    lower, upper = (0, 0, 0), (4, 4, 4)
    numerator = _reuse_dual_upper(lower, upper, requirements, queue, weights, lanes, threshold, prices) * threshold
    flags = tuple(weights[j] == prices[0] + sum(amount * prices[c + 1] for c, amount in enumerate(row)) + prices[3 + j]
                  for j, row in enumerate(requirements))
    certificate = engine._ZeroDeltaCertificate(prices, lower, upper, numerator, lower_zero_flags=flags)
    coefficients = engine._prepare_affine_certificate(certificate, requirements, weights)
    pending = engine._merge_pending_deltas((), ((0, 1, -3), (1, 0, -2), (2, 0, -3)))
    new_lower, new_upper = (1, 0, 0), (1, 2, 1)
    sparse = engine._sparse_zero_delta_guard(certificate, pending)
    full = previous._zero_delta_guard(certificate, lower, upper, new_lower, new_upper)
    assert sparse[0] == full[0] and sparse[2:] == full[2:]
    assert not sparse[0] and sparse[1] == 1
    actual, changed_lower, changed_upper = engine._update_sparse_affine_numerator(coefficients, coefficients.anchor_numerator, pending)
    assert (changed_lower, changed_upper) == (1, 3)
    expected = _reuse_dual_upper(new_lower, new_upper, requirements, queue, weights, lanes, threshold, prices)
    assert Fraction(actual * threshold.denominator, coefficients.denominator * threshold.numerator) == expected


def test_zero_skip_keeps_pending_changes_across_failed_lp_then_nonzero_fallback(monkeypatch):
    calls, guards = [], []
    real_guard = engine._sparse_zero_delta_guard

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        return fractional_root() if len(calls) == 1 else SimpleNamespace(success=False)

    def guard(certificate, pending):
        result = real_guard(certificate, pending)
        guards.append((pending, result[0]))
        return result

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "_sparse_zero_delta_guard", guard)
    search = engine.SparseZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    initial = search.advance_to(1)
    middle = search.advance_to(2)
    retained = [entry[-1] for entry in search._search.gi_frame.f_locals["frontier"]]
    assert any(region.pending_deltas == ((1, 0, -2),) for region in retained)
    final = search.advance_to(128)
    assert any(not skipped and len(pending) == 2 for pending, skipped in guards)
    assert final.zero_delta.zero_delta_skips >= 2 and final.dual_reuse.child_bound_evaluations >= 1
    assert final.dual_reuse.certificates_saved == 1 and final.selection.canonical_exact
    assert_fees(initial, 2)
    assert_fees(middle, 2)
    assert_fees(final, 2)
    actual_calls = calls[:]
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.ZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES).advance_to(128)
    assert calls == actual_calls and behavior(asdict(final)) == behavior(asdict(old))


def test_successful_child_certificate_clears_pending_changes_and_charges_the_reset(monkeypatch):
    real_lp, calls = engine.linprog, []

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        return fractional_root() if len(calls) == 1 else real_lp(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    final = engine.SparseZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES).advance_to(64)
    assert final.selection.canonical_exact and final.dual_reuse.certificates_saved > 1
    assert final.sparse_delta.anchor_entries_discarded > final.affine_dual.bound_variable_visits
    assert_fees(final, 2)
    actual_calls = calls[:]
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.ZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES).advance_to(64)
    assert calls == actual_calls and behavior(asdict(final)) == behavior(asdict(old))


def test_capacity_changes_outside_branch_coordinate_are_collected_and_used(monkeypatch):
    real_lp, calls = engine.linprog, []

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        return fractional_root() if len(calls) == 1 else real_lp(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    final = engine.SparseZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES).advance_to(2)
    assert final.sparse_delta.capacity_changes >= 1
    assert final.dual_reuse.child_bound_improvements == final.dual_reuse.child_bound_prunes == 1
    assert_fees(final, 2)


def test_continuation_preserves_lp_sequence_and_immutable_cumulative_metadata(monkeypatch):
    real_lp, calls = engine.linprog, []

    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        return real_lp(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    options = (*OPTIONS, Option((3, 2), 1.25), Option((0, 1), 2.0))
    direct = engine.SparseZeroDeltaAffineDualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    direct_calls = calls[:]
    calls.clear()
    search = engine.SparseZeroDeltaAffineDualReuseIntegerCapacitySearch(options, QUEUE, LANES)
    saved, prior = [], None
    for cap in (2, 3, 4, 8, 64):
        final = search.advance_to(cap)
        fee = {name: asdict(getattr(final, name)) for name in ("affine_dual", "zero_delta", "sparse_delta")}
        if prior is not None:
            assert all(fee[group][name] >= prior[group][name] for group in fee for name in fee[group])
        prior = fee
        assert_fees(final, 2)
        assert search.advance_to(cap) is final
        saved.append((final, asdict(final)))
    assert final == direct and calls == direct_calls
    assert all(asdict(stage) == record for stage, record in saved)
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    assert calls == direct_calls and behavior(asdict(final)) == behavior(asdict(old))


@pytest.mark.parametrize("queue", ((4, 3), (5, 2), (3, 1), (5, 4)))
def test_small_integer_enumeration_matches_the_original_canonical_argmax(queue):
    final = engine.SparseZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, queue, LANES).advance_to(128)
    old = previous.ZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, queue, LANES).advance_to(128)
    assert final.selection.canonical_exact
    assert selected_key(final.selection, queue, len(OPTIONS)) == oracle_key(OPTIONS, queue, LANES)
    assert behavior(asdict(final)) == behavior(asdict(old))
    assert_fees(final, 2)


@pytest.mark.parametrize("options,queue,lanes,initial_cap", (
    (OPTIONS, QUEUE, LANES, 1),
    ((Option((1, 0), 2.0), Option((1, 0), 1.0), Option((0, 2), 2.0)), (1, 2), 2, 2),
    (OPTIONS, (0, 0), LANES, 1),
    ((Option((1, 0), 1.0),), (1, 0), 1, 1),
))
def test_whole_router_keeps_all_nonfee_fields_and_original_action(options, queue, lanes, initial_cap):
    new = select_sparse_zero_delta_affine_dual_residual_capacity_frontier_frame(options, queue, lanes, initial_cap, 64, 8192)
    old = select_zero_delta_affine_dual_residual_capacity_frontier_frame(options, queue, lanes, initial_cap, 64, 8192)
    assert behavior(asdict(new)) == behavior(asdict(old))
    assert selected_key(new.selection, queue, len(options)) == oracle_key(options, queue, lanes)
    assert_fees(new.initial_lp, len(queue))
    assert_fees(new.final_lp, len(queue))


def test_snapshot_and_frontier_export_do_not_collect_or_merge_changes(monkeypatch):
    monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: fractional_root())
    search = engine.SparseZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    initial = search.advance_to(1)
    saved = asdict(initial)
    assert capture_strict_frontier(search).regions
    assert search.advance_to(1) is initial and asdict(initial) == saved
    assert initial.sparse_delta.pending_maps_created == initial.zero_delta.guard_calls == 0
