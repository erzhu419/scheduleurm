"""Check exact zero support, charged short circuits, and frozen search paths."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from types import SimpleNamespace

import pytest

import algorithm.affine_dual_reuse_integer_capacity_frame_candidates as previous
import algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as engine
from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
from algorithm.dual_reuse_integer_capacity_frame_candidates import _dual_upper_with_prices
from algorithm.frontier_score_proof_certificate_routed_frame_candidates import capture_strict_frontier
from algorithm.integer_capacity_shared_threshold_frame_candidates import _integer_capacity_upper_bounds
from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import select_zero_delta_affine_dual_residual_capacity_frontier_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (Option((3, 2), 1.0), Option((2, 0), 1.0))
QUEUE, LANES = (4, 3), 3
FEE_FIELDS = ("affine_dual", "zero_delta", "child_bound_evaluations")


def behavior(record):
    if isinstance(record, dict):
        return {name: behavior(value) for name, value in record.items() if name not in FEE_FIELDS}
    if isinstance(record, (tuple, list)):
        return [behavior(value) for value in record]
    return record


def assert_fees(stage, dimension):
    guard, fee, dual = stage.zero_delta, stage.affine_dual, stage.dual_reuse
    assert all(value >= 0 for value in asdict(guard).values())
    assert all(value >= 0 for value in asdict(fee).values())
    assert guard.lower_flags_saved == guard.lower_flag_comparisons == stage.selection.dual_column_checks
    assert guard.guard_calls == guard.zero_delta_skips + dual.child_bound_evaluations
    assert guard.box_coordinate_reads == 4 * guard.variable_visits
    assert guard.box_coordinate_comparisons == 2 * guard.variable_visits
    assert guard.lower_flag_reads <= guard.variable_visits
    assert guard.upper_price_reads <= guard.variable_visits
    assert guard.upper_price_zero_comparisons == guard.upper_price_reads
    assert guard.guard_calls <= guard.variable_visits <= guard.guard_calls * stage.preprocessing.requirement_group_count
    assert dual.child_rhs_entries == dual.child_requirement_reads == dual.child_weight_reads == 0
    prepared, variables = fee.certificates_prepared, fee.coefficient_weight_reads
    assert prepared <= dual.certificates_saved and prepared <= dual.child_bound_evaluations
    assert fee.certificate_numerator_products == dual.certificates_saved
    assert fee.price_reads == 2 * ((1 + dimension) * prepared + variables)
    assert fee.denominator_lcm_steps == (2 + dimension) * prepared + variables
    assert fee.integer_scalings == (2 + dimension) * prepared + 2 * variables
    assert fee.coefficient_requirement_reads == dimension * variables
    assert fee.coefficient_integer_operations == (2 * dimension + 3) * variables
    assert fee.box_coordinate_reads == 4 * fee.bound_variable_visits
    assert fee.box_coordinate_comparisons == 2 * fee.bound_variable_visits
    changed = fee.changed_lower_coordinates + fee.changed_upper_coordinates
    assert fee.changed_lower_coordinates <= fee.bound_variable_visits
    assert fee.changed_upper_coordinates <= fee.bound_variable_visits
    assert fee.delta_coefficient_reads == changed
    assert fee.delta_integer_operations == 3 * changed
    assert fee.bound_integer_products == 2 * dual.child_bound_evaluations
    assert fee.bound_rational_conversions == dual.child_bound_evaluations


def fractional_root(upper_price=6.0):
    return SimpleNamespace(success=True, x=(1.0, 0.5),
                           ineqlin=SimpleNamespace(marginals=(0.0, -4.0, 0.0, -upper_price, 0.0)))


@pytest.mark.parametrize("marginals", (
    (0.0, -1.0, 0.0, 0.0),
    (-1.0, 0.0, -2.0, -4.0),
    (0.0, 1.0, 0.0, 0.0),
))
def test_dual_helper_retains_exact_prices_repair_and_bound_and_saves_exact_zero_flags(marginals):
    matrix, rhs, weights = ((1, 1), (1, 2), (1, 0), (0, 1)), (2, 3, 2, 2), (3, 5)
    old = _dual_upper_with_prices(matrix, rhs, weights, 7, Fraction(5, 4), marginals)
    new = engine._zero_delta_dual_upper_with_prices(matrix, rhs, weights, 7, Fraction(5, 4), marginals)
    assert new[:3] == old
    assert new[3] == tuple(sum(Fraction(row[column]) * price for row, price in zip(matrix, new[2])) == weight
                          for column, weight in enumerate(weights))


def test_all_zero_changed_coefficients_skip_affine_preparation_and_keep_canonical_path(monkeypatch):
    calls = []

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        primal = (0.0, 1.0) if len(calls) == 1 else (0.0, 0.0)
        return SimpleNamespace(success=True, x=primal,
                               ineqlin=SimpleNamespace(marginals=(-1.0, 0.0, 0.0, 0.0, 0.0)))

    monkeypatch.setattr(engine, "linprog", lp)
    options = (Option((1, 0), 1.0), Option((0, 1), 1.0))
    search = engine.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, (1, 1), 1)
    initial = search.advance_to(1)
    saved = asdict(initial)
    final = search.advance_to(2)
    assert final.zero_delta.guard_calls == final.zero_delta.zero_delta_skips == 1
    assert final.dual_reuse.child_bound_evaluations == final.affine_dual.certificates_prepared == 0
    assert final.affine_dual.bound_rational_conversions == 0
    assert final.selection.canonical_exact and final.selection.option_indices == (0,)
    assert asdict(initial) == saved
    assert_fees(final, 2)
    new_calls = calls[:]
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.AffineDualReuseIntegerCapacitySearch(options, (1, 1), 1).advance_to(2)
    assert old.dual_reuse.child_bound_evaluations == 1
    assert old.affine_dual.certificates_prepared == 1
    assert calls == new_calls and behavior(asdict(final)) == behavior(asdict(old))


def test_capacity_tightening_of_an_extra_nonzero_upper_coefficient_forces_fallback():
    requirements, weights = ((3, 2), (2, 0)), (18, 8)
    matrix = ((1, 1), (3, 2), (2, 0), (1, 0), (0, 1))
    bound, _, prices, flags = engine._zero_delta_dual_upper_with_prices(
        matrix, (3, 4, 3, 1, 2), weights, 0, Fraction(1), fractional_root().ineqlin.marginals)
    certificate = engine._ZeroDeltaCertificate(prices, (0, 0), (1, 2), bound, lower_zero_flags=flags)
    assert flags == (True, True)
    child_lower, child_upper = (0, 1), (1, 2)
    assert engine._zero_delta_guard(certificate, certificate.lower, certificate.upper,
                                    child_lower, child_upper)[0]
    tightened = _integer_capacity_upper_bounds(child_lower, child_upper, requirements, QUEUE, LANES)
    assert tightened == (0, 2)
    assert engine._zero_delta_guard(certificate, certificate.lower, certificate.upper,
                                    child_lower, tightened) == (False, 1, 0, 1)


def test_loose_certificate_keeps_own_numerator_and_skips_only_a_zero_delta(monkeypatch):
    monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: fractional_root(16.0))
    search = engine.ZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    initial = search.advance_to(1)
    region = search._search.gi_frame.f_locals["frontier"][0][-1]
    assert region.bound == 25 and region.certificate.numerator == 32
    assert region.certificate.lower_zero_flags == (False, True)
    assert engine._zero_delta_guard(region.certificate, region.lower, region.upper,
                                    region.lower, (1, 0))[0]
    assert region.certificate.prepared is None
    assert initial.affine_dual.certificates_prepared == 0
    assert_fees(initial, 2)


def test_failed_lp_after_a_skip_retains_certificate_and_handles_later_nonzero_change(monkeypatch):
    calls, events = [], []
    real_guard = engine._zero_delta_guard
    # Both children retain a live second column. After a failed LP, a later
    # first-column upper change therefore reaches the nonzero-price guard.
    queue = (8, 3)

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        events.append(("lp", len(calls)))
        return (SimpleNamespace(success=True, x=(1.0, 1.5),
            ineqlin=SimpleNamespace(marginals=(0.0, -8.0, 0.0, -6.0, 0.0)))
            if len(calls) == 1 else SimpleNamespace(success=False))

    def guard(*args):
        result = real_guard(*args)
        events.append(("guard", result[0]))
        return result

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "_zero_delta_guard", guard)
    search = engine.ZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, queue, LANES)
    saved = []
    for cap in (1, 2, 128):
        final = search.advance_to(cap)
        assert_fees(final, 2)
        saved.append((final, asdict(final)))
    assert events.index(("guard", True)) < events.index(("lp", 2))
    assert ("guard", False) in events[events.index(("lp", 2)) + 1:]
    assert final.zero_delta.zero_delta_skips >= 2
    assert final.dual_reuse.child_bound_evaluations >= 1
    assert final.affine_dual.certificates_prepared == final.dual_reuse.certificates_saved == 1
    assert final.selection.safe_fallback_lp_calls == len(calls) - 1
    assert final.selection.canonical_exact
    assert all(asdict(stage) == record for stage, record in saved)
    new_calls = calls[:]
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.AffineDualReuseIntegerCapacitySearch(OPTIONS, queue, LANES).advance_to(128)
    assert calls == new_calls and behavior(asdict(final)) == behavior(asdict(old))


def test_resume_preserves_lp_sequence_cumulative_guard_fees_and_snapshot_immutability(monkeypatch):
    real_lp, calls = engine.linprog, []

    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        return real_lp(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    options = (*OPTIONS, Option((3, 2), 1.25), Option((0, 1), 2.0))
    direct = engine.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    direct_calls = calls[:]
    calls.clear()
    search = engine.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, QUEUE, LANES)
    prior_fee = None
    saved = []
    for cap in (2, 3, 4, 8, 64):
        final = search.advance_to(cap)
        assert_fees(final, 2)
        assert search.advance_to(cap) is final
        fee = {**asdict(final.affine_dual), **{"guard_" + name: value for name, value in asdict(final.zero_delta).items()}}
        if prior_fee is not None:
            assert all(fee[name] >= prior_fee[name] for name in fee)
        prior_fee = fee
        saved.append((final, asdict(final)))
    assert final == direct and calls == direct_calls
    assert all(asdict(stage) == record for stage, record in saved)
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.AffineDualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    assert calls == direct_calls and behavior(asdict(final)) == behavior(asdict(old))


@pytest.mark.parametrize("options,queue,lanes,initial_cap", (
    (OPTIONS, QUEUE, LANES, 1),
    ((Option((1, 0), 2.0), Option((1, 0), 1.0), Option((0, 2), 2.0)), (1, 2), 2, 2),
    (OPTIONS, (0, 0), LANES, 1),
    ((Option((1, 0), 1.0),), (1, 0), 1, 1),
))
def test_whole_router_keeps_action_proof_path_and_all_nonfee_counters(options, queue, lanes, initial_cap):
    new = select_zero_delta_affine_dual_residual_capacity_frontier_frame(options, queue, lanes, initial_cap, 64, 8192)
    old = select_affine_dual_residual_capacity_frontier_frame(options, queue, lanes, initial_cap, 64, 8192)
    assert behavior(asdict(new)) == behavior(asdict(old))
    assert new.selection.canonical_exact
    assert_fees(new.initial_lp, len(queue))
    assert_fees(new.final_lp, len(queue))


def test_snapshot_and_frontier_export_do_not_run_the_guard_or_prepare_coefficients(monkeypatch):
    monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: fractional_root())
    search = engine.ZeroDeltaAffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    initial = search.advance_to(1)
    saved = asdict(initial)
    assert capture_strict_frontier(search).regions
    assert search.advance_to(1) is initial and asdict(initial) == saved
    assert initial.zero_delta.guard_calls == initial.affine_dual.certificates_prepared == 0
    assert all(entry[-1].certificate.prepared is None for entry in search._search.gi_frame.f_locals["frontier"])


def test_failed_root_frontier_export_and_proof_routing_remain_identical(monkeypatch):
    failed = lambda *args, **kwargs: SimpleNamespace(success=False)
    monkeypatch.setattr(engine, "linprog", failed)
    monkeypatch.setattr(previous, "linprog", failed)
    options = (Option((1, 0), 1.0), Option((3, 0), 1.0), Option((0, 1), 1.0), Option((0, 3), 1.0))
    new = select_zero_delta_affine_dual_residual_capacity_frontier_frame(options, (4, 4), 3, 1, 2, 8192)
    old = select_affine_dual_residual_capacity_frontier_frame(options, (4, 4), 3, 1, 2, 8192)
    assert behavior(asdict(new)) == behavior(asdict(old))
    assert new.frontier_evidence.regions and new.probe.score_proof.status == "BETTER_ACTION_FOUND"
    assert all(value == 0 for value in asdict(new.final_lp.zero_delta).values())
    assert all(value == 0 for value in asdict(new.final_lp.affine_dual).values())
