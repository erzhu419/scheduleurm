"""Check exact dual algebra, lazy preparation, fees, and frozen search paths."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.affine_dual_reuse_integer_capacity_frame_candidates as engine
import algorithm.prune_first_dual_reuse_integer_capacity_frame_candidates as previous
from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
from algorithm.dual_reuse_integer_capacity_frame_candidates import _reuse_dual_upper
from algorithm.frontier_score_proof_certificate_routed_frame_candidates import capture_strict_frontier
from algorithm.prune_first_residual_capacity_frame_candidates import select_prune_first_residual_capacity_frontier_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (Option((3, 2), 1.0), Option((2, 0), 1.0))
QUEUE, LANES = (4, 3), 3
OLD_READS = ("child_rhs_entries", "child_requirement_reads", "child_weight_reads")


def behavior(record):
    if isinstance(record, dict):
        return {name: behavior(value) for name, value in record.items()
                if name != "affine_dual" and name not in OLD_READS}
    if isinstance(record, (tuple, list)):
        return [behavior(value) for value in record]
    return record


def assert_fees(stage, dimension):
    fee, dual = stage.affine_dual, stage.dual_reuse
    assert all(value >= 0 for value in asdict(fee).values())
    assert all(getattr(dual, name) == 0 for name in OLD_READS)
    prepared, variables = fee.certificates_prepared, fee.coefficient_weight_reads
    assert prepared <= dual.certificates_saved
    assert prepared <= dual.child_bound_evaluations
    assert fee.certificate_numerator_products == dual.certificates_saved
    assert fee.price_reads == 2 * ((1 + dimension) * prepared + variables)
    assert fee.denominator_lcm_steps == (2 + dimension) * prepared + variables
    assert fee.integer_scalings == (2 + dimension) * prepared + 2 * variables
    assert fee.coefficient_requirement_reads == dimension * variables
    assert fee.coefficient_integer_operations == (2 * dimension + 3) * variables
    assert fee.box_coordinate_reads == 4 * fee.bound_variable_visits
    assert fee.box_coordinate_comparisons == 2 * fee.bound_variable_visits
    assert fee.changed_lower_coordinates <= fee.bound_variable_visits
    assert fee.changed_upper_coordinates <= fee.bound_variable_visits
    changed = fee.changed_lower_coordinates + fee.changed_upper_coordinates
    assert fee.delta_coefficient_reads == changed
    assert fee.delta_integer_operations == 3 * changed
    assert fee.bound_integer_products == 2 * dual.child_bound_evaluations
    assert fee.bound_rational_conversions == dual.child_bound_evaluations


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


def selected_key(selection, option_count):
    return (-Fraction.from_float(selection.score), Fraction.from_float(selection.duration),
            tuple(option_count if index is None else index for index in selection.option_indices))


def fractional_root(upper_price=6.0):
    return SimpleNamespace(success=True, x=(1.0, 0.5),
                           ineqlin=SimpleNamespace(marginals=(0.0, -4.0, 0.0, -upper_price, 0.0)))


def test_affine_integer_update_matches_full_rational_rhs_after_multiple_box_changes():
    requirements = ((3, 1), (1, 2), (2, 3))
    queue, lanes, threshold = (8, 8), 4, Fraction(5, 4)
    weights = tuple(sum(q * r for q, r in zip(queue, requirement)) for requirement in requirements)
    prices = (Fraction(99, 8), Fraction(15, 4), Fraction(25, 8),
              Fraction(25, 2), Fraction(35, 4), Fraction(85, 4))
    lower, upper = (0, 0, 0), (4, 4, 4)
    numerator = _reuse_dual_upper(lower, upper, requirements, queue, weights, lanes, threshold, prices) * threshold
    certificate = engine._AffineDualCertificate(prices, lower, upper, numerator)
    coefficients = engine._prepare_affine_certificate(certificate, requirements, weights)
    scaled = coefficients.anchor_numerator
    for next_lower, next_upper in (((1, 0, 0), (1, 2, 1)), ((1, 1, 0), (1, 1, 1)), ((1, 1, 1), (1, 1, 1))):
        scaled, changed_lower, changed_upper = engine._update_affine_numerator(
            coefficients, scaled, lower, upper, next_lower, next_upper)
        assert changed_lower == sum(a != b for a, b in zip(lower, next_lower))
        assert changed_upper == sum(a != b for a, b in zip(upper, next_upper))
        actual = Fraction(scaled * threshold.denominator, coefficients.denominator * threshold.numerator)
        expected = _reuse_dual_upper(next_lower, next_upper, requirements, queue, weights, lanes, threshold, prices)
        assert actual == expected
        lower, upper = next_lower, next_upper
    assert coefficients.denominator == 8


def test_own_certificate_numerator_is_kept_when_an_inherited_bound_is_tighter(monkeypatch):
    monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: fractional_root(16.0))
    search = engine.AffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    initial = search.advance_to(1)
    assert initial.selection.upper_bound == 25.0
    region = search._search.gi_frame.f_locals["frontier"][0][-1]
    assert region.bound == 25 and region.certificate.numerator == 32
    assert region.certificate.prepared is None
    assert initial.affine_dual.certificates_prepared == 0
    assert initial.affine_dual.certificate_numerator_products == 1


def test_snapshot_and_frontier_export_do_not_prepare_a_certificate(monkeypatch):
    monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: fractional_root())
    search = engine.AffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    initial = search.advance_to(1)
    saved = asdict(initial)
    evidence = capture_strict_frontier(search)
    assert evidence.regions
    assert search.advance_to(1) is initial and asdict(initial) == saved
    assert initial.affine_dual.certificates_prepared == 0
    assert all(entry[-1].certificate.prepared is None
               for entry in search._search.gi_frame.f_locals["frontier"])


def test_already_pruned_child_does_not_pay_for_affine_preparation(monkeypatch):
    calls = []

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        primal = (0.0, 1.0) if len(calls) == 1 else (0.0, 0.0)
        return SimpleNamespace(success=True, x=primal,
                               ineqlin=SimpleNamespace(marginals=(-1.0, 0.0, 0.0, 0.0, 0.0)))

    monkeypatch.setattr(engine, "linprog", lp)
    options = (Option((1, 0), 1.0), Option((0, 1), 1.0))
    search = engine.AffineDualReuseIntegerCapacitySearch(options, (1, 1), 1)
    initial, final = search.advance_to(1), search.advance_to(2)
    assert initial.affine_dual.certificates_prepared == 0
    assert final.dual_reuse.child_bound_evaluations == 1
    assert final.affine_dual.certificates_prepared == 1
    assert final.affine_dual.bound_variable_visits == 2
    assert final.selection.canonical_exact and final.selection.option_indices == (0,)
    assert_fees(final, 2)
    new_calls = calls[:]
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.PruneFirstDualReuseIntegerCapacitySearch(options, (1, 1), 1).advance_to(2)
    assert new_calls == calls and behavior(asdict(final)) == behavior(asdict(old))


def test_failed_child_lp_keeps_and_reuses_one_prepared_parent_certificate(monkeypatch):
    calls = []

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        return fractional_root() if len(calls) == 1 else SimpleNamespace(success=False)

    monkeypatch.setattr(engine, "linprog", lp)
    search = engine.AffineDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    saved = []
    for cap in (1, 2, 128):
        final = search.advance_to(cap)
        assert_fees(final, 2)
        saved.append((final, asdict(final)))
    assert final.selection.canonical_exact
    assert final.dual_reuse.certificates_saved == final.affine_dual.certificates_prepared == 1
    assert final.dual_reuse.child_bound_evaluations >= 2
    assert final.selection.safe_fallback_lp_calls == len(calls) - 1
    assert selected_key(final.selection, len(OPTIONS)) == oracle_key(OPTIONS, QUEUE, LANES)
    assert all(asdict(stage) == record for stage, record in saved)
    new_calls = calls[:]
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.PruneFirstDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES).advance_to(128)
    assert calls == new_calls and behavior(asdict(final)) == behavior(asdict(old))


def test_resume_keeps_lp_sequence_cumulative_fees_and_exact_canonical_action(monkeypatch):
    real_lp, calls = engine.linprog, []

    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        return real_lp(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    options = (*OPTIONS, Option((3, 2), 1.25), Option((0, 1), 2.0))
    direct = engine.AffineDualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    direct_calls = calls[:]
    calls.clear()
    search = engine.AffineDualReuseIntegerCapacitySearch(options, QUEUE, LANES)
    prior_fee = None
    for cap in (2, 3, 4, 8, 64):
        final = search.advance_to(cap)
        assert_fees(final, 2)
        assert search.advance_to(cap) is final
        fee = asdict(final.affine_dual)
        if prior_fee is not None:
            assert all(fee[name] >= prior_fee[name] for name in fee)
        prior_fee = fee
    assert final == direct and calls == direct_calls
    assert selected_key(final.selection, len(options)) == oracle_key(options, QUEUE, LANES)
    with pytest.raises(ValueError, match="nondecreasing"):
        search.advance_to(8)
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.PruneFirstDualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    assert calls == direct_calls and behavior(asdict(final)) == behavior(asdict(old))


@pytest.mark.parametrize("options,queue,lanes,initial_cap", (
    (OPTIONS, QUEUE, LANES, 1),
    ((Option((1, 0), 2.0), Option((1, 0), 1.0), Option((0, 2), 2.0)), (1, 2), 2, 2),
    (OPTIONS, (0, 0), LANES, 1),
    ((Option((1, 0), 1.0),), (1, 0), 1, 1),
))
def test_residual_router_preserves_all_results_and_reports_actual_fees(options, queue, lanes, initial_cap):
    new = select_affine_dual_residual_capacity_frontier_frame(options, queue, lanes, initial_cap, 64, 8192)
    old = select_prune_first_residual_capacity_frontier_frame(options, queue, lanes, initial_cap, 64, 8192)
    assert behavior(asdict(new)) == behavior(asdict(old))
    assert new.selection.canonical_exact
    assert selected_key(new.selection, len(options)) == oracle_key(options, queue, lanes)
    assert_fees(new.initial_lp, len(queue))
    assert_fees(new.final_lp, len(queue))
    if new.final_lp.dual_reuse.child_bound_evaluations == 0:
        assert new.final_lp.affine_dual.certificates_prepared == 0


def test_open_failed_lp_frontier_remains_exportable_without_hidden_preparation(monkeypatch):
    failed = lambda *args, **kwargs: SimpleNamespace(success=False)
    monkeypatch.setattr(engine, "linprog", failed)
    monkeypatch.setattr(previous, "linprog", failed)
    options = (Option((1, 0), 1.0), Option((3, 0), 1.0), Option((0, 1), 1.0), Option((0, 3), 1.0))
    new = select_affine_dual_residual_capacity_frontier_frame(options, (4, 4), 3, 1, 2, 8192)
    old = select_prune_first_residual_capacity_frontier_frame(options, (4, 4), 3, 1, 2, 8192)
    assert behavior(asdict(new)) == behavior(asdict(old))
    assert new.frontier_evidence.regions
    assert new.probe.score_proof.status == "BETTER_ACTION_FOUND"
    assert not new.selection.score_exact
    assert all(value == 0 for value in asdict(new.final_lp.affine_dual).values())
