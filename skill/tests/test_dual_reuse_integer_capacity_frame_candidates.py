"""Check exact child duals, fallback inheritance, and resumable LP accounting."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement, product
from types import SimpleNamespace

import pytest

import algorithm.dual_reuse_integer_capacity_frame_candidates as engine


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (Option((3, 2), 1.0), Option((2, 0), 1.0))
QUEUE, LANES = (4, 3), 3
PRICES = tuple(map(Fraction, (0, 4, 0, 6, 0)))


def exhaustive_key(options, queue, lanes):
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        active = [options[index] for index in indices if index < len(options)]
        service = tuple(sum(option.requirements[c] for option in active) for c in range(len(queue)))
        if any(done > waiting for done, waiting in zip(service, queue)):
            continue
        duration = max([Fraction(1)] + [Fraction.from_float(option.duration) for option in active])
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        keys.append((-score, duration, indices))
    return min(keys)


def assert_certificate(stage, options, queue, lanes, expected):
    selected, roots, work = stage.selection, stage.root_pruning, stage.dual_reuse
    assert selected.service == tuple(sum(count * option.requirements[c]
                                         for count, option in zip(selected.option_counts, options))
                                     for c in range(len(queue)))
    assert all(done <= waiting for done, waiting in zip(selected.service, queue))
    score = Fraction(sum(q * s for q, s in zip(queue, selected.service))) / Fraction.from_float(selected.duration)
    upper = score if selected.score_exact else Fraction.from_float(selected.upper_bound)
    assert score <= -expected[0] <= upper
    assert roots.roots_seen == roots.roots_solved + roots.roots_skipped_empty + roots.roots_skipped_bound
    assert selected.lp_calls + roots.roots_seen - roots.roots_solved <= selected.lp_node_budget
    assert work.certificates_saved == selected.dual_certified_lp_calls
    assert 0 <= work.child_bound_prunes <= work.child_bound_improvements <= work.child_bound_evaluations
    if selected.score_exact:
        assert score == -expected[0] and selected.gap == 0.0
    if selected.canonical_exact:
        indices = tuple(len(options) if index is None else index for index in selected.option_indices)
        assert selected.exact and selected.score_exact and len(indices) == lanes
        assert (-score, Fraction.from_float(selected.duration), indices) == expected


def certified_fractional_root():
    # Feasible optimum x=(1,.5): resource 0 binds; option 0's integer cap binds.
    return SimpleNamespace(success=True, x=(1.0, 0.5),
                           ineqlin=SimpleNamespace(marginals=(0.0, -4.0, 0.0, -6.0, 0.0)))


def test_dual_repair_returns_exact_feasible_prices_without_recertification():
    matrix = ((1, 1), (2, 1), (1, 0), (0, 1))
    objective, rhs = (6, 5), (3, 4, 2, 3)
    upper, repair, prices = engine._dual_upper_with_prices(
        matrix, rhs, objective, 2, Fraction(2), (-1.0, -1.0, 1.0, 0.0))
    assert prices == tuple(map(Fraction, (4, 1, 0, 0)))
    assert repair == Fraction(3) and upper == Fraction(9)
    assert all(isinstance(price, Fraction) and price >= 0 for price in prices)
    assert all(sum(row[column] * price for row, price in zip(matrix, prices)) >= coefficient
               for column, coefficient in enumerate(objective))


@pytest.mark.parametrize("lower,upper,expected", (
    ((0, 0), (1, 0), 22), ((0, 1), (0, 2), 16), ((1, 0), (1, 0), 22),
))
def test_child_rhs_dual_bounds_cover_every_feasible_integer_count(lower, upper, expected):
    requirements = tuple(option.requirements for option in OPTIONS)
    weights = tuple(sum(q * amount for q, amount in zip(QUEUE, requirement)) for requirement in requirements)
    bound = engine._reuse_dual_upper(lower, upper, requirements, QUEUE, weights, LANES, Fraction(1), PRICES)
    assert bound == Fraction(expected) and bound <= Fraction(22)
    feasible_scores = []
    for counts in product(*(range(left, right + 1) for left, right in zip(lower, upper))):
        service = tuple(sum(count * requirement[c] for count, requirement in zip(counts, requirements))
                        for c in range(len(QUEUE)))
        if sum(counts) <= LANES and all(done <= waiting for done, waiting in zip(service, QUEUE)):
            feasible_scores.append(sum(q * s for q, s in zip(QUEUE, service)))
    assert feasible_scores and max(feasible_scores) <= bound


def test_parent_dual_prunes_a_child_and_reports_arithmetic_before_another_lp(monkeypatch):
    original, calls = engine.linprog, []

    def lp(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            assert kwargs["b_ub"] == [3, 4, 3, 1, 2]
            return certified_fractional_root()
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    search = engine.DualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    initial = search.advance_to(1)
    assert initial.selection.score == 18.0 and initial.selection.upper_bound == 22.0
    assert not initial.selection.score_exact
    assert initial.dual_reuse.certificates_saved == 1
    assert initial.dual_reuse.child_bound_evaluations == 0
    final = search.advance_to(2)
    work = final.dual_reuse
    assert len(calls) == final.selection.lp_calls == 2
    assert work.child_bound_evaluations == 2
    assert work.child_bound_improvements == work.child_bound_prunes == 1
    assert work.child_rhs_entries == 10
    assert work.child_requirement_reads == 8 and work.child_weight_reads == 4
    assert_certificate(final, OPTIONS, QUEUE, LANES, exhaustive_key(OPTIONS, QUEUE, LANES))
    assert final.selection.canonical_exact
    assert initial.dual_reuse.child_bound_evaluations == 0


@pytest.mark.parametrize("failure", ("exception", "unsuccessful", "uncertifiable"))
def test_failed_child_solver_or_certification_retains_the_parent_dual(monkeypatch, failure):
    calls = []

    def lp(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return certified_fractional_root()
        if failure == "exception":
            raise RuntimeError("controlled child solver failure")
        if failure == "unsuccessful":
            return SimpleNamespace(success=False)
        return SimpleNamespace(success=True, x=(0.0, 0.0),
                               ineqlin=SimpleNamespace(marginals=(float("nan"),) * len(kwargs["b_ub"])))

    monkeypatch.setattr(engine, "linprog", lp)
    search = engine.DualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    expected = exhaustive_key(OPTIONS, QUEUE, LANES)
    for cap in (1, 2, 128):
        final = search.advance_to(cap)
        assert_certificate(final, OPTIONS, QUEUE, LANES, expected)
        assert final.selection.lp_calls == len(calls)
    assert final.dual_reuse.certificates_saved == 1
    # The third evaluation belongs to a descendant of a failed child solve.
    assert final.dual_reuse.child_bound_evaluations > 2
    assert final.selection.safe_fallback_lp_calls == len(calls) - 1
    assert final.selection.canonical_exact


def test_uncertifiable_root_does_not_save_or_reuse_invalid_prices(monkeypatch):
    def lp(*args, **kwargs):
        return SimpleNamespace(success=True, x=(0.0, 0.0),
                               ineqlin=SimpleNamespace(marginals=(float("nan"),) * len(kwargs["b_ub"])))

    monkeypatch.setattr(engine, "linprog", lp)
    final = engine.DualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES).advance_to(128)
    assert final.dual_reuse.certificates_saved == 0
    assert final.dual_reuse.child_bound_evaluations == 0
    assert final.selection.safe_fallback_lp_calls == final.selection.lp_calls > 0
    assert_certificate(final, OPTIONS, QUEUE, LANES, exhaustive_key(OPTIONS, QUEUE, LANES))
    assert final.selection.canonical_exact


def test_root_slots_stay_reserved_and_failed_lp_attempts_remain_paid(monkeypatch):
    calls = []

    def lp(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(success=False)

    monkeypatch.setattr(engine, "linprog", lp)
    options = (Option((2, 1, 0), 1.0), Option((1, 2, 0), 1.0), Option((0, 0, 1), 2.0))
    search = engine.DualReuseIntegerCapacitySearch(options, (4, 4, 0), 3)
    initial = search.advance_to(2)
    assert initial.root_pruning.roots_seen == 2 and initial.root_pruning.roots_solved == 1
    assert initial.root_pruning.roots_skipped_empty == 1
    assert initial.selection.branch_splits == 0 and len(calls) == initial.selection.lp_calls == 1
    final = search.advance_to(3)
    assert len(calls) == final.selection.lp_calls == 2
    assert final.selection.safe_fallback_lp_calls == 2
    assert final.selection.lp_calls + final.root_pruning.roots_seen - final.root_pruning.roots_solved == 3
    assert final.dual_reuse.certificates_saved == 0


def test_real_resume_preserves_lp_sequence_and_cumulative_work(monkeypatch):
    original_lp, original_reduce = engine.linprog, engine.reduce_requirement_dominated_options
    calls, reductions = [], []

    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        return original_lp(*args, **kwargs)

    def reduce(*args, **kwargs):
        reductions.append(1)
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "reduce_requirement_dominated_options", reduce)
    options = (*OPTIONS, Option((3, 2), 1.25), Option((0, 1), 2.0))
    direct = engine.DualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    direct_calls = calls[:]
    calls.clear()
    reductions.clear()
    search = engine.DualReuseIntegerCapacitySearch(options, QUEUE, LANES)
    for cap in (2, 3, 4, 8, 64):
        final = search.advance_to(cap)
        assert len(calls) == final.selection.lp_calls and len(reductions) == 1
        assert search.advance_to(cap) is final
        assert_certificate(final, options, QUEUE, LANES, exhaustive_key(options, QUEUE, LANES))
    assert final == direct and calls == direct_calls


@pytest.mark.parametrize("queue,lanes", (((0, 0), 2), ((1, 0), 2), ((1, 2), 2),
                                          ((2, 2), 1), ((2, 3), 3), ((3, 4), 3)))
def test_original_option_oracle_preserves_score_and_canonical_ties(queue, lanes):
    # At duration 2 the slower original index 0 can replace index 1 canonically.
    options = (Option((1, 0), 2.0), Option((1, 0), 1.0), Option((0, 2), 2.0))
    expected = exhaustive_key(options, queue, lanes)
    search = engine.DualReuseIntegerCapacitySearch(options, queue, lanes)
    for cap in (2, 3, 4, 64):
        final = search.advance_to(cap)
        assert_certificate(final, options, queue, lanes, expected)
    assert final.selection.canonical_exact
    if queue == (1, 2) and lanes == 2:
        assert final.selection.option_indices == (0, 2)
