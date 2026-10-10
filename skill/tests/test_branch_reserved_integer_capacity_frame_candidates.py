"""Skipped roots reduce actual LP work without enlarging branch allowance."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.branch_reserved_integer_capacity_frame_candidates as engine


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (Option((2, 1, 0), 1.0), Option((1, 2, 0), 1.0), Option((0, 0, 1), 2.0))


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
    selected, roots = stage.selection, stage.root_pruning
    assert selected.option_counts == tuple(selected.option_indices.count(index) for index in range(len(options)))
    assert selected.service == tuple(sum(count * option.requirements[c]
                                         for count, option in zip(selected.option_counts, options))
                                     for c in range(len(queue)))
    assert all(done <= waiting for done, waiting in zip(selected.service, queue))
    score = Fraction(sum(q * s for q, s in zip(queue, selected.service))) / Fraction.from_float(selected.duration)
    upper = score if selected.score_exact else Fraction.from_float(selected.upper_bound)
    assert score <= -expected[0] <= upper
    assert roots.roots_seen == roots.roots_solved + roots.roots_skipped_empty + roots.roots_skipped_bound
    reserved = roots.roots_seen - roots.roots_solved
    assert selected.lp_calls + reserved <= selected.lp_node_budget
    assert selected.lp_calls - roots.roots_solved <= selected.lp_node_budget - roots.roots_seen
    if selected.score_exact:
        assert score == -expected[0] and selected.gap == 0.0
    if selected.canonical_exact:
        indices = tuple(len(options) if index is None else index for index in selected.option_indices)
        assert selected.exact and selected.score_exact
        assert len(indices) == lanes
        assert (-score, Fraction.from_float(selected.duration), indices) == expected


def test_root_count_cap_reserves_empty_root_and_does_not_start_branching(monkeypatch):
    original, calls = engine.linprog, []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", counted)
    stage = engine.BranchReservedIntegerCapacitySearch(OPTIONS, (4, 4, 0), 3).advance_to(2)
    assert stage.root_pruning.roots_seen == 2
    assert stage.root_pruning.roots_solved == 1 and stage.root_pruning.roots_skipped_empty == 1
    assert len(calls) == stage.selection.lp_calls == 1
    assert stage.selection.branch_splits == 0
    assert stage.selection.frontier_regions_remaining > 0
    assert not stage.selection.score_exact and not stage.selection.canonical_exact
    assert_certificate(stage, OPTIONS, (4, 4, 0), 3, exhaustive_key(OPTIONS, (4, 4, 0), 3))


def test_registered_caps_keep_branch_allowance_and_complete_original_option_oracle():
    options = (OPTIONS[0], OPTIONS[1], Option((2, 1, 0), 1.25),
               Option((1, 1, 0), 1.5), OPTIONS[2])
    for queue in ((0, 0, 0), (1, 1, 0), (4, 4, 0)):
        expected = exhaustive_key(options, queue, 3)
        search = engine.BranchReservedIntegerCapacitySearch(options, queue, 3)
        for cap in (3, 4, 5, 16, 128):
            stage = search.advance_to(cap)
            assert_certificate(stage, options, queue, 3, expected)
            if cap == 3:
                assert stage.selection.lp_calls == stage.root_pruning.roots_solved
                assert stage.selection.branch_splits == 0
        assert stage.selection.canonical_exact


@pytest.mark.parametrize("failure", ("exception", "unsuccessful"))
def test_failed_lp_attempts_also_consume_reserved_branch_slots(monkeypatch, failure):
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        if failure == "exception":
            raise RuntimeError("test solver failure")
        return SimpleNamespace(success=False)

    monkeypatch.setattr(engine, "linprog", failed)
    search = engine.BranchReservedIntegerCapacitySearch(OPTIONS, (4, 4, 0), 3)
    expected = exhaustive_key(OPTIONS, (4, 4, 0), 3)
    for cap in (2, 3, 4, 512):
        stage = search.advance_to(cap)
        assert len(calls) == stage.selection.lp_calls
        assert stage.selection.safe_fallback_lp_calls == len(calls)
        assert stage.selection.successful_lp_calls == 0
        assert_certificate(stage, OPTIONS, (4, 4, 0), 3, expected)
        if cap == 2:
            assert stage.selection.branch_splits == 0 and len(calls) == 1
    assert stage.selection.lp_calls > stage.root_pruning.roots_solved
    assert stage.selection.canonical_exact


def test_resuming_preserves_lp_signature_and_cumulative_root_preprocessing(monkeypatch):
    original_lp = engine.linprog
    original_reduce = engine.reduce_requirement_dominated_options
    original_capacity = engine._integer_capacity_upper_bounds
    calls, reductions, capacities = [], [], []

    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])),
                      tuple(kwargs["b_ub"]), tuple(kwargs["bounds"]), kwargs["method"]))
        return original_lp(*args, **kwargs)

    def reduction(*args, **kwargs):
        reductions.append(1)
        return original_reduce(*args, **kwargs)

    def capacity(*args, **kwargs):
        capacities.append(1)
        return original_capacity(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "reduce_requirement_dominated_options", reduction)
    monkeypatch.setattr(engine, "_integer_capacity_upper_bounds", capacity)
    direct = engine.BranchReservedIntegerCapacitySearch(OPTIONS, (4, 4, 0), 3).advance_to(64)
    direct_calls = calls[:]
    calls.clear()
    reductions.clear()
    capacities.clear()
    search = engine.BranchReservedIntegerCapacitySearch(OPTIONS, (4, 4, 0), 3)
    assert not calls and not reductions and not capacities
    for cap in (2, 3, 4, 8, 64):
        stage = search.advance_to(cap)
        assert len(calls) == stage.selection.lp_calls
        assert len(reductions) == 1
        assert len(capacities) == stage.capacity_propagation.region_visits
        assert stage.root_pruning == direct.root_pruning
        assert search.advance_to(cap) is stage
    assert stage == direct and calls == direct_calls


def test_legal_fractional_root_can_leave_a_lower_score_without_false_exactness(monkeypatch):
    # Controlled valid LP optimum: x=(1,.5), service4, resource dual4.
    # Integer-cap bounds are (1,2), so both coordinates respect the root box.
    # The residual heuristic returns service3; this is a test outcome, not a
    # new empirical loss claim about the historical confirmation.
    options, queue, lanes = (Option((3, 0), 1.0), Option((2, 0), 1.0), Option((0, 1), 2.0)), (4, 0), 3
    original, calls = engine.linprog, []

    def lp(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            assert kwargs["b_ub"][-2:] == [1, 2]
            return SimpleNamespace(success=True, x=(1.0, 0.5),
                                   ineqlin=SimpleNamespace(marginals=(0.0, -4.0, 0.0, 0.0, 0.0)))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    expected = exhaustive_key(options, queue, lanes)
    search = engine.BranchReservedIntegerCapacitySearch(options, queue, lanes)
    first = search.advance_to(2)
    assert first.selection.score == 12.0 < float(-expected[0]) == 16.0
    assert first.selection.upper_bound == 16.0 and first.selection.gap == 4.0
    assert not first.selection.score_exact and not first.selection.canonical_exact
    assert first.selection.branch_splits == 0 and first.selection.frontier_regions_remaining > 0
    assert_certificate(first, options, queue, lanes, expected)
    final = search.advance_to(64)
    assert len(calls) == final.selection.lp_calls
    assert_certificate(final, options, queue, lanes, expected)
    assert final.selection.canonical_exact and final.selection.option_indices == (1, 1, None)
