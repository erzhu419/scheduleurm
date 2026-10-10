"""Root pruning must retain original-option actions and unresolved certificates."""
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations_with_replacement
from random import Random
from types import SimpleNamespace

import pytest

import algorithm.root_pruned_integer_capacity_frame_candidates as candidate


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (
    Option((1, 0), 5.0), Option((0, 1), 2.0), Option((2, 1), 5.0),
    Option((1, 2), 3.0), Option((3, 0), 2.0), Option((0, 3), 3.0),
)


def exhaustive_key(options, queue, lanes, idle):
    """Enumerate original lane actions, independently of grouping and LP roots."""
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        active = [options[index] for index in indices if index < len(options)]
        service = tuple(sum(option.requirements[c] for option in active)
                        for c in range(len(queue)))
        if any(done > waiting for done, waiting in zip(service, queue)):
            continue
        duration = max([Fraction.from_float(idle)] +
                       [Fraction.from_float(option.duration) for option in active])
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        keys.append((-score, duration, indices))
    return min(keys)


def assert_bound_and_action(stage, options, queue, lanes, idle, expected):
    selected = stage.selection
    score = Fraction(sum(q * s for q, s in zip(queue, selected.service))) / \
        Fraction.from_float(selected.duration)
    assert selected.option_counts == tuple(selected.option_indices.count(index)
                                           for index in range(len(options)))
    assert len(selected.option_indices) == lanes
    assert selected.service == tuple(sum(count * option.requirements[coordinate]
                                         for count, option in zip(selected.option_counts, options))
                                     for coordinate in range(len(queue)))
    assert selected.duration == max([idle] + [option.duration
        for count, option in zip(selected.option_counts, options) if count])
    assert selected.score == float(score)
    assert all(done <= waiting for done, waiting in zip(selected.service, queue))
    upper = score if selected.score_exact else Fraction.from_float(selected.upper_bound)
    assert score <= -expected[0] <= upper
    if selected.score_exact:
        assert score == -expected[0] and selected.gap == 0.0
    if selected.canonical_exact:
        indices = tuple(len(options) if index is None else index
                        for index in selected.option_indices)
        assert selected.exact and selected.score_exact
        assert (-score, Fraction.from_float(selected.duration), indices) == expected
    root = stage.root_pruning
    assert root.roots_seen == selected.duration_root_count
    assert root.roots_seen == root.roots_solved + root.roots_skipped_empty + root.roots_skipped_bound
    assert root.roots_skipped_canonical_tie <= root.roots_skipped_bound
    assert root.roots_solved <= selected.lp_calls


@pytest.mark.parametrize("case", range(12))
def test_original_option_oracle_bounds_and_full_canonical_action(case):
    rng = Random(2026092800 + case)
    requirements = ((1, 0), (0, 1), (1, 1), (2, 1))
    options = [Option(requirement, rng.choice((0.5, 1.0, 1.25, 2.0)))
               for requirement in requirements for _ in range(2)]
    rng.shuffle(options)
    queue = tuple(rng.randrange(7) for _ in range(2))
    lanes, idle = 2 + case % 2, (1.0, 1.5)[case % 2]
    roots = len({min(max(idle, option.duration) for option in options
                     if option.requirements == requirement)
                 for requirement in requirements})
    expected = exhaustive_key(options, queue, lanes, idle)
    search = candidate.RootPrunedIntegerCapacitySearch(options, queue, lanes, idle)
    for budget in (roots, roots + 1, 256):
        actual = search.advance_to(budget)
        assert actual.selection.lp_calls <= budget
        assert_bound_and_action(actual, options, queue, lanes, idle, expected)
    assert actual.selection.canonical_exact


def test_zero_queue_discards_every_empty_root_without_calling_lp(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("zero queue with positive requirements needs no root LP")

    monkeypatch.setattr(candidate, "linprog", unexpected)
    search = candidate.RootPrunedIntegerCapacitySearch(OPTIONS, (0, 0), 4)
    first = search.advance_to(3)
    assert first.selection.lp_calls == 0
    assert first.selection.canonical_exact and first.selection.score_exact
    assert first.selection.option_indices == (None,) * 4
    assert first.selection.duration == 1.0 and first.selection.upper_bound == 0.0
    assert first.root_pruning.roots_seen == first.root_pruning.roots_skipped_empty == 3
    assert first.root_pruning.roots_solved == 0
    assert search.advance_to(3) is first
    later = search.advance_to(256)
    assert later == replace(first, selection=replace(first.selection, lp_node_budget=256))


@pytest.mark.parametrize("options,queue,lanes,skipped", (
    ((Option((1, 0), 1.0), Option((0, 1), 2.0)), (3, 0), 2, "roots_skipped_empty"),
    ((Option((1,), 1.0), Option((2,), 10.0)), (2,), 1, "roots_skipped_bound"),
    ((Option((1,), 1.0), Option((2,), 2.0)), (2,), 1, "roots_skipped_canonical_tie"),
))
def test_empty_strict_and_canonical_root_pruning_charge_only_actual_lp(
        monkeypatch, options, queue, lanes, skipped):
    original, calls = candidate.linprog, []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate, "linprog", counted)
    actual = candidate.RootPrunedIntegerCapacitySearch(options, queue, lanes).advance_to(64)
    assert len(calls) == actual.selection.lp_calls == 1
    assert actual.root_pruning.roots_seen == 2
    assert actual.root_pruning.roots_solved == 1
    assert getattr(actual.root_pruning, skipped) == 1
    assert_bound_and_action(actual, options, queue, lanes, 1.0,
                          exhaustive_key(options, queue, lanes, 1.0))
    assert actual.selection.canonical_exact


def test_one_root_budget_keeps_fractional_unresolved_region_live():
    options = (Option((2, 1), 1.0), Option((1, 2), 1.0))
    queue, lanes = (4, 4), 3
    search = candidate.RootPrunedIntegerCapacitySearch(options, queue, lanes)
    first = search.advance_to(1)
    assert first.selection.lp_calls == 1
    assert not first.selection.score_exact and not first.selection.canonical_exact
    assert first.selection.frontier_regions_remaining > 0
    assert first.selection.gap > 0
    expected = exhaustive_key(options, queue, lanes, 1.0)
    assert_bound_and_action(first, options, queue, lanes, 1.0, expected)
    final = search.advance_to(64)
    assert_bound_and_action(final, options, queue, lanes, 1.0, expected)
    assert final.selection.canonical_exact
    assert first.selection.lp_node_budget == 1 and first.selection.lp_calls == 1


def test_resume_matches_fresh_cumulative_cap_and_never_repeats_work(monkeypatch):
    original_solver = candidate.linprog
    original_reduce = candidate.reduce_requirement_dominated_options
    original_capacity = candidate._integer_capacity_upper_bounds
    calls, reductions, capacities = [], [], []

    def signature(args, kwargs):
        return (tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])),
                tuple(kwargs["b_ub"]), tuple(kwargs["bounds"]), kwargs["method"])

    def solver(*args, **kwargs):
        calls.append(signature(args, kwargs))
        return original_solver(*args, **kwargs)

    def reduction(*args, **kwargs):
        reductions.append(1)
        return original_reduce(*args, **kwargs)

    def capacity(*args, **kwargs):
        capacities.append(1)
        return original_capacity(*args, **kwargs)

    monkeypatch.setattr(candidate, "linprog", solver)
    monkeypatch.setattr(candidate, "reduce_requirement_dominated_options", reduction)
    monkeypatch.setattr(candidate, "_integer_capacity_upper_bounds", capacity)
    expected = candidate.RootPrunedIntegerCapacitySearch(OPTIONS, (5, 6), 4).advance_to(64)
    expected_calls = calls[:]
    calls.clear()
    reductions.clear()
    capacities.clear()
    search = candidate.RootPrunedIntegerCapacitySearch(OPTIONS, (5, 6), 4)
    assert not calls and not reductions and not capacities
    for cap in (3, 4, 5, 8, 16, 64):
        result = search.advance_to(cap)
        assert len(calls) == result.selection.lp_calls
        assert len(capacities) == result.capacity_propagation.region_visits
        assert len(reductions) == 1
        assert result.root_pruning == expected.root_pruning
        assert search.advance_to(cap) is result
    assert result == expected
    assert calls == expected_calls
    assert result.selection.canonical_exact


@pytest.mark.parametrize("failure", ("exception", "unsuccessful", "invalid_dual"))
def test_failed_solver_calls_and_branch_calls_remain_in_cumulative_count(monkeypatch, failure):
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        if failure == "exception":
            raise RuntimeError("test LP failure")
        if failure == "unsuccessful":
            return SimpleNamespace(success=False)
        return SimpleNamespace(success=True, x=[0.0] * len(args[0]))

    monkeypatch.setattr(candidate, "linprog", failed)
    options, queue, lanes = (Option((2, 1), 1.0), Option((1, 2), 1.0)), (4, 4), 3
    expected = exhaustive_key(options, queue, lanes, 1.0)
    search = candidate.RootPrunedIntegerCapacitySearch(options, queue, lanes)
    for cap in (1, 2, 512):
        actual = search.advance_to(cap)
        assert len(calls) == actual.selection.lp_calls <= cap
        assert actual.selection.safe_fallback_lp_calls == actual.selection.lp_calls
        assert actual.selection.successful_lp_calls == (len(calls) if failure == "invalid_dual" else 0)
        assert actual.root_pruning.roots_solved == 1
        assert_bound_and_action(actual, options, queue, lanes, 1.0, expected)
    assert actual.selection.lp_calls > actual.root_pruning.roots_solved
    assert actual.selection.canonical_exact


def test_rejected_decreasing_budget_leaves_live_frontier_usable():
    options = (Option((2, 1), 1.0), Option((1, 2), 1.0))
    search = candidate.RootPrunedIntegerCapacitySearch(options, (4, 4), 3)
    first = search.advance_to(1)
    with pytest.raises(ValueError, match="nondecreasing"):
        search.advance_to(0)
    assert search.advance_to(1) is first
    assert search.advance_to(64) == candidate.RootPrunedIntegerCapacitySearch(
        options, (4, 4), 3).advance_to(64)


def test_empty_catalogue_has_no_roots_and_needs_no_solver(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("empty catalogue has no LP root")

    monkeypatch.setattr(candidate, "linprog", unexpected)
    actual = candidate.RootPrunedIntegerCapacitySearch((), (3, 2), 4).advance_to(0)
    assert actual.selection.lp_calls == 0 and actual.selection.canonical_exact
    assert actual.root_pruning.roots_seen == 0
    assert_bound_and_action(actual, (), (3, 2), 4, 1.0,
                          exhaustive_key((), (3, 2), 4, 1.0))
