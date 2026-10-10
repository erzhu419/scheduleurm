from dataclasses import dataclass, replace
from random import Random
from types import SimpleNamespace

import pytest

import algorithm.integer_capacity_shared_threshold_frame_candidates as frozen
import algorithm.resumable_integer_capacity_frame_candidates as resumable


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (
    Option((1, 0), 5.0), Option((0, 1), 2.0), Option((2, 1), 5.0),
    Option((1, 2), 3.0), Option((3, 0), 2.0), Option((0, 3), 3.0),
)
CAPS = (3, 4, 5, 8, 16, 64)


def test_every_intermediate_snapshot_matches_frozen_search():
    search = resumable.IntegerCapacitySearch(OPTIONS, (5, 6), 4)
    snapshots = []
    for cap in CAPS:
        actual = search.advance_to(cap)
        expected = frozen.select_integer_capacity_shared_threshold_frame(OPTIONS, (5, 6), 4, cap)
        assert actual == expected
        snapshots.append(actual)
    assert not snapshots[0].selection.canonical_exact
    assert snapshots[0].selection.frontier_regions_remaining > 0
    assert snapshots[-1].selection.canonical_exact
    # Earlier immutable output must retain its cap and pending frontier.
    assert snapshots[0].selection.lp_node_budget == 3
    assert snapshots[0].selection.lp_calls == 3
    assert snapshots[0].selection.frontier_regions_remaining == 1


@pytest.mark.parametrize("case", range(12))
def test_split_search_preserves_all_work_and_certificates(case):
    rng = Random(2026092700 + case)
    requirements = ((1, 0), (0, 2), (1, 1), (2, 1))
    options = [Option(requirement, rng.choice((0.5, 1.0, 2.0, 3.0, 5.0)))
               for requirement in requirements for _ in range(2)]
    rng.shuffle(options)
    queue = tuple(rng.randrange(12) for _ in range(2))
    lanes = 2 + case % 4
    idle_duration = (1.0, 2.0)[case % 2]
    group_minima = {
        requirement: min(max(idle_duration, o.duration)
                         for o in options if o.requirements == requirement)
        for requirement in requirements
    }
    roots = len(set(group_minima.values()))
    search = resumable.IntegerCapacitySearch(options, queue, lanes, idle_duration)
    for cap in (roots, roots + 1, roots + 2, roots + 5, 256):
        assert search.advance_to(cap) == frozen.select_integer_capacity_shared_threshold_frame(
            options, queue, lanes, cap, idle_duration)
    assert search.advance_to(256).selection.canonical_exact


def test_roots_preprocessing_and_lp_prefix_are_never_repeated(monkeypatch):
    original_solver = frozen.linprog
    expected_calls = []

    def signature(args, kwargs):
        return (tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])),
                tuple(kwargs["b_ub"]), tuple(kwargs["bounds"]), kwargs["method"])

    def reference_solver(*args, **kwargs):
        expected_calls.append(signature(args, kwargs))
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(frozen, "linprog", reference_solver)
    expected = frozen.select_integer_capacity_shared_threshold_frame(OPTIONS, (5, 6), 4, 64)
    actual_calls, reduction_calls, capacity_calls = [], [], []
    original_reduction = resumable.reduce_requirement_dominated_options
    original_capacity = resumable._integer_capacity_upper_bounds

    def solver(*args, **kwargs):
        actual_calls.append(signature(args, kwargs))
        return original_solver(*args, **kwargs)

    def reduction(*args, **kwargs):
        reduction_calls.append(1)
        return original_reduction(*args, **kwargs)

    def capacity(*args, **kwargs):
        capacity_calls.append(1)
        return original_capacity(*args, **kwargs)

    monkeypatch.setattr(resumable, "linprog", solver)
    monkeypatch.setattr(resumable, "reduce_requirement_dominated_options", reduction)
    monkeypatch.setattr(resumable, "_integer_capacity_upper_bounds", capacity)
    search = resumable.IntegerCapacitySearch(OPTIONS, (5, 6), 4)
    assert not actual_calls and not reduction_calls and not capacity_calls
    for cap in CAPS:
        result = search.advance_to(cap)
        assert len(actual_calls) == result.selection.lp_calls
        assert len(capacity_calls) == result.capacity_propagation.region_visits
        assert len(reduction_calls) == 1
        assert search.advance_to(cap) is result
    assert result == expected
    assert actual_calls == expected_calls
    assert len(actual_calls) == 9


@pytest.mark.parametrize("failure", ("exception", "unsuccessful", "invalid_dual"))
def test_failed_lp_calls_resume_and_match_frozen_accounting(monkeypatch, failure):
    def failed(*args, **kwargs):
        if failure == "exception":
            raise RuntimeError("synthetic LP failure")
        if failure == "unsuccessful":
            return SimpleNamespace(success=False)
        return SimpleNamespace(success=True, x=[0.0] * len(args[0]))

    monkeypatch.setattr(frozen, "linprog", failed)
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return failed(*args, **kwargs)

    monkeypatch.setattr(resumable, "linprog", counted)
    search = resumable.IntegerCapacitySearch(OPTIONS, (5, 6), 4)
    for cap in (*CAPS, 4096):
        actual = search.advance_to(cap)
        expected = frozen.select_integer_capacity_shared_threshold_frame(OPTIONS, (5, 6), 4, cap)
        assert actual == expected
        assert len(calls) == actual.selection.lp_calls
        assert actual.selection.safe_fallback_lp_calls == actual.selection.lp_calls
    assert actual.selection.canonical_exact


def test_completed_search_changes_only_requested_cap(monkeypatch):
    calls = []
    original_solver = resumable.linprog

    def counted(*args, **kwargs):
        calls.append(1)
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(resumable, "linprog", counted)
    search = resumable.IntegerCapacitySearch(OPTIONS, (0, 0), 4)
    first = search.advance_to(3)
    assert first.selection.canonical_exact
    later = search.advance_to(4096)
    assert later == replace(first, selection=replace(first.selection, lp_node_budget=4096))
    assert len(calls) == 3


def test_empty_catalogue_needs_no_lp_calls(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("empty catalogue must not call LP")

    monkeypatch.setattr(resumable, "linprog", unexpected)
    search = resumable.IntegerCapacitySearch((), (3, 2), 4)
    for cap in (0, 1, 10):
        assert search.advance_to(cap) == frozen.select_integer_capacity_shared_threshold_frame(
            (), (3, 2), 4, cap)


def test_invalid_budget_and_decreasing_cap_leave_live_search_usable(monkeypatch):
    calls = []
    original_solver = resumable.linprog

    def counted(*args, **kwargs):
        calls.append(1)
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(resumable, "linprog", counted)
    search = resumable.IntegerCapacitySearch(OPTIONS, (5, 6), 4)
    for cap in (-1, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            search.advance_to(cap)
    with pytest.raises(ValueError, match="cover 3 duration thresholds"):
        search.advance_to(2)
    assert not calls
    first = search.advance_to(4)
    with pytest.raises(ValueError, match="nondecreasing"):
        search.advance_to(3)
    assert search.advance_to(4) is first
    assert len(calls) == 4
    assert search.advance_to(64) == frozen.select_integer_capacity_shared_threshold_frame(
        OPTIONS, (5, 6), 4, 64)
    assert len(calls) == 9
