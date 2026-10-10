from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement, product
import random

import pytest

import algorithm.queue_aware_canonical_integer_capacity_frame_candidates as queue_bound
from algorithm.integer_capacity_shared_threshold_frame_candidates import IntegerCapacityThresholdSelection


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def expanded_key(counts, lanes):
    indices = tuple(i for i, count in enumerate(counts) for _ in range(count))
    return indices + (len(counts),) * (lanes - len(indices))


@pytest.mark.parametrize("requirements", (
    ((2, 1), (1, 2), (1, 1)),
    ((0, 2), (2, 0), (1, 1)),
    ((1, 1), (1, 1), (2, 2)),
    ((0, 0), (1, 0), (0, 1)),
))
def test_resource_lex_bound_matches_entire_small_feasible_set(requirements):
    # Independently enumerate each box. A lexicographically too-large bound
    # could wrongly prune an optimal canonical action.
    for lower in product(range(2), repeat=3):
        reserved = tuple(sum(count * req[c] for count, req in zip(lower, requirements))
                         for c in range(2))
        for extra in product(range(3), repeat=3):
            upper = tuple(left + right for left, right in zip(lower, extra))
            for spare in product((0, 1, 3), repeat=2):
                queue = tuple(left + right for left, right in zip(reserved, spare))
                for free_lanes in (0, 1, 3):
                    lanes = sum(lower) + free_lanes
                    feasible = [counts for counts in product(*(range(a, b + 1)
                                for a, b in zip(lower, upper)))
                                if sum(counts) <= lanes and all(
                                    sum(n * req[c] for n, req in zip(counts, requirements)) <= q
                                    for c, q in enumerate(queue))]
                    expected = min(feasible, key=lambda counts: expanded_key(counts, lanes))
                    actual = queue_bound._queue_feasible_lex_counts(
                        lower, upper, requirements, queue, lanes)
                    assert actual == expected


def test_joint_queue_bound_excludes_individually_feasible_but_jointly_impossible_key():
    # Both upper counts are individually feasible; allocating both uses six
    # jobs of class 0 although only four are present.
    args = ((0, 0), (2, 2), ((2, 0), (1, 1)), (4, 4), 4)
    actual = queue_bound._queue_feasible_lex_counts(*args)
    assert actual == (2, 0)
    assert expanded_key(actual, 4) > expanded_key((2, 2), 4)
    # A mandatory later count must reserve its resources before the first fill.
    assert queue_bound._queue_feasible_lex_counts(
        (0, 1), (2, 2), ((2, 0), (1, 1)), (4, 4), 4) == (1, 2)


def oracle(options, queue, lanes, idle_duration=1.0):
    idle = len(options)
    best = None
    for indices in combinations_with_replacement(range(idle + 1), lanes):
        service = tuple(sum(options[i].requirements[c] for i in indices if i != idle)
                        for c in range(len(queue)))
        if any(s > q for s, q in zip(service, queue)):
            continue
        duration = max((Fraction.from_float(float(idle_duration)),
                        *(Fraction.from_float(float(options[i].duration))
                          for i in indices if i != idle)))
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        key = (-score, duration, indices)
        if best is None or key < best:
            best = key
    return best


def assert_certificate(result, expected, queue, option_count, budget):
    selected = result.selection
    duration = Fraction.from_float(selected.duration)
    score = Fraction(sum(q * s for q, s in zip(queue, selected.service))) / duration
    actual = (-score, duration, tuple(option_count if i is None else i
                                     for i in selected.option_indices))
    upper = score if selected.score_exact else Fraction.from_float(selected.upper_bound)
    assert score <= -expected[0] <= upper
    assert selected.lp_calls <= budget
    if selected.score_exact:
        assert score == -expected[0]
    if selected.canonical_exact:
        assert selected.score_exact and actual == expected
    assert selected.exact == selected.canonical_exact
    assert selected.option_counts == tuple(selected.option_indices.count(i)
                                           for i in range(option_count))


@pytest.mark.parametrize("case", range(12))
def test_full_selector_matches_exhaustive_original_actions(case):
    rng = random.Random(49100 + case)
    options = [Option(req, rng.choice((0.5, 1.0, 2.0, 3.0, 5.0)))
               for req in ((1, 0), (0, 2), (1, 1), (2, 1)) for _ in range(2)]
    rng.shuffle(options)
    queue = tuple(rng.randrange(9) for _ in range(2))
    lanes, idle = (2, 3, 4)[case % 3], (1.0, 2.0)[case % 2]
    result = queue_bound.select_queue_aware_canonical_integer_capacity_frame(
        options, queue, lanes, 4096, idle_duration=idle)
    assert_certificate(result, oracle(options, queue, lanes, idle), queue, len(options), 4096)
    assert result.selection.canonical_exact
    assert isinstance(result, IntegerCapacityThresholdSelection)


@pytest.mark.parametrize("queue", ((0, 0), (1, 1), (2, 3), (4, 4)))
@pytest.mark.parametrize("idle", (1.0, 2.0))
def test_original_representative_order_changes_with_duration(queue, idle):
    # Group A uses index 3 below duration 4, but index 0 at duration 4;
    # group B similarly moves from index 2 to index 1 at duration 3.
    options = (Option((1, 0), 4.0), Option((0, 1), 3.0),
               Option((0, 1), 1.0), Option((1, 0), 1.0),
               Option((1, 1), 2.0), Option((1, 1), 2.0))
    result = queue_bound.select_queue_aware_canonical_integer_capacity_frame(
        options, queue, 3, 4096, idle_duration=idle)
    assert_certificate(result, oracle(options, queue, 3, idle), queue, len(options), 4096)
    assert result.selection.canonical_exact


def budget_options():
    return (Option((1, 0), 5.0), Option((0, 1), 2.0), Option((2, 1), 5.0),
            Option((1, 2), 3.0), Option((3, 0), 2.0), Option((0, 3), 3.0))


@pytest.mark.parametrize("failed_solver", (False, True))
def test_budget_cutoff_and_failed_solver_keep_certificates_honest(monkeypatch, failed_solver):
    options, queue, lanes = budget_options(), (5, 6), 4
    expected = oracle(options, queue, lanes)
    calls = []
    original = queue_bound.linprog

    def counted_solver(*args, **kwargs):
        calls.append(1)
        if failed_solver:
            raise RuntimeError("LP unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_bound, "linprog", counted_solver)
    for budget in (3, 4, 5, 8, 16):
        calls.clear()
        result = queue_bound.select_queue_aware_canonical_integer_capacity_frame(
            options, queue, lanes, budget)
        assert_certificate(result, expected, queue, len(options), budget)
        selected = result.selection
        assert selected.lp_calls == len(calls)
        assert selected.dual_certified_lp_calls + selected.safe_fallback_lp_calls == len(calls)
        if failed_solver:
            assert selected.successful_lp_calls == 0
            assert selected.safe_fallback_lp_calls == len(calls)
    calls.clear()
    with pytest.raises(ValueError, match="cover 3 duration thresholds"):
        queue_bound.select_queue_aware_canonical_integer_capacity_frame(options, queue, lanes, 2)
    assert not calls


def test_queue_bound_work_includes_final_snapshot(monkeypatch):
    original = queue_bound._queue_feasible_lex_counts
    observed_sizes = []

    def observed(lower, upper, requirements, queue, lanes):
        observed_sizes.append(len(requirements) * len(queue))
        return original(lower, upper, requirements, queue, lanes)

    monkeypatch.setattr(queue_bound, "_queue_feasible_lex_counts", observed)
    options, queue = budget_options(), (1, 6)
    result = queue_bound.select_queue_aware_canonical_integer_capacity_frame(options, queue, 4, 3)
    assert_certificate(result, oracle(options, queue, 4), queue, len(options), 3)
    work = result.canonical_queue_bound
    assert work.bound_calls == len(observed_sizes) > 0
    assert work.requirement_coordinate_reads == 3 * sum(observed_sizes)
    # The legacy count omits account=False calls in the final snapshot.
    assert work.bound_calls > result.selection.canonical_key_bound_evaluations


def test_zero_queue_still_pays_all_duration_roots():
    options = (Option((3, 0), 2.0), Option((0, 2), 3.0), Option((2, 1), 5.0))
    result = queue_bound.select_queue_aware_canonical_integer_capacity_frame(options, (0, 0), 4, 3)
    assert result.selection.lp_calls == result.selection.duration_root_count == 3
    assert result.selection.option_indices == (None,) * 4
    assert result.selection.score_exact and result.selection.canonical_exact
