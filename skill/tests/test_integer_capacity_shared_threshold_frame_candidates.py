from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement, product
import random

import pytest

import algorithm.integer_capacity_shared_threshold_frame_candidates as capacity
from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_physical_catalogue_models import load_catalogues


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def feasible_counts(lower, upper, requirements, queues, lanes):
    """Enumerate the integer set without using the propagation formula."""
    return {
        counts
        for counts in product(*(range(left, right + 1)
                                for left, right in zip(lower, upper, strict=True)))
        if sum(counts) <= lanes
        and all(sum(count * requirement[c]
                    for count, requirement in zip(counts, requirements, strict=True)) <= q
                for c, q in enumerate(queues))
    }


@pytest.mark.parametrize("case", range(24))
def test_integer_propagation_preserves_entire_feasible_set(case):
    rng = random.Random(37100 + case)
    requirements = ((2, 1, 0), (0, 3, 1), (1, 0, 2), (1, 1, 1))
    lower = tuple(rng.randrange(2) for _ in requirements)
    upper = tuple(left + rng.randrange(5) for left in lower)
    queues = tuple(sum(count * requirement[c]
                       for count, requirement in zip(lower, requirements, strict=True))
                   + rng.randrange(7) for c in range(3))
    lanes = sum(lower) + rng.randrange(5)
    tightened = capacity._integer_capacity_upper_bounds(lower, upper, requirements, queues, lanes)
    assert all(left <= new <= old for left, new, old in zip(lower, tightened, upper, strict=True))
    assert feasible_counts(lower, upper, requirements, queues, lanes) == feasible_counts(
        lower, tightened, requirements, queues, lanes)


def test_capacity_reserves_all_mandatory_lower_counts():
    # Mandatory copies consume three jobs of class 0, four of class 1, and two lanes.
    assert capacity._integer_capacity_upper_bounds(
        (1, 1, 0), (5, 5, 5), ((2, 1), (1, 3), (0, 2)), (7, 9), 5
    ) == (3, 2, 2)
    # Eight queued jobs admit two whole batches of three, not a fractional third.
    assert capacity._integer_capacity_upper_bounds((0,), (5,), ((3,),), (8,), 5) == (2,)


def oracle(options, queues, lanes, idle_duration=1.0):
    """Enumerate original multisets, independently of the quotient/search."""
    idle = len(options)
    best = None
    for indices in combinations_with_replacement(range(idle + 1), lanes):
        service = tuple(sum(options[i].requirements[c] for i in indices if i != idle)
                        for c in range(len(queues)))
        if any(s > q for s, q in zip(service, queues, strict=True)):
            continue
        duration = max((Fraction.from_float(float(idle_duration)),
                        *(Fraction.from_float(float(options[i].duration))
                          for i in indices if i != idle)))
        score = Fraction(sum(q * s for q, s in zip(queues, service, strict=True))) / duration
        key = (-score, duration, indices)
        if best is None or key < best:
            best = key
    return best


def selection_key(selected, queues, option_count):
    score = Fraction(sum(q * s for q, s in zip(queues, selected.service, strict=True))) / Fraction.from_float(selected.duration)
    indices = tuple(option_count if i is None else i for i in selected.option_indices)
    return -score, Fraction.from_float(selected.duration), indices


def assert_certificate(selected, expected, queues, option_count, budget):
    actual = selection_key(selected, queues, option_count)
    actual_score, optimum = -actual[0], -expected[0]
    upper = actual_score if selected.score_exact else Fraction.from_float(selected.upper_bound)
    assert actual_score <= optimum <= upper
    assert selected.lp_calls <= budget
    if selected.score_exact:
        assert actual_score == optimum
    if selected.canonical_exact:
        assert selected.score_exact and actual == expected
    assert selected.option_counts == tuple(selected.option_indices.count(i) for i in range(option_count))


@pytest.mark.parametrize("case", range(24))
def test_capacity_search_matches_original_multiset_oracle(case):
    rng = random.Random(48100 + case)
    options = [Option(requirements, rng.choice((0.5, 1.0, 2.0, 3.0, 5.0)))
               for requirements in ((1, 0), (0, 2), (1, 1), (2, 1)) for _ in range(2)]
    rng.shuffle(options)
    queues = tuple(rng.randrange(9) for _ in range(2))
    lanes, idle_duration = (2, 3, 4)[case % 3], (1.0, 2.0)[case % 2]
    chosen = capacity.select_integer_capacity_shared_threshold_frame(
        options, queues, lanes, 4096, idle_duration=idle_duration).selection
    expected = oracle(options, queues, lanes, idle_duration)
    assert_certificate(chosen, expected, queues, len(options), 4096)
    assert chosen.exact and chosen.score_exact and chosen.canonical_exact


def test_reversed_duration_representatives_preserve_original_canonical_action():
    model = next(model for model in load_catalogues() if model.node == "node007")
    queues, lanes = (1, 0, 4), 2
    chosen = capacity.select_integer_capacity_shared_threshold_frame(model.options, queues, lanes, 4096).selection
    assert_certificate(chosen, oracle(model.options, queues, lanes), queues, len(model.options), 4096)
    assert chosen.canonical_exact


@pytest.mark.parametrize("failed_solver", (False, True))
def test_budget_cutoff_certificates_and_failed_call_accounting(monkeypatch, failed_solver):
    options = (Option((1, 0), 5.0), Option((0, 1), 2.0), Option((2, 1), 5.0),
               Option((1, 2), 3.0), Option((3, 0), 2.0), Option((0, 3), 3.0))
    queues, lanes = (5, 6), 4
    expected = oracle(options, queues, lanes)
    calls = []
    original_solver = capacity.linprog

    def counted_solver(*args, **kwargs):
        calls.append(1)
        if failed_solver:
            raise RuntimeError("LP unavailable")
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(capacity, "linprog", counted_solver)
    for budget in (3, 4, 5, 8, 16):
        calls.clear()
        chosen = capacity.select_integer_capacity_shared_threshold_frame(options, queues, lanes, budget).selection
        assert_certificate(chosen, expected, queues, len(options), budget)
        assert chosen.lp_calls == len(calls)
        assert chosen.dual_certified_lp_calls + chosen.safe_fallback_lp_calls == chosen.lp_calls
        if failed_solver:
            assert chosen.successful_lp_calls == 0
            assert chosen.safe_fallback_lp_calls == chosen.lp_calls
    calls.clear()
    with pytest.raises(ValueError, match="cover 3 duration thresholds"):
        capacity.select_integer_capacity_shared_threshold_frame(options, queues, lanes, 2)
    assert not calls


def test_propagation_work_counters_record_actual_bound_changes(monkeypatch):
    original = capacity._integer_capacity_upper_bounds
    calls = []

    def observed(lower, upper, requirements, queue, lanes):
        after = original(lower, upper, requirements, queue, lanes)
        calls.append((tuple(upper), after))
        return after

    monkeypatch.setattr(capacity, "_integer_capacity_upper_bounds", observed)
    options = (Option((3, 0), 2.0), Option((0, 2), 3.0), Option((2, 1), 2.0))
    result = capacity.select_integer_capacity_shared_threshold_frame(options, (5, 3), 4, 4096)
    work = result.capacity_propagation
    changes = [old - new for before, after in calls for old, new in zip(before, after, strict=True)]
    assert work.region_visits == len(calls) > 0
    assert work.requirement_coordinate_reads == 2 * 2 * sum(len(before) for before, _ in calls) > 0
    assert work.tightened_bounds == sum(change > 0 for change in changes) > 0
    assert work.removed_count_values == sum(changes) >= work.tightened_bounds
    assert_certificate(result.selection, oracle(options, (5, 3), 4), (5, 3), len(options), 4096)


def test_zero_queue_still_pays_each_duration_root_with_tightened_lp_bounds(monkeypatch):
    options = (Option((3, 0), 2.0), Option((0, 2), 3.0), Option((2, 1), 5.0))
    original_solver = capacity.linprog
    bounds = []

    def observed_solver(*args, **kwargs):
        variable_count = len(args[0])
        bounds.append(tuple(kwargs["b_ub"][-variable_count:]))
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(capacity, "linprog", observed_solver)
    chosen = capacity.select_integer_capacity_shared_threshold_frame(options, (0, 0), 4, 3).selection
    assert chosen.duration_root_count == chosen.lp_calls == len(bounds) == 3
    assert all(all(value == 0 for value in root_bounds) for root_bounds in bounds)
    assert chosen.option_indices == (None,) * 4
    assert chosen.score_exact and chosen.canonical_exact


@pytest.mark.parametrize(("node", "lanes", "queues", "budget"), (
    ("gpu_a", 8, (6, 6, 37), 288),
    ("gpu_b", 16, (35, 32, 77), 576),
))
def test_real_development_witness_certificates_are_honest(node, lanes, queues, budget):
    model = next(model for model in load_catalogues() if model.node == node)
    reference = select_canonical_count_space_bnb_frame(model.options, queues, lanes, 4096)
    assert reference.score_exact and reference.canonical_exact
    chosen = capacity.select_integer_capacity_shared_threshold_frame(model.options, queues, lanes, budget).selection
    expected = selection_key(reference, queues, len(model.options))
    assert_certificate(chosen, expected, queues, len(model.options), budget)
    print(f"\n{node} R={lanes} Q={queues} budget={budget}: "
          f"score_exact={chosen.score_exact}, canonical_exact={chosen.canonical_exact}, "
          f"action_matches={selection_key(chosen, queues, len(model.options)) == expected}, "
          f"lp_calls={chosen.lp_calls}, remaining={chosen.frontier_regions_remaining}")
