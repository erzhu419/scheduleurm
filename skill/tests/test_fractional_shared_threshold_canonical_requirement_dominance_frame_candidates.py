from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
import random

import pytest

import algorithm.fractional_shared_threshold_canonical_requirement_dominance_frame_candidates as fractional
from algorithm.canonical_count_space_bnb_frame_candidates import (
    _Region,
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_physical_catalogue_models import load_catalogues


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def oracle(options, queues, lanes, idle_duration=1.0):
    """Enumerate original multisets independently of quotient/search helpers."""
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
def test_fractional_branching_matches_independent_original_multiset_oracle(case):
    rng = random.Random(21800 + case)
    options = [Option(requirements, rng.choice((0.5, 1.0, 2.0, 3.0, 5.0)))
               for requirements in ((1, 0), (0, 1), (1, 1), (2, 0)) for _ in range(2)]
    rng.shuffle(options)
    queues = tuple(rng.randrange(9) for _ in range(2))
    lanes = (2, 3, 4)[case % 3]
    idle_duration = (1.0, 2.0)[case % 2]
    chosen = fractional.select_fractional_shared_threshold_canonical_requirement_dominance_frame(
        options, queues, lanes, 4096, idle_duration=idle_duration).selection
    expected = oracle(options, queues, lanes, idle_duration)
    assert_certificate(chosen, expected, queues, len(options), 4096)
    assert chosen.exact and chosen.score_exact and chosen.canonical_exact


def test_reversed_duration_dominance_keeps_original_canonical_action():
    model = next(model for model in load_catalogues() if model.node == "node007")
    queues, lanes = (1, 0, 4), 2
    chosen = fractional.select_fractional_shared_threshold_canonical_requirement_dominance_frame(
        model.options, queues, lanes, 4096).selection
    assert_certificate(chosen, oracle(model.options, queues, lanes), queues, len(model.options), 4096)
    assert chosen.canonical_exact


@pytest.mark.parametrize("failed_solver", (False, True))
def test_budget_cutoff_certificates_cover_oracle_and_charge_failed_calls(monkeypatch, failed_solver):
    options = (Option((1, 0), 5.0), Option((0, 1), 2.0), Option((2, 1), 5.0),
               Option((1, 2), 3.0), Option((3, 0), 2.0), Option((0, 3), 3.0))
    queues, lanes = (5, 6), 4
    expected = oracle(options, queues, lanes)
    calls = []
    original_solver = fractional.linprog

    def counted_solver(*args, **kwargs):
        calls.append(1)
        if failed_solver:
            raise RuntimeError("LP unavailable")
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(fractional, "linprog", counted_solver)
    for budget in (3, 4, 5, 8, 16):
        calls.clear()
        chosen = fractional.select_fractional_shared_threshold_canonical_requirement_dominance_frame(
            options, queues, lanes, budget).selection
        assert_certificate(chosen, expected, queues, len(options), budget)
        assert chosen.lp_calls == len(calls)
        assert chosen.dual_certified_lp_calls + chosen.safe_fallback_lp_calls == chosen.lp_calls
        if failed_solver:
            assert chosen.successful_lp_calls == 0
            assert chosen.safe_fallback_lp_calls == chosen.lp_calls
    calls.clear()
    with pytest.raises(ValueError, match="cover 3 duration thresholds"):
        fractional.select_fractional_shared_threshold_canonical_requirement_dominance_frame(options, queues, lanes, 2)
    assert not calls


@pytest.mark.parametrize(("lower", "upper", "primal", "expected"), (
    ((0, 0, 0), (3, 3, 3), (1.0, 1.5, 2.5), 1),
    ((1, 0, 0), (1, 3, 3), (1.5, 2.0, 0.5), 2),
    ((0, 0, 0), (3, 3, 3), (1.0, 2.0, 0.0), 0),
    ((0, 0, 0), (3, 3, 3), (1.0 + 1e-8, 1.0 + 1e-6, 0.0), 1),
    ((1, 0, 0), (1, 3, 3), None, 1),
    ((1, 2, 3), (1, 2, 3), (1.0, 2.0, 3.0), None),
))
def test_fractional_priority_is_stable_and_ignores_fixed_counts(lower, upper, primal, expected):
    region = _Region(0, Fraction(1), (0,), (0, 1, 2), lower, upper,
                     Fraction(100), True, primal)
    assert fractional._branch_position(region) == expected


@pytest.mark.parametrize(("node", "lanes", "queues", "budget"), (
    ("gpu_a", 8, (6, 6, 37), 288),
    ("gpu_b", 16, (35, 32, 77), 576),
))
def test_real_confirmation_regressions_have_honest_frozen_budget_certificates(node, lanes, queues, budget):
    model = next(model for model in load_catalogues() if model.node == node)
    reference = select_canonical_count_space_bnb_frame(model.options, queues, lanes, 4096)
    assert reference.score_exact and reference.canonical_exact
    chosen = fractional.select_fractional_shared_threshold_canonical_requirement_dominance_frame(
        model.options, queues, lanes, budget).selection
    expected = selection_key(reference, queues, len(model.options))
    assert_certificate(chosen, expected, queues, len(model.options), budget)
    assert selection_key(chosen, queues, len(model.options)) == expected
    if node == "gpu_b":
        assert chosen.score_exact and chosen.canonical_exact
    print(f"\n{node} R={lanes} Q={queues} budget={budget}: "
          f"score_exact={chosen.score_exact}, canonical_exact={chosen.canonical_exact}, "
          f"action_matches={selection_key(chosen, queues, len(model.options)) == expected}, "
          f"lp_calls={chosen.lp_calls}, remaining={chosen.frontier_regions_remaining}")
