from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
import random

import pytest

import algorithm.shared_threshold_canonical_requirement_dominance_frame_candidates as shared
from algorithm.experiments.aligned_hardware_physical_catalogue_models import load_catalogues
from algorithm.threshold_canonical_requirement_dominance_frame_candidates import (
    select_threshold_canonical_requirement_dominance_frame as independent,
)


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def oracle(options, queues, lanes, idle_duration=1.0):
    """Enumerate original lane multisets without any search/reduction helpers."""
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


def assert_exact(selected, options, queues, lanes, idle_duration=1.0):
    assert selection_key(selected, queues, len(options)) == oracle(options, queues, lanes, idle_duration)
    assert selected.exact and selected.score_exact and selected.canonical_exact
    assert selected.option_counts == tuple(selected.option_indices.count(i) for i in range(len(options)))


@pytest.mark.parametrize("case", range(24))
def test_shared_search_matches_original_catalogue_oracle(case):
    rng = random.Random(17300 + case)
    options = [Option(requirements, rng.choice((0.5, 1.0, 2.0, 3.0, 5.0)))
               for requirements in ((1, 0), (0, 1), (1, 1), (2, 0)) for _ in range(2)]
    rng.shuffle(options)
    queues = tuple(rng.randrange(9) for _ in range(2))
    lanes = (2, 3, 4)[case % 3]
    idle_duration = (1.0, 2.0)[case % 2]
    result = shared.select_shared_threshold_canonical_requirement_dominance_frame(
        options, queues, lanes, 4096, idle_duration=idle_duration)
    assert_exact(result.selection, options, queues, lanes, idle_duration)
    assert result.selection.lp_calls <= 4096
    assert result.thresholds == tuple(sorted({max(idle_duration, min(o.duration for o in options
        if o.requirements == requirements)) for requirements in {o.requirements for o in options}}))


def test_node007_reversed_dominance_retains_original_tie_order():
    model = next(model for model in load_catalogues() if model.node == "node007")
    queues, lanes = (1, 0, 4), 2
    result = shared.select_shared_threshold_canonical_requirement_dominance_frame(model.options, queues, lanes, 4096)
    assert_exact(result.selection, model.options, queues, lanes)


def test_shared_incumbent_prunes_inferior_threshold_branches():
    options = (Option((1, 0), 1.0), Option((0, 1), 4.0),
               Option((1, 1), 5.0), Option((2, 0), 8.0))
    queues, lanes = (10, 1), 3
    result = shared.select_shared_threshold_canonical_requirement_dominance_frame(options, queues, lanes, 4096)
    previous = independent(options, queues, lanes, 4096)
    assert_exact(result.selection, options, queues, lanes)
    assert result.selection.lp_calls < previous.selection.lp_calls
    assert result.selection.lp_matrix_coefficients < previous.selection.lp_matrix_coefficients


def test_lower_duration_residual_does_not_close_a_live_threshold_bound():
    options = (Option((1, 0), 3.0), Option((1, 0), 1.0),
               Option((2, 1), 2.0), Option((0, 1), 3.0))
    queues, lanes = (3, 3), 2
    stopped = shared.select_shared_threshold_canonical_requirement_dominance_frame(options, queues, lanes, 3).selection
    expected = oracle(options, queues, lanes)
    assert selection_key(stopped, queues, len(options)) == expected
    assert stopped.upper_bound > float(-expected[0])
    assert stopped.frontier_regions_remaining > 0
    assert not stopped.score_exact and not stopped.canonical_exact
    closed = shared.select_shared_threshold_canonical_requirement_dominance_frame(options, queues, lanes, 4096).selection
    assert_exact(closed, options, queues, lanes)
    assert closed.lp_calls > stopped.lp_calls


@pytest.mark.parametrize("failed_solver", (False, True))
def test_budget_cutoffs_preserve_oracle_coverage_and_charge_solver_calls(monkeypatch, failed_solver):
    options = (Option((1, 0), 5.0), Option((0, 1), 2.0), Option((2, 1), 5.0),
               Option((1, 2), 3.0), Option((3, 0), 2.0), Option((0, 3), 3.0))
    queues, lanes = (5, 6), 4
    expected = oracle(options, queues, lanes)
    calls = []
    original_solver = shared.linprog

    def counted_solver(*args, **kwargs):
        calls.append(1)
        if failed_solver:
            raise RuntimeError("LP unavailable")
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(shared, "linprog", counted_solver)
    for budget in (3, 4, 5, 8):
        calls.clear()
        chosen = shared.select_shared_threshold_canonical_requirement_dominance_frame(options, queues, lanes, budget).selection
        actual_score = -selection_key(chosen, queues, len(options))[0]
        upper = actual_score if chosen.score_exact else Fraction.from_float(chosen.upper_bound)
        assert actual_score <= -expected[0] <= upper
        assert len(calls) == chosen.lp_calls == budget
        assert chosen.frontier_regions_remaining > 0
        assert not chosen.score_exact and not chosen.canonical_exact
        if failed_solver:
            assert chosen.safe_fallback_lp_calls == chosen.lp_calls
            assert chosen.successful_lp_calls == 0
    calls.clear()
    with pytest.raises(ValueError, match="cover 3 duration thresholds"):
        shared.select_shared_threshold_canonical_requirement_dominance_frame(options, queues, lanes, 2)
    assert calls == []


def test_empty_catalogue_and_zero_queue_select_idle():
    empty = shared.select_shared_threshold_canonical_requirement_dominance_frame((), (0, 3), 2, 0).selection
    assert empty.option_counts == () and empty.option_indices == (None, None)
    assert empty.canonical_exact and empty.lp_calls == 0
    options = (Option((1,), 0.5), Option((1,), 0.25))
    zero = shared.select_shared_threshold_canonical_requirement_dominance_frame(options, (0,), 3, 16, idle_duration=2).selection
    assert_exact(zero, options, (0,), 3, idle_duration=2)
    assert zero.option_indices == (None, None, None)
