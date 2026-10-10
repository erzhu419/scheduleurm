from fractions import Fraction
from itertools import product
import random
from types import SimpleNamespace

import pytest

from algorithm.experiments.integer_capacity_canonical_oracle import (
    canonical_at_certified_score, certified_score_from_selection,
)


def exhaustive(options, queue, lanes, idle=1.0):
    feasible = []
    for counts in product(range(lanes + 1), repeat=len(options)):
        if sum(counts) > lanes:
            continue
        service = tuple(sum(c * o.requirements[j] for c, o in zip(counts, options))
                        for j in range(len(queue)))
        if any(s > q for s, q in zip(service, queue)):
            continue
        duration = Fraction.from_float(max([idle] + [float(o.duration) for c, o in zip(counts, options) if c]))
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        indices = tuple(i for i, c in enumerate(counts) for _ in range(c)) + (None,) * (lanes - sum(counts))
        key = duration, tuple(len(options) if i is None else i for i in indices)
        feasible.append(((-score, key), counts, service, indices))
    (_, key), counts, service, indices = min(feasible)
    score = -min(feasible)[0][0]
    return score, dict(option_counts=list(counts), option_indices=list(indices), service=list(service),
                       duration=float(key[0]), score=float(score), score_exact=True)


def solve(options, queue, lanes, score, maximum_duration):
    return canonical_at_certified_score(options, queue, lanes, score,
                                       state_visit_budget=10000, maximum_duration=maximum_duration)


def test_independent_exhaustive_small_models_cover_joint_resources_and_original_indices():
    rng = random.Random(2759)
    for _ in range(64):
        options = []
        for _ in range(rng.randint(1, 5)):
            requirements = rng.choice(((1, 0), (0, 1), (2, 0), (0, 2), (1, 1)))
            options.append(SimpleNamespace(requirements=requirements,
                                           duration=rng.choice((0.5, 1.0, 1.25, 2.0, 2.1))))
        queue = (rng.randint(0, 5), rng.randint(0, 5))
        lanes = rng.randint(1, 3)
        score, expected = exhaustive(options, queue, lanes)
        result = solve(options, queue, lanes, score, max(o.duration for o in options) + 1.0)
        assert result["complete"] and result["canonical_exact"]
        assert result["conditional_on_external_score_certificate"]
        for key in ("option_counts", "option_indices", "service", "duration", "score"):
            assert result[key] == expected[key]
        assert certified_score_from_selection(options, queue, lanes, expected) == score


def test_earlier_duration_wins_before_original_index_order():
    options = [SimpleNamespace(requirements=(2,), duration=2.0),
               SimpleNamespace(requirements=(1,), duration=1.0)]
    result = solve(options, (3,), 1, Fraction(3), 2.0)
    assert result["duration"] == 1.0
    assert result["option_counts"] == [0, 1]


def test_duplicate_original_options_keep_first_index_and_idle_last():
    options = [SimpleNamespace(requirements=(1,), duration=1.0)] * 2
    result = solve(options, (2,), 3, Fraction(4), 1.0)
    assert result["option_counts"] == [2, 0]
    assert result["option_indices"] == [0, 0, None]
    assert result["canonical_key"] == [1.0, [0, 0, 2]]


def test_noninteger_earlier_threshold_is_exactly_excluded():
    options = [SimpleNamespace(requirements=(1,), duration=1.1),
               SimpleNamespace(requirements=(2,), duration=2.0)]
    result = solve(options, (2,), 1, Fraction(2), 2.0)
    assert result["duration_fraction"] == [2, 1]
    assert result["work"]["noninteger_thresholds"] == 1


@pytest.mark.parametrize("budget", [0, 1])
def test_budget_exhaustion_never_returns_canonical_proof_or_action(budget):
    options = [SimpleNamespace(requirements=(1,), duration=1.0)]
    result = canonical_at_certified_score(options, (2,), 2, Fraction(4),
                                         state_visit_budget=budget, maximum_duration=1.0)
    assert result["status"] == "STATE_BUDGET_EXHAUSTED"
    assert not result["complete"] and not result["canonical_exact"]
    assert "option_counts" not in result
    assert result["work"]["states_visited"] == budget


def test_inconsistent_score_reports_exhausted_feasible_set_without_proof():
    options = [SimpleNamespace(requirements=(1,), duration=1.0)]
    result = solve(options, (1,), 1, Fraction(2), 1.0)
    assert result["complete"] and not result["canonical_exact"]
    assert result["status"] == "NO_ACTION_AT_CERTIFIED_SCORE"


@pytest.mark.parametrize("field,value", [("score_exact", False), ("service", [0]),
                                         ("score", 3.0), ("option_indices", [None, 0]),
                                         ("option_counts", [3]), ("duration", 2.0)])
def test_external_score_helper_rejects_incomplete_or_inconsistent_incumbent(field, value):
    options = [SimpleNamespace(requirements=(1,), duration=1.0)]
    _, selected = exhaustive(options, (2,), 2)
    selected[field] = value
    with pytest.raises(ValueError):
        certified_score_from_selection(options, (2,), 2, selected)
