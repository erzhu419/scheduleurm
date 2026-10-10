from dataclasses import dataclass
from fractions import Fraction
import random

import pytest

import algorithm.canonical_count_space_bnb_frame_candidates as canonical
from algorithm.experiments.aligned_hardware_physical_catalogue_screen import enumerate_frames, oracle
from algorithm.threshold_canonical_requirement_dominance_frame_candidates import (
    select_threshold_canonical_requirement_dominance_frame as select,
)


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


@pytest.mark.parametrize("case", range(24))
def test_group_orders_and_tied_minima_match_full_catalogue_oracle(case):
    rng = random.Random(7300 + case)
    options = [Option(requirement, rng.choice((0.5, 1.0, 2.0, 3.0, 5.0)))
               for requirement in ((1, 0), (0, 1), (1, 1), (2, 0)) for _ in range(2)]
    rng.shuffle(options)
    queues = tuple(rng.randrange(9) for _ in range(2))
    lanes = rng.choice((2, 3, 4))
    expected = oracle(enumerate_frames(options, lanes), queues)
    result = select(options, queues, lanes, 4096)
    chosen = result.selection
    score = Fraction(sum(q * s for q, s in zip(queues, chosen.service, strict=True))) / Fraction.from_float(chosen.duration)
    indices = tuple(len(options) if i is None else i for i in chosen.option_indices)
    assert (-score, Fraction.from_float(chosen.duration), indices) == expected
    assert chosen.exact and chosen.score_exact and chosen.canonical_exact
    assert chosen.option_counts == tuple(indices.count(i) for i in range(len(options)))
    assert chosen.lp_calls <= sum(result.threshold_budgets) == 4096
    assert chosen.duration_root_count == len(result.thresholds)


def test_failed_lp_calls_spend_the_budget_and_retain_a_valid_full_upper_bound(monkeypatch):
    options = (Option((1, 0), 4.0), Option((1, 0), 2.0),
               Option((0, 1), 3.0), Option((2, 1), 5.0))
    queues, lanes = (5, 5), 4
    expected = -oracle(enumerate_frames(options, lanes), queues)[0]
    calls = []

    def failed_lp(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("LP unavailable")

    monkeypatch.setattr(canonical, "linprog", failed_lp)
    result = select(options, queues, lanes, 3)
    chosen = result.selection
    score = Fraction(sum(q * s for q, s in zip(queues, chosen.service, strict=True))) / Fraction.from_float(chosen.duration)
    upper = score if chosen.score_exact else Fraction.from_float(chosen.upper_bound)
    assert score <= expected <= upper
    assert len(calls) == chosen.lp_calls == chosen.safe_fallback_lp_calls == 3
    assert result.threshold_budgets == (1, 1, 1)
    assert not chosen.canonical_exact
    calls.clear()
    with pytest.raises(ValueError, match="cover 3 duration thresholds"):
        select(options, queues, lanes, 2)
    assert not calls


def test_empty_catalogue_and_zero_queue_keep_the_idle_action():
    empty = select((), (0, 3), 2, 0).selection
    assert empty.option_counts == ()
    assert empty.option_indices == (None, None)
    assert empty.canonical_exact and empty.lp_calls == 0
    zero = select((Option((1,), 0.5), Option((1,), 0.25)), (0,), 3, 16, idle_duration=2).selection
    assert zero.option_counts == (0, 0)
    assert zero.option_indices == (None, None, None)
    assert zero.duration == 2 and zero.score == 0 and zero.canonical_exact
