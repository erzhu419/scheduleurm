"""Canonical completion against exhaustive original-action enumeration."""
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations_with_replacement
import random

import pytest

import algorithm.integer_capacity_shared_threshold_frame_candidates as stage_one_module
import algorithm.score_face_canonical_completion_frame_candidates as completion


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def enumerate_actions(options, queue, lanes, idle=1.0):
    best = None
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        service = tuple(sum(options[i].requirements[j] for i in indices if i < len(options))
                        for j in range(len(queue)))
        if any(s > q for s, q in zip(service, queue)):
            continue
        duration = max((Fraction.from_float(idle), *(Fraction.from_float(options[i].duration)
                                                    for i in indices if i < len(options))))
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        key = (-score, duration, indices)
        if best is None or key < best:
            best = key
    return best


def selection_key(selected, queue, count):
    return (-Fraction(sum(q * s for q, s in zip(queue, selected.service))) /
            Fraction.from_float(selected.duration), Fraction.from_float(selected.duration),
            tuple(count if i is None else i for i in selected.option_indices))


def open_baseline(monkeypatch, options, queue, lanes, *, counts=None, idle=1.0):
    original = stage_one_module.select_integer_capacity_shared_threshold_frame(options, queue, lanes, 4096, idle)
    selected = original.selection
    assert selected.canonical_exact
    if counts is not None:
        service = tuple(sum(c * opt.requirements[j] for c, opt in zip(counts, options))
                        for j in range(len(queue)))
        duration = max((idle, *(o.duration for o, c in zip(options, counts) if c)))
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / Fraction.from_float(duration)
        assert float(score) == selected.score
        selected = replace(selected, option_counts=counts,
                           option_indices=tuple(i for i, c in enumerate(counts) for _ in range(c)) +
                           (None,) * (lanes - sum(counts)), service=service, duration=duration)
    selected = replace(selected, exact=False, canonical_exact=False,
                       canonical_tie_regions_remaining=2, frontier_regions_remaining=2,
                       inherited_unsolved_regions=1)
    original = replace(original, selection=selected)
    monkeypatch.setattr(completion, "select_integer_capacity_shared_threshold_frame", lambda *args: original)
    return original


@pytest.mark.parametrize("case", range(18))
def test_completion_matches_exhaustive_original_actions(monkeypatch, case):
    rng = random.Random(65210 + case)
    options = tuple(Option(required, rng.choice((0.5, 1.0, 1.25, 1.75, 2.0)))
                    for required in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (1, 0)))
    queue = tuple(rng.randrange(5) for _ in range(2))
    lanes, idle = (2 + case % 2), (1.0, 1.5)[case % 2]
    before = open_baseline(monkeypatch, options, queue, lanes, idle=idle)
    chosen = completion.select_score_face_canonical_completion_frame(options, queue, lanes, 4096, 10000, idle)
    assert selection_key(chosen.selection, queue, len(options)) == enumerate_actions(options, queue, lanes, idle)
    assert chosen.completion.status == "COMPLETE"
    assert chosen.completion.complete and chosen.completion.attempted
    assert chosen.selection.exact and chosen.selection.score_exact and chosen.selection.canonical_exact
    assert chosen.completion.states_visited <= 10000
    assert chosen.baseline_selection == before.selection
    assert chosen.selection.lp_calls == before.selection.lp_calls
    assert chosen.selection.frontier_regions_remaining == chosen.selection.canonical_tie_regions_remaining == 0
    assert chosen.selection.inherited_unsolved_regions == 0


def test_tie_completion_can_change_to_earlier_duration(monkeypatch):
    options, queue = (Option((1,), 1.0), Option((2,), 2.0)), (4,)
    open_baseline(monkeypatch, options, queue, 1, counts=(0, 1))
    selected = completion.select_score_face_canonical_completion_frame(options, queue, 1, 4096, 100)
    assert selected.selection.option_counts == (1, 0)
    assert selected.selection.duration == 1.0
    assert selected.completion.action_changed
    assert selected.selection.score == selected.baseline_selection.score == 4.0


def test_original_index_precedes_faster_requirement_representative(monkeypatch):
    options = (Option((1, 0), 2.0), Option((1, 0), 1.0), Option((0, 3), 2.0))
    queue = (1, 3)
    open_baseline(monkeypatch, options, queue, 2, counts=(0, 1, 1))
    result = completion.select_score_face_canonical_completion_frame(options, queue, 2, 4096, 100)
    assert result.selection.option_counts == (1, 0, 1)
    assert result.selection.duration == 2.0
    assert result.completion.action_changed


@pytest.mark.parametrize("budget", (0, 1))
def test_exhaustion_preserves_entire_baseline(monkeypatch, budget):
    options, queue = (Option((1,), 1.0), Option((2,), 2.0)), (4,)
    before = open_baseline(monkeypatch, options, queue, 1, counts=(0, 1))
    result = completion.select_score_face_canonical_completion_frame(options, queue, 1, 4096, budget)
    assert result.selection == result.baseline_selection == before.selection
    assert result.completion.status == "STATE_BUDGET_EXHAUSTED"
    assert result.completion.states_visited == budget
    assert result.completion.attempted and not result.completion.complete
    assert not result.completion.action_changed


def test_completion_exactly_at_state_cap_succeeds(monkeypatch):
    options, queue = (Option((1,), 1.0), Option((2,), 2.0)), (4,)
    open_baseline(monkeypatch, options, queue, 1, counts=(0, 1))
    result = completion.select_score_face_canonical_completion_frame(options, queue, 1, 4096, 2)
    assert result.completion.status == "COMPLETE"
    assert result.completion.states_visited == 2


@pytest.mark.parametrize("closed", (False, True))
def test_only_score_proven_canonical_unresolved_enters_integer_stage(monkeypatch, closed):
    options, queue = (Option((1,), 1.0),), (2,)
    before = stage_one_module.select_integer_capacity_shared_threshold_frame(options, queue, 1, 1)
    if not closed:
        before = replace(before, selection=replace(before.selection, exact=False, score_exact=False,
                                                   canonical_exact=False, upper_bound=3.0, gap=1.0))
    monkeypatch.setattr(completion, "select_integer_capacity_shared_threshold_frame", lambda *args: before)
    result = completion.select_score_face_canonical_completion_frame(options, queue, 1, 1, 0)
    assert result.selection == before.selection
    assert not result.completion.attempted
    assert result.completion.complete == closed
    assert result.completion.states_visited == result.completion.thresholds_checked == 0
    assert result.completion.status == ("SKIPPED_CANONICAL_COMPLETE" if closed else "SKIPPED_SCORE_UNPROVEN")


def test_actual_wrapper_adds_no_lp_calls(monkeypatch):
    options = (Option((2, 0), 1.0), Option((0, 2), 1.0), Option((1, 1), 1.0))
    original = stage_one_module.linprog
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(stage_one_module, "linprog", counted)
    result = completion.select_score_face_canonical_completion_frame(options, (3, 3), 3, 1, 1000)
    assert len(calls) == result.selection.lp_calls == result.baseline_selection.lp_calls == 1
    if result.selection.canonical_exact:
        assert selection_key(result.selection, (3, 3), len(options)) == enumerate_actions(options, (3, 3), 3)


@pytest.mark.parametrize("budget", (-1, 1.5))
def test_invalid_integer_budget_rejected_before_lp(monkeypatch, budget):
    def unexpected(*args):
        pytest.fail("invalid integer budget must not spend an LP call")

    monkeypatch.setattr(completion, "select_integer_capacity_shared_threshold_frame", unexpected)
    with pytest.raises(ValueError, match="canonical_state_budget"):
        completion.select_score_face_canonical_completion_frame((Option((1,), 1.0),), (1,), 1, 1, budget)


def test_impossible_claimed_score_is_not_certified(monkeypatch):
    options, queue = (Option((2,), 1.0),), (4,)
    before = open_baseline(monkeypatch, options, queue, 1)
    # A broken upstream certificate cannot turn an absent score-face solution
    # into a successful canonical certificate.
    broken = replace(before, selection=replace(before.selection, service=(3,), score=12.0))
    monkeypatch.setattr(completion, "select_integer_capacity_shared_threshold_frame", lambda *args: broken)
    with pytest.raises(ArithmeticError, match="no canonical action"):
        completion.select_score_face_canonical_completion_frame(options, queue, 1, 4096, 100)
