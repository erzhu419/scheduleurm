"""Budget and certificate transitions for strict-score proof plus completion."""
from dataclasses import dataclass, replace
import random

import pytest

import algorithm.integer_capacity_shared_threshold_frame_candidates as stage_one
import algorithm.score_face_canonical_completion_frame_candidates as frozen
import algorithm.strict_score_proof_canonical_completion_frame_candidates as candidate


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def unresolved_stage(options, queue, lanes, *, score_exact=False, idle=1.0):
    before = stage_one.select_integer_capacity_shared_threshold_frame(options, queue, lanes, 4096, idle)
    assert before.selection.canonical_exact
    baseline = replace(before.selection, exact=False, canonical_exact=False, score_exact=score_exact,
                       upper_bound=before.selection.score if score_exact else before.selection.score + 1.0,
                       gap=0.0 if score_exact else 1.0, epsilon=0.0 if score_exact else 1.0,
                       canonical_tie_regions_remaining=1, frontier_regions_remaining=2,
                       inherited_unsolved_regions=1)
    return replace(before, selection=baseline)


@pytest.mark.parametrize("budget", (0, 1, 2, 3))
def test_proof_and_completion_share_budget_and_keep_intermediate_certificate(monkeypatch, budget):
    options, queue = (Option((1,), 1.0),), (1,)
    before = unresolved_stage(options, queue, 1)
    calls = []

    def baseline(*args):
        calls.append(args)
        return before

    monkeypatch.setattr(candidate, "select_integer_capacity_shared_threshold_frame", baseline)
    result = candidate.select_strict_score_proof_canonical_completion_frame(options, queue, 1, 4096, budget)
    assert len(calls) == 1
    assert result.baseline_selection == before.selection
    assert result.integer_state_budget == budget
    assert result.score_proof.states_visited + result.completion.states_visited == budget
    assert result.completion.state_budget == budget - result.score_proof.states_visited
    assert result.selection.lp_calls == before.selection.lp_calls
    if budget == 0:
        assert result.score_proof.status == "STATE_BUDGET_EXHAUSTED"
        assert result.selection == result.score_certified_selection == before.selection
        assert result.completion.status == "SKIPPED_SCORE_UNPROVEN"
        assert not result.score_proof.complete
        assert result.score_proof.score_fraction == (1, 1)
        assert result.completion.score_fraction is None
    else:
        assert result.score_proof.status == "COMPLETE" and result.score_proof.complete
        intermediate = replace(before.selection, score_exact=True, upper_bound=before.selection.score,
                               gap=0.0, epsilon=0.0)
        assert result.score_certified_selection == intermediate
        assert result.score_proof.witness is None
        if budget < 3:
            assert result.completion.status == "STATE_BUDGET_EXHAUSTED"
            assert result.selection == intermediate
            assert not result.selection.canonical_exact
            assert result.selection.frontier_regions_remaining == before.selection.frontier_regions_remaining
        else:
            assert result.completion.status == "COMPLETE"
            assert result.selection.exact and result.selection.canonical_exact and result.selection.score_exact
            assert result.selection.option_counts == before.selection.option_counts
            assert result.selection.frontier_regions_remaining == 0


@pytest.mark.parametrize("case", range(12))
def test_score_certified_path_exactly_preserves_frozen_completion(monkeypatch, case):
    rng = random.Random(88513 + case)
    options = tuple(Option(req, rng.choice((0.5, 1.0, 1.25, 2.0)))
                    for req in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (1, 0)))
    queue = (rng.randrange(5), rng.randrange(5))
    lanes, idle = 2 + case % 2, (1.0, 1.5)[case % 2]
    before = unresolved_stage(options, queue, lanes, score_exact=True, idle=idle)
    monkeypatch.setattr(candidate, "select_integer_capacity_shared_threshold_frame", lambda *args: before)
    monkeypatch.setattr(frozen, "select_integer_capacity_shared_threshold_frame", lambda *args: before)
    # Include exhausted paths as well as completed paths.
    budget = (0, 1, 10000)[case % 3]
    old = frozen.select_score_face_canonical_completion_frame(options, queue, lanes, 4096, budget, idle)
    new = candidate.select_strict_score_proof_canonical_completion_frame(options, queue, lanes, 4096, budget, idle)
    assert new.selection == old.selection
    assert new.baseline_selection == new.score_certified_selection == old.baseline_selection
    assert new.preprocessing == old.preprocessing
    assert new.capacity_propagation == old.capacity_propagation
    assert new.completion == old.completion
    assert new.score_proof.status == "SKIPPED_SCORE_CERTIFIED"
    assert new.score_proof.complete and not new.score_proof.attempted
    assert new.score_proof.states_visited == 0 and new.score_proof.score_fraction is None


def test_closed_baseline_skips_both_integer_stages(monkeypatch):
    options, queue = (Option((1,), 1.0),), (1,)
    before = stage_one.select_integer_capacity_shared_threshold_frame(options, queue, 1, 4096)
    monkeypatch.setattr(candidate, "select_integer_capacity_shared_threshold_frame", lambda *args: before)
    new = candidate.select_strict_score_proof_canonical_completion_frame(options, queue, 1, 4096, 0)
    assert new.selection == new.baseline_selection == new.score_certified_selection == before.selection
    assert new.score_proof.status == new.completion.status == "SKIPPED_CANONICAL_COMPLETE"
    assert new.score_proof.complete and new.completion.complete
    assert not new.score_proof.attempted and not new.completion.attempted
    assert new.score_proof.states_visited == new.completion.states_visited == 0
    assert new.score_proof.score_fraction is None


def test_actual_wrapper_spends_only_one_lp_budget(monkeypatch):
    options = (Option((2, 0), 1.0), Option((0, 2), 1.0), Option((1, 1), 1.0))
    linprog = stage_one.linprog
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return linprog(*args, **kwargs)

    monkeypatch.setattr(stage_one, "linprog", counted)
    new = candidate.select_strict_score_proof_canonical_completion_frame(options, (3, 3), 3, 1, 1000)
    assert len(calls) == new.selection.lp_calls == new.baseline_selection.lp_calls == 1
    assert new.score_proof.states_visited + new.completion.states_visited <= 1000


@pytest.mark.parametrize("budget", (-1, 1.5))
def test_invalid_budget_rejected_before_stage_one(monkeypatch, budget):
    def unexpected(*args):
        pytest.fail("invalid integer budget must not spend an LP call")

    monkeypatch.setattr(candidate, "select_integer_capacity_shared_threshold_frame", unexpected)
    with pytest.raises(ValueError, match="integer_state_budget"):
        candidate.select_strict_score_proof_canonical_completion_frame((Option((1,), 1.0),), (1,), 1, 1, budget)
