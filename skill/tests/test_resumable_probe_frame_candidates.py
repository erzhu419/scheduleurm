"""Certificate and accounting transitions for one LP frontier plus one probe."""
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations_with_replacement

import pytest

import algorithm.integer_capacity_shared_threshold_frame_candidates as frozen
import algorithm.resumable_integer_capacity_frame_candidates as engine
import algorithm.resumable_probe_frame_candidates as candidate
import algorithm.strict_score_proof_canonical_completion_frame_candidates as proof_module


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def staged_snapshot(options, queue, lanes, *, cap, calls, complete=False, score_exact=False):
    raw = frozen.select_integer_capacity_shared_threshold_frame(options, queue, lanes, 4096)
    assert raw.selection.canonical_exact
    selected = replace(
        raw.selection, lp_node_budget=cap, lp_calls=calls,
        exact=complete, canonical_exact=complete, score_exact=complete or score_exact,
        upper_bound=raw.selection.score if complete or score_exact else raw.selection.score + 1.0,
        gap=0.0 if complete or score_exact else 1.0,
        epsilon=0.0 if complete or score_exact else 1.0,
        frontier_regions_remaining=0 if complete else 2,
        inherited_unsolved_regions=0 if complete else 1,
        canonical_tie_regions_remaining=0 if complete else 1,
    )
    return replace(raw, selection=selected)


def install_snapshots(monkeypatch, initial, final=None):
    constructions, caps = [], []

    class Search:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

        def advance_to(self, cap):
            caps.append(cap)
            if len(caps) == 1:
                return initial
            assert final is not None, "a completed probe must not resume LP search"
            assert len(caps) == 2, "the LP frontier must not be restarted or repeatedly resumed"
            return final

    monkeypatch.setattr(candidate, "IntegerCapacitySearch", Search)
    return constructions, caps


def forbid_cold_restart(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("probe workflow must use the paid snapshot, not a cold LP selector")

    monkeypatch.setattr(frozen, "select_integer_capacity_shared_threshold_frame", unexpected)
    monkeypatch.setattr(proof_module, "select_integer_capacity_shared_threshold_frame", unexpected)


def test_initial_canonical_completion_skips_probe_and_resume(monkeypatch):
    options, queue = (Option((1,), 1.0),), (1,)
    initial = staged_snapshot(options, queue, 1, cap=4, calls=1, complete=True)
    constructions, caps = install_snapshots(monkeypatch, initial)

    def unexpected(*args, **kwargs):
        pytest.fail("canonical-complete initial LP must skip both integer stages")

    monkeypatch.setattr(candidate, "_prove_no_strict_improvement", unexpected)
    monkeypatch.setattr(candidate, "_complete_canonical", unexpected)
    forbid_cold_restart(monkeypatch)
    result = candidate.select_resumable_probe_frame(options, queue, 1, 4, 8, 20)
    assert len(constructions) == 1 and caps == [4]
    assert result.status == "LP_INITIAL_COMPLETE"
    assert result.selection == initial.selection
    assert result.initial_lp is result.final_lp is initial
    assert result.probe is None and not result.resumed
    assert result.initial_lp_budget == 4 and result.lp_node_budget == 8


def test_cheap_integer_probe_uses_initial_snapshot_and_avoids_lp_resume(monkeypatch):
    options, queue = (Option((1,), 1.0),), (1,)
    initial = staged_snapshot(options, queue, 1, cap=1, calls=1)
    constructions, caps = install_snapshots(monkeypatch, initial)
    forbid_cold_restart(monkeypatch)
    result = candidate.select_resumable_probe_frame(options, queue, 1, 1, 8, 3)
    assert len(constructions) == 1 and caps == [1]
    assert result.status == "INTEGER_PROBE_COMPLETE" and not result.resumed
    assert result.initial_lp is result.final_lp is initial
    assert result.probe.baseline_selection is initial.selection
    assert result.probe.score_proof.complete and result.probe.completion.complete
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited == 3
    assert result.selection.canonical_exact and result.selection.score_exact
    assert result.selection.lp_calls == 1 and result.selection.lp_node_budget == 1
    assert result.probe.preprocessing == initial.preprocessing
    assert result.probe.capacity_propagation == initial.capacity_propagation


@pytest.mark.parametrize("state_budget,score_exact", ((0, False), (1, False), (0, True)))
def test_exhausted_strict_or_canonical_stage_resumes_once_at_total_cap(monkeypatch, state_budget, score_exact):
    options, queue = (Option((1,), 1.0),), (1,)
    initial = staged_snapshot(options, queue, 1, cap=1, calls=1, score_exact=score_exact)
    final = staged_snapshot(options, queue, 1, cap=8, calls=6, complete=True)
    constructions, caps = install_snapshots(monkeypatch, initial, final)
    proof_calls, completion_calls = [], []
    original_proof, original_completion = candidate._prove_no_strict_improvement, candidate._complete_canonical

    def prove(*args):
        proof_calls.append(args)
        return original_proof(*args)

    def complete(*args):
        completion_calls.append(args)
        return original_completion(*args)

    monkeypatch.setattr(candidate, "_prove_no_strict_improvement", prove)
    monkeypatch.setattr(candidate, "_complete_canonical", complete)
    forbid_cold_restart(monkeypatch)
    result = candidate.select_resumable_probe_frame(options, queue, 1, 1, 8, state_budget)
    assert len(constructions) == 1 and caps == [1, 8]
    assert result.resumed and result.status == "LP_RESUME_COMPLETE"
    assert result.selection is final.selection and result.final_lp is final
    assert result.selection.lp_calls == 6, "initial LP calls are already included in the resumed counter"
    assert len(proof_calls) == int(not score_exact)
    assert len(completion_calls) == int(score_exact or state_budget > 0)
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= state_budget
    assert result.probe.completion.state_budget == state_budget - result.probe.score_proof.states_visited
    if not score_exact and state_budget == 0:
        assert result.probe.score_proof.status == "STATE_BUDGET_EXHAUSTED"
        assert result.probe.completion.status == "SKIPPED_SCORE_UNPROVEN"
    else:
        assert result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"


def test_strict_improvement_witness_is_retained_after_successful_lp_resume(monkeypatch):
    options, queue = (Option((1,), 1.0),), (2,)
    final = staged_snapshot(options, queue, 2, cap=8, calls=5, complete=True)
    initial = staged_snapshot(options, queue, 2, cap=1, calls=1)
    initial = replace(initial, selection=replace(
        initial.selection, option_counts=(1,), option_indices=(0, None), service=(1,),
        score=2.0, upper_bound=4.0, gap=2.0, epsilon=2.0))
    constructions, caps = install_snapshots(monkeypatch, initial, final)
    forbid_cold_restart(monkeypatch)
    result = candidate.select_resumable_probe_frame(options, queue, 2, 1, 8, 100)
    assert len(constructions) == 1 and caps == [1, 8]
    assert result.status == "LP_RESUME_COMPLETE" and result.selection == final.selection
    assert result.probe.score_proof.witness is not None
    assert not result.probe.score_proof.complete
    assert result.probe.score_proof.states_visited > 0
    assert result.probe.completion.status == "SKIPPED_SCORE_UNPROVEN"
    assert result.probe.baseline_selection == initial.selection


def test_unresolved_resume_keeps_honest_final_certificate_and_all_work(monkeypatch):
    options, queue = (Option((1,), 1.0),), (1,)
    initial = staged_snapshot(options, queue, 1, cap=1, calls=1)
    final = staged_snapshot(options, queue, 1, cap=8, calls=8)
    _, caps = install_snapshots(monkeypatch, initial, final)
    forbid_cold_restart(monkeypatch)
    result = candidate.select_resumable_probe_frame(options, queue, 1, 1, 8, 0)
    assert caps == [1, 8] and result.status == "LP_RESUME_INCOMPLETE"
    assert result.selection is final.selection and result.final_lp is final
    assert not result.selection.score_exact and not result.selection.canonical_exact
    assert result.selection.gap == 1.0 and result.selection.frontier_regions_remaining == 2
    assert result.selection.lp_calls == 8
    assert result.probe.score_proof.state_budget == 0


def test_proved_score_survives_incomplete_canonical_probe_and_lp_resume(monkeypatch):
    options, queue = (Option((1,), 1.0),), (1,)
    initial = staged_snapshot(options, queue, 1, cap=1, calls=1)
    final = staged_snapshot(options, queue, 1, cap=8, calls=8)
    _, caps = install_snapshots(monkeypatch, initial, final)
    forbid_cold_restart(monkeypatch)
    result = candidate.select_resumable_probe_frame(options, queue, 1, 1, 8, 1)
    assert caps == [1, 8] and result.status == "LP_RESUME_INCOMPLETE"
    assert result.probe.score_proof.complete and not result.probe.completion.complete
    assert result.probe.score_proof.score_fraction == (1, 1)
    assert result.probe.score_certified_selection.score_exact
    assert result.final_lp is final and not result.final_lp.selection.score_exact
    assert result.selection.score_exact and not result.selection.canonical_exact
    assert result.selection.upper_bound == result.selection.score and result.selection.gap == 0.0
    assert result.selection.lp_calls == 8 and result.selection.lp_node_budget == 8
    assert result.selection.frontier_regions_remaining == final.selection.frontier_regions_remaining
    assert result.initial_lp.selection.lp_calls == result.probe.selection.lp_calls == 1


def test_actual_frontier_has_one_preprocessing_pass_and_no_repeated_lp_prefix(monkeypatch):
    options = (Option((2, 1), 1.0), Option((1, 2), 1.0))
    queue, lanes = (4, 4), 3
    expected = frozen.select_integer_capacity_shared_threshold_frame(options, queue, lanes, 16)
    original_lp, original_reduce = engine.linprog, engine.reduce_requirement_dominated_options
    calls = {"lp": 0, "preprocessing": 0}

    def lp(*args, **kwargs):
        calls["lp"] += 1
        return original_lp(*args, **kwargs)

    def reduce(*args, **kwargs):
        calls["preprocessing"] += 1
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "reduce_requirement_dominated_options", reduce)
    forbid_cold_restart(monkeypatch)
    result = candidate.select_resumable_probe_frame(options, queue, lanes, 1, 16, 0)
    assert not result.initial_lp.selection.canonical_exact and result.resumed
    assert result.status == "LP_RESUME_COMPLETE"
    assert calls["preprocessing"] == 1
    assert calls["lp"] == result.final_lp.selection.lp_calls == result.selection.lp_calls
    assert result.final_lp == expected
    assert result.initial_lp.selection.lp_calls == 1


def exhaustive_key(options, queue, lanes, idle):
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        active = [options[index] for index in indices if index < len(options)]
        service = tuple(sum(option.requirements[c] for option in active) for c in range(len(queue)))
        if any(done > waiting for done, waiting in zip(service, queue)):
            continue
        duration = max([Fraction.from_float(idle)] + [Fraction.from_float(option.duration) for option in active])
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        keys.append((-score, duration, indices))
    return min(keys)


@pytest.mark.parametrize("queue", ((0, 0), (1, 3), (3, 3), (4, 2)))
@pytest.mark.parametrize("caps", ((1, 1, 0), (1, 4, 0), (1, 16, 1), (1, 64, 1000)))
def test_small_real_models_match_exhaustive_score_bounds_and_canonical_actions(queue, caps):
    options = (Option((2, 0), 1.0), Option((0, 2), 1.0), Option((1, 1), 1.0),
               Option((1, 0), 1.25), Option((1, 0), 1.0), Option((0, 1), 0.5))
    lanes, idle = 3, 1.0
    result = candidate.select_resumable_probe_frame(options, queue, lanes, *caps, idle_duration=idle)
    selected, expected = result.selection, exhaustive_key(options, queue, lanes, idle)
    actual_score = Fraction(sum(q * s for q, s in zip(queue, selected.service))) / Fraction.from_float(selected.duration)
    upper = actual_score if selected.score_exact else Fraction.from_float(selected.upper_bound)
    assert actual_score <= -expected[0] <= upper
    assert selected.lp_calls <= caps[1]
    assert selected.option_counts == tuple(selected.option_indices.count(i) for i in range(len(options)))
    if selected.score_exact:
        assert actual_score == -expected[0] and selected.gap == 0.0
    if selected.canonical_exact:
        actual_key = (-actual_score, Fraction.from_float(selected.duration),
                      tuple(len(options) if i is None else i for i in selected.option_indices))
        assert selected.score_exact and selected.exact and actual_key == expected
    if caps[1] == 64:
        assert selected.canonical_exact
    if result.probe is not None:
        assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= caps[2]


@pytest.mark.parametrize("caps", ((-1, 8, 1), (1, -1, 1), (1, 8, -1), (2, 1, 1), (1, 8, 0.5)))
def test_invalid_public_budgets_do_not_initialize_search(monkeypatch, caps):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid budgets must be rejected before starting the LP search")

    monkeypatch.setattr(candidate, "IntegerCapacitySearch", unexpected)
    with pytest.raises(ValueError):
        candidate.select_resumable_probe_frame((Option((1,), 1.0),), (1,), 1, *caps)
