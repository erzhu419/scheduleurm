"""Routing, certificate validity, and cumulative work for one LP frontier."""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import combinations_with_replacement

import pytest

import algorithm.certificate_routed_frame_candidates as candidate
import algorithm.integer_capacity_shared_threshold_frame_candidates as frozen
import algorithm.resumable_integer_capacity_frame_candidates as engine
import algorithm.resumable_probe_frame_candidates as probe_module
import algorithm.strict_score_proof_canonical_completion_frame_candidates as proof_module


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def snapshot(*, cap, calls, complete=False, score_exact=False):
    raw = frozen.select_integer_capacity_shared_threshold_frame(
        (Option((1,), 1.0),), (1,), 1, 4096)
    assert raw.selection.canonical_exact
    selected = replace(
        raw.selection, lp_node_budget=cap, lp_calls=calls,
        exact=complete, canonical_exact=complete, score_exact=complete or score_exact,
        upper_bound=1.0 if complete or score_exact else 2.0,
        gap=0.0 if complete or score_exact else 1.0,
        epsilon=0.0 if complete or score_exact else 1.0,
        frontier_regions_remaining=0 if complete else 2,
        inherited_unsolved_regions=0 if complete else 1,
        canonical_tie_regions_remaining=0 if complete else 1,
    )
    return replace(raw, selection=selected)


def install_search(monkeypatch, initial, final=None):
    constructions, caps = [], []

    class Search:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

        def advance_to(self, cap):
            caps.append(cap)
            if len(caps) == 1:
                return initial
            assert len(caps) == 2, "continue the existing frontier at most once"
            assert final is not None, "completed initial selection must not resume"
            return final

    monkeypatch.setattr(candidate, "IntegerCapacitySearch", Search)
    return constructions, caps


def forbid_score_proof_and_cold_restart(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("routed selector must not run strict integer score proof or restart LP")

    monkeypatch.setattr(probe_module, "_prove_no_strict_improvement", unexpected)
    monkeypatch.setattr(frozen, "select_integer_capacity_shared_threshold_frame", unexpected)
    monkeypatch.setattr(proof_module, "select_integer_capacity_shared_threshold_frame", unexpected)


def select(budget):
    return candidate.select_certificate_routed_frame(
        (Option((1,), 1.0),), (1,), 1, 1, 8, budget)


def assert_canonical_only_probe(result, stage, budget):
    assert result.probe is not None
    assert result.probe.baseline_selection is stage.selection
    assert result.probe.preprocessing is stage.preprocessing
    assert result.probe.capacity_propagation is stage.capacity_propagation
    assert result.probe.score_proof.status == "SKIPPED_SCORE_CERTIFIED"
    assert not result.probe.score_proof.attempted
    assert result.probe.score_proof.states_visited == 0
    assert result.probe.completion.state_budget == budget
    assert result.probe.completion.states_visited <= budget


def test_initial_completion_skips_both_probe_and_resume(monkeypatch):
    initial = snapshot(cap=1, calls=1, complete=True)
    constructions, caps = install_search(monkeypatch, initial)
    forbid_score_proof_and_cold_restart(monkeypatch)

    def unexpected(*args):
        pytest.fail("complete LP certificate must skip the integer probe")

    monkeypatch.setattr(candidate, "_probe", unexpected)
    result = select(8192)
    assert len(constructions) == 1 and caps == [1]
    assert result.selection is initial.selection
    assert result.initial_lp is result.final_lp is initial
    assert result.status == "LP_INITIAL_COMPLETE"
    assert result.probe_stage == "none" and result.probe is None and not result.resumed
    assert (result.initial_lp_budget, result.lp_node_budget, result.integer_state_budget) == (1, 8, 8192)
    assert asdict(result)["probe_stage"] == "none"


def test_unproved_score_resumes_without_integer_work(monkeypatch):
    initial = snapshot(cap=1, calls=1)
    final = snapshot(cap=8, calls=6, complete=True)
    constructions, caps = install_search(monkeypatch, initial, final)
    forbid_score_proof_and_cold_restart(monkeypatch)

    def unexpected(*args):
        pytest.fail("unproved initial score and complete final LP must skip integer work")

    monkeypatch.setattr(candidate, "_probe", unexpected)
    result = select(8192)
    assert len(constructions) == 1 and caps == [1, 8]
    assert result.selection is final.selection and result.final_lp is final
    assert result.status == "LP_RESUME_COMPLETE" and result.resumed
    assert result.probe_stage == "none" and result.probe is None
    assert result.selection.lp_calls == 6, "final count already contains the initial LP prefix"


def test_initial_certified_score_completes_canonical_probe_without_resume(monkeypatch):
    initial = snapshot(cap=1, calls=1, score_exact=True)
    constructions, caps = install_search(monkeypatch, initial)
    forbid_score_proof_and_cold_restart(monkeypatch)
    result = select(2)
    assert len(constructions) == 1 and caps == [1]
    assert result.status == "INTEGER_PROBE_COMPLETE" and not result.resumed
    assert result.probe_stage == "initial"
    assert result.initial_lp is result.final_lp is initial
    assert result.selection.canonical_exact and result.selection.score_exact
    assert result.selection.lp_calls == 1
    assert_canonical_only_probe(result, initial, 2)
    assert result.probe.completion.complete and result.probe.completion.states_visited == 2


@pytest.mark.parametrize("final_complete", (False, True))
def test_exhausted_initial_probe_resumes_once_and_is_not_retried(monkeypatch, final_complete):
    initial = snapshot(cap=1, calls=1, score_exact=True)
    final = snapshot(cap=8, calls=6 if final_complete else 8,
                     complete=final_complete, score_exact=True)
    constructions, caps = install_search(monkeypatch, initial, final)
    forbid_score_proof_and_cold_restart(monkeypatch)
    original_probe, probe_calls = candidate._probe, []

    def probe(*args):
        probe_calls.append(args)
        return original_probe(*args)

    monkeypatch.setattr(candidate, "_probe", probe)
    result = select(1)
    assert len(constructions) == 1 and caps == [1, 8] and len(probe_calls) == 1
    assert result.resumed and result.probe_stage == "initial"
    assert result.selection is final.selection and result.final_lp is final
    assert result.status == ("LP_RESUME_COMPLETE" if final_complete else "LP_RESUME_INCOMPLETE")
    assert_canonical_only_probe(result, initial, 1)
    assert result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"
    assert result.probe.completion.states_visited == 1
    assert result.probe.selection.lp_calls == 1
    assert result.selection.lp_calls == final.selection.lp_calls


@pytest.mark.parametrize("budget,complete", ((0, False), (1, False), (2, True)))
def test_newly_certified_resumed_score_gets_one_full_canonical_budget(monkeypatch, budget, complete):
    initial = snapshot(cap=1, calls=1)
    final = snapshot(cap=8, calls=8, score_exact=True)
    constructions, caps = install_search(monkeypatch, initial, final)
    forbid_score_proof_and_cold_restart(monkeypatch)
    original_probe, probe_calls = candidate._probe, []

    def probe(*args):
        assert caps == [1, 8], "unproved initial score must resume before the probe"
        probe_calls.append(args)
        return original_probe(*args)

    monkeypatch.setattr(candidate, "_probe", probe)
    result = select(budget)
    assert len(constructions) == 1 and caps == [1, 8] and len(probe_calls) == 1
    assert result.resumed and result.probe_stage == "resumed"
    assert result.final_lp is final and result.initial_lp is initial
    assert result.selection.canonical_exact is complete
    assert result.selection.score_exact
    assert result.status == ("INTEGER_PROBE_COMPLETE" if complete else "LP_RESUME_INCOMPLETE")
    assert result.selection.lp_calls == 8 and result.selection.lp_node_budget == 8
    assert_canonical_only_probe(result, final, budget)
    assert result.probe.completion.states_visited == budget
    assert result.probe.completion.complete is complete


def test_budget_exhaustion_preserves_unproved_final_score_and_gap(monkeypatch):
    initial = snapshot(cap=1, calls=1)
    final = snapshot(cap=8, calls=8)
    _, caps = install_search(monkeypatch, initial, final)
    forbid_score_proof_and_cold_restart(monkeypatch)

    def unexpected(*args):
        pytest.fail("an unproved final score must not enter canonical completion")

    monkeypatch.setattr(candidate, "_probe", unexpected)
    result = select(8192)
    assert caps == [1, 8] and result.status == "LP_RESUME_INCOMPLETE"
    assert result.probe is None and result.probe_stage == "none" and result.resumed
    assert result.selection is final.selection
    assert not result.selection.score_exact and not result.selection.canonical_exact
    assert result.selection.upper_bound == 2.0 and result.selection.gap == 1.0
    assert result.selection.frontier_regions_remaining == 2


def test_real_frontier_preprocesses_once_and_counts_the_lp_prefix_once(monkeypatch):
    options = (Option((2, 1), 1.0), Option((1, 2), 1.0))
    queue, lanes = (4, 4), 3
    expected = frozen.select_integer_capacity_shared_threshold_frame(options, queue, lanes, 16)
    original_lp, original_reduce = engine.linprog, engine.reduce_requirement_dominated_options
    counts = {"lp": 0, "preprocessing": 0}

    def lp(*args, **kwargs):
        counts["lp"] += 1
        return original_lp(*args, **kwargs)

    def reduce(*args, **kwargs):
        counts["preprocessing"] += 1
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "reduce_requirement_dominated_options", reduce)
    forbid_score_proof_and_cold_restart(monkeypatch)
    result = candidate.select_certificate_routed_frame(options, queue, lanes, 1, 16, 0)
    assert result.resumed and result.status == "LP_RESUME_COMPLETE"
    assert counts["preprocessing"] == 1
    assert counts["lp"] == result.final_lp.selection.lp_calls == result.selection.lp_calls
    assert result.final_lp == expected
    assert result.initial_lp.selection.lp_calls == 1


def exhaustive_key(options, queue, lanes, idle):
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        active = [options[i] for i in indices if i < len(options)]
        service = tuple(sum(option.requirements[c] for option in active) for c in range(len(queue)))
        if any(done > waiting for done, waiting in zip(service, queue)):
            continue
        duration = max([Fraction.from_float(idle)] + [Fraction.from_float(o.duration) for o in active])
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        keys.append((-score, duration, indices))
    return min(keys)


@pytest.mark.parametrize("queue", ((0, 0), (1, 3), (3, 3), (4, 2)))
@pytest.mark.parametrize("caps", ((1, 1, 0), (1, 4, 0), (1, 16, 1), (1, 64, 1000)))
def test_small_real_models_match_exhaustive_bounds_and_canonical_actions(monkeypatch, queue, caps):
    options = (Option((2, 0), 1.0), Option((0, 2), 1.0), Option((1, 1), 1.0),
               Option((1, 0), 1.25), Option((1, 0), 1.0), Option((0, 1), 0.5))
    lanes, idle = 3, 1.0
    forbid_score_proof_and_cold_restart(monkeypatch)
    result = candidate.select_certificate_routed_frame(options, queue, lanes, *caps, idle_duration=idle)
    selected, expected = result.selection, exhaustive_key(options, queue, lanes, idle)
    actual_score = Fraction(sum(q * s for q, s in zip(queue, selected.service))) / Fraction.from_float(selected.duration)
    upper = actual_score if selected.score_exact else Fraction.from_float(selected.upper_bound)
    assert actual_score <= -expected[0] <= upper
    assert selected.lp_calls == result.final_lp.selection.lp_calls <= caps[1]
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
        stage = result.initial_lp if result.probe_stage == "initial" else result.final_lp
        assert_canonical_only_probe(result, stage, caps[2])
    else:
        assert result.probe_stage == "none"


@pytest.mark.parametrize("caps", ((-1, 8, 1), (1, -1, 1), (1, 8, -1), (2, 1, 1), (1, 8, 0.5)))
def test_invalid_budgets_fail_before_starting_search(monkeypatch, caps):
    def unexpected(*args, **kwargs):
        pytest.fail("reject invalid budget before LP construction")

    monkeypatch.setattr(candidate, "IntegerCapacitySearch", unexpected)
    with pytest.raises(ValueError):
        candidate.select_certificate_routed_frame((Option((1,), 1.0),), (1,), 1, *caps)


def test_initial_budget_must_cover_duration_roots():
    with pytest.raises(ValueError, match="LP budget must cover 1 duration thresholds"):
        candidate.select_certificate_routed_frame((Option((1,), 1.0),), (1,), 1, 0, 8, 2)
