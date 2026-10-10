"""Keep one pruned LP frontier and at most one certificate-only integer probe."""
from dataclasses import asdict, dataclass, replace

import pytest

import algorithm.root_pruned_certificate_routed_frame_candidates as candidate
import algorithm.root_pruned_integer_capacity_frame_candidates as engine
import algorithm.resumable_probe_frame_candidates as probe_module


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def snapshot(*, cap, calls, complete=False, score_exact=False):
    raw = engine.RootPrunedIntegerCapacitySearch((Option((1,), 1.0),), (1,), 1).advance_to(64)
    selected = replace(raw.selection, lp_node_budget=cap, lp_calls=calls,
                       exact=complete, canonical_exact=complete,
                       score_exact=complete or score_exact,
                       upper_bound=1.0 if complete or score_exact else 2.0,
                       gap=0.0 if complete or score_exact else 1.0,
                       epsilon=0.0 if complete or score_exact else 1.0,
                       frontier_regions_remaining=0 if complete else 2,
                       inherited_unsolved_regions=0 if complete else 1,
                       canonical_tie_regions_remaining=0 if complete else 1)
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
            assert len(caps) == 2 and final is not None
            return final

    monkeypatch.setattr(candidate, "RootPrunedIntegerCapacitySearch", Search)
    return constructions, caps


def forbid_score_proof(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a certificate-routed probe must not run strict integer score proof")

    monkeypatch.setattr(probe_module, "_prove_no_strict_improvement", unexpected)


def select(budget):
    return candidate.select_root_pruned_certificate_routed_frame(
        (Option((1,), 1.0),), (1,), 1, 1, 8, budget)


def test_complete_initial_lp_preserves_root_work_and_skips_later_stages(monkeypatch):
    initial = snapshot(cap=1, calls=1, complete=True)
    constructions, caps = install_search(monkeypatch, initial)

    def unexpected(*args, **kwargs):
        pytest.fail("complete initial LP must skip the probe")

    monkeypatch.setattr(candidate, "_probe", unexpected)
    result = select(8192)
    assert len(constructions) == 1 and caps == [1]
    assert result.initial_lp is result.final_lp is initial
    assert result.selection is initial.selection
    assert result.status == "LP_INITIAL_COMPLETE" and not result.resumed
    assert result.probe_stage == "none" and result.probe is None
    assert asdict(result)["final_lp"]["root_pruning"]["roots_solved"] == 1


def test_unproved_initial_score_resumes_without_an_integer_probe(monkeypatch):
    initial = snapshot(cap=1, calls=1)
    final = snapshot(cap=8, calls=6, complete=True)
    constructions, caps = install_search(monkeypatch, initial, final)
    forbid_score_proof(monkeypatch)

    def unexpected(*args, **kwargs):
        pytest.fail("unproved score must go to LP continuation")

    monkeypatch.setattr(candidate, "_probe", unexpected)
    result = select(8192)
    assert len(constructions) == 1 and caps == [1, 8]
    assert result.final_lp is final and result.selection is final.selection
    assert result.status == "LP_RESUME_COMPLETE" and result.resumed
    assert result.probe is None and result.probe_stage == "none"
    assert result.selection.lp_calls == 6


@pytest.mark.parametrize("score_certified_initially", (False, True))
def test_canonical_probe_is_attempted_at_most_once_at_the_certified_stage(
        monkeypatch, score_certified_initially):
    initial = snapshot(cap=1, calls=1, score_exact=score_certified_initially)
    final = snapshot(cap=8, calls=6, score_exact=True)
    constructions, caps = install_search(monkeypatch, initial, final)
    forbid_score_proof(monkeypatch)
    original_probe, probe_calls = candidate._probe, []

    def probe(*args):
        probe_calls.append(args)
        return original_probe(*args)

    monkeypatch.setattr(candidate, "_probe", probe)
    # One state is insufficient for canonical completion, forcing continuation
    # after an initial probe and keeping the resumed-stage probe incomplete.
    result = select(1)
    assert len(constructions) == 1 and caps == [1, 8]
    assert len(probe_calls) == 1 and result.resumed
    assert result.status == "LP_RESUME_INCOMPLETE"
    assert result.probe_stage == ("initial" if score_certified_initially else "resumed")
    stage = initial if score_certified_initially else final
    assert result.probe.baseline_selection is stage.selection
    assert result.probe.preprocessing is stage.preprocessing
    assert result.probe.capacity_propagation is stage.capacity_propagation
    assert result.probe.score_proof.status == "SKIPPED_SCORE_CERTIFIED"
    assert result.probe.score_proof.states_visited == 0
    assert result.probe.completion.states_visited == 1
    assert result.probe.completion.state_budget == 1
    assert result.selection.lp_calls == 6
    assert result.final_lp.root_pruning is final.root_pruning


def test_real_wrapper_preprocesses_once_and_charges_one_cumulative_lp_prefix(monkeypatch):
    original_lp = engine.linprog
    original_reduce = engine.reduce_requirement_dominated_options
    counts = {"lp": 0, "preprocessing": 0}

    def lp(*args, **kwargs):
        counts["lp"] += 1
        return original_lp(*args, **kwargs)

    def reduction(*args, **kwargs):
        counts["preprocessing"] += 1
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "reduce_requirement_dominated_options", reduction)
    forbid_score_proof(monkeypatch)
    options = (Option((2, 1), 1.0), Option((1, 2), 1.0))
    result = candidate.select_root_pruned_certificate_routed_frame(options, (4, 4), 3, 1, 64, 0)
    assert result.resumed and result.status == "LP_RESUME_COMPLETE"
    assert counts["preprocessing"] == 1
    assert counts["lp"] == result.final_lp.selection.lp_calls == result.selection.lp_calls
    assert result.initial_lp.selection.lp_calls == 1
    assert result.final_lp.root_pruning.roots_seen == 1
    assert result.final_lp.root_pruning.roots_solved == 1
    assert result.selection.canonical_exact


def test_registered_budgets_use_new_pruned_search_without_empty_root_lp(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("registered zero-queue route must not pay for empty roots")

    monkeypatch.setattr(engine, "linprog", unexpected)
    options = (Option((1, 0), 1.0), Option((0, 1), 2.0), Option((1, 1), 3.0))
    result = candidate.select_root_pruned_certificate_routed_frame(options, (0, 0), 8, 144, 256, 8192)
    assert result.status == "LP_INITIAL_COMPLETE" and not result.resumed
    assert result.selection.option_indices == (None,) * 8
    assert result.selection.lp_calls == 0
    assert result.initial_lp.root_pruning.roots_skipped_empty == 3
    assert (result.initial_lp_budget, result.lp_node_budget, result.integer_state_budget) == (144, 256, 8192)


@pytest.mark.parametrize("budgets", ((-1, 8, 1), (2, 1, 1), (1, 8, -1)))
def test_invalid_budgets_do_not_construct_the_lp_frontier(monkeypatch, budgets):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid cumulative budgets must fail before search construction")

    monkeypatch.setattr(candidate, "RootPrunedIntegerCapacitySearch", unexpected)
    with pytest.raises(ValueError):
        candidate.select_root_pruned_certificate_routed_frame((Option((1,), 1.0),), (1,), 1, *budgets)
