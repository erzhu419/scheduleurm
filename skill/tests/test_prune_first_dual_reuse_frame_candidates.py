"""Check skipped dual arithmetic, live canonical ties, and frozen continuation."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.dual_reuse_integer_capacity_frame_candidates as previous
import algorithm.prune_first_dual_reuse_integer_capacity_frame_candidates as engine
from algorithm.prune_first_residual_capacity_frame_candidates import select_prune_first_residual_capacity_frontier_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (Option((3, 2), 1.0), Option((2, 0), 1.0))
QUEUE, LANES = (4, 3), 3


def original_action_key(options, queue, lanes):
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        active = [options[i] for i in indices if i < len(options)]
        service = tuple(sum(option.requirements[c] for option in active) for c in range(len(queue)))
        if any(done > waiting for done, waiting in zip(service, queue)):
            continue
        duration = max((Fraction(1), *(Fraction.from_float(option.duration) for option in active)))
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        keys.append((-score, duration, indices))
    return min(keys)


def selection_key(selected, option_count):
    return (-Fraction.from_float(selected.score), Fraction.from_float(selected.duration),
            tuple(option_count if i is None else i for i in selected.option_indices))


def fractional_root():
    return SimpleNamespace(success=True, x=(1.0, 0.5),
                           ineqlin=SimpleNamespace(marginals=(0.0, -4.0, 0.0, -6.0, 0.0)))


def test_inherited_canonical_prune_skips_dual_and_keeps_better_score_equal_child(monkeypatch):
    options = (Option((1, 0), 1.0), Option((0, 1), 1.0))
    calls, reused = [], []
    original_reuse = engine._reuse_dual_upper

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        # The first optimal LP action uses original index 1. The other child
        # has the same score and improves the original-index canonical key.
        primal = (0.0, 1.0) if len(calls) == 1 else (0.0, 0.0)
        return SimpleNamespace(success=True, x=primal,
                               ineqlin=SimpleNamespace(marginals=(-1.0, 0.0, 0.0, 0.0, 0.0)))

    def reuse(lower, upper, *args):
        reused.append((tuple(lower), tuple(upper)))
        return original_reuse(lower, upper, *args)

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "_reuse_dual_upper", reuse)
    search = engine.PruneFirstDualReuseIntegerCapacitySearch(options, (1, 1), 1)
    initial = search.advance_to(1)
    assert initial.selection.score_exact and not initial.selection.canonical_exact
    assert initial.selection.option_indices == (1,)
    final = search.advance_to(2)
    # Box (0, 0)..(0, 1) is already canonically closed. Only the better box
    # reaches parent-dual evaluation, whose unchanged bound needs no recheck.
    assert reused == [((1, 0), (1, 0))]
    assert final.dual_reuse.child_bound_evaluations == 1
    assert final.dual_reuse.child_bound_improvements == final.dual_reuse.child_bound_prunes == 0
    assert final.dual_reuse.child_rhs_entries == 5
    assert final.dual_reuse.child_requirement_reads == 4
    assert final.dual_reuse.child_weight_reads == 2
    assert final.selection.canonical_key_bound_evaluations == 6
    assert final.selection.lp_calls == len(calls) == 2
    assert final.selection.canonical_exact and final.selection.option_indices == (0,)
    assert selection_key(final.selection, len(options)) == original_action_key(options, (1, 1), 1)
    assert initial.dual_reuse.child_bound_evaluations == 0
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.DualReuseIntegerCapacitySearch(options, (1, 1), 1).advance_to(2)
    assert old.selection == final.selection
    assert old.dual_reuse.child_bound_evaluations == 2
    assert old.dual_reuse.child_rhs_entries == 10


def test_strictly_tightened_dual_still_prunes_a_surviving_child(monkeypatch):
    real_lp, calls = engine.linprog, []

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        return fractional_root() if len(calls) == 1 else real_lp(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    search = engine.PruneFirstDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    initial = search.advance_to(1)
    assert initial.selection.score == 18.0 and initial.selection.upper_bound == 22.0
    final = search.advance_to(2)
    work = final.dual_reuse
    assert work.child_bound_evaluations == 2
    assert work.child_bound_improvements == work.child_bound_prunes == 1
    assert work.child_rhs_entries == 10 and work.child_requirement_reads == 8
    assert work.child_weight_reads == 4
    assert final.selection.lp_calls == len(calls) == 2
    assert final.selection.canonical_exact
    assert selection_key(final.selection, len(OPTIONS)) == original_action_key(OPTIONS, QUEUE, LANES)


def test_failed_child_solve_keeps_the_inherited_certificate_and_charges_attempts(monkeypatch):
    calls = []

    def lp(*args, **kwargs):
        calls.append(tuple(kwargs["b_ub"]))
        return fractional_root() if len(calls) == 1 else SimpleNamespace(success=False)

    monkeypatch.setattr(engine, "linprog", lp)
    search = engine.PruneFirstDualReuseIntegerCapacitySearch(OPTIONS, QUEUE, LANES)
    for cap in (1, 2, 128):
        final = search.advance_to(cap)
        assert final.selection.lp_calls == len(calls)
        assert final.selection.lp_calls + final.root_pruning.roots_seen - final.root_pruning.roots_solved <= cap
    assert final.dual_reuse.certificates_saved == 1
    assert final.dual_reuse.child_bound_evaluations >= 2
    assert final.selection.safe_fallback_lp_calls == len(calls) - 1
    assert final.selection.canonical_exact
    assert selection_key(final.selection, len(OPTIONS)) == original_action_key(OPTIONS, QUEUE, LANES)


def test_resume_retains_lp_sequence_preprocessing_and_immutable_snapshots(monkeypatch):
    real_lp, reduce = engine.linprog, engine.reduce_requirement_dominated_options
    calls, reductions = [], []

    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        return real_lp(*args, **kwargs)

    def counted_reduce(*args, **kwargs):
        reductions.append(1)
        return reduce(*args, **kwargs)

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(engine, "reduce_requirement_dominated_options", counted_reduce)
    options = (*OPTIONS, Option((3, 2), 1.25), Option((0, 1), 2.0))
    direct = engine.PruneFirstDualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    direct_calls = calls[:]
    calls.clear()
    reductions.clear()
    search = engine.PruneFirstDualReuseIntegerCapacitySearch(options, QUEUE, LANES)
    snapshots = []
    for cap in (2, 3, 4, 8, 64):
        final = search.advance_to(cap)
        snapshots.append((final, asdict(final)))
        assert search.advance_to(cap) is final
        assert final.selection.lp_calls == len(calls) and len(reductions) == 1
    assert final == direct and calls == direct_calls
    assert all(asdict(stage) == saved for stage, saved in snapshots)
    assert final.selection.canonical_exact
    assert selection_key(final.selection, len(options)) == original_action_key(options, QUEUE, LANES)
    with pytest.raises(ValueError, match="nondecreasing"):
        search.advance_to(8)
    calls.clear()
    monkeypatch.setattr(previous, "linprog", lp)
    old = previous.DualReuseIntegerCapacitySearch(options, QUEUE, LANES).advance_to(64)
    assert calls == direct_calls
    assert (final.selection.option_counts, final.selection.service, final.selection.duration,
            final.selection.score, final.selection.upper_bound, final.selection.canonical_exact) == (
                old.selection.option_counts, old.selection.service, old.selection.duration,
                old.selection.score, old.selection.upper_bound, old.selection.canonical_exact)
    assert final.dual_reuse.child_bound_evaluations <= old.dual_reuse.child_bound_evaluations


@pytest.mark.parametrize("options,queue,lanes,initial_cap", (
    (OPTIONS, QUEUE, LANES, 1),
    ((Option((1, 0), 2.0), Option((1, 0), 1.0), Option((0, 2), 2.0)), (1, 2), 2, 2),
    (OPTIONS, (0, 0), LANES, 1),
))
def test_residual_router_keeps_original_action_and_exact_proof_shape(options, queue, lanes, initial_cap):
    routed = select_prune_first_residual_capacity_frontier_frame(options, queue, lanes, initial_cap, 64, 8192)
    assert routed.selection.canonical_exact and routed.selection.score_exact
    assert selection_key(routed.selection, len(options)) == original_action_key(options, queue, lanes)
    assert isinstance(routed.initial_lp, previous.DualReuseIntegerCapacityThresholdSelection)
    assert isinstance(routed.final_lp.dual_reuse, previous.DualReuseWork)
    assert routed.selection.lp_calls <= routed.final_lp.selection.lp_calls <= 64


def test_open_frontier_remains_exportable_and_reports_a_real_strict_improvement(monkeypatch):
    monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    options = (Option((1, 0), 1.0), Option((3, 0), 1.0),
               Option((0, 1), 1.0), Option((0, 3), 1.0))
    routed = select_prune_first_residual_capacity_frontier_frame(options, (4, 4), 3, 1, 2, 8192)
    assert routed.frontier_evidence is not None and routed.frontier_evidence.regions
    assert routed.probe_stage == "frontier_late" and routed.status == "FRONTIER_PROBE_INCOMPLETE"
    assert routed.probe.score_proof.status == "BETTER_ACTION_FOUND" and not routed.probe.score_proof.complete
    assert routed.probe.completion.status == "SKIPPED_SCORE_UNPROVEN"
    assert not routed.selection.score_exact and not routed.selection.canonical_exact
    assert routed.probe.score_proof.states_visited <= 8192
    assert routed.selection.lp_calls == 2
