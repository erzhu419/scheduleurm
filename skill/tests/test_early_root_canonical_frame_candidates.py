"""Check one paid LP search, one shared probe, and exact early completion."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.early_root_canonical_frame_candidates as router
import algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as engine
from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import select_zero_delta_affine_dual_residual_capacity_frontier_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (Option((1, 0), 1.0), Option((0, 1), 1.0))


def selection(*, score_exact, canonical_exact, cap):
    return SimpleNamespace(score_exact=score_exact, canonical_exact=canonical_exact,
        frontier_regions_remaining=0 if canonical_exact else 1, lp_node_budget=cap)


def fake_search(monkeypatch, certificates):
    constructors, advances = [], []
    stages = {cap: SimpleNamespace(selection=selection(score_exact=score, canonical_exact=canonical, cap=cap))
              for cap, (score, canonical) in certificates.items()}

    class Search:
        def __init__(self, *args, **kwargs):
            constructors.append(1)
            self._search = SimpleNamespace(gi_frame=SimpleNamespace(f_locals={"frontier": [object()]}))

        def advance_to(self, cap):
            advances.append(cap)
            return stages[cap]

    monkeypatch.setattr(router, "ZeroDeltaAffineDualReuseIntegerCapacitySearch", Search)
    return constructors, advances, stages


def probe(stage, *, complete, budget=8192):
    return SimpleNamespace(selection=selection(score_exact=True, canonical_exact=complete,
        cap=stage.selection.lp_node_budget), baseline_selection=stage.selection,
        score_proof=SimpleNamespace(status="SKIPPED_SCORE_CERTIFIED", states_visited=0),
        completion=SimpleNamespace(status="COMPLETE" if complete else "STATE_BUDGET_EXHAUSTED",
            state_budget=budget, states_visited=2 if complete else budget), integer_state_budget=budget)


def unexpected(*args, **kwargs):
    raise AssertionError("an unregistered second integer probe or search was executed")


def test_root_already_canonical_returns_raw_roots_without_probe_or_branch_lp(monkeypatch):
    constructors, advances, stages = fake_search(monkeypatch, {1: (True, True)})
    monkeypatch.setattr(router, "_probe", unexpected)
    monkeypatch.setattr(router, "_residual_frontier_probe", unexpected)
    result = router.select_early_root_canonical_frame(OPTIONS, (1, 1), 1, 144, 256, 8192)
    assert constructors == [1] and advances == [1]
    assert result.root_lp is result.initial_lp is result.final_lp is stages[1]
    assert result.root_lp_budget == 1 and result.initial_lp_budget == 144 and result.lp_node_budget == 256
    assert result.status == "LP_ROOT_COMPLETE" and result.probe is None and result.probe_stage == "none"
    assert not result.resumed and result.integer_state_budget == 8192


def test_early_success_uses_root_certificate_and_records_true_configured_budgets(monkeypatch):
    constructors, advances, stages = fake_search(monkeypatch, {1: (True, False)})
    calls = []

    def complete(options, queue, lanes, stage, budget, idle):
        calls.append((stage, budget))
        return probe(stage, complete=True, budget=budget)

    monkeypatch.setattr(router, "_probe", complete)
    monkeypatch.setattr(router, "_residual_frontier_probe", unexpected)
    result = router.select_early_root_canonical_frame(OPTIONS, (1, 1), 1, 144, 256, 8192)
    assert constructors == [1] and advances == [1] and calls == [(stages[1], 8192)]
    assert result.initial_lp is result.final_lp is result.root_lp
    assert result.root_lp_budget == 1 and result.initial_lp_budget == 144
    assert result.probe_stage == "root" and result.status == "INTEGER_PROBE_COMPLETE"
    assert result.selection.canonical_exact and not result.final_lp.selection.canonical_exact
    assert result.probe.score_proof.states_visited == 0 and result.probe.completion.states_visited == 2
    assert result.routing_preparation.root_snapshot_calls == 1
    assert result.routing_preparation.root_snapshot_heap_entries_read == 2
    assert result.routing_preparation.root_snapshot_retained_entries_read == 2


@pytest.mark.parametrize("final_canonical", (False, True))
def test_exhausted_early_probe_is_retained_and_never_retried_at_late_caps(monkeypatch, final_canonical):
    constructors, advances, stages = fake_search(monkeypatch,
        {1: (True, False), 144: (True, False), 256: (True, final_canonical)})
    calls, exhausted = [], probe(stages[1], complete=False)

    def early(*args):
        calls.append(args[3].selection.lp_node_budget)
        assert len(calls) == 1
        return exhausted

    monkeypatch.setattr(router, "_probe", early)
    monkeypatch.setattr(router, "_residual_frontier_probe", unexpected)
    result = router.select_early_root_canonical_frame(OPTIONS, (1, 1), 1, 144, 256, 8192)
    assert constructors == [1] and advances == [1, 144, 256] and calls == [1]
    assert result.probe is exhausted and result.probe_stage == "root"
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited == 8192
    assert result.initial_lp is stages[144] and result.final_lp is stages[256] and result.resumed
    assert result.status == ("LP_RESUME_COMPLETE" if final_canonical else "LP_RESUME_INCOMPLETE")


def test_exhausted_early_probe_can_finish_on_the_same_initial_lp_frontier(monkeypatch):
    constructors, advances, stages = fake_search(monkeypatch, {1: (True, False), 144: (True, True)})
    exhausted = probe(stages[1], complete=False)
    monkeypatch.setattr(router, "_probe", lambda *args: exhausted)
    result = router.select_early_root_canonical_frame(OPTIONS, (1, 1), 1, 144, 256, 8192)
    assert constructors == [1] and advances == [1, 144]
    assert result.initial_lp is result.final_lp is stages[144]
    assert result.probe is exhausted and result.probe_stage == "root"
    assert result.status == "LP_INITIAL_COMPLETE" and not result.resumed


@pytest.mark.parametrize("score_cap", (144, 256))
def test_open_root_score_retains_the_original_single_late_score_face_probe(monkeypatch, score_cap):
    constructors, advances, stages = fake_search(monkeypatch,
        {1: (False, False), 144: (score_cap == 144, False), 256: (True, False)})
    calls = []

    def complete(options, queue, lanes, stage, budget, idle):
        calls.append((stage.selection.lp_node_budget, budget))
        return probe(stage, complete=True, budget=budget)

    monkeypatch.setattr(router, "_probe", complete)
    monkeypatch.setattr(router, "_residual_frontier_probe", unexpected)
    result = router.select_early_root_canonical_frame(OPTIONS, (1, 1), 1, 144, 256, 8192)
    assert constructors == [1] and calls == [(score_cap, 8192)]
    assert advances == ([1, 144] if score_cap == 144 else [1, 144, 256])
    assert result.probe_stage == ("initial" if score_cap == 144 else "resumed")
    assert result.initial_lp is stages[144] and result.final_lp is stages[score_cap]


def test_minimum_duration_group_scan_and_true_root_budget_are_charged():
    options = (Option((1, 0), 2.0), Option((1, 0), 1.0), Option((0, 1), 2.0))
    cap, fee = router._root_budget(options, 1.0)
    assert cap == 2
    assert fee.option_visits == fee.duration_reads == fee.group_lookups == fee.group_writes == 3
    assert fee.requirement_coordinate_reads == 6 and fee.duration_comparisons == 4
    assert fee.threshold_value_reads == fee.threshold_set_insertions == 2
    assert fee.root_snapshot_calls == 0
    with pytest.raises(ValueError, match="all catalogue duration roots"):
        router.select_early_root_canonical_frame(options, (1, 1), 1, 1, 256, 8192)


def test_open_root_keeps_the_exact_lp_sequence_and_one_cold_construction(monkeypatch):
    options = (Option((3, 2), 1.0), Option((2, 0), 1.0))
    real_lp, native_type = engine.linprog, router.ZeroDeltaAffineDualReuseIntegerCapacitySearch
    calls, constructions, advances = [], [], []

    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        if len(calls) == 1:
            return SimpleNamespace(success=True, x=(1.0, 0.5),
                ineqlin=SimpleNamespace(marginals=(0.0, -4.0, 0.0, -6.0, 0.0)))
        return real_lp(*args, **kwargs)

    class CountedSearch(native_type):
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            super().__init__(*args, **kwargs)

        def advance_to(self, cap):
            advances.append(cap)
            return super().advance_to(cap)

    monkeypatch.setattr(engine, "linprog", lp)
    monkeypatch.setattr(router, "ZeroDeltaAffineDualReuseIntegerCapacitySearch", CountedSearch)
    new = router.select_early_root_canonical_frame(options, (4, 3), 3, 2, 64, 8192)
    actual_calls = calls[:]
    assert constructions == [1] and advances == [1, 2]
    assert not new.root_lp.selection.score_exact and new.probe is None
    calls.clear()
    old = select_zero_delta_affine_dual_residual_capacity_frontier_frame(options, (4, 3), 3, 2, 64, 8192)
    assert calls == actual_calls and asdict(new.final_lp) == asdict(old.final_lp)
    assert new.selection == old.selection


def test_open_root_score_uses_one_late_residual_shared_budget_on_real_frontier(monkeypatch):
    monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    options = (Option((1, 0), 1.0), Option((3, 0), 1.0), Option((0, 1), 1.0), Option((0, 3), 1.0))
    new = router.select_early_root_canonical_frame(options, (4, 4), 3, 1, 2, 8192)
    old = select_zero_delta_affine_dual_residual_capacity_frontier_frame(options, (4, 4), 3, 1, 2, 8192)
    assert not new.root_lp.selection.score_exact and new.probe_stage == "frontier_late"
    assert new.frontier_evidence == old.frontier_evidence and new.probe == old.probe
    assert new.final_lp == old.final_lp and new.selection == old.selection
    assert new.probe.completion.state_budget == 8192 - new.probe.score_proof.states_visited
    assert new.probe.score_proof.states_visited + new.probe.completion.states_visited <= 8192


@pytest.mark.parametrize("queue,lanes", (((1, 1), 1), ((2, 1), 2), ((0, 0), 2), ((3, 2), 3)))
def test_small_original_action_enumeration_matches_early_or_late_canonical_completion(queue, lanes):
    options = (Option((1, 0), 1.0), Option((0, 1), 1.0), Option((1, 0), 1.25))
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        active = [options[index] for index in indices if index < len(options)]
        service = tuple(sum(option.requirements[c] for option in active) for c in range(2))
        if any(done > waiting for done, waiting in zip(service, queue)):
            continue
        duration = max((Fraction(1), *(Fraction.from_float(option.duration) for option in active)))
        score = Fraction(sum(q * s for q, s in zip(queue, service))) / duration
        keys.append((-score, duration, indices))
    result = router.select_early_root_canonical_frame(options, queue, lanes, 144, 256, 8192)
    selected = result.selection
    duration = Fraction.from_float(selected.duration)
    key = (-Fraction(sum(q * s for q, s in zip(queue, selected.service))) / duration, duration,
           tuple(len(options) if index is None else index for index in selected.option_indices))
    assert selected.canonical_exact and key == min(keys)
    assert result.root_lp_budget == result.root_lp.selection.lp_node_budget == 1
    assert result.initial_lp_budget == 144 and result.lp_node_budget == 256
    if result.probe is not None:
        assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= 8192
