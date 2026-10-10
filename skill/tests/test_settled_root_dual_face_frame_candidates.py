"""No-LP settlement, original actions, and unchanged paid continuation."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.settled_root_dual_face_frame_candidates as router
import algorithm.settled_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as settled
import algorithm.native_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as native
from algorithm.native_root_dual_face_frame_candidates import select_native_root_dual_face_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


CATALOGUES = (
    (Option((1, 0), 1.), Option((0, 1), 1.)),
    (Option((2, 0), 2.), Option((2, 0), 1.), Option((0, 1), 1.5)),
    (Option((1, 0), 2.), Option((1, 0), 1.), Option((0, 2), 2.)),
)


def action_key(options, queue, lanes, counts):
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    if sum(counts) > lanes or any(done > q for done, q in zip(service, queue, strict=True)):
        return None
    duration = max((Fraction(1), *(Fraction.from_float(option.duration) for count, option in
                                zip(counts, options, strict=True) if count)))
    score = Fraction(sum(q * done for q, done in zip(queue, service, strict=True))) / duration
    indices = tuple(i for i, count in enumerate(counts) for _ in range(count))
    return -score, duration, indices + (len(options),) * (lanes - len(indices))


def argmax_key(options, queue, lanes):
    candidates = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        value = action_key(options, queue, lanes, tuple(indices.count(i) for i in range(len(options))))
        if value is not None:
            candidates.append(value)
    return min(candidates)


def valid_tied_root_then_real_lp(real_lp, calls):
    """Select an exact alternative LP optimum at the only duration root."""
    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        if len(calls) == 1:
            return SimpleNamespace(success=True, x=(0., 2.),
                ineqlin=SimpleNamespace(marginals=(-2., 0., 0., 0., 0.)))
        return real_lp(*args, **kwargs)
    return lp


def assert_phase_accounting(root, after, work, search):
    assert work.calls == work.snapshot_calls == 1
    assert work.lp_calls_before == root.selection.lp_calls == work.lp_calls_after == after.selection.lp_calls
    assert work.popped_regions == work.pruned_regions + work.split_regions + work.singleton_regions
    assert work.head_checks == work.popped_regions + work.stopped_for_lp
    assert work.unsolved_prune_checks <= work.pruned_regions + work.stopped_for_lp
    assert work.split_regions <= root.root_pruning.roots_solved <= root.selection.duration_root_count
    assert after.selection.branch_splits - root.selection.branch_splits == work.split_regions
    assert after.selection.singleton_regions_closed - root.selection.singleton_regions_closed == work.singleton_regions
    assert work.frontier_regions_after == len(search._search.gi_frame.f_locals["frontier"])
    assert work.snapshot_heap_entries_read == 2 * work.frontier_regions_after
    assert work.snapshot_retained_entries_read == 2 * after.selection.frontier_regions_remaining


def test_settlement_closes_root_with_native_split_and_dual_cost_but_no_new_lp(monkeypatch):
    options, queue, lanes = CATALOGUES[0], (1, 3), 2
    search = settled.SettledRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    old = native.NativeRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    root, old_root = search.advance_roots(144), old.advance_roots(144)
    assert asdict(root) == asdict(old_root) and root.selection.score_exact and not root.selection.canonical_exact
    generator = search._search

    def forbidden_lp(*args, **kwargs):
        raise AssertionError("settle_roots issued an extra LP")

    monkeypatch.setattr(settled, "linprog", forbidden_lp)
    after, work = search.settle_roots()
    assert search._search is generator and search.advance_to(root.selection.lp_node_budget) is after
    assert after.selection.canonical_exact and after.selection.branch_splits == 1
    assert work.stopped_for_lp == 0 and work.frontier_regions_after == 0
    assert after.dual_reuse.child_bound_evaluations == after.dual_reuse.child_bound_prunes == 1
    assert_phase_accounting(root, after, work, search)
    assert asdict(after.selection) == {**asdict(old.advance_to(144).selection), "lp_node_budget": root.selection.lp_node_budget}
    assert asdict(search.advance_to(144)) == asdict(old.advance_to(144))
    assert asdict(search.advance_to(256)) == asdict(old.advance_to(256))


def test_settlement_stops_at_live_unsolved_child_and_continuation_matches_frozen_native(monkeypatch):
    options, queue, lanes = CATALOGUES[0], (2, 2), 2
    real_lp, new_calls, old_calls = settled.linprog, [], []
    monkeypatch.setattr(settled, "linprog", valid_tied_root_then_real_lp(real_lp, new_calls))
    monkeypatch.setattr(native, "linprog", valid_tied_root_then_real_lp(real_lp, old_calls))
    search = settled.SettledRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    old = native.NativeRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    root, old_root = search.advance_roots(144), old.advance_roots(144)
    assert asdict(root) == asdict(old_root) and root.selection.option_counts == (0, 2)
    generator = search._search
    after, work = search.settle_roots()
    assert len(new_calls) == 1 and search._search is generator
    assert work.stopped_for_lp == 1 and work.unsolved_prune_checks == 1
    assert not after.selection.canonical_exact and after.selection.inherited_unsolved_regions == 1
    assert search._search.gi_frame.f_locals["frontier"][0][-1].solved is False
    assert_phase_accounting(root, after, work, search)
    for cap in (144, 256):
        assert asdict(search.advance_to(cap)) == asdict(old.advance_to(cap))
        assert search._search is generator
    assert new_calls == old_calls and len(new_calls) > 1


def test_whole_selector_returns_settled_closure_and_preserves_original_raw_root():
    options, queue, lanes = CATALOGUES[0], (1, 3), 2
    result = router.select_settled_root_dual_face_frame(options, queue, lanes, 144, 256, 8192)
    previous = select_native_root_dual_face_frame(options, queue, lanes, 144, 256, 8192)
    assert asdict(result.root_lp) == asdict(previous.root_lp)
    assert result.settled_lp is result.initial_lp is result.final_lp
    assert result.root_lp.selection.canonical_exact is False and result.settled_lp.selection.canonical_exact
    assert result.status == "LP_ROOT_SETTLE_COMPLETE" and result.probe is None and result.probe_stage == "none"
    assert not result.resumed and result.root_lp_budget == 1 and result.initial_lp_budget == 144
    assert result.lp_node_budget == 256 and result.integer_state_budget == 8192
    assert result.dual_face_evidence is None and all(value == 0 for value in asdict(result.dual_face_work).values())
    assert result.routing_preparation == previous.routing_preparation
    assert result.root_settle_work.frontier_regions_before == result.routing_preparation.root_snapshot_heap_entries_read // 2
    assert result.root_settle_work.lp_calls_before == result.root_settle_work.lp_calls_after


@pytest.mark.parametrize("budget", (0, 8192))
def test_partial_box_probe_uses_settled_baseline_and_retains_exhaustion_without_restart(monkeypatch, budget):
    options, queue, lanes = CATALOGUES[0], (2, 2), 2
    calls = []
    monkeypatch.setattr(settled, "linprog", valid_tied_root_then_real_lp(settled.linprog, calls))
    engine_type, constructions, roots, phases, advances, generators = router.SettledRootZeroDeltaAffineDualReuseIntegerCapacitySearch, [], [], [], [], []

    class CountedSearch(engine_type):
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            super().__init__(*args, **kwargs)

        def advance_roots(self, cap):
            roots.append(cap)
            result = super().advance_roots(cap)
            generators.append(self._search)
            return result

        def settle_roots(self):
            phases.append(1)
            assert self._search is generators[0]
            return super().settle_roots()

        def advance_to(self, cap):
            advances.append(cap)
            assert self._search is generators[0]
            return super().advance_to(cap)

    monkeypatch.setattr(router, "SettledRootZeroDeltaAffineDualReuseIntegerCapacitySearch", CountedSearch)
    result = router.select_settled_root_dual_face_frame(options, queue, lanes, 144, 256, budget)
    assert constructions == [1] and roots == [144] and phases == [1]
    assert result.root_settle_work.stopped_for_lp == 1
    assert result.probe_stage == "root" and result.probe.baseline_selection == result.settled_lp.selection
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= budget
    face, = result.dual_face_evidence.regions
    assert face.lower != face.certificate_lower or face.upper != face.certificate_upper
    assert face.own_numerator_fraction == face.target_numerator_fraction == (4, 1)
    assert result.selection.canonical_exact and result.selection.option_counts == (2, 0)
    if budget:
        assert advances == [] and len(calls) == 1
        assert result.initial_lp is result.final_lp is result.settled_lp and result.status == "INTEGER_PROBE_COMPLETE"
    else:
        assert advances == [144] and len(calls) > 1
        assert result.probe.selection == result.settled_lp.selection
        assert result.probe.completion.status == "STATE_BUDGET_EXHAUSTED" and result.status == "LP_INITIAL_COMPLETE"


@pytest.mark.parametrize("options", CATALOGUES)
@pytest.mark.parametrize("queue,lanes", (((0, 0), 2), ((1, 3), 2), ((2, 2), 3), ((3, 2), 2)))
def test_tiny_original_argmax_with_native_roots_and_settled_domain(options, queue, lanes):
    result = router.select_settled_root_dual_face_frame(options, queue, lanes, 144, 256, 8192)
    assert result.selection.canonical_exact
    assert action_key(options, queue, lanes, result.selection.option_counts) == argmax_key(options, queue, lanes)
    assert result.root_lp_budget == result.root_lp.selection.duration_root_count
    assert all(asdict(result.routing_preparation)[name] == 0 for name in
               tuple(router.EarlyRootRoutingWork.__dataclass_fields__)[:8])
    if result.root_lp.selection.canonical_exact or not result.root_lp.selection.score_exact:
        assert result.settled_lp is None and result.root_settle_work is None
    else:
        assert result.settled_lp is not None and result.root_settle_work is not None
        assert result.root_settle_work.split_regions <= result.root_lp_budget
        assert result.root_settle_work.lp_calls_before == result.root_settle_work.lp_calls_after


def test_score_open_failed_roots_keep_previous_route_and_no_settlement(monkeypatch):
    monkeypatch.setattr(settled, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    monkeypatch.setattr(native, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    options = (Option((1, 0), 1.), Option((3, 0), 1.), Option((0, 1), 1.), Option((0, 3), 1.))
    result = router.select_settled_root_dual_face_frame(options, (4, 4), 3, 1, 2, 8192)
    old = select_native_root_dual_face_frame(options, (4, 4), 3, 1, 2, 8192)
    record = asdict(result)
    assert record.pop("settled_lp") is None and record.pop("root_settle_work") is None
    assert record == asdict(old) and result.probe_stage == "frontier_late"


def test_empty_catalogue_is_already_canonical_and_never_settles():
    result = router.select_settled_root_dual_face_frame((), (0, 0), 2, 144, 256, 8192)
    assert result.status == "LP_ROOT_COMPLETE" and result.root_lp_budget == 0
    assert result.settled_lp is result.root_settle_work is result.probe is None
    assert result.root_lp is result.initial_lp is result.final_lp
    assert result.routing_preparation.root_snapshot_calls == 1
