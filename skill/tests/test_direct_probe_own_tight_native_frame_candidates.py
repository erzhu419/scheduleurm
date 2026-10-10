"""Preserve intact root domains, one probe, and ordinary paid continuation."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.direct_probe_own_tight_native_frame_candidates as router
import algorithm.direct_probe_own_tight_native_kernel as direct
import algorithm.own_tight_dual_native_frame_candidates as previous_router
import algorithm.own_tight_dual_native_kernel as previous


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


TIED = (Option((1, 0), 1.), Option((0, 1), 1.))
THREE = (Option((1, 0, 0), 1.), Option((0, 1, 0), 1.), Option((0, 0, 1), 1.))
CATALOGUES = (
    TIED,
    (Option((2, 0), 2.), Option((2, 0), 1.), Option((0, 1), 1.5)),
    (Option((1, 0), 2.), Option((1, 0), 1.), Option((0, 2), 2.)),
)


def key(options, queue, lanes, counts):
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    if sum(counts) > lanes or any(done > waiting for done, waiting in zip(service, queue, strict=True)):
        return None
    duration = max((Fraction(1), *(Fraction.from_float(option.duration) for count, option in
                                zip(counts, options, strict=True) if count)))
    score = Fraction(sum(q * done for q, done in zip(queue, service, strict=True))) / duration
    indices = tuple(i for i, count in enumerate(counts) for _ in range(count))
    return -score, duration, indices + (len(options),) * (lanes - len(indices))


def argmax(options, queue, lanes):
    return min(value for indices in combinations_with_replacement(range(len(options) + 1), lanes)
               if (value := key(options, queue, lanes, tuple(indices.count(i) for i in range(len(options))))) is not None)


def alternate_first(real_lp, calls, dimension):
    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        if len(calls) == 1:
            primal = (0., 2.) if dimension == 2 else (0., 2., 0.)
            price = -2. if dimension == 2 else -3.
            return SimpleNamespace(success=True, x=primal,
                ineqlin=SimpleNamespace(marginals=(price,) + (0.,) * (2 * dimension)))
        return real_lp(*args, **kwargs)
    return lp


def assert_phase(root, after, work, search):
    assert work.calls == work.snapshot_calls == 1 and work.split_regions == 0
    assert work.lp_calls_before == work.lp_calls_after == root.selection.lp_calls == after.selection.lp_calls
    assert work.popped_regions == work.pruned_regions + work.singleton_regions + work.stopped_for_branch
    assert work.head_checks == work.popped_regions + work.stopped_for_lp
    assert work.stopped_for_branch + work.stopped_for_lp <= 1
    assert work.frontier_regions_after == work.frontier_regions_before - work.pruned_regions - work.singleton_regions
    assert work.snapshot_heap_entries_read == work.frontier_regions_after
    assert work.snapshot_retained_entries_read == 0
    assert work.frontier_regions_after == len(search._search.gi_frame.f_locals["frontier"])
    if work.stopped_for_branch:
        head = search._search.gi_frame.f_locals["frontier"][0][-1]
        assert head.solved and direct._branch_position(head) is not None
    assert after.selection.canonical_exact == (not work.stopped_for_branch and not work.stopped_for_lp)


@pytest.mark.parametrize("options", CATALOGUES)
@pytest.mark.parametrize("queue,lanes", (((0, 0), 2), ((3, 2), 3)))
def test_real_raw_m_and_original_argmax_with_same_search_continuation(options, queue, lanes):
    search = direct.DirectProbeOwnTightNativeSearch(options, queue, lanes)
    parent = previous.OwnTightDualNativeSearch(options, queue, lanes)
    root, after, work = search.advance_roots_and_settle(144)
    old_root, _, _ = parent.advance_roots_and_settle(144)
    assert asdict(root) == asdict(old_root)
    generator = search._search
    if after is not None:
        assert_phase(root, after, work, search)
    for cap in (144, 256):
        stage = search.advance_to(cap)
        assert search._search is generator
        assert stage.selection.canonical_exact
        assert key(options, queue, lanes, stage.selection.option_counts) == argmax(options, queue, lanes)
    result = router.select_direct_probe_own_tight_native_frame(options, queue, lanes, 144, 256, 8192)
    assert result.selection.canonical_exact
    assert key(options, queue, lanes, result.selection.option_counts) == argmax(options, queue, lanes)
    assert result.initial_lp_budget == 144 and result.lp_node_budget == 256 and result.integer_state_budget == 8192


def test_root_branch_stop_preserves_unsplit_box_original_serial_and_lp_sequence(monkeypatch):
    calls, old_calls = [], []
    real_lp = direct.linprog
    monkeypatch.setattr(direct, "linprog", alternate_first(real_lp, calls, 2))
    monkeypatch.setattr(previous, "linprog", alternate_first(real_lp, old_calls, 2))
    search = direct.DirectProbeOwnTightNativeSearch(TIED, (2, 2), 2)
    parent = previous.OwnTightDualNativeSearch(TIED, (2, 2), 2)
    root, after, work = search.advance_roots_and_settle(144)
    old_root, old_after, old_work = parent.advance_roots_and_settle(144)
    assert asdict(root) == asdict(old_root)
    assert len(calls) == 1 and work.stopped_for_branch == 1 and work.stopped_for_lp == 0
    assert old_work.split_regions == 1 and old_work.stopped_for_lp == 1
    assert after.selection.branch_splits == 0 and after.selection.inherited_unsolved_regions == 0
    assert after.zero_delta.guard_calls == root.zero_delta.guard_calls == 0
    heap = search._search.gi_frame.f_locals["frontier"]
    assert len(heap) == 1 and heap[0][4] == 0
    region = heap[0][-1]
    assert region.lower == (0, 0) and region.upper == (2, 2) and region.solved
    assert after.selection.frontier_regions_remaining == 1
    assert_phase(root, after, work, search)
    generator = search._search
    for cap in (2, 144, 256):
        stage, old = search.advance_to(cap), parent.advance_to(cap)
        assert search._search is generator
        assert stage.selection.option_counts == old.selection.option_counts
        assert stage.selection.score == old.selection.score and stage.selection.canonical_exact == old.selection.canonical_exact
    assert calls == old_calls and len(calls) > 1
    assert stage.selection.branch_splits > 0


def test_changed_own_face_is_returned_with_current_keys_and_no_children(monkeypatch):
    calls = []
    monkeypatch.setattr(direct, "linprog", alternate_first(direct.linprog, calls, 3))
    search = direct.DirectProbeOwnTightNativeSearch(THREE, (3, 3, 1), 2)
    root, after, work = search.advance_roots_and_settle(144)
    assert not root.native_own_tight_evidence and len(after.native_own_tight_evidence) == 1
    assert work.stopped_for_branch == 1 and work.split_regions == 0 and len(calls) == 1
    heap = search._search.gi_frame.f_locals["frontier"]
    assert len(heap) == 1 and heap[0][4] == 0
    region = heap[0][-1]
    assert heap[0][2:4] == (region.lower, region.upper)
    assert region.lower == (0, 0, 0) and region.upper == (2, 2, 0) and region.solved
    assert region.certificate.upper == (2, 2, 1) and region.own_bound == Fraction(6)
    assert after.selection.branch_splits == after.zero_delta.guard_calls == 0
    assert after.capacity_propagation == root.capacity_propagation
    assert_phase(root, after, work, search)


def test_cheap_own_canonical_closure_keeps_parent_route_and_all_actual_work():
    result = router.select_direct_probe_own_tight_native_frame(TIED, (1, 3), 2, 144, 256, 8192)
    old = previous_router.select_own_tight_dual_native_frame(TIED, (1, 3), 2, 144, 256, 8192)
    record = asdict(result)
    assert record["root_settle_work"].pop("stopped_for_branch") == 0
    assert record == asdict(old)
    assert result.status == "LP_ROOT_SETTLE_COMPLETE" and result.probe is None
    assert result.root_settle_work.split_regions == 0 and result.root_settle_work.pruned_regions == 1


@pytest.mark.parametrize("budget", (0, 8192))
def test_single_probe_exhaustion_resumes_same_generator_and_counts_snapshot_copies(monkeypatch, budget):
    calls = []
    real_lp = direct.linprog
    monkeypatch.setattr(direct, "linprog", alternate_first(real_lp, calls, 3))
    engine = router.DirectProbeOwnTightNativeSearch
    constructions, advances, generators = [], [], []

    class CountedSearch(engine):
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            super().__init__(*args, **kwargs)

        def advance_roots_and_settle(self, cap):
            result = super().advance_roots_and_settle(cap)
            generators.append(self._search)
            return result

        def advance_to(self, cap):
            advances.append(cap)
            assert self._search is generators[0]
            return super().advance_to(cap)

    monkeypatch.setattr(router, "DirectProbeOwnTightNativeSearch", CountedSearch)
    result = router.select_direct_probe_own_tight_native_frame(THREE, (3, 3, 1), 2, 144, 256, budget)
    assert constructions == [1] and result.root_settle_work.stopped_for_branch == 1
    assert result.probe_stage == "root" and result.probe.baseline_selection == result.settled_lp.selection
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= budget
    assert result.selection.canonical_exact and result.selection.option_counts == (2, 0, 0)
    if budget:
        assert advances == [] and len(calls) == 1
        assert result.initial_lp is result.final_lp is result.settled_lp
        assert result.final_lp.selection.branch_splits == 0
    else:
        assert advances == [144] and len(calls) > 1
        assert result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"
        assert result.probe.selection == result.settled_lp.selection
        assert result.final_lp.selection.branch_splits > 0
        calls.clear()
        audit = direct.DirectProbeOwnTightNativeSearch(THREE, (3, 3, 1), 2)
        root, settled, _ = audit.advance_roots_and_settle(144)
        audit.advance_to(root.selection.duration_root_count + 1)
        final = audit.advance_to(144)
        a, b = asdict(result.final_lp), asdict(final)
        direct_copies = a["native_own_tight_work"].pop("event_snapshot_entries_copied")
        audit_copies = b["native_own_tight_work"].pop("event_snapshot_entries_copied")
        assert a == b and audit_copies > direct_copies
        assert direct_copies == len(settled.native_own_tight_evidence) + len(result.final_lp.native_own_tight_evidence)


def test_empty_catalogue_has_no_phase_and_keeps_real_schema():
    result = router.select_direct_probe_own_tight_native_frame((), (0, 0), 2, 144, 256, 8192)
    assert result.status == "LP_ROOT_COMPLETE" and result.root_lp_budget == 0
    assert result.root_lp is result.initial_lp is result.final_lp
    assert result.settled_lp is result.root_settle_work is result.probe is None
    assert direct.ROOT_SETTLE_COUNTERS == (*previous.ROOT_SETTLE_COUNTERS, "stopped_for_branch")
    assert direct.ROOT_SETTLE_FEE_COUNTERS == previous.ROOT_SETTLE_FEE_COUNTERS
    assert direct.NATIVE_OWN_TIGHT_COUNTERS == previous.NATIVE_OWN_TIGHT_COUNTERS
