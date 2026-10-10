"""Genuine raw roots, one-pass fees, and an unchanged paid scheduling path."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.fused_root_dual_face_frame_candidates as router
import algorithm.fused_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as fused
import algorithm.settled_root_dual_face_frame_candidates as previous_router
import algorithm.settled_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as previous


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


TIED = (Option((1, 0), 1.), Option((0, 1), 1.))
CASES = (
    (TIED, (0, 0), 2),
    (TIED, (1, 3), 2),
    (TIED, (2, 2), 3),
    ((Option((2, 0), 2.), Option((2, 0), 1.), Option((0, 1), 1.5)), (3, 2), 3),
    ((Option((1, 0), 2.), Option((1, 0), 1.), Option((0, 2), 2.)), (3, 2), 2),
)
SCAN_FIELDS = ("root_snapshot_heap_entries_read", "root_snapshot_retained_entries_read")
PHASE_SCAN_FIELDS = ("snapshot_heap_entries_read", "snapshot_retained_entries_read")


def action_key(options, queue, lanes, counts):
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    if sum(counts) > lanes or any(done > waiting for done, waiting in zip(service, queue, strict=True)):
        return None
    duration = max((Fraction(1), *(Fraction.from_float(option.duration) for count, option in
                                zip(counts, options, strict=True) if count)))
    score = Fraction(sum(waiting * done for waiting, done in zip(queue, service, strict=True))) / duration
    indices = tuple(i for i, count in enumerate(counts) for _ in range(count))
    return -score, duration, indices + (len(options),) * (lanes - len(indices))


def original_argmax(options, queue, lanes):
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        key = action_key(options, queue, lanes, tuple(indices.count(i) for i in range(len(options))))
        if key is not None:
            keys.append(key)
    return min(keys)


def same_route_except_actual_scans(result, old):
    record, prior = asdict(result), asdict(old)
    prep, old_prep = record.pop("routing_preparation"), prior.pop("routing_preparation")
    assert prep[SCAN_FIELDS[0]] * 2 == old_prep[SCAN_FIELDS[0]]
    assert prep[SCAN_FIELDS[1]] == 0
    assert {k: v for k, v in prep.items() if k not in SCAN_FIELDS} == {
        k: v for k, v in old_prep.items() if k not in SCAN_FIELDS}
    phase, old_phase = record.pop("root_settle_work"), prior.pop("root_settle_work")
    if old_phase is None:
        assert phase is None
    else:
        assert phase[PHASE_SCAN_FIELDS[0]] * 2 == old_phase[PHASE_SCAN_FIELDS[0]]
        assert phase[PHASE_SCAN_FIELDS[0]] == phase["frontier_regions_after"]
        assert phase[PHASE_SCAN_FIELDS[1]] == 0
        assert prep[SCAN_FIELDS[0]] == phase["frontier_regions_before"]
        assert {k: v for k, v in phase.items() if k not in PHASE_SCAN_FIELDS} == {
            k: v for k, v in old_phase.items() if k not in PHASE_SCAN_FIELDS}
    assert record == prior


def tied_first_lp(real_lp, calls, fail_second=False):
    """Return an exact alternative optimum, then use actual LPs or one failure."""
    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        if len(calls) == 1:
            return SimpleNamespace(success=True, x=(0., 2.),
                ineqlin=SimpleNamespace(marginals=(-2., 0., 0., 0., 0.)))
        if fail_second and len(calls) == 2:
            return SimpleNamespace(success=False)
        return real_lp(*args, **kwargs)
    return lp


@pytest.mark.parametrize("options,queue,lanes", CASES)
def test_true_raw_m_conditional_settle_continuation_and_original_argmax(options, queue, lanes):
    search = fused.FusedRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    old = previous.SettledRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    root, after, work = search.advance_roots_and_settle(144)
    raw = old.advance_roots(144)
    frozen_root = asdict(root)
    assert frozen_root == asdict(raw)
    assert root.selection.lp_node_budget == root.selection.duration_root_count
    assert root.selection.branch_splits == 0
    if raw.selection.score_exact and not raw.selection.canonical_exact:
        old_after, old_work = old.settle_roots()
        assert asdict(after) == asdict(old_after)
        for name in previous.ROOT_SETTLE_COUNTERS:
            if name not in PHASE_SCAN_FIELDS:
                assert getattr(work, name) == getattr(old_work, name)
        assert work.lp_calls_before == work.lp_calls_after
        assert work.snapshot_heap_entries_read == work.frontier_regions_after
        assert work.snapshot_retained_entries_read == 0
        assert after.thresholds is root.thresholds and after.root_pruning is root.root_pruning
        assert after.preprocessing is not root.preprocessing
    else:
        assert after is work is None
    generator = search._search
    assert search.advance_to(root.selection.lp_node_budget) is (after if after is not None else root)
    for cap in (144, 256):
        stage = search.advance_to(cap)
        assert search._search is generator and asdict(stage) == asdict(old.advance_to(cap))
        assert stage.thresholds is root.thresholds and stage.root_pruning is root.root_pruning
        assert stage.capacity_propagation is not root.capacity_propagation
    assert asdict(root) == frozen_root
    result = router.select_fused_root_dual_face_frame(options, queue, lanes, 144, 256, 8192)
    parent = previous_router.select_settled_root_dual_face_frame(options, queue, lanes, 144, 256, 8192)
    same_route_except_actual_scans(result, parent)
    assert result.selection.canonical_exact
    assert action_key(options, queue, lanes, result.selection.option_counts) == original_argmax(options, queue, lanes)


def test_fused_initialization_pays_one_preprocessing_and_identical_lp_sequence_after_failure(monkeypatch):
    calls, old_calls, reductions = [], [], []
    original_reduce, real_lp = fused.reduce_requirement_dominated_options, fused.linprog

    def counted_reduce(*args, **kwargs):
        reductions.append(1)
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(fused, "reduce_requirement_dominated_options", counted_reduce)
    monkeypatch.setattr(fused, "linprog", tied_first_lp(real_lp, calls, fail_second=True))
    monkeypatch.setattr(previous, "linprog", tied_first_lp(real_lp, old_calls, fail_second=True))
    search = fused.FusedRootZeroDeltaAffineDualReuseIntegerCapacitySearch(TIED, (2, 2), 2)
    old = previous.SettledRootZeroDeltaAffineDualReuseIntegerCapacitySearch(TIED, (2, 2), 2)
    root, after, phase = search.advance_roots_and_settle(144)
    old_root = old.advance_roots(144)
    old_after, old_phase = old.settle_roots()
    assert reductions == [1] and len(calls) == len(old_calls) == 1
    assert asdict(root) == asdict(old_root) and asdict(after) == asdict(old_after)
    assert phase.stopped_for_lp == old_phase.stopped_for_lp == 1
    assert not search._search.gi_frame.f_locals["frontier"][0][-1].solved
    generator = search._search
    for cap in (144, 256):
        actual = search.advance_to(cap)
        assert search._search is generator and asdict(actual) == asdict(old.advance_to(cap))
    assert calls == old_calls and len(calls) > 1
    assert actual.selection.safe_fallback_lp_calls >= 1 and reductions == [1]


@pytest.mark.parametrize("budget", (0, 8192))
def test_whole_fused_call_keeps_single_search_probe_and_true_raw_snapshot(monkeypatch, budget):
    calls, old_calls = [], []
    real_lp = fused.linprog
    monkeypatch.setattr(fused, "linprog", tied_first_lp(real_lp, calls))
    monkeypatch.setattr(previous, "linprog", tied_first_lp(real_lp, old_calls))
    engine = router.FusedRootZeroDeltaAffineDualReuseIntegerCapacitySearch
    constructions, initial_calls, advances, generators = [], [], [], []

    class CountedSearch(engine):
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            super().__init__(*args, **kwargs)

        def advance_roots_and_settle(self, cap):
            initial_calls.append(cap)
            record = super().advance_roots_and_settle(cap)
            generators.append(self._search)
            return record

        def advance_to(self, cap):
            advances.append(cap)
            assert self._search is generators[0]
            return super().advance_to(cap)

    monkeypatch.setattr(router, "FusedRootZeroDeltaAffineDualReuseIntegerCapacitySearch", CountedSearch)
    result = router.select_fused_root_dual_face_frame(TIED, (2, 2), 2, 144, 256, budget)
    parent = previous_router.select_settled_root_dual_face_frame(TIED, (2, 2), 2, 144, 256, budget)
    same_route_except_actual_scans(result, parent)
    assert constructions == [1] and initial_calls == [144]
    assert result.probe_stage == "root" and result.probe.baseline_selection == result.settled_lp.selection
    assert result.root_lp.selection.option_counts == (0, 2) and result.root_lp.selection.branch_splits == 0
    assert result.root_settle_work.lp_calls_before == result.root_settle_work.lp_calls_after == 1
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= budget
    assert result.selection.option_counts == (2, 0)
    assert calls == old_calls
    if budget:
        assert advances == [] and result.status == "INTEGER_PROBE_COMPLETE"
        assert result.initial_lp is result.final_lp is result.settled_lp
    else:
        assert advances == [144] and result.status == "LP_INITIAL_COMPLETE"
        assert result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"
        assert result.probe.selection == result.settled_lp.selection


def test_root_budget_validation_precedes_all_lp_and_requires_one_initialization(monkeypatch):
    options, queue, lanes = CASES[3]

    def forbidden_lp(*args, **kwargs):
        raise AssertionError("LP issued before validation of the actual duration root count")

    search = fused.FusedRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    real_lp = fused.linprog
    monkeypatch.setattr(fused, "linprog", forbidden_lp)
    with pytest.raises(ValueError, match="must cover 2 duration thresholds"):
        search.advance_roots_and_settle(1)
    assert search._search is None
    with pytest.raises(ValueError, match="nonnegative integer"):
        search.advance_roots_and_settle(-1)
    with pytest.raises(ValueError, match="advance_roots_and_settle"):
        search.advance_to(144)
    monkeypatch.setattr(fused, "linprog", real_lp)
    search.advance_roots_and_settle(144)
    with pytest.raises(ValueError, match="already initialized"):
        search.advance_roots_and_settle(144)
    search.advance_to(256)
    with pytest.raises(ValueError, match="nondecreasing"):
        search.advance_to(144)


def test_failed_score_open_roots_keep_the_previous_late_route_without_settlement(monkeypatch):
    fail = lambda *args, **kwargs: SimpleNamespace(success=False)
    monkeypatch.setattr(fused, "linprog", fail)
    monkeypatch.setattr(previous, "linprog", fail)
    options = (Option((1, 0), 1.), Option((3, 0), 1.), Option((0, 1), 1.), Option((0, 3), 1.))
    result = router.select_fused_root_dual_face_frame(options, (4, 4), 3, 1, 2, 8192)
    parent = previous_router.select_settled_root_dual_face_frame(options, (4, 4), 3, 1, 2, 8192)
    same_route_except_actual_scans(result, parent)
    assert result.settled_lp is result.root_settle_work is None and result.probe_stage == "frontier_late"


def test_empty_catalogue_returns_fully_built_raw_m_without_a_phase():
    result = router.select_fused_root_dual_face_frame((), (0, 0), 2, 144, 256, 8192)
    parent = previous_router.select_settled_root_dual_face_frame((), (0, 0), 2, 144, 256, 8192)
    same_route_except_actual_scans(result, parent)
    assert result.status == "LP_ROOT_COMPLETE" and result.root_lp_budget == 0
    assert result.root_lp is result.initial_lp is result.final_lp
    assert result.settled_lp is result.root_settle_work is result.probe is None
    assert result.routing_preparation.root_snapshot_calls == 1
    assert result.routing_preparation.root_snapshot_heap_entries_read == 0
