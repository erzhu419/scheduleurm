"""Check a native M checkpoint and unchanged, paid whole-selector paths."""
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from fractions import Fraction
from types import SimpleNamespace

import pytest

import algorithm.native_root_dual_face_frame_candidates as router
import algorithm.native_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as native
import algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as frozen
from algorithm.dual_face_root_canonical_frame_candidates import select_dual_face_root_canonical_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


CATALOGUES = (
    (Option((1, 0), 1.), Option((0, 1), 1.)),
    (Option((2, 0), 2.), Option((2, 0), 1.), Option((0, 1), 1.5)),
    (Option((1, 0), 2.), Option((1, 0), 1.), Option((0, 2), 2.)),
)
SCAN_FIELDS = tuple(router.EarlyRootRoutingWork.__dataclass_fields__)[:8]


def true_root_count(options):
    minima = {}
    for option in options:
        minima[option.requirements] = min(minima.get(option.requirements, float("inf")), max(1., option.duration))
    return len(set(minima.values()))


def key(options, queue, lanes, counts):
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    if sum(counts) > lanes or any(done > q for done, q in zip(service, queue, strict=True)):
        return None
    duration = max((Fraction(1), *(Fraction.from_float(option.duration) for count, option in
                                zip(counts, options, strict=True) if count)))
    score = Fraction(sum(q * done for q, done in zip(queue, service, strict=True))) / duration
    encoded = tuple(i for i, count in enumerate(counts) for _ in range(count))
    return -score, duration, encoded + (len(options),) * (lanes - len(encoded))


def argmax_key(options, queue, lanes):
    choices = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        value = key(options, queue, lanes, tuple(indices.count(i) for i in range(len(options))))
        if value is not None:
            choices.append(value)
    return min(choices)


def equivalent_except_cancelled_scan(result, old):
    record, previous = asdict(result), asdict(old)
    preparation, old_preparation = record.pop("routing_preparation"), previous.pop("routing_preparation")
    assert all(preparation[field] == 0 for field in SCAN_FIELDS)
    assert {field: value for field, value in preparation.items() if field not in SCAN_FIELDS} == {
        field: value for field, value in old_preparation.items() if field not in SCAN_FIELDS}
    assert record == previous


@pytest.mark.parametrize("options", CATALOGUES)
@pytest.mark.parametrize("queue,lanes", (((0, 0), 2), ((1, 3), 2), ((2, 2), 3)))
def test_native_actual_roots_and_continuation_equal_frozen_raw_ladder(options, queue, lanes):
    count = true_root_count(options)
    search = native.NativeRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    old = frozen.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    roots = search.advance_roots(144)
    generator = search._search
    assert roots.selection.lp_node_budget == roots.selection.duration_root_count == count
    assert roots.selection.branch_splits == 0 and asdict(roots) == asdict(old.advance_to(count))
    assert search.advance_to(count) is roots
    for cap in (count + 1, 144, 256):
        stage = search.advance_to(cap)
        assert search._search is generator and asdict(stage) == asdict(old.advance_to(cap))
    with pytest.raises(ValueError, match="nondecreasing"):
        search.advance_to(144)


@pytest.mark.parametrize("options", CATALOGUES)
@pytest.mark.parametrize("queue,lanes", (((0, 0), 2), ((1, 3), 2), ((2, 2), 3)))
def test_whole_route_removes_only_duplicate_scan_and_matches_original_action_argmax(options, queue, lanes):
    result = router.select_native_root_dual_face_frame(options, queue, lanes, 144, 256, 8192)
    old = select_dual_face_root_canonical_frame(options, queue, lanes, 144, 256, 8192)
    equivalent_except_cancelled_scan(result, old)
    assert result.selection.canonical_exact and key(options, queue, lanes, result.selection.option_counts) == argmax_key(options, queue, lanes)
    assert result.initial_lp_budget == 144 and result.lp_node_budget == 256 and result.integer_state_budget == 8192
    assert result.root_lp_budget == true_root_count(options)
    assert result.routing_preparation.root_snapshot_calls == 1
    assert result.final_lp.preprocessing.original_option_count == len(options)


def test_root_cap_uses_native_group_minima_and_refuses_short_initial_cap_before_any_lp(monkeypatch):
    options = CATALOGUES[1]
    calls = []

    def forbidden_lp(*args, **kwargs):
        calls.append(1)
        raise AssertionError("root LP started before actual root budget validation")

    monkeypatch.setattr(native, "linprog", forbidden_lp)
    search = native.NativeRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, (2, 2), 2)
    with pytest.raises(ValueError, match="initial LP budget must cover 2 duration thresholds"):
        search.advance_roots(1)
    assert calls == [] and search._search is None
    with pytest.raises(ValueError, match="nonnegative integer"):
        search.advance_roots(-1)
    with pytest.raises(ValueError, match="advance_roots"):
        search.advance_to(144)


def test_root_checkpoint_uses_one_native_preprocessing_and_one_lp_sequence(monkeypatch):
    options, queue, lanes = CATALOGUES[1], (3, 2), 3
    original_reduction, original_lp = native.reduce_requirement_dominated_options, native.linprog
    reductions, new_calls, old_calls = [], [], []

    def reduce(*args, **kwargs):
        reductions.append(1)
        return original_reduction(*args, **kwargs)

    def packet(args, kwargs):
        return tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])

    def new_lp(*args, **kwargs):
        new_calls.append(packet(args, kwargs))
        return original_lp(*args, **kwargs)

    def old_lp(*args, **kwargs):
        old_calls.append(packet(args, kwargs))
        return original_lp(*args, **kwargs)

    monkeypatch.setattr(native, "reduce_requirement_dominated_options", reduce)
    monkeypatch.setattr(native, "linprog", new_lp)
    monkeypatch.setattr(frozen, "linprog", old_lp)
    search = native.NativeRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    old = frozen.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    root = search.advance_roots(144)
    old_root = old.advance_to(true_root_count(options))
    assert asdict(root) == asdict(old_root)
    for cap in (144, 256):
        assert asdict(search.advance_to(cap)) == asdict(old.advance_to(cap))
    assert reductions == [1] and new_calls == old_calls


def test_early_exhaustion_keeps_same_search_actual_caps_and_unique_failed_probe(monkeypatch):
    options, queue, lanes = CATALOGUES[0], (1, 3), 2
    search_type = router.NativeRootZeroDeltaAffineDualReuseIntegerCapacitySearch
    constructions, roots, advances, generators = [], [], [], []

    class CountedSearch(search_type):
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            super().__init__(*args, **kwargs)

        def advance_roots(self, cap):
            roots.append(cap)
            result = super().advance_roots(cap)
            generators.append(self._search)
            return result

        def advance_to(self, cap):
            advances.append(cap)
            assert self._search is generators[0]
            return super().advance_to(cap)

    monkeypatch.setattr(router, "NativeRootZeroDeltaAffineDualReuseIntegerCapacitySearch", CountedSearch)
    result = router.select_native_root_dual_face_frame(options, queue, lanes, 144, 256, 0)
    old = select_dual_face_root_canonical_frame(options, queue, lanes, 144, 256, 0)
    equivalent_except_cancelled_scan(result, old)
    assert constructions == [1] and roots == [144] and advances == [144]
    assert result.probe_stage == "root" and result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"
    assert result.probe.selection == result.root_lp.selection and result.probe.integer_state_budget == 0
    assert result.status == "LP_INITIAL_COMPLETE" and result.root_lp_budget == 1


def test_failed_root_lp_keeps_frozen_late_frontier_probe_and_all_actual_costs(monkeypatch):
    monkeypatch.setattr(native, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    monkeypatch.setattr(frozen, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    options = (Option((1, 0), 1.), Option((3, 0), 1.), Option((0, 1), 1.), Option((0, 3), 1.))
    result = router.select_native_root_dual_face_frame(options, (4, 4), 3, 1, 2, 8192)
    old = select_dual_face_root_canonical_frame(options, (4, 4), 3, 1, 2, 8192)
    equivalent_except_cancelled_scan(result, old)
    assert result.probe_stage == "frontier_late" and result.final_lp.selection.safe_fallback_lp_calls == 2
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= 8192


def test_empty_catalogue_and_zero_queue_return_true_paid_root_checkpoint():
    result = router.select_native_root_dual_face_frame((), (0, 0), 2, 144, 256, 8192)
    assert result.root_lp_budget == result.root_lp.selection.lp_node_budget == 0
    assert result.selection.option_counts == () and result.selection.option_indices == (None, None)
    assert result.selection.canonical_exact and result.status == "LP_ROOT_COMPLETE" and result.probe is None
    assert result.root_lp is result.initial_lp is result.final_lp and not result.resumed
    assert result.routing_preparation.root_snapshot_calls == 1
    assert result.routing_preparation.root_snapshot_heap_entries_read == 0
    search = native.NativeRootZeroDeltaAffineDualReuseIntegerCapacitySearch(CATALOGUES[0], (0, 0), 2)
    roots = search.advance_roots(144)
    assert roots.selection.lp_calls == 0 and roots.selection.lp_node_budget == 1
    with pytest.raises(ValueError, match="already initialized"):
        search.advance_roots(144)
