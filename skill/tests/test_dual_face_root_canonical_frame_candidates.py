"""Independent tiny action enumerations and paid root-face routing checks."""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import combinations_with_replacement, product
from types import SimpleNamespace

import pytest

import algorithm.dual_face_root_canonical_frame_candidates as router
import algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as engine
from algorithm.early_root_canonical_frame_candidates import select_early_root_canonical_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def fees():
    return {name: 0 for name in router.DUAL_FACE_COUNTERS}


def baseline(options, queue, lanes, counts):
    minima = {}
    for option in options:
        minima[option.requirements] = min(minima.get(option.requirements, float("inf")), option.duration)
    raw = engine.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes).advance_to(len(set(minima.values())))
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    duration = max((Fraction(1), *(Fraction.from_float(option.duration) for count, option in
                                zip(counts, options, strict=True) if count)))
    indices = tuple(index for index, count in enumerate(counts) for _ in range(count))
    score = Fraction(sum(q * done for q, done in zip(queue, service, strict=True))) / duration
    return replace(raw.selection, option_counts=tuple(counts), option_indices=indices + (None,) * (lanes - len(indices)),
                   service=service, duration=float(duration), score=float(score), upper_bound=float(score),
                   gap=0.0, epsilon=0.0, exact=False, score_exact=True, canonical_exact=False,
                   canonical_tie_regions_remaining=1, frontier_regions_remaining=1)


def raw_box(threshold, eligible, lower, upper, *, certificate=None, bound=None, threshold_index=0,
            exact=None, solved=True):
    return SimpleNamespace(threshold=Fraction(threshold), threshold_index=threshold_index,
        eligible=tuple(eligible), lower=tuple(lower), upper=tuple(upper),
        bound=Fraction(bound), certificate=certificate, solved=solved,
        exact_duration_positions=tuple(range(len(eligible))) if exact is None else tuple(exact))


def capture(options, queue, lanes, incumbent, requirements, representatives, boxes, work):
    best = SimpleNamespace(score=Fraction(sum(q * done for q, done in zip(queue, incumbent.service, strict=True))) /
                           Fraction.from_float(incumbent.duration), duration=Fraction.from_float(incumbent.duration),
                           indices=incumbent.option_indices)
    local = dict(best=best, frontier=[(box,) for box in boxes], requirements=requirements,
        representatives=representatives, weights=tuple(sum(q * r for q, r in zip(queue, required, strict=True))
                                                       for required in requirements),
        lanes=lanes, dimension=len(queue), option_count=len(options), root_count=len(representatives))
    search = SimpleNamespace(_search=SimpleNamespace(gi_frame=SimpleNamespace(f_locals=local)))
    return router._capture_root_dual_faces(search, work)


def complete(options, queue, lanes, incumbent, evidence, work, budget=8192):
    return router._complete_root_dual_faces(options, queue, lanes, incumbent, budget, 1.0, evidence, work)


def original_key(options, queue, lanes, counts):
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    if sum(counts) > lanes or any(done > q for done, q in zip(service, queue, strict=True)):
        return None
    duration = max((Fraction(1), *(Fraction.from_float(option.duration) for count, option in
                                zip(counts, options, strict=True) if count)))
    score = Fraction(sum(q * done for q, done in zip(queue, service, strict=True))) / duration
    encoded = tuple(index for index, count in enumerate(counts) for _ in range(count))
    return -score, duration, encoded + (len(options),) * (lanes - len(encoded))


def enumerated_key(options, queue, lanes):
    keys = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        counts = tuple(indices.count(i) for i in range(len(options)))
        key = original_key(options, queue, lanes, counts)
        if key is not None:
            keys.append(key)
    return min(keys)


def test_tight_own_dual_fixes_both_signs_and_counts_actual_capture_work():
    options, queue, lanes = (Option((1, 0), 1.), Option((0, 1), 1.)), (1, 3), 2
    incumbent = baseline(options, queue, lanes, (0, 2))
    certificate = engine._ZeroDeltaCertificate(tuple(map(Fraction, (2, 0, 0, 0, 1))), (0, 0), (1, 2), Fraction(6))
    box = raw_box(1, (0, 1), (0, 0), (1, 2), certificate=certificate, bound=6)
    work = fees()
    evidence = capture(options, queue, lanes, incumbent, ((1, 0), (0, 1)), ({0: 0, 1: 1},), (box,), work)
    face, = evidence.regions
    assert face.tight and face.own_numerator_fraction == face.target_numerator_fraction == (6, 1)
    assert face.lower_coefficients == ((-1, 1), (0, 1)) and face.upper_coefficients == ((0, 1), (1, 1))
    assert face.face_lower == face.face_upper == (0, 2)
    assert certificate.prepared is None
    assert work == {**fees(), "capture_calls": 1, "heap_entries_read": 1, "canonical_box_checks": 1,
        "canonical_key_variable_visits": 2, "canonical_key_count_reads": 10,
        "canonical_key_integer_operations": 10, "canonical_key_indices_written": 4,
        "canonical_key_comparisons": 1, "boxes_exported": 1, "group_ids_copied": 2,
        "requirement_coordinates_copied": 4, "representative_options_copied": 2,
        "box_bound_values_copied": 4, "exact_positions_copied": 2, "fraction_pairs_copied": 16,
        "certificate_price_reads": 13, "certificate_bound_reads": 8, "coefficient_requirement_reads": 4,
        "exact_integer_operations": 4, "exact_rational_operations": 23, "own_bound_comparisons": 1,
        "coefficient_sign_comparisons": 4, "fixed_lower_coordinates": 1, "fixed_upper_coordinates": 1,
        "face_bound_values_written": 10}
    for counts in product(range(lanes + 1), repeat=2):
        key = original_key(options, queue, lanes, counts)
        if key is not None and key[0] == -6:
            assert all(left <= count <= right for left, count, right in
                       zip(face.face_lower, counts, face.face_upper, strict=True))
    selected, completion = complete(options, queue, lanes, incumbent, evidence, work)
    assert selected.canonical_exact and selected.option_counts == (0, 2)
    assert completion.states_visited == 1 and completion.complete


@pytest.mark.parametrize("solved", (False, True))
def test_failed_or_solved_child_uses_old_certificate_source_delta_without_mutating_cache(solved):
    options, queue, lanes = (Option((1, 0), 1.), Option((0, 1), 1.)), (1, 3), 2
    incumbent = baseline(options, queue, lanes, (0, 2))
    sentinel = object()
    certificate = engine._ZeroDeltaCertificate(tuple(map(Fraction, (2, 0, 0, 0, 1))),
        (0, 0), (1, 3), Fraction(7), prepared=sentinel)
    box = raw_box(1, (0, 1), (0, 0), (1, 2), certificate=certificate, bound=6, solved=solved)
    work = fees()
    evidence = capture(options, queue, lanes, incumbent, ((1, 0), (0, 1)), ({0: 0, 1: 1},), (box,), work)
    face, = evidence.regions
    assert face.certificate_numerator_fraction == (7, 1) and face.own_numerator_fraction == (6, 1)
    assert face.certificate_upper == (1, 3) and face.tight and face.face_lower == (0, 2)
    assert certificate.upper == (1, 3) and certificate.numerator == 7 and certificate.prepared is sentinel


@pytest.mark.parametrize("certificate", (None,
    engine._ZeroDeltaCertificate(tuple(map(Fraction, (4, 0, 0, 0, 0))), (0, 0), (1, 2), Fraction(8))))
def test_missing_or_loose_own_dual_keeps_whole_box_despite_tighter_inherited_bound(certificate):
    options, queue, lanes = (Option((1, 0), 1.), Option((0, 1), 1.)), (1, 3), 2
    incumbent = baseline(options, queue, lanes, (0, 2))
    box = raw_box(1, (0, 1), (0, 0), (1, 2), certificate=certificate, bound=6)
    work = fees()
    evidence = capture(options, queue, lanes, incumbent, ((1, 0), (0, 1)), ({0: 0, 1: 1},), (box,), work)
    face, = evidence.regions
    assert not face.tight and face.face_lower == (0, 0) and face.face_upper == (1, 2)
    assert face.own_numerator_fraction == (None if certificate is None else (8, 1))
    assert work["fixed_lower_coordinates"] == work["fixed_upper_coordinates"] == 0
    selected, completion = complete(options, queue, lanes, incumbent, evidence, work)
    assert completion.complete and original_key(options, queue, lanes, selected.option_counts) == enumerated_key(options, queue, lanes)


@pytest.mark.parametrize("budget,complete_expected", ((1, False), (2, True)))
def test_all_same_duration_boxes_compared_and_partial_winner_discarded_on_exhaustion(budget, complete_expected):
    options, queue, lanes = (Option((1, 0), 1.), Option((0, 1), 1.)), (2, 2), 2
    incumbent = baseline(options, queue, lanes, (0, 2))
    boxes = (raw_box(1, (0, 1), (1, 1), (1, 1), bound=4),
             raw_box(1, (0, 1), (2, 0), (2, 0), bound=4))
    work = fees()
    evidence = capture(options, queue, lanes, incumbent, ((1, 0), (0, 1)), ({0: 0, 1: 1},), boxes, work)
    selected, completion = complete(options, queue, lanes, incumbent, evidence, work, budget)
    assert completion.complete == complete_expected and completion.states_visited == budget
    if complete_expected:
        assert selected.canonical_exact and selected.option_counts == (2, 0)
        assert work["candidate_key_comparisons"] == 2
    else:
        assert selected == incumbent and not selected.canonical_exact
        assert completion.status == "STATE_BUDGET_EXHAUSTED" and work["candidate_key_comparisons"] == 1


def test_minimum_group_durations_and_threshold_specific_original_representatives():
    options = (Option((2, 0), 2.), Option((2, 0), 1.), Option((0, 2), 2.))
    queue, lanes = (2, 4), 1
    incumbent = baseline(options, queue, lanes, (0, 0, 1))
    boxes = (raw_box(2, (0, 1), (0, 0), (1, 1), bound=4, threshold_index=1, exact=(1,)),
             raw_box(1, (0,), (0,), (1,), bound=4, exact=(0,)))
    work = fees()
    evidence = capture(options, queue, lanes, incumbent, ((2, 0), (0, 2)),
                       ({0: 1}, {0: 0, 1: 2}), boxes, work)
    selected, completion = complete(options, queue, lanes, incumbent, evidence, work)
    assert selected.option_counts == (0, 1, 0) and selected.duration == 1.
    assert completion.thresholds_checked == 1 and work["domain_box_visits"] == 1
    assert original_key(options, queue, lanes, selected.option_counts) == enumerated_key(options, queue, lanes)


def test_longer_original_member_is_canonical_inside_an_attained_later_group_duration():
    options = (Option((1, 0), 2.), Option((1, 0), 1.), Option((0, 2), 2.))
    queue, lanes = (1, 2), 2
    incumbent = baseline(options, queue, lanes, (0, 1, 1))
    assert incumbent.score == 2.5
    box = raw_box(2, (0, 1), (0, 0), (1, 1), bound=Fraction(5, 2), threshold_index=1, exact=(1,))
    work = fees()
    evidence = capture(options, queue, lanes, incumbent, ((1, 0), (0, 2)),
                       ({0: 1}, {0: 0, 1: 2}), (box,), work)
    selected, completion = complete(options, queue, lanes, incumbent, evidence, work)
    assert completion.complete and selected.option_counts == (1, 0, 1)
    assert original_key(options, queue, lanes, selected.option_counts) == enumerated_key(options, queue, lanes)


def test_capture_excludes_raw_closed_boxes_and_domain_keeps_baseline_as_virtual_candidate():
    options, queue, lanes = (Option((1, 0), 1.), Option((0, 1), 1.)), (1, 3), 2
    incumbent = baseline(options, queue, lanes, (0, 2))
    boxes = (raw_box(1, (0, 1), (0, 0), (0, 2), bound=6),
             raw_box(1, (0, 1), (0, 0), (1, 2), bound=5))
    work = fees()
    evidence = capture(options, queue, lanes, incumbent, ((1, 0), (0, 1)), ({0: 0, 1: 1},), boxes, work)
    assert evidence.raw_region_count == 2 and evidence.live_region_count == 0
    selected, completion = complete(options, queue, lanes, incumbent, evidence, work)
    assert selected.option_counts == incumbent.option_counts and selected.canonical_exact
    assert completion.states_visited == 0 and work["boxes_exported"] == 0


def test_early_exhaustion_preserves_true_native_stages_one_search_and_one_probe(monkeypatch):
    options, queue, lanes = (Option((1, 0), 1.), Option((0, 1), 1.)), (1, 3), 2
    native = router.ZeroDeltaAffineDualReuseIntegerCapacitySearch
    constructions, advances, probes = [], [], []
    actual_probe = router._dual_face_root_probe

    class CountedSearch(native):
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            super().__init__(*args, **kwargs)

        def advance_to(self, cap):
            advances.append(cap)
            return super().advance_to(cap)

    def probe(*args):
        probes.append(args[3].selection.lp_node_budget)
        return actual_probe(*args)

    monkeypatch.setattr(router, "ZeroDeltaAffineDualReuseIntegerCapacitySearch", CountedSearch)
    monkeypatch.setattr(router, "_dual_face_root_probe", probe)
    result = router.select_dual_face_root_canonical_frame(options, queue, lanes, 144, 256, 0)
    old = select_early_root_canonical_frame(options, queue, lanes, 144, 256, 0)
    assert constructions == [1] and probes == [1] and advances == [1, 144]
    assert result.root_lp == old.root_lp and result.initial_lp == old.initial_lp and result.final_lp == old.final_lp
    assert result.selection == old.selection and result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"
    assert result.probe.selection == result.root_lp.selection and result.probe_stage == "root"
    assert result.root_lp_budget == 1 and result.initial_lp_budget == 144 and result.lp_node_budget == 256
    assert result.dual_face_work.capture_calls == 1 and result.routing_preparation == old.routing_preparation


def test_root_score_open_late_route_is_frozen_early_route_without_face_export(monkeypatch):
    monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    options = (Option((1, 0), 1.), Option((3, 0), 1.), Option((0, 1), 1.), Option((0, 3), 1.))
    result = router.select_dual_face_root_canonical_frame(options, (4, 4), 3, 1, 2, 8192)
    old = select_early_root_canonical_frame(options, (4, 4), 3, 1, 2, 8192)
    record = asdict(result)
    assert record.pop("dual_face_evidence") is None and record.pop("dual_face_work") == fees()
    assert record == asdict(old) and result.probe_stage == "frontier_late"


CATALOGUES = (
    (Option((1, 0), 1.), Option((0, 1), 1.)),
    (Option((2, 0), 2.), Option((2, 0), 1.), Option((0, 1), 1.5)),
    (Option((1, 0), 2.), Option((1, 0), 1.), Option((0, 2), 2.)),
    (Option((1, 1), 1.), Option((2, 0), 1.5), Option((0, 1), 1.), Option((1, 1), 2.)),
)


@pytest.mark.parametrize("options", CATALOGUES)
@pytest.mark.parametrize("queue,lanes", (((0, 0), 2), ((1, 3), 2), ((2, 2), 3), ((3, 2), 2)))
def test_tiny_whole_router_matches_independent_original_action_argmax(options, queue, lanes):
    result = router.select_dual_face_root_canonical_frame(options, queue, lanes, 144, 256, 8192)
    assert result.selection.canonical_exact
    assert original_key(options, queue, lanes, result.selection.option_counts) == enumerated_key(options, queue, lanes)
    assert result.root_lp_budget == result.root_lp.selection.lp_node_budget
    assert result.initial_lp_budget == 144 and result.lp_node_budget == 256 and result.integer_state_budget == 8192
    if result.probe is not None:
        assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= 8192
    if result.probe_stage == "root":
        assert result.dual_face_evidence.live_region_count == result.root_lp.selection.frontier_regions_remaining
        assert result.dual_face_work.capture_calls == 1
    else:
        assert result.dual_face_evidence is None and asdict(result.dual_face_work) == fees()
