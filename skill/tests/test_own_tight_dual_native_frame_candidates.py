"""Exact native faces, paid caching, and original scheduling argmax."""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import combinations_with_replacement, product
from types import SimpleNamespace

import pytest

import algorithm.fused_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as frozen
import algorithm.own_tight_dual_native_frame_candidates as router
import algorithm.own_tight_dual_native_kernel as native


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


TIED = (Option((1, 0), 1.), Option((0, 1), 1.))
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


def ordinary(stage):
    record = asdict(stage)
    record.pop("native_own_tight_work")
    record.pop("native_own_tight_evidence")
    return record


def check_cache(search):
    local = search._search.gi_frame.f_locals
    for entry in local["frontier"]:
        region, dimension = entry[-1], local["dimension"]
        certificate = region.certificate
        if certificate is None:
            assert region.own_bound is None
            continue
        prices = certificate.prices
        own = certificate.numerator
        for pos, group in enumerate(region.eligible):
            alpha = local["weights"][group] - prices[0] - prices[1 + dimension + pos]
            alpha -= sum(price * requirement for price, requirement in
                         zip(prices[1:1 + dimension], local["requirements"][group], strict=True))
            own += alpha * (region.lower[pos] - certificate.lower[pos])
            own += prices[1 + dimension + pos] * (region.upper[pos] - certificate.upper[pos])
        assert region.own_bound == own / region.threshold


def check_events(stage, queue):
    work, events = stage.native_own_tight_work, stage.native_own_tight_evidence
    assert work.checks == work.bound_comparisons
    assert work.lower_flag_reads == work.upper_price_reads == work.upper_price_comparisons == work.variable_visits
    assert work.box_coordinate_reads == 2 * work.variable_visits
    assert work.box_coordinate_comparisons == work.variable_visits
    assert work.bound_values_copied == 4 * work.variable_visits
    assert work.bound_values_written == work.changed_lower_coordinates + work.changed_upper_coordinates
    assert work.changed_regions == work.event_records_written == len(events)
    assert work.own_bound_comparisons == work.own_bound_reads <= work.certificate_reads <= work.checks
    assert work.event_fraction_pairs_written == sum(len(e.certificate_prices) + 9 for e in events)
    assert work.event_rational_operations == sum(2 + (e.anchor_kind == "affine") for e in events)
    assert work.event_bound_values_copied == sum(len(e.eligible) * (6 if e.anchor_kind == "affine" else 4) for e in events)
    for event in events:
        score, threshold = Fraction(*event.incumbent_score_fraction), Fraction(*event.threshold_fraction)
        assert Fraction(*event.current_own_numerator_fraction) == score * threshold
        assert Fraction(*event.own_bound_fraction) == score
        prices = tuple(Fraction(*pair) for pair in event.certificate_prices)
        dimension = len(event.requirements[0])
        own = anchor = Fraction(*event.certificate_numerator_fraction)
        expected_lower, expected_upper = list(event.before_lower), list(event.before_upper)
        for pos, required in enumerate(event.requirements):
            cap = prices[1 + dimension + pos]
            alpha = sum(q * r for q, r in zip(queue, required, strict=True)) - prices[0] - cap
            alpha -= sum(p * r for p, r in zip(prices[1:1 + dimension], required, strict=True))
            assert alpha <= 0 and cap >= 0
            own += alpha * (event.before_lower[pos] - event.certificate_lower[pos])
            own += cap * (event.before_upper[pos] - event.certificate_upper[pos])
            anchor += alpha * (event.anchor_lower[pos] - event.certificate_lower[pos])
            anchor += cap * (event.anchor_upper[pos] - event.certificate_upper[pos])
            if alpha < 0:
                expected_upper[pos] = event.before_lower[pos]
            if cap > 0:
                expected_lower[pos] = event.before_upper[pos]
        assert own == Fraction(*event.current_own_numerator_fraction)
        assert anchor == Fraction(*event.anchor_numerator_fraction)
        assert tuple(expected_lower) == event.face_lower and tuple(expected_upper) == event.face_upper
        assert all(left <= new_left and new_right <= right for left, right, new_left, new_right in
                   zip(event.before_lower, event.before_upper, event.face_lower, event.face_upper, strict=True))
        assert all(new_right <= face_right for new_right, face_right in zip(event.after_upper, event.face_upper, strict=True))


def tied_lp(real, calls, fail_after_first=False):
    def solve(*args, **kwargs):
        calls.append(tuple(args[0]))
        if len(calls) == 1:
            return SimpleNamespace(success=True, x=(0., 2.),
                ineqlin=SimpleNamespace(marginals=(-2., 0., 0., 0., 0.)))
        if fail_after_first:
            return SimpleNamespace(success=False)
        return real(*args, **kwargs)
    return solve


def test_simultaneous_endpoints_preserve_every_exact_dual_face_integer_point():
    # A nonnegative box price and nonpositive reduced coefficient yield a
    # nonnegative dual gap c(U-x) + (-alpha)(x-L). Every zero-gap point stays.
    for left, right in ((0, 2), (1, 1), (1, 3)):
        for alpha, cap in product((Fraction(0), Fraction(-1)), (Fraction(0), Fraction(2))):
            lower, upper, changed_l, changed_u, empty = native._own_tight_endpoints(
                (left,), (right,), (alpha == 0,), (cap,), 0)
            exact = [x for x in range(left, right + 1) if cap * (right - x) - alpha * (x - left) == 0]
            assert all(lower[0] <= x <= upper[0] for x in exact)
            assert empty == (alpha < 0 and cap > 0 and left < right)
            assert changed_l + changed_u == (lower[0] != left) + (upper[0] != right)
            if empty:
                assert not exact and lower[0] > upper[0]


@pytest.mark.parametrize("options", CATALOGUES)
@pytest.mark.parametrize("queue,lanes", (((0, 0), 2), ((3, 2), 3)))
def test_original_argmax_cross_duration_representatives_and_real_raw_m(options, queue, lanes):
    search = native.OwnTightDualNativeSearch(options, queue, lanes)
    old = frozen.FusedRootZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    root, after, phase = search.advance_roots_and_settle(144)
    prior, _, _ = old.advance_roots_and_settle(144)
    assert ordinary(root) == asdict(prior)
    assert root.native_own_tight_evidence == () and root.native_own_tight_work.checks == 0
    assert root.native_own_tight_work.own_bound_cache_reads >= root.selection.lp_calls
    assert root.native_own_tight_work.own_bound_cache_writes >= root.selection.lp_calls
    generator = search._search
    stages = [root]
    if after is not None:
        stages.append(after)
        assert phase.lp_calls_before == phase.lp_calls_after
    check_cache(search)
    for cap in (144, 256):
        stage = search.advance_to(cap)
        assert search._search is generator
        stages.append(stage)
        check_cache(search)
    for before, stage in zip(stages, stages[1:]):
        assert stage.native_own_tight_evidence[:len(before.native_own_tight_evidence)] == before.native_own_tight_evidence
        assert all(value >= getattr(before.native_own_tight_work, name) for name, value in asdict(stage.native_own_tight_work).items())
    check_events(stages[-1], queue)
    result = router.select_own_tight_dual_native_frame(options, queue, lanes, 144, 256, 8192)
    assert result.selection.canonical_exact and key(options, queue, lanes, result.selection.option_counts) == argmax(options, queue, lanes)
    assert result.initial_lp_budget == 144 and result.lp_node_budget == 256 and result.integer_state_budget == 8192
    check_events(result.final_lp, queue)


def test_exact_flags_fix_native_root_without_affine_preparation_or_extra_lp():
    search = native.OwnTightDualNativeSearch(TIED, (1, 3), 2)
    root, after, phase = search.advance_roots_and_settle(144)
    assert not root.selection.canonical_exact and after.selection.canonical_exact
    assert phase.lp_calls_before == phase.lp_calls_after == 1
    assert after.native_own_tight_evidence and after.affine_dual.certificates_prepared == 0
    assert after.native_own_tight_work.changed_upper_coordinates >= 1
    assert after.native_own_tight_work.capacity_rechecks == 0
    assert after.native_own_tight_work.lower_feasibility_rechecks == 0
    assert after.native_own_tight_work.event_snapshot_entries_copied == len(after.native_own_tight_evidence)
    assert search.advance_to(root.selection.lp_node_budget) is after
    check_events(after, (1, 3))


def test_tighter_inherited_bound_never_substitutes_for_loose_own_certificate(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "linprog", tied_lp(native.linprog, calls))
    search = native.OwnTightDualNativeSearch(TIED, (2, 2), 2)
    search.advance_roots_and_settle(144)
    local = search._search.gi_frame.f_locals
    original = local["frontier"][0][-1]
    certificate = native._ZeroDeltaCertificate((Fraction(3), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (0, 0), (2, 2), Fraction(6), lower_zero_flags=(False, False))
    loose = replace(original, certificate=certificate, dual_lower=None, dual_upper=None,
                    dual_numerator=None, bound=local["best"].score, own_bound=Fraction(6))
    count = len(local["own_events"])
    assert local["shrink_own_tight"](loose) is loose
    assert len(local["own_events"]) == count and certificate.prepared is None


def test_global_score_open_still_allows_local_own_tight_stratum_fixing(monkeypatch):
    monkeypatch.setattr(native, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    options = (Option((1, 0), 1.), Option((3, 0), 1.), Option((0, 1), 1.), Option((0, 3), 1.))
    search = native.OwnTightDualNativeSearch(options, (4, 1), 3)
    root, after, phase = search.advance_roots_and_settle(144)
    assert not root.selection.score_exact and after is phase is None
    local = search._search.gi_frame.f_locals
    region = local["frontier"][0][-1]
    certificate = native._ZeroDeltaCertificate(tuple(map(Fraction, (4, 0, 0, 0, 8, 0, 0))),
        (0, 0, 0, 0), (3, 0, 1, 0), Fraction(12), lower_zero_flags=(True, True, False, False))
    tied = replace(region, lower=certificate.lower, upper=certificate.upper,
                   certificate=certificate, own_bound=Fraction(12), bound=Fraction(12))
    assert local["shrink_own_tight"](tied) is None
    event = local["own_events"][-1]
    assert event.face_upper == (3, 0, 0, 0) and event.disposition == "canonical_pruned"
    assert root.selection.upper_bound > Fraction(*event.current_own_numerator_fraction)
    assert event.exact_duration_positions == tied.exact_duration_positions


def test_strict_lower_and_upper_prices_only_empty_an_open_coordinate(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "linprog", tied_lp(native.linprog, calls))
    search = native.OwnTightDualNativeSearch(TIED, (2, 2), 2)
    search.advance_roots_and_settle(144)
    local = search._search.gi_frame.f_locals
    region = local["frontier"][0][-1]
    certificate = native._ZeroDeltaCertificate(tuple(map(Fraction, (0, 0, 0, 4, 2))),
        (0, 0), (1, 0), Fraction(4), lower_zero_flags=(False, True))
    candidate = replace(region, lower=certificate.lower, upper=certificate.upper,
        certificate=certificate, own_bound=Fraction(4), bound=Fraction(4),
        dual_lower=None, dual_upper=None, dual_numerator=None)
    assert local["shrink_own_tight"](candidate) is None
    event = local["own_events"][-1]
    assert event.disposition == "direct_empty" and event.face_lower[0] > event.face_upper[0]
    lower, upper, changed_l, changed_u, empty = native._own_tight_endpoints((1,), (1,), (False,), (Fraction(4),), 0)
    assert lower == upper == (1,) and changed_l == changed_u == 0 and not empty


def test_failed_lp_keeps_current_own_cache_across_old_anchors_and_budget_continuation(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "linprog", tied_lp(native.linprog, calls, fail_after_first=True))
    search = native.OwnTightDualNativeSearch(TIED, (2, 2), 2)
    root, after, phase = search.advance_roots_and_settle(144)
    assert phase.stopped_for_lp and phase.lp_calls_before == phase.lp_calls_after == 1
    check_cache(search)
    assert any(entry[-1].certificate.lower != entry[-1].lower or entry[-1].certificate.upper != entry[-1].upper
               for entry in search._search.gi_frame.f_locals["frontier"])
    generator = search._search
    stage = search.advance_to(root.selection.duration_root_count + 1)
    assert stage.selection.safe_fallback_lp_calls == 1
    check_cache(search)
    final = search.advance_to(144)
    assert search._search is generator and final.selection.canonical_exact
    assert key(TIED, (2, 2), 2, final.selection.option_counts) == argmax(TIED, (2, 2), 2)
    check_events(final, (2, 2))


def test_mandatory_lower_capacity_only_changes_zero_price_upper_bounds(monkeypatch):
    options = (Option((1, 0), 1.), Option((0, 1), 1.), Option((1, 1), 1.))
    real = native.linprog
    calls = []

    def alternate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(success=True, x=(0., 0., 2.),
                ineqlin=SimpleNamespace(marginals=(0., -1., -2., -1., 0., -1.)))
        return real(*args, **kwargs)

    monkeypatch.setattr(native, "linprog", alternate)
    search = native.OwnTightDualNativeSearch(options, (2, 2), 2)
    search.advance_roots_and_settle(144)
    local = search._search.gi_frame.f_locals
    certificate = native._ZeroDeltaCertificate(tuple(map(Fraction, (0, 1, 2, 1, 0, 1))),
        (0, 0, 0), (2, 2, 2), Fraction(10), lower_zero_flags=(True, True, True))
    certificate.prepared = native._prepare_affine_certificate(certificate, ((1, 0), (0, 1), (1, 1)), (2, 2, 4))
    region = native._AffineRegion(0, Fraction(1), (0, 1, 2), (0, 1, 2),
        (0, 0, 0), (1, 2, 1), Fraction(8), False, None, certificate.prices, certificate,
        (0, 0, 0), (1, 2, 1), 8, Fraction(8))
    before_cap = local["own_fees"]["capacity_rechecks"]
    after = local["shrink_own_tight"](region)
    assert after is not None and after.lower == after.upper == (1, 0, 1)
    assert after.own_bound == Fraction(8)
    event = local["own_events"][-1]
    assert event.anchor_kind == "affine" and event.disposition == "kept"
    assert event.face_lower == (1, 0, 1) and event.face_upper == (1, 2, 1)
    assert event.after_upper == (1, 0, 1) and Fraction(*event.current_own_numerator_fraction) == 8
    assert local["own_fees"]["capacity_rechecks"] == before_cap + 1


def test_nonzero_events_failed_probe_and_different_snapshot_copy_costs_are_all_real(monkeypatch):
    options = (Option((1, 0, 0), 1.), Option((0, 1, 0), 1.), Option((0, 0, 1), 1.))
    real = native.linprog
    calls = []

    def alternate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(success=True, x=(0., 2., 0.),
                ineqlin=SimpleNamespace(marginals=(-3., 0., 0., 0., 0., 0., 0.)))
        return real(*args, **kwargs)

    monkeypatch.setattr(native, "linprog", alternate)
    result = router.select_own_tight_dual_native_frame(options, (3, 3, 1), 2, 144, 256, 0)
    assert result.settled_lp.native_own_tight_evidence
    assert result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"
    assert result.probe_stage == "root" and result.status == "LP_INITIAL_COMPLETE"
    assert result.selection.canonical_exact and result.selection.option_counts == (2, 0, 0)
    check_events(result.final_lp, (3, 3, 1))
    calls.clear()
    audit = native.OwnTightDualNativeSearch(options, (3, 3, 1), 2)
    root, settled, phase = audit.advance_roots_and_settle(144)
    audit.advance_to(root.selection.duration_root_count + 1)
    final = audit.advance_to(144)
    a, b = asdict(result.final_lp), asdict(final)
    direct_copies = a["native_own_tight_work"].pop("event_snapshot_entries_copied")
    audit_copies = b["native_own_tight_work"].pop("event_snapshot_entries_copied")
    assert a == b and audit_copies > direct_copies
    assert direct_copies == len(settled.native_own_tight_evidence) + len(result.final_lp.native_own_tight_evidence)


@pytest.mark.parametrize("budget", (0, 8192))
def test_single_probe_and_same_search_exhaustion_continuation_with_all_actual_event_copy_costs(monkeypatch, budget):
    calls = []
    monkeypatch.setattr(native, "linprog", tied_lp(native.linprog, calls))
    engine = router.OwnTightDualNativeSearch
    constructions, advances, generator = [], [], []

    class CountedSearch(engine):
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            super().__init__(*args, **kwargs)

        def advance_roots_and_settle(self, cap):
            result = super().advance_roots_and_settle(cap)
            generator.append(self._search)
            return result

        def advance_to(self, cap):
            advances.append(cap)
            assert self._search is generator[0]
            return super().advance_to(cap)

    monkeypatch.setattr(router, "OwnTightDualNativeSearch", CountedSearch)
    result = router.select_own_tight_dual_native_frame(TIED, (2, 2), 2, 144, 256, budget)
    assert constructions == [1] and result.probe_stage == "root"
    assert result.probe.baseline_selection == result.settled_lp.selection
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= budget
    assert result.selection.canonical_exact and result.selection.option_counts == (2, 0)
    if budget:
        assert advances == [] and len(calls) == 1
    else:
        assert advances == [144] and result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"
        assert result.probe.selection == result.settled_lp.selection
    check_events(result.final_lp, (2, 2))
