"""Exact projected LP proofs, original-action optimality and paid continuation."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

import algorithm.small_dimension_exact_native_kernel as native
import algorithm.small_dimension_exact_native_frame_candidates as router
import algorithm.direct_probe_own_tight_native_kernel as parent


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


THREE = (Option((1, 0, 0), 1.), Option((0, 1, 0), 1.), Option((0, 0, 1), 1.))
CATALOGUES = (
    (Option((1, 0), 1.), Option((0, 1), 1.)),
    (Option((2, 1), 1.), Option((1, 2), 1.)),
    (Option((2, 0), 2.), Option((2, 0), 1.), Option((0, 1), 1.5)),
    (Option((1, 0), 2.), Option((1, 0), 1.), Option((0, 2), 2.)),
)


def packing(requirements, queue, lanes, lower, upper):
    size = len(requirements)
    objective = tuple(sum(q * amount for q, amount in zip(queue, req, strict=True)) for req in requirements)
    matrix = [[1] * size]
    matrix.extend([[req[c] for req in requirements] for c in range(len(queue))])
    matrix.extend([[int(column == row) for column in range(size)] for row in range(size)])
    rhs = [lanes - sum(lower)]
    rhs.extend([queue[c] - sum(value * req[c] for value, req in zip(lower, requirements, strict=True))
                for c in range(len(queue))])
    rhs.extend(right - left for left, right in zip(lower, upper, strict=True))
    base = sum(value * weight for value, weight in zip(lower, objective, strict=True))
    return matrix, rhs, objective, base


def assert_proof(matrix, rhs, objective, base, primal, prices, flags, numerator):
    assert len(primal) == len(objective) and len(prices) == len(matrix)
    assert all(value >= 0 for value in primal) and all(price >= 0 for price in prices)
    assert all(sum(a * x for a, x in zip(row, primal, strict=True)) <= limit
               for row, limit in zip(matrix, rhs, strict=True))
    for column, weight in enumerate(objective):
        lhs = sum(row[column] * price for row, price in zip(matrix, prices, strict=True))
        assert lhs >= weight and flags[column] == (lhs == weight)
    assert numerator == base + sum(value * weight for value, weight in zip(primal, objective, strict=True))
    assert numerator == base + sum(price * limit for price, limit in zip(prices, rhs, strict=True))


def assert_event(event, queue, lanes):
    matrix, rhs, objective, base = packing(event.requirements, queue, lanes, event.lower, event.upper)
    primal = tuple(Fraction(*value) - left for value, left in zip(event.primal_fractions, event.lower, strict=True))
    prices = tuple(Fraction(*value) for value in event.price_fractions)
    flags = tuple(sum(row[column] * price for row, price in zip(matrix, prices, strict=True)) == weight
                  for column, weight in enumerate(objective))
    assert event.objective == objective and event.base_objective == base
    assert event.active_positions == tuple(i for i, (left, right) in enumerate(zip(event.lower, event.upper, strict=True))
                                          if left < right)
    assert len(event.active_positions) <= 2
    assert event.method == ("exact_0d", "exact_1d", "exact_2d")[len(event.active_positions)]
    assert_proof(matrix, rhs, objective, base, primal, prices, flags, Fraction(*event.numerator_fraction))


def action_key(options, queue, lanes, counts):
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    if sum(counts) > lanes or any(done > q for done, q in zip(service, queue, strict=True)):
        return None
    duration = max((Fraction(1), *(Fraction.from_float(option.duration) for count, option in
                                 zip(counts, options, strict=True) if count)))
    score = Fraction(sum(q * done for q, done in zip(queue, service, strict=True))) / duration
    expanded = tuple(i for i, count in enumerate(counts) for _ in range(count))
    return -score, duration, expanded + (len(options),) * (lanes - len(expanded))


def original_argmax(options, queue, lanes):
    return min(value for indices in combinations_with_replacement(range(len(options) + 1), lanes)
               if (value := action_key(options, queue, lanes, tuple(indices.count(i) for i in range(len(options))))) is not None)


@pytest.mark.parametrize("requirements,queue,lanes,lower,upper,expected", (
    (((1, 0), (0, 2), (2, 1)), (7, 7), 6, (1, 2, 2), (1, 2, 2), (Fraction(0),) * 3),
    (((1, 0), (0, 2), (2, 1)), (7, 7), 6, (1, 0, 2), (1, 2, 2), (Fraction(0), Fraction(2), Fraction(0))),
    (((2, 1), (1, 2), (1, 0)), (6, 6), 5, (0, 0, 1), (2, 3, 1),
     (Fraction(4, 3), Fraction(7, 3), Fraction(0))),
    (((1, 0), (0, 1)), (1, 3), 2, (0, 0), (1, 2), (Fraction(0), Fraction(2))),
))
def test_exact_geometry_has_full_primal_dual_proof_and_fixed_column_completion(
        requirements, queue, lanes, lower, upper, expected):
    matrix, rhs, objective, base = packing(requirements, queue, lanes, lower, upper)
    active = tuple(i for i, (left, right) in enumerate(zip(lower, upper, strict=True)) if left < right)
    fees = {name: 0 for name in native.SMALL_DIMENSION_LP_COUNTERS}
    primal, prices, flags, numerator = native._exact_small_packing_lp(matrix, rhs, objective, base, active, fees)
    assert primal == expected
    assert_proof(matrix, rhs, objective, base, primal, prices, flags, numerator)
    assert fees["full_dual_column_checks"] == len(lower)
    assert fees["full_dual_matrix_reads"] == len(matrix) * len(lower)
    assert fees["fixed_column_prices_written"] == len(lower) - len(active)
    assert fees["projection_matrix_reads"] == len(matrix) * len(active)
    assert fees["projection_rhs_reads"] == len(matrix)
    assert (fees["vertex_pairs_checked"] > 0) == (len(active) == 2)


def test_small_native_roots_replace_generic_calls_but_keep_logical_lp_budget(monkeypatch):
    def no_generic(*args, **kwargs):
        raise AssertionError("small projections must not call HiGHS")
    monkeypatch.setattr(native, "linprog", no_generic)
    options = (Option((1,), 1.), Option((3,), 2.))
    search = native.SmallDimensionExactNativeSearch(options, (20,), 4)
    root, settled, phase = search.advance_roots_and_settle(144)
    assert root.selection.duration_root_count == root.selection.lp_node_budget == 2
    assert root.selection.lp_calls == root.selection.successful_lp_calls == root.selection.dual_certified_lp_calls == 2
    assert root.small_dimension_lp_work.generic_lp_calls == 0
    assert root.small_dimension_lp_work.exact_lp_calls == 2
    assert root.small_dimension_lp_work.exact_one_calls == root.small_dimension_lp_work.exact_two_calls == 1
    assert len(root.small_dimension_lp_evidence) == 2
    assert [event.lp_call_index for event in root.small_dimension_lp_evidence] == [1, 2]
    assert root.native_own_tight_work.checks == 0 and not root.native_own_tight_evidence
    for event in root.small_dimension_lp_evidence:
        assert_event(event, (20,), 4)
    generator = search._search
    final = search.advance_to(256)
    assert search._search is generator and final.selection.canonical_exact
    assert action_key(options, (20,), 4, final.selection.option_counts) == original_argmax(options, (20,), 4)


@pytest.mark.parametrize("options", CATALOGUES)
@pytest.mark.parametrize("queue,lanes", (((0, 0), 2), ((3, 2), 3), ((6, 6), 4)))
def test_original_action_exhaustive_quality_across_duration_groups_and_ties(options, queue, lanes):
    result = router.select_small_dimension_exact_native_frame(options, queue, lanes, 144, 256, 8192)
    assert result.selection.canonical_exact
    assert action_key(options, queue, lanes, result.selection.option_counts) == original_argmax(options, queue, lanes)
    assert (result.initial_lp_budget, result.lp_node_budget, result.integer_state_budget) == (144, 256, 8192)
    final = result.final_lp
    work = final.small_dimension_lp_work
    assert work.generic_lp_calls + work.exact_lp_calls == final.selection.lp_calls
    assert work.exact_lp_calls == work.exact_zero_calls + work.exact_one_calls + work.exact_two_calls
    assert work.exact_lp_calls == len(final.small_dimension_lp_evidence)
    assert work.generic_lp_calls == 0
    for event in final.small_dimension_lp_evidence:
        assert_event(event, queue, lanes)


def controlled_generic(calls, fail_second=False):
    def lp(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(map(tuple, kwargs["A_ub"])), tuple(kwargs["b_ub"])))
        if fail_second and len(calls) == 2:
            return SimpleNamespace(success=False)
        assert len(calls) == 1
        return SimpleNamespace(success=True, x=(0., 2., 0.),
            ineqlin=SimpleNamespace(marginals=(-3., 0., 0., 0., 0., 0., 0.)))
    return lp


@pytest.mark.parametrize("probe_budget", (0, 8192))
def test_own_tight_projection_and_single_probe_resume_the_same_generator(monkeypatch, probe_budget):
    calls = []
    monkeypatch.setattr(native, "linprog", controlled_generic(calls))
    engine = router.SmallDimensionExactNativeSearch
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

    monkeypatch.setattr(router, "SmallDimensionExactNativeSearch", CountedSearch)
    result = router.select_small_dimension_exact_native_frame(THREE, (3, 3, 1), 2, 144, 256, probe_budget)
    assert constructions == [1] and len(calls) == 1
    assert result.root_settle_work.split_regions == 0 and result.root_settle_work.stopped_for_branch == 1
    assert result.probe_stage == "root" and result.probe.baseline_selection == result.settled_lp.selection
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= probe_budget
    assert result.selection.canonical_exact and result.selection.option_counts == (2, 0, 0)
    assert len(result.settled_lp.native_own_tight_evidence) == 1
    final = result.final_lp
    assert final.small_dimension_lp_work.generic_lp_calls == 1
    if probe_budget:
        assert advances == [] and final.small_dimension_lp_work.exact_lp_calls == 0
    else:
        assert advances == [144] and final.small_dimension_lp_work.exact_lp_calls > 0
        assert result.probe.completion.status == "STATE_BUDGET_EXHAUSTED"
        assert result.probe.selection == result.settled_lp.selection
        assert final.selection.branch_splits > 0
        for event in final.small_dimension_lp_evidence:
            assert_event(event, (3, 3, 1), 2)
        calls.clear()
        audit = engine(THREE, (3, 3, 1), 2)
        raw, after, _ = audit.advance_roots_and_settle(144)
        middle = audit.advance_to(raw.selection.duration_root_count + 1)
        audited = audit.advance_to(144)
        a, b = asdict(final), asdict(audited)
        for field in ("native_own_tight_work", "small_dimension_lp_work"):
            direct_copies = a[field].pop("event_snapshot_entries_copied")
            audit_copies = b[field].pop("event_snapshot_entries_copied")
            evidence_field = field.replace("work", "evidence")
            assert direct_copies == len(getattr(after, evidence_field)) + len(getattr(final, evidence_field))
            assert audit_copies == (len(getattr(after, evidence_field)) + len(getattr(middle, evidence_field))
                                    + len(getattr(audited, evidence_field)))
        assert a == b


def test_failed_generic_lp_retains_inherited_certificate_without_fabricating_exact_events(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "linprog", controlled_generic(calls, fail_second=True))
    search = native.SmallDimensionExactNativeSearch(THREE, (3, 3, 3), 2)
    root, after, phase = search.advance_roots_and_settle(144)
    assert root.small_dimension_lp_work.generic_lp_calls == 1 and phase.stopped_for_branch == 1
    generator = search._search
    final = search.advance_to(256)
    assert search._search is generator
    assert final.small_dimension_lp_work.generic_lp_calls == 2
    assert final.small_dimension_lp_work.exact_lp_calls == 0
    assert not final.small_dimension_lp_evidence
    assert final.selection.safe_fallback_lp_calls == 1 and final.selection.canonical_exact
    assert final.selection.option_counts == (2, 0, 0)
    assert final.selection.dual_certified_lp_calls == final.selection.lp_calls - 1
    assert final.dual_reuse.certificates_saved == 1
    assert final.selection.score == final.selection.upper_bound == 6.
    assert final.small_dimension_lp_work.generic_lp_calls + final.small_dimension_lp_work.exact_lp_calls == final.selection.lp_calls
    for event in final.small_dimension_lp_evidence:
        assert_event(event, (3, 3, 3), 2)


def test_empty_and_insufficient_root_budget_keep_contract_and_fee_schema():
    result = router.select_small_dimension_exact_native_frame((), (0, 0), 2, 144, 256, 8192)
    assert result.status == "LP_ROOT_COMPLETE" and result.root_lp_budget == 0
    assert result.root_lp is result.initial_lp is result.final_lp
    assert result.settled_lp is result.root_settle_work is result.probe is None
    assert result.root_lp.small_dimension_lp_work == native.SmallDimensionLPWork()
    assert not result.root_lp.small_dimension_lp_evidence
    with pytest.raises(ValueError, match="must cover 2 duration thresholds"):
        native.SmallDimensionExactNativeSearch((Option((1,), 1.), Option((3,), 2.)), (20,), 4).advance_roots_and_settle(1)
    assert native.ROOT_SETTLE_COUNTERS == parent.ROOT_SETTLE_COUNTERS
    assert native.ROOT_SETTLE_FEE_COUNTERS == parent.ROOT_SETTLE_FEE_COUNTERS
    assert native.NATIVE_OWN_TIGHT_COUNTERS == parent.NATIVE_OWN_TIGHT_COUNTERS
    assert len(native.SMALL_DIMENSION_LP_COUNTERS) == 30
    assert len(native.SMALL_DIMENSION_LP_DIAGNOSTIC_COUNTERS) == 6
    assert len(native.SMALL_DIMENSION_LP_FEE_COUNTERS) == 24
