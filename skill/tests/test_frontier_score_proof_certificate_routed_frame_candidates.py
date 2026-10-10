"""Check paid-frontier coverage, boxed integer proof, and shared completion cap."""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import product
from types import SimpleNamespace

import pytest

import algorithm.branch_reserved_integer_capacity_frame_candidates as branch
import algorithm.dual_reuse_integer_capacity_frame_candidates as dual
import algorithm.frontier_score_proof_certificate_routed_frame_candidates as routed
from algorithm.canonical_count_space_bnb_frame_candidates import _Region
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import StrictScoreProofWork


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


OPTIONS = (Option((1, 0), 2.0), Option((1, 0), 1.0), Option((0, 1), 2.0))
QUEUE, LANES = (3, 2), 3


def region(lower=(1, 0), upper=(1, 2), *, bound=(4, 1)):
    return routed.FrontierProofRegion((2, 1), bound, (0, 1), ((1, 0), (0, 1)),
                                     lower, upper, (1,), (0, 2))


def evidence(*regions, score=(3, 1)):
    groups = sum(len(item.eligible) for item in regions)
    return routed.PaidLpScoreFrontier(score, 2, len(regions), len(regions), 0, tuple(regions),
        routed.FrontierExportWork(len(regions), groups, groups,
            sum(sum(len(row) for row in item.requirements) for item in regions),
            2 * groups, sum(len(item.exact_duration_positions) for item in regions),
            groups, 1 + 2 * len(regions)))


def boxed_improvement(item, score=Fraction(3)):
    threshold = Fraction(*item.threshold_fraction)
    for counts in product(*(range(left, right + 1) for left, right in zip(item.lower, item.upper))):
        service = tuple(sum(count * required[c] for count, required in zip(counts, item.requirements))
                        for c in range(len(QUEUE)))
        attained = threshold == 1 or any(counts[pos] for pos in item.exact_duration_positions)
        if (sum(counts) <= LANES and attained and all(s <= q for s, q in zip(service, QUEUE))
                and Fraction(sum(q * s for q, s in zip(QUEUE, service))) / threshold > score):
            return True
    return False


@pytest.mark.parametrize("lower,upper", (((1, 0), (1, 0)), ((1, 0), (1, 2)),
                                         ((0, 1), (1, 1)), ((0, 1), (2, 2))))
def test_boxed_proof_matches_independent_integer_enumeration(lower, upper):
    item = region(lower, upper)
    proof = routed._prove_frontier_no_strict_improvement(OPTIONS, QUEUE, LANES, evidence(item), 128)
    expected = boxed_improvement(item)
    assert proof.status == ("BETTER_ACTION_FOUND" if expected else "COMPLETE")
    assert proof.complete == (not expected) and proof.states_visited <= 128
    if expected:
        counts = proof.witness["option_counts"]
        local = (counts[0], counts[2])
        assert counts[1] == 0
        assert all(left <= value <= right for left, value, right in zip(lower, local, upper))
        assert proof.witness["duration"] == 2.0
        assert Fraction(*proof.witness["score_fraction"]) > 3


def test_mandatory_lower_counts_and_duplicate_original_option_lifting_are_preserved():
    proof = routed._prove_frontier_no_strict_improvement(OPTIONS, QUEUE, LANES, evidence(region()), 128)
    assert proof.status == "BETTER_ACTION_FOUND"
    assert proof.witness["option_counts"] == (1, 0, 2)
    assert proof.witness["option_indices"] == (0, 2, 2)
    assert proof.witness["service"] == (1, 2)
    assert Fraction(*proof.witness["score_fraction"]) == Fraction(7, 2)


def test_duration_stratum_must_be_attained_even_if_lower_service_exceeds_target():
    item = region((3, 0), (3, 0), bound=(5, 1))
    # Group 0's minimum duration is 1; its slower lift cannot create this stratum.
    proof = routed._prove_frontier_no_strict_improvement(OPTIONS, QUEUE, LANES, evidence(item), 128)
    assert proof.status == "COMPLETE" and proof.witness is None
    assert proof.duration_prunes + proof.bound_prunes > 0


def test_idle_threshold_does_not_require_an_active_exact_duration_option():
    item = routed.FrontierProofRegion((1, 1), (10, 1), (0,), ((1, 0),), (1,), (2,), (), (1,))
    proof = routed._prove_frontier_no_strict_improvement(OPTIONS, QUEUE, LANES, evidence(item), 128)
    assert proof.status == "BETTER_ACTION_FOUND"
    assert proof.witness["option_counts"] == (0, 2, 0) and proof.witness["duration"] == 1.0


def test_region_memoization_is_not_shared_across_different_boxes():
    narrow, wide = region(upper=(1, 0)), region()
    proof = routed._prove_frontier_no_strict_improvement(OPTIONS, QUEUE, LANES,
                                                       evidence(narrow, wide), 128)
    assert proof.status == "BETTER_ACTION_FOUND" and proof.thresholds_checked == 2


@pytest.mark.parametrize("budget", (0, 1))
def test_global_state_cap_includes_root_bound_closure_across_boxes(budget):
    narrow = region(upper=(1, 1), bound=(61, 20))
    # floor(3.05*2)=6 is below the strict numerator target 7.
    proof = routed._prove_frontier_no_strict_improvement(OPTIONS, QUEUE, LANES,
                                                       evidence(narrow, narrow), budget)
    assert proof.status == "STATE_BUDGET_EXHAUSTED" and not proof.complete
    assert proof.states_visited == budget and proof.bound_prunes == budget
    assert proof.witness is None


def test_exact_region_bound_can_close_a_box_without_enumerating_children():
    narrow = region(upper=(1, 1), bound=(61, 20))
    proof = routed._prove_frontier_no_strict_improvement(OPTIONS, QUEUE, LANES, evidence(narrow), 1)
    assert proof.status == "COMPLETE" and proof.states_visited == 1
    assert proof.bound_prunes == 1 and proof.count_branches == 0


@pytest.mark.parametrize("engine,search_name", ((branch, "BranchReservedIntegerCapacitySearch"),
                                               (dual, "DualReuseIntegerCapacitySearch")))
def test_capture_includes_failed_and_unsolved_paid_regions_without_extra_native_work(
        monkeypatch, engine, search_name):
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(success=False)

    monkeypatch.setattr(engine, "linprog", failed)
    options = (Option((3, 2), 1.0), Option((2, 0), 1.0))
    search = getattr(engine, search_name)(options, (4, 3), 3)
    paid = search.advance_to(2)
    raw = search._search.gi_frame.f_locals["frontier"]
    assert any(not entry[-1].solved for entry in raw)
    assert any(entry[-1].solved for entry in raw)
    before = asdict(paid)
    captured = routed.capture_strict_frontier(search)
    assert len(calls) == paid.selection.lp_calls == 2
    assert search._snapshot is paid and asdict(paid) == before
    assert captured.incumbent_score_fraction == (18, 1)
    assert captured.root_count == 1 and captured.raw_region_count == len(raw)
    assert captured.live_region_count == paid.selection.frontier_regions_remaining
    assert len(captured.regions) == len(raw)
    assert tuple(item.lower for item in captured.regions) == tuple(entry[-1].lower for entry in raw)
    work = captured.export_work
    assert work.heap_entries_read == len(raw)
    assert work.group_ids_copied == work.requirement_rows_copied == work.representative_options_copied == 2 * len(raw)
    assert work.requirement_coordinates_copied == work.box_count_values_copied == 4 * len(raw)
    assert work.fraction_pairs_copied == 1 + 2 * len(raw)
    proof = routed._prove_frontier_no_strict_improvement(options, (4, 3), 3, captured, 128)
    assert proof.complete and proof.witness is None
    # Independent original-option search confirms that the paid incumbent is optimal.
    feasible_scores = []
    for counts in product(range(4), repeat=2):
        service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options))
                        for c in range(2))
        if sum(counts) <= 3 and all(s <= q for s, q in zip(service, (4, 3))):
            feasible_scores.append(4 * service[0] + 3 * service[1])
    assert max(feasible_scores) == 18


def test_capture_filters_only_exact_strict_bounds_and_keeps_raw_heap_order():
    boxes = tuple(_Region(0, Fraction(1), (0,), (0,), (0,), (2,), Fraction(value),
                          solved, None) for value, solved in ((5, False), (3, True), (4, True), (2, False)))
    heap = [(None, box) for box in boxes]
    native = SimpleNamespace(gi_frame=SimpleNamespace(f_locals={
        "best": SimpleNamespace(score=Fraction(3)), "frontier": heap,
        "requirements": ((1,),), "representatives": ({0: 0},), "root_count": 1}))
    search = SimpleNamespace(_search=native, _snapshot=SimpleNamespace(selection=SimpleNamespace(
        frontier_regions_remaining=3, canonical_tie_regions_remaining=1)))
    captured = routed.capture_strict_frontier(search)
    assert tuple(item.bound_fraction for item in captured.regions) == ((5, 1), (4, 1))
    assert captured.raw_region_count == 4 and captured.live_region_count == 3
    assert captured.live_tie_region_count == 1
    assert captured.export_work.heap_entries_read == 4
    assert captured.export_work.group_ids_copied == 2


@dataclass(frozen=True)
class Selection:
    lp_node_budget: int
    lp_calls: int
    score_exact: bool
    canonical_exact: bool
    score: float = 3.0
    upper_bound: float = 4.0
    gap: float = 1.0
    epsilon: float = 0.2


@dataclass(frozen=True)
class Stage:
    selection: Selection
    preprocessing: tuple[str, ...] = ("one_catalogue",)
    capacity_propagation: tuple[str, ...] = ("cumulative",)


@pytest.fixture(params=(("select_branch_reserved_frontier_score_proof_frame", "BranchReservedIntegerCapacitySearch"),
                        ("select_dual_reuse_frontier_score_proof_frame", "DualReuseIntegerCapacitySearch")))
def arm(request):
    return request.param


def install_search(monkeypatch, arm, initial, final=None):
    constructions, caps = [], []

    class Search:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

        def advance_to(self, cap):
            caps.append(cap)
            assert cap in (144, 256) and len(caps) <= 2
            return initial if cap == 144 else final

    monkeypatch.setattr(routed, arm[1], Search)
    return constructions, caps


def select(arm):
    return getattr(routed, arm[0])(OPTIONS, QUEUE, LANES, 144, 256, 8192)


def test_ordinary_initial_closure_keeps_no_frontier_evidence(monkeypatch, arm):
    initial = Stage(Selection(144, 142, True, True))
    constructions, caps = install_search(monkeypatch, arm, initial)

    def unexpected(*args, **kwargs):
        pytest.fail("closed initial route must not capture or probe")

    monkeypatch.setattr(routed, "capture_strict_frontier", unexpected)
    monkeypatch.setattr(routed, "_probe", unexpected)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144]
    assert result.frontier_evidence is None and result.final_lp is initial
    assert result.status == "LP_INITIAL_COMPLETE" and not result.resumed


def test_exhausted_ordinary_probe_never_enables_a_frontier_retry(monkeypatch, arm):
    initial, final = Stage(Selection(144, 142, True, False)), Stage(Selection(256, 254, True, False))
    constructions, caps = install_search(monkeypatch, arm, initial, final)
    probes = []

    def ordinary_probe(options, queue, lanes, source, budget, idle):
        probes.append((source, budget))
        return SimpleNamespace(selection=source.selection)

    def unexpected(*args, **kwargs):
        pytest.fail("an earlier probe must not trigger another frontier probe")

    monkeypatch.setattr(routed, "_probe", ordinary_probe)
    monkeypatch.setattr(routed, "capture_strict_frontier", unexpected)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144, 256]
    assert probes == [(initial, 8192)] and result.frontier_evidence is None
    assert result.probe_stage == "initial" and result.status == "LP_RESUME_INCOMPLETE"


@pytest.mark.parametrize("proof_complete", (False, True))
def test_frontier_probe_keeps_raw_lp_and_shares_completion_budget(monkeypatch, arm, proof_complete):
    initial, final = Stage(Selection(144, 142, False, False)), Stage(Selection(256, 254, False, False))
    constructions, caps = install_search(monkeypatch, arm, initial, final)
    captured, proof_calls, completions = evidence(region()), [], []
    visited = 23 if proof_complete else 8192
    monkeypatch.setattr(routed, "capture_strict_frontier", lambda search: captured)

    def prove(options, queue, lanes, source, budget, idle):
        proof_calls.append((source, budget))
        return StrictScoreProofWork("COMPLETE" if proof_complete else "STATE_BUDGET_EXHAUSTED",
                                    True, proof_complete, budget, (3, 1), states_visited=visited)

    def complete(options, queue, lanes, source, budget, idle):
        assert proof_complete and source.score_exact and not source.canonical_exact
        completions.append((source, budget))
        return replace(source, canonical_exact=True), ScoreFaceCompletionWork(
            "COMPLETE", True, True, budget, states_visited=2)

    monkeypatch.setattr(routed, "_prove_frontier_no_strict_improvement", prove)
    monkeypatch.setattr(routed, "_complete_canonical", complete)
    before = asdict(final)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144, 256]
    assert proof_calls == [(captured, 8192)] and len(completions) == int(proof_complete)
    assert result.frontier_evidence is captured and result.final_lp is final
    assert not final.selection.score_exact and asdict(final) == before
    assert result.probe.baseline_selection is final.selection
    assert result.probe.preprocessing is final.preprocessing
    assert result.probe.completion.state_budget == 8192 - visited
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= 8192
    assert result.probe_stage == "frontier_late" and result.resumed
    assert result.status == ("FRONTIER_PROBE_COMPLETE" if proof_complete else "FRONTIER_PROBE_INCOMPLETE")
