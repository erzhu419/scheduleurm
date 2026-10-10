"""Independent integer-capacity bounds and one shared frontier-proof budget."""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import product

import pytest

import algorithm.axis_capacity_frontier_score_proof_frame_candidates as axis
import algorithm.frontier_score_proof_certificate_routed_frame_candidates as previous
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def axis_options(dimension, batches):
    return tuple(Option(tuple(batch if position == coordinate else 0 for position in range(dimension)), 1.0)
                 for coordinate in range(dimension) for batch in batches)


def original_lane_upper(requirements, queue, lanes):
    """Enumerate original per-slot choices, including idle, without a capacity DP."""
    best = 0
    for choices in product(range(-1, len(requirements)), repeat=lanes):
        service = tuple(sum(requirements[index][coordinate] for index in choices if index >= 0)
                        for coordinate in range(len(queue)))
        if all(done <= cap for done, cap in zip(service, queue)):
            best = max(best, sum(weight * done for weight, done in zip(queue, service)))
    return best


def exact_slot_profiles(requirements, queue, lanes):
    profiles = []
    for coordinate, cap in enumerate(queue):
        amounts = tuple(required[coordinate] for required in requirements if required[coordinate])
        profiles.append(tuple(max((sum(choices) for choices in product(amounts, repeat=slots)
                                   if sum(choices) <= cap), default=-1) for slots in range(lanes + 1)))
    return tuple(profiles)


@pytest.mark.parametrize("queue,lanes,batches,expected", (
    ((10,), 2, (3, 8), 80), ((10,), 2, (1, 3, 8), 90),
    ((4, 4), 1, (1, 3, 8), 12), ((0, 10, 4), 2, (1, 3, 8), 92)))
def test_global_bound_and_exact_slot_profiles_match_independent_original_enumeration(queue, lanes, batches, expected):
    requirements = tuple(option.requirements for option in axis_options(len(queue), batches))
    bound, work, profiles = axis._axis_capacity_upper_bound(requirements, queue, lanes, 8192)
    assert bound == expected == original_lane_upper(requirements, queue, lanes)
    assert profiles == exact_slot_profiles(requirements, queue, lanes)
    assert work.status == "COMPLETE" and work.attempted
    assert work.states_visited == work.axis_states + work.merge_states <= 8192
    assert work.axis_states > 0 and work.merge_states > 0
    assert work.bitset_shifts > 0 and work.bitset_ors > 0 and work.bitset_masks > 0
    assert work.service_reads > 0 and work.merge_candidates > 0 and work.requirement_coordinate_reads > 0
    assert work.axis_states == len(queue) + len(requirements) * (lanes + 1)
    assert work.merge_states == 1 + len(queue) * (lanes + 1)
    assert work.bitset_ors == work.bitset_masks == len(requirements) * (lanes + 1)
    assert work.bitset_shifts == len(queue) + len(requirements) * lanes
    assert work.service_reads == len(queue) * (lanes + 1)
    assert work.merge_candidates == len(queue) * (lanes + 1) * (lanes + 2) // 2
    assert work.requirement_coordinate_reads == len(queue) * len(requirements)
    assert work.bounds == ()


@pytest.mark.parametrize("budget", (0, 1, "last_merge_cell"))
def test_unfinished_capacity_table_returns_no_bound_and_spends_only_its_available_cells(budget):
    requirements = ((1, 0), (3, 0), (8, 0), (0, 1), (0, 3), (0, 8))
    if budget == "last_merge_cell":
        _, complete, _ = axis._axis_capacity_upper_bound(requirements, (10, 4), 2, 8192)
        budget = complete.states_visited - 1
    bound, work, _ = axis._axis_capacity_upper_bound(requirements, (10, 4), 2, budget)
    assert bound is None and work.status == "STATE_BUDGET_EXHAUSTED" and work.attempted
    assert work.states_visited == work.axis_states + work.merge_states == budget
    assert work.bounds == ()


def proof_region(options, queue, lanes, *, threshold=Fraction(1), lower=None, upper=None):
    requirements = tuple(option.requirements for option in options)
    default_upper = tuple(min(lanes, *(cap // amount for cap, amount in zip(queue, required) if amount))
                          for required in requirements)
    return previous.FrontierProofRegion((threshold.numerator, threshold.denominator),
        (sum(cap * cap for cap in queue) + 1, 1), tuple(range(len(options))), requirements,
        (0,) * len(options) if lower is None else lower, default_upper if upper is None else upper,
        tuple(index for index, option in enumerate(options) if Fraction.from_float(max(1.0, option.duration)) == threshold),
        tuple(range(len(options))))


def frontier(regions, incumbent):
    groups = sum(len(region.eligible) for region in regions)
    return previous.PaidLpScoreFrontier((incumbent.numerator, incumbent.denominator),
        len({region.threshold_fraction for region in regions}),
        len(regions), len(regions), 0, tuple(regions), previous.FrontierExportWork(
            len(regions), groups, groups, sum(sum(len(row) for row in region.requirements) for region in regions),
            2 * groups, sum(len(region.exact_duration_positions) for region in regions), groups, 1 + 2 * len(regions)))


def split_axis_boxes(options, queue, lanes):
    whole = proof_region(options, queue, lanes)
    # Split on the first axis's batch8 count; boxes are disjoint and exhaustive.
    upper_without_eight = list(whole.upper)
    upper_without_eight[2] = 0
    lower_with_eight = list(whole.lower)
    lower_with_eight[2] = 1
    return (replace(whole, upper=tuple(upper_without_eight)), replace(whole, lower=tuple(lower_with_eight)))


def boxed_scores(options, queue, lanes, region):
    scores = []
    threshold = Fraction(*region.threshold_fraction)
    for counts in product(range(lanes + 1), repeat=len(options)):
        local_counts = tuple(counts[index] for index in region.representative_options)
        if any(count for index, count in enumerate(counts) if index not in region.representative_options):
            continue
        if sum(counts) > lanes or any(not left <= count <= right
                for left, count, right in zip(region.lower, local_counts, region.upper)):
            continue
        service = tuple(sum(count * option.requirements[coordinate] for count, option in zip(counts, options))
                        for coordinate in range(len(queue)))
        if any(done > cap for done, cap in zip(service, queue)):
            continue
        duration = max((Fraction(1), *(Fraction.from_float(option.duration)
                         for count, option in zip(counts, options) if count)))
        if duration == threshold:
            scores.append(Fraction(sum(weight * done for weight, done in zip(queue, service))) / duration)
    return scores


def test_same_threshold_table_is_charged_once_and_closes_both_disjoint_boxes():
    options, queue, lanes = axis_options(2, (1, 3, 8)), (10, 4), 2
    regions = split_axis_boxes(options, queue, lanes)
    assert max(score for region in regions for score in boxed_scores(options, queue, lanes, region)) == 92
    bound, table, profiles = axis._axis_capacity_upper_bound(regions[0].requirements, queue, lanes, 8192)
    proof = axis._prove_axis_capacity_frontier(options, queue, lanes, frontier(regions, Fraction(92)), 8192)
    assert proof.complete and proof.status == "COMPLETE" and proof.witness is None
    assert proof.states_visited == proof.capacity.states_visited == table.states_visited
    assert proof.dfs_states_visited == 0 and proof.thresholds_checked == 2
    assert proof.capacity.threshold_tables == 1 and proof.capacity.cache_hits == 1
    assert proof.capacity.bound_checks == proof.capacity.boxes_pruned == 2
    assert len(proof.capacity.bounds) == 1
    saved = proof.capacity.bounds[0]
    assert (saved.upper_numerator, saved.target_numerator, saved.box_indices) == (bound, 93, (0, 1))
    assert saved.upper_numerator == original_lane_upper(regions[0].requirements, queue, lanes)
    assert saved.axis_service_profiles == profiles and saved.requirements == regions[0].requirements
    assert saved.eligible == regions[0].eligible and saved.threshold_fraction == (1, 1)


def test_relaxed_box_bound_never_replaces_boxed_witness_search_or_the_remaining_dfs_cap():
    options, queue, lanes = axis_options(2, (1, 3, 8)), (10, 4), 2
    regions = split_axis_boxes(options, queue, lanes)
    assert max(boxed_scores(options, queue, lanes, regions[0])) < 90
    assert max(boxed_scores(options, queue, lanes, regions[1])) == 92
    _, table, _ = axis._axis_capacity_upper_bound(regions[0].requirements, queue, lanes, 8192)
    source = frontier(regions, Fraction(90))
    partial = axis._prove_axis_capacity_frontier(options, queue, lanes, source, table.states_visited + 1)
    assert partial.status == "STATE_BUDGET_EXHAUSTED" and not partial.complete and partial.witness is None
    assert partial.capacity.states_visited == table.states_visited and partial.dfs_states_visited == 1
    assert partial.states_visited == table.states_visited + 1 and partial.capacity.boxes_pruned == 0
    proof = axis._prove_axis_capacity_frontier(options, queue, lanes, source, 8192)
    assert proof.status == "BETTER_ACTION_FOUND" and not proof.complete
    assert proof.thresholds_checked == 2 and proof.capacity.boxes_pruned == 0
    assert proof.states_visited == proof.capacity.states_visited + proof.dfs_states_visited
    counts = proof.witness["option_counts"]
    assert all(left <= count <= right for left, count, right in zip(regions[1].lower, counts, regions[1].upper))
    assert counts[2] == counts[4] == 1 and sum(counts) == lanes
    assert Fraction(*proof.witness["score_fraction"]) == 92


def test_ignoring_duration_only_gives_an_upper_bound_not_a_strict_witness():
    options = (Option((1,), 1.0), Option((3,), 1.0), Option((2,), 2.0))
    queue, lanes = (4,), 1
    root = proof_region(options, queue, lanes, threshold=Fraction(2))
    slow_absent = replace(root, upper=(1, 1, 0))
    slow_present = replace(root, lower=(0, 0, 1))
    fast = proof_region(options[:2], queue, lanes)
    assert not boxed_scores(options, queue, lanes, slow_absent)
    assert max(boxed_scores(options, queue, lanes, slow_present)) == 4
    proof = axis._prove_axis_capacity_frontier(options, queue, lanes,
        frontier((slow_absent, slow_present, fast), Fraction(4)), 8192)
    assert proof.capacity.bounds[0].upper_numerator == 12
    assert proof.capacity.bounds[0].target_numerator == 9 and proof.capacity.boxes_pruned == 0
    assert proof.thresholds_checked == 3 and proof.status == "BETTER_ACTION_FOUND"
    assert proof.witness["duration"] == 1.0 and proof.witness["option_counts"] == (0, 1, 0)
    assert Fraction(*proof.witness["score_fraction"]) == 12


@pytest.mark.parametrize("case", ("one_box", "mixed_requirements"))
def test_nonattempted_capacity_preserves_every_original_dfs_result_and_accounts_only_real_scans(case):
    if case == "one_box":
        options, queue, lanes = axis_options(2, (1, 3, 8)), (10, 4), 2
        regions, incumbent = split_axis_boxes(options, queue, lanes)[:1], Fraction(92)
    else:
        options, queue, lanes = (Option((2, 1), 1.0), Option((1, 0), 1.0)), (4, 3), 2
        root = proof_region(options, queue, lanes)
        regions = (replace(root, upper=(1, 2)), replace(root, lower=(2, 0)))
        incumbent = Fraction(22)
    source = frontier(regions, incumbent)
    original = previous._prove_frontier_no_strict_improvement(options, queue, lanes, source, 128)
    proof = axis._prove_axis_capacity_frontier(options, queue, lanes, source, 128)
    original_fields = asdict(original)
    proof_fields = asdict(proof)
    assert {key: proof_fields[key] for key in original_fields} == original_fields
    assert proof.dfs_states_visited == original.states_visited
    capacity = proof.capacity
    assert not capacity.attempted and capacity.status == "NOT_ATTEMPTED" and capacity.bounds == ()
    assert capacity.states_visited == capacity.axis_states == capacity.merge_states == 0
    assert capacity.bitset_shifts == capacity.bitset_ors == capacity.bitset_masks == capacity.merge_candidates == 0
    assert (capacity.requirement_coordinate_reads == 0) if case == "one_box" else (capacity.requirement_coordinate_reads > 0)


@dataclass(frozen=True)
class NativeSelection:
    lp_node_budget: int
    lp_calls: int
    score_exact: bool
    canonical_exact: bool
    score: float = 92.0
    upper_bound: float = 116.0
    gap: float = 24.0
    epsilon: float = 0.2


@dataclass(frozen=True)
class NativeStage:
    selection: NativeSelection
    preprocessing: tuple[str, ...] = ("single preprocessing",)
    capacity_propagation: tuple[str, ...] = ("cumulative LP",)


def native_stage(cap, score_exact=False, canonical_exact=False):
    selected = NativeSelection(cap, cap - 2, score_exact, canonical_exact)
    if score_exact:
        selected = replace(selected, upper_bound=selected.score, gap=0.0, epsilon=0.0)
    return NativeStage(selected)


def test_completed_capacity_proof_shares_remaining_budget_with_canonical_and_exhaustion_skips_it(monkeypatch):
    options, queue, lanes = axis_options(2, (1, 3, 8)), (10, 4), 2
    source = frontier(split_axis_boxes(options, queue, lanes), Fraction(92))
    stage, calls = native_stage(256), []
    before = asdict(stage)

    def complete(options_arg, queue_arg, lanes_arg, certified, budget, idle):
        assert (options_arg, queue_arg, lanes_arg, idle) == (options, queue, lanes, 1.0)
        assert certified.score_exact and not certified.canonical_exact
        calls.append(budget)
        return replace(certified, canonical_exact=True), ScoreFaceCompletionWork(
            "COMPLETE", True, True, budget, states_visited=2)

    monkeypatch.setattr(axis, "_complete_canonical", complete)
    result = axis._axis_frontier_probe(options, queue, lanes, stage, 8192, 1.0, source)
    proof = result.score_proof
    assert proof.complete and proof.states_visited == proof.capacity.states_visited and proof.dfs_states_visited == 0
    assert calls == [8192 - proof.states_visited]
    assert proof.states_visited + result.completion.states_visited <= 8192
    assert result.completion.state_budget == 8192 - proof.states_visited
    assert result.baseline_selection is stage.selection and result.selection.canonical_exact
    assert result.score_certified_selection.score_exact and not result.score_certified_selection.canonical_exact
    assert asdict(stage) == before and not stage.selection.score_exact
    low_budget = proof.states_visited - 1
    exhausted = axis._axis_frontier_probe(options, queue, lanes, stage, low_budget, 1.0, source)
    assert not exhausted.score_proof.complete and exhausted.score_proof.status == "STATE_BUDGET_EXHAUSTED"
    assert exhausted.score_proof.states_visited == exhausted.score_proof.capacity.states_visited == low_budget
    assert exhausted.selection is exhausted.score_certified_selection is stage.selection
    assert exhausted.completion.status == "SKIPPED_SCORE_UNPROVEN" and exhausted.completion.state_budget == 0
    assert calls == [8192 - proof.states_visited]


@dataclass(frozen=True)
class OrdinaryProbe:
    selection: NativeSelection


@pytest.mark.parametrize("old_name,new_name,search_name", (
    ("select_branch_reserved_frontier_score_proof_frame", "select_branch_reserved_axis_capacity_frontier_frame",
     "BranchReservedIntegerCapacitySearch"),
    ("select_dual_reuse_frontier_score_proof_frame", "select_dual_reuse_axis_capacity_frontier_frame",
     "DualReuseIntegerCapacitySearch")))
@pytest.mark.parametrize("route", ("initial_complete", "exhausted_initial_probe", "resumed_ordinary_probe"))
def test_ordinary_routes_match_the_frozen_router_without_axis_work(monkeypatch, old_name, new_name, search_name, route):
    options, queue, lanes = axis_options(2, (1, 3, 8)), (10, 4), 2
    initial = native_stage(144, route != "resumed_ordinary_probe", route == "initial_complete")
    final = native_stage(256, True)
    logs, results = {}, {}

    def unexpected(*args, **kwargs):
        pytest.fail("an ordinary route must not capture or run a frontier capacity proof")

    for label, module, selector_name in (("old", previous, old_name), ("new", axis, new_name)):
        log = {"constructions": 0, "caps": [], "probes": []}
        logs[label] = log

        def search_class(log):
            class Search:
                def __init__(self, options_arg, queue_arg, lanes_arg, *, idle_duration):
                    assert (options_arg, queue_arg, lanes_arg, idle_duration) == (options, queue, lanes, 1.0)
                    log["constructions"] += 1

                def advance_to(self, cap):
                    log["caps"].append(cap)
                    return initial if cap == 144 else final
            return Search

        def ordinary_probe(log):
            def run(options_arg, queue_arg, lanes_arg, source, budget, idle):
                assert (options_arg, queue_arg, lanes_arg, budget, idle) == (options, queue, lanes, 8192, 1.0)
                log["probes"].append((source.selection.lp_node_budget, budget))
                return OrdinaryProbe(replace(source.selection, canonical_exact=route == "resumed_ordinary_probe"))
            return run

        monkeypatch.setattr(module, search_name, search_class(log))
        monkeypatch.setattr(module, "_probe", ordinary_probe(log))
        monkeypatch.setattr(module, "capture_strict_frontier", unexpected)
        results[label] = getattr(module, selector_name)(options, queue, lanes, 144, 256, 8192)
    assert results["new"] == results["old"] and logs["new"] == logs["old"]
    assert logs["new"]["constructions"] == 1 and results["new"].frontier_evidence is None
    if route == "initial_complete":
        assert logs["new"]["caps"] == [144] and not logs["new"]["probes"]
        assert results["new"].status == "LP_INITIAL_COMPLETE"
    else:
        assert logs["new"]["caps"] == [144, 256]
        assert logs["new"]["probes"] == [(144 if route == "exhausted_initial_probe" else 256, 8192)]
        assert results["new"].status == ("LP_RESUME_INCOMPLETE" if route == "exhausted_initial_probe" else "INTEGER_PROBE_COMPLETE")
