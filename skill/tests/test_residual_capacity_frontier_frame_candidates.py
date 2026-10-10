"""Canonical capacity reuse checked against independent original actions."""
from dataclasses import asdict, dataclass, fields, replace
from fractions import Fraction
from itertools import product

import pytest

import algorithm.axis_capacity_frontier_score_proof_frame_candidates as previous
import algorithm.residual_capacity_frontier_frame_candidates as residual
from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.frontier_score_proof_certificate_routed_frame_candidates import (
    FrontierExportWork, FrontierProofRegion, PaidLpScoreFrontier)
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def original_actions(options, queue, lanes):
    actions = []
    for choices in product(range(-1, len(options)), repeat=lanes):
        counts = tuple(choices.count(index) for index in range(len(options)))
        service = tuple(sum(count * option.requirements[coordinate] for count, option in zip(counts, options))
                        for coordinate in range(len(queue)))
        if any(done > cap for done, cap in zip(service, queue)):
            continue
        duration = max((Fraction(1), *(Fraction.from_float(max(1.0, option.duration))
                         for count, option in zip(counts, options) if count)))
        indices = tuple(index for index, count in enumerate(counts) for _ in range(count)) + (None,) * (lanes - sum(counts))
        actions.append({"counts": counts, "service": service, "indices": indices, "duration": duration,
                       "score": Fraction(sum(weight * done for weight, done in zip(queue, service))) / duration})
    return actions


def action_key(action, option_count):
    return action["duration"], tuple(option_count if index is None else index for index in action["indices"])


def certified_baseline(options, queue, lanes):
    actions = original_actions(options, queue, lanes)
    score = max(action["score"] for action in actions)
    optimal = [action for action in actions if action["score"] == score]
    canonical = min(optimal, key=lambda action: action_key(action, len(options)))
    old = max(optimal, key=lambda action: action_key(action, len(options)))
    values = {field.name: 0 for field in fields(CanonicalCountSpaceBnbSelection)}
    values.update(option_counts=old["counts"], option_indices=old["indices"], service=old["service"],
        duration=float(old["duration"]), score=float(score), upper_bound=float(score), gap=0.0, epsilon=0.0,
        exact=False, score_exact=True, canonical_exact=False, lp_node_budget=256, lp_calls=3,
        frontier_regions_remaining=1, canonical_tie_regions_remaining=1, inherited_unsolved_regions=1)
    return CanonicalCountSpaceBnbSelection(**values), canonical


def paid_bounds(options, queue, lanes, baseline):
    score = Fraction(sum(weight * done for weight, done in zip(queue, baseline.service))) / Fraction.from_float(baseline.duration)
    bounds, costs = [], []
    for threshold in sorted({Fraction.from_float(max(1.0, option.duration)) for option in options}):
        requirements = tuple(dict.fromkeys(option.requirements for option in options
                            if Fraction.from_float(max(1.0, option.duration)) <= threshold))
        upper, work, profiles, merged = residual._residual_axis_capacity_upper_bound(requirements, queue, lanes, 8192)
        assert work.status == "COMPLETE" and upper is not None
        assert work.merged_values_copied == len(merged) == lanes + 1
        target = score * threshold
        bounds.append(residual.ResidualAxisCapacityBound(
            (threshold.numerator, threshold.denominator), tuple(range(len(requirements))), requirements, upper,
            target.numerator // target.denominator + 1, (0, 1), profiles, merged))
        costs.append(work)
    return tuple(bounds), tuple(costs)


def case_fixture(case):
    if case == "aliases":
        return (Option((1,), 1.0), Option((3,), 1.0), Option((8,), 1.0), Option((8,), 1.0)), (8,), 2
    if case == "shorter_duration":
        return (Option((2,), 2.0), Option((1,), 1.0)), (4,), 2
    if case == "exact_slot_holes":
        return (Option((1, 0), 1.0), Option((0, 8), 1.0)), (0, 8), 3
    if case == "relaxed_suffix":
        return (Option((0, 1), 1.0), Option((1, 0), 1.0), Option((3, 0), 1.0), Option((8, 0), 1.0)), (10, 1), 3
    return (Option((1,), 1.0), Option((8,), 2.0)), (8,), 2


@pytest.mark.parametrize("case", ("aliases", "shorter_duration", "exact_slot_holes", "relaxed_suffix", "threshold_coverage"))
def test_paid_table_reuse_matches_independent_canonical_duration_and_original_indices(monkeypatch, case):
    options, queue, lanes = case_fixture(case)
    baseline, canonical = certified_baseline(options, queue, lanes)
    bounds, _ = paid_bounds(options, queue, lanes, baseline)
    before = asdict(baseline)

    def rebuild(*args, **kwargs):
        pytest.fail("canonical completion must use the already paid merged vector without another DP")

    monkeypatch.setattr(residual, "_residual_axis_capacity_upper_bound", rebuild)
    selected, work = residual._complete_residual_canonical(options, queue, lanes, baseline, 8192, 1.0, bounds)
    assert selected.canonical_exact and work.complete and work.status == "COMPLETE"
    assert (selected.option_counts, selected.option_indices, selected.service, selected.duration, selected.score) == (
        canonical["counts"], canonical["indices"], canonical["service"], float(canonical["duration"]), float(canonical["score"]))
    assert work.states_visited == work.prefix_states + work.dfs_states_visited <= 8192
    assert work.prefix_states == work.prefix_reads == work.tables_reused * (lanes + 1)
    assert work.capacity_queries > 0 and work.coverage_coordinate_reads > 0 and asdict(baseline) == before
    assert selected.lp_calls == baseline.lp_calls
    if case == "aliases":
        assert baseline.option_counts == (0, 0, 0, 1) and selected.option_counts == (0, 0, 1, 0)
        assert bounds[0].exact_slot_numerators == (0, 64, 48)
        assert work.capacity_prunes == 0
    elif case == "shorter_duration":
        assert baseline.duration == 2.0 and selected.duration == 1.0
        assert work.tables_reused == 1 and work.prefix_states == lanes + 1
    elif case == "exact_slot_holes":
        assert bounds[0].exact_slot_numerators == (0, 64, -1, -1)
        assert selected.option_indices == (1, None, None) and work.capacity_prunes == 0
    elif case == "relaxed_suffix":
        assert work.capacity_prunes > 0
        assert selected.option_indices == (1, 1, 3)
    else:
        assert work.tables_reused == 2 and selected.duration == 2.0


@pytest.mark.parametrize("coverage", ("missing_requirement", "different_threshold"))
def test_inapplicable_paid_tables_preserve_the_frozen_search_and_charge_actual_coverage_reads(coverage):
    options = (Option((1,), 1.0), Option((3,), 1.0), Option((8,), 1.0))
    queue, lanes = (8,), 2
    baseline, _ = certified_baseline(options, queue, lanes)
    if coverage == "missing_requirement":
        bounds, _ = paid_bounds(options[:2], queue, lanes, baseline)
    else:
        bounds, _ = paid_bounds(options, queue, lanes, baseline)
        bounds = (replace(bounds[0], threshold_fraction=(2, 1), target_numerator=129),)
    old_selected, old_work = previous._complete_canonical(options, queue, lanes, baseline, 8192, 1.0)
    selected, work = residual._complete_residual_canonical(options, queue, lanes, baseline, 8192, 1.0, bounds)
    assert selected == old_selected
    original = asdict(old_work)
    assert {key: asdict(work)[key] for key in original} == original
    if coverage == "missing_requirement":
        assert work.coverage_coordinate_reads > 0
        assert work.prefix_states == work.prefix_reads == work.tables_reused == work.capacity_queries == 0
        assert work.states_visited == work.dfs_states_visited
    else:
        assert type(work) is ScoreFaceCompletionWork and work == old_work


@pytest.mark.parametrize("budget", (0, 1, 3, 4))
def test_prefix_and_canonical_dfs_share_the_available_budget_without_false_closure(monkeypatch, budget):
    options = (Option((1,), 1.0), Option((3,), 1.0), Option((8,), 1.0))
    queue, lanes = (8,), 2
    baseline, _ = certified_baseline(options, queue, lanes)
    bounds, _ = paid_bounds(options, queue, lanes, baseline)

    def rebuild(*args, **kwargs):
        pytest.fail("an interrupted prefix cannot rebuild a paid DP")

    monkeypatch.setattr(residual, "_residual_axis_capacity_upper_bound", rebuild)
    selected, work = residual._complete_residual_canonical(options, queue, lanes, baseline, budget, 1.0, bounds)
    assert selected is baseline and selected.score_exact and not selected.canonical_exact
    assert work.status == "STATE_BUDGET_EXHAUSTED" and not work.complete
    assert work.states_visited == work.prefix_states + work.dfs_states_visited == budget
    assert work.prefix_states == work.prefix_reads == min(3, budget)
    assert work.dfs_states_visited == max(0, budget - 3) and work.tables_reused == 1


@pytest.mark.parametrize("budget", (0, 128))
def test_absent_paid_table_preserves_the_entire_original_completion_result(budget):
    options, queue, lanes = case_fixture("aliases")
    baseline, _ = certified_baseline(options, queue, lanes)
    old = previous._complete_canonical(options, queue, lanes, baseline, budget, 1.0)
    new = residual._complete_residual_canonical(options, queue, lanes, baseline, budget, 1.0, ())
    assert new == old and type(new[1]) is ScoreFaceCompletionWork


def test_partial_paid_merge_retains_no_vector_or_copy_charge():
    requirements, queue, lanes = ((1,), (3,), (8,)), (8,), 2
    _, completed, _, _ = residual._residual_axis_capacity_upper_bound(requirements, queue, lanes, 8192)
    upper, work, _, merged = residual._residual_axis_capacity_upper_bound(requirements, queue, lanes, completed.states_visited - 1)
    assert upper is None and merged == () and work.merged_values_copied == 0
    assert work.status == "STATE_BUDGET_EXHAUSTED" and work.states_visited == completed.states_visited - 1


@dataclass(frozen=True)
class Stage:
    selection: CanonicalCountSpaceBnbSelection
    preprocessing: tuple[str, ...] = ("one preprocessing",)
    capacity_propagation: tuple[str, ...] = ("paid LP",)


def split_frontier(options, queue, lanes, score):
    requirements = tuple(option.requirements for option in options)
    upper = tuple(min(lanes, *(cap // amount for cap, amount in zip(queue, required) if amount))
                  for required in requirements)
    root = FrontierProofRegion((1, 1), (sum(cap * cap for cap in queue) + 1, 1),
        tuple(range(len(options))), requirements, (0,) * len(options), upper,
        tuple(range(len(options))), tuple(range(len(options))))
    left_upper, right_lower = list(upper), [0] * len(options)
    left_upper[2], right_lower[2] = 0, 1
    boxes = (replace(root, upper=tuple(left_upper)), replace(root, lower=tuple(right_lower)))
    groups, coordinates = 2 * len(options), 2 * len(options) * len(queue)
    return PaidLpScoreFrontier((score.numerator, score.denominator), 1, 2, 2, 0, boxes,
        FrontierExportWork(2, groups, groups, coordinates, 2 * groups, groups, groups, 5))


def test_strict_proof_prefix_and_canonical_share_one_budget_and_keep_the_paid_stage(monkeypatch):
    options = tuple(Option(tuple(amount if axis == coordinate else 0 for axis in range(2)), 1.0)
                    for coordinate in range(2) for amount in (1, 3, 8))
    queue, lanes = (10, 4), 2
    certified, canonical = certified_baseline(options, queue, lanes)
    stage = Stage(replace(certified, score_exact=False, upper_bound=116.0, gap=24.0, epsilon=0.2))
    before = asdict(stage)
    source = split_frontier(options, queue, lanes, canonical["score"])
    helper, calls = residual._residual_axis_capacity_upper_bound, []

    def count_paid_tables(*args):
        calls.append(args[-1])
        return helper(*args)

    monkeypatch.setattr(residual, "_residual_axis_capacity_upper_bound", count_paid_tables)
    proof = residual._prove_residual_capacity_frontier(options, queue, lanes, source, 8192)
    assert proof.complete and len(calls) == 1
    assert proof.capacity.merged_values_copied == lanes + 1 and proof.capacity.threshold_tables == 1
    assert proof.capacity.bounds[0].exact_slot_numerators == (0, 80, 92)
    calls.clear()
    low = proof.states_visited + lanes
    exhausted = residual._residual_frontier_probe(options, queue, lanes, stage, low, 1.0, source)
    assert len(calls) == 1 and exhausted.score_proof.complete
    assert exhausted.selection.score_exact and not exhausted.selection.canonical_exact
    assert exhausted.completion.status == "STATE_BUDGET_EXHAUSTED"
    assert exhausted.completion.prefix_states == exhausted.completion.prefix_reads == lanes
    assert exhausted.completion.dfs_states_visited == 0
    assert exhausted.score_proof.states_visited + exhausted.completion.states_visited == low
    calls.clear()
    complete = residual._residual_frontier_probe(options, queue, lanes, stage, 8192, 1.0, source)
    assert len(calls) == 1 and complete.selection.canonical_exact
    assert complete.selection.option_indices == canonical["indices"]
    assert complete.completion.state_budget == 8192 - complete.score_proof.states_visited
    assert complete.completion.states_visited == complete.completion.prefix_states + complete.completion.dfs_states_visited
    assert complete.score_proof.states_visited + complete.completion.states_visited <= 8192
    assert complete.baseline_selection is stage.selection and asdict(stage) == before


@dataclass(frozen=True)
class Probe:
    selection: CanonicalCountSpaceBnbSelection


@pytest.mark.parametrize("old_name,new_name,search_name", (
    ("select_branch_reserved_axis_capacity_frontier_frame", "select_branch_reserved_residual_capacity_frontier_frame",
     "BranchReservedIntegerCapacitySearch"),
    ("select_dual_reuse_axis_capacity_frontier_frame", "select_dual_reuse_residual_capacity_frontier_frame",
     "DualReuseIntegerCapacitySearch")))
@pytest.mark.parametrize("route", ("initial_complete", "exhausted_initial_probe"))
def test_ordinary_routers_preserve_the_previous_results_without_capacity_completion(monkeypatch, old_name, new_name, search_name, route):
    options, queue, lanes = case_fixture("aliases")
    baseline, _ = certified_baseline(options, queue, lanes)
    initial = Stage(replace(baseline, lp_node_budget=144, lp_calls=142,
        canonical_exact=route == "initial_complete"))
    final = Stage(replace(baseline, lp_node_budget=256, lp_calls=254))
    logs, results = {}, {}

    def unexpected(*args, **kwargs):
        pytest.fail("ordinary routes cannot capture or reuse a frontier capacity table")

    for label, module, selector_name in (("old", previous, old_name), ("new", residual, new_name)):
        log = {"constructed": 0, "caps": [], "probes": []}
        logs[label] = log

        def search_class(log):
            class Search:
                def __init__(self, options_arg, queue_arg, lanes_arg, *, idle_duration):
                    assert (options_arg, queue_arg, lanes_arg, idle_duration) == (options, queue, lanes, 1.0)
                    log["constructed"] += 1

                def advance_to(self, cap):
                    log["caps"].append(cap)
                    return initial if cap == 144 else final
            return Search

        def ordinary_probe(log):
            def run(options_arg, queue_arg, lanes_arg, paid, budget, idle):
                assert (options_arg, queue_arg, lanes_arg, budget, idle) == (options, queue, lanes, 8192, 1.0)
                log["probes"].append((paid.selection.lp_node_budget, budget))
                return Probe(paid.selection)
            return run

        monkeypatch.setattr(module, search_name, search_class(log))
        monkeypatch.setattr(module, "_probe", ordinary_probe(log))
        monkeypatch.setattr(module, "capture_strict_frontier", unexpected)
        results[label] = getattr(module, selector_name)(options, queue, lanes, 144, 256, 8192)
    assert results["new"] == results["old"] and logs["new"] == logs["old"]
    assert logs["new"]["constructed"] == 1 and results["new"].frontier_evidence is None
    if route == "initial_complete":
        assert logs["new"]["caps"] == [144] and not logs["new"]["probes"]
        assert results["new"].status == "LP_INITIAL_COMPLETE"
    else:
        assert logs["new"]["caps"] == [144, 256] and logs["new"]["probes"] == [(144, 8192)]
        assert results["new"].status == "LP_RESUME_INCOMPLETE"
