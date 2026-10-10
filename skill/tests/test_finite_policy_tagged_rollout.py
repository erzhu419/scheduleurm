"""Fixed tagged jobs keep native approximation fees and free-policy semantics."""
from collections import Counter
from fractions import Fraction
from unittest.mock import patch

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundRootScores, _LazyBackgroundArrivals,
)
from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES
from algorithm.experiments.finite_policy_tagged_rollout import (
    _Population, _score_approx, _tree_semantic, evaluate_root_sample,
)
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree,
)
from algorithm import parallel_root_lane_rollout as native
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


OPTIONS = (
    FrameOption("single", (1,), 1.3, ((.3,),)),
    FrameOption("loaded-pair", (2,), 2.5, ((.1, 2.5),), loaded=True),
)
SAMPLES, CAP = 2, 16
PATHS = tuple(PlanningArrivals(events, False, 0, 0) for events in (
    ((.21, 0), (2.49, 0)), ((.19, 0), (2.47, 0))))
TIMINGS = {"worker_cpu_s", "worker_wall_s", "constructor_cpu_s", "constructor_wall_s"}


def exact(value):
    return Fraction.from_float(float(value))


def tree(planner):
    return capture_planner_tree(planner, Counter())


def anchor(seed, cap=CAP, *, idle=False):
    planner = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, seed, SAMPLES, cap)
    if idle:
        planner(OPTIONS, (4,), (1000.1, 1006.2), 1000.1, 0, ((.9, .6, .3, .1),), (4,))
        context = dict(queue=(1,), available=(1002.1, 1005.3), now=1002.1,
            lane=0, ages=((.2,),), counts=(4,))
    else:
        context = dict(queue=(4,), available=(1000.1, 1006.2), now=1000.1,
            lane=0, ages=((.9, .6, .3, .1),), counts=(100000 if cap == 2 else 200,))
    before = tree(planner)
    decision = planner(OPTIONS, context["queue"], context["available"], context["now"],
        context["lane"], context["ages"], context["counts"])
    return context, before, tree(planner), decision


def payload(context, before, after, root, sample):
    return dict(queue=context["queue"], available=context["available"], now=context["now"],
        lane=context["lane"], ages=context["ages"],
        outer_state=(after["planners"][0]["rng_state"], before["planners"][0]["decision_counter"],
            after["planners"][0]["arrival_rates"], context["counts"]),
        input_states=tuple(_tree_semantic(saved) for saved in before["planners"][1:]),
        root=root, sample_start=sample, paths=(PATHS[sample],))


def assert_population(result):
    population = result["population"]
    assert population["cost"] == result["cost"]
    assert population["squared"] == result["squared"]
    assert population["evaluations"] == result["evaluations"]
    assert population["tagged_remaining"] == population["tagged_total"] - result["evaluations"]
    assert len(set(population["completed_tagged_ids"])) == result["evaluations"]
    assert all(identity[0] in ("visible", "prefix") for identity in population["completed_tagged_ids"])
    for key in ("cost", "squared", "evaluations", "tagged_total"):
        assert sum(group[key] for group in population["sources"].values()) == population[key]
        assert sum(group[key] for group in population["per_coordinate"]) == population[key]


def test_instrumented_approx_reconstructs_complete_native_scores_fees_and_child_states():
    seed = "finite-tagged-shared"
    context, before, after, _ = anchor(seed)
    oracle = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, seed, SAMPLES, CAP)
    restore_planner_tree(oracle, before)
    native._restore_semantic(oracle, payload(context, before, after, 0, 0)["outer_state"])
    expected = oracle.score_background_roots(context["queue"], context["available"],
        context["now"], context["lane"], context["ages"], PATHS)
    assert not expected.background_rollout_overflow
    costs, squared = dict.fromkeys(expected.costs, Fraction(0)), dict.fromkeys(expected.costs, Fraction(0))
    totals, suffix_work = Counter(), [[0] * 4 for _ in PATHS]
    work = Counter(recourse_rng_words_copied=len(after["planners"][0]["rng_state"][1]),
        recourse_counter_reads=1)
    other_work = {name: Counter() for name in ("two_step_work", "inner_current_work", "inner_current_fallbacks")}
    states = list(payload(context, before, after, 0, 0)["input_states"])
    for root in expected.costs:
        for sample, path in enumerate(PATHS):
            current = payload(context, before, after, root, sample)
            native._initialize_worker(OPTIONS, seed, SAMPLES, CAP)
            raw = native._run_root(current)
            events = tuple((exact(birth), coordinate) for birth, coordinate in path.events)
            actual = _score_approx(OPTIONS, seed, SAMPLES, CAP, current,
                _Population(context["queue"], context["ages"], events))
            assert {key: actual[key] for key in raw if key not in TIMINGS} == {
                key: value for key, value in raw.items() if key not in TIMINGS}
            assert_population(actual)
            assert actual["complete"] and not actual["population"]["tagged_remaining"]
            costs[root] += actual["cost"]
            squared[root] += actual["squared"]
            for name in ("calls", "idle_calls", "evaluations", "admitted_reads", "tag_reads", "background_admissions"):
                totals[name] += actual[name]
            work.update(actual["recourse_work"])
            for name, target in other_work.items():
                target.update(actual[name])
            for index, value in enumerate(actual["suffix_work"][0]):
                suffix_work[sample][index] = max(suffix_work[sample][index], value)
            for level, flag in enumerate(("current_touched", "BG_touched")):
                if actual[flag]:
                    states[level] = actual["states"][level]
    fees = tuple(sum(row[i] for row in suffix_work) for i in range(4))
    reconstructed = BackgroundRootScores(costs, squared, len(costs), totals["calls"],
        totals["calls"] * len(OPTIONS), len(OPTIONS), totals["idle_calls"], SAMPLES,
        sum(context["queue"]), totals["evaluations"], *fees, totals["admitted_reads"],
        totals["tag_reads"], totals["background_admissions"], False)
    assert reconstructed == expected
    assert work == oracle._recourse_work
    for name, actual in other_work.items():
        assert actual == getattr(oracle, "_" + name)
    assert tuple(states) == (native._semantic_state(oracle.inner_planner),
        native._semantic_state(oracle.inner_planner.inner_planner))


def test_free_finite_consumes_complete_native_calls_after_full_tree_including_idle():
    seed = "finite-tagged-idle"
    context, before, after, _ = anchor(seed, idle=True)
    paths = (PlanningArrivals(((1.49, 0),), False, 0, 0), PlanningArrivals((), False, 0, 0))
    assert before["planners"][1:] != tree(
        ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, seed, SAMPLES, CAP))["planners"][1:]
    traced = []
    cls = native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout
    call = cls.__call__

    def observe(planner, *current):
        entry = tree(planner)
        decision = call(planner, *current)
        traced.append((current, entry, decision, tree(planner)))
        return decision

    with patch.object(cls, "__call__", observe):
        result = evaluate_root_sample(OPTIONS, seed, paths, before, after, context,
            0, 0, sample_count=SAMPLES, event_cap=CAP, workers=2)
    approx, free = result["approx"], result["free"]
    assert result["paired_complete"] and free["idle_calls"] == 1 and free["calls"] == len(traced) == 2
    assert free["initial_tree"] == traced[0][1] == after
    assert free["first_native_decision"] == traced[0][2]
    assert free["first_call_context"] == traced[0][0][1:]
    assert free["first_native_decision"].selection.option_indices == (None,)
    reference = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, seed, SAMPLES, CAP)
    restore_planner_tree(reference, after)
    logical = Counter()
    for current, entry, proposed, exit_state in traced:
        assert tree(reference) == entry
        expected = reference(*current)
        assert expected == proposed and tree(reference) == exit_state
        logical.update({name: getattr(expected, name) for name in FEES + TWO_STEP_WORK})
        for values in (expected.inner_current_work, expected.inner_current_fallbacks):
            logical.update({"inner_current_" + name: value for name, value in values.items()})
    assert Counter(free["logical_work"]) == logical and free["final_tree"] == tree(reference)
    assert free["final_tree"]["planners"][0]["decision_counter"] == after["planners"][0]["decision_counter"] + 2
    assert result["tagged_population"] == approx["population"]["tagged_population"] == free["population"]["tagged_population"]
    assert result["environment"]["seed"] == f"{seed}:background:{before['planners'][0]['decision_counter']}:0"
    assert result["environment"]["rates"] == after["planners"][0]["arrival_rates"]
    assert free["final_tree"]["planners"][0]["arrival_rates"] != result["environment"]["rates"]
    assert traced[0][0][2][1] == traced[1][0][2][1] == context["available"][1]
    duration = exact(1.3)
    last_start = duration + 1
    assert traced[0][0][3] == float(exact(context["now"]) + duration)
    assert traced[1][0][3] == float(exact(context["now"]) + last_start)
    assert exact(traced[1][0][3]) != exact(context["now"]) + last_start
    flows = (exact(.3) + exact(.2), last_start + exact(.3) - exact(1.49))
    assert free["cost"] == approx["cost"] == sum(flows)
    assert free["squared"] == approx["squared"] == sum(flow * flow for flow in flows)
    suffix = _LazyBackgroundArrivals(result["environment"]["rates"], OPTIONS[1].duration,
        result["environment"]["seed"], CAP, 1)
    assert suffix.release_through(last_start)
    assert free["background_events"] == approx["background_events"] == tuple(suffix.events) == ()
    assert free["suffix_work"] == approx["suffix_work"][0] == (len(suffix.events),
        suffix.random_draws, suffix.rate_reads, suffix.seed_initializations)
    for role in (approx, free):
        assert_population(role)


def test_overflow_retains_fixed_global_samples_native_partial_costs_and_births():
    seed, cap = "finite-tagged-overflow", 2
    context, before, after, _ = anchor(seed, cap)
    roster = []
    for root in range(len(OPTIONS)):
        for sample in range(SAMPLES):
            result = evaluate_root_sample(OPTIONS, seed, PATHS, before, after, context,
                root, sample, sample_count=SAMPLES, event_cap=cap, workers=2)
            roster.append((result["root"], result["sample_index"]))
            assert not result["paired_complete"]
            current = payload(context, before, after, root, sample)
            native._initialize_worker(OPTIONS, seed, SAMPLES, cap)
            expected = native._run_root(current)
            assert {key: result["approx"][key] for key in expected if key not in TIMINGS} == {
                key: value for key, value in expected.items() if key not in TIMINGS}
            suffix = _LazyBackgroundArrivals(result["environment"]["rates"], OPTIONS[1].duration,
                result["environment"]["seed"], cap, len(PATHS[sample].events))
            assert not suffix.release_through(Fraction(10))
            for role in (result["approx"], result["free"]):
                assert role["overflow"] and not role["complete"]
                assert role["population"]["tagged_remaining"] > 0
                assert 0 < role["evaluations"] < role["population"]["tagged_total"]
                assert isinstance(role["cost"], Fraction) and isinstance(role["squared"], Fraction)
                assert role["background_events"] == tuple(suffix.events)
                assert len(role["background_events"]) + len(PATHS[sample].events) == cap + 1
                assert_population(role)
            assert result["approx"]["population"]["tagged_population"] == result["free"]["population"]["tagged_population"]
            assert result["approx"]["recourse_work"]["recourse_calls"] > 0
            assert result["free"]["logical_work"]["planning_random_draws"] > 0
            assert result["free"]["final_tree"]["planners"][0]["decision_counter"] == (
                after["planners"][0]["decision_counter"] + result["free"]["calls"])
    assert roster == [(0, 0), (0, 1), (1, 0), (1, 1)]


if __name__ == "__main__":
    test_instrumented_approx_reconstructs_complete_native_scores_fees_and_child_states()
    test_free_finite_consumes_complete_native_calls_after_full_tree_including_idle()
    test_overflow_retains_fixed_global_samples_native_partial_costs_and_births()
    print("3 passed", flush=True)
