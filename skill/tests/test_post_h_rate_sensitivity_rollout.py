"""Post-H rate changes retain the H stream, native estimator and score roster."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json
from random import Random
from unittest.mock import patch

from algorithm import parallel_root_lane_rollout as native
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.common_horizon_flow_rollout import evaluate_root_sample
from algorithm.experiments.native_terminal_flow_rollout import evaluate_cached as original_evaluate
from algorithm.experiments.post_h_rate_sensitivity_rollout import (
    evaluate_cached, switch_post_h_rates,
)
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree
from skill.tests.test_competing_terminal_flow_rollout import json_value
from skill.tests.test_native_terminal_flow_rollout import RecordedAnchor


F = Fraction


def test_hand_residual_switch_retains_H_events_and_consumes_no_randomness():
    suffix = _LazyBackgroundArrivals((2., 4., 3.), 2., 17, 128, 0)
    assert suffix.release_through(F(4))
    # Pending residuals are positive supported clocks; these exact values make
    # the independent answer visible: .5*2/1=1 and .25*4/8=.125.
    suffix.next_times[:] = (4.5, 4.25, 4.75)
    events = tuple(suffix.events)
    rng_state = suffix.rng.getstate()
    counters = (suffix.random_draws, suffix.rate_reads, suffix.seed_initializations)
    switch = switch_post_h_rates(suffix, F(4), (1., 8., 3.))
    assert suffix.next_times == [5., 4.125, 4.75]
    assert switch["old_pending_times"] == (4.5, 4.25, 4.75)
    assert switch["new_pending_times"] == (5., 4.125, 4.75)
    assert suffix.rates == switch["new_rates"] == (1., 8., 3.)
    assert tuple(suffix.events) == events and suffix.rng.getstate() == rng_state
    assert (suffix.random_draws, suffix.rate_reads, suffix.seed_initializations) == counters
    work = switch["work"]
    assert work["random_draws"] == work["seed_initializations"] == 0
    assert work["rate_coordinate_reads"] == 6 and work["pending_time_coordinate_reads"] == 3
    assert work["rate_coordinate_writes"] == 3
    for field in ("residual_subtractions", "rate_ratio_divisions", "residual_multiplications",
            "horizon_additions", "pending_time_writes"):
        assert work[field] == 2
    # Subsequent new-rate gaps consume exactly the saved stream, with no
    # independent seed or replacement random draws at the intervention.
    independent_rng = Random()
    independent_rng.setstate(rng_state)
    pending = [5., 4.125, 4.75]
    expected = []
    while True:
        coordinate = min(range(3), key=lambda index: (pending[index], index))
        birth = pending[coordinate]
        if birth > 5.:
            break
        expected.append((F.from_float(birth), coordinate))
        pending[coordinate] = birth + independent_rng.expovariate((1., 8., 3.)[coordinate])
    assert suffix.release_through(F(5))
    assert tuple(suffix.events) == events + tuple(expected)
    assert suffix.next_times == pending and suffix.rng.getstate() == independent_rng.getstate()
    assert (suffix.random_draws, suffix.rate_reads, suffix.seed_initializations) == (
        counters[0] + len(expected), counters[1] + len(expected), counters[2])


def _real_cached_fixture():
    options = (FrameOption("single", (1,), 2., ((.5,),)),)
    seed, samples, cap = 0, 1, 128
    context = dict(queue=(2,), available=(100., 101.), now=100., lane=0,
        ages=((2., 1.),), counts=(200,))
    reference = RecordedAnchor(options, seed, samples, cap)
    before = capture_planner_tree(reference, Counter())
    proposal = reference(options, context["queue"], context["available"], context["now"],
        context["lane"], context["ages"], context["counts"])
    assert proposal.eligible and sum(context["queue"]) <= proposal.queue_limit
    assert len(reference.paths) == 1 and len(reference.paths[0].events) == 3
    after = capture_planner_tree(reference, Counter())
    result = evaluate_root_sample(options, seed, reference.paths, before, after, context,
        0, 0, sample_count=samples, event_cap=cap, workers=2)
    assert result["complete"] and sum(result["flow"]["terminal_state"]["queue"]) > 0
    cached = json.loads(json.dumps(json_value(result), default=lambda value: json_value(asdict(value))))
    return options, seed, samples, cap, context, cached


def _traced_evaluate(function, fixture, **extra):
    options, seed, samples, cap, context, cached = fixture
    traces = []
    cls = native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout
    native_call = cls.__call__

    def observe(planner, *arguments):
        before = capture_planner_tree(planner, Counter())
        decision = native_call(planner, *arguments)
        traces.append((arguments, before, decision, capture_planner_tree(planner, Counter())))
        return decision

    with patch.object(cls, "__call__", observe):
        result = function(options, seed, cached, context["now"], workers=2,
            sample_count=samples, event_cap=cap, **extra)
    return result, traces


def test_same_rates_public_real_JSON_retains_every_native_decision_tree_and_work():
    fixture = _real_cached_fixture()
    cached = fixture[-1]
    saved = deepcopy(cached)
    baseline, baseline_trace = _traced_evaluate(original_evaluate, fixture)
    repeated, repeated_trace = _traced_evaluate(evaluate_cached, fixture,
        post_h_rates=cached["environment"]["rates"])
    assert cached == saved and baseline["complete"] and repeated["complete"]
    assert baseline_trace and baseline_trace == repeated_trace
    for field in ("root", "sample_index", "environment", "scored_population"):
        assert repeated[field] == baseline[field]
    # Timing is measured separately; charged operation counts and every
    # semantic output must be identical when the external rates stay equal.
    for field in ("cost", "common_area", "waiting_area", "inflight_area", "terminal_cost",
            "population", "calls", "idle_calls", "terminal_calls", "terminal_idle_calls",
            "logical_work", "proposal_status", "dispatch_work", "incremental_tail_work",
            "final_lane_ends", "final_queue", "final_counts", "first_call_context",
            "first_native_decision", "background_events", "suffix_work", "initial_tree",
            "final_tree", "regeneration_work", "reused_window_work", "restore_work",
            "preprocessing", "controller_constructors"):
        assert repeated["flow"][field] == baseline["flow"][field], field
    for group, counts in baseline["flow"]["actual_work"].items():
        observed = repeated["flow"]["actual_work"][group]
        assert {name: value for name, value in observed.items()
            if not name.endswith(("cpu_s", "wall_s"))} == {
            name: value for name, value in counts.items()
            if not name.endswith(("cpu_s", "wall_s"))}, group
    switch = repeated["flow"]["rate_switch"]
    assert switch["old_pending_times"] == switch["new_pending_times"]
    assert switch["work"]["pending_time_writes"] == switch["work"]["random_draws"] == 0
    return fixture, baseline


def test_changed_rates_keep_native_counts_time_and_cap_crossing_population(fixture_and_baseline=None):
    if fixture_and_baseline is None:
        fixture = _real_cached_fixture()
        baseline, _ = _traced_evaluate(original_evaluate, fixture)
    else:
        fixture, baseline = fixture_and_baseline
    cached, saved = fixture[-1], deepcopy(fixture[-1])
    changed, traces = _traced_evaluate(evaluate_cached, fixture, post_h_rates=(4.,))
    assert changed["complete"] and not changed["flow"]["overflow"] and traces
    assert changed["flow"]["background_events"] != baseline["flow"]["background_events"]
    assert changed["scored_population"] == baseline["scored_population"]
    # Single-class FIFO forbids later births from passing the original H jobs.
    assert changed["flow"]["cost"] == baseline["flow"]["cost"]
    assert changed["flow"]["initial_tree"] == baseline["flow"]["initial_tree"]
    for arguments, before, decision, after in traces:
        now, counts = arguments[3], arguments[6]
        assert decision.observed_arrival_counts == counts
        assert decision.rate_exposure_s == now
        assert decision.estimated_arrival_rates == tuple(count / now for count in counts)
        assert after["planners"][0]["arrival_rates"] == decision.estimated_arrival_rates
        assert after["planners"][0]["decision_counter"] == before["planners"][0]["decision_counter"] + 1
    assert traces[0][0][6] == tuple(cached["flow"]["terminal_state"]["counts"])
    assert traces[0][2].estimated_arrival_rates != (4.,)
    capped, cap_trace = _traced_evaluate(evaluate_cached, fixture, post_h_rates=(64.,))
    flow = capped["flow"]
    assert flow["overflow"] and not capped["complete"] and cap_trace
    assert flow["calls"] == len(cap_trace)
    assert len(cached["environment"]["prefix_events"]) + len(flow["background_events"]) == fixture[3] + 1
    crossing = flow["background_events"][-1][0]
    assert flow["partial_terminal_observation_end"] == crossing
    assert flow["population"]["remaining"] > 0
    assert flow["population"]["pending_waiting_accrual"] == (
        flow["population"]["remaining"] * (crossing - flow["horizon"]))
    assert flow["terminal_cost"] == (flow["population"]["known_completion_residual"]
        + flow["population"]["pending_waiting_accrual"])
    for result in (changed, capped):
        assert result["scored_population"] == baseline["scored_population"]
        assert result["flow"]["population"]["total"] == baseline["flow"]["population"]["total"]
        for field in ("common_area", "waiting_area", "inflight_area", "regeneration_work"):
            assert result["flow"][field] == baseline["flow"][field]
        assert json_value(result["flow"]["background_events"][:len(cached["flow"]["background_events"])]) == (
            cached["flow"]["background_events"])
    assert cached == saved


if __name__ == "__main__":
    test_hand_residual_switch_retains_H_events_and_consumes_no_randomness()
    fixture_and_baseline = test_same_rates_public_real_JSON_retains_every_native_decision_tree_and_work()
    test_changed_rates_keep_native_counts_time_and_cap_crossing_population(fixture_and_baseline)
    print("3 passed", flush=True)
