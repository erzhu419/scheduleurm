"""Independent H timing examples, native replay identity and capped cost."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json
from unittest.mock import patch

from algorithm import parallel_root_lane_rollout as native
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.experiments import h_timing_mirror_rollout as mirror
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.common_horizon_flow_rollout import evaluate_root_sample
from algorithm.experiments.native_terminal_flow_rollout import evaluate_cached as original_evaluate
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree
from skill.tests.test_common_horizon_flow_rollout import DispatchFixture
from skill.tests.test_competing_terminal_flow_rollout import job, json_value
from skill.tests.test_native_terminal_flow_rollout import RecordedAnchor


F = Fraction


def test_hand_mirror_preserves_roster_reorders_FIFO_and_keeps_busy_reservation():
    options = (FrameOption("A", (1, 0), 2., ((.5,), ())),
        FrameOption("B", (0, 1), 2., ((), (1.5,))))
    roster = [job(("visible", 0, 0), 0, -1, "visible", 0, F(1, 2)),
        job(("visible", 1, 0), 1, -2, "visible", 2, F(7, 2)),
        job(("busy", 0), 1, -3, "committed", -1, F(9, 2)),
        job(("prefix", 0), 0, 1, "prefix", 8, F(17, 2)),
        job(("background", 0), 0, F(7, 2), "background", 10, F(21, 2)),
        job(("background", 1), 1, F(5, 2), "background", 6, F(15, 2))]
    saved = deepcopy(roster)
    reflected = mirror.reflect_roster(roster, F(4), F(2))
    observed = {entry["id"]: entry for entry in reflected}
    assert roster == saved and len(observed) == 6
    for entry in roster[:3]:
        assert observed[entry["id"]] == entry
    expected = {
        ("prefix", 0): (F(3), "background"),
        ("background", 0): (F(1, 2), "prefix"),
        ("background", 1): (F(3, 2), "prefix")}
    for identity, (birth, source) in expected.items():
        original = next(entry for entry in roster if entry["id"] == identity)
        assert observed[identity] == dict(original, birth=birth, source=source)
    assert Counter(entry["coordinate"] for entry in reflected) == Counter({0: 3, 1: 3})
    twice = mirror.reflect_roster(reflected, F(4), F(2))
    assert {entry["id"]: entry for entry in twice} == {entry["id"]: entry for entry in roster}
    assert mirror.reflect_roster(roster, F(4), F(2), reflect=False) == tuple(roster)

    context = dict(queue=(1, 1), available=(100., 105.), now=100., lane=0,
        ages=((1.,), (2.,)), counts=(10, 20))
    planner = DispatchFixture(options)
    flow = mirror.simulate_within_h(planner, context, 0, reflected,
        initial_inflight=(dict(id=("busy", 0), coordinate=1,
            birth=97., start=99., completion=104.5),))
    # A visible job starts at 0; B visible starts at 2. The three newborns
    # wait at H. Busy completion 4.5 does not release its lane before 5.
    assert flow["waiting_area"] == 9 and flow["inflight_area"] == 6
    assert flow["common_area"] == 15
    state = flow["terminal_state"]
    assert state["queue"] == (2, 1) and state["counts"] == (12, 21)
    assert state["lane_ends"] == (F(4), F(5))
    assert state["waiting_ids"] == ((("background", 0), ("prefix", 0)),
        (("background", 1),))
    assert [call["now"] for call in planner.calls] == [102.]
    assert planner.calls[0]["queue"] == (1, 2)
    assert planner.calls[0]["counts"] == (11, 21)
    assert planner.calls[0]["ages"] == ((1.5,), (4., .5))
    restored = {entry["id"]: entry for entry in flow["jobs"]}
    assert set(restored) == set(observed)
    for identity in expected:
        assert restored[identity]["start"] is restored[identity]["completion"] is None
    assert restored[("busy", 0)] == roster[2]


def _observe_calls(traces):
    native_call = native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout.__call__

    def observe(planner, *arguments):
        before = capture_planner_tree(planner, Counter())
        decision = native_call(planner, *arguments)
        traces.append((arguments, before, decision, capture_planner_tree(planner, Counter())))
        return decision

    return observe


def _real_cached_fixture(cap=128):
    options = (FrameOption("single", (1,), 2., ((.5,),)),)
    seed, samples = 0, 1
    context = dict(queue=(2,), available=(100., 101.), now=100., lane=0,
        ages=((2., 1.),), counts=(200,))
    reference = RecordedAnchor(options, seed, samples, cap)
    before = capture_planner_tree(reference, Counter())
    proposal = reference(options, context["queue"], context["available"], context["now"],
        context["lane"], context["ages"], context["counts"])
    assert proposal.eligible and sum(context["queue"]) <= proposal.queue_limit
    assert len(reference.paths) == 1 and len(reference.paths[0].events) == 3
    after = capture_planner_tree(reference, Counter())
    h_traces = []
    with patch.object(native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout,
            "__call__", _observe_calls(h_traces)):
        source = evaluate_root_sample(options, seed, reference.paths, before, after, context,
            0, 0, sample_count=samples, event_cap=cap, workers=2)
    assert source["complete"] and sum(source["flow"]["terminal_state"]["queue"]) > 0
    cached = json.loads(json.dumps(json_value(source), default=lambda value: json_value(asdict(value))))
    return options, seed, samples, cap, context, after, h_traces, cached


def _trace_mirror(fixture, *, reflect):
    options, seed, samples, cap, context, after, _, cached = fixture
    traces, suffix_snapshots = [], []
    cls = native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout
    terminal_call = mirror._simulate_native_terminal

    def observe_terminal(planner, jobs, state, suffix, **arguments):
        suffix_snapshots.append((tuple(suffix.events), tuple(suffix.next_times),
            suffix.rng.getstate(), suffix.random_draws, suffix.rate_reads, suffix.seed_initializations))
        return terminal_call(planner, jobs, state, suffix, **arguments)

    with patch.object(cls, "__call__", _observe_calls(traces)), patch.object(
            mirror, "_simulate_native_terminal", observe_terminal):
        result = mirror.evaluate_cached(options, seed, cached, context, after,
            workers=2, sample_count=samples, event_cap=cap, reflect=reflect)
    return result, traces, suffix_snapshots


def _expected_h_suffix(cached, cap):
    environment = cached["environment"]
    suffix = _LazyBackgroundArrivals(environment["rates"], environment["window_s"],
        environment["seed"], cap, len(environment["prefix_events"]))
    assert suffix.release_through(F(4))
    snapshot = (tuple(suffix.events), tuple(suffix.next_times), suffix.rng.getstate(),
        suffix.random_draws, suffix.rate_reads, suffix.seed_initializations)
    return suffix, snapshot


def test_identity_real_JSON_replays_every_H_and_terminal_native_tree(fixture=None):
    fixture = fixture or _real_cached_fixture()
    options, seed, samples, cap, context, after, h_traces, cached = fixture
    saved = deepcopy(cached)
    terminal_traces = []
    with patch.object(native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout,
            "__call__", _observe_calls(terminal_traces)):
        baseline = original_evaluate(options, seed, cached, context["now"],
            workers=2, sample_count=samples, event_cap=cap)
    repeated, traces, snapshots = _trace_mirror(fixture, reflect=False)
    assert cached == saved and baseline["complete"] and repeated["complete"]
    assert repeated["scored_population"] == baseline["scored_population"]
    flow, old_h = repeated["flow"], cached["flow"]
    assert flow["initial_tree"] == after
    assert json_value(flow["h_tree"]) == old_h["final_tree"]
    assert flow["h_tree"] == baseline["flow"]["initial_tree"]
    assert flow["final_tree"] == baseline["flow"]["final_tree"]
    for field in ("common_area", "waiting_area", "inflight_area", "terminal_state",
            "calls", "idle_calls", "logical_work", "proposal_status"):
        assert json_value(flow["window_flow"][field]) == old_h[field], field
    for field in ("cost", "common_area", "waiting_area", "inflight_area", "terminal_cost",
            "population", "calls", "idle_calls", "terminal_calls", "terminal_idle_calls",
            "logical_work", "proposal_status", "incremental_tail_work", "final_lane_ends",
            "final_queue", "final_counts", "first_call_context", "first_native_decision",
            "background_events", "suffix_work"):
        assert flow[field] == baseline["flow"][field], field
    assert traces[0][1] == after
    assert traces == h_traces + terminal_traces
    assert traces[flow["window_flow"]["calls"]][1] == baseline["flow"]["initial_tree"]
    suffix, expected = _expected_h_suffix(cached, cap)
    assert snapshots == [expected]
    assert suffix.release_through(F(traces[-1][0][3]) - F(context["now"]))
    assert flow["background_events"] == tuple(suffix.events)
    return fixture, baseline


def test_mirror_real_native_keeps_estimator_roster_and_exact_post_H_stream(fixture_and_baseline=None):
    if fixture_and_baseline is None:
        fixture_and_baseline = test_identity_real_JSON_replays_every_H_and_terminal_native_tree()
    fixture, baseline = fixture_and_baseline
    _, _, _, cap, context, after, _, cached = fixture
    saved = deepcopy(cached)
    result, traces, snapshots = _trace_mirror(fixture, reflect=True)
    assert cached == saved and result["complete"] and not result["flow"]["overflow"]
    assert traces and traces[0][1] == after
    original = {tuple(entry["id"]): entry for entry in baseline["scored_population"]}
    observed = {tuple(entry["id"]): entry for entry in result["scored_population"]}
    assert set(original) == set(observed)
    for identity, entry in original.items():
        assert observed[identity]["coordinate"] == entry["coordinate"]
        if entry["source"] in ("visible", "committed"):
            assert observed[identity] == entry
        else:
            assert observed[identity]["birth"] == F(4) - entry["birth"]
            assert observed[identity]["source"] == (
                "prefix" if observed[identity]["birth"] <= 2 else "background")
    assert Counter(entry["coordinate"] for entry in observed.values()) == Counter(
        entry["coordinate"] for entry in original.values())
    for arguments, before, decision, after_call in traces:
        now, counts = arguments[3], arguments[6]
        assert decision.observed_arrival_counts == counts
        assert decision.rate_exposure_s == now
        assert decision.estimated_arrival_rates == tuple(count / now for count in counts)
        assert after_call["planners"][0]["arrival_rates"] == decision.estimated_arrival_rates
        assert after_call["planners"][0]["decision_counter"] == (
            before["planners"][0]["decision_counter"] + 1)
    suffix, expected = _expected_h_suffix(cached, cap)
    assert snapshots == [expected]
    assert suffix.release_through(F(traces[-1][0][3]) - F(context["now"]))
    assert result["flow"]["background_events"] == tuple(suffix.events)
    assert result["flow"]["suffix_work"] == (len(suffix.events), suffix.random_draws,
        suffix.rate_reads, suffix.seed_initializations)


def test_original_cap_keeps_mirror_partial_cost_without_promoting_an_incomplete_score():
    # Three prefix births plus seven H-suffix births exactly fill cap 10.
    # The next original shared clock is 4.011655..., so terminal overflow
    # occurs after a consumed H=4 proposal rather than an artificial drain.
    fixture = _real_cached_fixture(cap=10)
    cached = fixture[-1]
    result, traces, snapshots = _trace_mirror(fixture, reflect=True)
    flow = result["flow"]
    assert flow["overflow"] and not flow["complete"] and not result["complete"]
    assert flow["population"]["remaining"] > 0 and flow["calls"] > 0
    assert flow["cost"] > 0 and flow["cost"] == flow["common_area"] + flow["terminal_cost"]
    suffix, expected = _expected_h_suffix(cached, 10)
    assert snapshots == [expected]
    crossing = F.from_float(suffix.next_times[0])
    assert 4 < crossing < F(9, 2)
    assert flow["partial_terminal_observation_end"] == crossing
    assert len(cached["environment"]["prefix_events"]) + len(flow["background_events"]) == 11
    population = flow["population"]
    assert population["pending_waiting_accrual"] == population["remaining"] * (crossing - 4)
    assert flow["terminal_cost"] == (
        population["known_completion_residual"] + population["pending_waiting_accrual"])
    assert flow["terminal_calls"] == flow["calls"]
    assert len(traces) == flow["window_flow"]["calls"] + flow["calls"]
    assert population["total"] == len(cached["flow"]["population"]["jobs"])


if __name__ == "__main__":
    test_hand_mirror_preserves_roster_reorders_FIFO_and_keeps_busy_reservation()
    fixture_and_baseline = test_identity_real_JSON_replays_every_H_and_terminal_native_tree()
    test_mirror_real_native_keeps_estimator_roster_and_exact_post_H_stream(fixture_and_baseline)
    test_original_cap_keeps_mirror_partial_cost_without_promoting_an_incomplete_score()
    print("4 passed", flush=True)
