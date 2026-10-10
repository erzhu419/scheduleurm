"""Parallel root/sample chunks preserve native scores, causal state, and fees."""
from collections import Counter

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.parallel_root_lane_rollout import ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import TwoStepCurrentPolicyBGIncumbentLaneRollout


OPTIONS = (
    FrameOption("a1", (1, 0, 0), 1., ((1.,), (), ())),
    FrameOption("b1", (0, 1, 0), 1.25, ((), (1.25,), ())),
    FrameOption("c1", (0, 0, 1), 1.5, ((), (), (1.5,))),
    FrameOption("loaded-ab", (1, 1, 0), 1.75, ((.5,), (1.75,), ()), loaded=True),
    FrameOption("a2", (2, 0, 0), 1.5, ((1.5, 1.5), (), ())),
)


def semantic(planner):
    return {"rng": planner.rng.getstate(), "counter": planner.decision_counter,
            "rates": planner.arrival_rates,
            "counts": getattr(planner, "_outer_observed_counts", None),
            "child": semantic(planner.inner_planner) if hasattr(planner, "inner_planner") else None}


def scorer_context():
    paths = tuple(PlanningArrivals(events, False, 0, 0) for events in (
        ((.2, 2), (.9, 1)), ((.1, 0), (.7, 2)), ((.6, 1),), ()))
    return ((2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
            ((2.1, .3), (1.2,), (.1,)), paths)


def same_logical_state(native, parallel):
    assert semantic(parallel) == semantic(native)
    for name in ("_recourse_work", "_two_step_work", "_inner_current_work", "_inner_current_fallbacks"):
        assert Counter(getattr(parallel, name)) == Counter(getattr(native, name)), name


def test_loaded_multiclass_chunks_keep_global_sample_seeds_two_calls_and_shared_suffix_fees():
    native = TwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "parallel-normal", 4, 16)
    with ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "parallel-normal", 4, 16, workers=2) as parallel:
        for planner in (native, parallel):
            planner.arrival_rates = (.12, .07, .08)
            planner._outer_observed_counts = (120, 70, 80)
            planner.decision_counter = 7
        before = semantic(parallel)
        expected = native.score_background_roots(*scorer_context())
        actual = parallel.score_background_roots(*scorer_context())
        assert actual == expected
        assert not actual.background_rollout_overflow and actual.background_planning_events > 0
        assert parallel.rng.getstate() == before["rng"] and parallel.decision_counter == 7
        same_logical_state(native, parallel)
        assert parallel._two_step_work["recourse_first_calls"] > 0
        assert parallel._two_step_work["recourse_second_calls"] > 0
        work = parallel.reuse_work["parallel_root_work"]
        assert work["submitted_roots"] == len(expected.costs) == 5
        assert work["submitted_chunks"] == work["completed_chunks"] == work["included_chunks"] == 10
        assert work["discarded_chunks"] == 0
        assert 0 < work["worker_planner_constructors"] <= 6


def test_first_overflow_truncates_native_root_sample_order_and_retains_actual_discarded_work():
    native = TwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "parallel-overflow", 4, 2)
    with ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "parallel-overflow", 4, 2, workers=2) as parallel:
        for planner in (native, parallel):
            planner.arrival_rates = (.5, .4, .3)
            planner._outer_observed_counts = (500, 400, 300)
            planner.decision_counter = 7
        context = (*scorer_context()[:-1], (
            PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * 4)
        expected = native.score_background_roots(*context)
        actual = parallel.score_background_roots(*context)
        assert expected.background_rollout_overflow
        assert actual == expected
        same_logical_state(native, parallel)
        work = parallel.reuse_work["parallel_root_work"]
        assert work["completed_chunks"] == work["submitted_chunks"] == 10
        assert 0 < work["included_chunks"] < work["completed_chunks"]
        assert work["included_chunks"] + work["discarded_chunks"] == work["completed_chunks"]
        assert work["discarded_chunks"] > 0 and work["worker_cpu_s"] > 0
        assert parallel.reuse_work["mw_reuse_work"]["actual_option_probes"] > 0
        assert (parallel.reuse_work["inner_decision_reuse_work"]["current_choose_calls"]
                > native._recourse_work["recourse_calls"])


def test_complete_native_loaded_trajectory_and_untouched_child_carry_match_every_frame():
    model = FrameModel("toy", "parallel-full", 2, ("a", "b", "c"), (1., 1., 1.), OPTIONS)
    tape = [(0., 0, 0), (0., 0, 1), (0., 1, 0), (0., 2, 0),
            (3., 0, 2), (3., 1, 1), (3., 2, 1), (6., 0, 3), (8., 2, 2)]
    native = TwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "parallel-full", 4, 8)
    native_states, parallel_states = [], []

    def capture(planner, states):
        def select(*context):
            decision = planner(*context)
            states.append(semantic(planner))
            return decision
        return select

    expected = simulate_observed_rate_contextual_trajectory(
        model, tape, 12., 16., capture(native, native_states), policy="parallel-comparison")
    with ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "parallel-full", 4, 8, workers=2) as parallel:
        actual = simulate_observed_rate_contextual_trajectory(
            model, tape, 12., 16., capture(parallel, parallel_states), policy="parallel-comparison")
        assert actual == expected
        assert parallel_states == native_states and semantic(parallel) == semantic(native)
        assert any(3 in frame["option_indices"] for frame in actual["frames"])
        seen_recourse = checked_carry = False
        previous = None
        for frame, state in zip(actual["frames"], parallel_states, strict=True):
            if frame["rollout"]["recourse_calls"]:
                seen_recourse = True
            elif seen_recourse:
                assert state["child"] == previous["child"]
                checked_carry = True
            previous = state
        assert seen_recourse and checked_carry
        work = parallel.reuse_work["parallel_root_work"]
        assert work["score_calls"] > 1
        assert work["completed_chunks"] == work["submitted_chunks"]
        assert work["included_chunks"] + work["discarded_chunks"] == work["completed_chunks"]
        assert 0 < work["worker_planner_constructors"] <= 6
