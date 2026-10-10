"""Original-state screening replays prefixes and preserves saved causal effects."""
from collections import Counter
from dataclasses import asdict, replace
from fractions import Fraction

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.bg_incumbent_background_redecision_lane_rollout import BGIncumbentBackgroundRedecisionLaneRollout
from algorithm.class_protected_bg_incumbent_background_redecision_lane_rollout import (
    ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.aligned_hardware_class_protected_original_root_screen import (
    replay_prefixes, score_original_root, summarize,
)


def test_prefix_state_matches_native_through_idle_ineligible_and_overflow_frames():
    options = (FrameOption("single", (1,), 1., ((1.,),)),)
    native = BGIncumbentBackgroundRedecisionLaneRollout(
        options, "frozen-prefix", sample_count=8, event_cap=1)
    replay = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        options, "frozen-prefix", sample_count=8, event_cap=1)
    frames, states = [], []
    for now, queue, counts in ((0., (0,), (0,)), (1., (1,), (1000,)),
                               (2., (3,), (1000,)), (1e6, (1,), (1000,))):
        available = (now, now + 2.)
        ages = (tuple(0. for _ in range(queue[0])),)
        decision = native(options, queue, available, now, 0, ages, counts)
        info = asdict(decision)
        del info["selection"]
        frames.append({"start": now, "queue": queue, "available_before": available,
                       "observed_arrival_counts": counts, "waiting_ages": ages,
                       "lane_ids": (0,), "rollout": info})
        states.append((native.rng.getstate(), native.decision_counter))
    assert not frames[0]["rollout"]["eligible"]
    assert frames[1]["rollout"]["planning_overflow"]
    assert frames[1]["rollout"]["incumbent_candidate_count"] == 0
    assert not frames[2]["rollout"]["eligible"]
    assert frames[3]["rollout"]["eligible"] and not frames[3]["rollout"]["planning_overflow"]
    work = Counter()
    populations = []
    for index, frame, paths in replay_prefixes(replay, frames, work):
        populations.append(paths)
        assert replay.rng.getstate() == states[index][0]
        assert replay.decision_counter == states[index][1] - 1
        assert replay._outer_observed_counts == frame["observed_arrival_counts"]
    assert replay.rng.getstate() == native.rng.getstate()
    assert replay.decision_counter == native.decision_counter == 4
    assert list(map(len, populations)) == [0, 1, 0, 8]
    assert populations[1][0].overflow
    assert work["prefix_overflow_frames"] == 1
    assert work["replayed_decision_counter_writes"] == work["cached_frame_reads"] == 4
    for fee in ("random_draws", "rate_reads"):
        assert work["prefix_planning_" + fee] == sum(
            frame["rollout"]["planning_" + fee]
            - frame["rollout"]["background_planning_" + fee] for frame in frames)
    assert work["rate_estimation_coordinate_reads"] == 4
    assert work["rate_estimation_divisions"] == 3
    assert not work["root_option_probes"]  # Prefix replay performs no MW scans.


def test_native_same_aggregate_class_veto_keeps_joined_outcomes_and_short_circuit_fees():
    options = (FrameOption("a1", (1, 0), 1., ((1.,), ())),
               FrameOption("a2", (2, 0), 1.5, ((1.5, 1.5), ())),
               FrameOption("b1", (0, 1), 1., ((), (1.,))))
    planner = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        options, "original-root", sample_count=1)
    planner._outer_observed_counts = (2, 1)
    planner.arrival_rates = (.002, .001)
    planner.decision_counter = 7
    native_score = planner.score_background_roots
    captured = []

    def contrast(*args):
        native = native_score(*args)
        assert all(sum(values, Fraction(0)) == native.squared_costs[root]
                   for root, values in native.class_squared_costs.items())
        classes = {**native.class_squared_costs,
                   0: (Fraction(101), Fraction(100)),
                   1: (Fraction(100), Fraction(200))}
        scored = replace(native,
            costs={**native.costs, 0: Fraction(20), 1: Fraction(25)},
            squared_costs={root: sum(values, Fraction(0)) for root, values in classes.items()},
            class_squared_costs=classes)
        captured.append(scored)
        return scored

    planner.score_background_roots = contrast
    frame = {"start": 1000., "queue": (2, 1), "available_before": (1000., 1005.),
        "lane_ids": (0,), "waiting_ages": ((10., 0.), (0.,)), "option_indices": (0,),
        "rollout": {"improved_incumbent": True, "planning_overflow": False,
            "incumbent_root_index": 1, "baseline_predicted_completion_s": 25.,
            "chosen_predicted_completion_s": 20., "baseline_predicted_squared_flow_s2": 300.,
            "chosen_predicted_squared_flow_s2": 201.}}
    outcomes = {"delta_T_mean": 2., "delta_T_common_p90": None,
        "delta_900_unfinished": 1, "delta_1200_unfinished": 0, "delta_middle_queue_area_s": 0.,
        "per_coordinate_deltas": [{"coordinate": coordinate, "workload": name,
            "delta_T_mean": mean, "delta_T_common_mean": mean, "delta_T_common_p90": None,
            "delta_900_unfinished": tail, "delta_1200_unfinished": 0}
            for coordinate, name, mean, tail in ((0, "a", 2., 0), (1, "b", -1., -1))]}
    original = {"frame_index": 7, "decision_counter": 7, "factual_option_index": 0,
                "replacement_option_index": 1, "outcomes": outcomes}
    rng_before, work = planner.rng.getstate(), Counter()
    result = score_original_root(planner, frame, (PlanningArrivals((), False, 0, 0),), original, work)
    assert result["same_model_aggregate_predictions_match_passed"]
    assert not result["class_gate_passed"] and result["first_failing_coordinate"] == 0
    assert result["candidate_class_squared_flow_s2"] == [101., 100.]
    assert result["BG_class_squared_flow_s2"] == [100., 200.]
    assert result["outcomes"] is outcomes  # No effect recomputation or outcome filtering.
    assert result["class_risk_comparisons"] == work["class_risk_comparisons"] == 1
    assert work["class_squared_score_reads"] == 6
    assert work["class_squared_initializations"] == captured[0].candidate_count * 2
    assert work["class_squared_accumulations"] == work["flow_evaluations"] > 0
    assert work["recourse_calls"] > 0 and work["recourse_root_option_probes"] > 0
    assert work["root_option_probes"] == len(options)
    assert not work["incumbent_root_option_probes"] and not work["override_root_option_probes"]
    assert planner.rng.getstate() == rng_before and planner.decision_counter == 7
    summary = summarize([result])
    assert summary["delta_T_mean"]["raw_signs"]["positive"] == 1
    assert summary["original_helps_mean_but_hurts_any_class_900_tail"] == 1
