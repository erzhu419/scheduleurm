"""The mean guard filters outer J ordering without changing native continuations."""
from collections import Counter
from dataclasses import asdict, fields
from fractions import Fraction
import json
from unittest.mock import patch

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundRootScores,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_fifo_waiting_penalty_development import action_comparison
from algorithm.fifo_mean_guard_lane_rollout import FIFOMeanGuardLaneRollout, ParallelFIFOMeanGuardLaneRollout
from algorithm.fifo_waiting_penalty_lane_rollout import FIFOBackgroundRootScores, WAITING_WORK
from skill.tests.test_fifo_waiting_penalty_lane_rollout import (
    CAP, SAMPLES, OPTIONS, NATIVE_WORK, base_scores, direct_context, initialize, semantic,
)


def test_mean_membership_is_separate_from_original_risk_and_j_ordering():
    from algorithm.experiments.fifo_mean_guard_checks import check_mean_guard_scores

    options = (
        FrameOption("single", (1,), 1., ((1.,),)),
        FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
        FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
        FrameOption("slow-pair", (2,), 2., ((2., 2.),)),
    )
    for mode in ("reject_trade", "allowed_trade", "old_illegal", "tie"):
        planner = FIFOMeanGuardLaneRollout(options, "fifo-mean:" + mode, SAMPLES, CAP)
        original = base_scores({0: 30, 1: 20, 2: 40, 3: 35},
            {0: 90, 1: 80, 2: 100, 3: 101 if mode == "old_illegal" else 95})
        if mode == "reject_trade":
            costs, penalties = {0: 26, 1: 25, 2: 30, 3: 29}, {0: 0, 1: 2, 2: 1, 3: 0}
            chosen, allowed, filtered = 1, 1, 3
        elif mode == "tie":
            costs, penalties = {0: 25, 1: 25, 2: 28, 3: 25}, dict.fromkeys(range(4), 2)
            chosen, allowed, filtered = 1, 3, 1
        else:
            # root0 minimizes S1; root3 is still no worse than BG, and has better J.
            costs, penalties = {0: 20, 1: 25, 2: 28, 3: 24}, {0: 30, 1: 10, 2: 0, 3: 0}
            chosen = 1 if mode == "old_illegal" else 3
            allowed, filtered = (2, 1) if mode == "old_illegal" else (3, 1)
        genuine = base_scores(costs, {0: 90, 1: 100, 2: 110, 3: 101})
        penalties = {root: Fraction(value) for root, value in penalties.items()}
        scored = FIFOBackgroundRootScores(
            **{field.name: getattr(genuine, field.name) for field in fields(BackgroundRootScores)},
            waiting_penalties=penalties,
            augmented_costs={root: genuine.costs[root] + penalties[root] for root in costs})

        def charged_scores(*context):
            planner.fifo_waiting_work.update({name: genuine.flow_evaluations for name in WAITING_WORK[:4]})
            planner.fifo_waiting_work["augmented_score_additions"] += genuine.candidate_count
            return scored

        with patch.object(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", return_value=original), \
                patch.object(planner, "score_background_roots", side_effect=charged_scores):
            decision = planner._choose(options, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),))
        assert decision.incumbent_root_index == 1 and decision.incumbent_maxweight_root_index == 2
        risk_count = 3 if mode == "old_illegal" else 4
        assert decision.risk_feasible_candidates == decision.incumbent_risk_feasible_candidates == risk_count
        assert decision.risk_filtered_candidates == decision.incumbent_risk_filtered_candidates == 4 - risk_count
        assert decision.mean_guard_comparisons == risk_count
        assert decision.mean_guard_allowed == allowed and decision.mean_guard_filtered == filtered
        assert allowed + filtered == risk_count
        assert decision.incumbent_lookahead_ordering_comparisons == 4 + allowed - 2
        assert decision.incumbent_lookahead_risk_comparisons == 0
        assert decision.selection.option_indices == (chosen,)
        assert decision.improved_incumbent == (mode == "allowed_trade")
        assert decision.chosen_predicted_completion_s == float(genuine.costs[chosen] / SAMPLES)
        assert decision.chosen_predicted_completion_s <= decision.baseline_predicted_completion_s
        assert decision.chosen_predicted_augmented_score_s == float(scored.augmented_costs[chosen] / SAMPLES)
        if mode == "reject_trade":
            assert scored.augmented_costs[0] < scored.augmented_costs[1]
            assert genuine.costs[0] > genuine.costs[1]
            assert decision.unconstrained_root_index == 0 and decision.risk_changed_unconstrained_root
        elif mode == "allowed_trade":
            assert genuine.costs[0] < genuine.costs[chosen] <= genuine.costs[1]
            # No new S2<=BG gate is added by the mean guard.
            assert decision.chosen_predicted_squared_flow_s2 > decision.baseline_predicted_squared_flow_s2
        elif mode == "old_illegal":
            assert decision.unconstrained_root_index == 3 and decision.risk_changed_unconstrained_root
        info = asdict(decision)
        assert check_mean_guard_scores(info)
        assert not check_mean_guard_scores({**info, "mean_guard_comparisons": risk_count + 1})
        if mode == "allowed_trade":
            # Still improves J: only the genuine-S1 mean constraint rejects this.
            bad_s1 = info["baseline_predicted_completion_s"] + .125
            bad_info = {**info, "chosen_predicted_completion_s": bad_s1,
                "chosen_predicted_augmented_score_s": bad_s1 + info["chosen_predicted_waiting_penalty_s"]}
            assert bad_info["chosen_predicted_augmented_score_s"] < info["baseline_predicted_augmented_score_s"]
            assert not check_mean_guard_scores(bad_info)


def test_serial_parallel_complete_native_decision_state_and_checker():
    from algorithm.experiments.fifo_mean_guard_checks import check_rollout

    serial = FIFOMeanGuardLaneRollout(OPTIONS, "fifo-mean-parallel", SAMPLES, CAP)
    with ParallelFIFOMeanGuardLaneRollout(OPTIONS, "fifo-mean-parallel", SAMPLES, CAP, workers=2) as parallel:
        context = (OPTIONS, (2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
            ((2.1, .3), (1.2,), (.1,)), (120, 70, 80))
        expected, actual = serial(*context), parallel(*context)
        assert actual == expected
        assert actual.chosen_predicted_waiting_penalty_s > 0
        assert actual.chosen_predicted_completion_s <= actual.baseline_predicted_completion_s
        assert actual.mean_guard_comparisons == actual.risk_feasible_candidates > 0
        assert actual.mean_guard_allowed + actual.mean_guard_filtered == actual.mean_guard_comparisons
        assert semantic(serial) == semantic(parallel)
        for name in NATIVE_WORK:
            assert Counter(getattr(serial, name)) == Counter(getattr(parallel, name)), name
        assert Counter(serial.fifo_waiting_work) == Counter(parallel.fifo_waiting_work)
        model = FrameModel("toy", "fifo-mean-parallel", 2, ("a", "b", "c"), (1., 1., 1.), OPTIONS)
        frame = {"start": context[3], "queue": context[1], "available_before": context[2],
            "lane_ids": (context[4],), "waiting_ages": context[5], "observed_arrival_counts": context[6],
            "option_indices": actual.selection.option_indices, "rollout": asdict(actual)}
        result = {"frames": [frame]}
        assert check_rollout(result, model, {"planning_samples": SAMPLES, "planning_event_cap": CAP}, parallel)
        comparison = action_comparison(result, json.loads(json.dumps(result)))
        assert comparison["first_divergence"] is None and comparison["completed_new_stage_frames"] == 1
        bad_info = {**frame["rollout"], "mean_guard_allowed": actual.mean_guard_allowed + 1}
        assert not check_rollout({"frames": [{**frame, "rollout": bad_info}]}, model,
            {"planning_samples": SAMPLES, "planning_event_cap": CAP}, parallel)
        work = parallel.reuse_work["parallel_root_work"]
        assert work["submitted_chunks"] == work["completed_chunks"] == work["included_chunks"]
        assert work["discarded_chunks"] == 0


def test_native_overflow_keeps_partial_fees_and_skips_mean_filtering():
    serial = FIFOMeanGuardLaneRollout(OPTIONS, "parallel-overflow", SAMPLES, 2)
    with ParallelFIFOMeanGuardLaneRollout(OPTIONS, "parallel-overflow", SAMPLES, 2, workers=2) as parallel:
        for planner in (serial, parallel):
            initialize(planner, (.5, .4, .3), (500, 400, 300))
        context = (*direct_context()[:-1],
            (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * SAMPLES)
        expected, actual = serial.score_background_roots(*context), parallel.score_background_roots(*context)
        assert actual == expected and actual.background_rollout_overflow
        assert actual.flow_evaluations > 0 and any(value > 0 for value in actual.waiting_penalties.values())
        assert semantic(serial) == semantic(parallel)
        for name in NATIVE_WORK:
            assert Counter(getattr(serial, name)) == Counter(getattr(parallel, name)), name
        partial_work = Counter(serial.fifo_waiting_work)
        assert partial_work == Counter(parallel.fifo_waiting_work)
        assert all(partial_work[name] == actual.flow_evaluations for name in WAITING_WORK[:4])
        work = parallel.reuse_work["parallel_root_work"]
        assert 0 < work["included_chunks"] < work["completed_chunks"]
        assert work["included_chunks"] + work["discarded_chunks"] == work["completed_chunks"]
        original = base_scores({0: 30, 1: 20, 2: 40, 3: 35, 4: 32},
            {0: 90, 1: 80, 2: 100, 3: 95, 4: 92})
        for planner in (serial, parallel):
            planner.arrival_rates = (0., 0., 0.)
            def cached_partial(*context):
                planner.fifo_waiting_work.update(partial_work)
                return actual
            with patch.object(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", return_value=original), \
                    patch.object(planner, "score_background_roots", side_effect=cached_partial):
                decision = planner._choose(OPTIONS, (2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
                    ((2.1, .3), (1.2,), (.1,)))
            assert decision.planning_overflow and decision.background_rollout_overflow
            assert decision.selection.option_indices == (decision.incumbent_root_index,)
            assert not decision.improved_incumbent
            assert decision.mean_guard_comparisons == decision.mean_guard_allowed == decision.mean_guard_filtered == 0
            assert decision.flow_evaluations == actual.flow_evaluations > 0
            assert decision.background_planning_events == actual.background_planning_events > 0
            assert Counter(decision.fifo_waiting_work) == partial_work


if __name__ == "__main__":
    test_mean_membership_is_separate_from_original_risk_and_j_ordering()
    test_serial_parallel_complete_native_decision_state_and_checker()
    test_native_overflow_keeps_partial_fees_and_skips_mean_filtering()
    print("3 passed", flush=True)
