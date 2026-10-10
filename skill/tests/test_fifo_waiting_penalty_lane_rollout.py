"""Keep FIFO waiting penalties separate from flow, risk, native state and fees."""
from collections import Counter
from dataclasses import asdict, fields
from fractions import Fraction
import json
from unittest.mock import patch

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundRootScores,
)
from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_fifo_waiting_penalty_development import action_comparison
from algorithm.experiments.fifo_waiting_penalty_checks import check_fifo_scores, check_rollout
from algorithm.fifo_waiting_penalty_lane_rollout import (
    FIFOBackgroundRootScores, FIFOWaitingPenaltyCurrentPolicyLaneRollout,
    waiting_penalty,
)
from algorithm.parallel_fifo_waiting_penalty_lane_rollout import (
    ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout,
)


SAMPLES, CAP = 8, 128
OPTIONS = (
    FrameOption("a1", (1, 0, 0), 1., ((1.,), (), ())),
    FrameOption("b1", (0, 1, 0), 1.25, ((), (1.25,), ())),
    FrameOption("c1", (0, 0, 1), 1.5, ((), (), (1.5,))),
    FrameOption("loaded-ab", (1, 1, 0), 1.75, ((.5,), (1.75,), ()), loaded=True),
    FrameOption("a2", (2, 0, 0), 1.5, ((1.5, 1.5), (), ())),
)
NATIVE_WORK = ("_recourse_work", "_two_step_work", "_inner_current_work", "_inner_current_fallbacks")


def semantic(planner):
    return (planner.rng.getstate(), planner.decision_counter, planner.arrival_rates,
        getattr(planner, "_outer_observed_counts", None),
        semantic(planner.inner_planner) if hasattr(planner, "inner_planner") else None)


def initialize(planner, rates=(.12, .07, .08), counts=(120, 70, 80)):
    planner.arrival_rates = rates
    planner._outer_observed_counts = counts
    planner.decision_counter = 7


def direct_context():
    patterns = (((.2, 2), (.9, 1)), ((.1, 0), (.7, 2)), ((.6, 1),), ())
    paths = tuple(PlanningArrivals(events, False, 0, 0) for events in patterns * 2)
    return ((2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
        ((2.1, .3), (1.2,), (.1,)), paths)


def base_scores(costs, squared):
    values = {field.name: 0 for field in fields(BackgroundRootScores)}
    values.update(costs={k: Fraction(v) for k, v in costs.items()},
        squared_costs={k: Fraction(v) for k, v in squared.items()},
        candidate_count=len(costs), sample_count=SAMPLES,
        root_option_probes=len(costs), waiting_age_reads=3,
        flow_evaluations=3 * SAMPLES, background_rollout_overflow=False)
    return BackgroundRootScores(**values)


def test_hand_fifo_quadratic_waiting_excludes_completion_offsets():
    # Two jobs born at 0/1, admitted at 2/5: W=2/4, P=3/4.
    assert waiting_penalty(Fraction(2), Fraction(0), Fraction(4)) == 1
    assert waiting_penalty(Fraction(5), Fraction(1), Fraction(4)) == 4
    assert waiting_penalty(Fraction(0), Fraction(-4), Fraction(4)) == 4
    assert Fraction(2 + 4 + 3 + 4) + 1 + 4 == 18

    totals = []
    for offsets in ((1., 2.), (4., 4.)):
        options = (FrameOption("single", (1,), 4., ((4.,),)),
            FrameOption("pair", (2,), 4., (offsets,)))
        planner = FIFOWaitingPenaltyCurrentPolicyLaneRollout(options, "fifo-hand", SAMPLES, CAP)
        planner.arrival_rates = (0.,)
        planner._outer_observed_counts = (0,)
        scored = planner.score_background_roots((4,), (1000.,), 1000., 0,
            ((4., 2., 1., 0.),), (PlanningArrivals((), False, 0, 0),) * SAMPLES)
        # FIFO W=4,2,5,4 in both runs; each pair reserves its full four seconds.
        assert scored.waiting_penalties[1] == SAMPLES * Fraction(61, 4)
        assert scored.augmented_costs[1] == scored.costs[1] + scored.waiting_penalties[1]
        assert scored.flow_evaluations == 2 * 4 * SAMPLES
        totals.append(scored.costs[1])
    assert totals == [SAMPLES * 21, SAMPLES * 31]


def test_original_risk_bg_ties_and_genuine_flow_metadata():
    options = (
        FrameOption("single", (1,), 1., ((1.,),)),
        FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
        FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
    )
    for mode in ("zero_tie", "equal_penalty_tie", "improve", "old_illegal"):
        planner = FIFOWaitingPenaltyCurrentPolicyLaneRollout(options, "fifo-risk:" + mode, SAMPLES, CAP)
        old_squared = {0: 101 if mode == "old_illegal" else 90, 1: 80, 2: 100}
        original = base_scores({0: 30, 1: 20, 2: 40}, old_squared)
        if mode in ("zero_tie", "equal_penalty_tie"):
            flow = {0: 25, 1: 25, 2: 30}
            penalties = dict.fromkeys(flow, Fraction(0 if mode == "zero_tie" else 2))
        else:
            # J favors root0 despite S1 and new S2 both exceeding BG's.
            flow, penalties = {0: 26, 1: 25, 2: 30}, {0: Fraction(0), 1: Fraction(2), 2: Fraction(1)}
        genuine = base_scores(flow, {0: 101, 1: 100, 2: 110})
        scored = FIFOBackgroundRootScores(
            **{field.name: getattr(genuine, field.name) for field in fields(BackgroundRootScores)},
            waiting_penalties=penalties,
            augmented_costs={root: genuine.costs[root] + penalty for root, penalty in penalties.items()})
        def charged_scores(*context):
            planner.fifo_waiting_work.update({name: genuine.flow_evaluations for name in (
                "tagged_waiting_subtractions", "tagged_waiting_squares", "tagged_tau_divisions", "tagged_penalty_additions")})
            planner.fifo_waiting_work["augmented_score_additions"] += genuine.candidate_count
            return scored
        with patch.object(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", return_value=original), \
                patch.object(planner, "score_background_roots", side_effect=charged_scores):
            decision = planner._choose(options, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),))
        assert decision.incumbent_root_index == 1 and decision.incumbent_maxweight_root_index == 2
        assert decision.risk_feasible_candidates == decision.incumbent_risk_feasible_candidates
        assert decision.incumbent_lookahead_risk_comparisons == 0
        chosen = 0 if mode == "improve" else 1
        assert decision.selection.option_indices == (chosen,)
        assert decision.improved_incumbent == (mode == "improve")
        assert decision.baseline_predicted_completion_s == float(genuine.costs[1] / SAMPLES)
        assert decision.chosen_predicted_completion_s == float(genuine.costs[chosen] / SAMPLES)
        assert decision.chosen_predicted_squared_flow_s2 == float(genuine.squared_costs[chosen] / SAMPLES)
        assert decision.baseline_predicted_waiting_penalty_s == float(penalties[1] / SAMPLES)
        assert decision.chosen_predicted_waiting_penalty_s == float(penalties[chosen] / SAMPLES)
        assert decision.baseline_predicted_augmented_score_s == float(scored.augmented_costs[1] / SAMPLES)
        assert decision.chosen_predicted_augmented_score_s == float(scored.augmented_costs[chosen] / SAMPLES)
        if mode == "old_illegal":
            assert decision.unconstrained_root_index == 0 and decision.risk_changed_unconstrained_root
            assert decision.risk_filtered_candidates == decision.incumbent_risk_filtered_candidates == 1
        elif mode == "improve":
            assert decision.chosen_predicted_completion_s > decision.baseline_predicted_completion_s
            assert decision.chosen_predicted_squared_flow_s2 > decision.baseline_predicted_squared_flow_s2
        info = asdict(decision)
        assert check_fifo_scores(info)
        bad_score = {**info, "chosen_predicted_augmented_score_s": info["chosen_predicted_augmented_score_s"] + .01}
        assert not check_fifo_scores(bad_score)
        bad_fee = {**info, "fifo_waiting_work": {**info["fifo_waiting_work"],
            "tagged_waiting_squares": info["fifo_waiting_work"]["tagged_waiting_squares"] + 1}}
        assert not check_fifo_scores(bad_fee)


def test_direct_native_scorer_changes_only_metric_and_preserves_child_state():
    parent = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "fifo-direct", SAMPLES, CAP)
    candidate = FIFOWaitingPenaltyCurrentPolicyLaneRollout(OPTIONS, "fifo-direct", SAMPLES, CAP)
    for planner in (parent, candidate):
        initialize(planner)
    before = candidate.rng.getstate()
    expected = parent.score_background_roots(*direct_context())
    actual = candidate.score_background_roots(*direct_context())
    for field in fields(BackgroundRootScores):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name
    assert not actual.background_rollout_overflow
    assert any(value > 0 for value in actual.waiting_penalties.values())
    assert all(actual.augmented_costs[root] == value + actual.waiting_penalties[root]
        for root, value in actual.costs.items())
    assert semantic(candidate) == semantic(parent)
    assert candidate.rng.getstate() == before and candidate.decision_counter == 7
    for name in NATIVE_WORK:
        assert Counter(getattr(candidate, name)) == Counter(getattr(parent, name)), name
    assert candidate._two_step_work["recourse_first_calls"] > 0
    assert candidate._two_step_work["recourse_second_calls"] > 0
    work = candidate.fifo_waiting_work
    for name in ("tagged_waiting_subtractions", "tagged_waiting_squares", "tagged_tau_divisions", "tagged_penalty_additions"):
        assert work[name] == actual.flow_evaluations, name
    assert work["augmented_score_additions"] == actual.candidate_count


def test_parallel_full_decision_and_native_overflow_retain_exact_state_and_partial_work():
    serial = FIFOWaitingPenaltyCurrentPolicyLaneRollout(OPTIONS, "fifo-parallel", SAMPLES, CAP)
    with ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout(OPTIONS, "fifo-parallel", SAMPLES, CAP, workers=2) as parallel:
        context = (OPTIONS, (2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
            ((2.1, .3), (1.2,), (.1,)), (120, 70, 80))
        expected, actual = serial(*context), parallel(*context)
        assert actual == expected
        assert actual.chosen_predicted_waiting_penalty_s > 0
        model = FrameModel("toy", "fifo-parallel", 2, ("a", "b", "c"), (1., 1., 1.), OPTIONS)
        frame = {"start": context[3], "queue": context[1], "available_before": context[2],
            "lane_ids": (context[4],), "waiting_ages": context[5], "observed_arrival_counts": context[6],
            "option_indices": actual.selection.option_indices, "rollout": asdict(actual)}
        assert check_rollout({"frames": [frame]}, model,
            {"planning_samples": SAMPLES, "planning_event_cap": CAP}, parallel)
        comparison = action_comparison({"frames": [frame]}, json.loads(json.dumps({"frames": [frame]})))
        assert comparison["first_divergence"] is None
        assert comparison["completed_new_stage_frames"] == 1
        assert semantic(parallel) == semantic(serial)
        for name in NATIVE_WORK:
            assert Counter(getattr(parallel, name)) == Counter(getattr(serial, name)), name
        assert Counter(parallel.fifo_waiting_work) == Counter(serial.fifo_waiting_work)
        work = parallel.reuse_work["parallel_root_work"]
        assert work["submitted_chunks"] == work["completed_chunks"] == work["included_chunks"]
        assert work["discarded_chunks"] == 0

    serial = FIFOWaitingPenaltyCurrentPolicyLaneRollout(OPTIONS, "parallel-overflow", SAMPLES, 2)
    with ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout(OPTIONS, "parallel-overflow", SAMPLES, 2, workers=2) as parallel:
        for planner in (serial, parallel):
            initialize(planner, (.5, .4, .3), (500, 400, 300))
        context = (*direct_context()[:-1],
            (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * SAMPLES)
        expected, actual = serial.score_background_roots(*context), parallel.score_background_roots(*context)
        assert actual == expected and actual.background_rollout_overflow
        assert actual.flow_evaluations > 0 and any(value > 0 for value in actual.waiting_penalties.values())
        assert semantic(parallel) == semantic(serial)
        for name in NATIVE_WORK:
            assert Counter(getattr(parallel, name)) == Counter(getattr(serial, name)), name
        assert Counter(parallel.fifo_waiting_work) == Counter(serial.fifo_waiting_work)
        work = parallel.reuse_work["parallel_root_work"]
        assert 0 < work["included_chunks"] < work["completed_chunks"]
        assert work["included_chunks"] + work["discarded_chunks"] == work["completed_chunks"]
        # A genuine cap-crossing score must retain the verified ordinary BG root.
        partial_waiting_work = Counter(serial.fifo_waiting_work)
        original = base_scores({0: 30, 1: 20, 2: 40, 3: 35, 4: 32}, {0: 90, 1: 80, 2: 100, 3: 95, 4: 92})
        for planner in (serial, parallel):
            planner.arrival_rates = (0., 0., 0.)
            def cached_partial(*context):
                planner.fifo_waiting_work.update(partial_waiting_work)
                return actual
            with patch.object(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", return_value=original), \
                    patch.object(planner, "score_background_roots", side_effect=cached_partial):
                decision = planner._choose(OPTIONS, (2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
                    ((2.1, .3), (1.2,), (.1,)))
            assert decision.planning_overflow and decision.background_rollout_overflow
            assert decision.selection.option_indices == (decision.incumbent_root_index,)
            assert not decision.improved_incumbent
            assert decision.flow_evaluations == actual.flow_evaluations > 0
            assert decision.background_planning_events == actual.background_planning_events > 0
            assert Counter(decision.fifo_waiting_work) == partial_waiting_work


if __name__ == "__main__":
    test_hand_fifo_quadratic_waiting_excludes_completion_offsets()
    test_original_risk_bg_ties_and_genuine_flow_metadata()
    test_direct_native_scorer_changes_only_metric_and_preserves_child_state()
    test_parallel_full_decision_and_native_overflow_retain_exact_state_and_partial_work()
    print("4 passed", flush=True)
