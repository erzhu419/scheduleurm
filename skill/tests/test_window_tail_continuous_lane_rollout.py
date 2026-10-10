from fractions import Fraction
import pytest

from algorithm.class_tail_continuous_lane_rollout import ClassTailContinuousLaneRollout
from algorithm.window_tail_continuous_lane_rollout import WindowTailContinuousLaneRollout
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.aligned_hardware_window_tail_continuous import build_protocol

OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))


def test_counts_use_completion_boundary_not_waiting_age_or_admission():
    planner = WindowTailContinuousLaneRollout(OPTIONS, "test")
    state = ((2, 1), (1000., 1003.), 1000., 0, ((0., 0.), (10.,)))
    scores = planner.score_age_roots(*state, ((), ()))
    # Batch a finishes exactly at tau_max=3; b finishes at4.
    assert scores.class_unfinished_counts[1] == (0, 2)
    # b-first finishes b at1, both a jobs at4, regardless of b's past age.
    assert scores.class_unfinished_counts[2] == (4, 0)


def test_scores_keep_global_S1_S2_and_continuation_work_identical():
    state = ((2, 1), (1000., 1003.), 1000., 0, ((3., 1.), (10.,)))
    paths = ((), ((.5, 0), (1., 1)))
    old = ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same").score_age_roots(*state, paths)
    new = WindowTailContinuousLaneRollout(OPTIONS, "same").score_age_roots(*state, paths)
    assert new.costs == old.costs and new.squared_costs == old.squared_costs
    assert new.flow_evaluations == old.flow_evaluations and new.rollout_calls == old.rollout_calls
    assert new.unfinished_accumulations == new.flow_evaluations + new.candidate_count * 2 * 2


def test_window_counts_allow_bounded_class_S2_tradeoff():
    options = OPTIONS + (FrameOption("slow_a1", (1, 0), 10., ((10.,), ())),)
    state = (options, (2, 1), (1000., 1003.), 1000., 0, ((0., 0.), (10.,)), (0, 0))
    strict = ClassTailContinuousLaneRollout(options, "same")(*state)
    candidate = WindowTailContinuousLaneRollout(options, "same")(*state)
    assert not strict.changed
    assert candidate.changed and candidate.selection.option_indices == (2,)
    assert candidate.chosen_class_unfinished_counts == candidate.baseline_class_unfinished_counts == (0, 0)
    assert candidate.chosen_predicted_squared_flow_s2 <= candidate.baseline_predicted_squared_flow_s2
    assert candidate.unfinished_completion_checks == candidate.flow_evaluations


def test_overflow_keeps_entire_mw_decision_without_scoring():
    candidate = WindowTailContinuousLaneRollout(OPTIONS, "overflow")
    result = candidate(OPTIONS, (2, 1), (.1, 3.1), .1, 0, ((0., 0.), (0.,)), (100, 100))
    assert result.planning_overflow and not result.changed
    assert not result.chosen_class_unfinished_counts
    assert result.unfinished_accumulations == result.unfinished_completion_checks == 0
    assert result.aggregate_risk_comparisons == result.class_unfinished_comparisons == 0


def test_no_stop_input_and_original_frozen_scope():
    candidate = WindowTailContinuousLaneRollout(OPTIONS, "test")
    result = candidate(OPTIONS, (2, 1), (1500., 1503.), 1500., 0, ((0., 0.), (10.,)), (150, 75))
    assert result.window_s == 3. and result.rate_exposure_s == 1500.
    assert result.estimated_arrival_rates == (.1, .05)
    with pytest.raises(TypeError):
        WindowTailContinuousLaneRollout(OPTIONS, "test", arrival_horizon=600.)
    protocol = build_protocol("development")
    assert len(protocol["tapes"]) == 24 and protocol["birth_cohort"] == [300., 900.]
    assert protocol["planning_samples"] == 8 and protocol["planning_event_cap"] == 128
    assert protocol["performance"] == {"min_gain": .01, "group_secondary_tolerance": .05}
