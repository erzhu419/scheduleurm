from fractions import Fraction
import pytest

from algorithm.class_tail_continuous_lane_rollout import ClassTailContinuousLaneRollout
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.aligned_hardware_class_tail_continuous import build_protocol


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))


def test_class_gate_blocks_risk_transfer_hidden_by_aggregate():
    state = (OPTIONS, (2, 1), (1000., 1003.), 1000., 0, ((0., 0.), (10.,)), (0, 0))
    aggregate = ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same")(*state)
    candidate = ClassTailContinuousLaneRollout(OPTIONS, "same")(*state)
    assert aggregate.changed and aggregate.selection.option_indices == (2,)
    assert aggregate.chosen_predicted_squared_flow_s2 < aggregate.baseline_predicted_squared_flow_s2
    assert not candidate.changed and candidate.selection.option_indices == (1,)
    assert candidate.baseline_class_squared_flow_s2 == (18., 196.)
    assert candidate.chosen_class_squared_flow_s2 == candidate.baseline_class_squared_flow_s2
    assert candidate.risk_changed_unconstrained_root


def test_class_sums_preserve_exact_global_scores_and_original_work():
    baseline = ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same")
    candidate = ClassTailContinuousLaneRollout(OPTIONS, "same")
    state = ((2, 1), (1000., 1003.), 1000., 0, ((3., 1.), (10.,)))
    paths = ((), ((.5, 0), (1., 1)))
    old = baseline.score_age_roots(*state, paths)
    new = candidate.score_age_roots(*state, paths)
    assert new.costs == old.costs and new.squared_costs == old.squared_costs
    assert new.rollout_calls == old.rollout_calls and new.flow_evaluations == old.flow_evaluations
    assert all(sum(new.class_squared_costs[root], Fraction(0)) == old.squared_costs[root] for root in old.costs)
    assert new.class_squared_accumulations == new.flow_evaluations + new.candidate_count * 2 * 2


def test_overflow_keeps_whole_decision_mw_without_class_scoring():
    candidate = ClassTailContinuousLaneRollout(OPTIONS, "overflow")
    result = candidate(OPTIONS, (2, 1), (.1, 3.1), .1, 0, ((0., 0.), (0.,)), (100, 100))
    assert result.planning_overflow and not result.changed
    assert result.class_squared_accumulations == result.class_risk_comparisons == 0
    assert not result.chosen_class_squared_flow_s2


def test_control_accepts_engine_context_but_ignores_observed_counts():
    candidate = ClassTailContinuousLaneRollout(OPTIONS, "control")
    state = (OPTIONS, (2, 1), (1500., 1503.), 1500., 0, ((0., 0.), (10.,)))
    assert candidate.maxweight(*state, (150, 75)) == candidate.maxweight(*state)
    result = candidate(*state, (150, 75))
    assert result.rate_exposure_s == 1500. and result.window_s == 3.
    assert result.estimated_arrival_rates == (.1, .05)
    assert all(a <= b for a, b in zip(result.chosen_class_squared_flow_s2, result.baseline_class_squared_flow_s2))
    with pytest.raises(TypeError):
        ClassTailContinuousLaneRollout(OPTIONS, "x", arrival_horizon=600.)


def test_protocol_keeps_previous_scope_and_budget():
    protocol = build_protocol("development")
    assert len(protocol["tapes"]) == 24
    assert protocol["birth_cohort"] == [300., 900.]
    assert protocol["planning_samples"] == 8 and protocol["planning_event_cap"] == 128
    assert protocol["performance"] == {"min_gain": .01, "group_secondary_tolerance": .05}
