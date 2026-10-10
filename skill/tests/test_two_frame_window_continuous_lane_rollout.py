"""The two-frame treatment changes arrival exposure, never service or root rules."""
from dataclasses import fields
import inspect

import algorithm.age_tail_constrained_lane_rollout as parent_module
import algorithm.two_frame_window_continuous_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals, sample_planning_arrivals
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.two_frame_window_continuous_lane_rollout import TwoFrameWindowContinuousLaneRollout


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))


class FourSecondGaps:
    def expovariate(self, rate):
        return 4.


def test_actual_sampler_uses_two_frames_after_evaluation_boundaries(monkeypatch):
    calls = []

    def sampled(rates, window, rng, cap):
        result = sample_planning_arrivals(rates, window, rng, cap)
        calls.append((window, cap, result))
        return result

    monkeypatch.setattr(module, "sample_planning_arrivals", sampled)
    candidate = TwoFrameWindowContinuousLaneRollout(OPTIONS, "two-frame-test")
    candidate.rng = FourSecondGaps()
    for now in (900., 1200., 1500.):
        actual = candidate(OPTIONS, (2, 1), (now, now + 3), now, 0,
                           ((10., 10.), (0.,)), (150, 75))
        assert actual.window_s == 6. and actual.sample_count == 8
        assert actual.planning_events == 16 and actual.planning_random_draws == 32
    assert len(calls) == 24
    assert all(window == 6. and cap == 128 for window, cap, _ in calls)
    assert all([event[0] for event in path.events] == [4., 4.]
               for _, _, path in calls)
    assert candidate.maximum_duration == 3.
    assert tuple(map(float, candidate.durations)) == (2., 3., 1.)


def test_rates_use_all_cumulative_births_and_current_time_without_stop_input():
    assert "arrival_horizon" not in inspect.signature(TwoFrameWindowContinuousLaneRollout).parameters
    candidate = TwoFrameWindowContinuousLaneRollout(OPTIONS, "rates-test")
    actual = candidate(OPTIONS, (1, 0), (1500., 1503.), 1500., 0,
                       ((4.,), ()), (150, 75))
    assert actual.observed_arrival_counts == (150, 75)
    assert actual.estimated_arrival_rates == (.1, .05)
    assert actual.rate_exposure_s == 1500.
    assert actual.rate_estimation_coordinate_reads == actual.rate_estimation_divisions == 2
    zero = candidate(OPTIONS, (2, 1), (0., 3.), 0., 0,
                     ((0., 0.), (0.,)), (2, 1))
    assert zero.estimated_arrival_rates == (0., 0.)
    assert zero.rate_estimation_divisions == zero.planning_random_draws == 0


def test_same_fixed_paths_reproduce_every_parent_field_except_window(monkeypatch):
    # Fix the population independently of the sampler's requested window, so
    # any changed score/gate/selection/accounting exposes an unintended change.
    paths = tuple(PlanningArrivals(((.5, 1, 0), (4.5, 0, 0)), False, 4, 2)
                  for _ in range(8))
    sampled_paths = []

    def sampled(rates, window, rng, cap):
        path = paths[len(sampled_paths) % len(paths)]
        sampled_paths.append(path)
        return path

    monkeypatch.setattr(module, "sample_planning_arrivals", sampled)
    monkeypatch.setattr(parent_module, "sample_planning_arrivals", sampled)
    current = TwoFrameWindowContinuousLaneRollout(OPTIONS, "same-paths")
    original = ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same-paths")
    args = (OPTIONS, (2, 1), (1000., 1003.), 1000., 0,
            ((8., 1.), (20.,)), (150, 75))
    actual, expected = current(*args), original(*args)
    for field in fields(actual):
        if field.name != "window_s":
            assert getattr(actual, field.name) == getattr(expected, field.name)
    assert actual.window_s == 6. and expected.window_s == 3.
    assert sampled_paths[:8] == sampled_paths[8:] == list(paths)
    assert current.score_age_roots(*args[1:6], paths) == original.score_age_roots(*args[1:6], paths)
    assert current.rng.getstate() == original.rng.getstate()


class LaterOverflowGaps:
    def __init__(self):
        self.draws = 0

    def expovariate(self, rate):
        self.draws += 1
        return 100. if self.draws <= 2 else .1


def test_later_sample_overflow_discards_entire_scoring_call(monkeypatch):
    candidate = TwoFrameWindowContinuousLaneRollout(OPTIONS, "overflow", event_cap=2)
    candidate.rng = LaterOverflowGaps()

    def forbidden(*args):
        raise AssertionError("any overflow must bypass all scoring")

    monkeypatch.setattr(candidate, "score_age_roots", forbidden)
    args = (OPTIONS, (2, 1), (100., 103.), 100., 0,
            ((10., 10.), (0.,)), (100, 100))
    actual = candidate(*args)
    assert actual.selection == candidate.maxweight(*args).selection
    assert actual.planning_overflow and actual.sample_count == 2
    assert actual.planning_events == 3 and actual.planning_random_draws == 5
    assert actual.planning_rate_reads == 3
    assert not actual.changed
    assert actual.candidate_count == actual.rollout_calls == actual.waiting_age_reads == 0
    assert actual.flow_evaluations == 0 and actual.baseline_predicted_squared_flow_s2 is None


def test_mw_and_queue_envelope_preserve_catalogue_duration_and_rng():
    current = TwoFrameWindowContinuousLaneRollout(OPTIONS, "mw-test")
    original = ContinuousArrivalAgeTailLaneRollout(OPTIONS, "mw-test")
    state = current.rng.getstate()
    for queue, ages in (((0, 0), ((), ())), ((4, 1), ((9., 8., 7., 6.), (2.,)))):
        args = (OPTIONS, queue, (1000., 1003.), 1000., 0, ages, (150, 75))
        assert current(*args) == original(*args)
        assert current.maxweight(*args) == original.maxweight(*args)
    args = (OPTIONS, (2, 1), (1000., 1003.), 1000., 0, ((9., 2.), (1.,)), (150, 75))
    assert current.maxweight(*args) == original.maxweight(*args)
    assert current.maxweight(*args).selection.duration == 3.
    assert current.rng.getstate() == state == original.rng.getstate()
    assert current.maximum_duration == original.maximum_duration == 3.


def test_aggregate_s2_gate_and_mw_ties_remain_parent_rules():
    candidate = TwoFrameWindowContinuousLaneRollout(OPTIONS, "s2-test")
    actual = candidate(OPTIONS, (2, 1), (1000., 1003.), 1000., 0,
                       ((10., 10.), (0.,)), (0, 0))
    assert actual.unconstrained_root_index == 2
    assert actual.risk_changed_unconstrained_root and actual.risk_filtered_candidates == 2
    assert actual.selection.option_indices == (1,) and not actual.changed
    tied = (FrameOption("first", (1,), 1., ((.5,),)),
            FrameOption("second", (1,), 1., ((.5,),)))
    decision = TwoFrameWindowContinuousLaneRollout(tied, "ties")(
        tied, (1,), (1000.,), 1000., 0, ((12.,),), (0,))
    assert decision.selection.option_indices == (0,) and not decision.changed
    assert decision.selection.canonical_exact
