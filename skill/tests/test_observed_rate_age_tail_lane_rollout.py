"""Observed-birth MLE, frozen parent planning and no true-law input."""
from dataclasses import fields

import pytest

from algorithm.age_tail_constrained_lane_rollout import AgeTailConstrainedLaneRollout, AgeTailDecision
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory
from algorithm.lane_refill_simple_controls import select_single_lane_control
from algorithm.observed_rate_age_tail_lane_rollout import ObservedRateAgeTailLaneRollout


def options():
    return (FrameOption("a1", (1, 0), 2., ((2.,), ())),
            FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
            FrameOption("b1", (0, 1), 1., ((), (1.,))))


def planner():
    return ObservedRateAgeTailLaneRollout(options(), 600., "observed-plan-v1:toy:online")


def test_constructor_accepts_no_true_arrival_rates_or_load():
    candidate = planner()
    assert candidate.arrival_rates == (0., 0.)
    assert candidate.sample_count == 8 and candidate.event_cap == 128
    with pytest.raises(TypeError, match="arrival_rates"):
        ObservedRateAgeTailLaneRollout(options(), 600., 123, arrival_rates=(.1, .2))
    with pytest.raises(TypeError, match="load"):
        ObservedRateAgeTailLaneRollout(options(), 600., 123, load=.95)
    assert candidate.preprocessing_work["rate_dimension_option_probes"] == 1
    assert candidate.preprocessing_work["option_probes"] == 3
    assert candidate.preprocessing_work["age_profile_option_probes"] == 3


def test_cumulative_birth_counts_include_already_admitted_and_completed_history():
    candidate = planner()
    actual = candidate(options(), (1, 0), (100., 103.), 100., 0, ((4.,), ()), (80, 20))
    assert actual.observed_arrival_counts == (80, 20)
    assert actual.estimated_arrival_rates == (.8, .2)
    assert actual.rate_exposure_s == 100.
    assert actual.estimated_arrival_rates != (.01, 0.)  # Waiting Q/exposure.
    assert actual.rate_estimation_coordinate_reads == actual.rate_estimation_divisions == 2


def test_empty_idle_calls_without_new_births_reduce_rates_with_more_exposure():
    candidate = planner()
    initial_rng = candidate.rng.getstate()
    first = candidate(options(), (0, 0), (5., 5.), 5., 0, ((), ()), (10, 2))
    later = candidate(options(), (0, 0), (10., 10.), 10., 0, ((), ()), (10, 2))
    assert first.estimated_arrival_rates == (2., .4)
    assert later.estimated_arrival_rates == (1., .2)
    assert not first.eligible and not later.eligible
    assert first.planning_random_draws == later.planning_random_draws == 0
    assert first.rate_estimation_coordinate_reads == later.rate_estimation_coordinate_reads == 2
    assert first.rate_estimation_divisions == later.rate_estimation_divisions == 2
    assert candidate.rng.getstate() == initial_rng


def test_time_zero_uses_zero_rates_without_division_or_pseudocount():
    candidate = planner()
    actual = candidate(options(), (2, 1), (0., 3.), 0., 0, ((0., 0.), (0.,)), (2, 1))
    assert actual.observed_arrival_counts == (2, 1)
    assert actual.rate_exposure_s == 0.
    assert actual.estimated_arrival_rates == (0., 0.)
    assert actual.rate_estimation_coordinate_reads == 2 and actual.rate_estimation_divisions == 0
    assert actual.planning_events == actual.planning_random_draws == 0
    assert actual.sample_count == 8


def test_public_h_stops_exposure_and_new_planning_arrivals():
    candidate = planner()
    initial_rng = candidate.rng.getstate()
    at_h = candidate(options(), (2, 1), (600., 603.), 600., 0, ((10., 10.), (0.,)), (60, 30))
    after_h = candidate(options(), (2, 1), (1200., 1203.), 1200., 0,
                         ((610., 610.), (600.,)), (60, 30))
    for actual in (at_h, after_h):
        assert actual.rate_exposure_s == 600.
        assert actual.estimated_arrival_rates == (.1, .05)
        assert actual.window_s == 0.
        assert actual.planning_events == actual.planning_random_draws == 0
        assert actual.rate_estimation_divisions == 2
    assert candidate.rng.getstate() == initial_rng


def test_initial_observed_burst_overflow_falls_back_for_whole_call():
    candidate = planner()
    actual = candidate(options(), (2, 1), (.1, 3.1), .1, 0,
                       ((0., 0.), (0.,)), (100, 100))
    assert actual.estimated_arrival_rates == (1000., 1000.)
    assert actual.planning_overflow and actual.planning_events == 129
    assert actual.sample_count == 1 and actual.candidate_count == actual.rollout_calls == 0
    assert not actual.changed and actual.waiting_age_reads == actual.flow_evaluations == 0
    assert actual.selection == select_single_lane_control(options(), (2, 1), 1, 0)
    assert actual.rate_estimation_coordinate_reads == actual.rate_estimation_divisions == 2


def test_same_estimated_rates_and_rng_reproduce_every_frozen_parent_field():
    observed = planner()
    parent = AgeTailConstrainedLaneRollout(options(), (.2, .1), 600.,
                                         "observed-plan-v1:toy:online")
    args = (options(), (2, 1), (10., 13.), 10., 0, ((9., 2.), (1.,)))
    actual = observed(*args, (2, 1))
    expected = parent(*args)
    for field in fields(AgeTailDecision):
        assert getattr(actual, field.name) == getattr(expected, field.name)
    assert observed.rng.getstate() == parent.rng.getstate()
    assert actual.estimated_arrival_rates == (.2, .1)


def test_each_call_overwrites_rate_state_using_only_observed_counts():
    first, second = planner(), planner()
    second.arrival_rates = (90., 900.)
    args = (options(), (2, 1), (10., 13.), 10., 0, ((9., 2.), (1.,)), (2, 1))
    assert first(*args) == second(*args)
    assert second.arrival_rates == (.2, .1)


def test_outside_queue_envelope_still_charges_each_observed_coordinate():
    candidate = planner()
    state = candidate.rng.getstate()
    actual = candidate(options(), (4, 1), (10., 13.), 10., 0,
                       ((9., 8., 7., 6.), (2.,)), (12, 7))
    assert not actual.eligible and not actual.changed
    assert actual.estimated_arrival_rates == (1.2, .7)
    assert actual.rate_estimation_coordinate_reads == actual.rate_estimation_divisions == 2
    assert actual.root_option_probes == 3 and actual.rollout_calls == 0
    assert candidate.rng.getstate() == state


def test_plain_control_ignores_counts_and_supports_real_parent_six_arg_call():
    candidate = planner()
    state = candidate.rng.getstate()
    args = (options(), (2, 1), (10., 13.), 10., 0, ((9., 2.), (1.,)))
    six = candidate.maxweight(*args)
    seven = candidate.maxweight(*args, (999, 999))
    assert six == seven
    assert seven.selection == select_single_lane_control(options(), (2, 1), 1, 0)
    assert seven.rate_estimation_coordinate_reads == seven.rate_estimation_divisions == 0
    assert seven.estimated_arrival_rates == ()
    assert candidate.rng.getstate() == state and candidate.arrival_rates == (0., 0.)


def test_same_released_history_and_different_future_produce_identical_decisions():
    current = FrameModel("toy", "observed", 2, ("a", "b"), (1., 1.), options())
    prefix = ((0., 0, 0), (0., 0, 1), (0., 1, 0), (.5, 0, 2))
    futures = (((4.1, 0, 3),), ((4.2, 1, 1), (4.3, 0, 3)))
    results = []
    for future in futures:
        tape, candidate, admitted = prefix + future, planner(), [0, 0]

        def visible_selector(options, queue, available, now, lane):
            ages, counts = [], []
            for coordinate in range(2):
                released = sorted(time for time, c, _ in tape if c == coordinate and time <= now)
                counts.append(len(released))
                ages.append(tuple(now - time for time in released[admitted[coordinate]:]))
            assert tuple(map(len, ages)) == tuple(queue)
            decision = candidate(options, queue, available, now, lane, tuple(ages), tuple(counts))
            for coordinate, count in enumerate(decision.selection.service):
                admitted[coordinate] += count
            return decision

        results.append(simulate_contextual_trajectory(current, tape, 6., 20.,
                       visible_selector, policy="observed_rate"))
    causal = lambda result: [frame for frame in result["frames"] if frame["start"] < 4.1]
    assert causal(results[0]) == causal(results[1])
    assert all(frame["decision_delay_s"] == 0. for result in results for frame in result["frames"])
