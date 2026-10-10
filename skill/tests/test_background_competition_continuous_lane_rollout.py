"""Background jobs compete for lanes but never enter the tagged flow objective."""
from dataclasses import fields
from fractions import Fraction
import inspect

import algorithm.background_competition_continuous_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, _LazyBackgroundArrivals,
)
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption


OPTIONS = (FrameOption("a", (1, 0), 2., ((2.,), ())),
           FrameOption("b", (0, 1), 1., ((), (1.,))))


def population(*events):
    return PlanningArrivals(tuple(events), False, 0, 0)


def scripted_random(monkeypatch, gaps):
    """Replace only the independent background RNG, keeping prefix RNG real."""
    instances = []

    class ScriptedRandom:
        def __init__(self, seed):
            self.seed = seed
            self.gaps = iter(gaps)
            self.rates = []
            instances.append(self)

        def expovariate(self, rate):
            self.rates.append(rate)
            return next(self.gaps)

    monkeypatch.setattr(module, "Random", ScriptedRandom)
    return instances


def test_suffix_starts_at_full_window_and_releases_shared_events_chronologically(monkeypatch):
    instances = scripted_random(monkeypatch, (.75, .5, .5, 100., 100.))
    suffix = _LazyBackgroundArrivals((1., 1.), 2., "suffix", 128, 1)
    # The last tagged birth could be at .5; background still starts after 2.
    assert suffix.release_through(Fraction(2))
    assert suffix.events == [] and suffix.random_draws == 0
    assert suffix.release_through(Fraction(13, 5))
    assert suffix.events == [(Fraction(5, 2), 1)]
    assert suffix.release_through(Fraction(14, 5))
    assert suffix.events == [(Fraction(5, 2), 1), (Fraction(11, 4), 0)]
    assert suffix.release_through(Fraction(3))
    assert suffix.events == [(Fraction(5, 2), 1), (Fraction(11, 4), 0),
                             (Fraction(3), 1)]
    # A second branch can release an earlier time using the same cache.
    assert suffix.release_through(Fraction(13, 5))
    assert suffix.random_draws == 5 and suffix.rate_reads == 5
    assert len(instances) == 1 and len(instances[0].rates) == 5


def test_background_changes_tagged_costs_preserving_busy_lane_and_same_time_birth(monkeypatch):
    instances = scripted_random(monkeypatch, (.5, 100.))
    candidate = BackgroundCompetitionContinuousLaneRollout(OPTIONS, "competition", sample_count=1)
    candidate.arrival_rates = (0., 1.)
    rng_before = candidate.rng.getstate()
    scored = candidate.score_background_roots(
        (2, 1), (1000., 1004.), 1000., 0, ((5., 1.), (2.,)),
        (population((2., 0)),))
    # At t=2 the tagged A birth is released before dispatch. The B background
    # birth at 2.5 wins the t=3 MW tie, reserving a lane until t=4. Both A jobs
    # can start at t=4 only because the other committed lane becomes free then.
    assert scored.costs == {0: Fraction(23), 1: Fraction(22)}
    assert scored.squared_costs == {0: Fraction(139), 1: Fraction(138)}
    assert scored.rollout_calls == 8 and scored.rollout_idle_calls == 0
    assert scored.rollout_option_probes == 8 * len(OPTIONS)
    assert scored.flow_evaluations == 8  # Four tagged jobs per root.
    assert scored.background_admissions == 2
    assert scored.admitted_job_reads == 10 and scored.tag_flag_reads == 10
    # One suffix is shared by both roots, rather than drawing two futures.
    assert scored.background_planning_events == 1
    assert scored.background_planning_random_draws == 2
    assert scored.background_planning_rate_reads == 3
    assert scored.background_seed_initializations == 1 and len(instances) == 1
    assert candidate.rng.getstate() == rng_before


def test_stop_when_final_tagged_job_is_admitted_with_background_still_waiting(monkeypatch):
    instances = scripted_random(monkeypatch, (.5, .1, 100.))
    options = (FrameOption("a", (1, 0), 1., ((1.,), ())),
               FrameOption("b", (0, 1), 2., ((), (2.,))))
    candidate = BackgroundCompetitionContinuousLaneRollout(options, "stop", sample_count=1)
    candidate.arrival_rates = (0., 1.)
    scored = candidate.score_background_roots(
        (4, 0), (1000.,), 1000., 0, ((0., 0., 0., 0.), ()), (population(),))
    # Two B background jobs are waiting at t=3. A wins the equal MW score by
    # shorter duration, and its completion at t=4 is known at admission t=3.
    assert scored.costs == {0: Fraction(10)}
    assert scored.squared_costs == {0: Fraction(30)}
    assert scored.rollout_calls == 3 and scored.flow_evaluations == 4
    assert scored.background_planning_events == 2
    assert scored.background_admissions == 0 and scored.admitted_job_reads == 4
    assert len(instances[0].rates) == 3


def test_zero_background_rates_reproduce_parent_exact_costs_and_scans():
    candidate = BackgroundCompetitionContinuousLaneRollout(OPTIONS, "zero", sample_count=1)
    parent = ContinuousArrivalAgeTailLaneRollout(OPTIONS, "zero", sample_count=1)
    args = ((2, 1), (1000., 1004.), 1000., 0, ((5., 1.), (2.,)),
            (population((2., 0)),))
    actual = candidate.score_background_roots(*args)
    expected = parent.score_age_roots(*args)
    assert actual.costs == expected.costs == {0: Fraction(22), 1: Fraction(21)}
    assert actual.squared_costs == expected.squared_costs == {
        0: Fraction(126), 1: Fraction(125)}
    for field in fields(expected):
        assert getattr(actual, field.name) == getattr(expected, field.name)
    assert actual.background_planning_events == actual.background_planning_random_draws == 0
    assert actual.background_admissions == 0


def test_combined_prefix_and_background_overflow_restores_mw_with_partial_work(monkeypatch):
    options = (*OPTIONS, FrameOption("a2", (2, 0), 3., ((3., 3.), ())))
    candidate = BackgroundCompetitionContinuousLaneRollout(
        options, "overflow", sample_count=1, event_cap=2)
    scripted_random(monkeypatch, (100., .1, .1))
    prefix = PlanningArrivals(((1., 0),), False, 2, 2)
    monkeypatch.setattr(module, "sample_planning_arrivals", lambda *args: prefix)
    args = (options, (2, 1), (1000., 1005.), 1000., 0,
            ((5., 1.), (2.,)), (200, 100))
    result = candidate(*args)
    assert result.selection == candidate.maxweight(*args).selection
    assert result.planning_overflow and result.background_rollout_overflow
    assert not result.changed
    assert result.tagged_planning_events == 1 and result.planning_events == 3
    assert result.background_planning_events == 2
    assert result.planning_random_draws == 5 and result.planning_rate_reads == 5
    # The root and one A2 continuation were actually simulated before the
    # second background birth crossed the combined cap; these fees remain.
    assert result.rollout_calls == 1 and result.rollout_option_probes == len(options)
    assert result.root_option_probes == 2 * len(options)
    assert result.flow_evaluations == result.admitted_job_reads == result.tag_flag_reads == 3
    assert result.baseline_predicted_completion_s is None
    assert result.chosen_predicted_completion_s is None
    assert result.baseline_predicted_squared_flow_s2 is None
    assert result.risk_feasible_candidates == result.risk_filtered_candidates == 0


def test_causal_rates_and_outside_envelope_do_not_sample(monkeypatch):
    assert "arrival_horizon" not in inspect.signature(BackgroundCompetitionContinuousLaneRollout).parameters
    candidate = BackgroundCompetitionContinuousLaneRollout(OPTIONS, "causal")
    assert candidate.event_cap == 128

    def forbidden(*args):
        raise AssertionError("empty/large queues must not sample a future")

    monkeypatch.setattr(module, "sample_planning_arrivals", forbidden)
    monkeypatch.setattr(module, "Random", forbidden)
    rng_before = candidate.rng.getstate()
    for queues, ages in (((0, 0), ((), ())), ((3, 1), ((9., 8., 7.), (2.,)))):
        args = (OPTIONS, queues, (1500., 1504.), 1500., 0, ages, (150, 75))
        result = candidate(*args)
        assert result.selection == candidate.maxweight(*args).selection
        assert not result.eligible and result.planning_random_draws == 0
        assert result.background_planning_events == result.background_seed_initializations == 0
        assert result.observed_arrival_counts == (150, 75)
        assert result.estimated_arrival_rates == (.1, .05)
        assert result.rate_exposure_s == 1500.
    assert candidate.rng.getstate() == rng_before
