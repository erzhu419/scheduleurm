"""Independent planner randomness, complete-lane costs and honest work/support."""
from fractions import Fraction
from math import nextafter
from random import Random

import pytest

import algorithm.arrival_aware_lane_rollout as rollout
from algorithm.arrival_aware_lane_rollout import (
    ArrivalAwareLaneRollout, PlanningArrivals, sample_planning_arrivals,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory
from algorithm.lane_refill_simple_controls import select_single_lane_control


def options():
    return (FrameOption("a1", (1, 0), 2., ((2.,), ())),
            FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
            FrameOption("b1", (0, 1), 1., ((), (1.,))))


def planner(current=None, rates=(0., 0.), **kwargs):
    return ArrivalAwareLaneRollout(options() if current is None else current,
                                  rates, 600., 987654, **kwargs)


def path(*events):
    return PlanningArrivals(tuple(events), False, 0, 0)


@pytest.mark.parametrize("queue", ((0, 0), (2, 1), (7, 4), (1, 0)))
def test_precomputed_plain_control_reproduces_ordinary_exact_maxweight(queue):
    current = options()
    candidate = planner(current)
    actual = candidate.maxweight(current, queue, (0., 3.), 0., 0)
    assert actual.selection == select_single_lane_control(current, queue, 1, 0)
    assert actual.root_option_probes == len(current)
    assert actual.rollout_calls == actual.sample_count == 0
    assert candidate.preprocessing_work == {
        "option_probes": 3, "requirement_reads": 6,
        "completion_offset_reads": 4, "singleton_checks": 3}


def test_zero_arrivals_full_clearing_counts_work_and_honest_support_gap():
    candidate = planner()
    actual = candidate(options(), (2, 1), (0., 3.), 0., 0)
    assert actual.selection.option_indices == (2,)
    assert actual.baseline_predicted_completion_s == 10.
    assert actual.chosen_predicted_completion_s == 9.
    assert actual.sample_count == actual.configured_sample_count == 8
    assert actual.rollout_calls == 8 * 4
    assert actual.rollout_option_probes == 3 * actual.rollout_calls
    assert actual.root_option_probes == 6
    assert actual.preprocessing_option_probes == 0
    assert actual.changed and actual.eligible
    assert not actual.selection.score_exact and not actual.selection.canonical_exact
    assert actual.selection.gap > 0
    assert Fraction.from_float(actual.selection.upper_bound) >= Fraction(4, 3)


def test_all_roots_share_each_generated_population_once(monkeypatch):
    generated = []
    shared = path((.5, 1, 0))

    def fixed(rates, window, rng, event_cap):
        generated.append((rates, window, event_cap))
        return shared

    monkeypatch.setattr(rollout, "sample_planning_arrivals", fixed)
    actual = planner(rates=(.3, .7))(options(), (2, 1), (0., 3.), 0., 0)
    assert len(generated) == 8
    assert actual.planning_events == 8 and actual.candidate_count == 3
    # Both lanes become free at3 after the a2 root, so both b jobs finish4.
    assert actual.baseline_predicted_completion_s == 13.5
    assert actual.chosen_predicted_completion_s == 12.5
    assert actual.selection.option_indices == (2,)


def test_busy_lane_availability_and_loaded_complete_period_are_preserved():
    candidate = planner()
    empty = (path(),)
    early = candidate.score_roots((2, 1), (0., 1.), 0., 0, empty)
    late = candidate.score_roots((2, 1), (0., 3.), 0., 0, empty)
    assert early.costs[1] == 8 and late.costs[1] == 10
    loaded = (FrameOption("loaded", (1, 1), 10., ((.2,), (10.,)), loaded=True),
              FrameOption("a1", (1, 0), 8., ((8.,), ())),
              FrameOption("b1", (0, 1), 8., ((), (8.,))))
    scored = planner(loaded).score_roots((2, 1), (0., 100.), 0., 0,
                                        (path((.5, 0, 0)),))
    # Resident ends at10; two later a jobs finish18 and26. Early target .2
    # does not release the busy lane, and the future arrival's .5 is deducted.
    assert float(scored.costs[0]) == pytest.approx(53.7)


def test_full_drain_keeps_late_future_job_and_charges_idle_polling():
    current = (FrameOption("a", (1,), 2., ((2.,),)),)
    candidate = planner(current, rates=(0.,))
    scored = candidate.score_roots((1,), (0.,), 0., 0,
                                   (path((10.25, 0, 0)),))
    assert scored.costs[0] == Fraction(19, 4)  # 2 + (13 - 10.25).
    assert scored.rollout_idle_calls == 9  # poll2,...,10, then dispatch11.
    assert scored.rollout_calls == 10 and scored.rollout_option_probes == 10


def test_other_free_lane_continues_at_same_time_and_ties_keep_mw():
    candidate = planner()
    scored = candidate.score_roots((2, 1), (0., 0.), 0., 0, (path(),))
    assert scored.costs[1] == 7 and scored.costs[0] == 6
    duplicate = (FrameOption("first", (1,), 1., ((.5,),)),
                 FrameOption("second", (1,), 1., ((.5,),)))
    actual = planner(duplicate, rates=(0.,))(duplicate, (1,), (0.,), 0., 0)
    assert actual.selection.option_indices == (0,) and not actual.changed
    assert actual.selection.canonical_exact and actual.selection.score_exact


def test_nonbinary_equal_support_changes_canonical_flag_without_gap():
    current = (FrameOption("first", (1,), 3., ((3.,),)),
               FrameOption("earlier", (1,), 3., ((1.,),)))
    actual = planner(current, rates=(0.,))(current, (1,), (0.,), 0., 0)
    assert actual.selection.option_indices == (1,) and actual.changed
    assert actual.selection.score_exact and actual.selection.gap == 0
    assert not actual.selection.canonical_exact and not actual.selection.exact


def test_exact_strict_improvement_uses_float_data_without_tuned_epsilon():
    current = (FrameOption("first", (1,), 1., ((.5,),)),
               FrameOption("earlier", (1,), 1., ((nextafter(.5, 0.),),)))
    actual = planner(current, rates=(0.,))(current, (1,), (0.,), 0., 0)
    assert actual.changed and actual.selection.option_indices == (1,)
    assert actual.chosen_predicted_completion_s < actual.baseline_predicted_completion_s


class ShortGapRng:
    def expovariate(self, rate):
        return .1


def test_sampler_overflow_retains_cap_plus_one_and_selector_falls_back():
    sampled = sample_planning_arrivals((1., 1.), 3., ShortGapRng(), event_cap=2)
    assert sampled.overflow and len(sampled.events) == 3
    assert sampled.random_draws == 3
    candidate = planner(rates=(1., 1.), event_cap=2)
    candidate.rng = ShortGapRng()
    actual = candidate(options(), (2, 1), (0., 3.), 0., 0)
    assert actual.planning_overflow and actual.sample_count == 1
    assert actual.planning_events == 3 and actual.planning_random_draws == 3
    assert actual.selection == select_single_lane_control(options(), (2, 1), 1, 0)
    assert actual.eligible and not actual.changed
    assert actual.rollout_calls == actual.candidate_count == 0
    assert actual.root_option_probes == 3
    with pytest.raises(ValueError, match="overflowing"):
        candidate.score_roots((2, 1), (0., 3.), 0., 0, (sampled,))


def test_sample_law_zero_rates_stop_draws_and_return_common_sorted_population():
    first = sample_planning_arrivals((0., .5, .3), 8., Random(12), 128)
    second = sample_planning_arrivals((0., .5, .3), 8., Random(12), 128)
    assert first == second and not first.overflow
    assert first.events == tuple(sorted(first.events))
    assert all(0 <= event[0] <= 8 and event[1] in (1, 2) for event in first.events)
    assert first.random_draws == len(first.events) + 2
    assert sample_planning_arrivals((1.,), 0., ShortGapRng(), 128).random_draws == 0


def test_high_and_empty_queues_do_not_consume_planning_randomness():
    candidate = planner(rates=(1., 1.))
    initial = candidate.rng.getstate()
    for queue in ((0, 0), (4, 1)):
        actual = candidate(options(), queue, (0., 3.), 0., 0)
        assert not actual.eligible and not actual.changed
        assert actual.sample_count == actual.planning_random_draws == 0
        assert candidate.rng.getstate() == initial


def test_window_uses_complete_period_and_registered_public_stop():
    candidate = planner(rates=(1., 1.))
    before = candidate(options(), (2, 1), (598., 601.), 598., 0)
    assert before.window_s == 2.
    state = candidate.rng.getstate()
    after = candidate(options(), (2, 1), (600., 603.), 600., 0)
    assert after.window_s == 0 and after.planning_events == 0
    assert after.planning_random_draws == 0 and candidate.rng.getstate() == state
    assert after.baseline_predicted_completion_s == 10.


def test_shifted_context_keeps_relative_costs():
    candidate = planner()
    first = candidate.score_roots((2, 1), (0., 3.), 0., 0, (path((.5, 1, 0)),))
    shifted = candidate.score_roots((2, 1), (7., 10.), 7., 0, (path((.5, 1, 0)),))
    assert first == shifted


def test_real_future_cannot_change_identical_history_with_same_planner_state():
    current = FrameModel("toy", "planning", 2, ("a", "b"), (1., 1.), options())
    prefix = ((0., 0, 0), (0., 0, 1), (0., 1, 0), (.5, 0, 2))
    futures = (((4.1, 0, 3),), ((4.2, 1, 1), (4.3, 0, 3)))
    results = []
    for future in futures:
        candidate = planner(rates=(.4, .2))
        results.append(simulate_contextual_trajectory(current, prefix + future, 6., 20.,
                       candidate, policy="arrival_aware"))
    causal = lambda result: [frame for frame in result["frames"] if frame["start"] < 4.1]
    assert causal(results[0]) == causal(results[1])
    assert all(frame["decision_delay_s"] == 0 for result in results for frame in result["frames"])


def test_small_option_duration_does_not_release_before_one_second_floor():
    current = (FrameOption("short", (1,), .2, ((.2,),)),
               FrameOption("batch", (2,), 3., ((3., 3.),)))
    candidate = planner(current, rates=(0.,))
    scored = candidate.score_roots((2,), (0.,), 0., 0, (path(),))
    assert float(scored.costs[0]) == pytest.approx(1.4)
    assert candidate(current, (2,), (0.,), 0., 0).selection.duration == 1.
