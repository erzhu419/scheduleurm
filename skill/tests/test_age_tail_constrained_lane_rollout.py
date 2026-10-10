"""FIFO flow squares, risk-constrained roots and unchanged causal planning."""
from fractions import Fraction

import pytest

import algorithm.age_tail_constrained_lane_rollout as module
from algorithm.age_tail_constrained_lane_rollout import AgeTailConstrainedLaneRollout
from algorithm.arrival_aware_lane_rollout import ArrivalAwareLaneRollout, PlanningArrivals
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory
from algorithm.lane_refill_simple_controls import select_single_lane_control


def options():
    return (FrameOption("a1", (1, 0), 2., ((2.,), ())),
            FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
            FrameOption("b1", (0, 1), 1., ((), (1.,))))


def planner(current=None, rates=(0., 0.), **kwargs):
    return AgeTailConstrainedLaneRollout(options() if current is None else current,
        rates, 600., "arrival-plan-v1:toy:0.95:online", **kwargs)


def path(*events):
    return PlanningArrivals(tuple(events), False, 0, 0)


@pytest.mark.parametrize("ages", (((0., 0.), (0.,)), ((10., 10.), (0.,)),
                                  ((8., 1.), (20.,))))
def test_ages_add_root_constant_to_s1_without_changing_old_root_order(ages):
    current = options()
    new = planner(current)
    old = ArrivalAwareLaneRollout(current, (0., 0.), 600., 123)
    paths = (path(), path((.5, 1, 0), (2.2, 0, 0)))
    plain = old.score_roots((2, 1), (0., 3.), 0., 0, paths)
    actual = new.score_age_roots((2, 1), (0., 3.), 0., 0, ages, paths)
    constant = len(paths) * sum(Fraction.from_float(age) for group in ages for age in group)
    assert actual.costs == {index: cost + constant for index, cost in plain.costs.items()}
    assert actual.rollout_calls == plain.rollout_calls
    assert actual.rollout_option_probes == plain.rollout_option_probes
    assert actual.waiting_age_reads == 3


def test_old_waiting_class_blocks_s1_favored_root_without_new_coefficient():
    candidate = planner()
    ages = ((10., 10.), (0.,))
    scored = candidate.score_age_roots((2, 1), (0., 3.), 0., 0, ages, (path(),))
    assert scored.costs[1] == 30 and scored.costs[2] == 29
    assert scored.squared_costs[1] == 354 and scored.squared_costs[2] == 393
    actual = candidate(options(), (2, 1), (0., 3.), 0., 0, ages)
    assert actual.selection.option_indices == (1,) and not actual.changed
    assert actual.unconstrained_root_index == 2 and actual.risk_changed_unconstrained_root
    assert actual.risk_filtered_candidates == 2 and actual.risk_feasible_candidates == 1
    assert actual.baseline_predicted_completion_s == actual.chosen_predicted_completion_s == 30
    assert actual.baseline_predicted_squared_flow_s2 == actual.chosen_predicted_squared_flow_s2 == 354
    assert actual.selection.canonical_exact and actual.selection.score_exact


def test_old_b_class_allows_strict_mean_gain_with_lower_squared_flow():
    candidate = planner()
    ages = ((0., 0.), (20.,))
    actual = candidate(options(), (2, 1), (0., 3.), 0., 0, ages)
    assert actual.changed and actual.selection.option_indices == (2,)
    assert actual.baseline_predicted_completion_s == 30
    assert actual.chosen_predicted_completion_s == 29
    assert actual.baseline_predicted_squared_flow_s2 == 594
    assert actual.chosen_predicted_squared_flow_s2 == 473
    assert actual.unconstrained_root_index == 2 and not actual.risk_changed_unconstrained_root
    assert not actual.selection.score_exact and not actual.selection.canonical_exact
    assert Fraction.from_float(actual.selection.upper_bound) >= Fraction(4, 3)
    assert actual.selection.gap > 0


def test_current_fifo_ages_bind_oldest_job_to_earliest_sorted_offset():
    current = (FrameOption("batch", (2,), 10., ((8., 2.),)),
               FrameOption("single", (1,), 10., ((10.,),)))
    scored = planner(current, rates=(0.,)).score_age_roots(
        (2,), (0.,), 0., 0, ((10., 1.),), (path(),))
    # FIFO gives flows12 and9. Reversed offsets would give18 and3, sharing
    # the same S1=21 but the wrong squared cost333 rather than225.
    assert scored.costs[0] == 21 and scored.squared_costs[0] == 225
    assert scored.flow_evaluations == 4


def test_future_job_appends_after_visible_fifo_and_uses_its_own_arrival():
    current = (FrameOption("single", (1,), 2., ((2.,),)),)
    scored = planner(current, rates=(0.,)).score_age_roots(
        (2,), (0.,), 0., 0, ((5., 1.),), (path((.5, 0, 0)),))
    # Current flows7 and5, then future completion6 minus arrival.5 =5.5.
    assert scored.costs[0] == Fraction(35, 2)
    assert scored.squared_costs[0] == Fraction(417, 4)
    assert scored.rollout_calls == 2 and scored.flow_evaluations == 3


def test_busy_resident_full_duration_and_early_target_offsets_affect_squares():
    current = (FrameOption("loaded", (1, 1), 10., ((.2,), (10.,)), loaded=True),
               FrameOption("a1", (1, 0), 8., ((8.,), ())),
               FrameOption("b1", (0, 1), 8., ((), (8.,))))
    scored = planner(current).score_age_roots((2, 1), (0., 100.), 0., 0,
                ((5., 1.), (2.,)), (path((.5, 0, 0)),))
    # Flows5.2,12,19,25.5; early target completion cannot start later a jobs.
    assert float(scored.costs[0]) == pytest.approx(61.7)
    assert float(scored.squared_costs[0]) == pytest.approx(1182.29)
    assert scored.flow_evaluations == 12


def test_complete_drain_keeps_late_future_cost_and_all_idle_calls():
    current = (FrameOption("single", (1,), 2., ((2.,),)),)
    scored = planner(current, rates=(0.,)).score_age_roots((1,), (0.,), 0., 0,
                ((5.,),), (path((10.25, 0, 0)),))
    assert scored.costs[0] == Fraction(39, 4)
    assert scored.squared_costs[0] == Fraction(905, 16)
    assert scored.rollout_calls == 10 and scored.rollout_idle_calls == 9
    assert scored.rollout_option_probes == 10


def test_distinct_samples_sum_individual_flow_squares_not_square_average():
    candidate = planner()
    paths = (path(), path((.5, 1, 0)))
    args = ((2, 1), (0., 3.), 0., 0, ((8., 1.), (20.,)))
    separate = [candidate.score_age_roots(*args, (population,)) for population in paths]
    combined = candidate.score_age_roots(*args, paths)
    assert combined.costs == {i: sum(part.costs[i] for part in separate) for i in combined.costs}
    assert combined.squared_costs == {i: sum(part.squared_costs[i] for part in separate)
                                      for i in combined.squared_costs}
    assert combined.flow_evaluations == 3 * (3 + 4)


def test_common_sample_generation_and_actual_counter_accounting(monkeypatch):
    generations = []
    population = path((.5, 1, 0))

    def sampled(rates, window, rng, event_cap):
        generations.append((rates, window, event_cap))
        return population

    monkeypatch.setattr(module, "sample_planning_arrivals", sampled)
    actual = planner(rates=(.3, .7))(options(), (2, 1), (0., 3.), 0., 0,
                                     ((0., 0.), (0.,)))
    assert len(generations) == 8 and actual.planning_events == 8
    assert actual.root_option_probes == 6
    assert actual.rollout_option_probes == 3 * actual.rollout_calls
    assert actual.waiting_age_reads == 3
    assert actual.flow_evaluations == 3 * 8 * 4
    assert actual.chosen_predicted_squared_flow_s2 <= actual.baseline_predicted_squared_flow_s2
    candidate = planner()
    assert candidate.preprocessing_work["option_probes"] == 3
    assert candidate.preprocessing_work["age_profile_option_probes"] == 3
    assert candidate.preprocessing_work["age_profile_completion_offset_reads"] == 4


class ShortGapRng:
    def expovariate(self, rate):
        return .1


def test_overflow_fallback_does_not_read_ages_or_score_truncated_population():
    candidate = planner(rates=(1., 1.), event_cap=2)
    candidate.rng = ShortGapRng()
    actual = candidate(options(), (2, 1), (0., 3.), 0., 0, ((10., 10.), (0.,)))
    assert actual.planning_overflow and actual.planning_events == 3
    assert actual.sample_count == 1 and actual.planning_random_draws == 3
    assert actual.selection == select_single_lane_control(options(), (2, 1), 1, 0)
    assert not actual.changed and actual.waiting_age_reads == actual.flow_evaluations == 0
    assert actual.rollout_calls == actual.risk_filtered_candidates == 0
    assert actual.baseline_predicted_squared_flow_s2 is None
    with pytest.raises(ValueError, match="overflowing"):
        candidate.score_age_roots((2, 1), (0., 3.), 0., 0, ((10., 10.), (0.,)),
                                  (PlanningArrivals(((.1, 0, 0),), True, 1, 1),))


def test_empty_high_queue_and_plain_mw_consume_no_planning_draws_or_age_reads():
    candidate = planner(rates=(1., 1.))
    state = candidate.rng.getstate()
    for queue, ages in (((0, 0), ((), ())), ((4, 1), ((9., 8., 7., 6.), (1.,)))):
        actual = candidate(options(), queue, (0., 3.), 0., 0, ages)
        control = candidate.maxweight(options(), queue, (0., 3.), 0., 0, ages)
        assert actual.selection == control.selection == select_single_lane_control(options(), queue, 1, 0)
        assert not actual.eligible and not actual.changed
        assert actual.planning_random_draws == actual.waiting_age_reads == 0
        assert actual.root_option_probes == 3 and candidate.rng.getstate() == state


def test_post_h_contract_has_no_new_arrivals_and_shifted_context_same_cost():
    candidate = planner(rates=(1., 1.))
    before = candidate(options(), (2, 1), (598., 601.), 598., 0, ((10., 10.), (0.,)))
    assert before.window_s == 2.
    state = candidate.rng.getstate()
    after = candidate(options(), (2, 1), (600., 603.), 600., 0, ((10., 10.), (0.,)))
    assert after.window_s == 0 and after.planning_events == after.planning_random_draws == 0
    assert candidate.rng.getstate() == state
    assert after.baseline_predicted_squared_flow_s2 == 354
    a = candidate.score_age_roots((2, 1), (0., 3.), 0., 0, ((10., 10.), (0.,)), (path(),))
    b = candidate.score_age_roots((2, 1), (7., 10.), 7., 0, ((10., 10.), (0.,)), (path(),))
    assert a == b


def test_equal_mean_and_risk_keep_the_mw_canonical_root():
    current = (FrameOption("first", (1,), 1., ((.5,),)),
               FrameOption("second", (1,), 1., ((.5,),)))
    actual = planner(current, rates=(0.,))(current, (1,), (0.,), 0., 0, ((12.,),))
    assert actual.selection.option_indices == (0,) and not actual.changed
    assert actual.unconstrained_root_index == 0 and not actual.risk_changed_unconstrained_root


def test_changed_equal_support_root_has_correct_canonical_and_gap_flags():
    current = (FrameOption("first", (1,), 3., ((3.,),)),
               FrameOption("earlier", (1,), 3., ((1.,),)))
    actual = planner(current, rates=(0.,))(current, (1,), (0.,), 0., 0, ((10.,),))
    assert actual.changed and actual.selection.option_indices == (1,)
    assert actual.selection.score_exact and actual.selection.gap == 0
    assert not actual.selection.canonical_exact and not actual.selection.exact
    assert actual.chosen_predicted_squared_flow_s2 == 121
    assert actual.baseline_predicted_squared_flow_s2 == 169


def test_waiting_age_counts_must_cover_every_visible_job():
    with pytest.raises(ValueError, match="age counts"):
        planner().score_age_roots((2, 1), (0., 3.), 0., 0, ((10.,), (0.,)), (path(),))


def test_real_future_does_not_change_decisions_on_identical_visible_history():
    current = FrameModel("toy", "planning", 2, ("a", "b"), (1., 1.), options())
    prefix = ((0., 0, 0), (0., 0, 1), (0., 1, 0), (.5, 0, 2))
    futures = (((4.1, 0, 3),), ((4.2, 1, 1), (4.3, 0, 3)))
    results = []
    for future in futures:
        tape = prefix + future
        candidate = planner(rates=(.4, .2))
        admitted = [0, 0]

        def visible_selector(options, queue, available, now, lane):
            ages = []
            for c in range(2):
                released = sorted(time for time, coordinate, _ in tape
                                  if coordinate == c and time <= now)
                ages.append(tuple(now - time for time in released[admitted[c]:]))
            assert tuple(map(len, ages)) == tuple(queue)
            result = candidate(options, queue, available, now, lane, tuple(ages))
            for c, count in enumerate(result.selection.service):
                admitted[c] += count
            return result

        results.append(simulate_contextual_trajectory(current, tape, 6., 20.,
                      visible_selector, policy="age_tail"))
    causal = lambda result: [frame for frame in result["frames"] if frame["start"] < 4.1]
    assert causal(results[0]) == causal(results[1])
    assert all(frame["decision_delay_s"] == 0 for result in results for frame in result["frames"])
