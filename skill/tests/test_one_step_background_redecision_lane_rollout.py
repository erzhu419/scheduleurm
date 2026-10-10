"""One causal virtual BG redecision, with its own state and work ledger."""
from dataclasses import replace
from fractions import Fraction
from itertools import chain, repeat
from random import Random

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.one_step_background_redecision_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, _LazyBackgroundArrivals,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_one_step_background_redecision_development import (
    check_rollout, redecision_work,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.one_step_background_redecision_lane_rollout import OneStepBackgroundRedecisionLaneRollout


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 2., ((), (2.,))),
           FrameOption("loaded", (1, 1), 2., ((.5,), (1.5,)), loaded=True))


def population(*events, overflow=False, draws=0):
    return PlanningArrivals(tuple(events), overflow, draws, 0)


def script_background(monkeypatch, gaps=(1000.,)):
    """Use the real chronological suffix and its metering, with chosen gaps."""
    class ScriptedRandom:
        def __init__(self, seed):
            self.gaps = chain(gaps, repeat(1000.))

        def expovariate(self, rate):
            return next(self.gaps)

    class ScriptedSuffix(_LazyBackgroundArrivals):
        def release_through(self, start):
            original = background_module.Random
            background_module.Random = ScriptedRandom
            try:
                return super().release_through(start)
            finally:
                background_module.Random = original

    monkeypatch.setattr(module, "_LazyBackgroundArrivals", ScriptedSuffix)
    monkeypatch.setattr(background_module, "_LazyBackgroundArrivals", ScriptedSuffix)


def capture_inner(monkeypatch, candidate):
    original = BackgroundCompetitionContinuousLaneRollout.__call__
    calls = []

    def traced(self, *args):
        if self is not candidate.inner_planner:
            return original(self, *args)
        record = {"args": args, "rng_state": self.rng.getstate(),
                  "counter": self.decision_counter}
        result = original(self, *args)
        record["decision"] = result
        calls.append(record)
        return result

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "__call__", traced)
    return calls


def split_prefixes(monkeypatch, candidate, outer=(), inner=(), consume=False):
    calls = []

    def sampled(rates, window, rng, cap):
        is_outer = rng is candidate.rng
        events = outer if is_outer else inner
        draws = len(events) + 1 if consume else 0
        for _ in range(draws):
            rng.random()
        calls.append((is_outer, tuple(rates), window, cap))
        return PlanningArrivals(tuple(events), False, draws, len(rates))

    monkeypatch.setattr(background_module, "sample_planning_arrivals", sampled)
    return calls


def test_real_inner_bg_improvement_preserves_fifo_loaded_period_and_outer_population(monkeypatch):
    candidate = OneStepBackgroundRedecisionLaneRollout(OPTIONS, "real", sample_count=1)
    candidate.arrival_rates = (.003, .001)
    candidate._outer_observed_counts = (3, 1)
    candidate.decision_counter = 7
    script_background(monkeypatch)
    split_prefixes(monkeypatch, candidate, inner=((.5, 1),))
    calls = capture_inner(monkeypatch, candidate)
    before = candidate.rng.getstate()
    args = ((3, 1), (1000., 1006.), 1000., 0, ((7., 4., 0.), (2.,)),
            (population(),))
    actual = candidate.score_background_roots(*args)
    reference = BackgroundCompetitionContinuousLaneRollout(OPTIONS, "real", sample_count=1)
    reference.arrival_rates = candidate.arrival_rates
    expected = reference.score_background_roots(*args)
    loaded_call, = [call for call in calls if call["args"][1] == (2, 0)]
    seen = loaded_call["args"]
    # Early .5/1.5 completions do not release the loaded lane before time2.
    assert seen[2:6] == ((1002., 1006.), 1002., 0, ((6., 2.), ()))
    assert seen[6] == (3, 1)  # Root admission never decrements observed history.
    inner = loaded_call["decision"]
    assert inner.changed and inner.selection.option_indices == (0,)
    assert inner.baseline_predicted_completion_s == 18.5
    assert inner.chosen_predicted_completion_s == 15.5
    assert actual.costs[3] == expected.costs[3] == Fraction(25)
    assert actual.squared_costs[3] == Fraction(337, 2)
    assert expected.squared_costs[3] == Fraction(349, 2)
    # The inner hypothetical B@.5 changes its action, but never joins the
    # outer tagged population; subsequent outer calls use ordinary MW.
    assert actual.flow_evaluations == 4 * 4 and actual.background_admissions == 0
    assert len(calls) == actual.candidate_count == 4
    assert all(call["rng_state"] == before and call["counter"] == 8 for call in calls)
    assert candidate.rng.getstate() == before and candidate.decision_counter == 7
    work = candidate._recourse_work
    assert work["recourse_rng_words_copied"] == len(before[1])
    assert work["recourse_rng_words_restored"] == 4 * len(before[1])
    assert work["recourse_counter_reads"] == 1 and work["recourse_counter_writes"] == 4
    for name in ("root_option_probes", "rollout_calls", "rollout_option_probes",
                 "planning_events", "planning_random_draws", "flow_evaluations",
                 "admitted_job_reads", "rate_estimation_coordinate_reads"):
        assert work["recourse_" + name] == sum(getattr(call["decision"], name) for call in calls)
    completed = [call["decision"] for call in calls
                 if call["decision"].eligible and not call["decision"].planning_overflow]
    assert work["recourse_risk_comparisons"] == sum(item.candidate_count for item in completed)
    assert work["recourse_ordering_comparisons"] == sum(
        item.candidate_count + item.risk_feasible_candidates - 2 for item in completed)
    assert work["recourse_strict_improvement_comparisons"] == len(completed)


def test_first_nonempty_context_counts_released_prefix_and_background_only(monkeypatch):
    options = (FrameOption("a", (1, 0), 2.5, ((2.5,), ())),
               FrameOption("b", (0, 1), 3., ((), (3.,))))
    candidate = OneStepBackgroundRedecisionLaneRollout(options, "released", sample_count=1)
    candidate.arrival_rates = (.02, .01)
    candidate._outer_observed_counts = (20, 10)
    candidate.decision_counter = 7
    script_background(monkeypatch, (1000., .25, 1000.))
    split_prefixes(monkeypatch, candidate)
    calls = capture_inner(monkeypatch, candidate)
    scored = candidate.score_background_roots(
        (1, 0), (1000.,), 1000., 0, ((5.,), ()), (population((3., 1)),))
    call, = calls
    # One empty poll at2.5 is skipped. At3.5 the prefix B@3 and background
    # B@3.25 are both visible; historical admitted jobs remain in counts.
    assert call["args"][1:] == ((0, 2), (1003.5,), 1003.5, 0,
                                  ((), (.5, .25)), (20, 12))
    assert call["counter"] == 9
    assert call["decision"].estimated_arrival_rates == (20 / 1003.5, 12 / 1003.5)
    assert candidate._recourse_work["recourse_arrival_counter_updates"] == 2
    assert scored.rollout_idle_calls == 1
    assert scored.costs == {0: Fraction(11)} and scored.squared_costs == {0: Fraction(137, 2)}
    # Once the tagged B is admitted its full completion is known, although
    # the later background B remains waiting and is not charged flow cost.
    assert scored.flow_evaluations == 2 and scored.background_admissions == 0


def test_unreleased_outer_future_does_not_enter_inner_redecision(monkeypatch):
    script_background(monkeypatch)
    contexts = []
    for coordinate in (0, 1):
        candidate = OneStepBackgroundRedecisionLaneRollout(OPTIONS, "unseen", sample_count=1)
        candidate.arrival_rates = (.02, .01)
        candidate._outer_observed_counts = (20, 10)
        split_prefixes(monkeypatch, candidate, inner=((.5, 1),))
        calls = capture_inner(monkeypatch, candidate)
        candidate.score_background_roots(
            (3, 1), (1000., 1006.), 1000., 0, ((7., 4., 0.), (2.,)),
            (population((2.5, coordinate)),))
        loaded_call, = [call for call in calls if call["args"][3] == 1002.
                       and call["args"][1] == (2, 0)]
        contexts.append((loaded_call["args"], loaded_call["rng_state"],
                         loaded_call["counter"], loaded_call["decision"]))
    assert contexts[0] == contexts[1]


def test_real_calls_freely_replan_without_inner_advancing_outer_rng_or_counter(monkeypatch):
    candidate = OneStepBackgroundRedecisionLaneRollout(OPTIONS, "online", sample_count=1)
    candidate.decision_counter = 7
    script_background(monkeypatch)
    sampled = split_prefixes(monkeypatch, candidate, inner=((.5, 1),), consume=True)
    calls = capture_inner(monkeypatch, candidate)
    expected_rng = Random("online")
    first = candidate(OPTIONS, (3, 1), (1000., 1006.), 1000., 0,
                      ((7., 4., 0.), (2.,)), (3, 1))
    expected_rng.random()
    assert candidate.rng.getstate() == expected_rng.getstate()
    assert all(call["rng_state"] == expected_rng.getstate() for call in calls)
    root, = first.selection.option_indices
    duration = first.selection.duration
    remaining = tuple(count - need for count, need in zip((3, 1), first.selection.service))
    ages = tuple(tuple(age + duration for age in group[need:])
                 for group, need in zip(((7., 4., 0.), (2.,)), first.selection.service))
    second = candidate(OPTIONS, remaining, (1000. + duration, 1006.),
                       1000. + duration, 0, ages, (3, 1))
    expected_rng.random()
    assert root is not None and second.eligible and second.sample_count == 1
    assert sum(item[0] for item in sampled) == 2
    assert candidate.decision_counter == 9 and candidate.rng.getstate() == expected_rng.getstate()
    assert not hasattr(candidate, "pending_successor_index")
    assert candidate.inner_planner.planning_seed == candidate.planning_seed
    assert {key: value for key, value in candidate.preprocessing_work.items()
            if key.startswith("recourse_")} == {
                "recourse_" + key: value for key, value in candidate.inner_planner.preprocessing_work.items()}


def test_inner_background_overflow_is_local_and_retains_partial_actual_work(monkeypatch):
    options = (FrameOption("a", (1,), 1., ((1.,),)),)
    candidate = OneStepBackgroundRedecisionLaneRollout(options, "local", sample_count=1, event_cap=2)
    script_background(monkeypatch, (.1, 1000.))
    split_prefixes(monkeypatch, candidate, inner=((.1, 0), (.2, 0)), consume=True)
    calls = capture_inner(monkeypatch, candidate)
    result = candidate(options, (2,), (1000., 1005.), 1000., 0, ((0., 0.),), (20,))
    inner, = [call["decision"] for call in calls]
    assert inner.planning_overflow and inner.background_rollout_overflow
    assert inner.rollout_calls == 1 and inner.flow_evaluations == 2
    assert result.recourse_planning_overflows == result.recourse_background_overflows == 1
    assert result.recourse_rollout_calls == 1 and result.recourse_flow_evaluations == 2
    assert result.recourse_tagged_planning_events == 2 and result.recourse_background_planning_events == 1
    assert result.recourse_planning_random_draws == inner.planning_random_draws > 0
    assert not result.planning_overflow and not result.background_rollout_overflow
    assert result.baseline_predicted_completion_s == result.chosen_predicted_completion_s == 3.
    assert result.flow_evaluations == 2 and result.background_admissions == 0


@pytest.mark.parametrize("prefix_overflow", [False, True])
def test_outer_overflow_restores_whole_decision_mw_and_preserves_work(monkeypatch, prefix_overflow):
    options = (FrameOption("a", (1,), 1., ((1.,),)),)
    candidate = OneStepBackgroundRedecisionLaneRollout(options, "outer", sample_count=1, event_cap=2)
    script_background(monkeypatch, (.1, .1, .1, 1000.))
    if prefix_overflow:
        monkeypatch.setattr(background_module, "sample_planning_arrivals",
                            lambda *args: population((.1, 0), (.2, 0), (.3, 0), overflow=True, draws=3))
    else:
        split_prefixes(monkeypatch, candidate, consume=True)
    calls = capture_inner(monkeypatch, candidate)
    args = (options, (3,), (1000., 1005., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
    result = candidate(*args)
    assert result.selection == candidate.maxweight(*args).selection
    assert result.planning_overflow and not result.changed
    assert result.baseline_predicted_completion_s is result.chosen_predicted_completion_s is None
    assert result.risk_feasible_candidates == result.risk_filtered_candidates == 0
    assert result.background_rollout_overflow == (not prefix_overflow)
    if prefix_overflow:
        assert not calls and result.recourse_calls == result.recourse_rng_words_copied == 0
        assert result.planning_random_draws == 3
    else:
        assert len(calls) == result.recourse_calls == 1
        assert result.background_planning_events == 3 and result.flow_evaluations == 2
        assert result.recourse_rollout_calls > 0 and result.recourse_planning_random_draws > 0
        assert result.recourse_rng_words_restored == len(candidate.rng.getstate()[1])
    assert candidate.decision_counter == 1


@pytest.mark.parametrize("cost,square,changed", [(19, 100, True), (20, 90, False), (19, 101, False)])
def test_outer_strict_s1_mw_ties_and_s2_gate_remain_unchanged(monkeypatch, cost, square, changed):
    candidate = OneStepBackgroundRedecisionLaneRollout(OPTIONS, "gate", sample_count=1)
    script_background(monkeypatch)
    split_prefixes(monkeypatch, candidate)
    score = candidate.score_background_roots

    def exact_contrast(*args):
        actual = score(*args)
        return replace(actual, costs={0: Fraction(cost), 1: Fraction(20)},
                       squared_costs={0: Fraction(square), 1: Fraction(100)})

    monkeypatch.setattr(candidate, "score_background_roots", exact_contrast)
    result = candidate(OPTIONS, (2, 0), (1000., 1006.), 1000., 0, ((4., 0.), ()), (20, 10))
    assert result.changed == changed and result.selection.option_indices == ((0,) if changed else (1,))
    assert result.recourse_calls > 0


def test_native_toy_driver_checks_and_work_include_both_layers_once():
    model = FrameModel("toy", "once", 2, ("a", "b"), (1., 1.), OPTIONS)
    tape = [(0., 0, 0), (0., 1, 0), (300., 0, 1), (300., 0, 2), (301., 1, 1)]
    candidate = OneStepBackgroundRedecisionLaneRollout(OPTIONS, "native")
    result = simulate_observed_rate_contextual_trajectory(model, tape, 305., 310., candidate, policy="once")
    assert result["frames"][0]["option_indices"] == (3,)
    assert result["frames"][1]["available_before"][0] == 2.
    assert sorted((job["coordinate"], job["completion"]) for job in result["jobs"]
                  if job["arrival"] == 0.) == [(0, .5), (1, 1.5)]
    protocol = {"planning_samples": 8, "planning_event_cap": 128}
    assert check_rollout(result, model, protocol, candidate)
    work = redecision_work(result, candidate, 305.)
    assert work["controller_constructors"] == work["recourse_controller_constructors"] == 1
    for key in ("recourse_calls", "recourse_root_option_probes", "recourse_rollout_option_probes",
                "recourse_rng_words_restored", "recourse_risk_comparisons"):
        assert work[key] == sum(frame["rollout"][key] for frame in result["frames"])
    assert work["recourse_calls"] > 0 and work["recourse_planning_random_draws"] > 0
    assert work["total_rollout_calls"] == work["rollout_calls"] + work["recourse_rollout_calls"]
    assert work["total_planning_events"] == work["planning_events"] + work["recourse_planning_events"]
    assert work["total_option_probes"] == (
        work["root_option_probes"] + work["rollout_option_probes"]
        + work["recourse_root_option_probes"] + work["recourse_rollout_option_probes"]
        + sum(work.get("preprocessing_" + prefix + key, 0)
              for prefix in ("", "recourse_")
              for key in ("option_probes", "age_profile_option_probes", "rate_dimension_option_probes")))
