"""Two virtual BG calls preserve causal FIFO, continued state, and inherited gates."""
from dataclasses import replace
from fractions import Fraction
from itertools import chain, repeat

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.bg_incumbent_background_redecision_lane_rollout as incumbent_module
import algorithm.two_step_bg_incumbent_background_redecision_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, _LazyBackgroundArrivals,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_two_step_bg_incumbent_development import (
    check_rollout, two_step_work,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import (
    TWO_STEP_WORK, TwoStepBGIncumbentBackgroundRedecisionLaneRollout,
)


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 2., ((), (2.,))),
           FrameOption("loaded", (1, 1), 2., ((.5,), (1.5,)), loaded=True))


def population(*events, overflow=False, draws=0):
    return PlanningArrivals(tuple(events), overflow, draws, 0)


def script_background(monkeypatch, gaps=(1000.,)):
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


def prefixes(monkeypatch, inner_events=()):
    def sampled(rates, window, rng, cap):
        draws = sum(rate > 0 for rate in rates) + len(inner_events)
        for _ in range(draws):
            rng.random()
        return PlanningArrivals(tuple(inner_events), False, draws, len(rates))

    monkeypatch.setattr(background_module, "sample_planning_arrivals", sampled)
    monkeypatch.setattr(incumbent_module, "sample_planning_arrivals", sampled)


def capture_inner(monkeypatch, candidate, force_index=None):
    original = BackgroundCompetitionContinuousLaneRollout.__call__
    calls = []

    def traced(self, *args):
        if self is not candidate.inner_planner:
            return original(self, *args)
        record = {"args": args, "rng_state": self.rng.getstate(),
                  "counter": self.decision_counter}
        result = original(self, *args)
        if force_index is not None:
            option = candidate.options[force_index]
            result = replace(result, selection=replace(result.selection,
                option_counts=tuple(int(index == force_index) for index in range(len(candidate.options))),
                option_indices=(force_index,), service=option.requirements,
                duration=option.duration, exact=False), changed=True)
        record.update(decision=result, rng_after=self.rng.getstate(), counter_after=self.decision_counter)
        calls.append(record)
        return result

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "__call__", traced)
    return calls


def test_native_two_bg_calls_keep_loaded_period_fifo_released_counts_and_continued_rng(monkeypatch):
    candidate = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "native-two", sample_count=1)
    candidate.arrival_rates = (.003, .001)
    candidate._outer_observed_counts = (3, 1)
    candidate.decision_counter = 7
    script_background(monkeypatch)
    prefixes(monkeypatch, inner_events=((.5, 1),))
    calls = capture_inner(monkeypatch, candidate)
    before = candidate.rng.getstate()
    scored = candidate.score_background_roots(
        (3, 1), (1000., 1006.), 1000., 0, ((7., 4., 0.), (2.,)),
        (population((2.5, 1)),))
    index, = [index for index, call in enumerate(calls)
              if call["args"][1] == (2, 0) and call["args"][3] == 1002.]
    first, second = calls[index:index + 2]
    assert first["args"][2:7] == ((1002., 1006.), 1002., 0, ((6., 2.), ()), (3, 1))
    assert first["decision"].selection.option_indices == (0,)
    assert first["decision"].changed  # Native BG differs from the Q2 pair MW root.
    assert second["args"][1:] == ((1, 1), (1004., 1006.), 1004., 0,
                                    ((4.,), (1.5,)), (3, 2))
    assert second["decision"].selection.option_indices == (3,)
    assert second["decision"].estimated_arrival_rates == (3 / 1004., 2 / 1004.)
    assert first["rng_state"] == before
    assert second["rng_state"] == first["rng_after"] != before
    assert first["counter"] == 8 and first["counter_after"] == second["counter"] == 9
    assert second["counter_after"] == 10
    assert scored.costs[3] == Fraction(53, 2)
    assert scored.squared_costs[3] == Fraction(647, 4)
    assert scored.flow_evaluations == 4 * 5 and scored.background_admissions == 0
    assert candidate.rng.getstate() == before and candidate.decision_counter == 7
    work, steps = candidate._recourse_work, candidate._two_step_work
    assert steps["recourse_first_calls"] == steps["recourse_second_calls"] == 4
    assert steps["recourse_virtual_idle_counter_updates"] == 0
    assert work["recourse_calls"] == len(calls) == 8
    assert work["recourse_rng_words_restored"] == 625 * 4
    assert work["recourse_counter_reads"] == 1 and work["recourse_counter_writes"] == 4
    for name in ("root_option_probes", "rollout_calls", "rollout_option_probes",
                 "planning_events", "planning_random_draws", "flow_evaluations"):
        assert work["recourse_" + name] == sum(getattr(call["decision"], name) for call in calls)


def test_idle_between_nonempty_steps_advances_counter_without_resetting_rng_or_spending_step(monkeypatch):
    options = (FrameOption("a1", (1, 0), 1., ((1.,), ())),
               FrameOption("a3", (3, 0), 4., ((4., 4., 4.), ())),
               FrameOption("b1", (0, 1), 1., ((), (1.,))))
    candidate = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(options, "two-idle", sample_count=1)
    candidate.arrival_rates = (.02, .01)
    candidate._outer_observed_counts = (20, 10)
    candidate.decision_counter = 7
    script_background(monkeypatch)
    prefixes(monkeypatch)
    calls = capture_inner(monkeypatch, candidate)
    before = candidate.rng.getstate()
    scored = candidate.score_background_roots(
        (1, 0), (1000.,), 1000., 0, ((5.,), ()),
        (population((1.5, 1), (3.2, 0)),))
    first, second = calls
    assert first["args"][1:] == ((0, 1), (1002.,), 1002., 0, ((), (.5,)), (20, 11))
    assert second["args"][1:5] == ((1, 0), (1004.,), 1004., 0)
    assert second["args"][5][0] == pytest.approx((.8,))
    assert second["args"][6] == (21, 11)
    assert first["counter"] == 9 and first["counter_after"] == 10
    assert second["counter"] == 11 and second["counter_after"] == 12
    assert second["rng_state"] == first["rng_after"] != before
    assert scored.rollout_idle_calls == 2 and scored.rollout_calls == 4
    assert candidate._two_step_work["recourse_first_calls"] == 1
    assert candidate._two_step_work["recourse_second_calls"] == 1
    assert candidate._two_step_work["recourse_virtual_idle_counter_updates"] == 1
    assert candidate._recourse_work["recourse_counter_reads"] == 2
    assert candidate._recourse_work["recourse_counter_writes"] == 2
    assert candidate._recourse_work["recourse_rng_words_restored"] == 625
    assert candidate._recourse_work["recourse_arrival_counter_updates"] == 2
    assert candidate.rng.getstate() == before and candidate.decision_counter == 7


def test_first_inner_overflow_spends_first_budget_and_second_continues_then_mw(monkeypatch):
    options = (FrameOption("single", (1,), 1., ((1.,),)),)
    candidate = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(
        options, "two-local-overflow", sample_count=1, event_cap=2)
    candidate.arrival_rates = (.004,)
    candidate._outer_observed_counts = (4,)
    candidate.decision_counter = 7
    script_background(monkeypatch)
    samples = []

    def sampled(rates, window, rng, cap):
        first = not samples
        draws = 3 if first else 1
        for _ in range(draws):
            rng.random()
        result = PlanningArrivals(((.1, 0), (.2, 0), (.3, 0)) if first else (),
                                  first, draws, len(rates))
        samples.append(result)
        return result

    monkeypatch.setattr(background_module, "sample_planning_arrivals", sampled)
    calls = capture_inner(monkeypatch, candidate)
    before = candidate.rng.getstate()
    scored = candidate.score_background_roots(
        (4,), (1000., 1005., 1005., 1005.), 1000., 0, ((0., 0., 0., 0.),), (population(),))
    first, second = calls
    assert first["decision"].planning_overflow and first["decision"].selection.option_indices == (0,)
    assert not second["decision"].planning_overflow
    assert second["rng_state"] == first["rng_after"] != before
    assert first["counter"] == 8 and second["counter"] == first["counter_after"] == 9
    assert len(samples) == 2 and candidate._recourse_work["recourse_planning_random_draws"] == 4
    assert candidate._recourse_work["recourse_planning_overflows"] == 1
    assert candidate._two_step_work["recourse_first_calls"] == candidate._two_step_work["recourse_second_calls"] == 1
    assert candidate._recourse_work["recourse_calls"] == 2 and scored.rollout_calls == 3
    assert scored.costs == {0: Fraction(10)} and scored.squared_costs == {0: Fraction(30)}
    assert not scored.background_rollout_overflow
    assert candidate._recourse_work["recourse_rng_words_restored"] == 625
    assert candidate.rng.getstate() == before and candidate.decision_counter == 7


def test_inherited_old_mask_new_aggregate_gate_and_bg_first_strict_ties(monkeypatch):
    options = (FrameOption("single", (1,), 1., ((1.,),)),
               FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
               FrameOption("triple", (3,), 2., ((2., 2., 2.),)))
    script_background(monkeypatch)
    prefixes(monkeypatch)
    ordinary_score = BackgroundCompetitionContinuousLaneRollout.score_background_roots
    for mode, actual in (("improve", 0), ("tie", 1), ("new_risk", 1), ("old_mask", 1)):
        candidate = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(options, mode, sample_count=1)

        def ordinary(self, *args):
            result = ordinary_score(self, *args)
            if self is candidate and mode == "old_mask":
                return replace(result, squared_costs={**result.squared_costs,
                    0: result.squared_costs[2] + 1})
            return result

        monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", ordinary)
        native_score = candidate.score_background_roots

        def contrast(*args):
            scored = native_score(*args)
            return replace(scored,
                costs={0: Fraction(25 if mode == "tie" else 20), 1: Fraction(25), 2: Fraction(30)},
                squared_costs={0: Fraction(101 if mode == "new_risk" else 90),
                               1: Fraction(100), 2: Fraction(110)})

        monkeypatch.setattr(candidate, "score_background_roots", contrast)
        result = candidate(options, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))
        assert result.incumbent_root_index == 1 and result.incumbent_maxweight_root_index == 2
        assert result.selection.option_indices == (actual,)
        assert result.improved_incumbent == (mode == "improve")
        assert result.incumbent_actual_predicted_squared_flow_s2 <= result.incumbent_mw_predicted_squared_flow_s2
        assert result.chosen_predicted_squared_flow_s2 <= result.baseline_predicted_squared_flow_s2
        assert result.recourse_first_calls > 0
        assert result.recourse_calls == result.recourse_first_calls + result.recourse_second_calls


def test_prefix_old_background_and_new_background_overflows_keep_correct_fallback_and_partial_fees(monkeypatch):
    options = (FrameOption("single", (1,), 1., ((1.,),)),
               FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
               FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
               FrameOption("slow-single", (1,), 2., ((2.,),)))
    with monkeypatch.context() as patch:
        candidate = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(options, "prefix-overflow", sample_count=1, event_cap=2)
        patch.setattr(incumbent_module, "sample_planning_arrivals",
                      lambda *args: population((.1, 0), (.2, 0), (.3, 0), overflow=True, draws=3))
        result = candidate(options, (3,), (1000., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
        assert result.selection.option_indices == (2,) and result.planning_overflow
        assert result.planning_random_draws == 3 and result.incumbent_candidate_count == 0
        assert result.recourse_first_calls == result.recourse_second_calls == result.recourse_calls == 0
    with monkeypatch.context() as patch:
        old_options = options[:1]
        candidate = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(old_options, "old-overflow", sample_count=1, event_cap=2)
        script_background(patch, (.1, .1, .1))
        prefixes(patch)
        result = candidate(old_options, (3,), (1000., 1005., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
        assert result.selection.option_indices == (0,) and result.incumbent_background_rollout_overflow
        assert result.incumbent_flow_evaluations == 2 and result.incumbent_background_planning_events == 3
        assert result.recourse_first_calls == result.recourse_second_calls == 0
    with monkeypatch.context() as patch:
        candidate = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(options, "new-overflow", sample_count=1, event_cap=2)
        script_background(patch, (.1, .1, .1))
        prefixes(patch)
        calls = capture_inner(patch, candidate, force_index=3)
        result = candidate(options, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))
        assert not result.incumbent_background_rollout_overflow
        assert result.background_rollout_overflow and result.planning_overflow
        assert result.incumbent_root_index == 1 and result.incumbent_maxweight_root_index == 2
        assert result.selection.option_indices == (1,) and result.changed and not result.improved_incumbent
        assert result.flow_evaluations == 2 and result.background_planning_events == 3
        assert result.recourse_first_calls == result.recourse_calls == len(calls) == 1
        assert result.recourse_second_calls == result.recourse_virtual_idle_counter_updates == 0
        assert result.recourse_rollout_option_probes > 0 and result.recourse_planning_random_draws > 0
        assert result.recourse_rng_words_restored == 625


def test_small_native_driver_checks_and_all_three_layers_are_charged_once():
    model = FrameModel("toy", "two-bg", 2, ("a", "b"), (1., 1.), OPTIONS)
    tape = [(0., 0, 0), (0., 1, 0), (300., 0, 1), (300., 0, 2), (300., 0, 3), (300., 1, 1)]
    candidate = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "two-native-driver")
    result = simulate_observed_rate_contextual_trajectory(model, tape, 305., 310., candidate, policy="two-bg")
    assert result["frames"][0]["option_indices"] == (3,)
    assert result["frames"][1]["available_before"][0] == 2.
    assert sorted((job["coordinate"], job["completion"]) for job in result["jobs"]
                  if job["arrival"] == 0.) == [(0, .5), (1, 1.5)]
    assert check_rollout(result, model, {"planning_samples": 8, "planning_event_cap": 128}, candidate)
    work = two_step_work(result, candidate, 305.)
    assert work["controller_constructors"] == work["recourse_controller_constructors"] == 1
    for name in TWO_STEP_WORK + ("recourse_calls", "incumbent_root_option_probes", "recourse_root_option_probes"):
        assert work[name] == sum(frame["rollout"][name] for frame in result["frames"])
    assert work["recourse_first_calls"] > 0 and work["recourse_second_calls"] > 0
    assert work["recourse_calls"] == work["recourse_first_calls"] + work["recourse_second_calls"]
    assert work["recourse_rng_words_restored"] == 625 * work["recourse_first_calls"]
    assert work["recourse_counter_writes"] == work["recourse_first_calls"] + work["recourse_virtual_idle_counter_updates"]
    assert work["total_option_probes"] == sum(work[name] for name in (
        "root_option_probes", "rollout_option_probes", "incumbent_root_option_probes",
        "incumbent_rollout_option_probes", "recourse_root_option_probes", "recourse_rollout_option_probes",
        "preprocessing_option_probes", "preprocessing_age_profile_option_probes",
        "preprocessing_rate_dimension_option_probes", "preprocessing_recourse_option_probes",
        "preprocessing_recourse_age_profile_option_probes", "preprocessing_recourse_rate_dimension_option_probes"))
