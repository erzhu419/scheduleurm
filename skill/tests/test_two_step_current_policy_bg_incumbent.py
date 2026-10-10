"""A finite current-policy continuation preserves native state, costs, and fees."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from itertools import chain, repeat

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.two_step_bg_incumbent_original_risk_lane_rollout as frozen_module
import algorithm.two_step_current_policy_bg_incumbent_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, _LazyBackgroundArrivals,
)
from algorithm.bg_incumbent_background_redecision_lane_rollout import INCUMBENT_WORK
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_two_step_current_policy_bg_incumbent_development import (
    check_rollout, two_step_work,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.one_step_background_redecision_lane_rollout import INNER_WORK, RECOURSE_WORK
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import TwoStepBGIncumbentOriginalRiskLaneRollout
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import TwoStepCurrentPolicyBGIncumbentLaneRollout


SINGLE = (FrameOption("single", (1,), 1., ((1.,),)),)
ROOTS = SINGLE + (FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
                  FrameOption("triple", (3,), 2., ((2., 2., 2.),)))
NATIVE_BG_SCORE = BackgroundCompetitionContinuousLaneRollout.score_background_roots
NESTED_WORK = INCUMBENT_WORK + RECOURSE_WORK + TWO_STEP_WORK


def populations(monkeypatch, gaps=(1000.,)):
    class ScriptedRandom:
        def __init__(self, seed):
            self.gaps = chain(gaps, repeat(1000.))

        def expovariate(self, rate):
            return next(self.gaps)

    def no_future(rates, window, rng, cap):
        draws = sum(rate > 0 for rate in rates)
        for _ in range(draws):
            rng.random()
        return PlanningArrivals((), False, draws, len(rates))

    monkeypatch.setattr(background_module, "Random", ScriptedRandom)
    monkeypatch.setattr(background_module, "sample_planning_arrivals", no_future)
    monkeypatch.setattr(frozen_module, "sample_planning_arrivals", no_future)


def trace_current(monkeypatch, candidate):
    original = TwoStepBGIncumbentOriginalRiskLaneRollout.__call__
    calls = []

    def traced(self, *args):
        if self is not candidate.inner_planner:
            return original(self, *args)
        record = {"args": args, "rng_state": self.rng.getstate(), "counter": self.decision_counter}
        result = original(self, *args)
        record.update(decision=result, rng_after=self.rng.getstate(), counter_after=self.decision_counter)
        calls.append(record)
        return result

    monkeypatch.setattr(TwoStepBGIncumbentOriginalRiskLaneRollout, "__call__", traced)
    return calls


def test_two_frozen_current_calls_keep_loaded_period_real_idle_and_continued_state(monkeypatch):
    options = (FrameOption("a1", (1, 0), 1., ((1.,), ())),
               FrameOption("a3", (3, 0), 4., ((4., 4., 4.), ())),
               FrameOption("b1", (0, 1), 1., ((), (1.,))),
               FrameOption("loaded-b1", (0, 1), 1.5, ((), (.5,)), loaded=True))
    populations(monkeypatch)
    candidate = TwoStepCurrentPolicyBGIncumbentLaneRollout(options, "current-two-idle", sample_count=1)
    assert type(candidate.inner_planner) is TwoStepBGIncumbentOriginalRiskLaneRollout
    assert type(candidate.inner_planner.inner_planner) is BackgroundCompetitionContinuousLaneRollout
    assert not hasattr(candidate.inner_planner.inner_planner, "inner_planner")
    candidate.arrival_rates = (.02, .01)
    candidate._outer_observed_counts = (20, 10)
    candidate.decision_counter = 7
    calls = trace_current(monkeypatch, candidate)
    before = candidate.rng.getstate()
    scored = candidate.score_background_roots((1, 0), (1000.,), 1000., 0, ((5.,), ()),
        (PlanningArrivals(((1.5, 1), (3.8, 0)), False, 0, 0),))
    first, second = calls
    assert first["args"][1:] == ((0, 1), (1002.,), 1002., 0, ((), (.5,)), (20, 11))
    assert first["decision"].selection.option_indices == (3,)
    assert first["decision"].incumbent_maxweight_root_index == 2
    assert second["args"][1:5] == ((1, 0), (1004.5,), 1004.5, 0)
    assert second["args"][5][0] == pytest.approx((.7,))
    assert second["args"][6] == (21, 11)
    assert first["rng_state"] == before and second["rng_state"] == first["rng_after"] != before
    assert first["counter"] == 9 and first["counter_after"] == 10
    assert second["counter"] == 11 and second["counter_after"] == 12
    assert scored.rollout_idle_calls == 2 and scored.flow_evaluations == 3
    assert candidate._two_step_work["recourse_first_calls"] == candidate._two_step_work["recourse_second_calls"] == 1
    assert candidate._two_step_work["recourse_virtual_idle_counter_updates"] == 1
    assert candidate._recourse_work["recourse_rng_words_restored"] == 625
    assert candidate._recourse_work["recourse_counter_reads"] == candidate._recourse_work["recourse_counter_writes"] == 2
    assert candidate._recourse_work["recourse_arrival_counter_updates"] == 2
    assert candidate.rng.getstate() == before and candidate.decision_counter == 7


def test_nested_native_work_and_three_controller_preprocessing_are_charged_once(monkeypatch):
    populations(monkeypatch)
    candidate = TwoStepCurrentPolicyBGIncumbentLaneRollout(SINGLE, "nested-native", sample_count=1)
    calls = trace_current(monkeypatch, candidate)
    decision = candidate(SINGLE, (4,), (1000., 1005., 1005., 1005.), 1000., 0,
                         ((0., 0., 0., 0.),), (4,))
    first, second = calls
    assert [call["args"][1] for call in calls] == [(3,), (2,)]
    assert decision.recourse_calls == decision.recourse_first_calls + decision.recourse_second_calls == 2
    assert first["decision"].recourse_calls == 2 and second["decision"].recourse_calls == 1
    assert decision.chosen_predicted_completion_s == 10. and decision.chosen_predicted_squared_flow_s2 == 30.
    expected = Counter()
    for call in calls:
        expected.update({name: getattr(call["decision"], name) for name in NESTED_WORK})
    assert Counter(decision.inner_current_work) == expected
    assert expected["recourse_calls"] == 3 and expected["recourse_first_calls"] == 2
    assert expected["recourse_second_calls"] == 1 and expected["incumbent_root_option_probes"] > 0
    assert decision.recourse_risk_comparisons == decision.recourse_ordering_comparisons == decision.recourse_strict_improvement_comparisons == 0
    for name in INNER_WORK:
        assert getattr(decision, "recourse_" + name) == sum(getattr(call["decision"], name) for call in calls)
    model = FrameModel("toy", "nested-driver", 4, ("a",), (1.,), SINGLE)
    planner = TwoStepCurrentPolicyBGIncumbentLaneRollout(SINGLE, "nested-driver")
    assert planner.sample_count == planner.inner_planner.sample_count == planner.inner_planner.inner_planner.sample_count == 8
    assert planner.event_cap == planner.inner_planner.event_cap == planner.inner_planner.inner_planner.event_cap == 128
    result = simulate_observed_rate_contextual_trajectory(model, [(300., 0, index) for index in range(4)],
        305., 310., planner, policy="current-policy-continuation")
    assert check_rollout(result, model, {"planning_samples": 8, "planning_event_cap": 128}, planner)
    work = two_step_work(result, planner, 305.)
    assert work["controller_constructors"] == 1 and work["recourse_controller_constructors"] == 2
    assert work["total_controller_constructors"] == 3
    assert work["preprocessing_option_probes"] == work["preprocessing_recourse_option_probes"] == work["preprocessing_recourse_recourse_option_probes"] == 1
    assert work["inner_current_recourse_calls"] > 0 and work["inner_current_recourse_second_calls"] > 0
    expected = Counter()
    for frame in result["frames"]:
        expected.update(frame["rollout"]["inner_current_work"])
    assert Counter({name: work["inner_current_" + name] for name in NESTED_WORK}) == expected
    assert work["total_option_probes"] == sum(value for name, value in work.items()
                                             if name.endswith("option_probes") and name != "total_option_probes")


def test_frozen_inner_second_stage_overflow_returns_verified_bg_with_partial_native_fees(monkeypatch):
    options = ROOTS + (FrameOption("slow-single", (1,), 2., ((2.,),)),)
    populations(monkeypatch, gaps=(.1, .1, .1))

    class QuietSuffix(_LazyBackgroundArrivals):
        def release_through(self, start):
            original = background_module.Random

            class QuietRandom:
                def __init__(self, seed):
                    pass

                def expovariate(self, rate):
                    return 1000.

            background_module.Random = QuietRandom
            try:
                return super().release_through(start)
            finally:
                background_module.Random = original

    monkeypatch.setattr(module, "_LazyBackgroundArrivals", QuietSuffix)
    candidate = TwoStepCurrentPolicyBGIncumbentLaneRollout(options, "nested-overflow", sample_count=1, event_cap=2)
    calls = trace_current(monkeypatch, candidate)
    ordinary_call = BackgroundCompetitionContinuousLaneRollout.__call__

    def slow_grandchild(self, *args):
        result = ordinary_call(self, *args)
        if self is candidate.inner_planner.inner_planner:
            return replace(result, selection=replace(result.selection,
                option_counts=(0, 0, 0, 1), option_indices=(3,), service=(1,), duration=2., exact=False), changed=True)
        return result

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "__call__", slow_grandchild)
    decision = candidate(options, (4,), (1000., 1005.), 1000., 0, ((20., 10., 0., 0.),), (20,))
    partial = [call["decision"] for call in calls if call["decision"].background_rollout_overflow
               and not call["decision"].incumbent_background_rollout_overflow]
    assert partial
    for inner in partial:
        assert inner.planning_overflow and not inner.improved_incumbent
        assert inner.selection.option_indices == (inner.incumbent_root_index,)
        assert inner.incumbent_root_index != inner.incumbent_maxweight_root_index
        assert inner.recourse_calls > 0 and inner.recourse_rollout_option_probes > 0
        assert inner.incumbent_flow_evaluations > 0 and inner.flow_evaluations > 0
    assert decision.inner_current_fallbacks["second_stage_overflows"] == len(partial)
    assert decision.inner_current_fallbacks["BG_fallbacks"] == len(partial)
    assert decision.inner_current_work["recourse_calls"] == sum(call["decision"].recourse_calls for call in calls)
    assert decision.inner_current_work["incumbent_flow_evaluations"] == sum(call["decision"].incumbent_flow_evaluations for call in calls)
    assert decision.inner_current_work["recourse_planning_random_draws"] > 0


def test_unreleased_outer_future_cannot_change_either_current_call_and_after_two_calls_uses_mw(monkeypatch):
    populations(monkeypatch)
    options = SINGLE + (FrameOption("five", (5,), 8., ((8., 8., 8., 8., 8.),)),)
    traces, costs = [], []
    for events in ((), ((5., 0), (5.5, 0))):
        candidate = TwoStepCurrentPolicyBGIncumbentLaneRollout(options, "causal-current-two", sample_count=1)
        candidate.arrival_rates = (.004,)
        candidate._outer_observed_counts = (4,)
        candidate.decision_counter = 7
        calls = trace_current(monkeypatch, candidate)
        before = candidate.rng.getstate()
        scored = candidate.score_background_roots((4,), (1000., 1010., 1010., 1010.), 1000., 0,
            ((0., 0., 0., 0.),), (PlanningArrivals(events, False, 0, 0),))
        assert len(calls) == 2 and [call["args"][1] for call in calls] == [(3,), (2,)]
        assert [call["args"][6] for call in calls] == [(4,), (4,)]
        assert candidate._recourse_work["recourse_arrival_counter_updates"] == 0
        assert candidate._two_step_work["recourse_first_calls"] == candidate._two_step_work["recourse_second_calls"] == 1
        assert scored.rollout_calls > len(calls)
        assert candidate.rng.getstate() == before and candidate.decision_counter == 7
        traces.append(calls)
        costs.append(scored.costs[0])
    assert traces[0] == traces[1]
    assert costs == [Fraction(10), Fraction(25, 2)]


def test_current_continuation_changes_root_only_inside_original_mask_and_strict_bg_ties(monkeypatch):
    populations(monkeypatch)
    for mode, selected in (("improve", 0), ("old_illegal", 1), ("tie", 1)):
        candidate = TwoStepCurrentPolicyBGIncumbentLaneRollout(ROOTS, "current-gate:" + mode, sample_count=1)

        def original_mask(self, *args):
            scored = NATIVE_BG_SCORE(self, *args)
            if self is candidate and mode == "old_illegal":
                return replace(scored, squared_costs={**scored.squared_costs, 0: scored.squared_costs[2] + 1})
            return scored

        monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", original_mask)
        native_score = candidate.score_background_roots

        def contrast(*args):
            scored = native_score(*args)
            return replace(scored, costs={0: Fraction(25 if mode == "tie" else 20), 1: Fraction(25), 2: Fraction(30)},
                           squared_costs={0: Fraction(101), 1: Fraction(100), 2: Fraction(110)})

        monkeypatch.setattr(candidate, "score_background_roots", contrast)
        result = candidate(ROOTS, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))
        assert result.incumbent_root_index == 1 and result.incumbent_maxweight_root_index == 2
        assert result.selection.option_indices == (selected,) and result.improved_incumbent == (mode == "improve")
        assert result.risk_feasible_candidates == result.incumbent_risk_feasible_candidates
        assert result.incumbent_lookahead_risk_comparisons == result.recourse_risk_comparisons == 0
        assert result.incumbent_actual_predicted_squared_flow_s2 <= result.incumbent_mw_predicted_squared_flow_s2
        assert result.inner_current_work["incumbent_root_option_probes"] > 0
        if mode == "improve":
            assert result.chosen_predicted_squared_flow_s2 > result.baseline_predicted_squared_flow_s2
        else:
            assert result.chosen_predicted_completion_s == result.baseline_predicted_completion_s == 25.
            assert result.chosen_predicted_squared_flow_s2 == result.baseline_predicted_squared_flow_s2 == 100.
