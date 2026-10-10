"""Original-risk ablation keeps the old mask, native two-BG state, and real fees."""
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import chain, repeat

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.two_step_bg_incumbent_background_redecision_lane_rollout as two_module
import algorithm.two_step_bg_incumbent_original_risk_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_two_step_bg_incumbent_original_risk_development import (
    check_rollout, two_step_work,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import (
    TWO_STEP_WORK, TwoStepBGIncumbentBackgroundRedecisionLaneRollout,
)
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import TwoStepBGIncumbentOriginalRiskLaneRollout


OPTIONS = (FrameOption("single", (1,), 1., ((1.,),)),
           FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
           FrameOption("triple", (3,), 2., ((2., 2., 2.),)))
NATIVE_BG_SCORE = BackgroundCompetitionContinuousLaneRollout.score_background_roots


def populations(monkeypatch, inner_events=(), gaps=(1000.,)):
    class ScriptedRandom:
        def __init__(self, seed):
            self.gaps = chain(gaps, repeat(1000.))

        def expovariate(self, rate):
            return next(self.gaps)

    def empty_prefix(rates, window, rng, cap):
        draws = sum(rate > 0 for rate in rates) + len(inner_events)
        for _ in range(draws):
            rng.random()
        return PlanningArrivals(tuple(inner_events), False, draws, len(rates))

    monkeypatch.setattr(background_module, "Random", ScriptedRandom)
    monkeypatch.setattr(background_module, "sample_planning_arrivals", empty_prefix)
    monkeypatch.setattr(module, "sample_planning_arrivals", empty_prefix)


def contrast_case(monkeypatch, mode):
    populations(monkeypatch)
    candidate = TwoStepBGIncumbentOriginalRiskLaneRollout(OPTIONS, "original-risk:" + mode, sample_count=1)

    def ordinary(self, *args):
        scored = NATIVE_BG_SCORE(self, *args)
        if self is candidate and mode == "old_illegal":
            return replace(scored, squared_costs={**scored.squared_costs,
                0: scored.squared_costs[2] + 1})
        return scored

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", ordinary)
    native_score = candidate.score_background_roots

    def new_contrast(*args):
        native = native_score(*args)
        return replace(native,
            costs={0: Fraction(25 if mode == "tie" else 26 if mode == "worse" else 20),
                   1: Fraction(25), 2: Fraction(30)},
            squared_costs={0: Fraction(101), 1: Fraction(100), 2: Fraction(110)})

    monkeypatch.setattr(candidate, "score_background_roots", new_contrast)
    context = (OPTIONS, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))
    result = candidate(*context)
    assert result.incumbent_root_index == 1 and result.incumbent_maxweight_root_index == 2
    assert result.risk_feasible_candidates == result.incumbent_risk_feasible_candidates
    assert result.incumbent_lookahead_risk_comparisons == 0
    assert result.incumbent_membership_checks == result.candidate_count
    assert result.incumbent_lookahead_ordering_comparisons == (
        result.candidate_count + result.risk_feasible_candidates - 2)
    assert result.incumbent_lookahead_strict_improvement_comparisons == 1
    return result, candidate, context


def test_new_squared_risk_can_exceed_bg_when_old_risk_is_legal_and_s1_improves(monkeypatch):
    result, planner, context = contrast_case(monkeypatch, "improve")
    assert result.selection.option_indices == (0,) and result.improved_incumbent
    assert result.chosen_predicted_completion_s < result.baseline_predicted_completion_s
    assert result.chosen_predicted_squared_flow_s2 > result.baseline_predicted_squared_flow_s2
    assert result.incumbent_actual_predicted_squared_flow_s2 <= result.incumbent_mw_predicted_squared_flow_s2
    info = asdict(result)
    del info["selection"]
    frame = {"queue": context[1], "available_before": context[2], "start": context[3],
        "lane_ids": (context[4],), "waiting_ages": context[5], "observed_arrival_counts": context[6],
        "option_indices": result.selection.option_indices, "rollout": info}
    model = FrameModel("toy", "original-risk-gate", 2, ("a",), (1.,), OPTIONS)
    assert check_rollout({"frames": [frame]}, model, {"planning_samples": 1, "planning_event_cap": 128}, planner)
    rejected_old_definition = {**frame, "rollout": {**info, "incumbent_lookahead_risk_comparisons": 1}}
    assert not check_rollout({"frames": [rejected_old_definition]}, model,
        {"planning_samples": 1, "planning_event_cap": 128}, planner)


def test_old_s2_above_mw_is_rejected_even_with_lower_two_bg_s1(monkeypatch):
    result, _, _ = contrast_case(monkeypatch, "old_illegal")
    assert result.selection.option_indices == (1,) and not result.improved_incumbent
    assert result.unconstrained_root_index == 0 and result.risk_changed_unconstrained_root
    assert result.risk_filtered_candidates == result.incumbent_risk_filtered_candidates == 1
    assert result.risk_feasible_candidates == 2
    assert result.incumbent_actual_predicted_squared_flow_s2 <= result.incumbent_mw_predicted_squared_flow_s2
    assert result.chosen_predicted_completion_s == result.baseline_predicted_completion_s
    assert result.chosen_predicted_squared_flow_s2 == result.baseline_predicted_squared_flow_s2


def test_bg_first_tie_and_nonstrict_s1_both_retain_bg(monkeypatch):
    for mode in ("tie", "worse"):
        result, _, _ = contrast_case(monkeypatch, mode)
        assert result.selection.option_indices == (1,) and not result.improved_incumbent
        assert result.risk_feasible_candidates == result.incumbent_risk_feasible_candidates == 3
        assert result.chosen_predicted_completion_s == result.baseline_predicted_completion_s == 25.
        assert result.chosen_predicted_squared_flow_s2 == result.baseline_predicted_squared_flow_s2 == 100.


def test_prefix_old_background_and_new_background_overflows_keep_fallback_and_partial_fees(monkeypatch):
    options = OPTIONS + (FrameOption("slow-single", (1,), 2., ((2.,),)),)
    with monkeypatch.context() as patch:
        candidate = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "prefix-overflow", sample_count=1, event_cap=2)
        patch.setattr(module, "sample_planning_arrivals",
            lambda *args: PlanningArrivals(((.1, 0), (.2, 0), (.3, 0)), True, 3, 1))
        result = candidate(options, (3,), (1000., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
        assert result.selection.option_indices == (2,) and result.planning_overflow
        assert result.planning_random_draws == 3 and result.incumbent_candidate_count == 0
        assert result.recourse_calls == result.recourse_first_calls == result.recourse_second_calls == 0
    with monkeypatch.context() as patch:
        old_options = OPTIONS[:1]
        candidate = TwoStepBGIncumbentOriginalRiskLaneRollout(old_options, "old-overflow", sample_count=1, event_cap=2)
        populations(patch, gaps=(.1, .1, .1))
        result = candidate(old_options, (3,), (1000., 1005., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
        assert result.selection.option_indices == (0,) and result.incumbent_background_rollout_overflow
        assert result.incumbent_flow_evaluations == 2 and result.incumbent_background_planning_events == 3
        assert result.recourse_calls == result.recourse_first_calls == result.recourse_second_calls == 0
    with monkeypatch.context() as patch:
        candidate = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "new-overflow", sample_count=1, event_cap=2)
        populations(patch, gaps=(.1, .1, .1))
        original_call = BackgroundCompetitionContinuousLaneRollout.__call__

        def slow_child(self, *args):
            decision = original_call(self, *args)
            if self is candidate.inner_planner:
                return replace(decision, selection=replace(decision.selection,
                    option_counts=(0, 0, 0, 1), option_indices=(3,), service=(1,), duration=2., exact=False),
                    changed=True)
            return decision

        patch.setattr(BackgroundCompetitionContinuousLaneRollout, "__call__", slow_child)
        result = candidate(options, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))
        assert not result.incumbent_background_rollout_overflow
        assert result.background_rollout_overflow and result.planning_overflow
        assert result.incumbent_root_index == 1 and result.incumbent_maxweight_root_index == 2
        assert result.selection.option_indices == (1,) and result.changed and not result.improved_incumbent
        assert result.flow_evaluations == 2 and result.background_planning_events == 3
        assert result.recourse_first_calls == result.recourse_calls == 1 and result.recourse_second_calls == 0
        assert result.recourse_rollout_option_probes > 0 and result.recourse_planning_random_draws > 0
        assert result.recourse_rng_words_restored == 625
        assert result.incumbent_membership_checks == result.incumbent_lookahead_risk_comparisons == 0


def test_native_two_bg_scorer_state_and_work_match_frozen_parent_and_driver_charges_once(monkeypatch):
    options = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
               FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
               FrameOption("b1", (0, 1), 2., ((), (2.,))),
               FrameOption("loaded", (1, 1), 2., ((.5,), (1.5,)), loaded=True))
    populations(monkeypatch, inner_events=((.5, 1),))
    new = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "original-risk-native", sample_count=1)
    parent = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(options, "original-risk-native", sample_count=1)
    for planner in (new, parent):
        planner.arrival_rates = (.003, .001)
        planner._outer_observed_counts = (3, 1)
        planner.decision_counter = 7
    before = new.rng.getstate()
    args = ((3, 1), (1000., 1006.), 1000., 0, ((7., 4., 0.), (2.,)),
            (PlanningArrivals(((2.5, 1),), False, 0, 0),))
    actual, expected = new.score_background_roots(*args), parent.score_background_roots(*args)
    assert actual == expected
    assert actual.costs[3] == Fraction(53, 2) and actual.squared_costs[3] == Fraction(647, 4)
    assert new._recourse_work == parent._recourse_work and new._two_step_work == parent._two_step_work
    assert new._two_step_work["recourse_first_calls"] == new._two_step_work["recourse_second_calls"] == 4
    assert new._recourse_work["recourse_calls"] == 8
    assert new.rng.getstate() == parent.rng.getstate() == before
    assert new.decision_counter == parent.decision_counter == 7
    populations(monkeypatch)
    model = FrameModel("toy", "original-risk-driver", 2, ("a", "b"), (1., 1.), options)
    tape = [(0., 0, 0), (0., 1, 0), (300., 0, 1), (300., 0, 2), (300., 0, 3), (300., 1, 1)]
    planner = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "original-risk-driver")
    result = simulate_observed_rate_contextual_trajectory(model, tape, 305., 310., planner, policy="original-risk")
    assert check_rollout(result, model, {"planning_samples": 8, "planning_event_cap": 128}, planner)
    work = two_step_work(result, planner, 305.)
    assert work["controller_constructors"] == work["recourse_controller_constructors"] == 1
    for name in TWO_STEP_WORK + ("recourse_calls", "incumbent_root_option_probes", "recourse_root_option_probes"):
        assert work[name] == sum(frame["rollout"][name] for frame in result["frames"])
    assert work["recourse_first_calls"] > 0 and work["recourse_second_calls"] > 0
    assert work["recourse_calls"] == work["recourse_first_calls"] + work["recourse_second_calls"]
    assert work["incumbent_lookahead_risk_comparisons"] == 0
    assert work["recourse_rng_words_restored"] == 625 * work["recourse_first_calls"]
    assert work["total_option_probes"] == sum(work[name] for name in (
        "root_option_probes", "rollout_option_probes", "incumbent_root_option_probes",
        "incumbent_rollout_option_probes", "recourse_root_option_probes", "recourse_rollout_option_probes",
        "preprocessing_option_probes", "preprocessing_age_profile_option_probes",
        "preprocessing_rate_dimension_option_probes", "preprocessing_recourse_option_probes",
        "preprocessing_recourse_age_profile_option_probes", "preprocessing_recourse_rate_dimension_option_probes"))
