"""Class protection applies to the same once-BG model and tagged FIFO jobs."""
from dataclasses import fields, replace
from fractions import Fraction
from itertools import chain, repeat

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.one_step_background_redecision_lane_rollout as recourse_module
import algorithm.class_protected_bg_incumbent_background_redecision_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundRootScores, _LazyBackgroundArrivals,
)
from algorithm.class_protected_bg_incumbent_background_redecision_lane_rollout import (
    ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.one_step_background_redecision_lane_rollout import OneStepBackgroundRedecisionLaneRollout


OPTIONS = (FrameOption("a1", (1, 0), 1., ((1.,), ())),
           FrameOption("a2", (2, 0), 1.5, ((1.5, 1.5), ())),
           FrameOption("a3", (3, 0), 2., ((2., 2., 2.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))


def population(*events, overflow=False):
    return PlanningArrivals(tuple(events), overflow, 0, 0)


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

    for target in (background_module, recourse_module, module):
        monkeypatch.setattr(target, "_LazyBackgroundArrivals", ScriptedSuffix)


def empty_prefixes(monkeypatch):
    for target in (module, background_module):
        monkeypatch.setattr(target, "sample_planning_arrivals", lambda *args: population())


@pytest.mark.parametrize("mode,actual,improved,aggregate_feasible,class_filtered", [
    ("class_transfer", 1, False, 2, 1),
    ("class_safe", 0, True, 2, 0),
    ("tie", 1, False, 2, 0),
    ("bg_relative", 0, True, 2, 0),
    ("old_illegal", 1, False, 1, 0),
])
def test_same_model_bg_class_gate_and_strict_improvement(
        monkeypatch, mode, actual, improved, aggregate_feasible, class_filtered):
    candidate = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        OPTIONS, "gate", sample_count=1)
    script_background(monkeypatch)
    empty_prefixes(monkeypatch)
    old_score = BackgroundCompetitionContinuousLaneRollout.score_background_roots

    def ordinary(self, *args):
        scored = old_score(self, *args)
        if self is candidate:
            # Old S2<=MW makes pair1 the verified BG incumbent. The old-mask
            # case must stay excluded even when it passes every new class.
            squared = {0: Fraction(501 if mode == "old_illegal" else 400),
                       1: Fraction(450), 2: Fraction(500), 3: Fraction(480)}
            return replace(scored, costs={0: Fraction(40), 1: Fraction(35),
                                          2: Fraction(40), 3: Fraction(39)},
                           squared_costs=squared)
        return scored

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout,
                        "score_background_roots", ordinary)
    new_score = candidate.score_background_roots

    def once(*args):
        scored = new_score(*args)
        classes = {0: (Fraction(101), Fraction(100)),
                   1: (Fraction(100), Fraction(200)),
                   2: (Fraction(250), Fraction(250)),
                   3: (Fraction(240), Fraction(240))}
        if mode in ("class_safe", "tie", "old_illegal"):
            classes[0] = (Fraction(99), Fraction(199))
        if mode == "bg_relative":
            # Candidate's class A risk exceeds MW's under this same model,
            # while improving both classes of the BG reference actually used.
            classes.update({0: (Fraction(110), Fraction(190)),
                            1: (Fraction(120), Fraction(200)),
                            2: (Fraction(100), Fraction(230))})
        return replace(scored,
            costs={0: Fraction(25 if mode == "tie" else 20),
                   1: Fraction(25), 2: Fraction(30), 3: Fraction(30)},
            squared_costs={root: sum(values, Fraction(0)) for root, values in classes.items()},
            class_squared_costs=classes)

    monkeypatch.setattr(candidate, "score_background_roots", once)
    result = candidate(OPTIONS, (3, 1), (1000., 1005.), 1000., 0,
                       ((10., 0., 0.), (0.,)), (20, 5))
    assert result.incumbent_maxweight_root_index == 2 and result.incumbent_root_index == 1
    assert result.selection.option_indices == (actual,) and result.improved_incumbent == improved
    assert result.aggregate_risk_feasible_candidates == aggregate_feasible
    assert result.class_risk_filtered_candidates == class_filtered
    assert result.risk_feasible_candidates == aggregate_feasible - class_filtered
    assert all(chosen <= baseline for chosen, baseline in zip(
        result.chosen_predicted_class_squared_flow_s2,
        result.baseline_predicted_class_squared_flow_s2, strict=True))
    assert result.incumbent_actual_predicted_squared_flow_s2 <= result.incumbent_mw_predicted_squared_flow_s2
    if mode == "class_transfer":
        assert result.risk_changed_unconstrained_root
        assert result.class_risk_comparisons == 3  # One rejected class; both incumbent classes.
    else:
        assert result.class_risk_comparisons == 2 * aggregate_feasible
    assert result.class_squared_score_reads == 2 * result.class_risk_comparisons + 4
    assert result.class_squared_initializations == 4 * 2
    assert result.class_squared_accumulations == result.flow_evaluations > 0


def test_loaded_fifo_class_costs_reuse_exact_once_bg_population_and_rng(monkeypatch):
    options = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
               FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
               FrameOption("b1", (0, 1), 2., ((), (2.,))),
               FrameOption("loaded", (1, 1), 2., ((.5,), (1.5,)), loaded=True))
    script_background(monkeypatch)
    # This virtual prefix changes the inner BG decision, but is not part of
    # the outer target population and therefore adds no class flow cost.
    monkeypatch.setattr(background_module, "sample_planning_arrivals",
                        lambda *args: population((.5, 1)))
    candidate = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        options, "loaded", sample_count=1)
    reference = OneStepBackgroundRedecisionLaneRollout(options, "loaded", sample_count=1)
    for planner in (candidate, reference):
        planner.arrival_rates = (.003, .001)
        planner._outer_observed_counts = (3, 1)
        planner.decision_counter = 7
    before = candidate.rng.getstate()
    args = ((3, 1), (1000., 1006.), 1000., 0, ((7., 4., 0.), (2.,)), (population(),))
    actual = candidate.score_background_roots(*args)
    expected = reference.score_background_roots(*args)
    assert all(getattr(actual, item.name) == getattr(expected, item.name)
               for item in fields(BackgroundRootScores))
    assert actual.class_squared_costs[3] == (Fraction(625, 4), Fraction(49, 4))
    assert all(sum(values, Fraction(0)) == actual.squared_costs[root]
               for root, values in actual.class_squared_costs.items())
    assert actual.flow_evaluations == 4 * 4 and actual.background_admissions == 0
    assert actual.class_squared_initializations == actual.candidate_count * 2
    assert actual.class_squared_accumulations == actual.flow_evaluations
    assert candidate._recourse_work == reference._recourse_work
    assert candidate.rng.getstate() == reference.rng.getstate() == before
    assert candidate.decision_counter == reference.decision_counter == 7


def test_outer_prefix_is_scored_and_admitted_later_background_is_excluded(monkeypatch):
    options = (FrameOption("a", (1, 0), 2., ((2.,), ())),
               FrameOption("b", (0, 1), 1., ((), (1.,))))
    script_background(monkeypatch, (1000., .5, 1000.))
    empty_prefixes(monkeypatch)
    candidate = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        options, "population", sample_count=1)
    reference = OneStepBackgroundRedecisionLaneRollout(options, "population", sample_count=1)
    for planner in (candidate, reference):
        planner.arrival_rates = (.002, 1.)
        planner._outer_observed_counts = (2, 1000)
    args = ((2, 1), (1000., 1004., 1008.), 1000., 0,
            ((5., 1.), (2.,)), (population((2., 0)),))
    actual = candidate.score_background_roots(*args)
    expected = reference.score_background_roots(*args)
    assert all(getattr(actual, item.name) == getattr(expected, item.name)
               for item in fields(BackgroundRootScores))
    # The outer A birth at2 joins the scored FIFO. The later B birth at2.5
    # wins the time3 MW tie and reserves a lane, but adds no class cost.
    assert actual.costs == {0: Fraction(23), 1: Fraction(22)}
    assert actual.class_squared_costs == {
        0: (Fraction(114), Fraction(25)), 1: (Fraction(129), Fraction(9))}
    assert all(sum(values, Fraction(0)) == actual.squared_costs[root]
               for root, values in actual.class_squared_costs.items())
    assert actual.flow_evaluations == actual.candidate_count * (3 + 1) == 8
    assert actual.background_admissions == 2 and actual.admitted_job_reads == 10
    assert actual.class_squared_accumulations == actual.flow_evaluations


@pytest.mark.parametrize("prefix_overflow", [False, True])
def test_original_overflow_returns_mw_without_class_scoring(monkeypatch, prefix_overflow):
    options = (FrameOption("single", (1,), 1., ((1.,),)),)
    candidate = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        options, "old-overflow", sample_count=1, event_cap=2)
    script_background(monkeypatch, (.1, .1, .1, 1000.))
    empty_prefixes(monkeypatch)
    if prefix_overflow:
        monkeypatch.setattr(module, "sample_planning_arrivals",
                            lambda *args: population((.1, 0), (.2, 0), (.3, 0), overflow=True))

    def forbidden(*args):
        raise AssertionError("incomplete first stage must not enter the class scorer")

    monkeypatch.setattr(candidate, "score_background_roots", forbidden)
    args = (options, (3,), (1000., 1005., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
    result = candidate(*args)
    assert result.selection == candidate.maxweight(*args).selection
    assert result.planning_overflow and not result.changed
    assert result.incumbent_background_rollout_overflow == (not prefix_overflow)
    assert result.class_squared_initializations == result.class_squared_accumulations == 0
    assert result.class_risk_comparisons == result.class_squared_score_reads == 0
    assert not result.baseline_predicted_class_squared_flow_s2
    if not prefix_overflow:
        assert result.incumbent_flow_evaluations == 2
        assert result.incumbent_background_planning_events == 3


def test_once_overflow_retains_verified_nonmw_bg_and_partial_class_costs(monkeypatch):
    options = (FrameOption("single", (1,), 1., ((1.,),)),
               FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
               FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
               FrameOption("slow-single", (1,), 2., ((2.,),)))
    candidate = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        options, "new-overflow", sample_count=1, event_cap=2)
    script_background(monkeypatch, (.1, .1, .1, 1000.))
    empty_prefixes(monkeypatch)
    call = BackgroundCompetitionContinuousLaneRollout.__call__

    def controlled_child(self, *args):
        result = call(self, *args)
        if self is candidate.inner_planner:
            selected = replace(result.selection, option_counts=(0, 0, 0, 1),
                               option_indices=(3,), service=(1,), duration=2., exact=False)
            return replace(result, selection=selected, changed=True)
        return result

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "__call__", controlled_child)
    result = candidate(options, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))
    assert result.incumbent_root_index == 1 and result.incumbent_maxweight_root_index == 2
    assert not result.incumbent_background_rollout_overflow
    assert result.planning_overflow and result.background_rollout_overflow
    assert result.selection.option_indices == (1,) and result.changed and not result.improved_incumbent
    assert result.class_squared_initializations == 4
    assert result.class_squared_accumulations == result.flow_evaluations == 2
    assert result.class_risk_comparisons == result.class_squared_score_reads == 0
    assert not result.baseline_predicted_class_squared_flow_s2
    assert not result.chosen_predicted_class_squared_flow_s2
    assert result.recourse_calls == 1 and result.recourse_rollout_calls > 0
