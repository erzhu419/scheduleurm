"""A once-BG improvement must preserve the verified ordinary-BG incumbent."""
from dataclasses import replace
from fractions import Fraction
from itertools import chain, repeat
from random import Random

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.bg_incumbent_background_redecision_lane_rollout as module
import algorithm.one_step_background_redecision_lane_rollout as recourse_module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, _LazyBackgroundArrivals,
)
from algorithm.bg_incumbent_background_redecision_lane_rollout import BGIncumbentBackgroundRedecisionLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_bg_incumbent_background_redecision_development import (
    check_rollout, incumbent_work,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory


OPTIONS = (FrameOption("single", (1,), 1., ((1.,),)),
           FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
           FrameOption("triple", (3,), 2., ((2., 2., 2.),)))


def population():
    return PlanningArrivals((), False, 0, 0)


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

    monkeypatch.setattr(background_module, "_LazyBackgroundArrivals", ScriptedSuffix)
    monkeypatch.setattr(recourse_module, "_LazyBackgroundArrivals", ScriptedSuffix)


def empty_prefixes(monkeypatch, candidate, consume=False):
    samples = []

    def sample(rates, window, rng, cap):
        samples.append(rng is candidate.rng)
        if consume:
            rng.random()
        return PlanningArrivals((), False, int(consume), len(rates))

    monkeypatch.setattr(module, "sample_planning_arrivals", sample)
    monkeypatch.setattr(background_module, "sample_planning_arrivals", sample)
    return samples


def current(options=OPTIONS):
    return (options, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))


@pytest.mark.parametrize("mode,actual,improved,old_feasible,new_feasible", [
    ("tie", 1, False, 3, 2),
    ("worse", 1, False, 3, 2),
    ("gain", 0, True, 3, 2),
    ("old_illegal", 1, False, 2, 1),
    ("return_mw", 2, True, 3, 3),
])
def test_real_nonmw_bg_incumbent_and_both_s2_constraints(monkeypatch, mode, actual,
                                                        improved, old_feasible, new_feasible):
    candidate = BGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "constraints", sample_count=2)
    script_background(monkeypatch)
    empty_prefixes(monkeypatch, candidate)
    old_score = BackgroundCompetitionContinuousLaneRollout.score_background_roots
    old_results = []

    def original(self, *args):
        scored = old_score(self, *args)
        if self is candidate:
            # Native ordinary BG selects pair1 instead of MW triple2. Only
            # the old-illegal case changes its otherwise permissive S2 mask.
            assert scored.costs == {0: Fraction(32), 1: Fraction(31), 2: Fraction(32)}
            assert scored.squared_costs == {0: Fraction(267), 1: Fraction(563, 2), 2: Fraction(304)}
            if mode == "old_illegal":
                scored = replace(scored, squared_costs={**scored.squared_costs, 0: Fraction(305)})
            old_results.append(scored)
        return scored

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", original)
    new_score = candidate.score_background_roots

    def once(*args):
        scored = new_score(*args)
        costs = {0: Fraction(31 if mode == "tie" else 32 if mode in ("worse", "return_mw") else 30),
                 1: Fraction(31), 2: Fraction(30 if mode == "return_mw" else 32)}
        squared = {0: Fraction(260), 1: Fraction(563, 2),
                   2: Fraction(280 if mode == "return_mw" else 304)}
        return replace(scored, costs=costs, squared_costs=squared)

    monkeypatch.setattr(candidate, "score_background_roots", once)
    result = candidate(*current())
    assert result.incumbent_maxweight_root_index == 2 and result.incumbent_root_index == 1
    assert result.incumbent_changed and result.selection.option_indices == (actual,)
    assert result.improved_incumbent == improved
    assert result.changed == (actual != 2)  # Returning to MW can improve BG.
    assert result.incumbent_risk_feasible_candidates == old_feasible
    assert result.risk_feasible_candidates == new_feasible
    assert result.incumbent_actual_predicted_squared_flow_s2 <= result.incumbent_mw_predicted_squared_flow_s2
    assert result.chosen_predicted_squared_flow_s2 <= result.baseline_predicted_squared_flow_s2
    assert (result.chosen_predicted_completion_s < result.baseline_predicted_completion_s) == improved
    assert result.incumbent_flow_evaluations == old_results[0].flow_evaluations == 18
    assert result.flow_evaluations == 18 and result.recourse_calls == 4
    assert result.incumbent_risk_comparisons == result.incumbent_membership_checks == 3
    assert result.incumbent_lookahead_risk_comparisons == old_feasible
    assert result.incumbent_ordering_comparisons == 2 + old_feasible - 1
    assert result.incumbent_lookahead_ordering_comparisons == 2 + new_feasible - 1
    assert result.incumbent_strict_improvement_comparisons == result.incumbent_lookahead_strict_improvement_comparisons == 1


def test_two_stages_share_one_prefix_and_inner_never_advances_online_state(monkeypatch):
    candidate = BGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "shared", sample_count=2)
    candidate.decision_counter = 7
    script_background(monkeypatch)
    sampled = empty_prefixes(monkeypatch, candidate, consume=True)
    paths = []
    old_score = BackgroundCompetitionContinuousLaneRollout.score_background_roots

    def original(self, *args):
        if self is candidate:
            paths.append(args[-1])
        return old_score(self, *args)

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", original)
    new_score = candidate.score_background_roots

    def once(*args):
        paths.append(args[-1])
        return new_score(*args)

    monkeypatch.setattr(candidate, "score_background_roots", once)
    call = BackgroundCompetitionContinuousLaneRollout.__call__
    states = []

    def inner(self, *args):
        if self is candidate.inner_planner:
            states.append((self.rng.getstate(), self.decision_counter, args[-1]))
        return call(self, *args)

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "__call__", inner)
    result = candidate(*current())
    expected = Random("shared")
    expected.random()
    expected.random()
    assert len(paths) == 2 and paths[0] is paths[1] and len(paths[0]) == 2
    assert sum(sampled) == 2 and len(sampled) - sum(sampled) == 8
    assert len(states) == 4 and all(state == (expected.getstate(), 8, (20,)) for state in states)
    assert candidate.rng.getstate() == expected.getstate() and candidate.decision_counter == 8
    assert result.planning_random_draws == 2
    assert result.recourse_rng_words_copied == 625 and result.recourse_rng_words_restored == 4 * 625
    assert result.recourse_counter_reads == 1 and result.recourse_counter_writes == 4
    assert result.incumbent_root_option_probes == 3 and result.root_option_probes == 6


def test_original_background_overflow_skips_once_stage_and_retains_partial_fees(monkeypatch):
    options = OPTIONS[:1]
    candidate = BGIncumbentBackgroundRedecisionLaneRollout(options, "old-overflow", sample_count=1, event_cap=2)
    script_background(monkeypatch, (.1, .1, .1, 1000.))
    empty_prefixes(monkeypatch, candidate, consume=True)

    def forbidden(*args):
        raise AssertionError("an incomplete original BG stage cannot enter the once stage")

    monkeypatch.setattr(candidate, "score_background_roots", forbidden)
    args = (options, (3,), (1000., 1005., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
    result = candidate(*args)
    assert result.selection == candidate.maxweight(*args).selection
    assert result.planning_overflow and result.incumbent_background_rollout_overflow
    assert not result.changed and result.incumbent_root_index is None
    assert result.incumbent_predicted_completion_s is None
    assert result.incumbent_rollout_calls == result.incumbent_rollout_option_probes == 1
    assert result.incumbent_flow_evaluations == result.incumbent_admitted_job_reads == 2
    assert result.incumbent_background_planning_events == 3
    assert result.incumbent_background_planning_random_draws == 3
    assert result.candidate_count == result.rollout_calls == result.recourse_calls == 0
    assert result.root_option_probes == 1 and result.incumbent_root_option_probes == 1
    assert result.recourse_rng_words_copied == result.incumbent_membership_checks == 0
    assert candidate.decision_counter == 1


def test_once_background_overflow_retains_nonmw_bg_and_all_partial_work(monkeypatch):
    options = (*OPTIONS, FrameOption("slow-single", (1,), 2., ((2.,),)))
    candidate = BGIncumbentBackgroundRedecisionLaneRollout(options, "new-overflow", sample_count=1, event_cap=2)
    script_background(monkeypatch, (.1, .1, .1, 1000.))
    empty_prefixes(monkeypatch, candidate, consume=True)
    call = BackgroundCompetitionContinuousLaneRollout.__call__
    proposals = []

    def controlled_child(self, *args):
        result = call(self, *args)
        if self is candidate.inner_planner:
            proposals.append(result)
            # Retain the real ordinary-BG computation, then control this one
            # executed child option. Its full period moves the outer call to3,
            # where real chronological births exceed the combined cap2.
            selected = replace(result.selection, option_counts=(0, 0, 0, 1),
                               option_indices=(3,), service=(1,), duration=2., exact=False)
            return replace(result, selection=selected, changed=True)
        return result

    monkeypatch.setattr(BackgroundCompetitionContinuousLaneRollout, "__call__", controlled_child)
    result = candidate(*current(options))
    assert result.incumbent_root_index == 1 and result.incumbent_maxweight_root_index == 2
    assert not result.incumbent_background_rollout_overflow
    assert result.planning_overflow and result.background_rollout_overflow
    assert result.selection.option_indices == (1,) and result.changed
    assert not result.improved_incumbent
    assert result.baseline_predicted_completion_s is result.chosen_predicted_completion_s is None
    assert result.incumbent_predicted_completion_s == 15.5
    assert result.incumbent_predicted_squared_flow_s2 == result.incumbent_actual_predicted_squared_flow_s2 == 140.75
    assert result.incumbent_flow_evaluations == 12 and result.incumbent_root_option_probes == 4
    assert result.flow_evaluations == result.admitted_job_reads == 2
    assert result.rollout_calls == 1 and result.rollout_option_probes == 4
    assert result.background_planning_events == 3 and result.root_option_probes == 8
    assert len(proposals) == result.recourse_calls == 1
    assert result.recourse_rollout_calls == proposals[0].rollout_calls > 0
    assert result.recourse_planning_random_draws == proposals[0].planning_random_draws > 0
    assert result.incumbent_membership_checks == result.incumbent_lookahead_risk_comparisons == 0
    assert candidate.decision_counter == 1


def test_native_toy_driver_counts_both_stage_ledgers_once():
    model = FrameModel("toy", "bg-incumbent", 1, ("a",), (1.,), OPTIONS)
    candidate = BGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "native")
    result = simulate_observed_rate_contextual_trajectory(
        model, [(0., 0, 0), (0., 0, 1), (0., 0, 2), (3., 0, 3)],
        8., 10., candidate, policy="bg-incumbent")
    protocol = {"planning_samples": 8, "planning_event_cap": 128}
    assert check_rollout(result, model, protocol, candidate)
    first = result["frames"][0]["rollout"]
    assert first["incumbent_maxweight_root_index"] == 2 and first["incumbent_root_index"] == 1
    assert result["frames"][0]["option_indices"] == (1,) and first["changed"]
    assert not first["improved_incumbent"]
    work = incumbent_work(result, candidate, 8.)
    for key in ("incumbent_root_option_probes", "incumbent_rollout_option_probes",
                "incumbent_membership_checks", "incumbent_lookahead_risk_comparisons"):
        assert work[key] == sum(frame["rollout"][key] for frame in result["frames"])
    assert work["total_option_probes"] == (
        work["root_option_probes"] + work["rollout_option_probes"]
        + work["incumbent_root_option_probes"] + work["incumbent_rollout_option_probes"]
        + work["recourse_root_option_probes"] + work["recourse_rollout_option_probes"]
        + sum(work.get("preprocessing_" + prefix + key, 0)
              for prefix in ("", "recourse_")
              for key in ("option_probes", "age_profile_option_probes", "rate_dimension_option_probes")))
    assert work["total_rollout_calls"] == (
        work["rollout_calls"] + work["incumbent_rollout_calls"] + work["recourse_rollout_calls"])
    assert work["total_planning_events"] == (
        work["planning_events"] + work["incumbent_background_planning_events"] + work["recourse_planning_events"])
