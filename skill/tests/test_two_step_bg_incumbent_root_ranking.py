"""Same-state ranking preserves native costs, causal state, and single-root effects."""
from collections import Counter
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import repeat
import json

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.bg_incumbent_background_redecision_lane_rollout as incumbent_module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.bg_incumbent_background_redecision_lane_rollout import (
    BGIncumbentBackgroundRedecisionLaneRollout, INCUMBENT_SCORE_WORK,
)
from algorithm.one_step_background_redecision_lane_rollout import RECOURSE_WORK
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import (
    TWO_STEP_WORK, TwoStepBGIncumbentBackgroundRedecisionLaneRollout,
)
from algorithm.experiments.aligned_hardware_class_protected_original_root_screen import replay_prefixes
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.bg_incumbent_counterfactual_replay import (
    BGIncumbentCounterfactualSelector, STATUS, restore_planner_state,
)
from algorithm.experiments.bg_incumbent_selected_root_replay import BGIncumbentSelectedRootSelector
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.two_step_bg_incumbent_root_ranking import compare_root_scores


OPTIONS = (FrameOption("single", (1,), 1., ((1.,),)),
           FrameOption("loaded-pair", (2,), 1.5, ((.5, 1.),), loaded=True),
           FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
           FrameOption("loaded-triple", (3,), 3., ((.5, 1., 2.),), loaded=True))
NATIVE_BG_SCORE = background_module.BackgroundCompetitionContinuousLaneRollout.score_background_roots
OLD_COSTS = {0: Fraction(40), 1: Fraction(35), 2: Fraction(40), 3: Fraction(39)}
OLD_SQUARED = {0: Fraction(400), 1: Fraction(450), 2: Fraction(500), 3: Fraction(480)}
ONCE_COSTS = {0: Fraction(20), 1: Fraction(25), 2: Fraction(30), 3: Fraction(24)}
ONCE_SQUARED = {0: Fraction(250), 1: Fraction(300), 2: Fraction(320), 3: Fraction(290)}


def normalized(value):
    return json.loads(json.dumps(value))


def quiet_population(monkeypatch):
    class QuietRandom:
        def __init__(self, seed):
            self.gaps = repeat(1000.)

        def expovariate(self, rate):
            return next(self.gaps)

    def empty(rates, window, rng, cap):
        draws = sum(rate > 0 for rate in rates)
        for _ in range(draws):
            rng.random()
        return PlanningArrivals((), False, draws, len(rates))

    monkeypatch.setattr(background_module, "Random", QuietRandom)
    monkeypatch.setattr(background_module, "sample_planning_arrivals", empty)
    monkeypatch.setattr(incumbent_module, "sample_planning_arrivals", empty)


def ranking_case(monkeypatch, mode):
    """Contrast objectives while retaining every native simulation/work fee."""
    quiet_population(monkeypatch)
    once = BGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "ranking", sample_count=1)
    twice = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "ranking", sample_count=1)
    captures = {}

    def ordinary(self, *args):
        native = NATIVE_BG_SCORE(self, *args)
        if self is once:
            square = {**OLD_SQUARED, **({3: Fraction(501)} if mode == "old_mask" else {})}
            result = replace(native, costs=OLD_COSTS, squared_costs=square)
            captures["old_stage"] = result
            return result
        return native

    monkeypatch.setattr(background_module.BackgroundCompetitionContinuousLaneRollout,
                        "score_background_roots", ordinary)
    native_once, native_twice = once.score_background_roots, twice.score_background_roots

    def one(*args):
        result = replace(native_once(*args), costs=ONCE_COSTS, squared_costs=ONCE_SQUARED)
        captures["once"] = result
        captures["once_work"] = Counter(once._recourse_work)
        return result

    def two(*args):
        native = native_twice(*args)
        costs = {0: Fraction(25 if mode == "BG_tie" else 15), 1: Fraction(25),
                 2: Fraction(30), 3: Fraction(26 if mode == "BG_tie" else 24 if mode == "original" else 10)}
        squared = {0: Fraction(260), 1: Fraction(300), 2: Fraction(310),
                   3: Fraction(301 if mode == "new_risk" else 280)}
        result = replace(native, costs=costs, squared_costs=squared)
        captures["twice"] = result
        captures["twice_work"] = Counter(twice._recourse_work)
        captures["two_step_work"] = Counter(twice._two_step_work)
        return result

    monkeypatch.setattr(once, "score_background_roots", one)
    monkeypatch.setattr(twice, "score_background_roots", two)
    once.decision_counter = 7
    context = (OPTIONS, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))
    decision = once(*context)
    assert decision.selection.option_indices == (0,) and decision.incumbent_root_index == 1
    fields = asdict(decision)
    del fields["selection"]
    frame = {"queue": context[1], "available_before": context[2], "start": context[3],
             "lane_ids": (context[4],), "waiting_ages": context[5],
             "observed_arrival_counts": context[6], "option_indices": decision.selection.option_indices,
             "rollout": fields}
    once.decision_counter = 7  # Preserve the native post-prefix/pre-increment state.
    captures.clear()
    prior = {"frame_index": 7, "decision_counter": 7, "factual_option_index": 0,
             "replacement_option_index": 1, "outcomes": {"saved_effect": "untouched"}}
    return once, twice, frame, (PlanningArrivals((), False, 1, 1),), prior, captures


def test_native_original_prefix_state_survives_idle_ineligible_and_overflow_frames():
    options = OPTIONS[:1]
    native = BGIncumbentBackgroundRedecisionLaneRollout(options, "ranking-prefix", sample_count=8, event_cap=1)
    once = BGIncumbentBackgroundRedecisionLaneRollout(options, "ranking-prefix", sample_count=8, event_cap=1)
    twice = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(options, "ranking-prefix", sample_count=8, event_cap=1)
    frames, states = [], []
    for now, queue, counts in ((0., (0,), (0,)), (1., (1,), (1000,)),
                               (2., (3,), (1000,)), (1e6, (1,), (1000,))):
        ages, available = (tuple(0. for _ in range(queue[0])),), (now, now + 2.)
        decision = native(options, queue, available, now, 0, ages, counts)
        info = asdict(decision)
        del info["selection"]
        frames.append({"start": now, "queue": queue, "available_before": available,
            "waiting_ages": ages, "observed_arrival_counts": counts, "lane_ids": (0,), "rollout": info})
        states.append((native.rng.getstate(), native.decision_counter))
    work, populations = Counter(), []
    for index, frame, paths in replay_prefixes(once, frames, work):
        assert once.rng.getstate() == states[index][0]
        assert once.decision_counter == states[index][1] - 1
        populations.append(len(paths))
        if index == 3:
            twice.rng.setstate(once.rng.getstate())
            twice.decision_counter = once.decision_counter
            twice._outer_observed_counts = once._outer_observed_counts
            twice.arrival_rates = once.arrival_rates
            scored = twice.score_background_roots(frame["queue"], frame["available_before"],
                frame["start"], 0, frame["waiting_ages"], paths)
            assert float(scored.costs[0] / len(paths)) == frame["rollout"]["chosen_predicted_completion_s"]
            assert twice.rng.getstate() == once.rng.getstate() and twice.decision_counter == once.decision_counter
    assert populations == [0, 1, 0, 8]
    assert work["prefix_overflow_frames"] == 1 and work["prefix_sample_calls"] == 9
    assert once.decision_counter == native.decision_counter == 4
    assert once.rng.getstate() == native.rng.getstate()
    assert not work["root_option_probes"]


def test_native_scores_use_both_original_masks_strict_bg_ties_and_charge_all_stages_once(monkeypatch):
    expected = {"third": ("third", 3), "old_mask": ("original", 0),
                "new_risk": ("original", 0), "BG_tie": ("BG", 1), "original": ("original", 0)}
    for mode, choice in expected.items():
        once, twice, frame, paths, prior, captures = ranking_case(monkeypatch, mode)
        rng_before, counter_before, work = once.rng.getstate(), once.decision_counter, Counter()
        result = compare_root_scores(once, twice, frame, paths, prior, work)
        assert (result["choice_kind"], result["two_step_option_index"]) == choice
        assert result["predictions_match_passed"] and result["once_option_index"] == 0
        assert result["BG_root_index"] == 1 and result["MW_root_index"] == 2
        assert result["outcomes"] is prior["outcomes"]
        by_index = {row["option_index"]: row for row in result["two_step_scores"]}
        assert by_index[3]["old_stage_permitted"] == (mode != "old_mask")
        assert by_index[3]["two_step_S2_permitted"] == (mode != "new_risk")
        assert by_index[3]["permitted"] == (mode not in ("old_mask", "new_risk"))
        for stage in ("old_stage", "once", "twice"):
            for name in INCUMBENT_SCORE_WORK:
                assert work[stage + "_" + name] == getattr(captures[stage], name)
        for stage in ("once", "twice"):
            for name in RECOURSE_WORK:
                assert work[stage + "_" + name] == captures[stage + "_work"][name]
        for name in TWO_STEP_WORK:
            assert work["twice_" + name] == captures["two_step_work"][name]
        assert work["comparison_maxweight_root_option_probes"] == len(OPTIONS)
        assert work["comparison_rng_words_copied"] == work["comparison_rng_words_restored"] == 625
        assert work["comparison_counter_reads"] == work["comparison_counter_writes"] == 1
        assert once.rng.getstate() == twice.rng.getstate() == rng_before
        assert once.decision_counter == twice.decision_counter == counter_before
        assert work["old_stage_risk_comparisons"] == len(OPTIONS)
        assert work["once_membership_checks"] == work["twice_membership_checks"] == len(OPTIONS)


def test_cache_prediction_mismatch_stops_before_twice_ranking(monkeypatch):
    once, twice, frame, paths, prior, _ = ranking_case(monkeypatch, "third")
    frame["rollout"]["chosen_predicted_completion_s"] += .125
    work = Counter()
    with pytest.raises(ValueError, match="predictions differ from the cache"):
        compare_root_scores(once, twice, frame, paths, prior, work)
    assert work["old_stage_root_option_probes"] > 0 and work["once_root_option_probes"] > 0
    assert not work["twice_root_option_probes"]


def test_driver_reuses_bg_effect_verbatim_and_original_choice_has_zero_single_action_effect():
    from algorithm.experiments.aligned_hardware_two_step_bg_incumbent_root_ranking import selected_outcomes
    factual = {"jobs": [{"coordinate": 0, "index": 0, "arrival": 300., "start": 301., "completion": 302.}]}
    factual["snapshots"] = {"observation_end": trajectory_snapshot(factual["jobs"], 1200., 1)}
    protocol = {"birth_cohort": [300., 900.], "arrival_horizon": 900., "observation_end": 1200.}
    saved = {"saved_BG_effect": {"delta_T_mean": -3.}}
    assert selected_outcomes({"choice_kind": "BG", "outcomes": saved}, factual, protocol, ("a",)) is saved
    original = selected_outcomes({"choice_kind": "original", "outcomes": saved}, factual, protocol, ("a",))
    for name in ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished",
                 "delta_1200_unfinished", "delta_middle_queue_area_s"):
        assert original[name] == 0
    assert saved == {"saved_BG_effect": {"delta_T_mean": -3.}}


def test_third_loaded_root_is_forced_once_then_native_once_bg_replans_and_null_matches(monkeypatch):
    quiet_population(monkeypatch)
    controlled = set()

    def ordinary(self, *args):
        native = NATIVE_BG_SCORE(self, *args)
        if self in controlled and tuple(args[0]) == (3,) and args[2] == 3.:
            return replace(native, costs=OLD_COSTS, squared_costs=OLD_SQUARED)
        return native

    monkeypatch.setattr(background_module.BackgroundCompetitionContinuousLaneRollout,
                        "score_background_roots", ordinary)

    def planner():
        candidate = BGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "selected-root", sample_count=1)
        controlled.add(candidate)
        native_score = candidate.score_background_roots

        def contrast(*args):
            native = native_score(*args)
            if tuple(args[0]) == (3,) and args[2] == 3.:
                return replace(native, costs=ONCE_COSTS, squared_costs=ONCE_SQUARED)
            return native

        candidate.score_background_roots = contrast
        return candidate

    model = FrameModel("toy", "ranking-replay", 2, ("a",), (1.,), OPTIONS)
    tape = [(0., 0, 0), (0., 0, 1), (3., 0, 2), (3., 0, 3), (3., 0, 4), (6., 0, 5)]
    captured = BGIncumbentCounterfactualSelector(planner(), capture=range(40))
    factual = simulate_observed_rate_contextual_trajectory(model, tape, 8., 10., captured, policy="factual")
    index, = [index for index, frame in enumerate(factual["frames"]) if frame["rollout"]["improved_incumbent"]]
    checkpoint, state = prepare_checkpoint(factual, index), captured.states[index]
    null = BGIncumbentCounterfactualSelector(planner())
    restore_planner_state(null.planner, state)
    resumed = resume_single_action_trajectory(model, 8., 10., null, checkpoint, policy="null")
    assert resumed["jobs"] == factual["jobs"]
    assert normalized(factual["frames"][:index] + resumed["frames"]) == normalized(factual["frames"])
    assert null.planner.rng.getstate() == captured.planner.rng.getstate()
    assert null.planner.decision_counter == captured.planner.decision_counter
    chosen_predictions = {"chosen_predicted_completion_s": 24.,
        "chosen_predicted_squared_flow_s2": 290., "incumbent_actual_predicted_squared_flow_s2": 480.}
    forced = BGIncumbentSelectedRootSelector(planner(), 3, chosen_predictions, capture=(1,))
    restore_planner_state(forced.planner, state)
    branch = resume_single_action_trajectory(model, 8., 10., forced, checkpoint, policy="third")
    assert branch["frames"][0]["option_indices"] == (3,) and branch["frames"][0]["duration"] == 3.
    assert normalized(forced.first_proposal["rollout"]) == normalized(factual["frames"][index]["rollout"])
    assert forced.first_proposal["original"] == 0 and forced.first_proposal["replacement"] == 1
    assert forced.intervention["original"] == 0 and forced.intervention["replacement"] == 3
    assert forced.intervention["BG_incumbent_option_index"] == 1
    assert forced.intervention["forced_single_action"]
    for name, value in chosen_predictions.items():
        assert branch["frames"][0]["rollout"][name] == value
    completions = {job["index"]: job["completion"] for job in branch["jobs"]}
    assert [completions[index] for index in range(5)] == [.5, 1., 3.5, 4., 5.]
    lane = branch["frames"][0]["lane_ids"][0]
    assert branch["frames"][1]["start"] == 3.5
    assert branch["frames"][1]["available_before"][lane] == 6.
    later, = [frame for frame in branch["frames"][1:] if frame["start"] == 6.]
    assert later["option_indices"] == (0,) and later["rollout"]["eligible"]
    assert forced.work["override_calls"] == 1 and forced.work["override_root_option_probes"] == len(OPTIONS)
    for name in ("root_option_probes", "incumbent_rollout_option_probes", "recourse_rollout_option_probes"):
        assert forced.work[name] == sum(frame["rollout"][name] for frame in branch["frames"])
    for name in STATUS:
        assert forced.status["applied_" + name] == sum(frame["rollout"][name] for frame in branch["frames"])
    reference = planner()
    restore_planner_state(reference, state)
    frame = factual["frames"][index]
    reference(OPTIONS, frame["queue"], frame["available_before"], frame["start"],
              frame["lane_ids"][0], frame["waiting_ages"], frame["observed_arrival_counts"])
    assert forced.states[1] == {"rng_state": reference.rng.getstate(), "decision_counter": reference.decision_counter}
    assert forced.planner.decision_counter == state["decision_counter"] + len(branch["frames"])
