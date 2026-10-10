"""Cross first roots with free suffixes without changing the native first proposal."""
from collections import Counter
from fractions import Fraction
from itertools import repeat
import json

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.two_step_bg_incumbent_original_risk_lane_rollout as policy_module
import algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_two_step_bg_incumbent_continuation_cross import cross_outcomes
from algorithm.experiments.bg_incumbent_counterfactual_replay import BGIncumbentCounterfactualSelector
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import (
    BG_FEES, TwoStepContinuationCrossSelector, collect_work, reconstruct_prefix_states,
    registered_anchors, restore_planner_state,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import TwoStepBGIncumbentOriginalRiskLaneRollout


OPTIONS = (FrameOption("single", (1,), 1., ((1.,),)),
           FrameOption("loaded-pair", (2,), 1.5, ((.5, 1.),), loaded=True),
           FrameOption("triple", (3,), 2., ((2., 2., 2.),)))
MODEL = FrameModel("toy", "continuation-cross", 2, ("a",), (1.,), OPTIONS)
TAPE = [(0., 0, 0), (0., 0, 1), (3., 0, 2), (3., 0, 3), (3., 0, 4), (6., 0, 5)]


def normalized(value):
    return json.loads(json.dumps(value))


@pytest.fixture(scope="module")
def native():
    with pytest.MonkeyPatch.context() as patch:
        class QuietRandom:
            def __init__(self, seed):
                self.gaps = repeat(1000.)

            def expovariate(self, rate):
                return next(self.gaps)

        def no_future(rates, window, rng, cap):
            draws = sum(rate > 0 for rate in rates)
            for _ in range(draws):
                rng.random()
            return PlanningArrivals((), False, draws, len(rates))

        patch.setattr(background_module, "Random", QuietRandom)
        for target in (policy_module, background_module, module):
            patch.setattr(target, "sample_planning_arrivals", no_future)

        def planner():
            candidate = TwoStepBGIncumbentOriginalRiskLaneRollout(OPTIONS, "cross-toy", sample_count=1)
            native_score = candidate.score_background_roots

            def contrast(*args):
                scored = native_score(*args)
                if tuple(args[0]) == (3,) and args[2] == 3.:
                    return type(scored)(**{**scored.__dict__,
                        "costs": {**scored.costs, 0: scored.costs[1] - Fraction(1, 2)},
                        "squared_costs": {**scored.squared_costs, 0: scored.squared_costs[1] + 1}})
                return scored

            candidate.score_background_roots = contrast
            return candidate

        captured = BGIncumbentCounterfactualSelector(planner(), capture=range(40))
        factual = simulate_observed_rate_contextual_trajectory(MODEL, TAPE, 8., 10., captured, policy="factual")
        index, = [index for index, frame in enumerate(factual["frames"])
                  if frame["rollout"]["improved_incumbent"]]
        frame = factual["frames"][index]
        assert frame["start"] == 3. and frame["option_indices"] == (0,)
        assert frame["rollout"]["incumbent_root_index"] == 1
        yield factual, captured, index, planner


def branch_from(native, first_root="original", suffix="current"):
    factual, captured, index, planner = native
    selector = TwoStepContinuationCrossSelector(planner(), first_root, suffix, capture=(1,))
    restore_planner_state(selector.planner, captured.states[index])
    branch = resume_single_action_trajectory(MODEL, 8., 10., selector,
        prepare_checkpoint(factual, index), policy=first_root + "_" + suffix)
    return branch, selector


def test_anchors_use_first_middle_visible_canonical_cohort_and_keep_missing_without_outcome_filter():
    def frame(start, counts, ages, outcome):
        return {"start": start, "queue": (len(ages),), "waiting_ages": (ages,),
            "observed_arrival_counts": (counts,), "available_before": (start, start + 1.5),
            "lane_ids": (0,), "service": (1,), "option_indices": (0,),
            "later_outcome": outcome, "rollout": {"improved_incumbent": True,
                "planning_overflow": False, "incumbent_root_index": 1,
                "incumbent_maxweight_root_index": 1}}

    jobs = [{"coordinate": 0, "index": index, "arrival": birth, "start": None, "completion": None}
            for index, birth in enumerate((298., 299., 300., 301., 302.))]
    factual = {
        "found": {"jobs": jobs, "frames": [frame(299., 2, (1., 0.), 1),
            frame(300., 3, (1., 0.), -1), frame(301., 4, (1., 0.), 1),
            frame(900., 5, (599., 598.), -1)]},
        "no_cohort": {"jobs": jobs[:2], "frames": [frame(300., 2, (2., 1.), -1)]},
        "late": {"jobs": jobs[2:4], "frames": [frame(900., 2, (600., 599.), -1)]},
    }
    tapes = [{"tape_id": name} for name in factual]
    actual = registered_anchors(tapes, factual)
    assert actual["missing_tape_ids"] == ["no_cohort", "late"]
    anchor, = actual["anchors"]
    assert (anchor["tape_id"], anchor["frame_index"], anchor["start"]) == ("found", 1, 300.)
    assert anchor["visible_cohort_jobs"] == 1
    assert normalized(anchor["visible_cohort_job_ids"]) == [[0, 2]]
    changed = normalized(factual)
    for result in changed.values():
        for row in result["frames"]:
            row["later_outcome"] *= -100
    assert normalized(registered_anchors(tapes, changed)) == normalized(actual)


def test_prefix_reconstruction_uses_real_fifo_births_and_precall_state_through_idle_ineligible_overflow(monkeypatch):
    options = OPTIONS[:1]
    model = FrameModel("toy", "cross-prefix", 2, ("a",), (1.,), options)

    def overflow(rates, window, rng, cap):
        for _ in range(2):
            rng.random()
        return PlanningArrivals(((.1, 0), (.2, 0)), True, 2, len(rates))

    monkeypatch.setattr(policy_module, "sample_planning_arrivals", overflow)
    monkeypatch.setattr(module, "sample_planning_arrivals", overflow)
    candidate = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "cross-prefix", sample_count=1, event_cap=1)
    captured = BGIncumbentCounterfactualSelector(candidate, capture=range(40))
    factual = simulate_observed_rate_contextual_trajectory(model,
        [(1., 0, index) for index in range(3)], 4., 5., captured, policy="prefix")
    assert [frame["rollout"]["eligible"] for frame in factual["frames"]] == [False, False, False, True, True]
    assert sum(frame["rollout"]["planning_overflow"] for frame in factual["frames"]) == 2
    fresh = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "cross-prefix", sample_count=1, event_cap=1)
    work = Counter()
    states = reconstruct_prefix_states(fresh, factual, range(len(factual["frames"])), work)
    assert states == captured.states
    assert fresh.rng.getstate() == candidate.rng.getstate() and fresh.decision_counter == candidate.decision_counter == 5
    assert work["prefix_fifo_job_birth_reads"] == work["prefix_arrival_counter_updates"] == 3
    assert work["prefix_fifo_admission_job_reads"] == 3
    assert work["prefix_overflow_frames"] == 2 and work["prefix_planning_random_draws"] == 4
    assert work["replayed_decision_counter_reads"] == work["replayed_decision_counter_writes"] == 5
    assert work["checkpoint_rng_words_copied"] == 625 * 5
    assert not work["root_option_probes"] and not work["recourse_calls"]


def test_current_null_reproduces_full_trajectory_and_bg_first_is_one_loaded_fifo_swap(native):
    factual, captured, index, _ = native
    resumed, null = branch_from(native)
    assert resumed["jobs"] == factual["jobs"]
    assert normalized(factual["frames"][:index] + resumed["frames"]) == normalized(factual["frames"])
    assert factual["lane_events"][:index] + resumed["lane_events"] == factual["lane_events"]
    assert resumed["last_frame_end"] == factual["last_frame_end"]
    assert null.active_planner.rng.getstate() == captured.planner.rng.getstate()
    assert null.active_planner.decision_counter == captured.planner.decision_counter
    forced, selector = branch_from(native, "BG", "current")
    assert forced["frames"][0]["option_indices"] == (1,)
    assert normalized(selector.first_proposal["rollout"]) == normalized(factual["frames"][index]["rollout"])
    assert selector.first_proposal["original"] == 0 and selector.first_proposal["replacement"] == 1
    assert not forced["frames"][0]["rollout"]["improved_incumbent"]
    assert forced["frames"][0]["rollout"]["chosen_predicted_completion_s"] == forced["frames"][0]["rollout"]["baseline_predicted_completion_s"]
    completions = {job["index"]: job["completion"] for job in forced["jobs"]}
    assert [completions[job] for job in range(4)] == [.5, 1., 3.5, 4.]
    lane = forced["frames"][0]["lane_ids"][0]
    assert forced["frames"][1]["start"] == 3.5
    assert forced["frames"][1]["available_before"][lane] == 4.5
    assert forced["frames"][1]["option_indices"] == (0,) and forced["frames"][1]["rollout"]["eligible"]
    assert selector.BG_planner is None and selector.active_planner is selector.planner
    assert selector.work["override_calls"] == 1 and selector.work["override_root_option_probes"] == len(OPTIONS)
    assert selector.status["current_proposal_calls"] == selector.calls
    assert selector.planner.decision_counter == captured.states[index]["decision_counter"] + selector.calls


def test_ordinary_bg_starts_after_full_current_proposal_with_postproposal_state_and_native_work(native):
    factual, captured, index, planner = native
    branch, selector = branch_from(native, "original", "BG")
    first = factual["frames"][index]
    reference = planner()
    restore_planner_state(reference, captured.states[index])
    reference(OPTIONS, first["queue"], first["available_before"], first["start"],
              first["lane_ids"][0], first["waiting_ages"], first["observed_arrival_counts"])
    assert selector.states[1] == {"rng_state": reference.rng.getstate(), "decision_counter": reference.decision_counter}
    assert branch["frames"][0]["option_indices"] == first["option_indices"]
    assert normalized(selector.first_proposal["rollout"]) == normalized(first["rollout"])
    assert "incumbent_root_index" not in branch["frames"][1]["rollout"]
    assert selector.active_planner is selector.BG_planner
    assert selector.planner.rng.getstate() == reference.rng.getstate()
    assert selector.planner.decision_counter == reference.decision_counter
    ordinary = BackgroundCompetitionContinuousLaneRollout(OPTIONS, "cross-toy", sample_count=1)
    restore_planner_state(ordinary, selector.states[1])
    combined = {**branch, "frames": factual["frames"][:index] + branch["frames"]}
    direct = resume_single_action_trajectory(MODEL, 8., 10., ordinary,
        prepare_checkpoint(combined, index + 1), policy="native_BG")
    assert normalized(direct["frames"]) == normalized(branch["frames"][1:])
    assert direct["jobs"] == branch["jobs"]
    assert ordinary.rng.getstate() == selector.active_planner.rng.getstate()
    assert ordinary.decision_counter == selector.active_planner.decision_counter
    assert selector.status["current_proposal_calls"] == 1
    assert selector.status["BG_suffix_proposal_calls"] == selector.calls - 1
    work = collect_work(selector, branch)
    assert work["controller_constructors"] == 2 and work["recourse_controller_constructors"] == 1
    assert work["preprocessing_option_probes"] == work["preprocessing_recourse_option_probes"] == len(OPTIONS)
    assert work["preprocessing_BG_suffix_option_probes"] == len(OPTIONS)
    assert work["suffix_rng_words_copied"] == work["suffix_rng_words_restored"] == 625
    assert work["suffix_decision_counter_reads"] == work["suffix_decision_counter_writes"] == 1
    for name in BG_FEES + TWO_STEP_WORK + ("incumbent_root_option_probes", "recourse_root_option_probes", "recourse_rollout_option_probes"):
        assert work[name] == sum(frame["rollout"].get(name, 0) for frame in branch["frames"])
    complete = [frame["rollout"] for frame in branch["frames"][1:]
                if frame["rollout"]["eligible"] and not frame["rollout"]["planning_overflow"]]
    assert work["BG_suffix_risk_comparisons"] == sum(info["candidate_count"] for info in complete)
    assert work["BG_suffix_ordering_comparisons"] == sum(info["candidate_count"] + info["risk_feasible_candidates"] - 2 for info in complete)
    assert work["BG_suffix_strict_improvement_comparisons"] == len(complete)
    assert work["age_context_job_reads"] == branch["rollout_work"]["age_context_job_reads"]
    assert not work["override_root_option_probes"]


def test_bg_first_bg_suffix_keeps_current_first_metadata_and_charges_only_one_override(native):
    factual, _, index, _ = native
    branch, selector = branch_from(native, "BG", "BG")
    assert branch["frames"][0]["option_indices"] == (1,)
    assert normalized(selector.first_proposal["rollout"]) == normalized(factual["frames"][index]["rollout"])
    assert selector.intervention["forced_single_action"] and selector.intervention["first_root"] == "BG"
    assert selector.intervention["suffix"] == "BG"
    assert selector.work["override_calls"] == 1 and selector.work["override_root_option_probes"] == len(OPTIONS)
    assert selector.status["current_proposal_calls"] == 1
    assert selector.status["BG_suffix_proposal_calls"] == len(branch["frames"]) - 1
    assert all("incumbent_root_index" not in frame["rollout"] for frame in branch["frames"][1:])
    assert branch["frames"][1]["option_indices"] == (0,)
    assert not hasattr(selector.active_planner, "pending_successor_index")
    work = collect_work(selector, branch)
    assert work["controller_constructors"] == 2 and work["recourse_controller_constructors"] == 1
    assert work["root_option_probes"] == sum(frame["rollout"]["root_option_probes"] for frame in branch["frames"])
    assert work["recourse_calls"] == branch["frames"][0]["rollout"]["recourse_calls"]


def test_cross_effects_use_all_four_completed_ids_full_cohort_and_original_minus_bg_direction():
    completion = {"original_current": (310., 320., None), "BG_current": (320., None, 330.),
                  "original_BG": (330., 340., 350.), "BG_BG": (340., 360., None)}
    arms = {}
    for name, values in completion.items():
        jobs = [{"coordinate": 0, "index": index, "arrival": 300.,
                 "start": 300. if end is not None else None, "completion": end}
                for index, end in enumerate(values)]
        jobs += [{"coordinate": 0, "index": 99, "arrival": 299., "start": 299., "completion": 300.},
                 {"coordinate": 0, "index": 98, "arrival": 901., "start": 901., "completion": 902.}]
        arms[name] = {"jobs": jobs, "snapshots": {"observation_end": trajectory_snapshot(jobs, 1200., 3)}}
    protocol = {"birth_cohort": [300., 900.], "arrival_horizon": 900., "observation_end": 1200.}
    result = cross_outcomes(arms, protocol, ("a", "b", "c"))
    for cutoff in ("arrival_horizon", "observation_end"):
        for name, mean in (("original_current", 10.), ("BG_current", 20.), ("original_BG", 30.), ("BG_BG", 40.)):
            info = result["snapshots"][cutoff][name]
            assert info["arrivals"] == 3 and info["common_jobs"] == 1
            assert info["common_p90_flow_s"] == mean
            assert info["per_coordinate"][0]["common_class_jobs"] == 1
            assert info["per_coordinate"][0]["common_class_p90_flow_s"] == mean
            assert info["per_coordinate"][1]["common_class_jobs"] == 0
            assert info["per_coordinate"][1]["common_class_p90_flow_s"] is None
    current, bg = result["root_effects"]["current"], result["root_effects"]["BG"]
    assert current["delta_T_mean"] == pytest.approx(-20 / 3)
    assert bg["delta_T_mean"] == pytest.approx(-880 / 3)
    assert current["delta_T_common_p90"] == bg["delta_T_common_p90"] == -10.
    assert current["delta_900_common_p90"] == bg["delta_900_common_p90"] == -10.
    assert current["delta_900_unfinished"] == 0 and bg["delta_900_unfinished"] == -1
    assert result["interaction"]["delta_T_mean"] == pytest.approx(860 / 3)
    assert result["interaction"]["delta_T_common_p90"] == 0
    assert result["interaction"]["delta_900_unfinished"] == 1
    assert result["interaction"]["delta_middle_queue_area_s"] == 600.
    assert result["interaction"]["per_coordinate_deltas"][0]["delta_T_common_p90"] == 0


def test_register_then_run_accepts_json_normalized_canonical_cohort_ids_without_execution(tmp_path, monkeypatch):
    import sys
    import algorithm.experiments.aligned_hardware_two_step_bg_incumbent_continuation_cross as driver

    class ReachedExecution(Exception):
        pass

    expected = {"anchors": [{"tape_id": "toy", "visible_cohort_job_ids": [(0, 2)]}],
                "missing_tape_ids": []}
    calls = []

    def saved_runs():
        calls.append("saved_runs")
        raise ReachedExecution

    monkeypatch.setattr(driver, "BASE", tmp_path / "cross")
    monkeypatch.setattr(driver, "build_protocol", lambda: expected)
    monkeypatch.setattr(driver, "saved_runs", saved_runs)
    monkeypatch.setattr(sys, "argv", ["continuation-cross", "--register"])
    driver.main()
    saved = json.loads(driver.path(driver.BASE, "_protocol.json").read_text())
    assert saved["anchors"][0]["visible_cohort_job_ids"] == [[0, 2]]
    assert {key: value for key, value in saved.items() if key != "registered_at_utc"} != expected
    assert not calls
    monkeypatch.setattr(sys, "argv", ["continuation-cross", "--run"])
    with pytest.raises(ReachedExecution):
        driver.main()
    assert calls == ["saved_runs"]
