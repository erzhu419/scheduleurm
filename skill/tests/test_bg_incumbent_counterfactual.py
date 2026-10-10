"""Controlled objective contrasts test replay mechanics, not scientific gain."""
from dataclasses import replace
from fractions import Fraction
import json

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.bg_incumbent_background_redecision_lane_rollout as policy_module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.bg_incumbent_background_redecision_lane_rollout import BGIncumbentBackgroundRedecisionLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.bg_incumbent_counterfactual_replay import (
    BGIncumbentCounterfactualSelector, collect_work, registered_targets,
    restore_planner_state,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory


OPTIONS = (FrameOption("single", (1,), 1., ((1.,),)),
           FrameOption("loaded-pair", (2,), 1.5, ((.5, 1.),), loaded=True),
           FrameOption("triple", (3,), 2., ((2., 2., 2.),)))
MODEL = FrameModel("toy", "incumbent-replay", 2, ("a",), (1.,), OPTIONS)
TAPE = [(0., 0, 0), (0., 0, 1), (3., 0, 2), (3., 0, 3), (3., 0, 4), (6., 0, 5)]


def normalized(value):
    return json.loads(json.dumps(value))


def controlled_planner():
    candidate = BGIncumbentBackgroundRedecisionLaneRollout(OPTIONS, "replay", sample_count=1)
    score = candidate.score_background_roots

    def contrast(*args):
        actual = score(*args)
        if tuple(args[0]) == (3,) and args[2] == 3.:
            # Ordinary BG naturally chooses non-MW loaded-pair1 at this
            # state. Only its once-stage objective contrast is controlled;
            # all native simulations, two-stage and child fees are retained.
            return replace(actual,
                costs={**actual.costs, 0: actual.costs[1] - Fraction(1, 2)},
                squared_costs={**actual.squared_costs, 0: actual.squared_costs[1]})
        return actual

    candidate.score_background_roots = contrast
    return candidate


@pytest.fixture(scope="module")
def native():
    with pytest.MonkeyPatch.context() as patch:
        def no_future(rates, window, rng, cap):
            positive = sum(rate > 0 for rate in rates)
            for _ in range(positive):
                rng.random()
            return PlanningArrivals((), False, positive, len(rates))

        patch.setattr(policy_module, "sample_planning_arrivals", no_future)
        patch.setattr(background_module, "sample_planning_arrivals", no_future)
        selector = BGIncumbentCounterfactualSelector(controlled_planner(), capture=range(40))
        factual = simulate_observed_rate_contextual_trajectory(MODEL, TAPE, 8., 10., selector, policy="factual")
        targets = registered_targets([{"tape_id": "toy"}], {"toy": factual})
        assert len(targets) == 1
        target = targets[0]
        assert (target["start"], target["factual_option_index"],
                target["replacement_option_index"], target["maxweight_option_index"]) == (3., 0, 1, 2)
        yield factual, selector, target


@pytest.fixture(scope="module")
def forced(native):
    factual, captured, target = native
    index = target["frame_index"]
    state = captured.states[index]
    candidate = controlled_planner()
    restore_fees = restore_planner_state(candidate, state)
    selector = BGIncumbentCounterfactualSelector(candidate, capture=(1,), force_first=True)
    branch = resume_single_action_trajectory(
        MODEL, 8., 10., selector, prepare_checkpoint(factual, index), policy="forced")
    return branch, selector, state, restore_fees


def test_rng_counter_null_reproduces_cache_fifo_and_full_loaded_commitments(native):
    factual, captured, target = native
    busy = prepare_checkpoint(factual, 1)
    assert busy["available"] == (1.5, 0.)
    assert [job["completion"] for job in busy["jobs"] if job["start"] is not None] == [.5, 1.]
    assert factual["frames"][1]["observed_arrival_counts"] == (2,)
    for index in (1, target["frame_index"]):
        candidate = controlled_planner()
        state = captured.states[index]
        fees = restore_planner_state(candidate, state)
        assert fees == {"checkpoint_rng_words_restored": 625,
                        "checkpoint_decision_counter_writes": 1}
        assert candidate.rng.getstate() == state["rng_state"]
        assert candidate.decision_counter == state["decision_counter"]
        selector = BGIncumbentCounterfactualSelector(candidate)
        resumed = resume_single_action_trajectory(
            MODEL, 8., 10., selector, prepare_checkpoint(factual, index), policy="null")
        assert resumed["jobs"] == factual["jobs"]
        assert normalized(factual["frames"][:index] + resumed["frames"]) == normalized(factual["frames"])
        assert normalized(selector.first_proposal["rollout"]) == normalized(factual["frames"][index]["rollout"])
        assert candidate.rng.getstate() == captured.planner.rng.getstate()
        assert candidate.decision_counter == captured.planner.decision_counter


def test_first_nonmw_bg_swap_keeps_early_offsets_period_and_freely_replans(native, forced):
    factual, _, target = native
    branch, selector, _, _ = forced
    assert branch["frames"][0]["option_indices"] == (1,)
    assert selector.intervention["original"] == 0
    assert selector.intervention["replacement"] == selector.intervention["actual_applied_option_index"] == 1
    assert selector.intervention["maxweight_option_index"] == 2
    assert selector.first_proposal["improved_incumbent"]
    assert normalized(selector.first_proposal["rollout"]) == normalized(factual["frames"][target["frame_index"]]["rollout"])
    completions = {job["index"]: job["completion"] for job in branch["jobs"]}
    assert completions[0] == .5 and completions[1] == 1.  # Earlier admissions remain known.
    assert completions[2] == 3.5 and completions[3] == 4.
    assert branch["frames"][1]["available_before"] == (3.5, 4.5)
    assert branch["frames"][1]["start"] == 3.5
    assert branch["frames"][1]["option_indices"] == (0,)  # A fresh Q1 decision.
    assert branch["frames"][1]["rollout"]["eligible"]
    assert branch["frames"][1]["rollout"]["sample_count"] == 1
    assert not branch["frames"][0]["rollout"]["improved_incumbent"]
    assert branch["frames"][0]["rollout"]["changed"]  # Forced BG also differs from MW.
    assert selector.work["override_calls"] == 1 and not hasattr(selector.planner, "pending_successor_index")


def test_swap_consumes_exact_proposal_state_and_charges_each_computation_once(native, forced):
    factual, _, target = native
    branch, selector, state, restore_fees = forced
    frame = factual["frames"][target["frame_index"]]
    reference = controlled_planner()
    restore_planner_state(reference, state)
    proposed = reference(OPTIONS, frame["queue"], frame["available_before"], frame["start"],
                         frame["lane_ids"][0], frame["waiting_ages"], frame["observed_arrival_counts"])
    assert proposed.improved_incumbent and proposed.incumbent_root_index == 1
    # The checkpoint immediately before call2 proves that the full proposal
    # advanced state once; the extra certificate MW scan advances neither.
    assert selector.states[1] == {"rng_state": reference.rng.getstate(),
                                  "decision_counter": reference.decision_counter}
    assert selector.planner.decision_counter == state["decision_counter"] + len(branch["frames"])
    assert selector.work["override_root_option_probes"] == len(OPTIONS)
    assert selector.intervention["override_root_option_probes"] == len(OPTIONS)
    for key in ("root_option_probes", "incumbent_rollout_option_probes",
                "recourse_rollout_option_probes", "planning_random_draws"):
        assert selector.work[key] == sum(item["rollout"][key] for item in branch["frames"])
    assert selector.work["incumbent_rollout_option_probes"] > 0
    assert selector.work["recourse_rollout_option_probes"] > 0
    work = collect_work(selector, branch)
    assert work["controller_constructors"] == work["recourse_controller_constructors"] == 1
    assert work["preprocessing_option_probes"] == work["preprocessing_recourse_option_probes"] == len(OPTIONS)
    assert work["root_option_probes"] == selector.work["root_option_probes"]
    assert work["override_root_option_probes"] == len(OPTIONS)
    assert work["age_context_job_reads"] == branch["rollout_work"]["age_context_job_reads"]
    assert work["observed_arrival_context_coordinate_reads"] == branch["rollout_work"]["observed_arrival_context_coordinate_reads"]
    assert restore_fees["checkpoint_rng_words_restored"] == 625
    assert selector.status["proposal_calls"] == selector.status["applied_calls"] == selector.calls
    assert selector.status["proposal_improved_incumbent"] == 1
    assert selector.status["applied_improved_incumbent"] == 0


def test_all_improved_targets_include_early_late_and_return_to_mw_without_outcome_filter():
    def frame(start, improved=True, actual=0, outcome=0):
        return {"start": start, "option_indices": (actual,),
                "rollout": {"improved_incumbent": improved, "incumbent_root_index": 1,
                            "incumbent_maxweight_root_index": 2, "changed": actual != 2},
                "later_flow_delta": outcome}

    tapes = [{"tape_id": key} for key in ("all", "zero", "last")]
    factual = {"all": {"frames": [frame(0., outcome=1), frame(299., outcome=-1),
                                   frame(300., improved=False), frame(900., outcome=0),
                                   frame(1200., actual=2, outcome=1)]},
               "zero": {"frames": [frame(500., improved=False)]},
               "last": {"frames": [frame(950., outcome=-1)]}}
    targets = registered_targets(tapes, factual)
    assert [(item["tape_id"], item["frame_index"], item["start"]) for item in targets] == [
        ("all", 0, 0.), ("all", 1, 299.), ("all", 3, 900.), ("all", 4, 1200.), ("last", 0, 950.)]
    assert targets[3]["factual_option_index"] == targets[3]["maxweight_option_index"] == 2
    assert [item["tape_id"] for item in tapes] == ["all", "zero", "last"]
