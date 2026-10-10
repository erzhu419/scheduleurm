"""One root intervention preserves proposal state and its chosen continuation."""
import json

import pytest

from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_background_root_counterfactual import (
    RootCounterfactualSelector, collect_work, paired_outcomes, registered_targets,
    restore_planner_state,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))),
           FrameOption("loaded", (1, 1), 2., ((.5,), (1.5,)), loaded=True))
MODEL = FrameModel("toy", "root", 2, ("a", "b"), (1., 1.), OPTIONS)
TAPE = [(0., 0, 0), (0., 1, 0), (300., 0, 1), (300., 0, 2),
        (301., 1, 1), (304., 0, 3), (306., 1, 2)]


def planner():
    return BackgroundCompetitionContinuousLaneRollout(OPTIONS, "root-null", sample_count=1)


@pytest.fixture(scope="module")
def native():
    selector = RootCounterfactualSelector(planner(), capture=range(700))
    result = simulate_observed_rate_contextual_trajectory(
        MODEL, TAPE, 320., 330., selector, policy="factual")
    targets = registered_targets([{"tape_id": "toy"}], {"toy": result})
    assert len(targets) == 1 and targets[0]["start"] == 300.
    return result, selector, targets[0]


def test_full_counter_rng_null_identity_keeps_loaded_prior_admissions_and_clock(native):
    factual, capture, target = native
    assert factual["frames"][0]["option_indices"] == (3,)
    assert factual["frames"][0]["start"] == factual["frames"][1]["start"] == 0.
    busy = prepare_checkpoint(factual, 1)
    assert busy["available"] == (2., 0.)
    assert [job["completion"] for job in busy["jobs"] if job["start"] is not None] == [.5, 1.5]
    for index in (0, 1, target["frame_index"]):
        candidate = planner()
        state = capture.states[index]
        fees = restore_planner_state(candidate, state)
        assert candidate.decision_counter == state["decision_counter"]
        assert candidate.rng.getstate() == state["rng_state"]
        assert fees == {"checkpoint_rng_words_restored": len(state["rng_state"][1]),
                        "checkpoint_decision_counter_writes": 1}
        null = RootCounterfactualSelector(candidate)
        resumed = resume_single_action_trajectory(
            MODEL, 320., 330., null, prepare_checkpoint(factual, index), policy="null")
        assert resumed["jobs"] == factual["jobs"]
        assert json.loads(json.dumps(factual["frames"][:index] + resumed["frames"])) == json.loads(json.dumps(factual["frames"]))
        assert candidate.rng.getstate() == capture.planner.rng.getstate()
        assert candidate.decision_counter == capture.planner.decision_counter


def test_single_root_override_consumes_same_bg_proposal_and_continuation_ledgers_do_not_double(native):
    factual, capture, target = native
    index = target["frame_index"]
    checkpoint = prepare_checkpoint(factual, index)
    state = capture.states[index]
    frame = factual["frames"][index]
    reference = planner()
    restore_planner_state(reference, state)
    proposed = reference(OPTIONS, frame["queue"], frame["available_before"], frame["start"],
                         frame["lane_ids"][0], frame["waiting_ages"], frame["observed_arrival_counts"])
    assert proposed.changed and proposed.selection.option_indices == (0,)
    initial_proposals = []
    for continuation in ("BG", "MW"):
        for forced in (False, True):
            candidate = planner()
            restore_planner_state(candidate, state)
            selector = RootCounterfactualSelector(candidate, force_first=forced, continuation=continuation)
            branch = resume_single_action_trajectory(
                MODEL, 320., 330., selector, checkpoint, policy=continuation)
            initial_proposals.append(selector.first_proposal)
            assert branch["frames"][0]["option_indices"] == ((1,) if forced else (0,))
            assert selector.work["override_calls"] == int(forced)
            if forced:
                assert selector.intervention["original"] == 0 and selector.intervention["replacement"] == 1
            if continuation == "MW":
                assert selector.status["BG_proposal_calls"] == 1
                assert selector.status["MW_continuation_calls"] == len(branch["frames"]) - 1
                assert candidate.rng.getstate() == reference.rng.getstate()
                assert candidate.decision_counter == state["decision_counter"] + 1
                assert selector.work["rollout_calls"] == proposed.rollout_calls
                assert selector.work["planning_random_draws"] == proposed.planning_random_draws
                assert selector.work["root_option_probes"] == proposed.root_option_probes + (
                    len(branch["frames"]) - 1 + int(forced)) * len(OPTIONS)
            else:
                assert selector.status["BG_proposal_calls"] == len(branch["frames"])
                assert selector.status["MW_continuation_calls"] == 0
                assert candidate.decision_counter == state["decision_counter"] + len(branch["frames"])
            work = collect_work(selector, branch)
            assert work["root_option_probes"] == selector.work["root_option_probes"]
            assert work["rollout_calls"] == selector.work["rollout_calls"]
            assert work["age_context_job_reads"] == branch["rollout_work"]["age_context_job_reads"]
    assert all(proposal == initial_proposals[0] for proposal in initial_proposals)


def test_roster_takes_first_in_window_per_tape_without_filtering_later_outcomes():
    def frame(start, changed=True, outcome=0):
        return {"start": start, "option_indices": (0,), "rollout": {"changed": changed},
                "later_outcome_delta": outcome}

    tapes = [{"tape_id": name} for name in ("one", "zero", "three")]
    runs = {"one": {"frames": [frame(299.9), frame(300., outcome=1),
                                 frame(400., outcome=-1), frame(900.)]},
            "zero": {"frames": [frame(300., changed=False), frame(900.)]},
            "three": {"frames": [frame(899.9, outcome=-1)]}}
    targets = registered_targets(tapes, runs)
    assert [(target["tape_id"], target["frame_index"], target["start"])
            for target in targets] == [("one", 1, 300.), ("three", 0, 899.9)]
    assert [tape["tape_id"] for tape in tapes] == ["one", "zero", "three"]


def test_endpoints_use_closed_birth_cohort_and_same_class_common_ids_with_all_job_queue_area():
    jobs = [{"coordinate": 0, "index": 0, "arrival": 300., "start": 400., "completion": 600.},
            {"coordinate": 0, "index": 1, "arrival": 500., "start": None, "completion": None},
            {"coordinate": 0, "index": 2, "arrival": 600., "start": 700., "completion": 800.},
            {"coordinate": 1, "index": 0, "arrival": 350., "start": 400., "completion": 550.},
            {"coordinate": 1, "index": 1, "arrival": 900., "start": 1000., "completion": 1300.},
            {"coordinate": 0, "index": 3, "arrival": 299., "start": 320., "completion": 350.},
            {"coordinate": 1, "index": 2, "arrival": 901., "start": 902., "completion": 903.}]
    replacement_jobs = [dict(job) for job in jobs]
    replacement_jobs[0].update(start=450., completion=650.)
    replacement_jobs[2].update(start=1000., completion=1250.)
    replacement_jobs[3].update(start=450., completion=600.)
    replacement_jobs[4]["completion"] = 1100.
    replacement_jobs[5].update(start=330., completion=2000.)
    replacement_jobs[6]["completion"] = 2000.
    original = {"jobs": jobs, "snapshots": {"observation_end": {"per_coordinate": [{}, {}]}}}
    replacement = {**original, "jobs": replacement_jobs}
    protocol = {"birth_cohort": [300., 900.], "arrival_horizon": 900., "observation_end": 1200.}
    outcomes = paired_outcomes(original, replacement, protocol, ("a", "b"))
    end = outcomes["snapshots"]["observation_end"]
    assert end["factual"]["arrivals"] == end["replacement"]["arrivals"] == 5
    assert outcomes["delta_T_mean"] == pytest.approx(80.)
    assert outcomes["delta_T_common_p90"] == 50.
    assert outcomes["delta_900_unfinished"] == 1 and outcomes["delta_1200_unfinished"] == 0
    assert outcomes["delta_middle_queue_area_s"] == 310.
    classes = outcomes["per_coordinate_deltas"]
    assert classes[0]["delta_T_mean"] == 150. and classes[1]["delta_T_mean"] == -25.
    # The factual-only completed A600 job cannot enter either arm's common
    # class P90. Exactly A300 belongs, so its paired P90 difference is50.
    assert end["factual"]["per_coordinate"][0]["common_class_jobs"] == 1
    assert end["replacement"]["per_coordinate"][0]["common_class_jobs"] == 1
    assert classes[0]["delta_T_common_p90"] == 50.
