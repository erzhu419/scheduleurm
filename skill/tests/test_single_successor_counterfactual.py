"""One successor intervention keeps the same pre-action policy and clock state."""
from dataclasses import asdict
import json

import pytest

from algorithm.deferred_incumbent_background_lane_rollout import DeferredIncumbentBackgroundLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_single_successor_counterfactual import (
    SuccessorCounterfactualSelector, paired_outcomes, registered_targets,
    restore_planner_state,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory


OPTIONS = (FrameOption("loaded", (1, 1), 4., ((1.,), (3.,)), loaded=True),
           FrameOption("a", (1, 0), 5., ((5.,), ())),
           FrameOption("b", (0, 1), 3., ((), (3.,))))
MODEL = FrameModel("toy", "successor", 2, ("a", "b"), (1., 1.), OPTIONS)
TAPE = [(0., 0, 0), (0., 1, 0), (0., 1, 1), (0., 1, 2),
        (1., 0, 1), (2., 1, 3), (5., 0, 2), (7., 1, 4)]


def planner():
    return DeferredIncumbentBackgroundLaneRollout(OPTIONS, "same", sample_count=1)


def factual():
    candidate = planner()
    candidate.pending_successor_index = 0
    selector = SuccessorCounterfactualSelector(candidate, capture=range(50))
    result = simulate_observed_rate_contextual_trajectory(
        MODEL, TAPE, 20., 20., selector, policy="factual")
    return result, selector


def context(frame):
    return (OPTIONS, frame["queue"], frame["available_before"], frame["start"],
            frame["lane_ids"][0], frame["waiting_ages"], frame["observed_arrival_counts"])


def test_null_restore_reproduces_full_policy_state_jobs_and_busy_clock():
    original, capture = factual()
    assert capture.work["checkpoint_planner_state_fields_copied"] == 2 * len(capture.states)
    assert capture.work["checkpoint_rng_words_copied"] == sum(
        len(state["rng_state"][1]) for state in capture.states.values())
    assert original["frames"][0]["rollout"]["successor_changed"]
    assert original["frames"][0]["option_indices"] == (0,)
    assert original["frames"][0]["start"] == original["frames"][1]["start"] == 0.
    # The loaded lane remains reserved until4, despite completions at1 and3.
    busy = prepare_checkpoint(original, 1)
    assert busy["available"] == (4., 0.)
    prior = [job for job in busy["jobs"] if job["start"] is not None]
    assert [job["completion"] for job in prior] == [1., 3.]
    assert all(job["completion"] is None for job in busy["jobs"] if job["start"] is None)
    for index in (0, 1, 3):
        restored = planner()
        state = capture.states[index]
        fees = restore_planner_state(restored, state)
        assert restored.rng.getstate() == state["rng_state"]
        assert restored.decision_counter == state["decision_counter"]
        assert restored.pending_successor_index == state["pending_successor_index"]
        assert fees == {"checkpoint_rng_words_restored": len(state["rng_state"][1]),
                        "checkpoint_planner_state_fields_restored": 2}
        null = SuccessorCounterfactualSelector(restored)
        resumed = resume_single_action_trajectory(
            MODEL, 20., 20., null, prepare_checkpoint(original, index), policy="null")
        assert resumed["jobs"] == original["jobs"]
        assert json.loads(json.dumps(original["frames"][:index] + resumed["frames"])) == json.loads(json.dumps(original["frames"]))
        assert restored.rng.getstate() == capture.planner.rng.getstate()
        assert restored.decision_counter == capture.planner.decision_counter
        assert restored.pending_successor_index == capture.planner.pending_successor_index


def test_exactly_one_successor_override_consumes_factual_state_and_charges_both_scans():
    original, capture = factual()
    candidate = planner()
    restore_planner_state(candidate, capture.states[0])
    selector = SuccessorCounterfactualSelector(candidate, force_first=True)
    before_rng = candidate.rng.getstate()
    result = selector(*context(original["frames"][0]))
    assert result.selection.option_indices == (2,)
    assert selector.intervention["original"] == 0 and selector.intervention["replacement"] == 2
    assert candidate.pending_successor_index is None and candidate.decision_counter == 1
    # The factual pending call sampled no prefix; overriding it must not start
    # a new planning decision or create an extra decision-counter increment.
    assert candidate.rng.getstate() == before_rng
    assert selector.status["proposal_calls"] == selector.work["override_calls"] == 1
    assert selector.status["proposal_successor_executed"] == selector.status["proposal_successor_changed"] == 1
    assert selector.status["applied_successor_executed"] == selector.status["applied_successor_changed"] == 0
    assert selector.work["root_option_probes"] == 2 * len(OPTIONS)
    assert selector.work["planning_random_draws"] == 0
    assert selector.work["successor_override_calls"] == selector.work["successor_envelope_checks"] == 1
    assert selector.work["successor_feasibility_checks"] == 1
    assert selector.work["successor_feasibility_coordinate_reads"] == 2
    reference = planner()
    restore_planner_state(reference, {"rng_state": candidate.rng.getstate(),
        "decision_counter": candidate.decision_counter,
        "pending_successor_index": candidate.pending_successor_index})
    # Actual MW replacement reserved lane0 for3; lane1 now sees the remaining
    # causal queue. Every later call is the unmodified policy at that state.
    next_context = (OPTIONS, (1, 2), (3., 0.), 0., 1, ((0.,), (0., 0.)), (1, 3))
    expected = reference(*next_context)
    second = selector(*next_context)
    assert asdict(second) == asdict(expected)
    assert candidate.rng.getstate() == reference.rng.getstate()
    assert candidate.decision_counter == reference.decision_counter == 2
    assert candidate.pending_successor_index == reference.pending_successor_index
    assert selector.work["override_calls"] == 1 and selector.status["proposal_calls"] == 2
    assert selector.work["root_option_probes"] == 2 * len(OPTIONS) + expected.root_option_probes


def test_roster_keeps_every_changed_successor_in_window_without_outcome_filtering():
    def frame(start, *, changed=False, successor_call=False, successor_changed=False):
        return {"start": start, "option_indices": (1,), "rollout": {
            "changed": changed, "successor_call": successor_call,
            "successor_executed": successor_call,
            "successor_changed": successor_changed, "planned_successor_index": 1},
            "later_outcome_delta": -1 if start < 500 else 1}

    rows = [frame(299.9, successor_call=True, successor_changed=True),
            frame(300., successor_call=True, successor_changed=True),
            frame(400., successor_call=True),  # Executed but equal to MW.
            frame(500., changed=True),        # Root planning, not successor.
            frame(899.9, successor_call=True, successor_changed=True),
            frame(900., successor_call=True, successor_changed=True)]
    targets = registered_targets([{"tape_id": "one"}], {"one": {"frames": rows}})
    assert [target["frame_index"] for target in targets] == [1, 4]
    assert [target["start"] for target in targets] == [300., 899.9]
    assert all(target["tape_id"] == "one" for target in targets)


def test_paired_outcomes_include_same_closed_birth_cohort_and_censor_both_cutoffs():
    jobs = [{"coordinate": 0, "index": 0, "arrival": 300., "start": 400., "completion": 600.},
            {"coordinate": 0, "index": 1, "arrival": 500., "start": None, "completion": None},
            {"coordinate": 1, "index": 0, "arrival": 900., "start": 1000., "completion": 1300.},
            {"coordinate": 0, "index": 2, "arrival": 299., "start": 300., "completion": 301.},
            {"coordinate": 1, "index": 1, "arrival": 901., "start": 902., "completion": 903.}]
    factual_result = {"jobs": jobs, "snapshots": {"observation_end": {"per_coordinate": [{}, {}]}}}
    replacement_jobs = [dict(job) for job in jobs]
    replacement_jobs[0]["completion"] = 650.
    replacement_jobs[2]["completion"] = 1100.
    # Outside-cohort differences must never contribute to paired endpoints.
    replacement_jobs[3]["completion"] = replacement_jobs[4]["completion"] = 2000.
    replacement = {**factual_result, "jobs": replacement_jobs}
    protocol = {"arrival_horizon": 900., "observation_end": 1200., "birth_cohort": [300., 900.]}
    outcomes = paired_outcomes(factual_result, replacement, protocol)
    end = outcomes["snapshots"]["observation_end"]
    checkpoint = outcomes["snapshots"]["arrival_horizon"]
    assert end["factual"]["arrivals"] == end["replacement"]["arrivals"] == 3
    assert end["factual"]["restricted_mean_flow_s"] == pytest.approx(1300 / 3)
    assert end["replacement"]["restricted_mean_flow_s"] == pytest.approx(1250 / 3)
    assert end["replacement"]["common_p90_flow_s"] - end["factual"]["common_p90_flow_s"] == 50.
    assert checkpoint["factual"]["unfinished_jobs"] == checkpoint["replacement"]["unfinished_jobs"] == 2
    assert end["factual"]["unfinished_jobs"] == 2 and end["replacement"]["unfinished_jobs"] == 1
    assert end["factual"]["per_coordinate"][1]["restricted_mean_flow_s"] == 300.
    assert end["replacement"]["per_coordinate"][1]["restricted_mean_flow_s"] == 200.
    assert outcomes["delta_T_mean"] == pytest.approx(-50 / 3)
    assert outcomes["delta_900_unfinished"] == 0 and outcomes["delta_1200_unfinished"] == -1
    assert outcomes["per_coordinate_deltas"][1]["delta_T_mean"] == -100.
