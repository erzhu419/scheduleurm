import json

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.aligned_hardware_single_action_counterfactual import PairedSelector, build_protocol


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))
MODEL = FrameModel("toy", "single_action", 2, ("a", "b"), (1., 1.), OPTIONS)
TAPE = [(0., 0, 0), (0., 0, 1), (0., 1, 0), (1., 1, 1), (2., 0, 2), (5., 1, 2)]


def factual():
    selector = PairedSelector(ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same"), capture=range(30))
    return simulate_observed_rate_contextual_trajectory(MODEL, TAPE, 20., 20., selector, policy="factual"), selector


def test_null_resume_preserves_same_time_busy_commitments_and_random_stream():
    original, capture = factual()
    for index in (0, 1, 3):
        state = prepare_checkpoint(original, index)
        selector = PairedSelector(ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same"))
        selector.planner.rng.setstate(capture.states[index])
        resumed = resume_single_action_trajectory(MODEL, 20., 20., selector, state, policy="null")
        assert resumed["jobs"] == original["jobs"]
        assert json.loads(json.dumps(original["frames"][:index] + resumed["frames"])) == json.loads(json.dumps(original["frames"]))
        assert selector.planner.rng.getstate() == capture.planner.rng.getstate()


def test_checkpoint_erases_later_admissions_even_at_the_same_timestamp():
    original, _ = factual()
    assert original["frames"][0]["start"] == original["frames"][1]["start"] == 0.
    checkpoint = prepare_checkpoint(original, 1)
    admitted = original["frames"][0]["service"]
    for c in range(2):
        jobs = [j for j in checkpoint["jobs"] if j["coordinate"] == c]
        assert all(j["start"] is not None for j in jobs[:admitted[c]])
        assert all(j["start"] is None and j["completion"] is None for j in jobs[admitted[c]:])
    assert checkpoint["available"] == original["frames"][1]["available_before"]


def test_override_runs_once_and_charges_the_original_proposal():
    original, capture = factual()
    index = next(i for i, f in enumerate(original["frames"]) if f["rollout"]["changed"])
    selector = PairedSelector(ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same"), force_first=True)
    selector.planner.rng.setstate(capture.states[index])
    result = resume_single_action_trajectory(MODEL, 20., 20., selector, prepare_checkpoint(original, index), policy="replacement")
    assert selector.work["override_calls"] == 1 and selector.calls == len(result["frames"])
    assert selector.intervention["original"] != selector.intervention["replacement"]
    assert result["frames"][0]["option_indices"] == (selector.intervention["replacement"],)
    assert selector.work["root_option_probes"] >= result["rollout_work"]["root_option_probes"]


def test_protocol_keeps_all_64_targets_without_outcome_filtering():
    protocol = build_protocol()
    assert len(protocol["targets"]) == 64 and len(protocol["tapes"]) == 3
    assert all(300 <= target["start"] < 900 for target in protocol["targets"])
    assert protocol["planning_samples"] == 8 and protocol["planning_event_cap"] == 128
