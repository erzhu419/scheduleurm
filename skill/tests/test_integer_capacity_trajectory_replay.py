"""Exercise causal frame execution and finite-horizon accounting on small tapes."""
from dataclasses import replace

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.integer_capacity_shared_threshold_frame_candidates import (
    select_integer_capacity_shared_threshold_frame,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments import integer_capacity_trajectory_replay as replay


def _model(*, duration=2.0, offsets=(2.0,), lanes=1):
    return FrameModel(
        node="small_replay", scenario_id="finite_cutoff", lanes=lanes,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a", (len(offsets), 0, 0), duration, (offsets, (), ())),),
    )


def _run(model, tape, *, horizon=5.0, cutoff=10.0, selector=select_canonical_count_space_bnb_frame, **kwargs):
    return replay.simulate_trajectory(model, tape, horizon, cutoff, 36, selector, **kwargs)


def test_same_released_history_produces_same_prefix_despite_different_futures():
    model = _model()
    common = ((0.0, 0, 0), (0.5, 0, 1))
    first = _run(model, common + ((3.1, 0, 2),))
    second = _run(model, common + ((3.2, 0, 2), (3.3, 0, 3)))
    prefix = lambda result: [
        (frame["start"], frame["queue"], frame["option_indices"], frame["service"])
        for frame in result["frames"] if frame["start"] < 3.1
    ]
    assert prefix(first) == prefix(second)
    assert [frame["queue"] for frame in first["frames"][:2]] == [(1, 0, 0), (1, 0, 0)]


def test_cutoff_counts_early_departures_inflight_and_queued_jobs_separately():
    model = _model(duration=10.0, offsets=(8.0, 2.0))
    tape = ((0.0, 0, 0), (0.0, 0, 1), (0.0, 0, 2), (1.0, 0, 3))
    result = _run(model, tape, horizon=2.0, cutoff=4.0)
    at_end = result["snapshots"]["observation_end"]
    assert at_end["arrivals"] == 4
    assert at_end["completed_jobs"] == 1
    assert at_end["inflight_jobs"] == 1
    assert at_end["queued_jobs"] == 2
    assert at_end["unfinished_jobs"] == 3
    assert at_end["restricted_mean_flow_s"] == pytest.approx(13 / 4)
    assert at_end["completed_mean_flow_s"] == 2.0
    assert at_end["completion_rate"] == 0.25
    jobs = {(job["coordinate"], job["index"]): job for job in result["jobs"]}
    assert jobs[(0, 0)]["completion"] == 2.0
    assert jobs[(0, 1)]["completion"] == 8.0
    assert jobs[(0, 2)]["start"] is None
    assert jobs[(0, 3)]["start"] is None
    assert all(result["invariants"].values())


def test_arrivals_during_frame_are_not_served_until_the_next_frame():
    model = _model(duration=5.0, offsets=(5.0,))
    result = _run(model, ((0.0, 0, 0), (0.25, 0, 1)), horizon=1.0, cutoff=8.0)
    assert result["frames"][0]["queue"] == (1, 0, 0)
    assert result["frames"][0]["service"] == (1, 0, 0)
    jobs = result["jobs"]
    assert jobs[0]["start"] == 0.0
    assert jobs[1]["start"] == 5.0
    assert jobs[1]["completion"] == 10.0
    assert result["snapshots"]["observation_end"]["inflight_jobs"] == 1


def test_idle_clock_keeps_one_second_frames_and_does_not_jump_to_arrival():
    calls = []

    def selector(options, queue, lanes, budget):
        calls.append(tuple(queue))
        return select_integer_capacity_shared_threshold_frame(options, queue, lanes, budget)

    result = _run(_model(), ((0.25, 0, 0),), horizon=1.0, cutoff=4.0, selector=selector)
    assert result["frames"][0]["start"] == 0.0
    assert result["frames"][0]["duration"] == 1.0
    assert result["jobs"][0]["start"] == 1.0
    assert result["jobs"][0]["completion"] == 3.0
    assert calls[0] == (0, 0, 0)
    assert len(calls) == len(result["frames"])
    assert result["selector_metrics"]["failures"] == 0


def test_policy_calls_are_not_reused_when_a_queue_state_repeats():
    calls = []

    def selector(options, queue, lanes, budget):
        calls.append(tuple(queue))
        return select_canonical_count_space_bnb_frame(options, queue, lanes, budget)

    result = _run(_model(), ((0.0, 0, 0), (0.5, 0, 1)), selector=selector)
    assert calls.count((1, 0, 0)) == 2
    assert len(calls) == len(result["frames"]) == result["selector_metrics"]["calls"]


def test_reference_audits_candidate_visited_states_after_path_divergence(monkeypatch):
    model = _model()
    tape = ((0.0, 0, 0), (0.5, 0, 1))
    audit = replay.ReferenceAudit(model, budget=4096)
    replay.simulate_trajectory(model, tape, 1.0, 6.0, 4096,
                               select_canonical_count_space_bnb_frame,
                               reference_cache=audit, policy="full_reference")
    assert set(audit.by_queue) == {(1, 0, 0)}
    audit_calls, policy_calls = [], []

    def exact_reference(options, queue, lanes, budget):
        audit_calls.append((tuple(queue), budget))
        return select_canonical_count_space_bnb_frame(options, queue, lanes, budget)

    def idle_once(options, queue, lanes, budget):
        policy_calls.append(tuple(queue))
        result = select_canonical_count_space_bnb_frame(options, queue, lanes, budget)
        if len(policy_calls) == 1:
            return replace(result, option_counts=(0,), option_indices=(None,), service=(0, 0, 0),
                           duration=1.0, score=0.0, gap=result.upper_bound,
                           exact=False, score_exact=False, canonical_exact=False)
        return result

    monkeypatch.setattr(replay, "select_canonical_count_space_bnb_frame", exact_reference)
    result = _run(model, tape, horizon=1.0, cutoff=6.0,
                  selector=idle_once, reference_cache=audit)
    assert (2, 0, 0) in policy_calls
    assert audit_calls == [((2, 0, 0), 4096)]
    assert result["reference_audit"]["actual_calls"] == 1
    assert result["reference_audit"]["incomplete_frame_count"] == 0
    assert result["selector_metrics"]["failures"] == 0
    assert result["selector_metrics"]["canonical_action_mismatch_count"] == 1
    assert len(policy_calls) == result["frame_count"]


def test_reconstructed_service_mismatch_is_rejected_before_job_mutation():
    def invalid(options, queue, lanes, budget):
        result = select_canonical_count_space_bnb_frame(options, queue, lanes, budget)
        return replace(result, service=(2, 0, 0))

    with pytest.raises(ValueError):
        _run(_model(), ((0.0, 0, 0),), selector=invalid)


def test_false_score_certificate_is_recorded_without_hiding_physical_results():
    def invalid(options, queue, lanes, budget):
        result = select_canonical_count_space_bnb_frame(options, queue, lanes, budget)
        return replace(result, score=result.score + 1.0)

    result = _run(_model(), ((0.0, 0, 0),), selector=invalid)
    assert result["selector_metrics"]["failures"] > 0
    assert result["selector_metrics"]["first_failure"] is not None
    assert result["snapshots"]["observation_end"]["completed_jobs"] == 1
