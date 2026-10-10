"""Contextual decisions preserve causal FIFO and committed lane service."""
from dataclasses import replace
from functools import partial

import pytest

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory
from algorithm.experiments.lane_refill_performance_replay import simulate_lane_refill_trajectory
from algorithm.lane_refill_simple_controls import select_single_lane_control
from algorithm.visible_queue_clearing_rollout import (
    select_context_maxweight, select_visible_queue_clearing,
)


def model(*, lanes=2, options=None):
    if options is None:
        options = (
            FrameOption("short_a", (1, 0), 2., ((2.,), ())),
            FrameOption("long_b", (0, 1), 3., ((), (3.,))),
        )
    return FrameModel("small", "contextual_refill", lanes,
                      ("a", "b"), (1., 1.), options)


def run(current, tape, *, horizon=6., cutoff=15., selector=select_context_maxweight):
    return simulate_contextual_trajectory(
        current, tape, horizon, cutoff, selector, policy="context_maxweight")


def frame_identity(result):
    return [(frame["start"], frame["queue"], frame["lane_ids"],
             frame["duration"], frame["option_indices"], frame["service"])
            for frame in result["frames"]]


@pytest.mark.parametrize("lanes", (2, 4))
def test_context_maxweight_reproduces_original_single_lane_replay(lanes):
    current = model(lanes=lanes)
    tape = ((0., 0, 0), (0., 1, 0), (0., 0, 1),
            (.5, 0, 2), (2.5, 1, 1), (4., 0, 3))
    old, _ = simulate_lane_refill_trajectory(
        current, tape, 6., 15., 0,
        partial(select_single_lane_control, policy="enum_maxweight"),
        policy="enum_maxweight", dispatch_rule="single")
    actual = run(current, tape)
    assert frame_identity(actual) == frame_identity(old)
    assert actual["jobs"] == old["jobs"]
    assert actual["snapshots"] == old["snapshots"]
    assert actual["lane_events"] == old["lane_events"]
    assert actual["last_frame_end"] == old["last_frame_end"]
    for key in ("occupied_lane_time_s", "barrier_idle_lane_time_s",
                "scheduled_idle_lane_time_s", "waiting_area_s"):
        assert actual[key] == old[key]


def test_selector_receives_busy_lane_availability_and_excludes_admitted_jobs():
    calls = []

    def recorder(options, queue, available, now, lane):
        calls.append((tuple(queue), tuple(available), now, lane))
        return select_context_maxweight(options, queue, available, now, lane)

    actual = run(model(), ((0., 0, 0), (0., 1, 0), (.5, 0, 1)),
                 selector=recorder)
    assert calls == [((1, 1), (0., 0.), 0., 0),
                     ((0, 1), (2., 0.), 0., 1),
                     ((1, 0), (2., 3.), 2., 0)]
    assert [frame["available_before"] for frame in actual["frames"]] == [
        call[1] for call in calls]
    assert actual["jobs"][1]["completion"] == 3.
    assert actual["jobs"][2]["start"] == 2.


def test_loaded_early_target_completion_keeps_lane_reserved_for_full_cycle():
    current = model(lanes=1, options=(
        FrameOption("loaded", (1, 1), 5., ((1.,), (5.,)), loaded=True),))
    actual = run(current, ((0., 0, 0), (0., 1, 0),
                           (.5, 0, 1), (.5, 1, 1)))
    assert actual["jobs"][0]["completion"] == 1.
    assert actual["jobs"][2]["start"] == 5.
    assert [frame["start"] for frame in actual["frames"]] == [0., 5.]
    assert actual["lane_events"][0]["local_end"] == 5.
    assert actual["lane_events"][0]["reserved_end"] == 5.


def test_different_future_arrivals_leave_released_prefix_decisions_unchanged():
    common = ((0., 0, 0), (0., 1, 0), (.5, 0, 1))
    first = run(model(), common + ((3.1, 0, 2),),
                selector=select_visible_queue_clearing)
    second = run(model(), common + ((3.2, 1, 1), (3.3, 0, 2)),
                 selector=select_visible_queue_clearing)
    prefix = lambda result: [row for row in frame_identity(result) if row[0] < 3.1]
    assert prefix(first) == prefix(second)


def test_fifo_offsets_and_cutoff_keep_inflight_and_queued_tails():
    current = model(lanes=1, options=(
        FrameOption("batch", (2, 0), 10., ((8., 2.), ())),))
    actual = run(current, ((0., 0, 0), (0., 0, 1),
                           (0., 0, 2), (1., 0, 3)), horizon=2., cutoff=4.)
    assert [job["completion"] for job in actual["jobs"]] == [2., 8., None, None]
    assert actual["jobs"][2]["start"] is None
    snapshot = actual["snapshots"]["observation_end"]
    assert (snapshot["completed_jobs"], snapshot["inflight_jobs"],
            snapshot["queued_jobs"], snapshot["unfinished_jobs"]) == (1, 1, 2, 3)
    assert snapshot["restricted_mean_flow_s"] == pytest.approx(13 / 4)
    assert actual["occupied_lane_time_s"] == 4.
    assert actual["waiting_area_s"] == 7.


def test_idle_polling_preserves_service_clock_and_stops_after_final_admission():
    actual = run(model(lanes=1), ((.25, 0, 0),))
    assert [frame["start"] for frame in actual["frames"]] == [0., 1.]
    assert actual["jobs"][0]["start"] == 1.
    assert actual["jobs"][0]["completion"] == 3.
    assert actual["frame_count"] == 2
    assert actual["scheduled_idle_lane_time_s"] == 1.
    assert actual["waiting_area_s"] == .75
    assert actual["total_decision_delay_s"] == 0.
    assert all(frame["decision_delay_s"] == 0. for frame in actual["frames"])


def test_prediction_and_rollout_work_are_recorded_without_advancing_service_clock():
    def annotated(options, queue, available, now, lane):
        decision = select_context_maxweight(options, queue, available, now, lane)
        return replace(decision, eligible=True, candidate_count=9,
                       rollout_calls=9, rollout_option_probes=18,
                       root_option_probes=2,
                       baseline_predicted_completion_s=88.,
                       chosen_predicted_completion_s=77., changed=True,
                       queue_limit=16)

    tape = ((0., 0, 0), (0., 1, 0), (.5, 0, 1))
    baseline = run(model(), tape)
    actual = run(model(), tape, selector=annotated)
    assert frame_identity(actual) == frame_identity(baseline)
    assert actual["jobs"] == baseline["jobs"]
    assert actual["snapshots"] == baseline["snapshots"]
    assert actual["lane_events"] == baseline["lane_events"]
    assert actual["total_decision_delay_s"] == 0.
    for frame in actual["frames"]:
        assert frame["rollout"] == {
            "eligible": True, "candidate_count": 9, "rollout_calls": 9,
            "rollout_option_probes": 18, "root_option_probes": 2,
            "baseline_predicted_completion_s": 88.,
            "chosen_predicted_completion_s": 77., "changed": True,
            "queue_limit": 16,
        }
    calls = actual["frame_count"]
    assert actual["rollout_work"]["candidate_count"] == 9 * calls
    assert actual["rollout_work"]["rollout_calls"] == 9 * calls
    assert actual["rollout_work"]["rollout_option_probes"] == 18 * calls
    assert actual["rollout_work"]["root_option_probes"] == 2 * calls
    assert actual["rollout_work"]["eligible_decisions"] == calls
    assert actual["rollout_work"]["changed_decisions"] == calls
    assert actual["rollout_work"]["maximum_rollout_calls"] == 9
