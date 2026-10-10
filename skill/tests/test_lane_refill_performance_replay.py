"""Causal lane refill, committed service, FIFO slots and censored tails."""
from dataclasses import replace

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.integer_capacity_decision_delay_replay import (
    simulate_delayed_trajectory,
)
from algorithm.experiments.lane_refill_performance_replay import (
    simulate_lane_refill_trajectory,
)


def model(*, lanes=2, options=None):
    if options is None:
        options = (FrameOption("short_a", (1, 0), 2., ((2.,), ())),
                   FrameOption("long_b", (0, 1), 3., ((), (3.,))))
    return FrameModel("small", "lane_refill", lanes, ("a", "b"), (1., 1.), options)


def run(current, tape, *, rule="refill", horizon=6., cutoff=15.,
        selector=select_canonical_count_space_bnb_frame):
    return simulate_lane_refill_trajectory(
        current, tape, horizon, cutoff, 64, selector,
        policy="test", dispatch_rule=rule)


def identity(result):
    return [(frame["start"], frame["queue"], frame["duration"],
             frame["option_indices"], frame["service"]) for frame in result["frames"]]


def test_synchronous_matches_original_service_only_fifo_replay():
    current = model()
    tape = ((0., 0, 0), (0., 1, 0), (.5, 0, 1), (3., 1, 1))
    old, _ = simulate_delayed_trajectory(current, tape, 6., 15., 64,
                                         select_canonical_count_space_bnb_frame,
                                         clock=lambda: 0)
    actual, originals = run(current, tape, rule="synchronous")
    assert identity(actual) == identity(old)
    assert actual["jobs"] == old["jobs"]
    assert actual["snapshots"] == old["snapshots"]
    assert actual["last_frame_end"] == old["last_frame_end"]
    assert len(originals) == actual["frame_count"]
    assert all(actual["invariants"].values())


def test_refill_uses_shorter_lane_before_longer_commitment_finishes():
    tape = ((0., 0, 0), (0., 1, 0), (.5, 0, 1))
    synchronous, _ = run(model(), tape, rule="synchronous")
    refill, _ = run(model(), tape)
    assert refill["jobs"][2]["start"] == 2.
    assert synchronous["jobs"][2]["start"] == 3.
    assert refill["frames"][1]["lane_ids"] == (0,)
    assert refill["frames"][1]["lanes_called"] == 1
    assert refill["barrier_idle_lane_time_s"] == 0.
    assert synchronous["barrier_idle_lane_time_s"] > 0.


def test_loaded_target_completion_does_not_release_lane_before_resident_finishes():
    loaded = FrameOption("loaded", (1, 1), 5., ((1.,), (5.,)), loaded=True)
    current = model(lanes=1, options=(loaded,))
    tape = ((0., 0, 0), (0., 1, 0), (.5, 0, 1), (.5, 1, 1))
    actual, _ = run(current, tape)
    assert actual["jobs"][0]["completion"] == 1.
    assert actual["jobs"][2]["start"] == 5.
    assert [frame["start"] for frame in actual["frames"]] == [0., 5.]
    assert actual["lane_events"][0]["local_end"] == 5.


def test_busy_lane_jobs_are_excluded_from_every_refill_queue():
    calls = []

    def recorder(options, queue, lanes, budget):
        calls.append((tuple(queue), lanes))
        return select_canonical_count_space_bnb_frame(options, queue, lanes, budget)

    actual, _ = run(model(), ((0., 0, 0), (0., 1, 0), (.5, 0, 1)),
                    selector=recorder)
    assert calls == [((1, 1), 2), ((1, 0), 1)]
    assert actual["jobs"][1]["completion"] == 3.
    assert actual["frame_count"] == 2


def test_idle_polling_is_not_triggered_by_intervening_arrival():
    actual, _ = run(model(lanes=1), ((.25, 0, 0),))
    assert [frame["start"] for frame in actual["frames"]] == [0., 1.]
    assert actual["jobs"][0]["start"] == 1.
    assert actual["scheduled_idle_lane_time_s"] == 1.
    assert actual["waiting_area_s"] == .75
    assert all(frame["decision_delay_s"] == 0 for frame in actual["frames"])


def test_same_released_prefix_is_invariant_to_later_arrival_suffix():
    common = ((0., 0, 0), (0., 1, 0), (.5, 0, 1))
    first, _ = run(model(), common + ((3.1, 0, 2),))
    second, _ = run(model(), common + ((3.2, 1, 1), (3.3, 0, 2)))
    prefix = lambda result: [row for row in identity(result) if row[0] < 3.1]
    assert prefix(first) == prefix(second)


def test_fifo_assigns_oldest_admitted_jobs_to_earliest_offsets_in_call():
    current = model(lanes=1, options=(
        FrameOption("batch", (2, 0), 10., ((8., 2.), ())),))
    actual, _ = run(current, ((0., 0, 0), (0., 0, 1)))
    assert [job["completion"] for job in actual["jobs"]] == [2., 8.]


def test_cutoff_retains_late_completion_inflight_and_queued_tail():
    current = model(lanes=1, options=(
        FrameOption("batch", (2, 0), 10., ((8., 2.), ())),))
    tape = ((0., 0, 0), (0., 0, 1), (0., 0, 2), (1., 0, 3))
    actual, _ = run(current, tape, horizon=2., cutoff=4.)
    snapshot = actual["snapshots"]["observation_end"]
    assert (snapshot["completed_jobs"], snapshot["inflight_jobs"],
            snapshot["queued_jobs"], snapshot["unfinished_jobs"]) == (1, 1, 2, 3)
    assert actual["jobs"][1]["completion"] == 8.
    assert actual["jobs"][2]["start"] is None
    assert snapshot["restricted_mean_flow_s"] == pytest.approx(13 / 4)
    assert actual["occupied_lane_time_s"] == 4.
    assert actual["waiting_area_s"] == 7.


def test_single_dispatch_calls_one_lane_and_updates_visible_queues_immediately():
    calls = []

    def recorder(options, queue, lanes, budget):
        calls.append((tuple(queue), lanes))
        return select_canonical_count_space_bnb_frame(options, queue, lanes, budget)

    actual, _ = run(model(), ((0., 0, 0), (0., 1, 0)),
                    rule="single", selector=recorder)
    assert calls == [((1, 1), 1), ((0, 1), 1)]
    assert [frame["start"] for frame in actual["frames"]] == [0., 0.]
    assert [frame["lane_ids"] for frame in actual["frames"]] == [(0,), (1,)]


def test_do_not_add_idle_calls_after_last_registered_job_is_admitted():
    actual, originals = run(model(), ((0., 0, 0),), rule="single")
    assert len(originals) == actual["frame_count"] == 1
    assert actual["frames"][0]["lane_ids"] == (0,)
    assert actual["scheduled_idle_lane_time_s"] == 0.


def test_lane_accounting_retains_barrier_and_clips_all_exposure_at_cutoff():
    tape = ((0., 0, 0), (0., 1, 0))
    actual, _ = run(model(), tape, rule="synchronous", horizon=1., cutoff=3.)
    assert actual["occupied_lane_time_s"] == 5.
    assert actual["barrier_idle_lane_time_s"] == 1.
    assert actual["scheduled_idle_lane_time_s"] == 0.
    assert sum(actual[key] for key in ("occupied_lane_time_s", "barrier_idle_lane_time_s",
                                      "scheduled_idle_lane_time_s")) == 6.


def test_invalid_executable_service_is_rejected_but_score_audit_is_deferred():
    def invalid_service(options, queue, lanes, budget):
        chosen = select_canonical_count_space_bnb_frame(options, queue, lanes, budget)
        return replace(chosen, service=(chosen.service[0] + 1, chosen.service[1]))

    with pytest.raises(ValueError, match="infeasible"):
        run(model(), ((0., 0, 0),), selector=invalid_service)

    returned = []

    def false_score(options, queue, lanes, budget):
        chosen = select_canonical_count_space_bnb_frame(options, queue, lanes, budget)
        chosen = replace(chosen, score=chosen.score + 1.)
        returned.append(chosen)
        return chosen

    actual, originals = run(model(), ((0., 0, 0),), selector=false_score)
    assert actual["snapshots"]["observation_end"]["completed_jobs"] == 1
    assert originals[0] is returned[0]
