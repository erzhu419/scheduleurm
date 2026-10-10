"""Waiting ages preserve the existing causal FIFO and lane-reservation clock."""
from copy import deepcopy

import pytest

from algorithm.arrival_aware_lane_rollout import ArrivalAwareLaneRollout
from algorithm.experiments.age_contextual_lane_refill_replay import simulate_age_contextual_trajectory
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory
from algorithm.visible_queue_clearing_rollout import select_context_maxweight


def _model(lanes=2, options=None):
    return FrameModel("toy", "age_context", lanes, ("a", "b"), (1., 1.), options or (
        FrameOption("short_a", (1, 0), 2., ((2.,), ())),
        FrameOption("long_b", (0, 1), 3., ((), (3.,))),
    ))


def _run(model, tape, selector, horizon=6., cutoff=15.):
    return simulate_age_contextual_trajectory(model, tape, horizon, cutoff, selector, policy="test")


def _ignore_age(options, queue, available, now, lane, ages):
    return select_context_maxweight(options, queue, available, now, lane)


@pytest.mark.parametrize("lanes", (2, 4))
def test_age_ignored_maxweight_reproduces_every_existing_job_frame_and_fee(lanes):
    model = _model(lanes)
    tape = ((0., 0, 0), (0., 1, 0), (0., 0, 1),
            (.5, 0, 2), (2.5, 1, 1), (4., 0, 3))
    core = ArrivalAwareLaneRollout(model.options, (.1, .1), 6., "no-planning-in-test")
    original = simulate_contextual_trajectory(model, tape, 6., 15., core.maxweight, policy="test")
    actual = _run(model, tape, lambda options, queue, available, now, lane, ages:
                  core.maxweight(options, queue, available, now, lane))
    normalized = deepcopy(actual)
    ages_read = normalized["rollout_work"].pop("age_context_job_reads")
    assert ages_read == sum(sum(frame["queue"]) for frame in actual["frames"])
    for frame in normalized["frames"]:
        frame.pop("waiting_ages")
    assert normalized == original


def test_fifo_ages_exclude_jobs_admitted_on_an_earlier_lane_at_same_time():
    calls = []

    def recorder(options, queue, available, now, lane, ages):
        calls.append((now, lane, queue, ages))
        return _ignore_age(options, queue, available, now, lane, ages)

    result = _run(_model(), ((0., 0, 0), (0., 0, 1), (0., 1, 0),
                             (.5, 0, 2), (1.25, 1, 1)), recorder)
    assert calls[0] == (0., 0, (2, 1), ((0., 0.), (0.,)))
    assert calls[1] == (0., 1, (1, 1), ((0.,), (0.,)))
    assert calls[2] == (2., 0, (1, 2), ((1.5,), (2., .75)))
    assert [frame["waiting_ages"] for frame in result["frames"]] == [call[3] for call in calls]
    for _, _, queue, ages in calls:
        assert tuple(map(len, ages)) == queue


def test_early_loaded_completion_does_not_release_lane_or_waiting_age_context():
    model = _model(lanes=1, options=(
        FrameOption("loaded", (1, 1), 5., ((1.,), (5.,)), loaded=True),))
    result = _run(model, ((0., 0, 0), (0., 1, 0),
                          (.5, 0, 1), (.5, 1, 1)), _ignore_age)
    assert result["jobs"][0]["completion"] == 1.
    assert result["jobs"][2]["start"] == 5.
    assert [frame["start"] for frame in result["frames"]] == [0., 5.]
    assert result["frames"][1]["waiting_ages"] == ((4.5,), (4.5,))
    assert result["lane_events"][0]["reserved_end"] == 5.


def test_selector_cannot_distinguish_actual_future_until_release():
    common = ((0., 0, 0), (0., 1, 0), (.5, 0, 1))

    def contexts(tape):
        calls = []

        def recorder(options, queue, available, now, lane, ages):
            calls.append((queue, available, now, lane, ages))
            return _ignore_age(options, queue, available, now, lane, ages)

        _run(_model(), tape, recorder)
        return [call for call in calls if call[2] < 3.1]

    assert contexts(common + ((3.1, 0, 2),)) == contexts(common + ((3.2, 1, 1), (3.3, 0, 2)))


def test_idle_polling_ages_and_unfinished_waiting_tail_keep_original_clock():
    idle = _run(_model(lanes=1), ((.25, 0, 0),), _ignore_age)
    assert [frame["start"] for frame in idle["frames"]] == [0., 1.]
    assert [frame["waiting_ages"] for frame in idle["frames"]] == [((), ()), ((.75,), ())]
    assert idle["rollout_work"]["age_context_job_reads"] == 1
    assert idle["waiting_area_s"] == .75
    model = _model(lanes=1, options=(FrameOption("batch", (2, 0), 10., ((8., 2.), ())),))
    result = _run(model, ((0., 0, 0), (0., 0, 1), (0., 0, 2), (1., 0, 3)),
                  _ignore_age, horizon=2., cutoff=4.)
    assert result["jobs"][2]["start"] is None
    assert result["snapshots"]["observation_end"]["unfinished_jobs"] == 3
    assert result["waiting_area_s"] == 7.
    assert result["occupied_lane_time_s"] == 4.
