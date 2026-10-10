"""Observed births are cumulative, causal and counted once per released job."""
from copy import deepcopy

import pytest

from algorithm.arrival_aware_lane_rollout import ArrivalAwareLaneRollout
from algorithm.experiments.age_contextual_lane_refill_replay import simulate_age_contextual_trajectory
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.visible_queue_clearing_rollout import select_context_maxweight


def _model(lanes=2, options=None):
    return FrameModel("toy", "observed_context", lanes, ("a", "b"), (1., 1.), options or (
        FrameOption("short_a", (1, 0), 2., ((2.,), ())),
        FrameOption("long_b", (0, 1), 3., ((), (3.,))),
    ))


def _run(model, tape, selector, horizon=6., cutoff=15.):
    return simulate_observed_rate_contextual_trajectory(model, tape, horizon, cutoff, selector, policy="test")


def _ignore_history(options, queue, available, now, lane, ages, births):
    return select_context_maxweight(options, queue, available, now, lane)


@pytest.mark.parametrize("lanes", (2, 4))
def test_history_ignored_maxweight_reproduces_complete_age_engine_output(lanes):
    model = _model(lanes)
    tape = ((0., 0, 0), (0., 1, 0), (0., 0, 1),
            (.5, 0, 2), (2.5, 1, 1), (4., 0, 3))
    core = ArrivalAwareLaneRollout(model.options, (.1, .1), 6., "no-planning-in-test")
    original = simulate_age_contextual_trajectory(model, tape, 6., 15.,
        lambda options, queue, available, now, lane, ages:
        core.maxweight(options, queue, available, now, lane), policy="test")
    actual = _run(model, tape, lambda options, queue, available, now, lane, ages, births:
                  core.maxweight(options, queue, available, now, lane))
    normalized = deepcopy(actual)
    assert normalized["rollout_work"].pop("observed_arrival_counter_updates") == len(tape)
    assert normalized["rollout_work"].pop("observed_arrival_context_coordinate_reads") == len(model.workloads) * actual["frame_count"]
    for frame in normalized["frames"]:
        frame.pop("observed_arrival_counts")
    assert normalized == original


def test_same_timestamp_calls_do_not_repeat_birth_updates_and_history_keeps_admitted():
    result = _run(_model(), ((0., 0, 0), (0., 0, 1), (0., 1, 0),
                             (.5, 0, 2), (1.25, 1, 1)), _ignore_history)
    first, second, third = result["frames"][:3]
    assert first["start"] == second["start"] == 0.
    assert first["observed_arrival_counts"] == second["observed_arrival_counts"] == (2, 1)
    assert first["queue"] == (2, 1) and second["queue"] == (1, 1)
    assert third["start"] == 2. and third["queue"] == (1, 2)
    assert third["observed_arrival_counts"] == (3, 2)
    assert result["rollout_work"]["observed_arrival_counter_updates"] == 5


def test_idle_exposure_advances_without_new_births_and_already_served_counts_remain():
    result = _run(_model(lanes=1), ((0., 0, 0), (5.2, 1, 0)), _ignore_history)
    idle = [frame for frame in result["frames"] if frame["option_indices"] == (None,)]
    assert [frame["start"] for frame in idle] == [2., 3., 4., 5.]
    assert all(frame["queue"] == (0, 0) for frame in idle)
    assert all(frame["observed_arrival_counts"] == (1, 0) for frame in idle)
    assert result["frames"][-1]["observed_arrival_counts"] == (1, 1)
    assert result["frames"][-1]["start"] == 6.
    assert result["rollout_work"]["observed_arrival_counter_updates"] == 2


def test_unreleased_actual_futures_leave_all_seven_argument_contexts_unchanged():
    common = ((0., 0, 0), (0., 1, 0), (.5, 0, 1))

    def contexts(tape):
        calls = []

        def recorder(options, queue, available, now, lane, ages, births):
            calls.append((queue, available, now, lane, ages, births))
            return _ignore_history(options, queue, available, now, lane, ages, births)

        _run(_model(), tape, recorder)
        return [call for call in calls if call[2] < 3.1]

    assert contexts(common + ((3.1, 0, 2),)) == contexts(common + ((3.2, 1, 1), (3.3, 0, 2)))


def test_loaded_early_completion_preserves_complete_commitment_and_releases_births_once():
    model = _model(lanes=1, options=(
        FrameOption("loaded", (1, 1), 5., ((1.,), (5.,)), loaded=True),))
    result = _run(model, ((0., 0, 0), (0., 1, 0),
                          (.5, 0, 1), (.5, 1, 1)), _ignore_history)
    assert result["jobs"][0]["completion"] == 1.
    assert result["jobs"][2]["start"] == 5.
    assert [frame["start"] for frame in result["frames"]] == [0., 5.]
    assert [frame["observed_arrival_counts"] for frame in result["frames"]] == [(1, 1), (2, 2)]
    assert result["frames"][1]["waiting_ages"] == ((4.5,), (4.5,))
    assert result["lane_events"][0]["reserved_end"] == 5.
    assert result["rollout_work"]["observed_arrival_counter_updates"] == 4
