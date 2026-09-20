from copy import deepcopy
from dataclasses import replace
import json

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import (
    exact_search_budget, simulate_frames,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    FrameModel, FrameOption, capacity_certificate,
)


def _model():
    return FrameModel(
        "fixture", "fixture", 2, ("a", "b"), (1.0, 1.0),
        (
            FrameOption("a", (1, 0), 2.0, ((2.0,), ())),
            FrameOption("b", (0, 1), 4.0, ((), (4.0,))),
            FrameOption("ab", (1, 1), 3.0, ((1.0,), (3.0,)), loaded=True),
        ),
    )


def _tape_and_frames():
    tape = ((0.0, 0, 0), (0.0, 0, 1), (0.0, 1, 0), (1.0, 0, 2), (4.0, 1, 1))
    frames = [
        {"start_s": 0.0, "duration_s": 4.0, "queue_before": (2, 1), "departures": (1, 1), "option_indices": (0, 1)},
        {"start_s": 4.0, "duration_s": 3.0, "queue_before": (2, 1), "departures": (1, 1), "option_indices": (2, None)},
        {"start_s": 7.0, "duration_s": 2.0, "queue_before": (1, 0), "departures": (1, 0), "option_indices": (0, None)},
    ]
    return tape, frames


def test_hand_computed_flow_components_classes_and_lane_seconds():
    tape, frames = _tape_and_frames()
    result = decompose_flow(_model(), tape, frames, 4.0)

    # Completed flows are a=(2,5,8), b=(4,3). The arrival at t=4 has zero barrier wait.
    assert result["completed_jobs"] == 5
    assert result["flow_area_components"] == {
        "deferred_visible": 7.0, "new_arrival_barrier": 3.0, "dispatch_to_completion": 12.0,
    }
    assert result["mean_flow_s"] == 22 / 5
    assert result["mean_deferred_visible_wait_s"] == 7 / 5
    assert result["mean_arrival_barrier_wait_s"] == 3 / 5
    assert result["mean_dispatch_to_completion_s"] == 12 / 5
    assert result["by_class"]["a"]["mean_flow_s"] == 5.0
    assert result["by_class"]["b"]["mean_flow_s"] == 3.5
    assert result["flow_identity_error_job_seconds"] == 0.0
    assert result["frame_duration_mean_s"] == 3.0
    assert result["frame_duration_time_weighted_mean_s"] == 29 / 9
    assert result["occupied_lane_seconds"] == 11.0
    assert result["global_barrier_idle_lane_seconds"] == 7.0
    assert result["unassigned_lane_seconds"] == 5.0
    assert result["completed_lane_barrier_seconds"] == 2.0
    assert result["total_lane_seconds"] == 18.0
    assert result["loaded_frame_count"] == 1
    assert result["time_regions"]["arrival_window"]["flow_area_job_seconds"] == 13.0
    assert result["time_regions"]["drain"]["flow_area_job_seconds"] == 9.0
    for component, area in result["flow_area_components"].items():
        assert sum(row["flow_area_components"][component] for row in result["by_action_multiset"]) == area
    swapped = [dict(frame, option_indices=tuple(reversed(frame["option_indices"]))) for frame in frames]
    assert decompose_flow(_model(), tape, swapped, 4.0)["by_action_multiset"] == result["by_action_multiset"]
    json.dumps(result)


def test_cross_horizon_intervals_split_wait_and_completion_offsets_exactly():
    model = _model()
    tape = ((0.0, 0, 0), (0.0, 0, 1), (1.0, 0, 2))
    frames = [
        {"start_s": 0.0, "duration_s": 2.0, "queue_before": (2, 0), "departures": (1, 0), "option_indices": (0, None)},
        {"start_s": 2.0, "duration_s": 2.0, "queue_before": (2, 0), "departures": (2, 0), "option_indices": (0, 0)},
    ]
    result = decompose_flow(model, tape, frames, 1.5)

    assert result["flow_area_components"] == {"deferred_visible": 2.0, "new_arrival_barrier": 1.0, "dispatch_to_completion": 6.0}
    assert result["time_regions"]["arrival_window"]["flow_area_components"] == {
        "deferred_visible": 1.5, "new_arrival_barrier": 0.5, "dispatch_to_completion": 1.5,
    }
    assert result["time_regions"]["drain"]["flow_area_components"] == {
        "deferred_visible": 0.5, "new_arrival_barrier": 0.5, "dispatch_to_completion": 4.5,
    }
    assert result["mean_flow_s"] == 3.0


def test_selector_callback_trace_reproduces_existing_runner_mean_flow():
    base = _model()
    model = replace(base, options=(*base.options[:2], replace(base.options[2], duration=4.0, completion_offsets=((2.0,), (4.0,)))))
    tape, _ = _tape_and_frames()
    frames = []
    now = 0.0

    def recorded_selector(options, queues, lanes, budget):
        nonlocal now
        selected = select_budgeted_frame(options, queues, lanes, budget)
        frames.append({
            "start_s": now, "duration_s": selected.duration, "queue_before": queues,
            "departures": selected.service, "option_indices": selected.option_indices,
        })
        now += selected.duration
        return selected

    run = simulate_frames(
        model, tape, 4.0, capacity_certificate(model, 0.5),
        exact_search_budget(len(model.options), model.lanes), selector=recorded_selector,
    )
    result = decompose_flow(model, tape, frames, 4.0)
    assert result["mean_flow_s"] == pytest.approx(run["mean_flow_s"])
    assert result["completed_jobs"] == run["completed_jobs"]
    assert result["elapsed_including_drain_s"] == run["elapsed_including_drain_s"]
    assert result["flow_area_job_seconds"] / result["elapsed_including_drain_s"] == pytest.approx(run["time_average_unfinished_jobs"])


@pytest.mark.parametrize("change, message", (
    ("queue", "queue_before"), ("departure", "departures"),
    ("duration", "synchronized action duration"), ("truncated", "drained"),
))
def test_trace_drift_is_rejected(change, message):
    tape, frames = _tape_and_frames()
    frames = deepcopy(frames)
    if change == "queue":
        frames[0]["queue_before"] = (1, 1)
    elif change == "departure":
        frames[0]["departures"] = (2, 1)
    elif change == "duration":
        frames[-1]["duration_s"] = 3.0
    else:
        frames.pop()
    with pytest.raises(ValueError, match=message):
        decompose_flow(_model(), tape, frames, 4.0)
