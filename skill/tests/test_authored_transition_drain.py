"""Small execution contracts for the authored transition drain helper."""
from __future__ import annotations

from algorithm.authored_transition_drain import (
    authored_arrival_times,
    drain_initial_population,
    fixed_frame_action,
)
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import (
    FrameModel,
    FrameOption,
    load_model,
)
import pytest


def test_authored_profiles_encode_coincident_and_one_second_fifo_ages():
    queues = (3, 0, 2)

    assert authored_arrival_times(queues, "coincident") == (
        (0.0, 0.0, 0.0),
        (),
        (0.0, 0.0),
    )
    assert authored_arrival_times(queues, "unit_fifo") == (
        (-2.0, -1.0, 0.0),
        (),
        (-1.0, 0.0),
    )


def _flows(result):
    return sorted(row["flow_s"] for row in result["completion_records"])


def _record(result, coordinate, job_index):
    return next(
        row
        for row in result["completion_records"]
        if row["coordinate"] == coordinate and row["job_index"] == job_index
    )


def test_real_raw_lane_order_changes_fifo_p90_but_not_aggregate_physics():
    model = load_model()
    queues = (1, 0, 4)
    forward_action = fixed_frame_action(model, queues, (0, 7))
    reversed_action = fixed_frame_action(model, queues, (7, 0))

    def forbidden_continuation(*args, **kwargs):
        pytest.fail("the two-option first action must drain this fixture")

    results = {}
    for profile in ("coincident", "unit_fifo"):
        arrivals = authored_arrival_times(queues, profile)
        results[(profile, "forward")] = drain_initial_population(
            model,
            arrivals,
            26,
            forward_action,
            continuation=forbidden_continuation,
        )
        results[(profile, "reversed")] = drain_initial_population(
            model,
            arrivals,
            26,
            reversed_action,
            continuation=forbidden_continuation,
        )

    assert tuple(sorted(forward_action.option_indices)) == tuple(
        sorted(reversed_action.option_indices)
    ) == (0, 7)
    assert forward_action.service == reversed_action.service == queues
    assert forward_action.duration == reversed_action.duration
    assert forward_action.score == reversed_action.score

    coincident_forward = results[("coincident", "forward")]
    coincident_reversed = results[("coincident", "reversed")]
    assert _flows(coincident_forward) == _flows(coincident_reversed)
    assert coincident_forward["mean_flow_s"] == coincident_reversed["mean_flow_s"]
    assert coincident_forward["p90_flow_s"] == coincident_reversed["p90_flow_s"]

    fifo_forward = results[("unit_fifo", "forward")]
    fifo_reversed = results[("unit_fifo", "reversed")]
    assert fifo_forward["mean_flow_s"] == pytest.approx(
        fifo_reversed["mean_flow_s"]
    )
    assert fifo_forward["mean_flow_s"] == pytest.approx(11.26660908087538)
    assert fifo_forward["makespan_s"] == fifo_reversed["makespan_s"]
    assert fifo_forward["makespan_s"] == pytest.approx(11.514578111707017)
    assert fifo_forward["p90_flow_s"] == pytest.approx(14.514578111707017)
    assert fifo_reversed["p90_flow_s"] == pytest.approx(12.462480184740949)
    assert fifo_forward["flow_area_components_job_s"] == pytest.approx({
        "initial_age": 6.0,
        "deferred_visible": 0.0,
        "dispatch_to_completion": 50.333045404376904,
    })
    assert fifo_forward["flow_area_job_s"] == pytest.approx(56.333045404376904)
    assert _record(fifo_forward, 2, 0)["option_index"] == 0
    assert _record(fifo_reversed, 2, 0)["option_index"] == 7
    assert _record(fifo_forward, 2, 3)["option_index"] == 7
    assert _record(fifo_reversed, 2, 3)["option_index"] == 0

    for result in results.values():
        assert result["job_count"] == result["completed_jobs"] == 5
        assert result["frame_count"] == 1
        assert result["continuation_selector_calls"] == 0
        assert result["flow_area_identity_error_job_s"] == pytest.approx(0.0)
        assert result["flow_area_job_s"] == pytest.approx(
            sum(result["flow_area_components_job_s"].values())
        )
        assert all(value == 0 for value in result["selection_work"].values())


def _toy_model():
    return FrameModel(
        node="toy",
        scenario_id="toy",
        lanes=1,
        workloads=("jobs",),
        scales=(1.0,),
        options=(
            FrameOption("single", (1,), 1.0, ((1.0,),)),
            FrameOption("batch", (2,), 1.5, ((0.5, 1.5),)),
        ),
    )


def test_both_first_actions_use_the_same_v4_continuation_and_close_work_area():
    model = _toy_model()
    queues = (3,)
    arrivals = authored_arrival_times(queues, "unit_fifo")
    first_actions = (
        fixed_frame_action(model, queues, (0,)),
        fixed_frame_action(model, queues, (1,)),
    )
    calls = []

    def common_continuation(options, visible, lanes, budget):
        selected = select_budgeted_frame(options, visible, lanes, budget)
        calls.append((tuple(visible), selected))
        return selected

    branches = []
    call_ranges = []
    for first_action in first_actions:
        start = len(calls)
        result = drain_initial_population(
            model,
            arrivals,
            3,
            first_action,
            continuation=common_continuation,
        )
        branches.append(result)
        call_ranges.append(calls[start:])

    assert [row[0] for row in call_ranges[0]] == [(2,)]
    assert [row[0] for row in call_ranges[1]] == [(1,)]
    for result, first_action, branch_calls in zip(
        branches, first_actions, call_ranges, strict=True
    ):
        assert result["frame_trace"][0]["option_indices"] == (
            first_action.option_indices
        )
        assert result["frame_trace"][0]["queue_after"] == branch_calls[0][0]
        assert result["frame_trace"][1]["option_indices"] == (
            branch_calls[0][1].option_indices
        )
        assert result["continuation_selector_calls"] == len(branch_calls) == 1
        assert result["frame_count"] == 2
        assert result["job_count"] == result["completed_jobs"] == 3
        assert all(
            tuple(after) == tuple(q - d for q, d in zip(before, departures))
            for before, departures, after in (
                (
                    frame["queue_before"],
                    frame["departures"],
                    frame["queue_after"],
                )
                for frame in result["frame_trace"]
            )
        )
        assert result["flow_area_job_s"] == pytest.approx(
            sum(result["flow_area_components_job_s"].values())
        )
        assert result["flow_area_job_s"] == pytest.approx(
            sum(row["flow_s"] for row in result["completion_records"])
        )
        assert result["flow_area_identity_error_job_s"] == pytest.approx(0.0)
        for field, value in result["selection_work"].items():
            assert value == sum(
                getattr(selected, field, 0) for _, selected in branch_calls
            )


def test_invalid_profiles_fixed_actions_and_drain_inputs_are_rejected():
    model = _toy_model()

    with pytest.raises(ValueError, match="nonnegative integer"):
        authored_arrival_times((1.5,), "coincident")
    with pytest.raises(ValueError, match="nonnegative integer"):
        authored_arrival_times((-1,), "coincident")
    with pytest.raises(ValueError, match="profile must be"):
        authored_arrival_times((1,), "unknown")
    with pytest.raises(ValueError, match="lane count"):
        fixed_frame_action(model, (1,), (0, None))
    with pytest.raises(ValueError, match="unknown option"):
        fixed_frame_action(model, (1,), (2,))
    with pytest.raises(ValueError, match="exceeds the visible queue"):
        fixed_frame_action(model, (0,), (0,))
    with pytest.raises(ValueError, match="coordinates"):
        fixed_frame_action(model, (1, 0), (0,))
    with pytest.raises(ValueError, match="nonnegative integer"):
        fixed_frame_action(model, (1.5,), (0,))

    action = fixed_frame_action(model, (1,), (0,))
    with pytest.raises(ValueError, match="coordinates"):
        drain_initial_population(model, ((0.0,), ()), 3, action)
    with pytest.raises(ValueError, match="finite and nonpositive"):
        drain_initial_population(model, ((0.1,),), 3, action)
    with pytest.raises(ValueError, match="FIFO order"):
        drain_initial_population(model, ((0.0, -1.0),), 3, action)
    with pytest.raises(ValueError, match="nonempty queue"):
        drain_initial_population(model, ((),), 3, action)

    idle = fixed_frame_action(model, (1,), (None,))
    with pytest.raises(ValueError, match="selected no service"):
        drain_initial_population(model, ((0.0,),), 3, idle)
