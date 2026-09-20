from types import SimpleNamespace

import pytest

from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import LookaheadRecorder, terminal_service_rates


def test_terminal_rates_use_feasible_time_allocation_instead_of_arrival_direction():
    model = SimpleNamespace(lanes=2, workloads=("a", "b"), options=(
        SimpleNamespace(requirements=(1, 0), duration=2.0),
        SimpleNamespace(requirements=(0, 2), duration=4.0),
    ))
    certificate = {"boundary_time_weights": (0.25, 0.75), "arrival_direction": (9.0, 9.0)}
    assert terminal_service_rates(model, certificate) == pytest.approx((0.25, 0.75))


def test_lookahead_trace_executes_same_departures_and_accounts_for_both_prediction_weights():
    model = load_model()
    certificate = capacity_certificate(model, 0.8)
    rates = terminal_service_rates(model, certificate)
    # Explicit arrivals exercise class queues, in-frame releases and draining;
    # no new experimental arrival seed is consumed by this integration test.
    tape = [(0.0, 0, 0), (0.0, 1, 0), (0.0, 2, 0),
            (0.5, 0, 1), (1.0, 2, 1), (4.0, 1, 1)]
    for weight in (0.0, 1.0):
        recorder = LookaheadRecorder(certificate["arrival_rates"], rates, weight)
        metrics = simulate_frames(model, tape, 5.0, certificate, 26, selector=recorder)
        flow = decompose_flow(model, tape, recorder.frames, 5.0)
        assert metrics["completed_jobs"] == len(tape)
        assert metrics["maximum_queue_recursion_violation"] == 0
        assert metrics["mean_flow_s"] == pytest.approx(flow["mean_flow_s"])
        accounting = recorder.accounting()
        assert accounting["minimum_seed_support_margin"] >= 0
        assert accounting["minimum_support_floor_margin"] >= 0
        assert accounting["maximum_relative_support_loss"] <= 0.01 + 1e-12
        assert accounting["maximum_objective_excess_over_support_incumbent"] <= 1e-8
        assert accounting["next_queue_coordinate_evaluations"] == 3 * accounting["cost_candidate_evaluations"]
        assert accounting["terminal_value_coordinate_evaluations"] == accounting["next_queue_coordinate_evaluations"]
