"""Exercise gated dispatch through the frozen simulator and flow accounting."""
from copy import deepcopy

import pytest

from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_backlog_gate import BacklogGateRecorder, _assert_control_reproduced
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates


@pytest.mark.parametrize("burst", (False, True))
def test_trace_draining_and_prediction_accounting(burst):
    model = load_model()
    certificate = capacity_certificate(model, 0.8)
    if burst:
        tape = [(0.0, coordinate, index) for coordinate, count in enumerate((8, 8, 16))
                for index in range(count)] + [(0.5, 2, 16)]
    else:
        tape = [(0.0, 0, 0), (0.5, 2, 0)]
    recorder = BacklogGateRecorder(model.options, model.lanes, certificate["arrival_rates"],
                                  terminal_service_rates(model, certificate))
    metrics = simulate_frames(model, tape, 2.0, certificate, 26, selector=recorder)
    flow = decompose_flow(model, tape, recorder.frames, 2.0)
    accounting = recorder.accounting()
    assert metrics["completed_jobs"] == len(tape)
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert metrics["maximum_expected_drift_bound_violation"] <= 1e-8
    assert flow["mean_flow_s"] == pytest.approx(metrics["mean_flow_s"])
    assert accounting["gate_setup_requirement_reads"] == 54
    assert accounting["gate_queue_coordinate_evaluations"] == 3 * len(recorder.frames)
    expected_active = [frame for frame in recorder.frames
                       if sum(w * q for w, q in zip((8, 8, 3), frame["queue_before"])) > 48]
    assert accounting["lookahead_active_frames"] == len(expected_active)
    assert accounting["lookahead_active_duration_s"] == sum(frame["duration_s"] for frame in expected_active)
    assert accounting["v4_inactive_frames"] > 0
    assert accounting["maximum_relative_support_loss"] <= 0.01 + 1e-12
    assert accounting["next_queue_coordinate_evaluations"] == accounting["terminal_value_coordinate_evaluations"]
    assert accounting["next_queue_coordinate_evaluations"] == 3 * accounting["cost_candidate_evaluations"]
    if burst:
        assert accounting["lookahead_active_frames"] > 0
        assert accounting["cost_candidate_evaluations"] > 0
        assert accounting["minimum_active_seed_support_margin"] >= 0
        assert accounting["minimum_active_support_floor_margin"] >= 0
    else:
        assert accounting["lookahead_active_frames"] == 0
        assert accounting["cost_candidate_evaluations"] == accounting["completion_offset_reads"] == 0
        assert accounting["minimum_active_seed_support_margin"] is None
        assert accounting["minimum_active_support_floor_margin"] is None


def test_control_reproduction_rejects_changed_results_and_work_but_not_clock_noise():
    old = {"load": 0.8, "seed": 41, "budget": 26, "policy": "lookahead_cap_1pct",
           "metrics": {"mean_flow_s": 3.0}, "decomposition": {"mean_flow_s": 3.0},
           "selector_accounting": {"selector_wall_s": 1.0, "selector_mean_wall_ms": 0.2,
                                   "cost_candidate_evaluations": 18}}
    row = deepcopy(old)
    row["selector_accounting"]["selector_wall_s"] = 2.0
    row["selector_accounting"]["selector_mean_wall_ms"] = 0.4
    _assert_control_reproduced(row, old)
    for section, field in (("metrics", "mean_flow_s"), ("decomposition", "mean_flow_s"),
                           ("selector_accounting", "cost_candidate_evaluations")):
        changed = deepcopy(row)
        changed[section][field] += 1
        with pytest.raises(ValueError, match="frozen"):
            _assert_control_reproduced(changed, old)
