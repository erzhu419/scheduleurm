import pytest

from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_arrival_rollout import ArrivalRolloutRecorder
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_recycled_rollout import RecycledRolloutRecorder


@pytest.mark.parametrize("budget", (14, 26))
def test_arrival_proxy_work_and_executed_poisson_interface_remain_separate(budget):
    model = load_model()
    certificate = capacity_certificate(model, 0.5)
    rates = certificate["arrival_rates"]
    mu = terminal_service_rates(model, certificate)
    tape = [(0.0, coordinate, index) for coordinate, count in enumerate((1, 4, 8)) for index in range(count)]
    tape += [(0.5, 2, 8), (1.0, 0, 1)]
    recorder = ArrivalRolloutRecorder(rates, mu)
    metrics = simulate_frames(model, tape, 2.0, certificate, budget, selector=recorder)
    parts = decompose_flow(model, tape, recorder.frames, 2.0)
    accounting = recorder.accounting()
    assert metrics["completed_jobs"] == metrics["job_count"] == len(tape)
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert metrics["maximum_expected_drift_bound_violation"] <= 1e-8
    assert metrics["maximum_candidate_attempts_per_frame"] <= budget
    assert metrics["mean_flow_s"] == pytest.approx(parts["mean_flow_s"])
    assert accounting["root_attempts"] + accounting["rollout_attempts"] == metrics["candidate_attempts"]
    assert accounting["initial_root_attempts"] + accounting["recycled_candidate_attempts"] == accounting["root_attempts"]
    assert accounting["initial_cost_candidate_evaluations"] + accounting["recycled_cost_candidate_evaluations"] == accounting["cost_candidate_evaluations"]
    assert accounting["maximum_budget_identity_error"] == accounting["maximum_stage_identity_error"] == 0
    assert accounting["rollout_attempts"] == 4 * accounting["rollout_frames"]
    assert accounting["rollout_cost_candidate_evaluations"] == 2 * accounting["rollout_frames"]
    assert accounting["minimum_seed_support_margin"] >= 0
    assert accounting["minimum_support_floor_margin"] >= 0
    assert accounting["maximum_relative_support_loss"] <= .01 + 1e-12
    assert accounting["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8
    assert accounting["maximum_path_population_error"] == 0
    if budget == 14:
        assert accounting["current_search_resumed_frames"] > 0
        assert accounting["rollout_attempts"] == accounting["rollout_arrival_event_generations"] == 0
        old = RecycledRolloutRecorder(rates, mu)
        reference = simulate_frames(model, tape, 2.0, certificate, budget, selector=old)
        assert recorder.frames == old.frames
        assert metrics == reference
    else:
        assert accounting["rollout_frames"] > 0
        assert accounting["rollout_arrival_event_generations"] > 0
        assert accounting["rollout_arrival_event_reads"] >= 2 * accounting["rollout_arrival_event_generations"]
        assert accounting["rollout_current_completion_offset_reads"] > 0
        assert accounting["rollout_tail_segments"] > 0
