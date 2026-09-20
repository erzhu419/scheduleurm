import pytest

from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_batch_rollout import AllocatedLookaheadRecorder, BatchRolloutRecorder
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import LookaheadRecorder, terminal_service_rates


@pytest.mark.parametrize("budget", (10, 26))
def test_executed_flow_and_total_budget_include_successor_work(budget):
    model = load_model()
    certificate = capacity_certificate(model, 0.5)
    tape = [(0.0, coordinate, index) for coordinate, count in enumerate((1, 4, 8)) for index in range(count)]
    tape += [(0.5, 2, 8), (1.0, 0, 1)]
    recorder = BatchRolloutRecorder(certificate["arrival_rates"], terminal_service_rates(model, certificate))
    metrics = simulate_frames(model, tape, 2.0, certificate, budget, selector=recorder)
    parts = decompose_flow(model, tape, recorder.frames, 2.0)
    accounting = recorder.accounting()
    assert metrics["completed_jobs"] == len(tape)
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert metrics["maximum_expected_drift_bound_violation"] <= 1e-8
    assert metrics["maximum_candidate_attempts_per_frame"] <= budget
    assert metrics["mean_flow_s"] == pytest.approx(parts["mean_flow_s"])
    assert accounting["root_attempts"] + accounting["rollout_attempts"] == metrics["candidate_attempts"]
    assert accounting["maximum_budget_identity_error"] == 0
    assert accounting["rollout_attempts"] == 4 * accounting["rollout_frames"]
    assert accounting["rollout_cost_candidate_evaluations"] == 2 * accounting["rollout_frames"]
    assert accounting["rollout_accepted_frames"] + accounting["rollout_vetoed_frames"] == accounting["rollout_frames"]
    assert accounting["minimum_seed_support_margin"] >= 0
    assert accounting["minimum_support_floor_margin"] >= 0
    assert accounting["maximum_relative_support_loss"] <= .01 + 1e-12
    assert accounting["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8
    if budget == 10:
        assert accounting["rollout_attempts"] == 0


def test_allocation_control_passes_identical_reduced_budget_to_frozen_objective(monkeypatch):
    calls = []

    def record_budget(self, options, queues, lanes, budget):
        calls.append(budget)

    monkeypatch.setattr(LookaheadRecorder, "__call__", record_budget)
    recorder = AllocatedLookaheadRecorder((0.1,), (0.2,), 1.0)
    for budget in (10, 14, 26, 120):
        recorder((None,) * 9, (1,), 2, budget)
    assert calls == [10, 10, 22, 116]
