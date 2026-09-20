import pytest

from algorithm.experiments.aligned_hardware_mixed_successor_rollout import (
    CONTROLS, MIXED_COUNTERS, NEW_POLICY, MixedSuccessorRolloutRecorder,
    summarize_mixed_runs,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model, capacity_certificate
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_waiting_cost import COMPONENTS


@pytest.mark.parametrize("budget", (14, 26))
def test_recorder_charges_mixed_work_inside_original_candidate_budget(budget):
    model = load_model()
    certificate = capacity_certificate(model, .5)
    tape = [(0., coordinate, index) for coordinate, count in enumerate((1, 4, 8))
            for index in range(count)] + [(.5, 2, 8), (1., 0, 1)]
    recorder = MixedSuccessorRolloutRecorder(
        certificate["arrival_rates"], terminal_service_rates(model, certificate), 1)
    metrics = simulate_frames(model, tape, 2., certificate, budget, selector=recorder)
    parts = decompose_flow(model, tape, recorder.frames, 2.)
    counts = recorder.accounting()
    assert metrics["completed_jobs"] == len(tape)
    assert metrics["mean_flow_s"] == pytest.approx(parts["mean_flow_s"])
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert metrics["maximum_expected_drift_bound_violation"] <= 1e-8
    assert metrics["maximum_candidate_attempts_per_frame"] <= budget
    assert counts["root_attempts"] + counts["rollout_attempts"] == metrics["candidate_attempts"]
    assert counts["rollout_attempts"] == 4 * counts["rollout_frames"]
    assert counts["rollout_mixed_priority_coordinate_evaluations"] == counts["rollout_class_evaluations"]
    assert counts["rollout_mixed_profile_probes"] == counts["rollout_profile_probes"]
    assert counts["rollout_temporary_queue_updates"] <= counts["rollout_queue_coordinate_evaluations"]
    assert counts["rollout_temporary_queue_updates"] <= counts["rollout_lane_fill_evaluations"] <= 8 * counts["rollout_frames"]
    assert counts["rollout_random_draws"] == counts["rollout_arrival_event_generations"] + 3 * counts["rollout_frames"]
    assert counts["minimum_seed_support_margin"] >= 0
    assert counts["minimum_support_floor_margin"] >= 0
    assert counts["maximum_relative_support_loss"] <= .01 + 1e-12
    assert counts["maximum_budget_identity_error"] == counts["maximum_stage_identity_error"] == 0
    if budget == 14:
        assert counts["rollout_frames"] == counts["rollout_lane_fill_evaluations"] == 0
    else:
        assert counts["rollout_frames"] > 0


def test_summary_pairs_stochastic_control_before_hierarchical_average():
    accounting_keys = ("selector_mean_wall_ms", "rollout_frames", "rollout_vetoed_frames",
                       "rollout_random_draws", *MIXED_COUNTERS)

    def row(seed, policy, flow, planning_seed=None):
        return {"load": .5, "seed": seed, "budget": 26, "policy": policy,
                "planning_seed": planning_seed,
                "metrics": {"mean_flow_s": flow, "p90_flow_s": 2 * flow, "job_count": 10},
                "decomposition": {key: flow / 3 for key in COMPONENTS},
                "selector_accounting": dict.fromkeys(accounting_keys, 0)}

    runs = []
    for seed, ratios in ((41, (.5, 1., 1.8)), (42, (.7, .8, .9)), (43, (1., 1.1, 1.2))):
        runs.extend(row(seed, policy, 10.) for policy in CONTROLS[:2])
        for planning_seed, ratio in enumerate(ratios):
            old = 10 * 2 ** planning_seed
            runs.append(row(seed, CONTROLS[-1], old, planning_seed))
            runs.append(row(seed, NEW_POLICY, old * ratio, planning_seed))
    pairs, summaries = summarize_mixed_runs(runs, (.5,), (26,))
    ratio_key = f"flow_ratio_over_{CONTROLS[-1]}"
    assert [pair[ratio_key] for pair in pairs] == pytest.approx(
        (.5, 1., 1.8, .7, .8, .9, 1., 1.1, 1.2))
    assert summaries[0][f"mean_paired_{ratio_key}"] == pytest.approx(1.)
    assert [row[f"mean_{ratio_key}"] for row in summaries[0]["by_arrival_seed"]] == pytest.approx((1.1, .8, 1.1))
    with pytest.raises(ValueError, match="every planning repetition"):
        summarize_mixed_runs(runs[:-1], (.5,), (26,))
