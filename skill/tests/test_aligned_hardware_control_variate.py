from copy import deepcopy

import pytest

from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_arrival_rollout import ARRIVAL_COUNTERS, ArrivalRolloutRecorder
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_control_variate import (
    CONTROLS, NEW_POLICY, ControlVariateRolloutRecorder, summarize_control_variate_runs, _assert_control_reproduced,
)


@pytest.mark.parametrize("budget", (14, 26))
def test_executed_flow_and_rng_costs_remain_in_the_original_budget(budget):
    model = load_model()
    certificate = capacity_certificate(model, 0.5)
    rates = certificate["arrival_rates"]
    mu = terminal_service_rates(model, certificate)
    tape = [(0.0, coordinate, index) for coordinate, count in enumerate((1, 4, 8)) for index in range(count)]
    tape += [(0.5, 2, 8), (1.0, 0, 1)]
    recorder = ControlVariateRolloutRecorder(rates, mu, 0)
    initial_states = [rng.getstate() for rng in recorder.arrival_rngs]
    metrics = simulate_frames(model, tape, 2.0, certificate, budget, selector=recorder)
    parts = decompose_flow(model, tape, recorder.frames, 2.0)
    accounting = recorder.accounting()
    assert metrics["completed_jobs"] == metrics["job_count"] == len(tape)
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert metrics["maximum_expected_drift_bound_violation"] <= 1e-8
    assert metrics["maximum_candidate_attempts_per_frame"] <= budget
    assert metrics["mean_flow_s"] == pytest.approx(parts["mean_flow_s"])
    assert accounting["root_attempts"] + accounting["rollout_attempts"] == metrics["candidate_attempts"]
    assert accounting["maximum_budget_identity_error"] == accounting["maximum_stage_identity_error"] == 0
    assert accounting["rollout_attempts"] == 4 * accounting["rollout_frames"]
    assert accounting["minimum_seed_support_margin"] >= 0
    assert accounting["minimum_support_floor_margin"] >= 0
    assert accounting["maximum_relative_support_loss"] <= .01 + 1e-12
    assert accounting["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8
    assert accounting["maximum_path_population_error"] == accounting["maximum_random_draw_identity_error"] == 0
    assert accounting["rollout_random_draws"] == accounting["rollout_arrival_event_generations"] + 3 * accounting["rollout_frames"]
    assert accounting["rollout_rng_initializations"] == 3
    assert accounting["cv_terminal_value_coordinate_evaluations"] == 6 * accounting["rollout_frames"]
    assert accounting["maximum_cv_cost_identity_error"] <= 1e-8
    if budget == 14:
        assert accounting["current_search_resumed_frames"] > 0
        assert accounting["rollout_random_draws"] == accounting["rollout_attempts"] == 0
        assert [rng.getstate() for rng in recorder.arrival_rngs] == initial_states
        old = ArrivalRolloutRecorder(rates, mu)
        reference = simulate_frames(model, tape, 2.0, certificate, budget, selector=old)
        assert recorder.frames == old.frames and metrics == reference
    else:
        assert accounting["rollout_frames"] > 0 and accounting["rollout_random_draws"] > 0


def test_stochastic_control_is_matched_on_planning_seed_before_ratio_averaging():
    components = ("mean_deferred_visible_wait_s", "mean_arrival_barrier_wait_s", "mean_dispatch_to_completion_s")
    count_keys = (*ARRIVAL_COUNTERS, "recycled_candidate_attempts", "rollout_frames", "rollout_vetoed_frames", "rollout_random_draws", "rng_setup_wall_s", "cv_terminal_value_coordinate_evaluations")

    def row(seed, policy, flow, planning_seed=None, budget=26):
        return {"load": .5, "seed": seed, "budget": budget, "policy": policy, "planning_seed": planning_seed,
                "metrics": {"job_count": 10, "mean_flow_s": flow, "p90_flow_s": 2 * flow,
                            "makespan_s": 100.0, "mean_candidate_attempts": 12.0},
                "decomposition": {key: flow / 3 for key in components},
                "selector_accounting": {"selector_mean_wall_ms": 1.0, **dict.fromkeys(count_keys, 0)}}

    runs = []
    for seed, baseline, ratios in ((41, 10., (.5, 1., 1.8)), (42, 20., (.7, .8, .9)), (43, 40., (1., 1.1, 1.2))):
        runs.extend(row(seed, policy, baseline) for policy in CONTROLS[:2])
        runs.append(row(seed, "v4_support", baseline, budget=120))
        for planning_seed, ratio in enumerate(ratios):
            control_flow = baseline * (2 ** planning_seed)
            runs.append(row(seed, CONTROLS[-1], control_flow, planning_seed))
            runs.append(row(seed, NEW_POLICY, control_flow * ratio, planning_seed))
    paired, summaries = summarize_control_variate_runs(runs, (.5,), (26,))
    pairs = [r for r in paired if r["policy"] == NEW_POLICY]
    summary = next(r for r in summaries if r["policy"] == NEW_POLICY)
    assert len(pairs) == len({(r["seed"], r["planning_seed"]) for r in pairs}) == 9
    assert summary["seeds"] == [41, 42, 43]
    assert summary["run_count"] == 9
    control_ratios = [r["flow_ratio_over_poisson_rollout_cap_1pct"] for r in pairs]
    assert control_ratios == pytest.approx((.5, 1., 1.8, .7, .8, .9, 1., 1.1, 1.2))
    assert [r["mean_flow_ratio_over_poisson_rollout_cap_1pct"] for r in summary["by_arrival_seed"]] == pytest.approx((1.1, .8, 1.1))
    assert summary["mean_paired_flow_ratio_over_poisson_rollout_cap_1pct"] == pytest.approx(1.0)
    assert [r["mean_paired_flow_ratio_over_poisson_rollout_cap_1pct"] for r in summary["by_planning_seed"]] == pytest.approx((2.2 / 3, 2.9 / 3, 1.3))
    assert all(r["policy"] != CONTROLS[-1] for r in summaries)  # no misleading nine-environment summary.
    with pytest.raises(ValueError, match="every planning repetition"):
        summarize_control_variate_runs(runs[:-1], (.5,), (26,))


def test_control_reproduction_ignores_only_runtime_measurements():
    old = {"load": .5, "seed": 41, "budget": 26, "policy": CONTROLS[-1], "planning_seed": 1,
           "metrics": {"mean_flow_s": 15.0}, "decomposition": {"mean_flow_s": 15.0},
           "selector_accounting": {"selector_wall_s": 1.0, "selector_mean_wall_ms": 1.0,
                                   "rng_setup_wall_s": .01, "rollout_random_draws": 24}}
    new = deepcopy(old)
    for key in ("selector_wall_s", "selector_mean_wall_ms", "rng_setup_wall_s"):
        new["selector_accounting"][key] *= 2
    _assert_control_reproduced(new, old)
    new["selector_accounting"]["rollout_random_draws"] += 1
    with pytest.raises(ValueError, match="work accounting"):
        _assert_control_reproduced(new, old)
