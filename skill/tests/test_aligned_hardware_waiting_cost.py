import pytest

from algorithm.experiments.aligned_hardware_waiting_cost import COMPONENTS, summarize_runs


def test_waiting_comparison_keeps_frozen_v4_as_exact_reference():
    def run(policy, budget, flow):
        return {
            "load": 0.5, "seed": 41, "budget": budget, "policy": policy,
            "metrics": {"mean_flow_s": flow, "p90_flow_s": 2 * flow,
                        "makespan_s": 100.0, "mean_candidate_attempts": budget,
                        "job_count": 50},
            "decomposition": dict(zip(COMPONENTS, (flow - 3, 1.0, 2.0))),
            "selector_accounting": {"selector_mean_wall_ms": 1.0},
        }

    # The new K120 rule can have worse flow and lower support after a complete
    # search. It must never become the denominator of an exact comparison.
    runs = [run("v4_support", 26, 10.0), run("v4_support", 120, 20.0),
            run("seed_guarded_holding_v1", 26, 8.0),
            run("seed_guarded_holding_v1", 120, 30.0)]
    paired, summary = summarize_runs(runs, [0.5], [26, 120])
    assert paired[0]["waiting_over_v4_same_budget"]["mean_flow_s"] == pytest.approx(0.8)
    assert paired[0]["waiting_over_v4_exact"]["mean_flow_s"] == pytest.approx(0.4)
    assert paired[1]["waiting_over_v4_exact"]["mean_flow_s"] == pytest.approx(1.5)
    assert paired[0]["component_deltas_s"][COMPONENTS[0]] == pytest.approx(-2.0)
    assert summary[0]["improved_seed_count"] == 1
    assert summary[1]["improved_seed_count"] == 0
