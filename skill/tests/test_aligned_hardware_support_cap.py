import pytest

from algorithm.experiments.aligned_hardware_waiting_cost import COMPONENTS
from algorithm.experiments.aligned_hardware_support_cap import summarize_runs


def test_cap_policies_are_compared_separately_against_v4():
    def run(policy, budget, flow):
        return {
            "load": 0.5, "seed": 41, "budget": budget, "policy": policy,
            "metrics": {"mean_flow_s": flow, "p90_flow_s": 2 * flow,
                        "makespan_s": 100.0, "mean_candidate_attempts": budget,
                        "job_count": 50},
            "decomposition": dict(zip(COMPONENTS, (flow - 3, 1.0, 2.0))),
            "selector_accounting": {"selector_mean_wall_ms": 1.0},
        }

    runs = [run(policy, budget, flow) for budget in (26, 120)
            for policy, flow in (("v4_support", 10.0), ("holding_cap_zero", 10.0),
                                 ("holding_cap_1pct", 8.0))]
    paired, summary = summarize_runs(runs, [0.5], [26, 120])
    assert len(paired) == len(summary) == 4
    for row in summary:
        expected = 1.0 if row["policy"] == "holding_cap_zero" else 0.8
        assert row["mean_paired_flow_ratio_over_v4"] == pytest.approx(expected)
        assert row["mean_paired_flow_ratio_over_v4_exact"] == pytest.approx(expected)
        assert row["seeds"] == [41]
