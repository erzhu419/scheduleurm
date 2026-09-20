from algorithm.experiments.aligned_hardware_mixed_successor_continuation import (
    _sum_work, _summarize_subset,
)


def _case(state_id, realized):
    estimate = {
        "fresh_mean_minus_discovery_sample_job_s": 2.,
        "realized_minus_fresh_proxy_mean_job_s": realized - 1.,
        "proxy_point_estimate_disagrees_with_realized": realized < 0,
    }
    return {"state_id": state_id, "effect": {"outcome": "improved" if realized < 0 else "worsened",
                                               "realized_total_delta_job_s": realized},
            "estimate_comparison": {"raw": estimate, "mixed": estimate}}


def test_source_summary_uses_only_states_visited_by_that_source_subset():
    runs = [{"cases": [_case("a", -2.)]}, {"cases": [_case("b", 3.)]}]
    states = [
        {"state_id": "a", "raw_margin_variance": 2., "mixed_margin_variance": 1.},
        {"state_id": "b", "raw_margin_variance": 1., "mixed_margin_variance": 2.},
    ]
    summary = _summarize_subset(runs[:1], states)
    assert summary["source_count"] == summary["disagreements"] == summary["unique_states"] == 1
    assert summary["improved"] == 1 and summary["worsened"] == 0
    assert summary["state_variance_comparison"] == {"decreased": 1, "equal": 0, "increased": 0}


def test_sampling_work_union_preserves_raw_and_mixed_only_fields():
    states = [{"raw_work": {"calls": 2}, "mixed_work": {"calls": 2, "lane": 7}},
              {"raw_work": {"calls": 3}, "mixed_work": {"calls": 3, "lane": 9}}]
    assert _sum_work(states, "raw_work") == {"calls": 5}
    assert _sum_work(states, "mixed_work") == {"calls": 5, "lane": 16}
