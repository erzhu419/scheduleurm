from math import sqrt
from statistics import variance

import pytest

from algorithm.experiments.aligned_hardware_mixed_successor_rollout_diagnostic import (
    compare_case, paired_summary, summarize,
)


def test_model_change_uncertainty_uses_paired_samples():
    # Large shared path noise cancels from the difference between models.
    old = (101.0, -99.0, 201.0, -199.0)
    changes = (1.0, 2.0, 3.0, 2.0)
    rows = [{"old_raw_margin": raw, "mixed_margin": raw + change}
            for raw, change in zip(old, changes, strict=True)]
    summary = paired_summary(rows)
    assert summary["old_raw_margin_mean"] == 1.0
    assert summary["mixed_margin_mean"] == 3.0
    assert summary["paired_mean_change"] == 2.0
    assert summary["paired_mean_change_mc_se"] == pytest.approx(sqrt(variance(changes) / 4))
    assert summary["paired_mean_change_mc_se"] < summary["mixed_margin_mc_se"] / 100


@pytest.mark.parametrize("orientation", (-1, 1))
def test_retained_intervention_orientation_applies_to_both_model_estimates(orientation):
    state = {"state_id": "toy", "old_raw_margin_mean": 2.0, "mixed_margin_mean": -4.0,
             "old_raw_margin_mc_se": .5, "mixed_margin_mc_se": 1.0,
             "paired_mean_change": -6.0, "paired_mean_change_mc_se": .75,
             "old_raw_margin_variance": 1.0, "mixed_margin_variance": 4.0}
    case = {"orientation_cv_minus_raw": orientation, "effect": {"realized_total_delta_job_s": -10 * orientation},
            "frame_index": 7, "start_s": 20.0, "phase": "arrival_window"}
    result = compare_case(case, state)
    assert result["estimates"]["old_raw"]["proxy_mean_delta_job_s"] == 2 * orientation
    assert result["estimates"]["mixed"]["proxy_mean_delta_job_s"] == -4 * orientation
    assert result["estimates"]["old_raw"]["point_estimate_disagrees_with_realized"]
    assert not result["estimates"]["mixed"]["point_estimate_disagrees_with_realized"]
    assert result["oriented_paired_mean_change_job_s"] == -6 * orientation
    assert result["paired_mean_change_mc_se_job_s"] == .75
    assert result["residual_identity_error_job_s"] == 0
    # Repeated occurrences remain explicit; absolute residuals do not cancel
    # simply because the retained intervention orientation is reversed.
    summary = summarize([state], [result, result])
    assert summary["unique_state_count"] == 1 and summary["retained_case_occurrences"] == 2
    assert summary["old_raw"]["point_estimate_disagrees_with_realized"] == 2
    assert summary["old_raw"]["mean_absolute_realized_minus_proxy_mean_job_s"] == 12
    assert summary["mixed"]["median_absolute_realized_minus_proxy_mean_job_s"] == 6
