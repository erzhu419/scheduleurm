from types import SimpleNamespace

import pytest

from algorithm.experiments.aligned_hardware_lookahead_continuation import action_audit, paired_effect, summarize
from algorithm.experiments.aligned_hardware_waiting_cost import COMPONENTS


def test_cutoff_and_realized_boundary_separate_stationary_prediction_from_actual_arrivals():
    model = SimpleNamespace(options=[SimpleNamespace(completion_offsets=((1.0,),))])
    action = SimpleNamespace(option_indices=(0,), service=(1,), duration=4.0)
    audit = action_audit(model, (3,), action, 8.0, 10.0, (2.0,), (4.0,),
                         (8.0, 9.0, 10.0), (0.0, 8.0, 17.0, 27.0), ((8.0, 9.0, 10.0),))
    assert audit["predicted_current_area"] == 25.0
    assert audit["predicted_terminal_area"] == 13.5
    assert audit["truncated_expected_current_area"] == 21.0
    assert audit["truncated_expected_terminal_area"] == 5.0
    assert audit["realized_current_area"] == 14.0
    assert audit["realized_next_queue"] == (4,)
    assert audit["realized_terminal_value"] == 2.0
    drain = action_audit(model, (3,), action, 10.0, 10.0, (2.0,), (4.0,),
                         (8.0, 9.0, 10.0), (0.0, 8.0, 17.0, 27.0), ((8.0, 9.0, 10.0),))
    assert drain["truncated_expected_current_area"] == drain["realized_current_area"] == 9.0
    assert drain["truncated_expected_terminal_area"] == drain["realized_terminal_value"] == 0.5
    assert drain["realized_next_queue"] == (2,)


def test_paired_effect_sign_units_and_four_residual_terms():
    before = {"metrics": {"mean_flow_s": 10.0}, "decomposition": dict(zip(COMPONENTS, (3.0, 3.0, 4.0)))}
    after = {"metrics": {"mean_flow_s": 12.0}, "decomposition": dict(zip(COMPONENTS, (5.0, 2.0, 5.0)))}
    keys = ("predicted_current_area", "predicted_terminal_area", "predicted_total_area",
            "truncated_expected_current_area", "truncated_expected_terminal_area",
            "realized_current_area", "realized_terminal_value")
    audits = [dict(zip(keys, values)) for values in ((10, 30, 40, 9, 29, 8, 28), (8, 27, 35, 6, 24, 7, 30))]
    for audit in audits:
        audit["realized_next_queue"] = (1,)
    effect = paired_effect(before, after, audits, 10)
    assert effect["predicted_total_delta_job_s"] == -5
    assert effect["realized_total_delta_job_s"] == 20
    assert effect["realized_current_delta_job_s"] == -1
    assert effect["realized_future_delta_job_s"] == 21
    assert effect["cutoff_correction_delta_job_s"] == -3
    assert effect["current_arrival_residual_delta_job_s"] == 2
    assert effect["next_queue_residual_delta_job_s"] == 7
    assert effect["continuation_residual_delta_job_s"] == 19
    assert effect["prediction_residual_delta_job_s"] == 25
    assert effect["residual_identity_error_job_s"] == effect["flow_component_identity_error_job_s"] == 0
    assert effect["outcome"] == "worsened"
    assert effect["flow_component_deltas_job_s"] == dict(zip(COMPONENTS, (20, -10, 10)))


def test_seed_without_substitutions_stays_in_the_summary():
    summary = summarize([{"cases": []}])
    assert summary["substitutions"] == summary["improved"] == summary["worsened"] == 0
    assert summary["median_realized_delta_job_s"] is None
    assert all(value is None for value in summary["mean_deltas_job_s"].values())
