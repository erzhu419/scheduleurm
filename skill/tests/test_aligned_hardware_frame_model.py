from dataclasses import replace

import numpy as np
import pytest

from algorithm.experiments.aligned_hardware_frame_model import (
    FrameModel, FrameOption, capacity_certificate, load_model,
)
from algorithm.experiments import aligned_hardware_frame_model as frame_model


def test_registered_catalog_contains_only_executable_classes_and_complete_jobs():
    model = load_model()
    assert model.lanes == 2
    assert model.scales == (60.0, 40.0, 120.0)
    assert len(model.options) == 9
    assert sum(option.loaded for option in model.options) == 2
    assert len(model.excluded_actions) == 1
    assert "cnn_after_rl" in model.excluded_actions[0]["action_id"]
    for option in model.options:
        assert tuple(map(len, option.completion_offsets)) == option.requirements
        assert all(
            0 < offset <= option.duration
            for offsets in option.completion_offsets for offset in offsets
        )


def test_loaded_frames_charge_fresh_resident_initialization_and_finalization_once():
    model = load_model()
    for option in model.options:
        if not option.loaded:
            continue
        phases = dict(option.phases)
        assert phases["resident_initialization_s"] > 0
        assert phases["resident_finalization_s"] >= 0
        assert option.duration == pytest.approx(sum(
            phases[key] for key in (
                "resident_initialization_s", "target_overlap_s",
                "resident_dedicated_drain_s", "resident_finalization_s",
            )
        ))
        assert phases["resident_remaining_native_units"] == 0
        assert phases["resident_dedicated_drain_s"] == 0
        assert len(option.source_action_ids) == 2


def _toy_model(*, loaded_duration=1.0):
    return FrameModel(
        node="test", scenario_id="test", lanes=2, workloads=("a", "b"),
        scales=(1.0, 1.0), options=(
            FrameOption("a", (1, 0), 1.0, ((1.0,), ())),
            FrameOption("b", (0, 1), 1.0, ((), (1.0,))),
            FrameOption("ab", (1, 1), loaded_duration,
                        ((loaded_duration,), (loaded_duration,)), loaded=True),
        ),
    )


def test_loaded_capacity_and_duration_change_the_actual_arrival_scale():
    model = _toy_model()
    certificate = capacity_certificate(model, 0.8)
    assert certificate["full_boundary_scale"] == pytest.approx(2.0)
    assert certificate["arrival_rates"] == pytest.approx((1.6, 1.6))
    assert certificate["support_slack_delta"] == pytest.approx(0.4)
    assert certificate["capacity_projection_passed"]
    assert certificate["conditional_model_drift_certified"]
    slower = capacity_certificate(_toy_model(loaded_duration=2.0), 0.8)
    assert slower["full_boundary_scale"] == pytest.approx(1.0)
    assert slower["arrival_rates"] == pytest.approx((0.8, 0.8))


def test_infeasible_loaded_projection_is_not_a_global_stability_certificate():
    # The loaded a-coordinate is faster than any a-only option, so arbitrarily
    # large a queues with b=0 violate the proposed constant support-loss bound.
    model = _toy_model()
    faster = FrameOption("ab", (2, 1), 1.0, ((1.0, 1.0), (1.0,)), loaded=True)
    model = replace(model, options=(*model.options[:2], faster))
    certificate = capacity_certificate(model, 0.8)
    assert not certificate["capacity_projection_passed"]
    assert not certificate["conditional_model_drift_certified"]
    assert certificate["negative_drift_backlog_threshold"] is None
    assert certificate["projection_failures"]


def test_short_options_use_the_selector_one_second_duration_floor():
    certificate = capacity_certificate(_toy_model(loaded_duration=0.5), 0.8)
    assert certificate["full_boundary_scale"] == pytest.approx(2.0)
    assert certificate["arrival_rates"] == pytest.approx((1.6, 1.6))
    assert certificate["tau_min"] == 1.0
    assert certificate["tau_max"] == 1.0
    assert certificate["capacity_projection_passed"]


def test_lp_witnesses_determine_slack_and_preserve_positive_projection_loss(monkeypatch):
    solve = frame_model.linprog

    def numerical_witness(objective, **kwargs):
        result = solve(objective, **kwargs)
        if result.success and np.all(objective == 0):
            # A solver may accept a slightly underserving projection witness.
            result.x *= 1.0 - 1e-8
        elif result.success and isinstance(kwargs["bounds"], list):
            # Its reported objective cannot certify more than its time shares.
            result.x[:-1] *= 1.0 + 1e-8
            result.x[-1] += 0.1
        return result

    monkeypatch.setattr(frame_model, "linprog", numerical_witness)
    certificate = capacity_certificate(_toy_model(), 0.8)
    assert sum(certificate["slack_time_weights"]) <= 1.0
    assert min(certificate["slack_time_weights"]) >= 0.0
    assert certificate["support_slack_delta"] == pytest.approx(0.4)
    assert certificate["projection_support_loss_rate"] > 0.0
    assert certificate["effective_slack_eta"] == pytest.approx(
        certificate["support_slack_delta"] - certificate["projection_support_loss_rate"]
    )
    assert certificate["effective_slack_eta"] < certificate["support_slack_delta"]


def test_reference_certificate_bounds_integer_queue_service_and_poisson_drift():
    model = load_model()
    certificate = capacity_certificate(model, 0.85)
    assert certificate["capacity_projection_passed"]
    assert certificate["conditional_model_drift_certified"]
    assert certificate["projection_check_count"] == 7 * len(model.options)
    assert certificate["max_departures"] == (6, 6, 16)
    assert certificate["support_slack_delta"] > 0
    rates = np.asarray([option.requirements for option in model.options]) / np.asarray(
        [option.duration for option in model.options]
    )[:, None]
    actual_mean = model.lanes * np.asarray(certificate["slack_time_weights"]) @ rates
    assert np.all(actual_mean >= np.asarray(certificate["arrival_rates"])
                  + certificate["support_slack_delta"] - 1e-9)
    outside = capacity_certificate(model, 1.05)
    assert outside["support_slack_delta"] < 0
    assert not outside["conditional_model_drift_certified"]
