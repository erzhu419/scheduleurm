"""Matched shadow calls must preserve the source and intervention direction."""
import json

import pytest

import algorithm.experiments.aligned_control_variate_continuation_replay as replay
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_control_variate import ControlVariateRolloutRecorder
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_poisson_rollout import PoissonRolloutRecorder


@pytest.fixture(scope="module")
def example():
    model = load_model()
    certificate = capacity_certificate(model, .5)
    mu = terminal_service_rates(model, certificate)
    tape = [(0., coordinate, index) for coordinate, count in enumerate((1, 4, 8))
            for index in range(count)] + [(.5, 2, 8), (1., 0, 1)]
    return model, certificate, mu, tape


def _native(policy, certificate, mu):
    cls = PoissonRolloutRecorder if policy == replay.RAW_POLICY else ControlVariateRolloutRecorder
    return cls(certificate["arrival_rates"], mu, 1)


def _counts(recorder):
    timing = {"selector_wall_s", "selector_mean_wall_ms", "rng_setup_wall_s"}
    return {key: value for key, value in recorder.accounting().items() if key not in timing}


@pytest.mark.parametrize("policy", (replay.RAW_POLICY, replay.CV_POLICY))
def test_shadow_preserves_native_stream_trajectory_and_accounting(example, policy):
    model, certificate, mu, tape = example
    reference = _native(policy, certificate, mu)
    expected = simulate_frames(model, tape, 2., certificate, 26, selector=reference)
    discovery = replay.DisagreementDiscovery(model, certificate["arrival_rates"], mu, 1, policy)
    actual = simulate_frames(model, tape, 2., certificate, 26, selector=discovery)
    assert actual == expected
    assert discovery.frames == reference.frames
    assert _counts(discovery.native) == _counts(reference)
    assert [r.getstate() for r in discovery.native.arrival_rngs] == [r.getstate() for r in reference.arrival_rngs]
    assert len(discovery.prefix) == len(reference.frames)
    active = reference.accounting()["rollout_frames"]
    assert 0 < active < len(reference.frames)  # Both active and inactive source calls occur.
    assert discovery.shadow_work["selector_calls"] == active
    assert discovery.shadow_work["rng_initializations"] == 3 * active
    assert discovery.shadow_work["rng_state_writes"] == 3 * active
    assert discovery.shadow_work["rng_state_reads"] == 3 * (len(reference.frames) + 2 * active)
    assert len(discovery.switches) == 1
    switch = discovery.switches[0]
    assert switch["frame_index"] == 0
    assert switch["raw_action"].option_indices == (7, 7)
    assert switch["cv_action"].option_indices == (8, 5)
    assert switch["orientation_cv_minus_raw"] == -1
    assert switch["raw_rollout_margin"] < 0 < switch["corrected_rollout_margin"]


@pytest.mark.parametrize("policy", (replay.RAW_POLICY, replay.CV_POLICY))
def test_inactive_frame_does_not_construct_or_run_shadow(example, policy, monkeypatch):
    model, certificate, mu, _ = example
    discovery = replay.DisagreementDiscovery(model, certificate["arrival_rates"], mu, 1, policy)
    before = [rng.getstate() for rng in discovery.native.arrival_rngs]

    def forbidden(*args, **kwargs):
        pytest.fail("inactive frame must not execute the opposite selector")

    monkeypatch.setattr(replay, "select_poisson_rollout_frame", forbidden)
    monkeypatch.setattr(replay, "select_control_variate_rollout_frame", forbidden)
    selected = discovery(model.options, (0, 0, 0), model.lanes, 26)
    assert not selected.rollout_used and not discovery.switches
    assert [rng.getstate() for rng in discovery.native.arrival_rngs] == before
    assert discovery.shadow_work["selector_calls"] == discovery.shadow_work["rng_initializations"] == 0
    assert discovery.shadow_work["rng_state_writes"] == 0


def test_paired_effect_has_cv_minus_raw_sign_on_identical_population(example, monkeypatch):
    model, certificate, mu, tape = example
    native = _native(replay.RAW_POLICY, certificate, mu)
    metrics = simulate_frames(model, tape, 2., certificate, 26, selector=native)
    original = {
        "load": .5, "seed": 41, "budget": 26, "planning_seed": 1, "policy": replay.RAW_POLICY,
        "metrics": metrics, "decomposition": decompose_flow(model, tape, native.frames, 2.),
        "selector_accounting": native.accounting(),
    }
    original = json.loads(json.dumps(original))  # Frozen rows are loaded from JSON.
    # Use this authored tape rather than any new development or confirmation run.
    monkeypatch.setattr(replay, "poisson_arrivals", lambda *args: tape)
    result = replay.run_source((model, certificate, .5, 41, 1, replay.RAW_POLICY,
                               {"budget": 26, "arrival_horizon_s": 2., "terminal_service_rates": mu}, original))
    assert result["source_reproduced"] and result["paired_replay_count"] == 1
    assert result["job_count"] == len(tape)
    case = result["cases"][0]
    assert case["phase"] == "crosses_cutoff"
    assert case["oriented_raw_sample_delta_job_s"] > 0 > case["oriented_cv_sample_delta_job_s"]
    effect = case["effect"]
    expected = len(tape) * (case["branch_mean_flow_s"][1] - case["branch_mean_flow_s"][0])
    assert effect["realized_total_delta_job_s"] == pytest.approx(expected)
    assert effect["realized_current_delta_job_s"] == pytest.approx(
        case["realized_current_area_job_s"][1] - case["realized_current_area_job_s"][0])
    assert effect["realized_current_delta_job_s"] + effect["realized_future_delta_job_s"] == pytest.approx(expected)
    assert sum(effect["flow_component_deltas_job_s"].values()) == pytest.approx(expected)
    assert result["diagnostic_work"]["replay_runs"] == 2
