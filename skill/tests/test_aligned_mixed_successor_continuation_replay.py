"""Matched shadows preserve native sources; paired effects use mixed-minus-raw."""
from dataclasses import replace
import json

import pytest

import algorithm.experiments.aligned_mixed_successor_continuation_replay as replay
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_mixed_successor_rollout import MixedSuccessorRolloutRecorder
from algorithm.experiments.aligned_hardware_poisson_rollout import PoissonRolloutRecorder


@pytest.fixture(scope="module")
def example():
    model = load_model()
    certificate = capacity_certificate(model, .5)
    mu = terminal_service_rates(model, certificate)
    # The initial single CNN makes an inactive, nonempty common prefix. A
    # subsequent authored wave creates Q=(1,4,8), followed by two arrivals.
    tape = [(0., 0, 0)] + [
        (.5, coordinate, index + int(coordinate == 0))
        for coordinate, count in enumerate((1, 4, 8)) for index in range(count)
    ] + [(9., 2, 8), (9.5, 0, 2)]
    return model, certificate, mu, tape


def _native(policy, certificate, mu, planning_seed):
    cls = PoissonRolloutRecorder if policy == replay.RAW_POLICY else MixedSuccessorRolloutRecorder
    return cls(certificate["arrival_rates"], mu, planning_seed)


def _counts(recorder):
    timing = {"selector_wall_s", "selector_mean_wall_ms", "rng_setup_wall_s"}
    return {key: value for key, value in recorder.accounting().items() if key not in timing}


@pytest.mark.parametrize("policy", replay.SOURCE_POLICIES)
@pytest.mark.parametrize("planning_seed,orientation", ((0, -1), (7, 1)))
def test_shadow_preserves_native_frames_counts_rng_and_both_disagreement_orientations(
        example, policy, planning_seed, orientation, monkeypatch):
    model, certificate, mu, tape = example
    reference = _native(policy, certificate, mu, planning_seed)
    expected = simulate_frames(model, tape, 10., certificate, 26, selector=reference)
    name = ("select_mixed_successor_rollout_frame" if policy == replay.RAW_POLICY
            else "select_poisson_rollout_frame")
    selector, shadows = getattr(replay, name), []

    def observe(*args, **kwargs):
        selected = selector(*args, **kwargs)
        shadows.append(selected)
        return selected

    monkeypatch.setattr(replay, name, observe)
    discovery = replay.DisagreementDiscovery(model, certificate["arrival_rates"], mu, planning_seed, policy)
    actual = simulate_frames(model, tape, 10., certificate, 26, selector=discovery)
    assert actual == expected
    assert discovery.frames == reference.frames
    assert _counts(discovery.native) == _counts(reference)
    assert [rng.getstate() for rng in discovery.native.arrival_rngs] == [
        rng.getstate() for rng in reference.arrival_rngs]
    assert len(discovery.prefix) == len(reference.frames)
    active = reference.accounting()["rollout_frames"]
    assert 0 < active < len(reference.frames)
    assert discovery.shadow_work["selector_calls"] == len(shadows) == active == 1
    assert discovery.shadow_work["rng_initializations"] == discovery.shadow_work["rng_state_writes"] == 3 * active
    assert discovery.shadow_work["rng_state_reads"] == 3 * (len(reference.frames) + 2 * active)
    for key, attribute in replay.WORK_FIELDS.items():
        assert discovery.shadow_work[key] == sum(getattr(selected, attribute, 0) for selected in shadows), key
    assert len(discovery.switches) == 1
    switch = discovery.switches[0]
    assert switch["frame_index"] == 1 and switch["queue"] == (1, 4, 8)
    assert switch["start_s"] == reference.frames[0]["duration_s"]
    assert switch["orientation_mixed_minus_raw"] == orientation
    assert switch["raw_action"].rollout_vetoed == (orientation == 1)
    assert switch["mixed_action"].rollout_vetoed == (orientation == -1)
    assert orientation * switch["raw_rollout_margin"] > 0
    assert orientation * switch["mixed_rollout_margin"] < 0
    assert set((switch["raw_action"].option_indices, switch["mixed_action"].option_indices)) == {(7, 7), (8, 5)}
    assert discovery.shadow_work["physical_action_descriptions"] == 2
    assert discovery.shadow_work["comparison_completion_offset_reads"] == 17


@pytest.mark.parametrize("policy", replay.SOURCE_POLICIES)
def test_inactive_call_does_not_execute_shadow_or_change_rng(example, policy, monkeypatch):
    model, certificate, mu, _ = example
    discovery = replay.DisagreementDiscovery(model, certificate["arrival_rates"], mu, 0, policy)
    before = [rng.getstate() for rng in discovery.native.arrival_rngs]

    def forbidden(*args, **kwargs):
        pytest.fail("inactive native selection must not call the opposite selector")

    monkeypatch.setattr(replay, "select_poisson_rollout_frame", forbidden)
    monkeypatch.setattr(replay, "select_mixed_successor_rollout_frame", forbidden)
    result = discovery(model.options, (1, 0, 0), model.lanes, 26)
    assert not result.rollout_used and not discovery.switches
    assert [rng.getstate() for rng in discovery.native.arrival_rngs] == before
    assert discovery.shadow_work["selector_calls"] == discovery.shadow_work["rng_initializations"] == 0
    assert discovery.shadow_work["rng_state_writes"] == 0
    assert all(discovery.shadow_work[key] == 0 for key in replay.WORK_FIELDS)


@pytest.mark.parametrize("policy", replay.SOURCE_POLICIES)
def test_lane_permutation_is_not_a_physical_disagreement(example, policy, monkeypatch):
    model, certificate, mu, _ = example
    name = ("select_mixed_successor_rollout_frame" if policy == replay.RAW_POLICY
            else "select_poisson_rollout_frame")
    selector = getattr(replay, name)
    observed = []

    def permute(*args, **kwargs):
        selected = selector(*args, **kwargs)
        assert selected.option_indices == (8, 5)
        result = replace(selected, option_indices=(5, 8))
        observed.append(result)
        return result

    monkeypatch.setattr(replay, name, permute)
    discovery = replay.DisagreementDiscovery(model, certificate["arrival_rates"], mu, 2, policy)
    selected = discovery(model.options, (1, 4, 8), model.lanes, 26)
    assert selected.rollout_used and selected.option_indices == (8, 5)
    assert observed[0].option_indices != selected.option_indices
    assert not discovery.switches and discovery.shadow_work["selector_calls"] == 1


@pytest.mark.parametrize("policy", replay.SOURCE_POLICIES)
@pytest.mark.parametrize("planning_seed,orientation", ((0, -1), (7, 1)))
def test_run_source_replays_shared_prefix_population_and_signed_area_identity(
        example, policy, planning_seed, orientation, monkeypatch):
    model, certificate, mu, tape = example
    native = _native(policy, certificate, mu, planning_seed)
    metrics = simulate_frames(model, tape, 10., certificate, 26, selector=native)
    original = {
        "load": .5, "seed": 41, "budget": 26, "planning_seed": planning_seed, "policy": policy,
        "metrics": metrics, "decomposition": decompose_flow(model, tape, native.frames, 10.),
        "selector_accounting": native.accounting(),
    }
    original = json.loads(json.dumps(original))
    branches = []

    class CapturedReplay(replay.PrefixReplay):
        def __init__(self, *args):
            super().__init__(*args)
            branches.append(self)

    monkeypatch.setattr(replay, "PrefixReplay", CapturedReplay)
    monkeypatch.setattr(replay, "poisson_arrivals", lambda *args: tape)
    result = replay.run_source((model, certificate, .5, 41, planning_seed, policy,
                               {"budget": 26, "arrival_horizon_s": 10., "terminal_service_rates": mu}, original))
    assert result["source_reproduced"] and result["paired_replay_count"] == 1
    assert result["job_count"] == len(tape) == 16
    case = result["cases"][0]
    assert case["frame_index"] == 1 and case["phase"] == "crosses_cutoff"
    assert case["orientation_mixed_minus_raw"] == orientation
    assert case["oriented_raw_sample_delta_job_s"] > 0 > case["oriented_mixed_sample_delta_job_s"]
    assert len(branches) == 2
    population = tuple(sum(coordinate == i for _, coordinate, _ in tape) for i in range(3))
    for branch, description in zip(branches, (case["raw_action"], case["mixed_action"]), strict=True):
        assert branch.frames[0] == native.frames[0]
        assert tuple(sum(frame["departures"][i] for frame in branch.frames) for i in range(3)) == population
        assert branch.frames[1]["departures"] == description["departures"]
    effect = case["effect"]
    total = len(tape) * (case["branch_mean_flow_s"][1] - case["branch_mean_flow_s"][0])
    assert effect["realized_total_delta_job_s"] == pytest.approx(total)
    assert effect["realized_current_delta_job_s"] == pytest.approx(
        case["realized_current_area_job_s"][1] - case["realized_current_area_job_s"][0])
    assert effect["realized_current_delta_job_s"] + effect["realized_future_delta_job_s"] == pytest.approx(total)
    assert sum(effect["flow_component_deltas_job_s"].values()) == pytest.approx(total)
    assert abs(effect["flow_component_identity_error_job_s"]) < 1e-6
    work = result["diagnostic_work"]
    assert work["replay_runs"] == work["action_audit_evaluations"] == 2
    assert work["replay_frames"] == sum(len(branch.frames) for branch in branches)
    assert work["action_audit_completion_offset_reads"] == 34
    assert work["replay_action_description_completion_offset_reads"] == 17
    assert work["continuation"]["support_calls"] > 0
    assert work["continuation"]["lookahead_calls"] == 0
