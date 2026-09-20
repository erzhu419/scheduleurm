"""Tests for the fixed seed-41 first-divergence diagnosis."""
import json
from pathlib import Path

import pytest

import algorithm.experiments.aligned_hardware_rollout_depth_gate_first_divergence as diagnosis
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_rollout_depth_gate import RolloutDepthGateRecorder
from algorithm.experiments.aligned_hardware_waiting_cost import PolicyRecorder
from algorithm.experiments.aligned_lookahead_continuation import DecisionPairs


@pytest.fixture(scope="module")
def example():
    model = load_model()
    certificate = capacity_certificate(model, .8)
    service_rates = terminal_service_rates(model, certificate)
    return model, certificate, service_rates


def _row(model, certificate, service_rates, tape, horizon, policy, planning_seed=None):
    if policy == diagnosis.NEW_POLICY:
        recorder = RolloutDepthGateRecorder(
            model.options, model.lanes, certificate["arrival_rates"], service_rates, planning_seed,
        )
    else:
        recorder = PolicyRecorder(policy, sum(certificate["arrival_rates"]), max_relative_support_loss=.01)
    metrics = simulate_frames(model, tape, horizon, certificate, 26, selector=recorder)
    return json.loads(json.dumps({
        "load": .8, "seed": 41, "budget": 26, "policy": policy,
        "planning_seed": planning_seed, "metrics": metrics,
        "decomposition": decompose_flow(model, tape, recorder.frames, horizon),
        "selector_accounting": recorder.accounting(),
    }))


def test_discovery_preserves_native_selection_rng_and_only_records_first_difference(example):
    model, certificate, service_rates = example
    rates = certificate["arrival_rates"]
    pairs = DecisionPairs(model, rates, service_rates, budget=26)
    discovery = diagnosis.FirstDivergenceDiscovery(model, rates, service_rates, 1, pairs)
    reference = RolloutDepthGateRecorder(model.options, model.lanes, rates, service_rates, 1)
    queues = ((1, 0, 0), (0, 8, 13), (6, 6, 10), (8, 8, 8))
    observed, expected = [], []
    for queue in queues:
        observed.append(discovery(model.options, queue, model.lanes, 26))
        expected.append(reference(model.options, queue, model.lanes, 26))
    assert observed == expected
    assert diagnosis._deterministic_accounting(discovery.native.accounting()) == diagnosis._deterministic_accounting(reference.accounting())
    assert [rng.getstate() for rng in discovery.native.arrival_rngs] == [rng.getstate() for rng in reference.arrival_rngs]
    assert discovery.first["frame_index"] == 2 and discovery.first["queue"] == (6, 6, 10)
    assert discovery.first["weighted_backlog"] > 96
    assert discovery.inactive_prefix_frames == discovery.equal_active_comparisons == 1
    assert discovery.active_v4_comparisons == 2
    # No shadow comparison is made after the first physical divergence.
    assert pairs.accounting()["support_calls"] == 2
    assert discovery.accounting()["extra_random_draws"] == 0


def test_run_source_replays_first_difference_and_closes_full_first_suffix_identity(example, monkeypatch):
    model, certificate, service_rates = example
    tape = tuple(
        (0., coordinate, index)
        for coordinate, count in enumerate((6, 6, 10)) for index in range(count)
    )
    horizon = 1.0
    gated = _row(model, certificate, service_rates, tape, horizon, diagnosis.NEW_POLICY, 1)
    v4 = _row(model, certificate, service_rates, tape, horizon, diagnosis.V4_POLICY)
    protocol = {
        "load": .8, "real_seed": 41, "planning_seeds": [0, 1, 2],
        "budget": 26, "arrival_horizon_s": horizon,
        "terminal_service_rates": list(service_rates),
    }
    monkeypatch.setattr(diagnosis, "poisson_arrivals", lambda *args: tape)
    result = diagnosis._run_source((model, certificate, protocol, 1, gated, v4))
    assert result["source_reproduced"] and result["first_divergence_count"] == 1
    assert result["replay_branch_count"] == result["diagnostic_work"]["replay_runs"] == 2
    case = result["case"]
    assert case["frame_index"] == 0 and case["queue"] == (6, 6, 10)
    assert case["v4_branch_matches_frozen_physical_result"]
    assert abs(case["first_action_effect"]["residual_identity_error_job_s"]) <= 1e-5
    assert abs(case["first_action_effect"]["flow_component_identity_error_job_s"]) <= 1e-5
    assert abs(case["full_first_suffix_identity_error_job_s"]) <= 1e-5
    assert max(map(abs, case["flow_component_identity_errors_job_s"].values())) <= 1e-5
    assert result["diagnostic_work"]["continuation_v4_cache_miss_work"]["lookahead_calls"] == 0


def test_no_divergence_is_retained_without_replay(example, monkeypatch):
    model, certificate, service_rates = example
    tape = ((0., 0, 0),)
    horizon = 1.0
    gated = _row(model, certificate, service_rates, tape, horizon, diagnosis.NEW_POLICY, 0)
    v4 = _row(model, certificate, service_rates, tape, horizon, diagnosis.V4_POLICY)
    protocol = {
        "load": .8, "real_seed": 41, "planning_seeds": [0, 1, 2],
        "budget": 26, "arrival_horizon_s": horizon,
        "terminal_service_rates": list(service_rates),
    }
    monkeypatch.setattr(diagnosis, "poisson_arrivals", lambda *args: tape)
    result = diagnosis._run_source((model, certificate, protocol, 0, gated, v4))
    assert result["source_reproduced"] and result["first_divergence_count"] == 0
    assert result["case"] is None and result["replay_branch_count"] == 0
    assert result["diagnostic_work"]["replay_runs"] == result["diagnostic_work"]["replay_frames"] == 0


def test_execution_checks_keep_registered_zero_case_sources_valid():
    runs = []
    for planning_seed in (0, 1, 2):
        runs.append({
            "planning_seed": planning_seed, "source_reproduced": True,
            "first_divergence_count": 0, "replay_branch_count": 0, "case": None,
            "diagnostic_work": {
                "discovery": {
                    "inactive_prefix_frames": 1, "active_v4_comparisons": 0,
                    "equal_active_comparisons": 0,
                    "first_divergence_weighted_queue_coordinate_evaluations": 0,
                    "extra_rng_initializations": 0, "extra_rng_state_reads": 0,
                    "extra_rng_state_writes": 0, "extra_random_draws": 0,
                },
                "discovery_v4_cache_miss_work": {"lookahead_calls": 0},
                "continuation_v4_cache_miss_work": {"lookahead_calls": 0},
                "replay_runs": 0, "action_audit_evaluations": 0,
                "action_audit_completion_offset_reads": 0,
            },
            "source_frame_count": 1,
        })
    assert all(diagnosis._execution_checks(runs, {}).values())
    runs[0]["replay_branch_count"] = 2
    assert not diagnosis._execution_checks(runs, {})["two_common_v4_replay_branches_per_divergence"]


def test_registration_freezes_only_failed_development_rows_without_arrivals(tmp_path, monkeypatch):
    output = tmp_path / "diagnosis.json"
    read_text = Path.read_text

    def allowed_read(path, *args, **kwargs):
        assert "mixed_successor_confirmation" not in str(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", allowed_read)
    monkeypatch.setattr(diagnosis, "poisson_arrivals", lambda *args: pytest.fail("registration must not generate arrivals"))
    protocol = diagnosis.register(output)
    assert protocol["load"] == .8 and protocol["real_seed"] == 41
    assert protocol["planning_seeds"] == (0, 1, 2) and protocol["budget"] == 26
    assert protocol["gate_weights"] == (8, 8, 3)
    assert protocol["gate_capacity"] == 48 and protocol["strict_threshold"] == 96
    assert protocol["selected_failed_cell"] == {"load": .8, "real_seed": 41}
    assert len(protocol["source_rows"]) == 4
    assert {path.name for path in tmp_path.iterdir()} == {"diagnosis_protocol.json", "diagnosis_fixed_sources.json"}
    saved = tmp_path / "diagnosis_protocol.json"
    changed = json.loads(saved.read_text())
    changed["planning_seeds"] = [0, 1, 4041]
    saved.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="roster or gate changed"):
        diagnosis.run_registered(output)

    bad = json.loads(read_text(diagnosis.DEVELOPMENT))
    medium = next(row for row in bad["summary_by_load"] if row["load"] == .8)
    next(seed for seed in medium["by_arrival_seed"] if seed["seed"] == 41)["flow_ratio_over_v4_support"] = .99
    bad_source = tmp_path / "wrong_failure.json"
    bad_source.write_text(json.dumps(bad))
    monkeypatch.setattr(diagnosis, "DEVELOPMENT", bad_source)
    with pytest.raises(ValueError, match="unique registered medium/high-load failure"):
        diagnosis.register(tmp_path / "wrong_diagnosis.json")
