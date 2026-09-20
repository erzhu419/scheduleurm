from __future__ import annotations

from dataclasses import replace
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_count_space_bnb_temporal_incumbent_development as temporal,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_model import load_model


def _stub_source():
    return (
        {
            "escalation_result_path": "/tmp/stub-escalation.json",
            "escalation_experiment": temporal.SOURCE_EXPERIMENT,
            "escalation_decision": temporal.SOURCE_DECISION,
            "escalation_advance": False,
            "prior_case_gates": [],
        },
        {"lanes": 2, "options": []},
        {"upstream.py": "frozen"},
    )


def test_small_warm_and_cold_simulators_are_exact_and_reconstruct_fifo_flow():
    model = replace(load_model(), lanes=8)
    arrivals = (
        (0.10, 0, 0),
        (0.20, 1, 0),
        (0.30, 2, 0),
        (0.40, 2, 1),
    )
    warm_metrics, warm_frames, warm_flows = (
        temporal.simulate_temporal_incumbent_frames(
            model, arrivals, arrival_horizon=2.0, warm=True
        )
    )
    cold_metrics, cold_frames, cold_flows = (
        temporal.simulate_temporal_incumbent_frames(
            model, arrivals, arrival_horizon=2.0, warm=False
        )
    )
    decomposition = decompose_flow(model, arrivals, warm_frames, 2.0)

    assert warm_frames == cold_frames
    assert warm_flows == cold_flows
    assert warm_metrics["job_count"] == warm_metrics["completed_jobs"] == 4
    assert cold_metrics["job_count"] == cold_metrics["completed_jobs"] == 4
    assert warm_metrics["exact_selection_frame_count"] == warm_metrics["frame_count"]
    assert cold_metrics["exact_selection_frame_count"] == cold_metrics["frame_count"]
    assert warm_metrics["projection_call_count"] == warm_metrics["frame_count"] - 1
    assert cold_metrics["projection_call_count"] == 0
    assert warm_metrics["invalid_projection_count"] == 0
    assert warm_metrics["maximum_queue_recursion_violation"] == 0
    assert decomposition["mean_flow_s"] == pytest.approx(warm_metrics["mean_flow_s"])


def test_registration_does_no_model_capacity_tape_projection_or_selector_work(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "temporal.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("registration performed execution work")

    monkeypatch.setattr(temporal, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        temporal,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "temporal.py": "frozen"},
    )
    monkeypatch.setattr(temporal, "load_model", forbidden)
    monkeypatch.setattr(temporal, "capacity_certificate", forbidden)
    monkeypatch.setattr(temporal, "poisson_arrivals", forbidden)
    monkeypatch.setattr(temporal, "project_temporal_incumbent", forbidden)
    monkeypatch.setattr(temporal, "select_warm_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(temporal, "select_count_space_bnb_frame", forbidden)

    protocol = temporal.register(output, tmp_path / "escalation.json")
    result_path, protocol_path, fixed_path = temporal._paths(output)
    assert not result_path.exists()
    assert protocol_path.exists() and fixed_path.exists()
    assert protocol["arrival_seeds"] == list(range(8041, 8049))
    assert protocol["loads"] == [0.8, 0.95]
    assert protocol["lanes"] == [8, 16]
    assert protocol["lp_node_budget"] == 4096
    assert "lower" in protocol["temporal_projection_rule"]["search_role"]


def test_run_rejects_protocol_change_before_model_or_tape_work(tmp_path, monkeypatch):
    output = tmp_path / "temporal.json"
    monkeypatch.setattr(temporal, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        temporal,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "temporal.py": "frozen"},
    )
    temporal.register(output, tmp_path / "escalation.json")
    _, protocol_path, _ = temporal._paths(output)
    protocol = json.loads(protocol_path.read_text())
    protocol["arrival_horizon_s"] = 601.0
    protocol_path.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("execution began before protocol validation")

    monkeypatch.setattr(temporal, "load_model", forbidden)
    monkeypatch.setattr(temporal, "poisson_arrivals", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        temporal.run_registered(output)


def test_decision_requires_each_case_and_aggregate_strict_work_gates():
    passing_cases = [{"case_gate_passed": True} for _ in range(4)]
    assert temporal._decision(True, passing_cases, True) == "TEMPORAL_INCUMBENT_PASS"
    assert (
        temporal._decision(True, passing_cases, False)
        == "TEMPORAL_INCUMBENT_WORK_FAIL_STOP"
    )
    failing_cases = [*passing_cases[:3], {"case_gate_passed": False}]
    assert (
        temporal._decision(True, failing_cases, True)
        == "TEMPORAL_INCUMBENT_WORK_FAIL_STOP"
    )
    assert temporal._decision(False, passing_cases, True) == "INVALID_EXECUTION"
