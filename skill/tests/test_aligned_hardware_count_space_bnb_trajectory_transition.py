from __future__ import annotations

from dataclasses import replace
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_count_space_bnb_trajectory_transition as transition,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_model import load_model


def _stub_source():
    return (
        {
            "confirmation_result_path": "/tmp/stub-confirmation.json",
            "confirmation_experiment": transition.SOURCE_EXPERIMENT,
            "confirmation_decision": transition.SOURCE_DECISION,
            "confirmed_lane_budgets": {},
            "inherited_r2_oracle_evidence": {},
        },
        {"lanes": 2, "options": []},
        {"upstream.py": "frozen"},
    )


def test_small_count_space_simulator_completes_and_decomposes_fifo_flow():
    model = replace(load_model(), lanes=8)
    arrivals = (
        (0.10, 0, 0),
        (0.20, 1, 0),
        (0.30, 2, 0),
        (0.40, 2, 1),
    )
    metrics, frames = transition.simulate_count_space_frames(
        model,
        arrivals,
        arrival_horizon=2.0,
        lp_node_budget=transition.REFERENCE_LP_BUDGET,
    )
    decomposition = decompose_flow(model, arrivals, frames, 2.0)
    assert metrics["job_count"] == metrics["completed_jobs"] == 4
    assert metrics["valid_certificate_frame_count"] == metrics["frame_count"]
    assert metrics["exact_selection_frame_count"] == metrics["frame_count"]
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert metrics["maximum_lp_calls_per_frame"] <= transition.REFERENCE_LP_BUDGET
    assert decomposition["mean_flow_s"] == pytest.approx(metrics["mean_flow_s"])


def test_registration_does_not_load_model_build_tapes_or_run_selectors(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "trajectory.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("registration performed trajectory work")

    monkeypatch.setattr(transition, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        transition,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "transition.py": "frozen"},
    )
    monkeypatch.setattr(transition, "load_model", forbidden)
    monkeypatch.setattr(transition, "capacity_certificate", forbidden)
    monkeypatch.setattr(transition, "poisson_arrivals", forbidden)
    monkeypatch.setattr(transition, "select_count_space_bnb_frame", forbidden)

    protocol = transition.register(output, tmp_path / "confirmation.json")
    result_path, protocol_path, fixed_path = transition._paths(output)
    assert not result_path.exists()
    assert protocol_path.exists() and fixed_path.exists()
    assert protocol["arrival_seeds"] == list(range(6041, 6049))
    assert protocol["loads"] == [0.8, 0.95]
    assert protocol["reference"]["lp_node_budget"] == 4096
    assert protocol["paired_mean_flow_noninferiority"]["acceptance_upper"] == 1.01


def test_run_rejects_protocol_mutation_before_model_or_tape_work(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "trajectory.json"
    monkeypatch.setattr(transition, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        transition,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "transition.py": "frozen"},
    )
    transition.register(output, tmp_path / "confirmation.json")
    _, protocol_path, _ = transition._paths(output)
    protocol = json.loads(protocol_path.read_text())
    protocol["arrival_horizon_s"] = 601.0
    protocol_path.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("execution work began before protocol validation")

    monkeypatch.setattr(transition, "load_model", forbidden)
    monkeypatch.setattr(transition, "poisson_arrivals", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        transition.run_registered(output)


def _synthetic_runs(ratio):
    rows = []
    for seed in transition.ARRIVAL_SEEDS:
        def policy(name, mean_flow, exact, epsilon):
            return {
                "policy": name,
                "metrics": {
                    "job_count": 100,
                    "completed_jobs": 100,
                    "mean_flow_s": mean_flow,
                    "frame_count": 10,
                    "valid_certificate_frame_count": 10,
                    "exact_selection_frame_count": 10 if exact else 8,
                    "maximum_certified_gap_per_job": epsilon,
                    "work": {"lp_calls": 90},
                },
                "flow_decomposition": {"status": "PASS", "mean_flow_s": mean_flow},
            }

        rows.append({
            "lanes": 8,
            "load": 0.8,
            "seed": seed,
            "arrival_tape": {"job_count": 100},
            "policies": [
                policy("fixed", 10.0 * ratio, False, 0.01),
                policy("exact_reference", 10.0, True, 0.0),
            ],
        })
    return rows


def test_paired_noninferiority_and_support_gates_are_separate():
    passing = transition._paired_case_summary(
        _synthetic_runs(1.005), 8, 0.8, eta=0.02
    )
    assert passing["paired_mean_flow_noninferiority_passed"]
    assert passing["trajectory_support_gate_passed"]
    assert passing["case_gate_passed"]

    slow = transition._paired_case_summary(
        _synthetic_runs(1.02), 8, 0.8, eta=0.02
    )
    assert not slow["paired_mean_flow_noninferiority_passed"]
    assert slow["trajectory_support_gate_passed"]
    assert not slow["case_gate_passed"]

    loose = transition._paired_case_summary(
        _synthetic_runs(1.005), 8, 0.8, eta=0.005
    )
    assert loose["paired_mean_flow_noninferiority_passed"]
    assert not loose["trajectory_support_gate_passed"]
    assert not loose["case_gate_passed"]
