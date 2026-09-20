from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import (
    aligned_hardware_count_space_bnb_exactness_escalation_development as escalation,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_model import load_model


def _stub_source():
    return (
        {
            "trajectory_result_path": "/tmp/stub-transition.json",
            "trajectory_experiment": escalation.SOURCE_EXPERIMENT,
            "trajectory_decision": escalation.SOURCE_DECISION,
            "trajectory_advance": False,
            "prior_case_gates": [],
        },
        {"lanes": 2, "options": []},
        {"upstream.py": "frozen"},
    )


def test_budget_schedules_double_and_end_at_hard_cap():
    assert escalation._budget_schedule(36) == (
        36, 72, 144, 288, 576, 1152, 2304, 4096
    )
    assert escalation._budget_schedule(72) == (
        72, 144, 288, 576, 1152, 2304, 4096
    )
    assert escalation._budget_schedule(4096) == (4096,)


def test_fresh_root_retries_charge_every_actual_attempt(monkeypatch):
    calls = []

    def selection(budget, exact, lp_calls, matrix_cells):
        values = {field: 0 for field in escalation.WORK_FIELDS}
        values.update({
            "lp_calls": lp_calls,
            "lp_matrix_coefficients": matrix_cells,
            "candidate_evaluations": lp_calls + 1,
            "exact": exact,
        })
        return SimpleNamespace(**values)

    returned = {
        36: selection(36, False, 36, 360),
        72: selection(72, True, 50, 500),
    }

    def fake_selector(options, queue, lanes, budget):
        calls.append(budget)
        return returned[budget]

    monkeypatch.setattr(escalation, "select_count_space_bnb_frame", fake_selector)
    monkeypatch.setattr(
        escalation, "_selection_certificate_valid", lambda *args: True
    )
    selected, attempts, charged = escalation._select_exact_with_restarts(
        SimpleNamespace(options=(), lanes=8), (4, 3, 2), 36
    )
    assert selected is returned[72]
    assert calls == [36, 72]
    assert [row["lp_node_budget"] for row in attempts] == [36, 72]
    assert charged["lp_calls"] == 86
    assert charged["lp_matrix_coefficients"] == 860
    assert charged["candidate_evaluations"] == 88


def test_small_simulator_finishes_exact_and_reconstructs_fifo_flow():
    model = replace(load_model(), lanes=8)
    arrivals = (
        (0.10, 0, 0),
        (0.20, 1, 0),
        (0.30, 2, 0),
        (0.40, 2, 1),
    )
    metrics, frames, flows = escalation.simulate_exactness_escalation_frames(
        model,
        arrivals,
        arrival_horizon=2.0,
        start_budget=36,
    )
    decomposition = decompose_flow(model, arrivals, frames, 2.0)
    assert metrics["job_count"] == metrics["completed_jobs"] == len(flows) == 4
    assert metrics["exact_selection_frame_count"] == metrics["frame_count"]
    assert (
        metrics["valid_certificate_attempt_count"]
        == metrics["selector_attempt_count"]
    )
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert decomposition["mean_flow_s"] == pytest.approx(metrics["mean_flow_s"])


def test_registration_does_no_model_capacity_tape_or_selector_work(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "escalation.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("registration performed execution work")

    monkeypatch.setattr(escalation, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        escalation,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "escalation.py": "frozen"},
    )
    monkeypatch.setattr(escalation, "load_model", forbidden)
    monkeypatch.setattr(escalation, "capacity_certificate", forbidden)
    monkeypatch.setattr(escalation, "poisson_arrivals", forbidden)
    monkeypatch.setattr(escalation, "select_count_space_bnb_frame", forbidden)

    protocol = escalation.register(output, tmp_path / "transition.json")
    result_path, protocol_path, fixed_path = escalation._paths(output)
    assert not result_path.exists()
    assert protocol_path.exists() and fixed_path.exists()
    assert protocol["arrival_seeds"] == list(range(7041, 7049))
    assert protocol["loads"] == [0.8, 0.95]
    assert protocol["lane_start_budgets"][0]["restart_budget_schedule"][-1] == 4096
    assert "no frontier or work is reused" in protocol["escalation_rule"]["restart"]


def test_run_rejects_protocol_change_before_model_or_tape_work(tmp_path, monkeypatch):
    output = tmp_path / "escalation.json"
    monkeypatch.setattr(escalation, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        escalation,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "escalation.py": "frozen"},
    )
    escalation.register(output, tmp_path / "transition.json")
    _, protocol_path, _ = escalation._paths(output)
    protocol = json.loads(protocol_path.read_text())
    protocol["arrival_horizon_s"] = 601.0
    protocol_path.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("execution began before protocol validation")

    monkeypatch.setattr(escalation, "load_model", forbidden)
    monkeypatch.setattr(escalation, "poisson_arrivals", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        escalation.run_registered(output)


def test_decision_requires_each_case_and_aggregate_strict_work_gates():
    passing_cases = [{"case_gate_passed": True} for _ in range(4)]
    assert (
        escalation._decision(True, passing_cases, True)
        == "EXACTNESS_ESCALATION_PASS"
    )
    assert (
        escalation._decision(True, passing_cases, False)
        == "EXACTNESS_ESCALATION_WORK_FAIL_STOP"
    )
    failing_cases = [*passing_cases[:3], {"case_gate_passed": False}]
    assert (
        escalation._decision(True, failing_cases, True)
        == "EXACTNESS_ESCALATION_WORK_FAIL_STOP"
    )
    assert escalation._decision(False, passing_cases, True) == "INVALID_EXECUTION"
