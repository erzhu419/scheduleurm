from __future__ import annotations

import json

import pytest

from algorithm.experiments import (
    aligned_hardware_count_space_bnb_confirmation as confirmation,
)


def _stub_source():
    return (
        {
            "development_result_path": "/tmp/stub-development.json",
            "development_experiment": confirmation.DEVELOPMENT_EXPERIMENT,
            "development_decision": confirmation.DEVELOPMENT_DECISION,
            "primary_development_rows": {},
            "inherited_r2_oracle_evidence": {
                "oracle_calls": 1000,
                "ordered_action_evaluations": 100000,
                "checkpoint_inequality_comparisons": 4000,
                "score_optimum_upper_check": True,
            },
        },
        {"lanes": 2, "options": []},
        {"upstream.py": "frozen"},
    )


def test_fixed_roster_is_deterministic_unique_and_disjoint():
    first = confirmation.generate_roster()
    second = confirmation.generate_roster()
    assert first == second
    queues = tuple(tuple(queue) for queue in first["queues"])
    assert first["seed"] == 260913
    assert first["candidate_draws"] == 530
    assert len(queues) == len(set(queues)) == 512
    assert queues[:3] == ((30, 6, 8), (3, 27, 11), (31, 11, 0))
    assert (0, 0, 0) not in queues
    assert not set(queues) & confirmation._old_base_grid()


def test_registration_only_generates_and_writes_the_roster(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "confirmation.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("registration performed execution work")

    monkeypatch.setattr(confirmation, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        confirmation,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "confirmation.py": "frozen"},
    )
    monkeypatch.setattr(confirmation, "trace_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(confirmation, "capacity_certificate", forbidden)
    monkeypatch.setattr(confirmation, "load_model", forbidden)

    protocol = confirmation.register(output, tmp_path / "development.json")
    result_path, protocol_path, fixed_path, roster_path = confirmation._paths(output)
    assert not result_path.exists()
    assert protocol_path.exists() and fixed_path.exists() and roster_path.exists()
    roster = json.loads(roster_path.read_text())
    assert roster["queues"] == protocol["registered_base_queues"]
    assert protocol["primary_budget_by_lane"] == {"8": 36, "16": 72}
    assert protocol["exploratory_lane_counts"] == [32, 64]
    assert protocol["lp_checkpoint_budgets"] == [9, 18, 36, 72, 144, 288]


def test_run_rejects_roster_mutation_before_loading_model(tmp_path, monkeypatch):
    output = tmp_path / "confirmation.json"
    monkeypatch.setattr(confirmation, "_source_contract", lambda path: _stub_source())
    monkeypatch.setattr(
        confirmation,
        "_current_fixed_sources",
        lambda upstream: {"upstream.py": "frozen", "confirmation.py": "frozen"},
    )
    confirmation.register(output, tmp_path / "development.json")
    _, _, _, roster_path = confirmation._paths(output)
    roster = json.loads(roster_path.read_text())
    roster["queues"][0] = roster["queues"][1]
    roster_path.write_text(json.dumps(roster))

    def forbidden(*args, **kwargs):
        raise AssertionError("model loaded before roster validation")

    monkeypatch.setattr(confirmation, "load_model", forbidden)
    with pytest.raises(ValueError, match="roster changed or is invalid"):
        confirmation.run_registered(output)


def test_primary_decision_ignores_exploratory_rows():
    cases = {
        "8": {"primary_gate_passed": True},
        "16": {"primary_gate_passed": True},
    }
    assert confirmation._primary_decision(True, cases) == (
        "PRIMARY_LARGE_ACTION_CONFIRMATION_PASS"
    )
    cases["16"]["primary_gate_passed"] = False
    assert confirmation._primary_decision(True, cases) == "PRIMARY_R8_ONLY_PASS_STOP"
    cases["8"]["primary_gate_passed"] = False
    cases["16"]["primary_gate_passed"] = True
    assert confirmation._primary_decision(True, cases) == "PRIMARY_R16_ONLY_PASS_STOP"
    assert confirmation._primary_decision(False, cases) == "INVALID_EXECUTION"
