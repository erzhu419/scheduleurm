"""New arrivals must confirm the frozen method, without tuning its clock or gates."""
from collections import Counter
from dataclasses import replace
import json
import random
import sys
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_small_dimension_exact_native_common_cadence_confirmation as experiment
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from skill.tests.test_small_dimension_exact_native_common_cadence import previous_replay
from skill.tests.test_small_dimension_exact_native_decision_delay import registered_inputs


@pytest.fixture
def completed_development(previous_replay, monkeypatch):
    models, previous, _ = previous_replay
    development = experiment.development
    previous["capacities"] = [
        {"node": model.node, "lanes": 8, "load": load,
         "certificate": {"arrival_rates": [.012 * load, .008 * load]}}
        for model in models for load in previous["loads"]
    ]
    development._protocol_path(development.PREVIOUS_OUTPUT).write_text(json.dumps(previous))
    protocol = development.build_protocol(models)
    protocol["registered_at_utc"] = "development frozen before outcomes"
    result = {key: True for key in (
        "integration_gate_passed", "runtime_checks_passed", "no_overrun_gate_passed",
        "trajectory_identity_passed", "timing_gate_passed", "common_cadence_gate_passed")}
    result["decision"] = "SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_TIMING_ADVANCE"
    monkeypatch.setattr(experiment, "ROOT", development.ROOT)
    monkeypatch.setattr(experiment, "PREVIOUS_OUTPUT", development.ROOT / "common_development.json")
    monkeypatch.setattr(experiment, "SOURCE", "confirmation_driver.py")
    (experiment.ROOT / experiment.SOURCE).write_text("frozen confirmation driver\n")
    experiment.PREVIOUS_OUTPUT.write_text(json.dumps(result))
    development._protocol_path(experiment.PREVIOUS_OUTPUT).write_text(json.dumps(protocol))
    return models, protocol, result


def test_confirmation_keeps_method_window_measurement_and_historical_hold(completed_development):
    models, previous, result = completed_development
    protocol = experiment.build_protocol(models)
    for key in (
        "models", "capacities", "lanes", "loads", "arrival_horizon", "observation_end",
        "initial_lp_budget", "lp_node_budget", "integer_state_budget", "reference_budget",
        "oracle_state_budget", "budgets", "labels", "repeats", "computation_window_s",
        "maximum_pooled_wall_ratio", "maximum_group_wall_ratio", "cpu", "execution_order",
        "float_relative_tolerance", "expected_cost_group_count", "expected_speed_gate_count",
        "window_origin", "clock", "reference", "measurement", "integration_gate", "cost_gate", "cohort",
    ):
        assert protocol[key] == previous[key]
    assert [protocol[key] for key in (
        "initial_lp_budget", "lp_node_budget", "integer_state_budget", "reference_budget", "oracle_state_budget",
    )] == [144, 256, 8192, 4096, 5_000_000]
    assert protocol["computation_window_s"] == 1.0
    assert (protocol["maximum_pooled_wall_ratio"], protocol["maximum_group_wall_ratio"]) == (.95, 1.05)
    assert protocol["expected_speed_gate_count"] == 9
    assert protocol["prior_decisions"]["small_dimension_exact_native_decision_delay"].endswith("UTILITY_HOLD")
    for key, decision in previous["prior_decisions"].items():
        assert protocol["prior_decisions"][key] == decision
    assert protocol["prior_decisions"]["small_dimension_exact_native_common_cadence_development"] == result["decision"]
    assert protocol["execution_sources"] == {
        **previous["execution_sources"], experiment.SOURCE: (experiment.ROOT / experiment.SOURCE).read_text(),
    }
    assert "registered_at_utc" not in protocol


def test_new_seed_roster_generates_all_64_literal_poisson_tapes(completed_development):
    models, previous, _ = completed_development
    protocol = experiment.build_protocol(models)
    assert protocol["arrival_seeds"] == list(range(13051, 13059))
    assert not set(protocol["arrival_seeds"]).intersection(previous["arrival_seeds"])
    assert (protocol["expected_tape_count"], protocol["expected_reference_trajectory_count"],
            protocol["expected_measured_trajectory_count"], protocol["expected_trajectory_count"]) == (64, 64, 768, 832)
    expected_cells = {(model.node, 8, load, seed)
                      for model in models for load in previous["loads"] for seed in experiment.ARRIVAL_SEEDS}
    assert {(tape["node"], tape["lanes"], tape["load"], tape["seed"])
            for tape in protocol["tapes"]} == expected_cells
    assert len(protocol["tapes"]) == len({tape["tape_id"] for tape in protocol["tapes"]}) == 64
    for tape in protocol["tapes"]:
        rates = next(entry["certificate"]["arrival_rates"] for entry in previous["capacities"]
                     if (entry["node"], entry["lanes"], entry["load"]) ==
                     (tape["node"], tape["lanes"], tape["load"]))
        generated = poisson_arrivals(rates, previous["arrival_horizon"], tape["seed"])
        assert tape["arrivals"] == json.loads(json.dumps(generated))
        assert tape["arrivals"] != previous["tapes"][0]["arrivals"]
    assert protocol["tapes"][0]["arrivals"] != protocol["tapes"][1]["arrivals"]


def test_all_tapes_keep_six_deterministic_balanced_orders(completed_development):
    models, _, _ = completed_development
    protocol = experiment.build_protocol(models)
    assert protocol["order_seed"] == 2026092917
    assert protocol["timing_orders"] == experiment.build_protocol(models)["timing_orders"]
    assert set(protocol["timing_orders"]) == {tape["tape_id"] for tape in protocol["tapes"]}
    for tape_id, orders in protocol["timing_orders"].items():
        assert Counter(map(tuple, orders)) == {("candidate", "direct"): 3, ("direct", "candidate"): 3}
        expected = [["candidate", "direct"], ["direct", "candidate"]] * 3
        random.Random(f"2026092917:{tape_id}").shuffle(expected)
        assert orders == expected


@pytest.mark.parametrize("failure", ("decision", "integration"))
def test_failed_development_cannot_be_promoted(completed_development, failure):
    models, _, result = completed_development
    if failure == "decision":
        result["decision"] = "SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_TIMING_HOLD"
    else:
        result["integration_gate_passed"] = False
    experiment.PREVIOUS_OUTPUT.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="development advance"):
        experiment.build_protocol(models)


def test_changed_development_source_blocks_new_registration(completed_development):
    models, _, _ = completed_development
    (experiment.ROOT / "whole_selector.py").write_text("retuned native source\n")
    with pytest.raises(ValueError, match="frozen development source changed"):
        experiment.build_protocol(models)


def test_changed_catalogue_blocks_new_registration(completed_development):
    models, _, _ = completed_development
    changed = [replace(models[0], lanes=3), *models[1:]]
    with pytest.raises(ValueError, match="same cached complete-frame catalogues"):
        experiment.build_protocol(changed)


def test_reusing_a_development_seed_blocks_new_registration(completed_development, monkeypatch):
    models, previous, _ = completed_development
    monkeypatch.setattr(experiment, "ARRIVAL_SEEDS", (previous["arrival_seeds"][0], *experiment.ARRIVAL_SEEDS[1:]))
    with pytest.raises(ValueError, match="seeds must be new"):
        experiment.build_protocol(models)


@pytest.mark.parametrize("drift", ("arrival", "order", "gate"))
def test_registered_drift_is_rejected_before_any_cell_runs(completed_development, monkeypatch, drift):
    models, _, _ = completed_development
    protocol = experiment.build_protocol(models)
    protocol["registered_at_utc"] = "before confirmation outcomes"
    experiment.verify_protocol(protocol, models)
    if drift == "arrival":
        protocol["tapes"][0]["arrivals"][0][0] += .5
    elif drift == "order":
        tape_id = protocol["tapes"][0]["tape_id"]
        protocol["timing_orders"][tape_id][0].reverse()
    else:
        protocol["maximum_pooled_wall_ratio"] = 1.0
    output = experiment.ROOT / "confirmation.json"
    experiment.development._protocol_path(output).write_text(json.dumps(protocol))
    monkeypatch.setattr(experiment, "runtime_setup", lambda: SimpleNamespace(load_catalogues=lambda: models))
    ran = []
    monkeypatch.setattr(experiment.development, "run_cell", lambda *args: ran.append(args))
    monkeypatch.setattr(sys, "argv", ["confirmation", "--run", "--output", str(output)])
    with pytest.raises(ValueError, match="registered confirmation"):
        experiment.main()
    assert ran == []
    assert not output.exists()
