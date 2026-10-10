"""The new experiment borrows the full prior roster and executes only two arms."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_residual_capacity_frontier_development as runner


@dataclass(frozen=True)
class Result:
    label: str
    complete: bool


def harness(monkeypatch):
    caps = [2, 3, 144, 256]
    saved = [{"state_id": name, "queue": queue, "source": source, "caps": caps,
        **{key: {"borrowed": name, "field": key} for key in runner.REFERENCE_FIELDS},
        "traces": {label: {str(cap): {"tag": [label, cap]} for cap in caps} for label in ("old", "new")},
        "old_routed": {"tag": "prior old " + name}, "routed": {"tag": "prior new " + name}}
        for name, queue, source in (("history", [1, 0, 0], "development"),
                                   ("axis_failure:trigger", [120, 123, 258], "axis_failure_development"))]
    states = [{"state_id": state["state_id"], "queue": state["queue"], "source": state["source"],
        "caps": caps, "cohort": state["source"], "borrowed_from": "axis_capacity_development",
        "borrowed_index": index, "source_state_id": state["state_id"], "source_record_source": state["source"]}
        for index, state in enumerate(saved)]
    model = SimpleNamespace(node="toy", lanes=2, options=(
        SimpleNamespace(requirements=(1, 0, 0), duration=1.),
        SimpleNamespace(requirements=(0, 1, 0), duration=2.)))
    protocol = {"registered_at_utc": "residual", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "borrowed_sources": {"axis_capacity_development": {
            "path": "axis.json", "registered_at_utc": "axis"}}}
    calls, loads = [], []

    def load(path, node, lanes, registration):
        assert (path, node, lanes, registration) == (runner.ROOT / "axis.json", "toy", 64, "axis")
        loads.append(registration)
        return {"states": saved}

    def selector(label):
        def select(options, queue, lanes, *budgets):
            assert options is model.options and (lanes, budgets) == (64, (144, 256, 8192))
            calls.append((label, queue))
            return Result(label, label == "old")
        return select

    def unexpected(*args, **kwargs):
        pytest.fail("borrowing must not execute references, native ladders or prior routers")

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    runtime = SimpleNamespace(_new_reference=unexpected, reference_key=unexpected,
        select_canonical_count_space_bnb_frame=unexpected)
    return model, {"node": "toy", "lanes": 64, "states": states}, protocol, runtime, {
        label: selector(label) for label in ("old", "new")}, saved, calls, loads


def test_borrowed_reference_native_and_prior_outputs_survive_without_reexecution(monkeypatch):
    model, cell, protocol, runtime, selectors, saved, calls, loads = harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert loads == ["axis"]
    assert calls == [(label, tuple(state["queue"])) for state in cell["states"] for label in ("old", "new")]
    assert row["new_reference_calls"] == row["new_reference_oracle_calls"] == 0
    assert row["new_old_routed_calls"] == 2
    for state, original in zip(row["states"], saved):
        assert all(state[key] == original[key] for key in (*runner.REFERENCE_FIELDS, "traces"))
        assert state["prior_routed"] == {"old": original["old_routed"], "new": original["routed"]}
        assert state["old_routed"] == {"label": "old", "complete": True}
        assert state["routed"] == {"label": "new", "complete": False}


@pytest.mark.parametrize("key,value", [("state_id", "wrong"), ("queue", [0, 0, 0]),
                                      ("source", "wrong"), ("caps", [2, 4, 144, 256])])
def test_wrong_prior_state_stops_before_both_cold_calls(monkeypatch, key, value):
    model, cell, protocol, runtime, selectors, saved, calls, _ = harness(monkeypatch)
    saved[0][key] = value
    with pytest.raises(ValueError, match="borrowed state"):
        runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert not calls


def test_wrong_prior_source_stops_before_both_cold_calls(monkeypatch):
    model, cell, protocol, runtime, selectors, _, calls, _ = harness(monkeypatch)
    cell["states"][0]["borrowed_from"] = "trigger_validation"
    with pytest.raises(ValueError, match="borrowed state"):
        runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert not calls


def protocol_fixture(monkeypatch, tmp_path):
    options = (SimpleNamespace(requirements=(1, 0, 0), duration=1.),
               SimpleNamespace(requirements=(0, 1, 0), duration=2.))
    models = [SimpleNamespace(node=f"node{i}", options=options,
        snapshot=lambda i=i: {"node": f"node{i}", "catalogue": i}) for i in range(4)]
    caps = [2, 3, 144, 256]
    cells = [{"node": m.node, "lanes": 64, "states": [
        {"state_id": "history", "queue": [1, 0, 0], "source": "development", "cohort": "development",
         "caps": caps, "historical_regression": True, "borrowed_from": "older", "borrowed_index": 9},
        {"state_id": "axis_failure:trigger", "queue": [120, 123, 258], "source": "axis_failure_development",
         "cohort": "axis_failure_development", "seed": 54, "caps": caps, "historical_regression": False,
         "borrowed_from": "trigger_validation", "borrowed_index": 0}]} for m in models]
    previous = {"registered_at_utc": "axis", "models": [m.snapshot() for m in models], "cells": cells,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": 4096, "oracle_state_budget": 5000000, "execution_sources": {"frozen.py": "same\n"},
        "historical_regression_keys": [{"node": m.node, "lanes": 64, "queue": [1, 0, 0]} for m in models],
        "failure_development_keys": [], "axis_failure_development_keys": [
            {"node": m.node, "lanes": 64, "queue": [120, 123, 258]} for m in models],
        "expected_state_count": 8, "expected_nonzero_state_count": 8,
        "expected_failure_development_count": 0, "expected_axis_failure_development_count": 4}
    result = {"decision": "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD", "execution_checks_passed": True,
        "effective_references_complete": True, "predecessor_work_gate_passed": True,
        "routed_quality_gate_passed": False, "state_count": 8,
        "prior_decisions": {"frontier_trigger_discovery": "FRONTIER_TRIGGER_VALIDATION_HOLD"}}
    path = tmp_path / "axis.json"
    path.write_text(json.dumps(result))
    path.with_name("axis_protocol.json").write_text(json.dumps(previous))
    (tmp_path / "frozen.py").write_text("same\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PREVIOUS_OUTPUT", path)
    monkeypatch.setattr(runner, "SOURCES", ())
    return models, previous, result, path


def test_protocol_preserves_every_prior_state_seed_control_and_hold(monkeypatch, tmp_path):
    models, previous, _, _ = protocol_fixture(monkeypatch, tmp_path)
    protocol = runner.build_protocol(models)
    assert protocol["expected_state_count"] == protocol["expected_nonzero_state_count"] == 8
    assert protocol["expected_selector_calls"] == protocol["expected_interface_calls"] == 16
    assert protocol["expected_borrowed_native_snapshots"] == 64 and protocol["expected_native_advance_calls"] == 0
    assert protocol["axis_failure_development_keys"] == previous["axis_failure_development_keys"]
    assert protocol["prior_decisions"] == {
        "frontier_trigger_discovery": "FRONTIER_TRIGGER_VALIDATION_HOLD",
        "axis_capacity_development": "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"}
    for cell, old_cell in zip(protocol["cells"], previous["cells"]):
        for index, (state, old) in enumerate(zip(cell["states"], old_cell["states"])):
            assert all(state[key] == value for key, value in old.items() if key not in ("borrowed_from", "borrowed_index"))
            assert (state["borrowed_from"], state["borrowed_index"], state["source_state_id"], state["source_record_source"]) == (
                "axis_capacity_development", index, old["state_id"], old["source"])
    runner.verify_protocol({**protocol, "registered_at_utc": "residual"}, models)
    (tmp_path / "frozen.py").write_text("changed\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.build_protocol(models)


def test_prior_result_cannot_omit_a_registered_state(monkeypatch, tmp_path):
    models, _, result, path = protocol_fixture(monkeypatch, tmp_path)
    result["state_count"] = 7
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="complete prior development roster"):
        runner.build_protocol(models)
