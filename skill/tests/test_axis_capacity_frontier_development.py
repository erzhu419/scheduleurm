"""Borrow both frozen cohorts while executing only the new cold routers."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_axis_capacity_frontier_development as runner


@dataclass(frozen=True)
class Result:
    label: str
    complete: bool


def harness(monkeypatch):
    caps = [2, 3, 144, 256]
    saved = {}
    for name, queue, source in (("frontier_development", [54, 58, 290], "failure_development"),
                                ("trigger_validation", [120, 123, 258], "fresh_confirmation")):
        saved[name] = {"state_id": name, "queue": queue, "source": source, "caps": caps,
            **{key: {"borrowed": name, "field": key} for key in runner.REFERENCE_FIELDS},
            "traces": {label: {str(cap): {"tag": [label, cap]} for cap in caps} for label in ("old", "new")},
            "old_routed": {"tag": "prior old " + name}, "routed": {"tag": "prior new " + name}}
    states = [{"state_id": name, "queue": state["queue"], "source": state["source"], "caps": caps,
        "cohort": name, "borrowed_from": name, "borrowed_index": 0,
        "source_state_id": state["state_id"], "source_record_source": state["source"]}
        for name, state in saved.items()]
    model = SimpleNamespace(node="toy", lanes=2, options=(
        SimpleNamespace(requirements=(1, 0, 0), duration=1.), SimpleNamespace(requirements=(0, 1, 0), duration=2.)))
    protocol = {"registered_at_utc": "axis", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "borrowed_sources": {
            name: {"path": name + ".json", "registered_at_utc": name} for name in saved}}
    calls, loads = [], []

    def load(path, node, lanes, registration):
        assert (node, lanes) == ("toy", 64) and path == runner.ROOT / (registration + ".json")
        loads.append(registration)
        return {"states": [saved[registration]]}

    def selector(label):
        def select(options, queue, lanes, *budgets):
            assert options is model.options and (lanes, budgets) == (64, (144, 256, 8192))
            calls.append((label, queue))
            return Result(label, label == "old")
        return select

    def unexpected(*args, **kwargs):
        pytest.fail("borrowing must not re-execute references, native ladders or previous routers")

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    runtime = SimpleNamespace(_new_reference=unexpected, reference_key=unexpected,
        select_canonical_count_space_bnb_frame=unexpected)
    return model, {"node": "toy", "lanes": 64, "states": states}, protocol, runtime, {
        label: selector(label) for label in ("old", "new")}, saved, calls, loads


def test_both_source_cohorts_borrow_reference_and_native_work_without_hiding_incomplete_outputs(monkeypatch):
    model, cell, protocol, runtime, selectors, saved, calls, loads = harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert loads == ["frontier_development", "trigger_validation"]
    assert calls == [(label, tuple(state["queue"])) for state in cell["states"] for label in ("old", "new")]
    assert row["new_reference_calls"] == row["new_reference_oracle_calls"] == 0
    assert row["new_old_routed_calls"] == 2
    for state, original in zip(row["states"], saved.values()):
        assert all(state[key] == original[key] for key in (*runner.REFERENCE_FIELDS, "traces"))
        assert state["prior_routed"] == {"old": original["old_routed"], "new": original["routed"]}
        assert state["old_routed"] == {"label": "old", "complete": True}
        assert state["routed"] == {"label": "new", "complete": False}


@pytest.mark.parametrize("key,value", [("state_id", "wrong"), ("queue", [0, 0, 0]),
                                      ("source", "development"), ("caps", [2, 4, 144, 256])])
def test_wrong_borrowed_state_stops_before_either_cold_router(monkeypatch, key, value):
    model, cell, protocol, runtime, selectors, saved, calls, _ = harness(monkeypatch)
    saved["frontier_development"][key] = value
    with pytest.raises(ValueError, match="borrowed state"):
        runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert not calls


def test_protocol_retains_old_development_and_relabels_all_new_failure_controls(monkeypatch, tmp_path):
    options = (SimpleNamespace(requirements=(1, 0, 0), duration=1.),
               SimpleNamespace(requirements=(0, 1, 0), duration=2.))
    models = [SimpleNamespace(node=f"node{i}", options=options,
        snapshot=lambda i=i: {"node": f"node{i}", "catalogue": i}) for i in range(4)]
    caps = [2, 3, 144, 256]
    cells = [{"node": m.node, "lanes": 64, "states": [
        {"state_id": "history", "queue": [1, 0, 0], "source": "development", "cohort": "development",
         "caps": caps, "historical_regression": True},
        {"state_id": "failure", "queue": [54, 58, 290], "source": "failure_development", "cohort": "failure_development",
         "caps": caps, "historical_regression": False},
        {"state_id": "zero", "queue": [0, 0, 0], "source": "new_zero", "cohort": "zero_queue",
         "caps": caps, "historical_regression": False}]} for m in models]
    previous = {"registered_at_utc": "frontier", "models": [m.snapshot() for m in models], "cells": cells,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": 4096, "oracle_state_budget": 5000000, "execution_sources": {"frozen.py": "same\n"},
        "historical_regression_keys": [{"node": m.node, "lanes": 64, "queue": [1, 0, 0]} for m in models],
        "failure_development_keys": [{"node": m.node, "lanes": 64, "queue": [54, 58, 290]} for m in models],
        "expected_failure_development_count": 4}
    validation = {**previous, "registered_at_utc": "trigger", "cells": [
        {"node": m.node, "lanes": 64, "states": [{"state_id": "new_trigger", "queue": [120, 123, 258],
            "source": "fresh_confirmation", "cohort": "conditional_trigger_validation", "panel": "trigger_54", "seed": 54,
            "caps": caps, "historical_regression": False}]} for m in models]}
    dev = tmp_path / "dev.json"
    trigger = tmp_path / "trigger.json"
    dev.write_text(json.dumps({"decision": "FRONTIER_SCORE_PROOF_DEVELOPMENT_ADVANCE", "execution_checks_passed": True,
        "effective_references_complete": True, "development_gate_passed": True}))
    dev.with_name("dev_protocol.json").write_text(json.dumps(previous))
    trigger.write_text(json.dumps({"decision": "FRONTIER_TRIGGER_VALIDATION_HOLD", "routed_quality_gate_passed": False,
        "execution_checks_passed": True, "effective_references_complete": True,
        "prior_decisions": {"dual_reuse_confirmation": "DUAL_REUSE_FRESH_CONFIRMATION_HOLD"}}))
    trigger.with_name("trigger_protocol.json").write_text(json.dumps({"registered_at_utc": "discovery"}))
    trigger.with_name("trigger_validation_protocol.json").write_text(json.dumps(validation))
    (tmp_path / "frozen.py").write_text("same\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "DEVELOPMENT_OUTPUT", dev)
    monkeypatch.setattr(runner, "TRIGGER_OUTPUT", trigger)
    monkeypatch.setattr(runner, "SOURCES", ())
    protocol = runner.build_protocol(models)
    assert protocol["expected_state_count"] == 16 and protocol["expected_nonzero_state_count"] == 12
    assert protocol["expected_selector_calls"] == protocol["expected_interface_calls"] == 32
    assert protocol["expected_borrowed_native_snapshots"] == 128 and protocol["expected_native_advance_calls"] == 0
    assert protocol["expected_axis_failure_development_count"] == protocol["expected_failure_development_count"] == 4
    for registered, original in zip(protocol["cells"], cells):
        for index, (state, old) in enumerate(zip(registered["states"][:3], original["states"])):
            assert all(state[key] == value for key, value in old.items())
            assert (state["borrowed_from"], state["borrowed_index"]) == ("frontier_development", index)
        new = registered["states"][3]
        assert (new["state_id"], new["source_state_id"], new["source_record_source"], new["borrowed_index"]) == (
            "axis_failure:new_trigger", "new_trigger", "fresh_confirmation", 0)
        assert new["source"] == new["cohort"] == new["panel"] == "axis_failure_development"
        assert new["seed"] == 54 and not new["historical_regression"]
    assert protocol["prior_decisions"]["frontier_trigger_discovery"] == "FRONTIER_TRIGGER_VALIDATION_HOLD"
    runner.verify_protocol({**protocol, "registered_at_utc": "axis"}, models)
    (tmp_path / "frozen.py").write_text("changed\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.build_protocol(models)
