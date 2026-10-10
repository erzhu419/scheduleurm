"""Freeze late-proof evidence while executing two frontier-proof routers."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_frontier_score_proof_development as runner


@dataclass(frozen=True)
class Evidence:
    retained_regions: int
    complete: bool


@dataclass(frozen=True)
class Result:
    label: str
    canonical_exact: bool
    frontier_evidence: Evidence | None


def harness(monkeypatch):
    caps = [2, 3, 144, 256]
    saved = []
    for index, (queue, source) in enumerate((([1, 0, 0], "development"),
        ([54, 58, 290], "failure_development"), ([0, 0, 0], "new_zero"))):
        saved.append({"state_id": f"state{index}", "queue": queue, "source": source, "cohort": source, "caps": caps,
            "reference": {"independent": index}, "reference_metrics": {"borrowed": index},
            "reference_complete": index == 2, "oracle": None if index == 2 else {"complete": True},
            "effective_reference_complete": True,
            "effective_reference_source": "full_K4096" if index == 2 else "independent_score_face_oracle",
            "traces": {label: {str(cap): {"tag": f"{label} native {index}", "cap": cap} for cap in caps}
                       for label in ("old", "new")},
            "original_routed": {label: {"tag": f"original {label} {index}"} for label in ("old", "new")},
            "old_routed": {"tag": f"late old {index}"}, "routed": {"tag": f"late new {index}"}})
    states = [{"state_id": state["state_id"], "queue": state["queue"], "source": state["source"],
        "cohort": state["cohort"], "caps": caps, "borrowed_from": "late_development", "borrowed_index": index,
        "source_state_id": state["state_id"], "source_record_source": state["source"]}
        for index, state in enumerate(saved)]
    model = SimpleNamespace(node="toy", lanes=2, options=(SimpleNamespace(requirements=(1, 0, 0), duration=1.0),
        SimpleNamespace(requirements=(0, 1, 0), duration=2.0)))
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192,
        "borrowed_sources": {"late_development": {"path": "late.json", "registered_at_utc": "late registered"}}}
    calls = []

    def selector(label):
        def run(options, queue, lanes, *budgets):
            assert options is model.options and lanes == 64 and budgets == (144, 256, 8192)
            calls.append((label, queue))
            complete = label == "old" or not any(queue)
            return Result(label, complete, Evidence(2, complete) if any(queue) else None)
        return run

    def load(path, node, lanes, registration):
        assert (path, node, lanes, registration) == (runner.ROOT / "late.json", "toy", 64, "late registered")
        return {"states": saved}

    def unexpected(*args, **kwargs):
        pytest.fail("borrowed references, oracles, native ladders and prior routers must not be re-executed")

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    runtime = SimpleNamespace(_new_reference=unexpected, _record=unexpected, reference_key=unexpected,
        select_canonical_count_space_bnb_frame=unexpected)
    return model, {"node": "toy", "lanes": 64, "states": states}, protocol, runtime, {
        label: selector(label) for label in ("old", "new")}, saved, calls


def test_references_native_traces_and_both_prior_router_generations_remain_separate(monkeypatch):
    model, cell, protocol, runtime, selectors, saved, calls = harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert calls == [(label, tuple(state["queue"])) for state in cell["states"] for label in ("old", "new")]
    assert row["new_reference_calls"] == row["new_reference_oracle_calls"] == 0
    assert row["new_old_routed_calls"] == 3
    for state, prior in zip(row["states"], saved):
        for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                    "effective_reference_complete", "effective_reference_source", "traces", "original_routed"):
            assert state[key] == prior[key]
        assert state["prior_routed"] == {"old": prior["old_routed"], "new": prior["routed"]}
        assert state["old_routed"]["label"] == "old" and state["routed"]["label"] == "new"
    assert [state["routed"]["canonical_exact"] for state in row["states"]] == [False, False, True]
    assert row["states"][1]["routed"]["frontier_evidence"] == {"retained_regions": 2, "complete": False}
    assert row["states"][2]["routed"]["frontier_evidence"] is None


@pytest.mark.parametrize("changed,value", (("queue", [9, 9, 9]), ("caps", [2, 4, 144, 256]),
    ("source", "fresh_confirmation"), ("state_id", "different state"), ("borrowed_index", 1)))
def test_wrong_borrowed_record_blocks_all_router_calls(monkeypatch, changed, value):
    model, cell, protocol, runtime, selectors, saved, calls = harness(monkeypatch)
    if changed == "borrowed_index":
        cell["states"][0][changed] = value
    else:
        saved[0][changed] = value
    with pytest.raises(ValueError):
        runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert not calls


def test_registration_preserves_roster_holds_and_frozen_execution_sources(monkeypatch, tmp_path):
    options = (SimpleNamespace(requirements=(1, 0, 0), duration=1.0),
               SimpleNamespace(requirements=(0, 1, 0), duration=2.0))
    models = [SimpleNamespace(node=f"node{index}", options=options,
        snapshot=lambda index=index: {"node": f"node{index}", "catalogue": index}) for index in range(4)]
    cells = [{"node": model.node, "lanes": 64, "states": [
        {"state_id": "historical", "queue": [1, 0, 0], "source": "development", "cohort": "development",
         "seed": 27, "caps": [2, 3, 144, 256], "historical_regression": True},
        {"state_id": "failure_development:held", "queue": [54, 58, 290], "source": "failure_development",
         "cohort": "failure_development", "seed": 2026092831, "caps": [2, 3, 144, 256],
         "historical_regression": False},
        {"state_id": "zero", "queue": [0, 0, 0], "source": "new_zero", "cohort": "zero_queue",
         "seed": None, "caps": [2, 3, 144, 256], "historical_regression": False}]} for model in models]
    previous = {"registered_at_utc": "late registered", "models": [model.snapshot() for model in models],
        "cells": cells, "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "execution_sources": {"frozen.py": "original source\n"},
        "historical_regression_keys": [[model.node, 64, [1, 0, 0]] for model in models],
        "failure_development_keys": [[model.node, 64, [54, 58, 290]] for model in models],
        "expected_failure_development_count": 4,
        "prior_decisions": {"dual_reuse_confirmation": "DUAL_REUSE_CONFIRMATION_HOLD",
                            "branch_reserved_timing": "BRANCH_RESERVED_TIMING_HOLD"}}
    prior = {"decision": "LATE_SCORE_PROOF_DEVELOPMENT_HOLD", "execution_checks_passed": True,
        "effective_references_complete": True, "final_score_gate_passed": True,
        "work_gate_passed": True, "routed_quality_gate_passed": False}
    output = tmp_path / "late.json"
    output.write_text(json.dumps(prior))
    output.with_name("late_protocol.json").write_text(json.dumps(previous))
    (tmp_path / "frozen.py").write_text("original source\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_OUTPUT", output)
    monkeypatch.setattr(runner, "SOURCES", ())

    protocol = runner.build_protocol(models)
    assert protocol["borrowed_sources"] == {
        "late_development": {"path": "late.json", "registered_at_utc": "late registered"}}
    for registered, original in zip(protocol["cells"], cells):
        assert (registered["node"], registered["lanes"]) == (original["node"], original["lanes"])
        for index, (state, old) in enumerate(zip(registered["states"], original["states"])):
            assert {key: state[key] for key in old} == old
            assert state["borrowed_from"] == "late_development" and state["borrowed_index"] == index
            assert state["source_state_id"] == old["state_id"] and state["source_record_source"] == old["source"]
            assert "borrowed_index" not in old
    assert protocol["historical_regression_keys"] == previous["historical_regression_keys"]
    assert protocol["failure_development_keys"] == previous["failure_development_keys"]
    assert protocol["prior_decisions"] == {**previous["prior_decisions"],
        "late_score_proof_development": "LATE_SCORE_PROOF_DEVELOPMENT_HOLD"}
    assert protocol["execution_sources"] == previous["execution_sources"]
    assert (protocol["expected_state_count"], protocol["expected_nonzero_state_count"],
            protocol["expected_failure_development_count"]) == (12, 8, 4)
    assert protocol["expected_selector_calls"] == protocol["expected_native_constructions"] == 24
    assert protocol["expected_native_advance_calls"] == 0
    assert protocol["expected_borrowed_native_snapshots"] == 96
    runner.verify_protocol({**protocol, "registered_at_utc": "new registration"}, models)
    (tmp_path / "frozen.py").write_text("changed source\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.build_protocol(models)
