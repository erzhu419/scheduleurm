"""Borrow both native ladders and independent references; execute only routers."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_late_score_proof_development as runner


@dataclass(frozen=True)
class Result:
    label: str
    canonical_exact: bool


def harness(monkeypatch):
    caps = [2, 3, 144, 256]

    def saved(state_id, queue, source, label, confirmation=False):
        branch = {str(cap): {"label": label + " branch", "cap": cap} for cap in caps}
        dual = {str(cap): {"label": label + " dual", "cap": cap} for cap in caps}
        return {"state_id": state_id, "queue": queue, "source": source, "caps": caps,
            "reference": {"label": label + " reference"}, "reference_metrics": {"label": label + " reference metrics"},
            "reference_complete": False, "oracle": {"complete": True, "label": label + " oracle"},
            "effective_reference_complete": True, "effective_reference_source": "independent_score_face_oracle",
            "old_routed": {"label": label + " original branch route"}, "routed": {"label": label + " original dual route"},
            **({"traces": {"old": branch, "new": dual}} if confirmation else {"old_traces": branch, "traces": dual})}

    development = [saved("dev0", [1, 0, 0], "development", "development"),
                   saved("zero", [0, 0, 0], "new_zero", "zero")]
    confirmation = [saved("fresh0", [54, 58, 290], "fresh_confirmation", "confirmation", True)]
    states = [{"state_id": state["state_id"], "queue": state["queue"], "source": state["source"],
        "cohort": state["source"], "borrowed_from": "development", "borrowed_index": index,
        "source_state_id": state["state_id"], "source_record_source": state["source"], "caps": caps}
        for index, state in enumerate(development)]
    states.append({"state_id": "failure_development:fresh0", "queue": [54, 58, 290], "source": "failure_development",
        "cohort": "failure_development", "borrowed_from": "confirmation", "borrowed_index": 0,
        "source_state_id": "fresh0", "source_record_source": "fresh_confirmation", "caps": caps})
    model = SimpleNamespace(node="toy", lanes=2, options=(SimpleNamespace(requirements=(1, 0, 0), duration=1.0),
        SimpleNamespace(requirements=(0, 1, 0), duration=2.0)))
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "borrowed_sources": {"development": {"path": "development.json", "registered_at_utc": "development"},
                             "confirmation": {"path": "confirmation.json", "registered_at_utc": "confirmation"}}}
    calls = []

    def selector(label):
        def run(options, queue, lanes, *budgets):
            assert options is model.options and lanes == 64 and budgets == (144, 256, 8192)
            calls.append((label, queue))
            return Result(label, label == "old" or not any(queue))
        return run

    def load(path, node, lanes, registration):
        assert (node, lanes) == ("toy", 64)
        assert path == runner.ROOT / (registration + ".json")
        return {"states": development if registration == "development" else confirmation}

    def unexpected(*args, **kwargs):
        pytest.fail("borrowed native ladders, references, and oracles must not be re-executed")

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    runtime = SimpleNamespace(_new_reference=unexpected, _record=unexpected, reference_key=unexpected,
        select_canonical_count_space_bnb_frame=unexpected)
    selectors = {label: selector(label) for label in ("old", "new")}
    return model, {"node": "toy", "lanes": 64, "states": states}, protocol, runtime, selectors, development, confirmation, calls


def test_both_source_layouts_borrow_the_correct_ladders_and_execute_two_cold_routers(monkeypatch):
    model, cell, protocol, runtime, selectors, development, confirmation, calls = harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert calls == [(label, tuple(state["queue"])) for state in cell["states"] for label in ("old", "new")]
    assert row["new_reference_calls"] == row["new_reference_oracle_calls"] == 0
    assert row["new_old_routed_calls"] == 3
    for state, borrowed in zip(row["states"], [*development, *confirmation]):
        expected = {"old": borrowed["old_traces"], "new": borrowed["traces"]} if "old_traces" in borrowed else borrowed["traces"]
        assert state["traces"] == expected
        assert state["original_routed"] == {"old": borrowed["old_routed"], "new": borrowed["routed"]}
        for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                    "effective_reference_complete", "effective_reference_source"):
            assert state[key] == borrowed[key]
        assert state["old_routed"]["label"] == "old" and state["routed"]["label"] == "new"
    assert [state["routed"]["canonical_exact"] for state in row["states"]] == [False, True, False]


@pytest.mark.parametrize("changed,value", (("queue", [9, 9, 9]), ("caps", [2, 4, 144, 256]),
    ("source", "fresh_confirmation"), ("state_id", "different state"), ("borrowed_index", 1)))
def test_wrong_borrowed_identity_or_budget_ladder_blocks_all_cold_calls(monkeypatch, changed, value):
    model, cell, protocol, runtime, selectors, development, _, calls = harness(monkeypatch)
    if changed == "borrowed_index":
        cell["states"][0][changed] = value
    else:
        development[0][changed] = value
    with pytest.raises(ValueError):
        runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert not calls


def test_registration_retains_development_holds_and_four_paired_failure_states(monkeypatch, tmp_path):
    options = (SimpleNamespace(requirements=(1, 0, 0), duration=1.0),
               SimpleNamespace(requirements=(0, 1, 0), duration=2.0))
    models = [SimpleNamespace(node=f"node{index}", options=options,
        snapshot=lambda index=index: {"node": f"node{index}"}) for index in range(4)]
    budgets = {"initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192}
    caps = [2, 3, 144, 256]
    historical = [{"node": model.node, "lanes": 64, "queue": [1, 0, 0]} for model in models]
    previous = {**budgets, "models": [model.snapshot() for model in models],
        "execution_sources": {"frozen.py": "original\n"}, "registered_at_utc": "development registered",
        "reference_budget": 4096, "oracle_state_budget": 5000000, "historical_regression_keys": historical,
        "prior_decisions": {"branch_reserved_timing": "BRANCH_RESERVED_TIMING_HOLD"},
        "cells": [{"node": model.node, "lanes": 64, "states": [
            {"state_id": "original development", "queue": [1, 0, 0], "source": "development", "cohort": "development",
             "caps": caps, "seed": 27, "historical_regression": True},
            {"state_id": "zero", "queue": [0, 0, 0], "source": "new_zero", "cohort": "zero", "caps": caps,
             "seed": None, "historical_regression": False}]} for model in models]}
    confirmation = {**budgets, "models": previous["models"], "execution_sources": previous["execution_sources"],
        "registered_at_utc": "confirmation registered", "cells": [
            {"node": model.node, "lanes": 64, "states": [
                {"state_id": "unselected fresh", "queue": [99, 99, 99], "source": "fresh_confirmation", "caps": caps},
                {"state_id": "selected failure", "queue": [54, 58, 290], "source": "fresh_confirmation",
                 "cohort": "fresh_confirmation", "caps": caps, "seed": 2026092831}]} for model in models]}
    prior, held = tmp_path / "prior.json", tmp_path / "held.json"
    prior.write_text(json.dumps({"decision": "DUAL_REUSE_DEVELOPMENT_ADVANCE", "development_gate_passed": True}))
    held.write_text(json.dumps({"decision": "DUAL_REUSE_FRESH_CONFIRMATION_HOLD", "execution_checks_passed": True,
        "effective_references_complete": True, "final_score_gate_passed": True}))
    for path, source in ((prior, previous), (held, confirmation)):
        path.with_name(path.stem + "_protocol.json").write_text(json.dumps(source))
    (tmp_path / "frozen.py").write_text("original\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_OUTPUT", prior)
    monkeypatch.setattr(runner, "HELD_OUTPUT", held)
    monkeypatch.setattr(runner, "SOURCES", ())
    protocol = runner.build_protocol(models)
    assert protocol["expected_state_count"] == 12
    assert protocol["expected_failure_development_count"] == 4
    assert protocol["expected_selector_calls"] == 24 and protocol["expected_native_advance_calls"] == 0
    assert protocol["historical_regression_keys"] == historical
    assert protocol["prior_decisions"]["dual_reuse_confirmation"] == "DUAL_REUSE_FRESH_CONFIRMATION_HOLD"
    assert protocol["prior_decisions"]["branch_reserved_timing"] == "BRANCH_RESERVED_TIMING_HOLD"
    assert protocol["execution_sources"]["frozen.py"] == "original\n"
    for original, cell in zip(previous["cells"], protocol["cells"]):
        assert [(state["state_id"], state["queue"], state["seed"]) for state in cell["states"][:2]] == [
            (state["state_id"], state["queue"], state["seed"]) for state in original["states"]]
        assert [state["borrowed_index"] for state in cell["states"][:2]] == [0, 1]
        failure = cell["states"][2]
        assert failure["queue"] == [54, 58, 290] and failure["borrowed_index"] == 1
        assert failure["state_id"] == "failure_development:selected failure"
        assert (failure["source"], failure["cohort"], failure["borrowed_from"]) == (
            "failure_development", "failure_development", "confirmation")
        assert failure["source_state_id"] == "selected failure" and failure["source_record_source"] == "fresh_confirmation"
        assert failure["historical_regression"] is False
    assert "borrowed_from" not in previous["cells"][0]["states"][0]
    runner.verify_protocol({**protocol, "registered_at_utc": "new"}, models)
    (tmp_path / "frozen.py").write_text("changed\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.build_protocol(models)
