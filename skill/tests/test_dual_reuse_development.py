"""Freeze the existing roster and execute new work without rebuilding references."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_dual_reuse_development as runner


@dataclass(frozen=True)
class Snapshot:
    cap: int
    queue: tuple[int, ...]


@dataclass(frozen=True)
class Result:
    label: str
    canonical_exact: bool


def harness(monkeypatch):
    model = SimpleNamespace(node="toy", lanes=4, options=(
        SimpleNamespace(requirements=(1, 0), duration=1.0),
        SimpleNamespace(requirements=(0, 1), duration=2.0)))
    caps = [2, 3, 144, 256]
    registered = [{"state_id": f"dev:{i}", "source_state_id": f"dev:{i}",
                   "queue": queue, "source": source, "cohort": source,
                   "prior_index": i, "caps": caps}
                  for i, (queue, source) in enumerate((([2, 1], "development"), ([0, 0], "new_zero")))]
    saved = [{**state, "reference": {"index": i}, "reference_metrics": {"independent": i},
              "reference_complete": i == 1, "oracle": {"complete": True} if i == 0 else None,
              "effective_reference_complete": True,
              "effective_reference_source": "independent_score_face_oracle" if i == 0 else "full_K4096",
              "traces": {"new": {str(cap): {"borrowed": cap} for cap in caps}, "old": {"wrong_arm": True}},
              "routed": {"frozen": i}, "old_routed": {"wrong_control": i}}
             for i, state in enumerate(registered)]
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
                "integer_state_budget": 8192, "prior_source": {"path": "source.json", "registered_at_utc": "prior"}}
    constructions, advances, calls = [], [], []

    class Search:
        def __init__(self, options, queue, lanes):
            assert options is model.options and lanes == 8
            self.queue = queue
            constructions.append(queue)

        def advance_to(self, cap):
            advances.append((self.queue, cap))
            return Snapshot(cap, self.queue)

    def selector(label):
        def run(options, queue, lanes, *budgets):
            assert options is model.options and lanes == 8 and budgets == (144, 256, 8192)
            calls.append((label, queue))
            return Result(label, label == "old" or queue == (0, 0))
        return run

    def load(path, node, lanes, registered_at):
        assert path == runner.ROOT / "source.json" and (node, lanes, registered_at) == ("toy", 8, "prior")
        return {"states": saved}

    def unexpected(*args, **kwargs):
        pytest.fail("a borrowed reference, oracle or old native ladder must not be re-executed")

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": True})
    import algorithm.branch_reserved_integer_capacity_frame_candidates as old_engine
    monkeypatch.setattr(old_engine.BranchReservedIntegerCapacitySearch, "__init__", unexpected)
    runtime = SimpleNamespace(_new_reference=unexpected, reference_key=unexpected, _record=unexpected)
    return model, {"lanes": 8, "states": registered}, protocol, runtime, Search, {
        "old": selector("old"), "new": selector("new")}, saved, constructions, advances, calls


def test_four_new_native_caps_and_two_cold_arms_borrow_the_right_predecessor(monkeypatch):
    model, cell, protocol, runtime, search, selectors, saved, constructions, advances, calls = harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, search, selectors, progress=False)
    assert constructions == [(2, 1), (0, 0)]
    assert advances == [(queue, cap) for queue in constructions for cap in (2, 3, 144, 256)]
    assert calls == [(label, queue) for queue in constructions for label in ("old", "new")]
    assert row["new_reference_calls"] == row["new_reference_oracle_calls"] == 0
    assert row["new_old_routed_calls"] == 2
    for index, state in enumerate(row["states"]):
        assert {key: state[key] for key in runner.REFERENCE_FIELDS} == {
            key: saved[index][key] for key in runner.REFERENCE_FIELDS}
        assert state["old_traces"] == saved[index]["traces"]["new"]
        assert state["expected_old_routed"] == saved[index]["routed"]
        assert set(state["traces"]) == {"2", "3", "144", "256"}
    assert row["states"][0]["routed"]["canonical_exact"] is False
    assert row["states"][1]["routed"]["canonical_exact"] is True


@pytest.mark.parametrize("field,value", (("queue", [9, 9]), ("caps", [2, 4, 144, 256])))
def test_mismatched_borrowed_state_or_cap_blocks_execution(monkeypatch, field, value):
    model, cell, protocol, runtime, search, selectors, saved, constructions, advances, calls = harness(monkeypatch)
    saved[0][field] = value
    with pytest.raises(ValueError, match="state or budget ladder changed"):
        runner.run_cell(model, cell, protocol, runtime, search, selectors, progress=False)
    assert not constructions and not advances and not calls


def test_registration_keeps_roster_and_holds_and_checks_frozen_source(monkeypatch, tmp_path):
    model = SimpleNamespace(node="toy", options=(
        SimpleNamespace(requirements=(1, 0), duration=1.0),
        SimpleNamespace(requirements=(1, 0), duration=3.0),
        SimpleNamespace(requirements=(0, 1), duration=2.0)), snapshot=lambda: {"node": "toy"})
    budgets = {"initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192}
    source = {**budgets, "models": [model.snapshot()], "execution_sources": {"old.py": "frozen\n"},
              "registered_at_utc": "prior", "reference_budget": 4096, "oracle_state_budget": 5000000,
              "historical_regression_keys": [{"node": "toy", "lanes": 8, "queue": [2, 1]}],
              "cells": [{"node": "toy", "lanes": 8, "states": [
                  {"state_id": "one", "queue": [2, 1], "source": "development", "seed": 27},
                  {"state_id": "zero", "queue": [0, 0], "source": "new_zero", "seed": None}]}]}
    prior, held = tmp_path / "prior.json", tmp_path / "held.json"
    prior.write_text(json.dumps({"decision": "BRANCH_RESERVED_DEVELOPMENT_ADVANCE", "development_gate_passed": True}))
    held.write_text(json.dumps({"decision": "SHORT_LOOKAHEAD_DEVELOPMENT_HOLD", "execution_checks_passed": True,
        "effective_references_complete": True, "routed_quality_gate_passed": True, "final_score_gate_passed": True,
        "prior_decisions": {"branch_reserved_timing": "BRANCH_RESERVED_TIMING_HOLD"}}))
    for path in (prior, held):
        path.with_name(path.stem + "_protocol.json").write_text(json.dumps(source))
    (tmp_path / "old.py").write_text("frozen\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_OUTPUT", prior)
    monkeypatch.setattr(runner, "HELD_OUTPUT", held)
    monkeypatch.setattr(runner, "SOURCES", ())
    protocol = runner.build_protocol([model])
    assert [state["queue"] for state in protocol["cells"][0]["states"]] == [[2, 1], [0, 0]]
    assert [state["seed"] for state in protocol["cells"][0]["states"]] == [27, None]
    assert [state["caps"] for state in protocol["cells"][0]["states"]] == [[2, 3, 144, 256]] * 2
    assert "caps" not in source["cells"][0]["states"][0]
    assert protocol["prior_decisions"] == {"branch_reserved_timing": "BRANCH_RESERVED_TIMING_HOLD",
                                           "short_lookahead_development": "SHORT_LOOKAHEAD_DEVELOPMENT_HOLD"}
    (tmp_path / "old.py").write_text("changed\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.build_protocol([model])
