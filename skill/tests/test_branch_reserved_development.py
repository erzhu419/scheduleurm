"""Borrow all development references while executing both cold selector arms."""
from dataclasses import asdict, dataclass, fields
import json
from types import SimpleNamespace

import pytest

import algorithm.resumable_integer_capacity_frame_candidates as old_engine
import algorithm.branch_reserved_integer_capacity_frame_candidates as new_engine
import algorithm.certificate_routed_frame_candidates as old_routed
import algorithm.branch_reserved_certificate_routed_frame_candidates as new_routed
from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.experiments import aligned_hardware_branch_reserved_development as runner


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def test_borrowed_references_include_zero_and_both_cold_arms_still_execute(monkeypatch):
    model = SimpleNamespace(node="toy", lanes=1, options=(Option((1, 0), 1.0), Option((0, 1), 2.0)))
    states = [{"state_id": "one", "queue": [1, 0], "source": "development", "cohort": "development", "prior_index": 0},
              {"state_id": "zero", "queue": [0, 0], "source": "new_zero", "cohort": "zero", "prior_index": 1}]
    borrowed = []
    for index, state in enumerate(states):
        reference = {field.name: 0 for field in fields(CanonicalCountSpaceBnbSelection)}
        reference.update(option_counts=(1, 0) if index == 0 else (0, 0),
            option_indices=(0, None) if index == 0 else (None, None), service=tuple(state["queue"]),
            duration=1.0, score=float(1 - index), upper_bound=float(1 - index),
            exact=index == 1, canonical_exact=index == 1, score_exact=True, lp_node_budget=4096)
        oracle = {"complete": True, "canonical_exact": True, "score_fraction": [1, 1],
                  "duration_fraction": [1, 1], "option_indices": [0, None]} if index == 0 else None
        borrowed.append({**state, "reference": asdict(CanonicalCountSpaceBnbSelection(**reference)),
            "oracle": oracle, "effective_reference_complete": True,
            "reference_metrics": {"stale": True}, "old_routed": {"stale": True}})
    constructions = {"old": [], "new": []}

    def spy_class(base, label):
        class Search(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.caps = []
                constructions[label].append(self)

            def advance_to(self, cap):
                self.caps.append(cap)
                return super().advance_to(cap)
        return Search

    old_search = spy_class(old_engine.IntegerCapacitySearch, "old")
    new_search = spy_class(new_engine.BranchReservedIntegerCapacitySearch, "new")
    monkeypatch.setattr(old_engine, "IntegerCapacitySearch", old_search)
    monkeypatch.setattr(old_routed, "IntegerCapacitySearch", old_search)
    monkeypatch.setattr(new_engine, "BranchReservedIntegerCapacitySearch", new_search)
    monkeypatch.setattr(new_routed, "BranchReservedIntegerCapacitySearch", new_search)
    monkeypatch.setattr(runner, "load_cell", lambda *args: {"states": borrowed})
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    records = []

    def unexpected(*args, **kwargs):
        pytest.fail("all references, including zero, must be borrowed without a new solve")

    def record(metrics, reference, model, queue, lanes, budget, expected):
        records.append((queue, lanes, budget, expected))
        metrics["recorded_queue"] = list(queue)

    runtime = SimpleNamespace(_metrics=lambda: {}, _record=record,
        reference_key=lambda reference, queue, count: ("full",) if reference.canonical_exact else None,
        select_canonical_count_space_bnb_frame=unexpected, select_reference=unexpected)
    protocol = {"registered_at_utc": "current", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_budget": 4096,
        "prior_source": {"path": "prior.json", "registered_at_utc": "prior"}}
    row = runner.run_cell(model, {"lanes": 2, "states": states}, protocol, runtime, progress=False)
    assert row["new_reference_calls"] == 0 and row["new_old_routed_calls"] == 2
    assert [record[:3] for record in records] == [((1, 0), 2, 4096), ((0, 0), 2, 4096)]
    assert [record[3] for record in records] == [None, ("full",)]
    for label in ("old", "new"):
        assert len(constructions[label]) == 4
        assert [search.caps for search in constructions[label]] == [[2, 3, 144, 256], [144]] * 2
    for index, state in enumerate(row["states"]):
        assert state["reference"] == json.loads(json.dumps(borrowed[index]["reference"]))
        assert state["oracle"] == borrowed[index]["oracle"]
        assert state["reference_metrics"] == {"recorded_queue": state["queue"]}
        assert state["effective_reference_complete"] is True
        assert state["old_routed"]["selection"]["lp_calls"] == 2
        assert state["routed"]["selection"]["lp_calls"] == 1 - index
        for cap, snapshot in state["traces"]["new"].items():
            root = snapshot["root_pruning"]
            assert snapshot["selection"]["lp_calls"] + root["roots_seen"] - root["roots_solved"] <= int(cap)
        assert state["traces"]["new"]["2"]["selection"]["branch_splits"] == 0
    assert row["states"][0]["reference_complete"] is False
    assert row["states"][0]["effective_reference_source"] == "independent_score_face_oracle"
    assert row["states"][1]["reference_complete"] is True
    assert row["states"][1]["effective_reference_source"] == "full_K4096"


def test_historical_regressions_must_already_belong_to_development(monkeypatch, tmp_path):
    model = SimpleNamespace(snapshot=lambda: {"node": "toy"})
    budgets = {"initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192}
    original = {**budgets, "models": [model.snapshot()], "execution_sources": {}, "registered_at_utc": "prior",
        "prior_decisions": {"confirmation": "retained timing HOLD"}, "cells": [{"node": "toy", "lanes": 8,
            "states": [{"state_id": "one", "queue": [1, 0], "source": "development"},
                       {"state_id": "zero", "queue": [0, 0], "source": "new_zero"}]}]}
    held = {**budgets, "models": [model.snapshot()], "execution_sources": {}, "cells": [{"node": "toy", "lanes": 8,
        "states": [{"queue": [1, 0], "source": "historical_regression"},
                   {"queue": [9, 9], "source": "fresh_confirmation"}]}]}
    prior_path, held_path = tmp_path / "prior.json", tmp_path / "held.json"
    prior_path.write_text(json.dumps({"decision": "ROOT_PRUNING_DEVELOPMENT_ADVANCE", "development_gate_passed": True}))
    held_path.write_text(json.dumps({"decision": "ROOT_PRUNING_FRESH_CONFIRMATION_HOLD", "execution_checks_passed": True,
        "effective_references_complete": True, "routed_quality_gate_passed": True, "finite_budget_gate_passed": True}))
    prior_path.with_name("prior_protocol.json").write_text(json.dumps(original))
    held_protocol = held_path.with_name("held_protocol.json")
    held_protocol.write_text(json.dumps(held))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_OUTPUT", prior_path)
    monkeypatch.setattr(runner, "CONFIRMATION_OUTPUT", held_path)
    monkeypatch.setattr(runner, "SOURCES", ())
    protocol = runner.build_protocol([model])
    assert [state["queue"] for state in protocol["cells"][0]["states"]] == [[1, 0], [0, 0]]
    assert [state["historical_regression"] for state in protocol["cells"][0]["states"]] == [True, False]
    assert [state["prior_index"] for state in protocol["cells"][0]["states"]] == [0, 1]
    assert "prior_index" not in original["cells"][0]["states"][0]
    held["cells"][0]["states"][0]["queue"] = [2, 0]
    held_protocol.write_text(json.dumps(held))
    with pytest.raises(ValueError, match="historical regression keys must be in the development roster"):
        runner.build_protocol([model])
