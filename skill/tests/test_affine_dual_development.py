"""A fresh native ladder and cold router borrow the exact frozen controls."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_affine_dual_development as runner


@dataclass(frozen=True)
class Record:
    value: int


def harness(monkeypatch):
    model = SimpleNamespace(node="toy", lanes=4, options=())
    saved = {"state_id": "one", "queue": [2, 1], "caps": [2, 3, 144, 256],
             "timing_group": "ordinary_development", "seed": 27,
             **{key: {"borrowed": key} for key in runner.REFERENCE_FIELDS},
             "traces": {"old": {"branch": True}, "new": {"dual": True}},
             "prior_routed": {"axis": True},
             "first_results": {"branch": {"branch": True}, "candidate": {"dual": True}}}
    registered = {key: saved[key] for key in ("state_id", "queue", "caps", "timing_group", "seed")}
    registered.update(affine_source_index=0, affine_source_state_id="one")
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
                "integer_state_budget": 8192,
                "reference_source": {"path": "source.json", "registered_at_utc": "prior"}}
    calls = []

    class Search:
        def __init__(self, options, queue, lanes):
            assert options is model.options and (queue, lanes) == ((2, 1), 8)
            calls.append("construct")

        def advance_to(self, cap):
            calls.append(cap)
            return Record(cap)

    def selector(options, queue, lanes, *budgets):
        assert options is model.options and (queue, lanes) == ((2, 1), 8)
        assert budgets == (144, 256, 8192)
        calls.append("selector")
        return Record(1)

    def load(path, node, lanes, stamp):
        assert path == runner.ROOT / "source.json" and (node, lanes, stamp) == ("toy", 8, "prior")
        return {"states": [saved]}

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": True})
    return model, {"lanes": 8, "states": [registered]}, protocol, Search, selector, saved, calls


def test_independent_cold_ladder_and_router_do_not_execute_borrowed_controls(monkeypatch):
    model, cell, protocol, search, selector, saved, calls = harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, SimpleNamespace(), search, selector, progress=False)
    assert calls == ["construct", 2, 3, 144, 256, "selector"]
    assert (row["native_advance_calls"], row["candidate_calls"], row["native_search_constructions"]) == (4, 1, 2)
    assert row["new_reference_calls"] == row["new_oracle_calls"] == 0
    state = row["states"][0]
    assert state["previous_dual_traces"] == saved["traces"]["new"]
    assert state["old_routed"] == saved["first_results"]["branch"]
    assert state["previous_dual_routed"] == saved["first_results"]["candidate"]
    assert state["prior_routed"] == saved["prior_routed"]
    assert state["traces"]["new"] == {str(cap): {"value": cap} for cap in cell["states"][0]["caps"]}
    assert {key: state[key] for key in runner.REFERENCE_FIELDS} == {key: saved[key] for key in runner.REFERENCE_FIELDS}


@pytest.mark.parametrize("field,value", (("state_id", "wrong"), ("queue", [9, 9]), ("caps", [2, 4, 144, 256])))
def test_changed_source_identity_queue_or_budget_stops_before_execution(monkeypatch, field, value):
    model, cell, protocol, search, selector, saved, calls = harness(monkeypatch)
    saved[field] = value
    with pytest.raises(ValueError, match="frozen timing source"):
        runner.run_cell(model, cell, protocol, SimpleNamespace(), search, selector, progress=False)
    assert calls == []
