"""Execute only the sparse candidate while retaining real borrowed controls."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_sparse_zero_delta_affine_development as runner


@dataclass(frozen=True)
class Record:
    value: int


def harness(monkeypatch, group="independent_fresh"):
    model = SimpleNamespace(node="toy", lanes=4, options=())
    saved = {"state_id": "one", "queue": [2, 1], "caps": [2, 3, 144, 256],
        "timing_group": group, "confirmation_source": "fresh" if group != "trigger_control" else "borrowed_control",
        **{key: {"borrowed": key} for key in runner.REFERENCE_FIELDS},
        "traces": {"old": {"branch": True}, "new": {"zero": True}},
        "previous_dual_traces": {"affine": True},
        "first_results": {"branch": {"branch": True}, "candidate": {"zero": True}, "affine": {"affine": True}}}
    if group == "trigger_control":
        saved["prior_routed"] = {"real_axis": True}
    registered = {key: saved[key] for key in ("state_id", "queue", "caps", "timing_group", "confirmation_source")}
    registered.update(sparse_delta_source_index=0, sparse_delta_source_state_id="one")
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_source": {"path": "source.json", "registered_at_utc": "prior"}}
    calls = []

    class Search:
        def __init__(self, options, queue, lanes):
            assert (queue, lanes) == ((2, 1), 8)
            calls.append("construct")

        def advance_to(self, cap):
            calls.append(cap)
            return Record(cap)

    def selector(options, queue, lanes, *budgets):
        assert (queue, lanes, budgets) == ((2, 1), 8, (144, 256, 8192))
        calls.append("selector")
        return Record(1)

    def load(path, node, lanes, stamp):
        assert path == runner.ROOT / "source.json" and (node, lanes, stamp) == ("toy", 8, "prior")
        return {"states": [saved]}

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": True})
    return model, {"lanes": 8, "states": [registered]}, protocol, Search, selector, saved, calls


@pytest.mark.parametrize("group", ("independent_fresh", "fresh_confirmation", "trigger_control"))
def test_real_control_lineage_and_candidate_execution(monkeypatch, group):
    model, cell, protocol, search, selector, saved, calls = harness(monkeypatch, group)
    row = runner.run_cell(model, cell, protocol, None, search, selector, progress=False)
    assert calls == ["construct", 2, 3, 144, 256, "selector"]
    assert (row["native_advance_calls"], row["candidate_calls"], row["native_search_constructions"]) == (4, 1, 2)
    assert row["new_reference_calls"] == row["new_oracle_calls"] == 0
    state = row["states"][0]
    assert state["previous_zero_traces"] == saved["traces"]["new"]
    assert state["previous_zero_routed"] == saved["first_results"]["candidate"]
    assert state["original_affine_traces"] == saved["previous_dual_traces"]
    assert state["original_affine_routed"] == saved["first_results"]["affine"]
    assert ("prior_routed" in state) == (group == "trigger_control")


@pytest.mark.parametrize("field,value", (("state_id", "wrong"), ("queue", [9, 9]), ("caps", [2, 4, 144, 256])))
def test_source_drift_stops_before_candidate_execution(monkeypatch, field, value):
    model, cell, protocol, search, selector, saved, calls = harness(monkeypatch)
    saved[field] = value
    with pytest.raises(ValueError, match="frozen timing source"):
        runner.run_cell(model, cell, protocol, None, search, selector, progress=False)
    assert calls == []


def test_timing_plan_cannot_change_after_quality_registration(monkeypatch):
    monkeypatch.setattr(runner, "build_protocol", lambda models: {"subsequent_timing_plan": runner.TIMING_PLAN})
    with pytest.raises(ValueError, match="rule changed"):
        runner.verify_protocol({"registered_at_utc": "now", "subsequent_timing_plan": {**runner.TIMING_PLAN, "order_seed": 1}}, [])
