"""New cold calls retain real borrowed sources and reject source drift."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_zero_delta_affine_development as runner


@dataclass(frozen=True)
class Record:
    value: int


def harness(monkeypatch, control=False):
    model = SimpleNamespace(node="toy", lanes=4, options=())
    saved = {"state_id": "one", "queue": [2, 1], "caps": [2, 3, 144, 256],
        "timing_group": "trigger_control" if control else "fresh_confirmation",
        "confirmation_source": "borrowed_control" if control else "fresh",
        **{key: {"borrowed": key} for key in runner.REFERENCE_FIELDS},
        "traces": {"old": {"branch": True}, "new": {"affine": True}},
        "first_results": {"branch": {"branch": True}, "candidate": {"affine": True}}}
    if control:
        saved["prior_routed"] = {"real_axis": True}
    registered = {key: saved[key] for key in ("state_id", "queue", "caps", "timing_group", "confirmation_source")}
    registered.update(zero_delta_source_index=0, zero_delta_source_state_id="one")
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_source": {"path": "source.json", "registered_at_utc": "prior"}}
    calls = []

    class Search:
        def __init__(self, options, queue, lanes):
            assert options is model.options and (queue, lanes) == ((2, 1), 8)
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


@pytest.mark.parametrize("control", (False, True))
def test_only_candidate_ladder_and_cold_router_execute(monkeypatch, control):
    model, cell, protocol, search, selector, saved, calls = harness(monkeypatch, control)
    row = runner.run_cell(model, cell, protocol, SimpleNamespace(), search, selector)
    assert calls == ["construct", 2, 3, 144, 256, "selector"]
    assert (row["native_advance_calls"], row["candidate_calls"], row["native_search_constructions"]) == (4, 1, 2)
    assert row["new_reference_calls"] == row["new_oracle_calls"] == 0
    state = row["states"][0]
    assert state["previous_dual_traces"] == saved["traces"]["new"]
    assert state["previous_dual_routed"] == saved["first_results"]["candidate"]
    assert ("prior_routed" in state) == control
    if control:
        assert state["prior_routed"] == saved["prior_routed"]


@pytest.mark.parametrize("field,value", (("state_id", "wrong"), ("queue", [9, 9]), ("caps", [2, 4, 144, 256])))
def test_changed_source_stops_before_any_execution(monkeypatch, field, value):
    model, cell, protocol, search, selector, saved, calls = harness(monkeypatch)
    saved[field] = value
    with pytest.raises(ValueError, match="frozen timing source"):
        runner.run_cell(model, cell, protocol, SimpleNamespace(), search, selector)
    assert calls == []


def test_registered_rule_change_is_detected(monkeypatch):
    monkeypatch.setattr(runner, "build_protocol", lambda models: {"rule": "frozen"})
    with pytest.raises(ValueError, match="rule changed"):
        runner.verify_protocol({"registered_at_utc": "now", "rule": "changed"}, [])
