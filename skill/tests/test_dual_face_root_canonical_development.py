"""Execute dual-face completion once, borrowing every real old control."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_dual_face_root_canonical_development as runner


@dataclass(frozen=True)
class Record:
    value: int


def harness(monkeypatch, group="independent_fresh"):
    model = SimpleNamespace(node="toy", lanes=4, options=())
    saved = {"state_id": "one", "queue": [2, 1], "caps": [2, 3, 144, 256],
        "timing_group": group, "confirmation_source": "fresh" if group != "trigger_control" else "borrowed_control",
        **{key: {"borrowed": key} for key in runner.REFERENCE_FIELDS},
        "traces": {"old": {"branch": True}, "new": {"sparse": True}},
        "previous_zero_traces": {"zero": True}, "original_affine_traces": {"affine": True},
        "first_results": {"branch": {"branch": True}, "candidate": {"early": True},
                          "zero": {"zero": True}, "affine": {"affine": True}}}
    if group == "trigger_control":
        saved["prior_routed"] = {"real_axis": True}
    registered = {key: saved[key] for key in ("state_id", "queue", "caps", "timing_group", "confirmation_source")}
    registered.update(dual_face_source_index=0, dual_face_source_state_id="one")
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_source": {"path": "source.json", "registered_at_utc": "prior"}}
    calls = []

    def selector(options, queue, lanes, *budgets):
        assert (queue, lanes, budgets) == ((2, 1), 8, (144, 256, 8192))
        calls.append("selector")
        return Record(1)

    def load(path, node, lanes, stamp):
        assert path == runner.ROOT / "source.json" and (node, lanes, stamp) == ("toy", 8, "prior")
        return {"states": [saved]}

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": True})
    return model, {"lanes": 8, "states": [registered]}, protocol, selector, saved, calls


@pytest.mark.parametrize("group", ("independent_fresh", "fresh_confirmation", "trigger_control"))
def test_real_zero_controls_and_single_candidate_execution(monkeypatch, group):
    model, cell, protocol, selector, saved, calls = harness(monkeypatch, group)
    row = runner.run_cell(model, cell, protocol, None, selector, progress=False)
    assert calls == ["selector"]
    assert (row["native_advance_calls"], row["candidate_calls"], row["native_search_constructions"]) == (0, 1, 1)
    assert row["new_reference_calls"] == row["new_oracle_calls"] == 0
    state = row["states"][0]
    assert state["traces"]["new"] == saved["previous_zero_traces"]
    assert state["previous_zero_routed"] == saved["first_results"]["zero"]
    assert state["original_affine_traces"] == saved["original_affine_traces"]
    assert state["original_affine_routed"] == saved["first_results"]["affine"]
    assert state["old_routed"] == saved["first_results"]["branch"]
    assert state["previous_early_routed"] == saved["first_results"]["candidate"]
    assert ("prior_routed" in state) == (group == "trigger_control")


@pytest.mark.parametrize("field,value", (("state_id", "wrong"), ("queue", [9, 9]), ("caps", [2, 4, 144, 256])))
def test_source_drift_stops_before_candidate_execution(monkeypatch, field, value):
    model, cell, protocol, selector, saved, calls = harness(monkeypatch)
    saved[field] = value
    with pytest.raises(ValueError, match="frozen timing source"):
        runner.run_cell(model, cell, protocol, None, selector, progress=False)
    assert calls == []


def test_timing_plan_cannot_change_after_quality_registration(monkeypatch):
    monkeypatch.setattr(runner, "build_protocol", lambda models: {"subsequent_timing_plan": runner.TIMING_PLAN})
    with pytest.raises(ValueError, match="rule changed"):
        runner.verify_protocol({"registered_at_utc": "now", "subsequent_timing_plan": {**runner.TIMING_PLAN, "order_seed": 1}}, [])
