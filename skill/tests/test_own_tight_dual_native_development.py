"""The own-tight development pays its real native prefixes and whole call."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_own_tight_dual_native_development as runner


@dataclass(frozen=True)
class Phase:
    calls: int


@dataclass(frozen=True)
class Stage:
    budget: int
    source: str


@dataclass(frozen=True)
class Record:
    value: int
    root_settle_work: Phase | None


def harness(monkeypatch, group="independent_fresh", settled=True):
    model = SimpleNamespace(node="toy", lanes=4, options=())
    saved = {"state_id": "one", "queue": [2, 1], "caps": [2, 3, 144, 256],
        "timing_group": group, "confirmation_source": "fresh" if group != "trigger_control" else "borrowed_control",
        "fused_root_source_index": 0, "fused_root_source_state_id": "original settled",
        **{key: {"borrowed": key} for key in runner.REFERENCE_FIELDS},
        "traces": {"old": {"branch": True}, "new": {"zero_native": True}},
        "previous_zero_traces": {"zero": True}, "original_affine_traces": {"affine": True},
        "old_routed": {"branch": True}, "previous_early_routed": {"early": True},
        "previous_zero_routed": {"zero": True}, "original_affine_routed": {"affine": True},
        "previous_dual_face_routed": {"actual_dual_face": True},
        "previous_dual_face_traces": {"actual_face_native": True},
        "previous_native_root_routed": {"actual_native_root": True},
        "previous_native_root_traces": {"actual_native_ladder": True},
        "previous_settled_root_routed": {"actual_settled": True},
        "previous_settled_root_traces": {"actual_settled_ladder": True}, "routed": {"actual_fused": True}}
    if group == "trigger_control":
        saved["prior_routed"] = {"real_axis": True}
    registered = {key: saved[key] for key in ("state_id", "queue", "caps", "timing_group", "confirmation_source",
        "fused_root_source_index", "fused_root_source_state_id")}
    registered.update(own_tight_source_index=0, own_tight_source_state_id="one")
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_source": {"path": "source.json", "registered_at_utc": "original run"}}
    calls = []

    class Audit:
        def __init__(self, options, queue, lanes):
            assert (options, queue, lanes) == ((), (2, 1), 8)
            calls.append("audit_construct")

        def advance_roots_and_settle(self, cap):
            calls.append(("root", cap))
            return Stage(cap, "raw"), Stage(cap, "settled") if settled else None, Phase(1) if settled else None

        def advance_to(self, cap):
            calls.append(("advance", cap))
            return Stage(cap, "new_native")

    def selector(options, queue, lanes, *budgets):
        assert (queue, lanes, budgets) == ((2, 1), 8, (144, 256, 8192))
        calls.append("whole")
        return Record(1, Phase(1) if settled else None)

    def load(path, node, lanes, stamp):
        assert path == runner.ROOT / "source.json" and (node, lanes, stamp) == ("toy", 8, "original run")
        return {"states": [saved]}

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": True})
    return model, {"lanes": 8, "states": [registered]}, protocol, selector, Audit, saved, calls


@pytest.mark.parametrize("group", ("independent_fresh", "fresh_confirmation", "trigger_control"))
@pytest.mark.parametrize("settled", (True, False))
def test_new_prefixes_are_paid_and_old_controls_stay_distinct(monkeypatch, group, settled):
    model, cell, protocol, selector, audit, saved, calls = harness(monkeypatch, group, settled)
    row = runner.run_cell(model, cell, protocol, None, selector, audit, progress=False)
    assert calls == ["audit_construct", ("root", 2), ("advance", 3), ("advance", 144), ("advance", 256), "whole"]
    assert (row["native_advance_calls"], row["candidate_calls"], row["native_search_constructions"]) == (4, 1, 2)
    assert row["root_settle_calls"] == int(settled)
    assert row["new_reference_calls"] == row["new_oracle_calls"] == 0
    state = row["states"][0]
    assert state["previous_fused_root_routed"] == saved["routed"]
    assert state["previous_fused_root_traces"] == saved["traces"]["new"]
    assert state["candidate_native_root"]["root_lp"] == {"budget": 2, "source": "raw"}
    assert state["candidate_native_traces"]["2"] == {"budget": 2, "source": "settled" if settled else "raw"}
    assert state["candidate_native_traces"]["256"] == {"budget": 256, "source": "new_native"}
    for key in ("traces", "previous_native_root_routed", "previous_native_root_traces", "previous_zero_routed",
                "previous_zero_traces", "original_affine_routed", "original_affine_traces", "old_routed",
                "previous_early_routed", "previous_dual_face_routed", "previous_dual_face_traces",
                "previous_settled_root_routed", "previous_settled_root_traces"):
        assert state[key] == saved[key]
    assert ("prior_routed" in state) == (group == "trigger_control")


@pytest.mark.parametrize("field,value", (("state_id", "wrong"), ("queue", [9, 9]), ("caps", [2, 4, 144, 256]),
    ("fused_root_source_state_id", "fabricated predecessor")))
def test_parent_drift_stops_before_any_native_or_whole_execution(monkeypatch, field, value):
    model, cell, protocol, selector, audit, saved, calls = harness(monkeypatch)
    saved[field] = value
    with pytest.raises(ValueError, match="frozen fused development source"):
        runner.run_cell(model, cell, protocol, None, selector, audit, progress=False)
    assert calls == []


def test_ten_arm_plan_is_frozen_before_outcome_access(monkeypatch):
    assert runner.TIMING_PLAN["expected_timed_calls"] == 2802 * 6 * 10
    assert runner.TIMING_PLAN["expected_warmup_calls"] == 16 * 2 * 10
    assert len(set(runner.TIMING_PLAN["labels"])) == 10
    monkeypatch.setattr(runner, "build_protocol", lambda models: {"subsequent_timing_plan": runner.TIMING_PLAN})
    with pytest.raises(ValueError, match="rule changed"):
        runner.verify_protocol({"registered_at_utc": "now", "subsequent_timing_plan": {**runner.TIMING_PLAN, "order_seed": 1}}, [])
