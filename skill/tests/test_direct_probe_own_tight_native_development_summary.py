"""Paid complete-box root probes and ordinary same-search continuation."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import algorithm.direct_probe_own_tight_native_kernel as native
from algorithm.direct_probe_own_tight_native_frame_candidates import select_direct_probe_own_tight_native_frame
from algorithm.experiments import direct_probe_own_tight_native_development_summary as summary
from skill.tests.test_own_tight_dual_native_development_summary import (
    lp_factory, records as own_records, tiny_records as own_tiny,
)


def tiny_records(monkeypatch, path):
    protocol, state, model, lanes, previous, expected = own_tiny(monkeypatch, path)
    state.update(previous_own_tight_routed=previous,
        previous_own_tight_native_root=state["candidate_native_root"],
        previous_own_tight_native_traces=state["candidate_native_traces"])
    mode = "face" if path == "face_exhausted" else "weak" if path == "score_open" else "real" if path == "closed" else "tie"
    options = tuple(SimpleNamespace(**option) for option in model["options"])
    real = native.linprog
    monkeypatch.setattr(native, "linprog", lp_factory(real, mode))
    engine = native.DirectProbeOwnTightNativeSearch(options, state["queue"], lanes)
    root, settled, phase = engine.advance_roots_and_settle(state["caps"][0])
    state["candidate_native_root"] = {"root_lp": asdict(root), "settled_lp": asdict(settled) if settled is not None else None,
                                      "root_settle_work": asdict(phase) if phase is not None else None}
    state["candidate_native_traces"] = {str(state["caps"][0]): asdict(settled if settled is not None else root)}
    state["candidate_native_traces"].update({str(cap): asdict(engine.advance_to(cap)) for cap in state["caps"][1:]})
    monkeypatch.setattr(native, "linprog", lp_factory(real, mode))
    result = asdict(select_direct_probe_own_tight_native_frame(options, state["queue"], lanes,
        protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"]))
    return protocol, state, model, lanes, result, expected


@pytest.mark.parametrize("path", ("closed", "open", "exhausted", "face_exhausted", "score_open"))
def test_true_direct_root_box_probe_and_four_native_prefixes(monkeypatch, path):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, path)
    _, errors, quality = summary._audit_direct_probe_result(result, state, protocol, model, lanes, "candidate", expected)
    assert not errors and quality, errors
    assert result["root_lp"] == state["previous_own_tight_routed"]["root_lp"]
    assert result["root_lp"]["native_own_tight_work"]["own_bound_cache_writes"] > 0
    work = summary._work([result], "candidate")
    assert work["search_work"]["lp_calls"] == result["final_lp"]["selection"]["lp_calls"]
    assert work["native_own_tight_work"]["event_snapshot_entries_copied"] == result["final_lp"]["native_own_tight_work"]["event_snapshot_entries_copied"]
    if path in ("open", "exhausted", "face_exhausted"):
        phase = result["root_settle_work"]
        assert phase["stopped_for_branch"] == 1 and phase["stopped_for_lp"] == phase["split_regions"] == 0
        assert phase["lp_calls_before"] == phase["lp_calls_after"]
        assert result["settled_lp"]["selection"]["branch_splits"] == 0
        assert phase["frontier_regions_after"] == phase["frontier_regions_before"]
        assert result["dual_face_evidence"]["regions"][0]["lower"] == (0,) * len(model["options"])
        assert result["probe_stage"] == "root" and work["root_probe_calls"] == 1
    if path in ("exhausted", "face_exhausted"):
        assert result["probe"]["completion"]["status"] == "STATE_BUDGET_EXHAUSTED"
        assert result["final_lp"]["selection"]["branch_splits"] > 0
        assert work["root_probe_exhausted_calls"] == 1
    if path in ("face_exhausted", "score_open"):
        assert result["final_lp"]["native_own_tight_evidence"]
        assert result["final_lp"]["native_own_tight_work"]["event_snapshot_entries_copied"] < state["candidate_native_traces"]["144"]["native_own_tight_work"]["event_snapshot_entries_copied"]
    if path == "closed":
        assert result["status"] == "LP_ROOT_SETTLE_COMPLETE" and result["probe"] is None


def test_parent_control_uses_its_genuine_native_engine_and_original_phase_work(monkeypatch):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, "face_exhausted")
    previous = state["previous_own_tight_routed"]
    _, errors, quality = summary._audit_direct_probe_result(previous, state, protocol, model, lanes, "own", expected)
    assert not errors and quality, errors
    assert "stopped_for_branch" not in summary._work([previous], "own")["root_settle_diagnostics"]
    changed = deepcopy(state)
    changed["previous_own_tight_native_traces"] = changed["candidate_native_traces"]
    _, errors, _ = summary._audit_direct_probe_result(previous, changed, protocol, model, lanes, "own", expected)
    assert errors


def test_branch_stop_raw_cache_and_actual_snapshot_fee_mutations_are_rejected(monkeypatch):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, "face_exhausted")
    for name in ("stop", "split", "lost_box", "raw_cache", "whole_copy", "audit_copy", "source"):
        changed, changed_state = deepcopy(result), deepcopy(state)
        if name == "stop":
            changed["root_settle_work"]["stopped_for_branch"] = 0
        elif name == "split":
            changed["root_settle_work"]["split_regions"] = 1
        elif name == "lost_box":
            changed["root_settle_work"]["frontier_regions_after"] -= 1
        elif name == "raw_cache":
            changed["root_lp"]["native_own_tight_work"]["own_bound_cache_writes"] += 1
        elif name == "whole_copy":
            changed["final_lp"]["native_own_tight_work"]["event_snapshot_entries_copied"] += 1
        elif name == "audit_copy":
            changed_state["candidate_native_traces"]["256"]["native_own_tight_work"]["event_snapshot_entries_copied"] -= 1
        else:
            changed["final_lp"]["native_own_tight_evidence"][0]["certificate_numerator_fraction"] = (7, 1)
        _, errors, _ = summary._audit_direct_probe_result(changed, changed_state, protocol, model, lanes, "candidate", expected)
        assert errors, name


@pytest.fixture
def records():
    protocol, rows = deepcopy(own_records.__wrapped__())
    protocol["experiment"] = "direct_probe_own_tight_native_development_v1"
    for row, cell in zip(rows, protocol["cells"], strict=True):
        for index, (registered, state) in enumerate(zip(cell["states"], row["states"], strict=True)):
            registered.update(direct_probe_source_index=index, direct_probe_source_state_id=state["state_id"])
            state.update(registered)
            state.update(previous_own_tight_routed=deepcopy(state["routed"]),
                previous_own_tight_native_root=deepcopy(state["candidate_native_root"]),
                previous_own_tight_native_traces=deepcopy(state["candidate_native_traces"]))
    return protocol, rows


def test_full_roster_actual_interface_counts_and_each_final_ledger_once(records):
    result = summary.summarize(*records)
    assert result["decision"] == "DIRECT_PROBE_OWN_TIGHT_NATIVE_DEVELOPMENT_ADVANCE", result["failures"]
    assert all(result[name] for name in ("execution_checks_passed", "references_complete", "quality_gate_passed", "work_gate_passed", "development_gate_passed", "prior_hold_retained"))
    count = result["state_count"]
    assert result["executed_selector_calls"] == count == 12
    assert result["executed_native_advance_calls"] == result["executed_candidate_native_snapshots"] == 4 * count
    assert result["native_search_constructions"] == 2 * count and result["executed_interface_calls"] == 5 * count
    assert "stopped_for_branch" in result["root_settle_diagnostics"]
    assert "stopped_for_branch" not in result["executed_root_settle_work"]
    assert "stopped_for_branch" not in result["borrowed"]["whole_selector"]["own"]["root_settle_diagnostics"]


def test_new_nonclosure_keeps_frozen_controls_full_roster_and_exhausted_cost(records):
    protocol, rows = records
    state = rows[0]["states"][2]
    result, probe = state["routed"], state["routed"]["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    result.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "DIRECT_PROBE_OWN_TIGHT_NATIVE_DEVELOPMENT_HOLD", result["failures"]
    assert result["execution_checks_passed"] and result["work_gate_passed"]
    assert result["state_count"] == 12 and len(result["nonclosures"]) == 1
    assert result["executed"]["whole_selector"]["shared_integer_states"] >= 8192
