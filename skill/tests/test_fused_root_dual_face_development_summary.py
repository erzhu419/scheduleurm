"""Preserve real tiny settled paths and independently charge fused snapshots."""
from copy import deepcopy
from dataclasses import asdict

import pytest

import algorithm.fused_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as fused
from algorithm.fused_root_dual_face_frame_candidates import select_fused_root_dual_face_frame
from algorithm.experiments import fused_root_dual_face_development_summary as summary
from algorithm.experiments.early_root_canonical_development_summary import LEGACY_FIELDS
from skill.tests.test_settled_root_dual_face_development_summary import records as settled_records, tiny_records as settled_tiny
from skill.tests.test_settled_root_dual_face_frame_candidates import CATALOGUES, valid_tied_root_then_real_lp


@pytest.fixture
def records():
    protocol, rows = deepcopy(settled_records.__wrapped__())
    protocol["experiment"] = "fused_root_dual_face_development_v1"
    for index, (registered, state) in enumerate(zip(protocol["cells"][0]["states"], rows[0]["states"], strict=True)):
        registered.update(fused_root_source_index=index, fused_root_source_state_id=state["state_id"])
        state.update(registered)
        state["previous_settled_root_routed"] = deepcopy(state["routed"])
        state["previous_settled_root_traces"] = deepcopy(state["traces"]["new"])
        state["routed"]["routing_preparation"]["root_snapshot_heap_entries_read"] //= 2
        state["routed"]["routing_preparation"]["root_snapshot_retained_entries_read"] = 0
    return protocol, rows


def tiny_records(monkeypatch, path):
    protocol, state, model, lanes, previous, expected = settled_tiny(monkeypatch, path)
    state["previous_settled_root_routed"] = previous
    state["previous_settled_root_traces"] = state["previous_zero_traces"]
    if path != "closed":
        monkeypatch.setattr(fused, "linprog", valid_tied_root_then_real_lp(fused.linprog, []))
    result = asdict(select_fused_root_dual_face_frame(CATALOGUES[0], state["queue"], lanes,
        protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"]))
    return protocol, state, model, lanes, result, expected


@pytest.mark.parametrize("path", ("closed", "open", "exhausted"))
def test_true_raw_settled_probe_and_continuation_paths_only_change_scan_fees(monkeypatch, path):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, path)
    _, errors, quality = summary._audit_fused_root_result(result, state, protocol, model, lanes, "candidate", expected)
    assert not errors and quality
    phase, previous = result["root_settle_work"], state["previous_settled_root_routed"]
    assert result["root_lp"] == previous["root_lp"] and result["settled_lp"] == previous["settled_lp"]
    assert result["selection"] == previous["selection"] and result["probe"] == previous["probe"]
    assert result["routing_preparation"]["root_snapshot_heap_entries_read"] == phase["frontier_regions_before"]
    assert phase["snapshot_heap_entries_read"] == phase["frontier_regions_after"]
    assert result["routing_preparation"]["root_snapshot_retained_entries_read"] == phase["snapshot_retained_entries_read"] == 0
    new, old = summary._work([result], "candidate"), summary._work([previous], "settled")
    assert new["search_work"] == old["search_work"] and new["shared_integer_states"] == old["shared_integer_states"]
    assert new["root_settle_work"]["snapshot_heap_entries_read"] * 2 == old["root_settle_work"]["snapshot_heap_entries_read"]
    assert new["root_settle_diagnostics"] == old["root_settle_diagnostics"]
    if path == "closed":
        assert result["status"] == "LP_ROOT_SETTLE_COMPLETE" and result["probe"] is None
    if path == "exhausted":
        assert result["probe"]["completion"]["status"] == "STATE_BUDGET_EXHAUSTED"
        assert result["status"] == "LP_INITIAL_COMPLETE" and new["root_probe_exhausted_calls"] == 1


def test_true_fused_checkpoint_fee_action_and_native_work_changes_are_rejected(monkeypatch):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, "open")
    for change in ("root_scan", "retained_scan", "settled_scan", "phase_head", "raw_m", "canonical", "affine_fee"):
        changed = deepcopy(result)
        if change == "root_scan":
            changed["routing_preparation"]["root_snapshot_heap_entries_read"] += 1
        elif change == "retained_scan":
            changed["routing_preparation"]["root_snapshot_retained_entries_read"] = 1
        elif change == "settled_scan":
            changed["root_settle_work"]["snapshot_heap_entries_read"] += 1
        elif change == "phase_head":
            changed["root_settle_work"]["head_checks"] += 1
        elif change == "raw_m":
            changed["root_lp"]["selection"]["lp_node_budget"] = 144
        elif change == "canonical":
            changed["selection"]["canonical_exact"] = False
        else:
            changed["final_lp"]["affine_dual"]["price_reads"] += 1
        _, errors, _ = summary._audit_fused_root_result(changed, state, protocol, model, lanes, "candidate", expected)
        assert errors, change


def test_full_mock_roster_retains_controls_and_actual_new_scan_ledger(records):
    result = summary.summarize(*records)
    assert result["decision"] == "FUSED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE"
    assert all(result[name] for name in ("execution_checks_passed", "references_complete", "quality_gate_passed",
        "work_gate_passed", "development_gate_passed", "behavior_preserved", "prior_hold_retained"))
    assert result["state_count"] == result["executed_selector_calls"] == result["native_search_constructions"] == 12
    assert result["executed_native_advance_calls"] == result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert result["borrowed_settled_root_snapshots"] == 48
    current, previous = result["executed"]["whole_selector"], result["borrowed"]["whole_selector"]["settled"]
    assert current["search_work"] == previous["search_work"] and current["shared_integer_states"] == previous["shared_integer_states"]
    assert current["routing_preparation"]["root_snapshot_heap_entries_read"] * 2 == previous["routing_preparation"]["root_snapshot_heap_entries_read"]
    assert current["routing_preparation"]["root_snapshot_retained_entries_read"] == 0
    assert result["removed_snapshot_reads"]["raw_root"]["root_snapshot_retained_entries_read"] > 0


@pytest.mark.parametrize("change", ("untriggered_phase", "native_trace", "root_snapshot", "late_probe"))
def test_untriggered_and_late_paths_preserve_all_records_and_fees(records, change):
    state = records[1][0]["states"][2]
    if change == "untriggered_phase":
        state["routed"]["root_settle_work"] = dict.fromkeys(summary.ROOT_SETTLE_COUNTERS, 0)
    elif change == "native_trace":
        state["previous_settled_root_traces"]["144"]["selection"]["lp_calls"] += 1
    elif change == "root_snapshot":
        state["routed"]["root_lp"]["selection"]["canonical_key_bound_evaluations"] += 1
    else:
        state["routed"]["probe"]["completion"]["resource_coordinate_reads"] += 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION" and result["state_count"] == 12
    assert not result["behavior_preserved"] and not result["work_gate_passed"]


def test_matching_parent_nonclosure_keeps_all8192_work_and_roster(records):
    state = records[1][0]["states"][2]
    for field in ("routed", "previous_settled_root_routed", "previous_native_root_routed", "previous_dual_face_routed"):
        routed, probe = state[field], state[field]["probe"]
        probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8184, dfs_states_visited=8181)
        probe["selection"] = deepcopy(probe["score_certified_selection"])
        routed.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    state["previous_early_routed"] = {name: value for name, value in state["previous_dual_face_routed"].items()
                                      if name not in ("dual_face_evidence", "dual_face_work")}
    state["previous_zero_routed"] = {name: state["previous_early_routed"][name] for name in LEGACY_FIELDS}
    result = summary.summarize(*records)
    assert result["decision"] == "FUSED_ROOT_DUAL_FACE_DEVELOPMENT_HOLD"
    assert result["execution_checks_passed"] and result["work_gate_passed"] and result["behavior_preserved"]
    assert result["state_count"] == 12 and len(result["nonclosures"]) == 1
    assert result["executed"]["whole_selector"]["shared_integer_states"] >= 8192
