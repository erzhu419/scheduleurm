"""Only the eight explicitly eliminated routing scans may change."""
from copy import deepcopy

import pytest

from algorithm.experiments import native_root_dual_face_development_summary as summary
from algorithm.experiments.early_root_canonical_development_summary import LEGACY_FIELDS
from skill.tests.test_dual_face_root_canonical_development_summary import records as face_records


@pytest.fixture
def records():
    protocol, rows = deepcopy(face_records.__wrapped__())
    protocol["experiment"] = "native_root_dual_face_development_v1"
    for index, (registered, state) in enumerate(zip(protocol["cells"][0]["states"], rows[0]["states"], strict=True)):
        registered.update(native_root_source_index=index, native_root_source_state_id=state["state_id"])
        state.update(registered)
        state["previous_dual_face_routed"] = deepcopy(state["routed"])
        state["previous_dual_face_traces"] = deepcopy(state["traces"]["new"])
        for name in summary.REMOVED_ROUTING_COUNTERS:
            state["routed"]["routing_preparation"][name] = 0
    return protocol, rows


def test_full_path_and_real_work_match_with_eight_scans_removed(records):
    result = summary.summarize(*records)
    assert result["decision"] == "NATIVE_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE"
    assert all(result[name] for name in ("execution_checks_passed", "references_complete", "quality_gate_passed",
        "work_gate_passed", "development_gate_passed", "behavior_preserved", "prior_hold_retained"))
    assert result["state_count"] == result["executed_selector_calls"] == result["native_search_constructions"] == 12
    assert not result["independent_confirmation"] and result["executed_native_advance_calls"] == 0
    assert result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert result["borrowed_dual_face_native_snapshots"] == 48
    assert not any(result["executed_routing_preparation_work"][name] for name in summary.REMOVED_ROUTING_COUNTERS)
    assert all(result["removed_routing_preparation_work"][name] > 0 for name in summary.REMOVED_ROUTING_COUNTERS)
    assert result["executed"]["whole_selector"]["search_work"] == result["borrowed"]["whole_selector"]["face"]["search_work"]


@pytest.mark.parametrize("kind", ("root_lp", "preprocessing", "probe", "probe_stage", "evidence_work", "action", "native_trace"))
def test_any_remaining_action_path_or_fee_change_is_invalid(records, kind):
    state = records[1][0]["states"][2]
    routed = state["routed"]
    if kind == "root_lp":
        routed["root_lp"]["selection"]["lp_node_budget"] = 144
    elif kind == "preprocessing":
        routed["final_lp"]["preprocessing"]["requirement_group_count"] += 1
    elif kind == "probe":
        routed["probe"]["completion"]["resource_coordinate_reads"] += 1
    elif kind == "probe_stage":
        routed["probe_stage"] = "root"
    elif kind == "evidence_work":
        routed["dual_face_work"]["capture_calls"] += 1
    elif kind == "action":
        routed["selection"]["duration"] += .25
    else:
        state["previous_dual_face_traces"]["144"]["selection"]["lp_calls"] += 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION" and not result["behavior_preserved"]
    assert not result["work_gate_passed"] and result["state_count"] == 12


@pytest.mark.parametrize("fee", ("option_visits", "root_snapshot_heap_entries_read"))
def test_cancelled_scans_and_retained_snapshot_fees_are_actual(records, fee):
    records[1][0]["states"][0]["routed"]["routing_preparation"][fee] += 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION"
    assert result["executed_routing_preparation_work"][fee] > 0


def test_matching_parent_nonclosure_preserves_complete_roster_and_all_failed_work(records):
    state = records[1][0]["states"][2]
    for field in ("routed", "previous_dual_face_routed"):
        routed, probe = state[field], state[field]["probe"]
        probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8184, dfs_states_visited=8181)
        probe["selection"] = deepcopy(probe["score_certified_selection"])
        routed.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    state["previous_early_routed"] = {key: value for key, value in state["previous_dual_face_routed"].items()
                                      if key not in ("dual_face_evidence", "dual_face_work")}
    state["previous_zero_routed"] = {name: state["previous_early_routed"][name] for name in LEGACY_FIELDS}
    result = summary.summarize(*records)
    assert result["decision"] == "NATIVE_ROOT_DUAL_FACE_DEVELOPMENT_HOLD"
    assert result["execution_checks_passed"] and result["work_gate_passed"] and result["behavior_preserved"]
    assert result["state_count"] == 12 and len(result["nonclosures"]) == 1
    assert result["executed"]["whole_selector"]["shared_integer_states"] >= 8192
