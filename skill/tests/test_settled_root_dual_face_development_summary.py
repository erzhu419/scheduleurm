"""Actual tiny settled boxes, phase fees, and complete-cohort HOLD accounting."""
from copy import deepcopy
from dataclasses import asdict

import pytest

import algorithm.native_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as native
import algorithm.settled_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as settled
import algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as zero
from algorithm.native_root_dual_face_frame_candidates import select_native_root_dual_face_frame
from algorithm.settled_root_dual_face_frame_candidates import select_settled_root_dual_face_frame
from algorithm.experiments import settled_root_dual_face_development_summary as summary
from algorithm.experiments.early_root_canonical_development_summary import LEGACY_FIELDS
from algorithm.experiments.score_face_completion_confirmation_summary import _key
from skill.tests.test_native_root_dual_face_development_summary import records as native_records
from skill.tests.test_settled_root_dual_face_frame_candidates import CATALOGUES, valid_tied_root_then_real_lp


@pytest.fixture
def records():
    protocol, rows = deepcopy(native_records.__wrapped__())
    protocol.update(experiment="settled_root_dual_face_development_v1", expected_root_settle_calls=0)
    rows[0]["root_settle_calls"] = 0
    for index, (registered, state) in enumerate(zip(protocol["cells"][0]["states"], rows[0]["states"], strict=True)):
        registered.update(settled_root_source_index=index, settled_root_source_state_id=state["state_id"])
        state.update(registered)
        state["previous_native_root_routed"] = deepcopy(state["routed"])
        state["previous_native_root_traces"] = deepcopy(state["traces"]["new"])
        state["routed"].update(settled_lp=None, root_settle_work=None)
    return protocol, rows


def tiny_records(monkeypatch, path):
    options, lanes = CATALOGUES[0], 2
    queue, budget = ((1, 3), 8192) if path == "closed" else ((2, 2), 0 if path == "exhausted" else 8192)
    if path != "closed":
        for engine in (native, settled, zero):
            monkeypatch.setattr(engine, "linprog", valid_tied_root_then_real_lp(engine.linprog, []))
    result = asdict(select_settled_root_dual_face_frame(options, queue, lanes, 144, 256, budget))
    previous = asdict(select_native_root_dual_face_frame(options, queue, lanes, 144, 256, budget))
    search = zero.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    traces = {str(cap): asdict(search.advance_to(cap)) for cap in (1, 2, 144, 256)}
    state = dict(queue=list(queue), reference=traces["256"]["selection"], caps=[1, 2, 144, 256],
        previous_native_root_routed=previous, previous_native_root_traces=traces,
        previous_zero_traces=traces, traces={"new": traces})
    protocol = dict(initial_lp_budget=144, lp_node_budget=256, integer_state_budget=budget)
    model = {"options": [asdict(option) for option in options]}
    return protocol, state, model, lanes, result, _key(state["reference"], queue, len(options))


@pytest.mark.parametrize("path", ("closed", "open", "exhausted"))
def test_real_settlement_raw_m_child_box_and_one_paid_probe(monkeypatch, path):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, path)
    _, errors, quality = summary._audit_settled_root_result(result, state, protocol, model, lanes, "candidate", expected)
    assert not errors and quality
    phase, work = result["root_settle_work"], summary._work([result], "candidate")
    assert phase["lp_calls_before"] == phase["lp_calls_after"] == result["root_lp"]["selection"]["lp_calls"]
    assert work["search_work"]["lp_calls"] == result["final_lp"]["selection"]["lp_calls"]
    assert set(work["root_settle_work"]) == set(summary.ROOT_SETTLE_FEE_COUNTERS)
    assert work["root_settle_work"]["calls"] == work["root_settle_work"]["snapshot_calls"] == 1
    assert "popped_regions" not in work["root_settle_work"] and work["root_settle_diagnostics"]["split_regions"] == 1
    if path == "closed":
        assert result["status"] == "LP_ROOT_SETTLE_COMPLETE" and result["probe"] is None
        assert work["root_probe_calls"] == work["shared_integer_states"] == 0
        assert result["final_lp"]["selection"]["lp_node_budget"] == 1
        assert phase["pruned_regions"] == 0 and result["settled_lp"]["selection"]["bound_pruned_regions"] > 0
    else:
        box, = result["dual_face_evidence"]["regions"]
        assert box["lower"] != box["certificate_lower"] or box["upper"] != box["certificate_upper"]
        assert work["root_probe_calls"] == 1 and result["probe"]["baseline_selection"] == result["settled_lp"]["selection"]
        assert (result["probe"]["completion"]["status"] == "STATE_BUDGET_EXHAUSTED") == (path == "exhausted")
        assert work["root_probe_exhausted_calls"] == int(path == "exhausted")
        assert result["status"] == ("LP_INITIAL_COMPLETE" if path == "exhausted" else "INTEGER_PROBE_COMPLETE")


def test_real_phase_raw_cap_certificate_and_fee_mutations_are_rejected(monkeypatch):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, "open")
    for change in ("cap", "extra_lp", "raw_heap", "own_numerator", "face_fee"):
        changed = deepcopy(result)
        if change == "cap":
            changed["root_lp"]["selection"]["lp_node_budget"] = 144
        elif change == "extra_lp":
            changed["settled_lp"]["selection"]["lp_calls"] += 1
        elif change == "raw_heap":
            changed["root_settle_work"]["frontier_regions_before"] += 1
        elif change == "own_numerator":
            changed["dual_face_evidence"]["regions"][0]["own_numerator_fraction"] = (5, 1)
        else:
            changed["dual_face_work"]["face_bound_values_written"] += 1
        _, errors, _ = summary._audit_settled_root_result(changed, state, protocol, model, lanes, "candidate", expected)
        assert errors, change


def test_complete_roster_retains_true_parent_controls_and_no_extra_native_calls(records):
    result = summary.summarize(*records)
    assert result["decision"] == "SETTLED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE"
    assert all(result[name] for name in ("execution_checks_passed", "references_complete", "quality_gate_passed",
        "work_gate_passed", "development_gate_passed", "source_native_behavior_preserved", "prior_hold_retained"))
    assert result["state_count"] == result["executed_selector_calls"] == result["native_search_constructions"] == 12
    assert result["executed_native_advance_calls"] == result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert result["executed_root_settle_calls"] == 0 and not any(result["executed_root_settle_work"].values())
    assert result["borrowed_native_root_snapshots"] == 48 and "native" in result["borrowed"]["whole_selector"]
    assert result["executed"]["whole_selector"]["search_work"] == result["borrowed"]["whole_selector"]["native"]["search_work"]


@pytest.mark.parametrize("change", ("untriggered_phase", "parent_snapshot", "row_calls", "registered_calls"))
def test_untriggered_fields_and_actual_phase_coverage_are_not_free(records, change):
    protocol, rows = records
    state = rows[0]["states"][0]
    if change == "untriggered_phase":
        state["routed"]["root_settle_work"] = dict.fromkeys(summary.ROOT_SETTLE_COUNTERS, 0)
        state["routed"]["root_settle_work"]["calls"] = 1
    elif change == "parent_snapshot":
        state["previous_native_root_traces"]["144"]["selection"]["lp_calls"] += 1
    elif change == "row_calls":
        rows[0]["root_settle_calls"] = 1
    else:
        protocol["expected_root_settle_calls"] = 1
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["state_count"] == 12
    assert not result["work_gate_passed"]


def test_matching_parent_nonclosure_keeps_full_roster_and_all_exhausted_work(records):
    state = records[1][0]["states"][2]
    for field in ("routed", "previous_native_root_routed", "previous_dual_face_routed"):
        routed, probe = state[field], state[field]["probe"]
        probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8184, dfs_states_visited=8181)
        probe["selection"] = deepcopy(probe["score_certified_selection"])
        routed.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    state["previous_early_routed"] = {name: value for name, value in state["previous_dual_face_routed"].items()
                                      if name not in ("dual_face_evidence", "dual_face_work")}
    state["previous_zero_routed"] = {name: state["previous_early_routed"][name] for name in LEGACY_FIELDS}
    result = summary.summarize(*records)
    assert result["decision"] == "SETTLED_ROOT_DUAL_FACE_DEVELOPMENT_HOLD"
    assert result["execution_checks_passed"] and result["work_gate_passed"]
    assert result["state_count"] == 12 and len(result["nonclosures"]) == 1
    assert result["executed"]["whole_selector"]["shared_integer_states"] >= 8192
