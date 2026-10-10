"""Audit real tiny roots and continuations, plus complete-cohort HOLD accounting."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import algorithm.early_root_canonical_frame_candidates as router
import algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as engine
from algorithm.experiments import early_root_canonical_development_summary as summary
from algorithm.experiments.score_face_completion_confirmation_summary import _key
from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import (
    select_zero_delta_affine_dual_residual_capacity_frontier_frame as select_zero,
)
from skill.tests.test_zero_delta_affine_development_summary import records as zero_records


def real_records(monkeypatch, path):
    pairs, queue, lanes, budget = (((1, 0), (0, 1)), (1, 1), 1, 8192)
    if path == "root_complete":
        queue, lanes = (2, 1), 2
    elif path == "early_exhausted":
        budget = 1  # Actually exhaust the tiny canonical search; native LP then closes.
    elif path == "open_root":
        pairs, queue, lanes = ((1, 0), (3, 0), (0, 1), (0, 3)), (4, 4), 3
        monkeypatch.setattr(engine, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False))
    options = tuple(SimpleNamespace(requirements=pair, duration=1.) for pair in pairs)
    protocol = dict(initial_lp_budget=144, lp_node_budget=256, integer_state_budget=budget)
    result = asdict(router.select_early_root_canonical_frame(options, queue, lanes, 144, 256, budget))
    search = engine.ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes)
    traces = {str(cap): asdict(search.advance_to(cap)) for cap in (1, 2, 144, 256)}
    state = dict(queue=list(queue), reference=deepcopy(result["selection"]), caps=[1, 2, 144, 256],
        traces={"new": traces}, previous_zero_traces=traces,
        previous_zero_routed=asdict(select_zero(options, queue, lanes, 144, 256, budget)))
    model = {"options": [vars(option) for option in options]}
    return protocol, state, model, lanes, result, _key(result["selection"], queue, len(options))


@pytest.mark.parametrize("path,stage,status", (("root_complete", "none", "LP_ROOT_COMPLETE"),
    ("early", "root", "INTEGER_PROBE_COMPLETE"), ("early_exhausted", "root", "LP_INITIAL_COMPLETE"),
    ("open_root", "none", "LP_INITIAL_COMPLETE")))
def test_real_root_actual_cap_native_ladder_and_unique_paid_probe(monkeypatch, path, stage, status):
    protocol, state, model, lanes, result, expected = real_records(monkeypatch, path)
    _, errors, quality = summary._audit_early_result(result, state, protocol, model, lanes, "candidate", expected)
    assert not errors and quality
    assert result["root_lp_budget"] == result["root_lp"]["selection"]["lp_node_budget"] == 1
    assert result["probe_stage"] == stage and result["status"] == status
    work = summary._work([result], "candidate")
    assert work["search_work"]["lp_calls"] == result["final_lp"]["selection"]["lp_calls"]
    assert work["routing_preparation"]["root_snapshot_calls"] == 1
    assert work["shared_integer_states"] == (0 if result["probe"] is None else
        result["probe"]["score_proof"]["states_visited"] + result["probe"]["completion"]["states_visited"])
    if path == "early_exhausted":
        assert work["root_probe_exhausted_calls"] == 1 and work["shared_integer_states"] == 1
        assert result["probe"]["completion"]["status"] == "STATE_BUDGET_EXHAUSTED"
        assert result["final_lp"]["selection"]["canonical_exact"] and result["probe_stage"] == "root"


@pytest.mark.parametrize("mutation", ("cap", "canonical_fee", "routing_fee"))
def test_real_root_cap_and_full_fees_cannot_be_normalized_away(monkeypatch, mutation):
    protocol, state, model, lanes, result, expected = real_records(monkeypatch, "early")
    if mutation == "cap":
        result["root_lp"]["selection"]["lp_node_budget"] = 144
    elif mutation == "canonical_fee":
        result["root_lp"]["selection"]["canonical_key_bound_evaluations"] += 1
    else:
        result["routing_preparation"]["duration_comparisons"] += 1
    _, errors, _ = summary._audit_early_result(result, state, protocol, model, lanes, "candidate", expected)
    assert errors


@pytest.fixture
def records():
    protocol, rows = deepcopy(zero_records.__wrapped__())
    states = rows[0]["states"]
    for state in states:
        state["original_affine_traces"] = state.pop("previous_dual_traces")
        state["original_affine_routed"] = state.pop("previous_dual_routed")
        for trace in (state["traces"]["old"], state["traces"]["new"], state["original_affine_traces"]):
            trace["1"]["selection"]["frontier_regions_remaining"] = min(1, trace["1"]["selection"]["frontier_regions_remaining"])
        state["previous_zero_traces"] = deepcopy(state["traces"]["new"])
        state["previous_zero_routed"] = deepcopy(state["routed"])
        result, root = state["routed"], deepcopy(state["traces"]["new"]["1"])
        result.update(root_lp=root, root_lp_budget=1, routing_preparation=dict.fromkeys(summary.ROUTING_COUNTERS, 1))
        result["routing_preparation"].update(root_snapshot_heap_entries_read=2 * root["selection"]["frontier_regions_remaining"],
            root_snapshot_retained_entries_read=2 * root["selection"]["frontier_regions_remaining"])
        if root["selection"]["canonical_exact"]:
            result.update(selection=deepcopy(root["selection"]), initial_lp=deepcopy(root), final_lp=deepcopy(root),
                probe=None, resumed=False, status="LP_ROOT_COMPLETE", probe_stage="none", frontier_evidence=None)
    count = len(states)
    protocol.update(experiment="pure_mock_early_root", expected_state_count=count, expected_selector_calls=count,
        expected_native_advance_calls=0, expected_native_constructions=count, expected_interface_calls=count)
    rows[0].update(candidate_calls=count, native_advance_calls=0, native_search_constructions=count,
        new_reference_calls=0, new_oracle_calls=0)
    return protocol, rows


def test_complete_cohort_has_one_cold_call_and_borrowed_native_work(records):
    result = summary.summarize(*records)
    assert result["decision"] == "EARLY_ROOT_CANONICAL_DEVELOPMENT_ADVANCE"
    assert result["work_gate_passed"] and result["execution_checks_passed"] and not result["independent_confirmation"]
    assert result["executed_selector_calls"] == result["native_search_constructions"] == result["state_count"] == 12
    assert result["executed_native_advance_calls"] == result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert set(result["borrowed"]["native"]) == {"zero", "affine", "branch"}
    assert result["executed_routing_preparation_work"]["root_snapshot_calls"] == 12


def test_nonclosure_retains_all_states_and_full_failed_probe_work(records):
    result = records[1][0]["states"][2]["routed"]
    probe = result["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False,
        states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    result.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    records[1][0]["states"][2]["previous_zero_routed"] = deepcopy({key: result[key] for key in summary.LEGACY_FIELDS})
    actual = summary.summarize(*records)
    assert actual["decision"] == "EARLY_ROOT_CANONICAL_DEVELOPMENT_HOLD"
    assert actual["execution_checks_passed"] and actual["work_gate_passed"] and not actual["quality_gate_passed"]
    assert actual["state_count"] == 12 and len(actual["nonclosures"]) == 1
    assert actual["executed"]["whole_selector"]["shared_integer_states"] >= 8192
