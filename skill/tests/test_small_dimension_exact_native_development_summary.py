"""Full exact LP certificates and separate native/whole checkpoint ledgers."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from types import SimpleNamespace

import pytest

import algorithm.small_dimension_exact_native_kernel as native
from algorithm.small_dimension_exact_native_frame_candidates import select_small_dimension_exact_native_frame
from algorithm.experiments import small_dimension_exact_native_development_summary as summary
from skill.tests.test_direct_probe_own_tight_native_development_summary import (
    records as direct_records, tiny_records as direct_tiny,
)
from skill.tests.test_own_tight_dual_native_development_summary import lp_factory


def new_records(monkeypatch, path):
    protocol, state, model, lanes, previous, expected = direct_tiny(monkeypatch, path)
    state.update(previous_direct_probe_routed=previous,
        previous_direct_probe_native_root=state["candidate_native_root"],
        previous_direct_probe_native_traces=state["candidate_native_traces"])
    mode = "face" if path == "face_exhausted" else "weak" if path == "score_open" else "real"
    options = tuple(SimpleNamespace(**option) for option in model["options"])
    real = native.linprog
    monkeypatch.setattr(native, "linprog", lp_factory(real, mode))
    engine = native.SmallDimensionExactNativeSearch(options, state["queue"], lanes)
    root, settled, phase = engine.advance_roots_and_settle(state["caps"][0])
    state["candidate_native_root"] = {"root_lp": asdict(root), "settled_lp": asdict(settled) if settled is not None else None,
        "root_settle_work": asdict(phase) if phase is not None else None}
    state["candidate_native_traces"] = {str(state["caps"][0]): asdict(settled if settled is not None else root)}
    state["candidate_native_traces"].update({str(cap): asdict(engine.advance_to(cap)) for cap in state["caps"][1:]})
    monkeypatch.setattr(native, "linprog", lp_factory(real, mode))
    result = asdict(select_small_dimension_exact_native_frame(options, state["queue"], lanes,
        protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"]))
    return protocol, state, model, lanes, result, expected


@pytest.mark.parametrize("path", ("closed", "open", "face_exhausted", "score_open"))
def test_real_geometry_roots_continuations_and_both_actual_copy_ledgers(monkeypatch, path):
    protocol, state, model, lanes, result, expected = new_records(monkeypatch, path)
    _, errors, quality = summary._audit_small_dimension_result(result, state, protocol, model, lanes, "candidate", expected)
    assert not errors and quality, errors
    work = summary._work([result], "candidate")
    calls = work["small_dimension_lp_diagnostics"]
    assert calls["generic_lp_calls"] + calls["exact_lp_calls"] == work["search_work"]["lp_calls"]
    for field in ("native_own_tight_work", "small_dimension_lp_work"):
        assert work[field]["event_snapshot_entries_copied"] == result["final_lp"][field]["event_snapshot_entries_copied"]
    if path == "open":
        assert result["root_lp"]["selection"] != state["previous_direct_probe_routed"]["root_lp"]["selection"]
        assert result["root_lp"]["small_dimension_lp_evidence"][0]["method"] == "exact_2d"
    if path in ("face_exhausted", "score_open"):
        assert calls["generic_lp_calls"] > 0 and calls["exact_lp_calls"] > 0
        assert result["final_lp"]["native_own_tight_evidence"]
        assert result["final_lp"]["native_own_tight_work"]["event_snapshot_entries_copied"] < state["candidate_native_traces"]["144"]["native_own_tight_work"]["event_snapshot_entries_copied"]
    if path == "face_exhausted":
        assert result["probe"]["completion"]["status"] == "STATE_BUDGET_EXHAUSTED"
        assert result["root_settle_work"]["split_regions"] == 0 and result["final_lp"]["selection"]["branch_splits"] > 0
        assert result["final_lp"]["small_dimension_lp_work"]["event_snapshot_entries_copied"] < state["candidate_native_traces"]["144"]["small_dimension_lp_work"]["event_snapshot_entries_copied"]
    if path == "score_open":
        assert not result["root_lp"]["selection"]["score_exact"] and result["settled_lp"] is None


def test_zero_active_and_fixed_column_certificate_prove_the_original_lp():
    model, queue, lanes = {"options": [dict(requirements=(1, 0), duration=1.), dict(requirements=(0, 1), duration=1.)]}, (1, 5), 2
    event = dict(lp_call_index=1, method="exact_0d", threshold_fraction=(1, 1), eligible=(0, 1),
        requirements=((1, 0), (0, 1)), lower=(1, 0), upper=(1, 0), active_positions=(),
        base_objective=1, objective=(1, 5), primal_fractions=((1, 1), (0, 1)),
        price_fractions=((0, 1), (0, 1), (0, 1), (1, 1), (5, 1)), numerator_fraction=(1, 1))
    assert not summary._small_event_errors(event, model, queue, lanes)
    changed = deepcopy(event)
    changed["price_fractions"] = ((0, 1),) * 5
    assert summary._small_event_errors(changed, model, queue, lanes)


def test_bad_primal_full_fixed_dual_strong_equality_and_fees_are_rejected(monkeypatch):
    protocol, state, model, lanes, result, expected = new_records(monkeypatch, "face_exhausted")
    for change in ("primal", "negative_price", "fixed_column", "strong", "projection_fee", "active_fee", "whole_copy", "audit_copy", "cap"):
        changed, changed_state = deepcopy(result), deepcopy(state)
        event = changed["final_lp"]["small_dimension_lp_evidence"][0]
        if change == "primal":
            event["primal_fractions"] = ((99, 1), *event["primal_fractions"][1:])
        elif change == "negative_price":
            event["price_fractions"] = ((-1, 1), *event["price_fractions"][1:])
        elif change == "fixed_column":
            event["objective"] = (*event["objective"][:-1], 99)
        elif change == "strong":
            event["numerator_fraction"] = (7, 1)
        elif change == "projection_fee":
            changed["final_lp"]["small_dimension_lp_work"]["primal_inequality_checks"] += 1
        elif change == "active_fee":
            changed["final_lp"]["small_dimension_lp_work"]["active_columns_total"] += 1
        elif change == "whole_copy":
            changed["final_lp"]["small_dimension_lp_work"]["event_snapshot_entries_copied"] += 1
        elif change == "audit_copy":
            changed_state["candidate_native_traces"]["256"]["small_dimension_lp_work"]["event_snapshot_entries_copied"] -= 1
        else:
            changed["root_lp"]["selection"]["lp_node_budget"] = 144
        _, errors, _ = summary._audit_small_dimension_result(changed, changed_state, protocol, model, lanes, "candidate", expected)
        assert errors, change
    event = deepcopy(result["final_lp"]["small_dimension_lp_evidence"][0])
    primal = [Fraction(*pair) for pair in event["primal_fractions"]]
    assert any(left == right == value for left, right, value in zip(event["lower"], event["upper"], primal, strict=True))


def test_direct_control_uses_the_genuine_parent_native_context(monkeypatch):
    protocol, state, model, lanes, result, expected = new_records(monkeypatch, "face_exhausted")
    previous = state["previous_direct_probe_routed"]
    _, errors, quality = summary._audit_small_dimension_result(previous, state, protocol, model, lanes, "direct", expected)
    assert not errors and quality, errors
    assert "small_dimension_lp_work" not in summary._work([previous], "direct")
    state["previous_direct_probe_native_traces"] = state["candidate_native_traces"]
    _, errors, _ = summary._audit_small_dimension_result(previous, state, protocol, model, lanes, "direct", expected)
    assert errors


@pytest.fixture
def records():
    protocol, rows = deepcopy(direct_records.__wrapped__())
    protocol.update(experiment="small_dimension_exact_native_development_v1", parent_root_settle_calls=0)
    protocol.pop("expected_root_settle_calls", None)
    models = {model["node"]: model for model in protocol["models"]}
    for row, cell in zip(rows, protocol["cells"], strict=True):
        options = tuple(SimpleNamespace(**option) for option in models[row["node"]]["options"])
        for index, (registered, state) in enumerate(zip(cell["states"], row["states"], strict=True)):
            registered.update(small_dimension_source_index=index, small_dimension_source_state_id=state["state_id"])
            state.update(registered)
            state.update(previous_direct_probe_routed=deepcopy(state["routed"]),
                previous_direct_probe_native_root=deepcopy(state["candidate_native_root"]),
                previous_direct_probe_native_traces=deepcopy(state["candidate_native_traces"]))
            engine = native.SmallDimensionExactNativeSearch(options, state["queue"], row["lanes"])
            root, settled, phase = engine.advance_roots_and_settle(state["caps"][0])
            state["candidate_native_root"] = {"root_lp": asdict(root), "settled_lp": asdict(settled) if settled is not None else None,
                "root_settle_work": asdict(phase) if phase is not None else None}
            state["candidate_native_traces"] = {str(state["caps"][0]): asdict(settled if settled is not None else root)}
            state["candidate_native_traces"].update({str(cap): asdict(engine.advance_to(cap)) for cap in state["caps"][1:]})
            state["routed"] = asdict(select_small_dimension_exact_native_frame(options, state["queue"], row["lanes"],
                protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"]))
        row["root_settle_calls"] = sum(state["routed"]["root_settle_work"] is not None for state in row["states"])
    return protocol, rows


def test_complete_toy_roster_preserves_controls_and_counts_both_real_ledgers_once(records):
    result = summary.summarize(*records)
    assert result["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_DEVELOPMENT_ADVANCE", result["failures"]
    assert all(result[name] for name in ("execution_checks_passed", "references_complete", "quality_gate_passed", "work_gate_passed", "development_gate_passed", "prior_hold_retained"))
    assert result["state_count"] == result["executed_selector_calls"] == 12
    assert result["executed_native_advance_calls"] == 48 and result["executed_interface_calls"] == 60
    assert result["native_search_constructions"] == 24 and result["executed_root_settle_calls"] == 0
    assert result["executed_small_dimension_lp_diagnostics"]["exact_one_calls"] == 11
    assert result["executed_small_dimension_lp_work"]["event_snapshot_entries_copied"] == 11
    assert result["executed"]["native"]["small_dimension_lp_work"]["event_snapshot_entries_copied"] == 44
    assert "small_dimension_lp_work" not in result["borrowed"]["whole_selector"]["direct"]


def test_invalid_exact_proof_keeps_the_complete_roster_and_prior_hold(records):
    protocol, rows = records
    rows[0]["states"][0]["routed"]["root_lp"]["small_dimension_lp_evidence"][0]["numerator_fraction"] = (0, 1)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["state_count"] == 12
    assert not result["work_gate_passed"] and result["prior_hold_retained"]
