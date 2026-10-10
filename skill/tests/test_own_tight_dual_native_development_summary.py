"""Own-tight proof and separately paid whole/native checkpoint integration."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import algorithm.fused_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as frozen
import algorithm.own_tight_dual_native_kernel as native
from algorithm.fused_root_dual_face_frame_candidates import select_fused_root_dual_face_frame
from algorithm.own_tight_dual_native_frame_candidates import select_own_tight_dual_native_frame
from algorithm.experiments import own_tight_dual_native_development_summary as summary
from skill.tests.test_fused_root_dual_face_development_summary import records as fused_records
from skill.tests.test_own_tight_dual_native_frame_candidates import Option, TIED, argmax


def lp_factory(real, mode):
    calls = []

    def solve(*args, **kwargs):
        calls.append(1)
        if mode == "tie" and len(calls) == 1:
            return SimpleNamespace(success=True, x=(0., 2.),
                ineqlin=SimpleNamespace(marginals=(-2., 0., 0., 0., 0.)))
        if mode == "face" and len(calls) == 1:
            return SimpleNamespace(success=True, x=(0., 2., 0.),
                ineqlin=SimpleNamespace(marginals=(-3., 0., 0., 0., 0., 0., 0.)))
        if mode == "weak" and len(calls) <= 2:
            count, cap_price = (2., -1.) if len(calls) == 1 else (1., 0.)
            return SimpleNamespace(success=True, x=(0., count, 0., 0.),
                ineqlin=SimpleNamespace(marginals=(-5., 0., 0., 0., cap_price, 0., 0., -17.)))
        return real(*args, **kwargs)

    return solve


def tiny_records(monkeypatch, path):
    if path in ("face_exhausted", "score_open"):
        options = tuple(Option(tuple(int(c == index) for c in range(3)), 1.) for index in range(3))
        queue, mode, budget = ((3, 3, 1), "face", 0) if path == "face_exhausted" else ((5, 5, 1), "weak", 8192)
        if path == "score_open":
            options += (Option((2, 2, 2), 1.),)
    else:
        options, queue, mode = TIED, (1, 3) if path == "closed" else (2, 2), "real" if path == "closed" else "tie"
        budget = 0 if path == "exhausted" else 8192
    real = native.linprog
    monkeypatch.setattr(frozen, "linprog", lp_factory(real, mode))
    previous = asdict(select_fused_root_dual_face_frame(options, queue, 2, 144, 256, budget))
    monkeypatch.setattr(native, "linprog", lp_factory(real, mode))
    engine = native.OwnTightDualNativeSearch(options, queue, 2)
    raw, settled, phase = engine.advance_roots_and_settle(1)
    checkpoints = {"root_lp": asdict(raw), "settled_lp": asdict(settled) if settled is not None else None,
                   "root_settle_work": asdict(phase) if phase is not None else None}
    traces = {"1": asdict(settled if settled is not None else raw)}
    traces.update({str(cap): asdict(engine.advance_to(cap)) for cap in (2, 144, 256)})
    monkeypatch.setattr(native, "linprog", lp_factory(real, mode))
    result = asdict(select_own_tight_dual_native_frame(options, queue, 2, 144, 256, budget))
    protocol = dict(initial_lp_budget=144, lp_node_budget=256, integer_state_budget=budget)
    state = dict(queue=list(queue), caps=[1, 2, 144, 256], reference=traces["256"]["selection"],
        previous_fused_root_routed=previous, candidate_native_root=checkpoints,
        candidate_native_traces=traces, traces={"new": {}})
    model = {"options": [asdict(option) for option in options]}
    return protocol, state, model, 2, result, argmax(options, queue, 2)


@pytest.mark.parametrize("path", ("closed", "open", "exhausted", "face_exhausted", "score_open"))
def test_true_whole_four_prefix_and_own_face_paths(monkeypatch, path):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, path)
    _, errors, quality = summary._audit_own_tight_result(result, state, protocol, model, lanes, "candidate", expected)
    assert not errors and quality, errors
    work = summary._work([result], "candidate")
    assert work["search_work"]["lp_calls"] == result["final_lp"]["selection"]["lp_calls"]
    assert work["native_own_tight_work"]["event_snapshot_entries_copied"] == result["final_lp"]["native_own_tight_work"]["event_snapshot_entries_copied"]
    assert len(summary.OWN_TIGHT_COUNTERS) == 33 and len(summary.OWN_TIGHT_FEE_COUNTERS) == 28
    assert summary._without_own(result["root_lp"]) == state["previous_fused_root_routed"]["root_lp"]
    assert result["root_lp"]["native_own_tight_work"]["own_bound_cache_writes"] > 0
    if path == "face_exhausted":
        assert result["probe_stage"] == "root" and result["probe"]["completion"]["status"] == "STATE_BUDGET_EXHAUSTED"
        assert result["final_lp"]["native_own_tight_evidence"]
        assert result["final_lp"]["native_own_tight_work"]["event_snapshot_entries_copied"] < state["candidate_native_traces"]["144"]["native_own_tight_work"]["event_snapshot_entries_copied"]
    if path == "score_open":
        assert not result["root_lp"]["selection"]["score_exact"] and result["settled_lp"] is None
        assert result["final_lp"]["native_own_tight_evidence"]
        assert result["final_lp"]["native_own_tight_work"]["event_snapshot_entries_copied"] < state["candidate_native_traces"]["144"]["native_own_tight_work"]["event_snapshot_entries_copied"]


def test_actual_own_objective_current_box_direction_and_fees_cannot_be_mutated(monkeypatch):
    protocol, state, model, lanes, result, expected = tiny_records(monkeypatch, "face_exhausted")
    for name in ("own", "source", "direction", "cache_fee", "evidence_fee", "whole_copy", "audit_copy", "raw_m"):
        changed, changed_state = deepcopy(result), deepcopy(state)
        if name in ("own", "source", "direction"):
            event = changed["final_lp"]["native_own_tight_evidence"][0]
            if name == "own":
                event["current_own_numerator_fraction"] = (7, 1)
            elif name == "source":
                event["certificate_numerator_fraction"] = (7, 1)
            else:
                event["face_upper"] = event["before_upper"]
            assert summary._event_errors(event, model, state["queue"], lanes), name
        elif name == "cache_fee":
            changed["final_lp"]["native_own_tight_work"]["own_bound_cache_writes"] += 1
        elif name == "evidence_fee":
            changed["final_lp"]["native_own_tight_work"]["event_fraction_pairs_written"] += 1
        elif name == "whole_copy":
            changed["final_lp"]["native_own_tight_work"]["event_snapshot_entries_copied"] += 1
        elif name == "audit_copy":
            changed_state["candidate_native_traces"]["256"]["native_own_tight_work"]["event_snapshot_entries_copied"] -= 1
        else:
            changed["root_lp"]["selection"]["lp_node_budget"] = 144
        _, errors, _ = summary._audit_own_tight_result(changed, changed_state, protocol, model, lanes, "candidate", expected)
        assert errors, name


@pytest.fixture
def records():
    protocol, rows = deepcopy(fused_records.__wrapped__())
    count = sum(len(row["states"]) for row in rows)
    protocol.update(experiment="own_tight_dual_native_development_v1", expected_native_advance_calls=4 * count,
        expected_native_constructions=2 * count, expected_interface_calls=5 * count, expected_candidate_native_snapshots=4 * count)
    for row, cell in zip(rows, protocol["cells"], strict=True):
        row.update(native_advance_calls=4 * len(row["states"]), native_search_constructions=2 * len(row["states"]))
        for index, (registered, state) in enumerate(zip(cell["states"], row["states"], strict=True)):
            registered.update(own_tight_source_index=index, own_tight_source_state_id=state["state_id"])
            state.update(registered)
            state["previous_fused_root_routed"] = deepcopy(state["routed"])
            state["previous_fused_root_traces"] = deepcopy(state["previous_zero_traces"])

            def annotate(stage):
                work = dict.fromkeys(summary.OWN_TIGHT_COUNTERS, 0)
                work["own_bound_cache_reads"] = stage["selection"]["lp_calls"] + stage["selection"]["branch_splits"]
                work["own_bound_cache_writes"] = work["own_bound_cache_reads"] + stage["selection"]["branch_splits"] + stage["dual_reuse"]["child_bound_evaluations"]
                return {**deepcopy(stage), "native_own_tight_work": work, "native_own_tight_evidence": []}

            routed = state["routed"]
            for name in ("root_lp", "initial_lp", "final_lp"):
                routed[name] = annotate(routed[name])
            state["candidate_native_root"] = {"root_lp": deepcopy(routed["root_lp"]), "settled_lp": None, "root_settle_work": None}
            state["candidate_native_traces"] = {cap: annotate(stage) for cap, stage in state["previous_zero_traces"].items()}
    return protocol, rows


def test_complete_mock_roster_counts_two_actual_searches_and_final_work_once(records):
    result = summary.summarize(*records)
    assert result["decision"] == "OWN_TIGHT_DUAL_NATIVE_DEVELOPMENT_ADVANCE", result["failures"]
    assert all(result[name] for name in ("execution_checks_passed", "references_complete", "quality_gate_passed", "work_gate_passed", "development_gate_passed", "prior_hold_retained"))
    count = result["state_count"]
    assert result["executed_selector_calls"] == count == 12
    assert result["executed_native_advance_calls"] == result["executed_candidate_native_snapshots"] == 4 * count
    assert result["native_search_constructions"] == 2 * count and result["executed_interface_calls"] == 5 * count
    assert result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert "native_own_tight_work" in result["executed"]["native"]
    assert "native_own_tight_work" not in result["borrowed"]["whole_selector"]["fused"]


def test_native_actual_calls_and_snapshot_fee_mutations_hold_full_roster(records):
    protocol, rows = records
    rows[0]["native_advance_calls"] -= 1
    rows[0]["states"][0]["candidate_native_traces"]["256"]["native_own_tight_work"]["event_snapshot_entries_copied"] = 1
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["state_count"] == 12
    assert not result["work_gate_passed"]


def test_nonclosed_sole_probe_keeps_all_states_and_full_exhausted_cost(records):
    protocol, rows = records
    state = rows[0]["states"][2]
    result, probe = state["routed"], state["routed"]["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    result.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "OWN_TIGHT_DUAL_NATIVE_DEVELOPMENT_HOLD", result["failures"]
    assert result["execution_checks_passed"] and result["work_gate_passed"]
    assert result["state_count"] == 12 and len(result["nonclosures"]) == 1
    assert result["executed"]["whole_selector"]["shared_integer_states"] >= 8192
