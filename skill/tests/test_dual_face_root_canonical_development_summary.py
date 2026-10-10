"""Real root-face algebra, complete accounting and unchanged control routes."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import algorithm.dual_face_root_canonical_frame_candidates as router
import algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates as engine
from algorithm.early_root_canonical_frame_candidates import select_early_root_canonical_frame
from algorithm.experiments import dual_face_root_canonical_development_summary as summary
from algorithm.experiments.early_root_canonical_development_summary import LEGACY_FIELDS
from algorithm.experiments.score_face_completion_confirmation_summary import _key
from algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates import ZeroDeltaAffineDualReuseIntegerCapacitySearch
from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import select_zero_delta_affine_dual_residual_capacity_frontier_frame
from skill.tests.test_early_root_canonical_development_summary import real_records as early_real, records as early_records


def real_records(monkeypatch, path):
    if path == "loose":
        real_lp = engine.linprog

        def weaker_dual(*args, **kwargs):
            result = real_lp(*args, **kwargs)
            prices = list(result.ineqlin.marginals)
            prices[0] -= 1  # Feasible own dual becomes loose; the inherited analytic bound stays tight.
            return SimpleNamespace(success=result.success, x=result.x, ineqlin=SimpleNamespace(marginals=prices))

        monkeypatch.setattr(engine, "linprog", weaker_dual)
    protocol, state, model, lanes, early, expected = early_real(monkeypatch, "early" if path in ("fixed", "loose") else path)
    if path == "fixed":
        model["options"] = [dict(requirements=pair, duration=1.) for pair in ((1, 0), (2, 0), (0, 1))]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        state["queue"] = [2, 2]
        early = asdict(select_early_root_canonical_frame(options, state["queue"], lanes, 144, 256, 8192))
        search = ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, state["queue"], lanes)
        state["previous_zero_traces"] = {str(cap): asdict(search.advance_to(cap)) for cap in state["caps"]}
        state["traces"]["new"] = state["previous_zero_traces"]
        state["previous_zero_routed"] = asdict(select_zero_delta_affine_dual_residual_capacity_frontier_frame(
            options, state["queue"], lanes, 144, 256, 8192))
        state["reference"] = state["previous_zero_traces"]["256"]["selection"]
        expected = _key(state["reference"], state["queue"], len(options))
    state["previous_early_routed"] = early
    options = tuple(SimpleNamespace(**option) for option in model["options"])
    result = asdict(router.select_dual_face_root_canonical_frame(options, state["queue"], lanes,
        protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"]))
    return protocol, state, model, lanes, result, expected


@pytest.mark.parametrize("path", ("root_complete", "early", "fixed", "loose", "early_exhausted", "open_root"))
def test_real_face_exact_math_actual_m_and_one_paid_probe(monkeypatch, path):
    protocol, state, model, lanes, result, expected = real_records(monkeypatch, path)
    _, errors, quality = summary._audit_dual_face_result(result, state, protocol, model, lanes, "candidate", expected)
    assert not errors and quality
    assert set(result["dual_face_work"]) == set(router.DUAL_FACE_COUNTERS) == set(summary.DUAL_FACE_COUNTERS)
    work = summary._work([result], "candidate")
    assert work["search_work"]["lp_calls"] == result["final_lp"]["selection"]["lp_calls"]
    assert work["dual_face_work"] == result["dual_face_work"]
    if path == "fixed":
        assert work["dual_face_work"]["fixed_lower_coordinates"] + work["dual_face_work"]["fixed_upper_coordinates"] > 0
    if path == "loose":
        assert not work["dual_face_work"]["fixed_lower_coordinates"] and not work["dual_face_work"]["fixed_upper_coordinates"]
        assert all(not box["tight"] and box["face_lower"] == box["lower"] and box["face_upper"] == box["upper"]
            for box in result["dual_face_evidence"]["regions"])
    if path == "early_exhausted":
        assert work["shared_integer_states"] == 1 and work["root_probe_exhausted_calls"] == 1
        assert result["probe"]["completion"]["status"] == "STATE_BUDGET_EXHAUSTED"
        assert result["status"] == "LP_INITIAL_COMPLETE"
    if path in ("root_complete", "open_root"):
        assert result["dual_face_evidence"] is None and not any(result["dual_face_work"].values())
        assert {key: value for key, value in result.items() if key not in summary.FACE_FIELDS} == state["previous_early_routed"]


@pytest.mark.parametrize("mutation", ("own_numerator", "face_bounds", "price_fee", "actual_m"))
def test_independent_face_algebra_raw_cap_and_fees_reject_mutations(monkeypatch, mutation):
    protocol, state, model, lanes, result, expected = real_records(monkeypatch, "early")
    box = result["dual_face_evidence"]["regions"][0]
    if mutation == "own_numerator":
        pair = box["own_numerator_fraction"]
        box["own_numerator_fraction"] = (pair[0] + pair[1], pair[1])
    elif mutation == "face_bounds":
        box["face_lower"] = (box["face_lower"][0] + 1, *box["face_lower"][1:])
    elif mutation == "price_fee":
        result["dual_face_work"]["certificate_price_reads"] += 1
    else:
        result["root_lp"]["selection"]["lp_node_budget"] = 144
    _, errors, _ = summary._audit_dual_face_result(result, state, protocol, model, lanes, "candidate", expected)
    assert errors


@pytest.fixture
def records():
    protocol, rows = deepcopy(early_records.__wrapped__())
    protocol["experiment"] = "pure_mock_dual_face_root"
    for state in rows[0]["states"]:
        state["previous_early_routed"] = deepcopy(state["routed"])
        state["routed"].update(dual_face_evidence=None, dual_face_work=dict.fromkeys(summary.DUAL_FACE_COUNTERS, 0))
    return protocol, rows


def test_complete_roster_advances_with_borrowed_early_and_zero_native_work(records):
    result = summary.summarize(*records)
    assert result["decision"] == "DUAL_FACE_ROOT_CANONICAL_DEVELOPMENT_ADVANCE"
    assert all(result[key] for key in ("execution_checks_passed", "references_complete", "quality_gate_passed",
        "work_gate_passed", "development_gate_passed"))
    assert not result["independent_confirmation"] and result["state_count"] == 12
    assert result["executed_native_advance_calls"] == result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert "early" in result["borrowed"]["whole_selector"]
    assert not any(result["executed_dual_face_work"].values())


def test_nonclosure_remains_in_complete_roster_and_all8192_work_is_retained(records):
    state = records[1][0]["states"][2]
    result, probe = state["routed"], state["routed"]["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    result.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    state["previous_early_routed"] = deepcopy({key: value for key, value in result.items() if key not in summary.FACE_FIELDS})
    state["previous_zero_routed"] = {key: state["previous_early_routed"][key] for key in LEGACY_FIELDS}
    actual = summary.summarize(*records)
    assert actual["decision"] == "DUAL_FACE_ROOT_CANONICAL_DEVELOPMENT_HOLD"
    assert actual["execution_checks_passed"] and actual["work_gate_passed"]
    assert actual["state_count"] == 12 and len(actual["nonclosures"]) == 1
    assert actual["executed"]["whole_selector"]["shared_integer_states"] >= 8192
