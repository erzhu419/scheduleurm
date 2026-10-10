"""Late proof shares its remainder and never reruns frozen LP/native work."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from algorithm.experiments import late_score_proof_development_summary as summary_module
from algorithm.experiments.late_score_proof_development_summary import summarize
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import StrictScoreProofWork
from skill.tests.test_dual_reuse_development_summary import records as dual_records, new_trace
from skill.tests.test_root_pruning_confirmation_summary import selection, stage, routed


def late(original, proof_states):
    result = deepcopy(original)
    snapshot = original["final_lp"]
    baseline = deepcopy(snapshot["selection"])
    certified = {**deepcopy(baseline), "score_exact": True, "upper_bound": baseline["score"], "gap": 0., "epsilon": 0.}
    selected = {**deepcopy(certified), "exact": True, "canonical_exact": True, "frontier_regions_remaining": 0,
                "inherited_unsolved_regions": 0, "canonical_tie_regions_remaining": 0}
    score = (int(baseline["score"]), 1)
    probe = {"selection": selected, "baseline_selection": baseline, "score_certified_selection": certified,
        "preprocessing": deepcopy(snapshot["preprocessing"]), "capacity_propagation": deepcopy(snapshot["capacity_propagation"]),
        "integer_state_budget": 8192,
        "score_proof": asdict(StrictScoreProofWork("COMPLETE", True, True, 8192, score_fraction=score, states_visited=proof_states)),
        "completion": asdict(ScoreFaceCompletionWork("COMPLETE", True, True, 8192 - proof_states, states_visited=5, score_fraction=score))}
    result.update(probe=probe, probe_stage="late", status="LATE_PROBE_COMPLETE", selection=deepcopy(selected))
    return result


@pytest.fixture
def records():
    protocol, rows = deepcopy(dual_records.__wrapped__())
    for index, state in enumerate(rows[0]["states"]):
        state.update(borrowed_from="development", borrowed_index=index, source_record_source=state["source"])
        state["traces"] = {"old": state.pop("old_traces"), "new": state["traces"]}
        state["original_routed"] = {"old": deepcopy(state["old_routed"]), "new": deepcopy(state["routed"])}
        state.pop("expected_old_routed")
    failure = deepcopy(rows[0]["states"][0])
    failure.update(state_id="failure_development:q6", source="failure_development", cohort="failure_development", queue=[6],
        source_state_id="q6", source_record_source="fresh_confirmation", borrowed_from="confirmation", borrowed_index=9,
        historical_regression=False, reference=selection(6, 4096, 1))
    failure["traces"] = {"old": {str(cap): stage(6, cap, new=True, lp=2) for cap in failure["caps"]}, "new": new_trace(failure)}
    for trace in failure["traces"].values():
        for snapshot in trace.values():
            snapshot["selection"].update(exact=False, score_exact=False, canonical_exact=False,
                upper_bound=13., gap=1., epsilon=1 / 6, frontier_regions_remaining=1)
    originals = {}
    for label in ("old", "new"):
        original = routed(failure["traces"][label]["144"])
        original.update(final_lp=deepcopy(failure["traces"][label]["256"]),
            selection=deepcopy(failure["traces"][label]["256"]["selection"]), resumed=True, status="LP_RESUME_INCOMPLETE")
        originals[label] = original
    failure.update(original_routed=originals, old_routed=late(originals["old"], 30), routed=late(originals["new"], 10))
    rows[0]["states"].append(failure)
    protocol["cells"][0]["states"] = [{key: deepcopy(value) for key, value in state.items() if key in (
        "state_id", "queue", "source", "cohort", "panel", "seed", "source_index", "source_state_id", "prior_index",
        "historical_regression", "caps", "borrowed_from", "borrowed_index", "source_record_source")} for state in rows[0]["states"]]
    protocol.update(expected_state_count=4, expected_nonzero_state_count=3, expected_failure_development_count=1,
        expected_selector_calls=8, expected_interface_calls=8, expected_native_constructions=8,
        expected_native_advance_calls=0, expected_borrowed_native_snapshots=32,
        failure_development_keys=[{"node": "toy", "lanes": 2, "queue": [6]}],
        borrowed_sources={"development": {"path": "frozen_dev", "registered_at_utc": "previous"},
                          "confirmation": {"path": "frozen_hold", "registered_at_utc": "previous"}})
    rows[0]["new_old_routed_calls"] = 4
    return protocol, rows


def test_late_closure_uses_only_two_actual_routes_and_final_paid_work(records):
    result = summarize(*records)
    assert result["decision"] == "LATE_SCORE_PROOF_DEVELOPMENT_ADVANCE" and result["prior_hold_retained"]
    assert result["routed_quality_gate_passed"] and not result["failures"]
    assert result["state_count"] == result["borrowed_reference_count"] == 4
    assert result["executed_selector_calls"] == result["executed_interface_calls"] == result["native_search_constructions"] == 8
    assert result["executed_native_advance_calls"] == result["new_reference_calls"] == result["new_reference_oracle_calls"] == 0
    assert result["borrowed_native_snapshots"] == 32 and result["failure_development_count"] == 1
    work = result["all_executed_routed_work"]
    assert work["routed_old"]["search_work"]["lp_calls"] == 8 and work["routed_new"]["search_work"]["lp_calls"] == 4
    assert result["executed_search_work"]["lp_calls"] == 12
    assert work["routed_old"]["shared_integer_states"] == 35 and work["routed_new"]["shared_integer_states"] == 15
    assert work["routed_new"]["score_proof"]["work"]["states_visited"] == 10
    assert work["routed_new"]["completion"]["work"]["states_visited"] == 5
    assert work["probe_stage_counts"] == {"old": {"none": 3, "late": 1}, "new": {"none": 3, "late": 1}}
    assert result["executed_dual_reuse_work"]["certificates_saved"] == 4
    assert result["counterexample_states"][0]["quality"] == {"old": True, "new": True}


@pytest.mark.parametrize("mutation", ["shared_remainder", "extra_lp", "wrong_baseline", "missing_late", "ordinary_changed", "source_pointer"])
def test_late_proof_must_follow_the_frozen_route_and_shared_budget(records, mutation):
    protocol, rows = records
    failure = rows[0]["states"][-1]
    if mutation == "shared_remainder":
        failure["routed"]["probe"]["completion"]["state_budget"] = 8192
    elif mutation == "extra_lp":
        failure["routed"]["final_lp"]["selection"]["lp_calls"] += 1
    elif mutation == "wrong_baseline":
        failure["routed"]["probe"]["baseline_selection"] = deepcopy(failure["original_routed"]["new"]["initial_lp"]["selection"])
    elif mutation == "missing_late":
        failure["routed"] = deepcopy(failure["original_routed"]["new"])
    elif mutation == "ordinary_changed":
        rows[0]["states"][0]["routed"]["status"] = "LATE_PROBE_COMPLETE"
    else:
        failure["borrowed_index"] += 1
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"] and not result["development_gate_passed"]


def test_honest_strict_proof_exhaustion_remains_a_quality_hold(records):
    protocol, rows = records
    new = rows[0]["states"][-1]["routed"]
    probe = new["probe"]
    baseline = deepcopy(probe["baseline_selection"])
    probe.update(selection=deepcopy(baseline), score_certified_selection=deepcopy(baseline),
        score_proof=asdict(StrictScoreProofWork("STATE_BUDGET_EXHAUSTED", True, False, 8192,
            score_fraction=(12, 1), states_visited=8192)),
        completion=asdict(ScoreFaceCompletionWork("SKIPPED_SCORE_UNPROVEN", False, False, 0)))
    new.update(selection=deepcopy(baseline), status="LATE_PROBE_INCOMPLETE")
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and not result["routed_quality_gate_passed"]
    assert result["decision"] == "LATE_SCORE_PROOF_DEVELOPMENT_HOLD"
    assert result["counterexample_states"][0]["status"]["new"] == "LATE_PROBE_INCOMPLETE"
    assert result["routed_nonclosures"][0]["reference_complete"]
    assert result["all_executed_routed_work"]["routed_new"]["shared_integer_states"] == 8192


def test_historical_cost_is_independent_of_nonzero_pooled_gain(records, monkeypatch):
    original = summary_module._work

    def mock_historical_integer_work(states):
        work = original(states)
        work["routed_new"]["shared_integer_states"] += 10 * sum(state["historical_regression"] for state in states)
        return work

    monkeypatch.setattr(summary_module, "_work", mock_historical_integer_work)
    result = summarize(*records)
    assert result["execution_checks_passed"] and result["nonzero_work_gate_passed"]
    assert not result["historical_work_gate_passed"] and result["decision"] == "LATE_SCORE_PROOF_DEVELOPMENT_HOLD"
    assert result["work_increases"][0]["historical_regression"]


def test_borrowed_reference_nonclosure_has_its_own_decision(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and not result["effective_references_complete"]
    assert result["decision"] == "INCOMPLETE_REFERENCE" and not result["failures"]
