"""Short-prefix routing, direct-predecessor costs and honest borrowed references."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from algorithm.experiments.short_lookahead_development_summary import audit_short_lookahead_router, summarize
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import StrictScoreProofWork
from skill.tests.test_root_pruning_confirmation_summary import records as root_records, stage, routed


def probe(initial, queue, complete=False, states=8192):
    baseline = deepcopy(initial["selection"])
    selected = deepcopy(baseline)
    if complete:
        selected.update(exact=True, canonical_exact=True, frontier_regions_remaining=0,
                        inherited_unsolved_regions=0, canonical_tie_regions_remaining=0)
    return {"selection": selected, "baseline_selection": baseline, "score_certified_selection": deepcopy(baseline),
        "preprocessing": deepcopy(initial["preprocessing"]), "capacity_propagation": deepcopy(initial["capacity_propagation"]),
        "integer_state_budget": 8192,
        "score_proof": asdict(StrictScoreProofWork("SKIPPED_SCORE_CERTIFIED", False, True, 8192)),
        "completion": asdict(ScoreFaceCompletionWork("COMPLETE" if complete else "STATE_BUDGET_EXHAUSTED", True, complete, 8192,
            states_visited=states, score_fraction=(queue * 2, 1)))}


@pytest.fixture
def records(root_records):
    protocol, rows = deepcopy(root_records)
    for index, state in enumerate(rows[0]["states"]):
        state.update(source=("development", "confirmation", "new_zero")[index], prior_index=index,
            historical_regression=index == 1, source_state_id=state["state_id"])
        queue = state["queue"][0]
        if queue:
            initial, prefix, full = stage(queue, 144, new=True, lp=144), stage(queue, 145, new=True, lp=145), stage(queue, 256, new=True, lp=145)
            initial["selection"].update(exact=False, canonical_exact=False, frontier_regions_remaining=1,
                                        inherited_unsolved_regions=1, canonical_tie_regions_remaining=1)
            old = routed(initial)
            old.update(selection=deepcopy(full["selection"]), final_lp=deepcopy(full), probe=probe(initial, queue),
                       resumed=True, probe_stage="initial", status="LP_RESUME_COMPLETE")
            new = routed(initial)
            new.update(selection=deepcopy(prefix["selection"]), final_lp=deepcopy(prefix), resumed=True,
                       lookahead=[deepcopy(prefix)], status="LP_LOOKAHEAD_COMPLETE")
        else:
            initial, full = stage(queue, 144, new=True, lp=1), stage(queue, 256, new=True, lp=1)
            old = routed(initial)
            new = {**deepcopy(old), "lookahead": []}
        state.update(expected_initial_lp=initial, expected_native_final=full, expected_old_routed=deepcopy(old), old_routed=old, routed=new)
        state.pop("traces")
        state.pop("caps")
    protocol["cells"][0]["states"] = [{key: deepcopy(value) for key, value in state.items() if key in (
        "state_id", "source", "cohort", "queue", "source_index", "prior_index", "source_state_id", "historical_regression")}
        for state in rows[0]["states"]]
    protocol.update(expected_selector_calls=6, historical_regression_keys=[{"node": "toy", "lanes": 2, "queue": [5]}])
    rows[0].update(new_reference_calls=0, new_reference_oracle_calls=0)
    return protocol, rows


def test_prefix_counts_actual_advances_and_charges_only_final_cumulative_lp(records):
    result = summarize(*records)
    assert result["decision"] == "SHORT_LOOKAHEAD_DEVELOPMENT_ADVANCE" and result["prior_hold_retained"]
    assert result["executed_selector_calls"] == result["native_search_constructions"] == 6
    assert result["lp_advance_calls"] == {"old": 5, "new": 5} and result["integer_probe_calls"] == {"old": 2, "new": 0}
    assert result["executed_interface_calls"] == 6
    assert result["all_executed_routed_work"]["old"]["search_work"]["lp_calls"] == 291
    assert result["all_executed_routed_work"]["new"]["search_work"]["lp_calls"] == 291
    assert result["all_executed_routed_work"]["old"]["shared_integer_states"] == 16384
    assert result["all_executed_routed_work"]["new"]["shared_integer_states"] == 0
    assert result["new_reference_calls"] == result["new_reference_oracle_calls"] == 0
    assert result["lookahead"] == {"attempted": 2, "canonical_successes": 2, "steps": 2}


@pytest.mark.parametrize("mutation", ["skip_cap", "continue_closed", "mixed_initial", "wrong_native_final"])
def test_new_prefix_rule_and_unchanged_kernel_snapshots_are_enforced(records, mutation):
    protocol, rows = records
    state, new = rows[0]["states"][0], rows[0]["states"][0]["routed"]
    if mutation == "skip_cap":
        new["lookahead"][0]["selection"]["lp_node_budget"] = 146
    elif mutation == "continue_closed":
        extra = deepcopy(new["lookahead"][0])
        extra["selection"]["lp_node_budget"] = 146
        new["lookahead"].append(extra)
    elif mutation == "mixed_initial":
        new["initial_lp"]["selection"]["inherited_unsolved_regions"] = 0
    elif mutation == "wrong_native_final":
        new.update(lookahead=[], final_lp=deepcopy(state["expected_native_final"]),
                   selection=deepcopy(state["expected_native_final"]["selection"]), status="LP_RESUME_COMPLETE", probe_stage="initial",
                   probe=deepcopy(state["old_routed"]["probe"]))
        new["final_lp"]["selection"]["lp_calls"] = 144
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]
    assert not result["development_gate_passed"]


def test_probe_must_reuse_latest_prefix_and_stop_after_canonical_completion(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    new = state["routed"]
    current = new["lookahead"][0]
    current["selection"].update(exact=False, canonical_exact=False, frontier_regions_remaining=1,
                               inherited_unsolved_regions=0, canonical_tie_regions_remaining=1)
    paid_probe = probe(current, 4, complete=True, states=100)
    new.update(final_lp=deepcopy(current), selection=deepcopy(paid_probe["selection"]), probe=paid_probe,
               probe_stage="lookahead", status="INTEGER_PROBE_COMPLETE")
    expected, _ = _reference(state, protocol, 1)
    metric, errors, quality = audit_short_lookahead_router(new, state, protocol, protocol["models"][0], 2, expected)
    assert not errors and quality and not metric["failures"]
    new["probe"]["baseline_selection"] = deepcopy(state["expected_initial_lp"]["selection"])
    assert audit_short_lookahead_router(new, state, protocol, protocol["models"][0], 2, expected)[1]


def test_integer_saving_cannot_pay_for_extra_lp_against_direct_predecessor(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    initial = state["old_routed"]["initial_lp"]
    paid_probe = probe(initial, 4, complete=True, states=200)
    state["old_routed"] = routed(initial)
    state["old_routed"].update(selection=deepcopy(paid_probe["selection"]), probe=paid_probe,
                               probe_stage="initial", status="INTEGER_PROBE_COMPLETE")
    state["expected_old_routed"] = deepcopy(state["old_routed"])
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["strict_integer_reduction_gate_passed"]
    assert not result["work_nonincrease_gate_passed"] and result["decision"] == "SHORT_LOOKAHEAD_DEVELOPMENT_HOLD"
    assert result["work_increases"][0]["increases"]["lp_calls"] == {"old": 144, "new": 145}


def test_historical_cost_increase_cannot_hide_in_the_nonzero_pool(records):
    protocol, rows = records
    # Pure cost-gate mocks keep this test focused on the historical subset.
    import algorithm.experiments.short_lookahead_development_summary as module
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "audit_candidate", lambda *args: ({}, [], True))
        patch.setattr(module, "audit_short_lookahead_router", lambda *args: ({}, [], True))
        state = rows[0]["states"][1]
        state["routed"]["final_lp"]["selection"]["lp_matrix_coefficients"] += 1
        rows[0]["states"][0]["old_routed"]["final_lp"]["selection"]["lp_matrix_coefficients"] += 1
        rows[0]["states"][0]["expected_old_routed"] = deepcopy(rows[0]["states"][0]["old_routed"])
        rows[0]["states"][0]["expected_native_final"] = deepcopy(rows[0]["states"][0]["old_routed"]["final_lp"])
        result = summarize(protocol, rows)
    assert result["cells"][0]["nonzero_cost_nonincrease"] and not result["cells"][0]["historical_cost_nonincrease"]
    assert result["decision"] == "SHORT_LOOKAHEAD_DEVELOPMENT_HOLD" and result["strict_integer_reduction_gate_passed"]


def test_borrowed_reference_nonclosure_is_retained_as_incomplete(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"]
    assert not result["development_gate_passed"] and result["borrowed_reference_count"] == 3
