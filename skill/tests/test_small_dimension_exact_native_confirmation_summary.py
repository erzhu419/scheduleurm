"""New confirmation coverage, honest nonclosure and independently paid proofs."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from types import SimpleNamespace

import pytest

from algorithm.experiments import small_dimension_exact_native_confirmation_summary as summary
from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
from skill.tests.test_affine_dual_fresh_confirmation_summary import _open_reference
from skill.tests.test_small_dimension_exact_native_development_summary import records as development_records


@pytest.fixture
def records():
    protocol, rows = deepcopy(development_records.__wrapped__())
    cell, row = protocol["cells"][0], rows[0]
    models = {model["node"]: model for model in protocol["models"]}
    fresh_states, metadata = [], []
    options = tuple(SimpleNamespace(**option) for option in models[row["node"]]["options"])
    for index, source_index in enumerate((0, 2)):
        state, registered = deepcopy(row["states"][source_index]), deepcopy(cell["states"][source_index])
        for key in tuple(registered):
            if "source_index" in key or "source_state_id" in key or key.startswith("timing_source"):
                registered.pop(key)
                state.pop(key, None)
        registered.update(state_id=f"new:{index}", source="small_dimension_exact_native_confirmation",
            cohort=summary.FRESH_GROUP, panel="independent_2026092911", confirmation_source="fresh",
            timing_group=summary.FRESH_GROUP, historical_regression=False, seed=2026092911,
            small_dimension_confirmation_source="fresh")
        state.update(registered)
        state.pop("prior_routed", None)
        state.pop("independent_source", None)
        state["long_routed"] = asdict(select_strict_score_proof_canonical_completion_frame(
            options, tuple(state["queue"]), row["lanes"], 4096, 100000))
        fresh_states.append(state)
        metadata.append(registered)
    for index, (state, registered) in enumerate(zip(row["states"], cell["states"], strict=True)):
        pointer = dict(small_dimension_confirmation_source="borrowed_control",
            small_dimension_confirmation_source_index=index, small_dimension_confirmation_source_state_id=state["state_id"])
        state.update(pointer)
        registered.update(pointer)
    # A previously independent sample is still an old borrowed control.
    row["states"][-1]["independent_source"] = cell["states"][-1]["independent_source"] = "fresh"
    cell["states"] = metadata + cell["states"]
    row["states"] = fresh_states + row["states"]
    fresh, old = len(fresh_states), len(row["states"]) - len(fresh_states)
    protocol.update(experiment="toy_small_dimension_confirmation", expected_state_count=fresh + old,
        expected_independent_state_count=fresh, expected_borrowed_state_count=old,
        expected_selector_calls=12 * fresh, expected_native_advance_calls=24 * fresh,
        expected_native_constructions=18 * fresh, expected_interface_calls=36 * fresh,
        expected_new_reference_calls=fresh, expected_candidate_native_snapshots=4 * fresh,
        long_lp_budget=4096, control_integer_state_budget=100000,
        expected_group_counts=dict(Counter(state["timing_group"] for state in row["states"])))
    row.update(new_reference_calls=fresh, new_oracle_calls=0, selector_calls=dict.fromkeys(summary.LABELS, fresh),
        native_advance_calls=24 * fresh, native_search_constructions=18 * fresh,
        root_settle_calls={label: sum(state[field]["root_settle_work"] is not None for state in fresh_states)
                           for label, field in summary.ROOT_ARMS.items()})
    return protocol, rows


def test_complete_new_and_borrowed_roster_charges_six_ladders_and_twelve_whole_calls(records):
    result = summary.summarize(*records)
    assert result["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_ADVANCE", result["failures"]
    assert result["state_count"] == 14 and result["independent_state_count"] == 2 and result["borrowed_state_count"] == 12
    assert result["new_reference_calls"] == 2 and result["borrowed_reference_count"] == 12
    assert result["executed_selector_calls"] == 24 and result["executed_native_advance_calls"] == 48
    assert result["executed_interface_calls"] == 72 and result["native_search_constructions"] == 36
    assert set(result["executed"]["whole_selector"]) == set(summary.LABELS)
    assert "long" not in result["borrowed"]["whole_selector"]
    assert len(result["executed"]["native"]) == len(result["borrowed"]["native"]) == 6
    for field in ("small_dimension_lp_work", "native_own_tight_work"):
        expected = sum(state["routed"]["final_lp"][field]["event_snapshot_entries_copied"] for state in records[1][0]["states"][:2])
        assert result["executed"]["whole_selector"]["candidate"][field]["event_snapshot_entries_copied"] == expected
    assert all(result[name] for name in ("execution_checks_passed", "references_complete", "quality_gate_passed",
        "work_gate_passed", "confirmation_gate_passed", "prior_hold_retained"))


def test_honest_method_nonclosure_keeps_full_roster_and_all_exhausted_work(records):
    routed = records[1][0]["states"][1]["old_routed"]
    probe = routed["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    routed.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    result = summary.summarize(*records)
    assert result["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_HOLD", result["failures"]
    assert result["execution_checks_passed"] and result["work_gate_passed"] and not result["quality_gate_passed"]
    assert result["state_count"] == 14 and len(result["nonclosures"]) == 1
    assert result["nonclosures"][0]["label"] == "branch"
    assert result["executed"]["whole_selector"]["branch"]["shared_integer_states"] >= 8192


def test_unproved_reference_retains_all_states_and_prevents_confirmation(records):
    _open_reference(records[1][0]["states"][0], score_exact=False)
    result = summary.summarize(*records)
    assert result["decision"] == "INCOMPLETE_REFERENCE", result["failures"]
    assert result["execution_checks_passed"] and not result["references_complete"]
    assert result["state_count"] == 14 and result["new_reference_calls"] == 2


@pytest.mark.parametrize("complete", (True, False))
def test_reference_own_score_certificate_triggers_independent_oracle_and_keeps_exhaustion(records, complete):
    state = records[1][0]["states"][0]
    _open_reference(state, score_exact=True)
    reference = state["reference"]
    score = Fraction(sum(waiting * done for waiting, done in zip(state["queue"], reference["service"], strict=True))) / Fraction.from_float(reference["duration"])
    duration = Fraction.from_float(reference["duration"])
    oracle = {**{name: deepcopy(reference[name]) for name in ("option_counts", "option_indices", "service", "duration", "score")},
        "conditional_on_external_score_certificate": True, "state_visit_budget": 5000000,
        "score_fraction": [score.numerator, score.denominator], "duration_fraction": [duration.numerator, duration.denominator],
        "complete": complete, "canonical_exact": complete, "status": "COMPLETE" if complete else "STATE_BUDGET_EXHAUSTED",
        "work": dict.fromkeys(summary.ORACLE_COUNTERS, 0)}
    oracle["work"]["states_visited"] = 2 if complete else 5000000
    state.update(oracle=oracle, effective_reference_complete=complete,
        effective_reference_source="independent_score_face_oracle" if complete else None)
    records[1][0]["new_oracle_calls"] = 1
    result = summary.summarize(*records)
    assert result["execution_checks_passed"], result["failures"]
    assert result["decision"] == ("SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_ADVANCE" if complete else "INCOMPLETE_REFERENCE")
    assert result["state_count"] == 14 and result["oracle_work"]["new"]["work"]["states_visited"] == oracle["work"]["states_visited"]
    state["reference"]["score_exact"] = False
    state["reference_metrics"]["score_exact_count"] = 0
    assert summary.summarize(*records)["decision"] == "INVALID_EXECUTION"


@pytest.mark.parametrize("change", ("copy_fee", "native_budget", "long_budget", "method_calls", "borrowed_long"))
def test_real_ledger_budget_and_call_mutations_do_not_advance(records, change):
    protocol, rows = records
    state = rows[0]["states"][0]
    if change == "copy_fee":
        state["routed"]["final_lp"]["small_dimension_lp_work"]["event_snapshot_entries_copied"] += 1
    elif change == "native_budget":
        state["previous_direct_probe_native_traces"]["256"]["selection"]["lp_node_budget"] = 144
    elif change == "long_budget":
        state["long_routed"]["selection"]["lp_node_budget"] = 256
    elif change == "method_calls":
        rows[0]["selector_calls"]["long"] -= 1
    else:
        rows[0]["states"][-1]["long_routed"] = deepcopy(state["long_routed"])
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["state_count"] == 14
    assert not result["work_gate_passed"] and result["prior_hold_retained"]


@pytest.mark.parametrize("field", ("independent_source", "small_dimension_confirmation_source_state_id"))
def test_borrowed_source_metadata_is_preserved_verbatim(records, field):
    records[1][0]["states"][-1][field] = "changed"
    with pytest.raises(ValueError, match="original metadata/source pointer"):
        summary.summarize(*records)
