"""Audit fresh shared-budget confirmation separately from fixed regressions."""
from __future__ import annotations

from collections import defaultdict

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.score_face_completion_confirmation_summary import (
    LABELS, _aggregate as _completion_aggregate, _key, _pointer, _reference,
)
from algorithm.experiments.strict_score_proof_development_summary import PROOF_COUNTERS, _proof_errors


COHORTS = {"fresh_confirmation", "old_regression", "score_proof_regression", "canonical_regression"}


def _aggregate(entries):
    result = _completion_aggregate(entries)
    proofs = [state["completed"]["score_proof"] for _, state in entries]
    shared = [proof["states_visited"] + state["completed"]["completion"]["states_visited"]
              for proof, (_, state) in zip(proofs, entries)]
    result.update(
        score_proof={
            "attempted_count": sum(proof["attempted"] for proof in proofs),
            "completed_count": sum(proof["status"] == "COMPLETE" for proof in proofs),
            "better_action_found_count": sum(proof["status"] == "BETTER_ACTION_FOUND" for proof in proofs),
            "exhausted_count": sum(proof["status"] == "STATE_BUDGET_EXHAUSTED" for proof in proofs),
            "work": {key: sum(proof[key] for proof in proofs) for key in PROOF_COUNTERS},
            "max_states_visited": max((proof["states_visited"] for proof in proofs), default=0),
        },
        shared_integer_states=sum(shared), max_shared_integer_states=max(shared, default=0),
        trigger_points=[{
            **_pointer(row, state),
            "score_proof_status": state["completed"]["score_proof"]["status"],
            "proof_states": state["completed"]["score_proof"]["states_visited"],
            "completion_status": state["completed"]["completion"]["status"],
            "completion_states": state["completed"]["completion"]["states_visited"],
            "final_score_exact": state["completed"]["selection"]["score_exact"],
            "final_canonical_exact": state["completed"]["selection"]["canonical_exact"],
        } for row, state in entries
          if state["completed"]["score_proof"]["attempted"] or state["completed"]["completion"]["attempted"]],
    )
    return result


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    expected_cells = {(node, lanes) for node in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        raise ValueError("all registered cells must occur exactly once")
    entries, cells, cohorts, lane_groups, failures = [], [], defaultdict(list), defaultdict(list), []
    for row in rows:
        model = models[row["node"]]
        if (row["registered_at_utc"] != protocol["registered_at_utc"]
                or row["physical_lanes"] != model["lanes"] or row["option_count"] != len(model["options"])):
            raise ValueError("cell registration or catalogue metadata differs from the protocol")
        panels = protocol["queue_panels"][str(row["lanes"])]
        if any(panel["cohort"] not in COHORTS for panel in panels):
            raise ValueError("unknown registered confirmation cohort")
        expected = [(state["state_id"], panel["panel"], tuple(state["queue"]), panel["cohort"])
                    for panel in panels for state in panel["states"]]
        actual = [(state["state_id"], state["panel"], tuple(state["queue"]), state["cohort"])
                  for state in row["states"]]
        if (actual != expected or not actual or len({item[0] for item in actual}) != len(actual)
                or len({item[2] for item in actual}) != len(actual)):
            raise ValueError("stored states differ from the deduplicated registered roster")
        cell_entries = []
        for state in row["states"]:
            _check_single_call(state["reference_metrics"], state["reference"],
                               reference_complete=state["reference_complete"])
            if (state["reference"]["lp_node_budget"] != protocol["reference_budget"]
                    or state["reference_metrics"]["reference_comparable_count"] != int(state["reference_complete"])):
                raise ValueError("reference budget or original reference comparability differs from protocol")
            expected_key, errors = _reference(state, protocol, row["option_count"])
            records = [(label, state[label]["metrics"], state[label]["selection"]) for label in LABELS]
            records.append(("score_certified", state["score_certified_metrics"],
                            state["completed"]["score_certified_selection"]))
            for label, metrics, selected in records:
                _check_single_call(metrics, selected)
                if (selected["lp_node_budget"] != protocol["budget"]
                        or metrics["reference_comparable_count"] != int(expected_key is not None)):
                    raise ValueError("candidate budget or reference comparability differs from protocol")
                if (selected["lp_calls"] > protocol["budget"] or metrics["failures"]
                        or any(metrics["search_work"][key] != selected[key] for key in COUNTERS)):
                    errors.append(f"{label} has invalid selection metrics or LP accounting")
                if expected_key is not None:
                    key = _key(selected, state["queue"], row["option_count"])
                    if (metrics["score_mismatch_count"] != int(key[0] != expected_key[0])
                            or metrics["canonical_action_mismatch_count"] != int(key[1:] != expected_key[1:])):
                        errors.append(f"{label} reference comparison disagrees with its exact action")
            if state["baseline_source"] != "embedded_stage_one":
                errors.append("baseline must be identified as embedded stage one")
            errors.extend(_proof_errors(state, protocol, model, row["lanes"]))
            reference = state["reference"]
            if (state["reference_metrics"]["failures"] or reference["lp_calls"] > protocol["reference_budget"]
                    or any(state["reference_metrics"]["search_work"][key] != reference[key] for key in COUNTERS)):
                errors.append("full reference has invalid selection metrics or LP accounting")
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entry = (row, state)
            entries.append(entry)
            cell_entries.append(entry)
            cohorts[state["cohort"]].append(entry)
            lane_groups[row["lanes"]].append(entry)
        cells.append({"node": row["node"], "lanes": row["lanes"], **_aggregate(cell_entries)})
    total = _aggregate(entries)
    fresh = _aggregate(cohorts["fresh_confirmation"])
    regression = _aggregate([entry for name, group in cohorts.items() if name != "fresh_confirmation" for entry in group])
    valid, complete = not failures, not total["effective_incomplete_reference_count"]
    fresh_gate = bool(valid and fresh["queue_count"] and not fresh["effective_incomplete_reference_count"]
                      and fresh["action_gate_passed"])
    regression_gate = bool(valid and not regression["effective_incomplete_reference_count"]
                           and regression["action_gate_passed"])
    action = fresh_gate and regression_gate
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "STRICT_SCORE_PROOF_CONFIRMATION_ADVANCE" if action else "STRICT_SCORE_PROOF_CONFIRMATION_HOLD")
    references = _combine_metrics(state["reference_metrics"] for _, state in entries)
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "budget": protocol["budget"], "integer_state_budget": protocol["integer_state_budget"],
        "cell_count": len(rows), "executed_selector_calls": 2 * len(entries),
        "executed_reference_calls": len(entries), "executed_candidate_calls": len(entries),
        "embedded_baseline_count": len(entries), "intermediate_certificate_count": len(entries),
        "executed_oracle_calls": total["reference_oracle"]["calls"],
        "certificate_checks_passed": valid, "proof_accounting_failures": failures,
        "original_references_complete": not total["original_incomplete_reference_count"],
        "effective_references_complete": complete,
        "fresh_confirmation_gate_passed": fresh_gate, "regression_gate_passed": regression_gate,
        "confirmation_action_gate_passed": action, **total,
        "reference_metrics": references,
        "executed_search_work": {key: references["search_work"][key] + total["metrics"]["completed"]["search_work"][key]
                                 for key in COUNTERS},
        "fresh_confirmation": fresh, "regression": regression, "cells": cells,
        "cohorts": {name: _aggregate(group) for name, group in cohorts.items()},
        "by_lanes": [{"lanes": lanes, **_aggregate(group)} for lanes, group in lane_groups.items()],
        "limitations": protocol["limitations"],
    }
