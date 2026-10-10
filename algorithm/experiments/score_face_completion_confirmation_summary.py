"""Summarize frozen two-stage confirmation, keeping old regressions separate."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    CAPACITY_COUNTERS, COUNTERS, _combine_metrics,
)
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.score_face_completion_development_summary import (
    ACTION_FIELDS, COMPLETION_COUNTERS, DISCHARGED_FIELDS, _selection_equal,
)


LABELS = ("baseline", "completed")
ORACLE_COUNTERS = COMPLETION_COUNTERS[:-2]


def _pointer(row, state):
    return {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "panel": state["panel"], "cohort": state["cohort"], "queue": state["queue"]}


def _score(selection, queue):
    return Fraction(sum(q * s for q, s in zip(queue, selection["service"], strict=True))) / Fraction.from_float(selection["duration"])


def _key(selection, queue, option_count):
    return (-_score(selection, queue), Fraction.from_float(selection["duration"]),
            tuple(option_count if index is None else index for index in selection["option_indices"]))


def _completion_errors(state, protocol):
    record, baseline = state["completed"], state["baseline"]["selection"]
    work, selected = record["completion"], record["selection"]
    if work["state_budget"] != protocol["canonical_state_budget"]:
        raise ValueError("completion state budget differs from the registered cap")
    errors = []
    if state["baseline_source"] != "embedded_stage_one":
        errors.append("baseline must be identified as embedded stage one")
    if not _selection_equal(baseline, record["baseline_selection"]):
        errors.append("baseline must be the stored embedded stage-one result")
    if state["baseline"]["capacity_propagation"] != record["capacity_propagation"]:
        errors.append("completion changed embedded capacity-propagation accounting")
    if state["baseline"]["metrics"]["preprocessing"] != record["metrics"]["preprocessing"]:
        errors.append("completion changed embedded preprocessing accounting")
    attempted = bool(baseline["score_exact"] and not baseline["canonical_exact"])
    if work["attempted"] != attempted:
        errors.append("completion trigger disagrees with the stage-one certificates")
    if any(type(work[key]) is not int or work[key] < 0 for key in COMPLETION_COUNTERS):
        errors.append("completion work counters must be nonnegative integers")
    if work["states_visited"] > work["state_budget"]:
        errors.append("completion exceeded its integer-state budget")
    if any(selected[key] != baseline[key] for key in COUNTERS):
        errors.append("completion changed the recorded LP-stage work")
    changed = any(tuple(selected[key]) != tuple(baseline[key]) if key in ACTION_FIELDS[:3]
                  else selected[key] != baseline[key] for key in ACTION_FIELDS)
    if work["action_changed"] != changed:
        errors.append("completion action-change flag disagrees with the stored action")
    if not attempted:
        expected = "SKIPPED_CANONICAL_COMPLETE" if baseline["canonical_exact"] else "SKIPPED_SCORE_UNPROVEN"
        if (work["status"] != expected or work["complete"] != baseline["canonical_exact"]
                or any(work[key] for key in COMPLETION_COUNTERS) or work["score_fraction"] is not None
                or not _selection_equal(selected, baseline)):
            errors.append("a skipped completion must preserve stage one and charge zero integer work")
    else:
        score = _score(baseline, state["queue"])
        if work["score_fraction"] is None or tuple(work["score_fraction"]) != (score.numerator, score.denominator):
            errors.append("completion target differs from its own certified incumbent score")
        if work["status"] == "COMPLETE":
            if (not work["complete"] or not selected["canonical_exact"] or not selected["score_exact"]
                    or not selected["exact"] or any(selected[key] for key in DISCHARGED_FIELDS)):
                errors.append("completed score-face search lacks a discharged canonical certificate")
        elif work["status"] == "STATE_BUDGET_EXHAUSTED":
            if work["complete"] or work["states_visited"] != work["state_budget"] or not _selection_equal(selected, baseline):
                errors.append("exhausted completion must retain stage one and its honest nonclosure")
        else:
            errors.append("attempted completion has an unsupported outcome")
    return errors


def _reference(state, protocol, option_count):
    reference, oracle = state["reference"], state["oracle"]
    trigger = bool(reference["score_exact"] and not reference["canonical_exact"])
    errors = []
    if (oracle is not None) != trigger:
        errors.append("reference oracle must follow only the full-reference certificate trigger")
    expected = _key(reference, state["queue"], option_count) if state["reference_complete"] else None
    source = "full_K4096" if expected is not None else None
    if oracle is not None:
        if oracle["state_visit_budget"] != protocol["oracle_state_budget"]:
            raise ValueError("reference oracle budget differs from the registered cap")
        work = oracle["work"]
        if (any(type(work[key]) is not int or work[key] < 0 for key in ORACLE_COUNTERS)
                or work["states_visited"] > protocol["oracle_state_budget"]):
            errors.append("reference oracle exceeded its separately recorded integer budget")
        score = _score(reference, state["queue"])
        if (not oracle["conditional_on_external_score_certificate"]
                or Fraction(*oracle["score_fraction"]) != score):
            errors.append("reference oracle target differs from the full-reference certified score")
        if oracle["complete"] and oracle["canonical_exact"]:
            if oracle["status"] != "COMPLETE":
                errors.append("completed reference oracle has inconsistent status")
            oracle_key = (-Fraction(*oracle["score_fraction"]), Fraction(*oracle["duration_fraction"]),
                          tuple(option_count if index is None else index for index in oracle["option_indices"]))
            if (Fraction.from_float(oracle["duration"]) != oracle_key[1]
                    or _score(oracle, state["queue"]) != -oracle_key[0]
                    or oracle["duration"] > reference["duration"]):
                errors.append("reference oracle action does not realize its external score and duration bound")
            if expected is not None and expected != oracle_key:
                errors.append("reference oracle contradicts the completed full reference")
            elif expected is None:
                expected, source = oracle_key, "independent_score_face_oracle"
        elif (oracle["complete"] or oracle["canonical_exact"]
              or oracle["status"] != "STATE_BUDGET_EXHAUSTED"
              or work["states_visited"] != protocol["oracle_state_budget"]):
            errors.append("reference oracle outcome is inconsistent or exposes an infeasible certified score")
    if state["effective_reference_complete"] != (expected is not None) or state["effective_reference_source"] != source:
        raise ValueError("effective reference must come from the full reference or its independent oracle")
    return expected, errors


def _observed(record):
    metrics = record["metrics"]
    return (metrics["reference_comparable_count"] == 1 and not metrics["score_mismatch_count"]
            and not metrics["canonical_action_mismatch_count"])


def _aggregate(entries):
    metrics = {label: _combine_metrics(state[label]["metrics"] for _, state in entries) for label in LABELS}
    work = [state["completed"]["completion"] for _, state in entries]
    oracles = [state["oracle"] for _, state in entries if state["oracle"] is not None]
    rates = {label: {
        "score_exact_count": metrics[label]["score_exact_count"],
        "canonical_exact_count": metrics[label]["canonical_exact_count"],
        "observed_canonical_action_equal_count": sum(_observed(state[label]) for _, state in entries),
        "certified_canonical_action_equal_count": sum(_observed(state[label]) and state[label]["selection"]["canonical_exact"] for _, state in entries),
    } for label in LABELS}
    return {
        "queue_count": len(entries), "metrics": metrics, "label_rates": rates,
        "action_gate_passed": rates["completed"]["certified_canonical_action_equal_count"] == len(entries),
        "original_incomplete_reference_count": sum(not state["reference_complete"] for _, state in entries),
        "effective_incomplete_reference_count": sum(not state["effective_reference_complete"] for _, state in entries),
        "completion": {
            "attempted_count": sum(item["attempted"] for item in work),
            "completed_count": sum(item["status"] == "COMPLETE" for item in work),
            "exhausted_count": sum(item["status"] == "STATE_BUDGET_EXHAUSTED" for item in work),
            "skipped_canonical_complete_count": sum(item["status"] == "SKIPPED_CANONICAL_COMPLETE" for item in work),
            "skipped_score_unproven_count": sum(item["status"] == "SKIPPED_SCORE_UNPROVEN" for item in work),
            "action_changed_count": sum(item["action_changed"] for item in work),
            "work": {key: sum(item[key] for item in work) for key in COMPLETION_COUNTERS},
            "max_states_visited": max((item["states_visited"] for item in work), default=0),
        },
        "reference_oracle": {
            "calls": len(oracles), "completed_count": sum(item["complete"] and item["canonical_exact"] for item in oracles),
            "work": {key: sum(item["work"][key] for item in oracles) for key in ORACLE_COUNTERS},
            "max_states_visited": max((item["work"]["states_visited"] for item in oracles), default=0),
        },
        "capacity_propagation": {label: {key: sum(state[label]["capacity_propagation"][key] for _, state in entries)
                                         for key in CAPACITY_COUNTERS} for label in LABELS},
        "first_unresolved": next((_pointer(row, state) for row, state in entries if not state["completed"]["selection"]["canonical_exact"]), None),
        "first_mismatch": next((_pointer(row, state) for row, state in entries if state["completed"]["metrics"]["score_mismatch_count"] or state["completed"]["metrics"]["canonical_action_mismatch_count"]), None),
    }


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
        expected = [(state["state_id"], panel["panel"], tuple(state["queue"]), panel["cohort"]) for panel in panels for state in panel["states"]]
        actual = [(state["state_id"], state["panel"], tuple(state["queue"]), state["cohort"]) for state in row["states"]]
        if actual != expected or not actual or len({x[0] for x in actual}) != len(actual) or len({x[2] for x in actual}) != len(actual):
            raise ValueError("stored states differ from the deduplicated registered roster")
        cell_entries = []
        for state in row["states"]:
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"]:
                raise ValueError("full-reference LP budget differs from the registered cap")
            if state["reference_metrics"]["reference_comparable_count"] != int(state["reference_complete"]):
                raise ValueError("full-reference metrics must retain original reference nonclosure")
            expected_key, errors = _reference(state, protocol, row["option_count"])
            for label in LABELS:
                metrics, selected = state[label]["metrics"], state[label]["selection"]
                _check_single_call(metrics, selected)
                if selected["lp_node_budget"] != protocol["budget"]:
                    raise ValueError("stored LP budget differs from the registered cap")
                if metrics["reference_comparable_count"] != int(expected_key is not None):
                    raise ValueError("reference comparability differs from the available reference")
                if (selected["lp_calls"] > protocol["budget"]
                        or any(metrics["search_work"][key] != selected[key] for key in COUNTERS)):
                    errors.append(f"{label} LP work exceeds its cap or differs from stored selection")
                if expected_key is not None:
                    actual_key = _key(selected, state["queue"], row["option_count"])
                    if (metrics["score_mismatch_count"] != int(actual_key[0] != expected_key[0])
                            or metrics["canonical_action_mismatch_count"] != int(actual_key[1:] != expected_key[1:])):
                        errors.append(f"{label} comparison metrics disagree with the recorded exact actions")
            errors.extend(_completion_errors(state, protocol))
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entry = (row, state)
            entries.append(entry)
            cell_entries.append(entry)
            cohorts[state["cohort"]].append(entry)
            lane_groups[row["lanes"]].append(entry)
        cells.append({"node": row["node"], "lanes": row["lanes"], **_aggregate(cell_entries)})
    total = _aggregate(entries)
    valid = not failures and all(not state["reference_metrics"]["failures"]
                and all(not state[label]["metrics"]["failures"] for label in LABELS) for _, state in entries)
    complete = not total["effective_incomplete_reference_count"]
    fresh = _aggregate(cohorts["fresh_confirmation"])
    regression = _aggregate([entry for cohort, group in cohorts.items() if cohort != "fresh_confirmation" for entry in group])
    fresh_gate = bool(valid and fresh["queue_count"] and not fresh["effective_incomplete_reference_count"] and fresh["action_gate_passed"])
    regression_gate = bool(valid and not regression["effective_incomplete_reference_count"] and regression["action_gate_passed"])
    action = fresh_gate and regression_gate
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "SCORE_FACE_COMPLETION_CONFIRMATION_ADVANCE" if action else "SCORE_FACE_COMPLETION_CONFIRMATION_HOLD")
    references = _combine_metrics(state["reference_metrics"] for _, state in entries)
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "budget": protocol["budget"], "canonical_state_budget": protocol["canonical_state_budget"],
        "cell_count": len(rows), "executed_selector_calls": 2 * len(entries),
        "executed_reference_calls": len(entries), "executed_candidate_calls": len(entries),
        "embedded_baseline_count": len(entries), "executed_oracle_calls": total["reference_oracle"]["calls"],
        "certificate_checks_passed": bool(valid), "completion_accounting_failures": failures,
        "original_references_complete": not total["original_incomplete_reference_count"],
        "effective_references_complete": complete,
        "fresh_confirmation_gate_passed": fresh_gate, "regression_gate_passed": regression_gate,
        "confirmation_action_gate_passed": action, **total,
        "reference_metrics": references,
        "executed_search_work": {key: references["search_work"][key] + total["metrics"]["completed"]["search_work"][key] for key in COUNTERS},
        "fresh_confirmation": fresh, "regression": regression, "cells": cells,
        "cohorts": {name: _aggregate(group) for name, group in cohorts.items()},
        "by_lanes": [{"lanes": lanes, **_aggregate(group)} for lanes, group in lane_groups.items()],
        "limitations": protocol["limitations"],
    }
