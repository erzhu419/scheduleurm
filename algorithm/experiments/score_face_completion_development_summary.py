"""Audit a conditional integer completion stage with a separate work budget."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    CAPACITY_COUNTERS, COUNTERS, _combine_metrics,
)
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call


LABELS = ("legacy", "completed")
COMPLETION_COUNTERS = (
    "states_visited", "memo_hits", "count_branches", "bound_prunes", "gcd_prunes",
    "duration_prunes", "failed_states_cached", "thresholds_checked", "noninteger_thresholds",
    "suffix_coordinate_reads", "resource_coordinate_reads",
)
ACTION_FIELDS = ("option_counts", "option_indices", "service", "duration")
DISCHARGED_FIELDS = ("canonical_tie_regions_remaining", "frontier_regions_remaining", "inherited_unsolved_regions")


def _pointer(row, state):
    return {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "panel": state["panel"], "queue": state["queue"]}


def _selection_equal(left, right):
    return left.keys() == right.keys() and all(
        tuple(left[key]) == tuple(right[key]) if key in ACTION_FIELDS[:3]
        else left[key] == right[key] for key in left)


def _observed(record):
    metrics = record["metrics"]
    return (metrics["reference_comparable_count"] == 1
            and not metrics["score_mismatch_count"]
            and not metrics["canonical_action_mismatch_count"])


def _completion_errors(state, protocol):
    record, baseline = state["completed"], state["completed"]["baseline_selection"]
    work, selected = record["completion"], record["selection"]
    if work["state_budget"] != protocol["canonical_state_budget"]:
        raise ValueError("completion state budget differs from the registered cap")
    attempted = bool(baseline["score_exact"] and not baseline["canonical_exact"])
    errors = []
    if work["attempted"] != attempted:
        errors.append("completion trigger disagrees with the stage-one certificates")
    if any(type(work[key]) is not int or work[key] < 0 for key in COMPLETION_COUNTERS):
        errors.append("completion work counters must be nonnegative integers")
    if work["states_visited"] > work["state_budget"]:
        errors.append("completion exceeded its integer-state budget")
    if any(selected[key] != baseline[key] for key in COUNTERS):
        errors.append("completion changed the recorded LP-stage work")
    if not state["baseline_reproduced"] or not _selection_equal(baseline, state["legacy"]["selection"]):
        errors.append("stage one did not reproduce the frozen legacy result")
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
        score = Fraction(sum(q * s for q, s in zip(state["queue"], baseline["service"], strict=True))) / Fraction.from_float(baseline["duration"])
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


def _aggregate(entries):
    metrics = {label: _combine_metrics(state[label]["metrics"] for _, state in entries) for label in LABELS}
    rates = {label: {
        "score_exact_count": metrics[label]["score_exact_count"],
        "canonical_exact_count": metrics[label]["canonical_exact_count"],
        "observed_canonical_action_equal_count": sum(_observed(state[label]) for _, state in entries),
        "certified_canonical_action_equal_count": sum(_observed(state[label]) and state[label]["selection"]["canonical_exact"] for _, state in entries),
    } for label in LABELS}
    work = [state["completed"]["completion"] for _, state in entries]
    return {
        "queue_count": len(entries), "metrics": metrics, "label_rates": rates,
        "original_incomplete_reference_count": sum(not state["reference_complete"] for _, state in entries),
        "effective_incomplete_reference_count": sum(not state["effective_reference_complete"] for _, state in entries),
        "baseline_reproduced_count": sum(state["baseline_reproduced"] for _, state in entries),
        "completion": {
            "attempted_count": sum(item["attempted"] for item in work),
            "completed_count": sum(item["status"] == "COMPLETE" for item in work),
            "exhausted_count": sum(item["status"] == "STATE_BUDGET_EXHAUSTED" for item in work),
            "skipped_canonical_complete_count": sum(item["status"] == "SKIPPED_CANONICAL_COMPLETE" for item in work),
            "skipped_score_unproven_count": sum(item["status"] == "SKIPPED_SCORE_UNPROVEN" for item in work),
            "action_changed_count": sum(item["action_changed"] for item in work),
            "work": {key: sum(item[key] for item in work) for key in COMPLETION_COUNTERS},
            "max_states_visited": max(item["states_visited"] for item in work),
        },
        "capacity_propagation": {label: {key: sum(state[label]["capacity_propagation"].get(key, 0) for _, state in entries)
                                         for key in CAPACITY_COUNTERS} for label in LABELS},
        "first_unresolved": next((_pointer(row, state) for row, state in entries if not state["completed"]["selection"]["canonical_exact"]), None),
        "first_mismatch": next((_pointer(row, state) for row, state in entries if state["completed"]["metrics"]["score_mismatch_count"] or state["completed"]["metrics"]["canonical_action_mismatch_count"]), None),
    }


def summarize(models, protocol, rows, oracles):
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        raise ValueError("all registered cells must occur exactly once")
    signature = lambda item: (item["node"], item["lanes"], item["state_id"], tuple(item["queue"]))
    expected_oracles, actual_oracles = ([signature(item) for item in items]
                                      for items in (protocol["oracle_witnesses"], oracles))
    if actual_oracles != expected_oracles or len(set(actual_oracles)) != len(actual_oracles):
        raise ValueError("reused oracle coverage differs from the frozen witness roster")
    oracle_lookup = {(item["node"], item["lanes"], item["state_id"]): item["result"] for item in oracles}
    for item in oracles:
        if item["result"]["state_visit_budget"] != protocol["oracle_state_budget"]:
            raise ValueError("reused independent oracle budget changed")
    oracle_invalid = any(item["result"]["complete"] != item["result"]["canonical_exact"] for item in oracles)
    oracle_complete = all(item["result"]["complete"] and item["result"]["canonical_exact"] for item in oracles)
    entries, cells, cohorts, lane_groups, failures = [], [], defaultdict(list), defaultdict(list), []
    for row in rows:
        panels = protocol["queue_panels"][str(row["lanes"])]
        expected = [(state["state_id"], panel["panel"], tuple(state["queue"]), panel["cohort"]) for panel in panels for state in panel["states"]]
        actual = [(state["state_id"], state["panel"], tuple(state["queue"]), state["cohort"]) for state in row["states"]]
        if actual != expected or not actual or len({x[0] for x in actual}) != len(actual) or len({x[2] for x in actual}) != len(actual):
            raise ValueError("stored states differ from the deduplicated registered roster")
        cell_entries = []
        for state in row["states"]:
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            oracle = oracle_lookup.get((row["node"], row["lanes"], state["state_id"]))
            supplemented = oracle is not None and oracle["complete"] and oracle["canonical_exact"]
            complete = bool(state["reference_complete"] or supplemented)
            source = "full_K4096" if state["reference_complete"] else "independent_score_face_oracle" if supplemented else None
            if state["effective_reference_complete"] != complete or state["effective_reference_source"] != source:
                raise ValueError("only a frozen independent reference can make a state comparable")
            records = [(state[label]["metrics"], state[label]["selection"]) for label in LABELS]
            records.append((state["baseline_metrics"], state["completed"]["baseline_selection"]))
            for metrics, selection in records:
                _check_single_call(metrics, selection)
                if selection["lp_node_budget"] != protocol["budget"]:
                    raise ValueError("stored LP budget differs from the registered cap")
                if metrics["reference_comparable_count"] != int(complete):
                    raise ValueError("reference comparability differs from the available reference")
            errors = _completion_errors(state, protocol)
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entry = (row, state)
            entries.append(entry)
            cell_entries.append(entry)
            cohorts[state["cohort"]].append(entry)
            lane_groups[row["lanes"]].append(entry)
        cells.append({"node": row["node"], "lanes": row["lanes"], **_aggregate(cell_entries)})
    total = _aggregate(entries)
    valid = not oracle_invalid and not failures and all(
        not state["reference_metrics"]["failures"] and not state["oracle_crosscheck_failed"]
        and not state["baseline_metrics"]["failures"]
        and all(not state[label]["metrics"]["failures"] for label in LABELS) for _, state in entries)
    effective_complete = all(state["effective_reference_complete"] for _, state in entries)
    complete = oracle_complete and effective_complete
    action = valid and complete and total["label_rates"]["completed"]["certified_canonical_action_equal_count"] == len(entries)
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "SCORE_FACE_COMPLETION_HOLD" if not action else "SCORE_FACE_COMPLETION_DEVELOPMENT_ADVANCE")
    witnesses = []
    for witness in protocol["oracle_witnesses"]:
        group = [(row, state) for row, state in entries if (row["node"], row["lanes"], state["state_id"]) == (witness["node"], witness["lanes"], witness["state_id"])]
        if len(group) != 1:
            raise ValueError("registered oracle witness must occur exactly once")
        witnesses.append({**witness, **_aggregate(group)})
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "budget": protocol["budget"], "canonical_state_budget": protocol["canonical_state_budget"],
        "cell_count": len(rows), "executed_selector_calls": len(entries),
        "reused_legacy_calls": len(entries), "reused_reference_calls": len(entries),
        "executed_oracle_calls": 0, "reused_oracle_calls": len(oracles),
        "certificate_checks_passed": bool(valid), "completion_accounting_failures": failures,
        "baseline_reproduced": total["baseline_reproduced_count"] == len(entries),
        "original_references_complete": all(state["reference_complete"] for _, state in entries),
        "effective_references_complete": effective_complete, "independent_oracles_complete": oracle_complete,
        "development_action_gate_passed": bool(action),
        **total, "cells": cells, "cohorts": {name: _aggregate(group) for name, group in cohorts.items()},
        "by_lanes": [{"lanes": lanes, **_aggregate(group)} for lanes, group in lane_groups.items()],
        "witnesses": witnesses, "oracles": oracles, "prior_decision": protocol["prior_decision"],
        "limitations": protocol["limitations"],
    }
