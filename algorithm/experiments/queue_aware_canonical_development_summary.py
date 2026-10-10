"""Compare new joint-queue bounds with frozen predecessor actions and work."""
from __future__ import annotations

from collections import defaultdict

from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    _combine_metrics, CAPACITY_COUNTERS, COMPARISON_COUNTERS,
)
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call


LABELS = ("legacy", "queue_aware")


def _pointer(row, state):
    return {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "panel": state["panel"], "queue": state["queue"]}


def _aggregate(entries):
    metrics = {label: _combine_metrics(state[label]["metrics"] for _, state in entries) for label in LABELS}
    rates = {}
    for label in LABELS:
        current = metrics[label]
        observed = sum(state[label]["metrics"]["reference_comparable_count"] == 1
                       and not state[label]["metrics"]["score_mismatch_count"]
                       and not state[label]["metrics"]["canonical_action_mismatch_count"] for _, state in entries)
        rates[label] = {"score_exact_count": current["score_exact_count"],
                        "canonical_exact_count": current["canonical_exact_count"],
                        "observed_canonical_action_equal_count": observed,
                        "certified_canonical_action_equal_count": sum(
                            state[label]["selection"]["canonical_exact"]
                            and state[label]["metrics"]["reference_comparable_count"] == 1
                            and not state[label]["metrics"]["score_mismatch_count"]
                            and not state[label]["metrics"]["canonical_action_mismatch_count"]
                            for _, state in entries)}
    candidate, legacy = (metrics[label]["search_work"] for label in ("queue_aware", "legacy"))
    return {
        "queue_count": len(entries), "metrics": metrics, "label_rates": rates,
        "original_incomplete_reference_count": sum(not state["reference_complete"] for _, state in entries),
        "effective_incomplete_reference_count": sum(not state["effective_reference_complete"] for _, state in entries),
        "max_candidate_lp_calls": max(state["queue_aware"]["selection"]["lp_calls"] for _, state in entries),
        "first_candidate_unresolved": next((_pointer(row, state) for row, state in entries
                                            if not state["queue_aware"]["selection"]["canonical_exact"]), None),
        "first_candidate_mismatch": next((_pointer(row, state) for row, state in entries
                                          if state["queue_aware"]["metrics"]["score_mismatch_count"]
                                          or state["queue_aware"]["metrics"]["canonical_action_mismatch_count"]), None),
        "cost_ratios": {key: candidate[key] / legacy[key] if legacy[key] else None for key in COMPARISON_COUNTERS},
        "cost_nonincrease": all(candidate[key] <= legacy[key] for key in COMPARISON_COUNTERS),
        "cost_strict_decrease": all(candidate[key] < legacy[key] for key in COMPARISON_COUNTERS),
        "cost_regressions": {key: {
            "count": sum(state["queue_aware"]["selection"][key] > state["legacy"]["selection"][key] for _, state in entries),
            "first": next((_pointer(row, state) for row, state in entries
                           if state["queue_aware"]["selection"][key] > state["legacy"]["selection"][key]), None),
        } for key in COMPARISON_COUNTERS},
        "canonical_queue_bound": {key: sum(state["queue_aware"]["canonical_queue_bound"][key] for _, state in entries)
                                  for key in ("bound_calls", "requirement_coordinate_reads")},
        "capacity_propagation": {label: {key: sum(state[label]["capacity_propagation"].get(key, 0) for _, state in entries)
                                         for key in CAPACITY_COUNTERS} for label in LABELS},
    }


def summarize(models, protocol, rows, oracles):
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        raise ValueError("all registered cells must occur exactly once")
    expected_oracles = [(w["node"], w["lanes"], w["state_id"], tuple(w["queue"])) for w in protocol["oracle_witnesses"]]
    actual_oracles = [(w["node"], w["lanes"], w["state_id"], tuple(w["queue"])) for w in oracles]
    if actual_oracles != expected_oracles or len(set(actual_oracles)) != len(actual_oracles):
        raise ValueError("independent oracle coverage differs from the frozen witness roster")
    oracle_lookup = {(item["node"], item["lanes"], item["state_id"]): item["result"] for item in oracles}
    for item in oracles:
        if item["result"]["state_visit_budget"] != protocol["oracle_state_budget"]:
            raise ValueError("independent oracle state budget changed")
    oracle_invalid = any(item["result"]["complete"] and not item["result"]["canonical_exact"] for item in oracles)
    oracle_complete = all(item["result"]["complete"] and item["result"]["canonical_exact"] for item in oracles)
    entries, cells, cohorts, lane_groups = [], [], defaultdict(list), defaultdict(list)
    for row in rows:
        panels = protocol["queue_panels"][str(row["lanes"])]
        expected_states = [(state["state_id"], panel["panel"], tuple(state["queue"])) for panel in panels for state in panel["states"]]
        actual_states = [(state["state_id"], state["panel"], tuple(state["queue"])) for state in row["states"]]
        if (actual_states != expected_states or not actual_states
                or len({x[0] for x in actual_states}) != len(actual_states)
                or len({x[2] for x in actual_states}) != len(actual_states)):
            raise ValueError("stored states differ from the deduplicated registered roster")
        registered_cohorts = {panel["panel"]: panel["cohort"] for panel in panels}
        cell_entries = []
        for state in row["states"]:
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            independent = oracle_lookup.get((row["node"], row["lanes"], state["state_id"]))
            supplemented = independent is not None and independent["complete"] and independent["canonical_exact"]
            expected_complete = bool(state["reference_complete"] or supplemented)
            expected_source = ("full_K4096" if state["reference_complete"] else
                               "independent_score_face_oracle" if supplemented else None)
            if state["effective_reference_complete"] != expected_complete or state["effective_reference_source"] != expected_source:
                raise ValueError("only an original or independent complete reference can make a state comparable")
            for label in LABELS:
                record = state[label]
                _check_single_call(record["metrics"], record["selection"])
                if record["selection"]["lp_node_budget"] != protocol["budget"]:
                    raise ValueError("stored selector budget differs from the registered cap")
                if record["metrics"]["reference_comparable_count"] != int(expected_complete):
                    raise ValueError("reference comparability differs from the available reference")
            entry = (row, state)
            entries.append(entry)
            cell_entries.append(entry)
            cohorts[registered_cohorts[state["panel"]]].append(entry)
            lane_groups[row["lanes"]].append(entry)
        cells.append({"node": row["node"], "lanes": row["lanes"], **_aggregate(cell_entries)})
    total = _aggregate(entries)
    valid = (not oracle_invalid and all(not state["reference_metrics"]["failures"]
             and not state["oracle_crosscheck_failed"]
             and all(not state[label]["metrics"]["failures"] for label in LABELS) for _, state in entries))
    effective_complete = all(state["effective_reference_complete"] for _, state in entries)
    complete = oracle_complete and effective_complete
    action = valid and complete and total["label_rates"]["queue_aware"]["certified_canonical_action_equal_count"] == len(entries)
    cost = valid and complete and all(cell["cost_nonincrease"] for cell in cells) and total["cost_strict_decrease"]
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_INDEPENDENT_REFERENCE" if not complete else
                "QUEUE_AWARE_CANONICAL_ACTION_HOLD" if not action else
                "QUEUE_AWARE_CANONICAL_COST_HOLD" if not cost else "QUEUE_AWARE_CANONICAL_DEVELOPMENT_ADVANCE")
    witnesses = []
    for witness in protocol["oracle_witnesses"]:
        selected = [(row, state) for row, state in entries
                    if (row["node"], row["lanes"], state["state_id"]) == (witness["node"], witness["lanes"], witness["state_id"])]
        if len(selected) != 1:
            raise ValueError("registered independent witness must occur exactly once")
        witnesses.append({**witness, **_aggregate(selected)})
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "budget": protocol["budget"], "cell_count": len(rows),
        "executed_selector_calls": len(entries), "reused_legacy_calls": len(entries),
        "reused_reference_calls": len(entries), "executed_oracle_calls": len(oracles),
        "certificate_checks_passed": valid,
        "original_references_complete": all(state["reference_complete"] for _, state in entries),
        "effective_references_complete": effective_complete, "independent_oracles_complete": oracle_complete,
        "development_action_gate_passed": bool(action), "development_cost_gate_passed": bool(cost),
        **total, "cells": cells,
        "cohorts": {name: _aggregate(group) for name, group in cohorts.items()},
        "by_lanes": [{"lanes": lanes, **_aggregate(group)} for lanes, group in lane_groups.items()],
        "witnesses": witnesses, "oracles": oracles, "prior_decision": protocol["prior_decision"],
        "limitations": protocol["limitations"],
    }
