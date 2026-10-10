"""Characterize tested budget frontiers without assuming monotone closure."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    CAPACITY_COUNTERS, COMPARISON_COUNTERS, CONTROLS, _combine_metrics,
)
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call


CANDIDATE = "integer_capacity"


def _pointer(row, state, budget=None, label=None):
    result = {"node": row["node"], "lanes": row["lanes"],
              "state_id": state["state_id"], "panel": state["panel"],
              "queue": state["queue"]}
    if budget is not None:
        result["budget"] = budget
    if label is not None:
        result["label"] = label
    return result


def _score(selection, queue):
    return Fraction(sum(q * s for q, s in zip(queue, selection["service"], strict=True))) / Fraction.from_float(selection["duration"])


def _upper(selection, queue):
    return (_score(selection, queue) if selection["score_exact"]
            else Fraction.from_float(selection["upper_bound"]))


def _observed(state, result):
    metrics = result["metrics"]
    score_equal = bool(metrics["score_reference_comparable_count"]
                       and not metrics["score_mismatch_count"])
    action_equal = bool(state["reference_complete"] and score_equal
                        and not metrics["canonical_action_mismatch_count"])
    return score_equal, action_equal


def _curve(states, budget, labels):
    metrics = {label: _combine_metrics(state["budgets"][str(budget)][label]["metrics"]
                                      for _, state in states) for label in labels}
    rates, gaps = {}, {}
    for label in labels:
        results = [(state, state["budgets"][str(budget)][label]) for _, state in states]
        current = metrics[label]
        observations = [_observed(state, result) for state, result in results]
        rates[label] = {
            "score_exact_count": current["score_exact_count"],
            "canonical_exact_count": current["canonical_exact_count"],
            "score_reference_comparable_count": current["score_reference_comparable_count"],
            "reference_comparable_count": current["reference_comparable_count"],
            "observed_score_equal_count": sum(item[0] for item in observations),
            "observed_canonical_action_equal_count": sum(item[1] for item in observations),
            "certified_canonical_action_equal_count": sum(
                observed[1] and result["selection"]["canonical_exact"]
                for observed, (_, result) in zip(observations, results, strict=True)),
        }
        normalized = [result["selection"]["gap"] / result["selection"]["upper_bound"]
                      if result["selection"]["upper_bound"] > 0 else 0.0
                      for _, result in results]
        gaps[label] = {
            "mean": sum(normalized) / len(normalized) if normalized else None,
            "max": max(normalized, default=None),
            "zero_score_count": sum(result["selection"]["score"] == 0 for _, result in results),
            "zero_upper_bound_count": sum(result["selection"]["upper_bound"] == 0 for _, result in results),
            "zero_gap_uncertified_count": sum(
                result["selection"]["gap"] == 0 and not result["selection"]["canonical_exact"]
                for _, result in results),
        }
        for field, predicate in (
            ("first_failure", lambda item: item["metrics"]["failures"]),
            ("first_mismatch", lambda item: item["metrics"]["score_mismatch_count"]
             or item["metrics"]["canonical_action_mismatch_count"]),
        ):
            current[field] = next((_pointer(row, state, budget, label) for row, state in states
                                   if predicate(state["budgets"][str(budget)][label])), None)
    return {
        "budget": budget, "queue_count": len(states), "metrics": metrics,
        "label_rates": rates, "normalized_gaps": gaps,
        "capacity_propagation": {
            key: sum(state["budgets"][str(budget)][CANDIDATE]["capacity_propagation"].get(key, 0)
                     for _, state in states) for key in CAPACITY_COUNTERS
        },
        "cost_ratios": {control: {
            key: (metrics[CANDIDATE]["search_work"][key] / metrics[control]["search_work"][key]
                  if metrics[control]["search_work"][key] else None)
            for key in COMPARISON_COUNTERS} for control in CONTROLS},
    }


def _boundaries(curves, labels, eligible):
    fields = {"observed_canonical_action": "observed_canonical_action_equal_count",
              "certified_canonical": "certified_canonical_action_equal_count"}
    return {label: {name: next((curve["budget"] for index, curve in enumerate(curves)
                               if eligible and all(
                                   later["label_rates"][label][field] == later["queue_count"]
                                   for later in curves[index:])), None)
                    for name, field in fields.items()} for label in labels}


def _group(states, budgets, labels):
    curves = [_curve(states, budget, labels) for budget in budgets]
    eligible = all(state["reference_complete"] and not state["reference_metrics"]["failures"]
                   and all(not result["metrics"]["failures"]
                           for selected in state["budgets"].values() for result in selected.values())
                   for _, state in states)
    return {"queue_count": len(states), "curves": curves,
            "budget_boundaries": _boundaries(curves, labels, eligible)}


def _regressions(states, budgets, labels):
    names = ("score_certificate_lost", "canonical_certificate_lost", "observed_action_lost",
             "score_worsened", "upper_bound_increased")
    result = {label: {name: {"count": 0, "first": None} for name in names} for label in labels}
    for row, state in states:
        for low, high in zip(budgets, budgets[1:]):
            for label in labels:
                before, after = (state["budgets"][str(budget)][label] for budget in (low, high))
                a, b = before["selection"], after["selection"]
                checks = {
                    "score_certificate_lost": a["score_exact"] and not b["score_exact"],
                    "canonical_certificate_lost": a["canonical_exact"] and not b["canonical_exact"],
                    "observed_action_lost": _observed(state, before)[1] and not _observed(state, after)[1],
                    "score_worsened": _score(b, state["queue"]) < _score(a, state["queue"]),
                    "upper_bound_increased": _upper(b, state["queue"]) > _upper(a, state["queue"]),
                }
                for name, regressed in checks.items():
                    if regressed:
                        item = result[label][name]
                        item["count"] += 1
                        if item["first"] is None:
                            item["first"] = {**_pointer(row, state, label=label),
                                             "lower_budget": low, "higher_budget": high,
                                             "before_score": str(_score(a, state["queue"])),
                                             "after_score": str(_score(b, state["queue"])),
                                             "before_upper": str(_upper(a, state["queue"])),
                                             "after_upper": str(_upper(b, state["queue"]))}
    return result


def summarize(models, protocol, rows):
    labels, budgets = tuple(protocol["labels"]), tuple(protocol["budget_grid"])
    if set(labels) != {*CONTROLS, CANDIDATE} or len(labels) != 3:
        raise ValueError("the three registered selectors must appear exactly once")
    if not budgets or budgets != tuple(sorted(set(budgets))):
        raise ValueError("budget grid must be nonempty, sorted and unique")
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        raise ValueError("all registered node/R cells must appear exactly once")
    states, all_metrics, references = [], [], []
    for row in rows:
        lanes = str(row["lanes"])
        if tuple(protocol["unique_budgets"][lanes]) != budgets:
            raise ValueError("every node/R cell must execute every registered budget")
        expected_states = [(state["state_id"], panel["panel"], tuple(state["queue"]))
                           for panel in protocol["queue_panels"][lanes] for state in panel["states"]]
        actual_states = [(state["state_id"], state["panel"], tuple(state["queue"])) for state in row["states"]]
        if (actual_states != expected_states or not actual_states
                or len({item[0] for item in actual_states}) != len(actual_states)
                or len({item[2] for item in actual_states}) != len(actual_states)):
            raise ValueError("cell state coverage or queue differs from the deduplicated registered panel")
        for state in row["states"]:
            _check_single_call(state["reference_metrics"], state["reference"],
                               reference_complete=state["reference_complete"])
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"]:
                raise ValueError("reference budget differs from registration")
            references.append(state["reference_metrics"])
            all_metrics.append(state["reference_metrics"])
            states.append((row, state))
            if set(state["budgets"]) != {str(budget) for budget in budgets}:
                raise ValueError("stored budget coverage differs from registration")
            for budget, selected in state["budgets"].items():
                if set(selected) != set(labels):
                    raise ValueError("stored selectors differ from registration")
                for item in selected.values():
                    metrics, selection = item["metrics"], item["selection"]
                    _check_single_call(metrics, selection)
                    if selection["lp_node_budget"] != int(budget):
                        raise ValueError("selection budget differs from registered cold call")
                    if metrics["reference_comparable_count"] != int(state["reference_complete"]):
                        raise ValueError("reference comparability must retain incomplete references")
                    all_metrics.append(metrics)
    valid = all(not metrics["failures"] for metrics in all_metrics)
    complete = all(state["reference_complete"] for _, state in states)
    pooled = _group(states, budgets, labels)
    high_cap = (valid and complete and pooled["curves"][-1]["label_rates"][CANDIDATE]
                ["certified_canonical_action_equal_count"] == len(states))
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "INTEGER_CAPACITY_BUDGET_FRONTIER_CHARACTERIZED" if high_cap else
                "BUDGET_FRONTIER_HIGH_CAP_HOLD")
    grouped_cells, grouped_lanes, grouped_panels = defaultdict(list), defaultdict(list), defaultdict(list)
    for row, state in states:
        grouped_cells[(row["node"], row["lanes"])].append((row, state))
        grouped_lanes[row["lanes"]].append((row, state))
        grouped_panels[state["panel"]].append((row, state))
    witnesses = []
    for witness in protocol["retained_witnesses"]:
        selected = [(row, state) for row, state in states
                    if row["node"] == witness["node"] and row["lanes"] == witness["lanes"]
                    and tuple(state["queue"]) == tuple(witness["queue"])]
        if len(selected) != 1:
            raise ValueError("registered witness must occur exactly once at its original node/R")
        witnesses.append({**witness, **_group(selected, budgets, labels)})
    return {
        "schema_version": 1, "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models), "cell_count": len(rows),
        "distinct_lane_queue_count": sum(len(panel["states"])
                                         for panels in protocol["queue_panels"].values() for panel in panels),
        "queue_count": len(states), "executed_reference_calls": len(references),
        "executed_candidate_calls": len(all_metrics) - len(references),
        "executed_selector_calls": len(all_metrics),
        "certificate_checks_passed": valid, "references_complete": complete,
        "high_cap_action_gate_passed": high_cap, "decision": decision,
        "reference_metrics": _combine_metrics(references),
        "first_incomplete_reference": next((_pointer(row, state) for row, state in states
                                            if not state["reference_complete"]), None),
        "budget_boundaries": pooled["budget_boundaries"], "curves": pooled["curves"],
        "budget_regressions": _regressions(states, budgets, labels),
        "breakdowns": {
            "cells": [{"node": node, "lanes": lanes, **_group(group, budgets, labels)}
                      for (node, lanes), group in grouped_cells.items()],
            "by_lanes": [{"lanes": lanes, **_group(group, budgets, labels)}
                         for lanes, group in grouped_lanes.items()],
            "panels": [{"panel": panel, **_group(group, budgets, labels)}
                       for panel, group in grouped_panels.items()],
        },
        "witnesses": witnesses, "limitations": protocol["limitations"],
    }
