"""Summarize the fixed-catalogue large-R screen without merging budget regimes."""
from __future__ import annotations

from collections import defaultdict

from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    CAPACITY_COUNTERS, COMPARISON_COUNTERS, CONTROLS, _combine_metrics,
)


CANDIDATE = "integer_capacity"


def _pointer(row, state, *, budget=None, label=None):
    point = {"node": row["node"], "lanes": row["lanes"],
             "state_id": state["state_id"], "panel": state["panel"],
             "queue": state["queue"]}
    if budget is not None:
        point["budget"] = budget
    if label is not None:
        point["label"] = label
    return point


def _rate_metrics(metrics):
    count = metrics["calls"]
    return {
        "calls": count,
        "score_exact_count": metrics["score_exact_count"],
        "canonical_exact_count": metrics["canonical_exact_count"],
        "canonical_exact_fraction": metrics["canonical_exact_count"] / count if count else None,
        "reference_comparable_count": metrics["reference_comparable_count"],
        "reference_comparable_fraction": metrics["reference_comparable_count"] / count if count else None,
        "score_reference_comparable_count": metrics["score_reference_comparable_count"],
        "score_mismatch_count": metrics["score_mismatch_count"],
        "canonical_action_mismatch_count": metrics["canonical_action_mismatch_count"],
    }


def _aggregate(entries, labels):
    """Each entry contains one actual state and the chosen regime's cold calls."""
    metrics = {label: _combine_metrics(entry[2][label]["metrics"] for entry in entries)
               for label in labels}
    first_unresolved = dict.fromkeys(labels)
    first_mismatch = dict.fromkeys(labels)
    first_failure = dict.fromkeys(labels)
    for row, state, selected, budget in entries:
        for label in labels:
            current = selected[label]["metrics"]
            if not current["canonical_exact_count"] and first_unresolved[label] is None:
                first_unresolved[label] = _pointer(row, state, budget=budget, label=label)
            if ((current["score_mismatch_count"] or current["canonical_action_mismatch_count"])
                    and first_mismatch[label] is None):
                first_mismatch[label] = _pointer(row, state, budget=budget, label=label)
            if current["failures"] and first_failure[label] is None:
                first_failure[label] = _pointer(row, state, budget=budget, label=label)
    for label in labels:
        metrics[label]["first_failure"] = first_failure[label]
        metrics[label]["first_mismatch"] = first_mismatch[label]
    candidate = metrics[CANDIDATE]
    return {
        "queue_count": len(entries), "metrics": metrics,
        "label_rates": {label: _rate_metrics(value) for label, value in metrics.items()},
        "action_gate_passed": (
            candidate["canonical_exact_count"] == len(entries)
            and candidate["reference_comparable_count"] == len(entries)
            and not candidate["score_mismatch_count"]
            and not candidate["canonical_action_mismatch_count"]
        ),
        "cost_nonincrease_by_control": {
            control: all(candidate["search_work"][counter] <= metrics[control]["search_work"][counter]
                         for counter in COMPARISON_COUNTERS)
            for control in CONTROLS
        },
        "capacity_propagation": {
            key: sum(entry[2][CANDIDATE]["capacity_propagation"].get(key, 0) for entry in entries)
            for key in CAPACITY_COUNTERS
        },
        "first_unresolved": first_unresolved, "first_mismatch": first_mismatch,
    }


def _check_single_call(metrics, selection, *, reference_complete=None):
    if metrics["calls"] != 1:
        raise ValueError("each stored selection must account for exactly one cold call")
    for flag in ("score_exact", "canonical_exact"):
        if metrics[f"{flag}_count"] != int(selection[flag]):
            raise ValueError("selection exact flags disagree with stored metrics")
    if reference_complete is not None:
        certified = bool(selection["exact"] and selection["score_exact"] and selection["canonical_exact"])
        if reference_complete != certified:
            raise ValueError("stored reference completeness disagrees with its certificate")


def summarize(models, protocol, rows):
    labels = tuple(protocol["labels"])
    if set(labels) != {*CONTROLS, CANDIDATE} or len(labels) != 3:
        raise ValueError("the three registered selectors must appear exactly once")
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        raise ValueError("all registered node/R cells must appear exactly once")

    all_states, all_call_metrics, references = [], [], []
    for row in rows:
        lanes = str(row["lanes"])
        expected_states = [
            (state["state_id"], panel["panel"], state["queue"])
            for panel in protocol["queue_panels"][lanes] for state in panel["states"]
        ]
        actual_states = [(state["state_id"], state["panel"], state["queue"])
                         for state in row["states"]]
        if actual_states != expected_states or len({item[0] for item in actual_states}) != len(actual_states):
            raise ValueError("cell state coverage or queue differs from the registered panel")
        expected_budgets = {str(cap) for cap in protocol["unique_budgets"][lanes]}
        regime_budgets = {str(caps[lanes]) for caps in protocol["budget_regimes"].values()}
        if expected_budgets != regime_budgets:
            raise ValueError("unique budgets do not represent the registered regimes")
        for state in row["states"]:
            _check_single_call(state["reference_metrics"], state["reference"],
                               reference_complete=state["reference_complete"])
            references.append(state["reference_metrics"])
            all_call_metrics.append(state["reference_metrics"])
            all_states.append((row, state))
            if set(state["budgets"]) != expected_budgets:
                raise ValueError("stored budget coverage differs from registered unique caps")
            for results in state["budgets"].values():
                if set(results) != set(labels):
                    raise ValueError("stored selectors differ from the registered labels")
                for item in results.values():
                    _check_single_call(item["metrics"], item["selection"])
                    if item["metrics"]["reference_comparable_count"] != int(state["reference_complete"]):
                        raise ValueError("reference comparability must retain incomplete reference states")
                    all_call_metrics.append(item["metrics"])

    valid = all(not metrics["failures"] for metrics in all_call_metrics)
    complete = all(state["reference_complete"] for _, state in all_states)
    regimes = {}
    for name, caps in protocol["budget_regimes"].items():
        entries = [(row, state, state["budgets"][str(caps[str(row["lanes"])])], caps[str(row["lanes"])])
                   for row, state in all_states]
        result = _aggregate(entries, labels)
        grouped_cells, grouped_lanes, grouped_panels = defaultdict(list), defaultdict(list), defaultdict(list)
        for entry in entries:
            row, state = entry[:2]
            grouped_cells[(row["node"], row["lanes"])].append(entry)
            grouped_lanes[row["lanes"]].append(entry)
            grouped_panels[state["panel"]].append(entry)
        result["cells"] = [
            {"node": node, "lanes": lanes, "budget": caps[str(lanes)], **_aggregate(group, labels)}
            for (node, lanes), group in grouped_cells.items()
        ]
        result["by_lanes"] = [
            {"lanes": lanes, "budget": caps[str(lanes)], **_aggregate(group, labels)}
            for lanes, group in grouped_lanes.items()
        ]
        result["panels"] = [
            {"panel": panel, **_aggregate(group, labels)} for panel, group in grouped_panels.items()
        ]
        result["cost_gate_by_control"] = {
            control: (
                all(cell["cost_nonincrease_by_control"][control] for cell in result["cells"])
                and all(result["metrics"][CANDIDATE]["search_work"][key]
                        < result["metrics"][control]["search_work"][key]
                        for key in COMPARISON_COUNTERS)
            ) for control in CONTROLS
        }
        result["cost_gate_passed"] = all(result["cost_gate_by_control"].values())
        regimes[name] = result

    scaled = regimes["scaled_36r"]
    action = valid and complete and scaled["action_gate_passed"]
    cost = scaled["cost_gate_passed"]
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "LARGE_R_ACTION_REVIEW_REQUIRED" if not action else
                "INTEGER_CAPACITY_LARGE_R_SCALING_ADVANCE" if cost else
                "LARGE_R_SEARCH_COST_REVIEW_REQUIRED")
    reference_metrics = _combine_metrics(references)
    reference_metrics["first_failure"] = next(
        (_pointer(row, state, budget=protocol["reference_budget"], label="reference")
         for row, state in all_states if state["reference_metrics"]["failures"]), None)
    reference_metrics["first_mismatch"] = next(
        (_pointer(row, state, budget=protocol["reference_budget"], label="reference")
         for row, state in all_states if state["reference_metrics"]["score_mismatch_count"]
         or state["reference_metrics"]["canonical_action_mismatch_count"]), None)
    return {
        "schema_version": 1, "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models), "cell_count": len(rows),
        "distinct_lane_queue_count": sum(len(panel["states"])
                                         for panels in protocol["queue_panels"].values() for panel in panels),
        "queue_count": len(all_states), "executed_reference_calls": len(references),
        "executed_candidate_calls": len(all_call_metrics) - len(references),
        "executed_selector_calls": len(all_call_metrics),
        "certificate_checks_passed": valid, "references_complete": complete,
        "scaled_action_gate_passed": action, "scaled_cost_gate_passed": cost,
        "decision": decision, "reference_metrics": reference_metrics,
        "first_incomplete_reference": next((_pointer(row, state) for row, state in all_states
                                            if not state["reference_complete"]), None),
        "regimes": regimes, "limitations": protocol["limitations"],
    }
