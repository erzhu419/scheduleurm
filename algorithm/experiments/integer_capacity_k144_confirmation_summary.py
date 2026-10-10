"""Evaluate a fixed K144 confirmation without pooling in regression states."""
from __future__ import annotations

from collections import defaultdict

from algorithm.experiments import integer_capacity_budget_frontier_summary as frontier
from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    COMPARISON_COUNTERS, CONTROLS,
)


BUDGET = 144
CANDIDATE = "integer_capacity"


def _group(states, labels):
    result = frontier._group(states, (BUDGET,), labels)
    result.pop("budget_boundaries")
    return result


def _cost_regressions(states):
    result = {control: {counter: {"count": 0, "first": None}
                        for counter in COMPARISON_COUNTERS} for control in CONTROLS}
    for row, state in states:
        selected = state["budgets"][str(BUDGET)]
        candidate = selected[CANDIDATE]["metrics"]["search_work"]
        for control in CONTROLS:
            baseline = selected[control]["metrics"]["search_work"]
            for counter in COMPARISON_COUNTERS:
                if candidate[counter] > baseline[counter]:
                    item = result[control][counter]
                    item["count"] += 1
                    if item["first"] is None:
                        item["first"] = {
                            **frontier._pointer(row, state, BUDGET, CANDIDATE),
                            "candidate": candidate[counter], "control": baseline[counter],
                        }
    return result


def _cohort(states, labels):
    if not states:
        raise ValueError("fresh and regression cohorts must both contain registered states")
    result = _group(states, labels)
    cells, lanes, panels = defaultdict(list), defaultdict(list), defaultdict(list)
    for row, state in states:
        cells[(row["node"], row["lanes"])].append((row, state))
        lanes[row["lanes"]].append((row, state))
        panels[state["panel"]].append((row, state))
    result.update(
        cells=[{"node": node, "lanes": lane, **_group(group, labels)}
               for (node, lane), group in cells.items()],
        by_lanes=[{"lanes": lane, **_group(group, labels)} for lane, group in lanes.items()],
        panels=[{"panel": panel, **_group(group, labels)} for panel, group in panels.items()],
        cost_regressions=_cost_regressions(states),
        max_candidate_lp_calls=max(state["budgets"][str(BUDGET)][CANDIDATE]
                                   ["metrics"]["search_work"]["lp_calls"] for _, state in states),
        first_unresolved=next((frontier._pointer(row, state, BUDGET, CANDIDATE)
                               for row, state in states
                               if not state["budgets"][str(BUDGET)][CANDIDATE]
                               ["selection"]["canonical_exact"]), None),
        first_mismatch=result["curves"][0]["metrics"][CANDIDATE]["first_mismatch"],
    )
    valid = all(not state["reference_metrics"]["failures"]
                and all(not item["metrics"]["failures"]
                        for item in state["budgets"][str(BUDGET)].values()) for _, state in states)
    complete = all(state["reference_complete"] for _, state in states)
    result["certificate_checks_passed"] = valid
    result["references_complete"] = complete
    result["action_gate_passed"] = bool(
        valid and complete and result["curves"][0]["label_rates"][CANDIDATE]
        ["certified_canonical_action_equal_count"] == len(states))
    pooled = result["curves"][0]["metrics"]
    result["search_cost_gate_passed"] = bool(
        valid and complete
        and all(cell["curves"][0]["metrics"][CANDIDATE]["search_work"][counter]
                <= cell["curves"][0]["metrics"][control]["search_work"][counter]
                for cell in result["cells"] for control in CONTROLS for counter in COMPARISON_COUNTERS)
        and all(pooled[CANDIDATE]["search_work"][counter] < pooled[control]["search_work"][counter]
                for control in CONTROLS for counter in COMPARISON_COUNTERS))
    return result


def summarize(models, protocol, rows):
    if tuple(protocol["budget_grid"]) != (BUDGET,):
        raise ValueError("confirmation executes the frozen K144 budget only")
    base = frontier.summarize(models, protocol, rows)
    registered_cohorts = {}
    for lanes in protocol["lanes"]:
        for panel in protocol["queue_panels"][str(lanes)]:
            cohort = panel["cohort"]
            key = (lanes, panel["panel"])
            if cohort not in ("fresh", "regression") or key in registered_cohorts:
                raise ValueError("each registered panel must have one fresh or regression cohort")
            registered_cohorts[key] = cohort
    cohorts = {name: [] for name in ("fresh", "regression")}
    for row in rows:
        for state in row["states"]:
            cohorts[registered_cohorts[(row["lanes"], state["panel"])]].append((row, state))
    fresh, regression = (_cohort(cohorts[name], protocol["labels"])
                         for name in ("fresh", "regression"))
    valid, complete = base["certificate_checks_passed"], base["references_complete"]
    fresh_pass = valid and complete and fresh["action_gate_passed"]
    regression_pass = valid and complete and regression["action_gate_passed"]
    decision = (
        "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
        "K144_FRESH_CONFIRMATION_HOLD" if not fresh_pass else
        "K144_REGRESSION_HOLD" if not regression_pass else "INTEGER_CAPACITY_K144_CONFIRMED"
    )
    witnesses = [{key: value for key, value in witness.items() if key != "budget_boundaries"}
                 for witness in base["witnesses"]]
    return {
        **{key: base[key] for key in (
            "schema_version", "experiment", "registered_at_utc", "hardware_catalogue_count",
            "cell_count", "distinct_lane_queue_count", "queue_count", "executed_reference_calls",
            "executed_candidate_calls", "executed_selector_calls", "certificate_checks_passed",
            "references_complete", "reference_metrics", "first_incomplete_reference")},
        "budget": BUDGET, "decision": decision,
        "confirmation_action_gate_passed": bool(fresh_pass),
        "regression_action_gate_passed": bool(regression_pass),
        "confirmation_search_cost_gate_passed": bool(valid and complete and fresh["search_cost_gate_passed"]),
        "fresh": fresh, "regression": regression, "witnesses": witnesses,
        "limitations": protocol["limitations"],
    }
