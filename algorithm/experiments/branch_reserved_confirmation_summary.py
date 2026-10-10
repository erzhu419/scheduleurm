"""Fresh confirmation uses existing gates plus the reserved branch LP allowance."""
from __future__ import annotations

from algorithm.experiments.root_pruning_confirmation_summary import summarize as root_pruning_summary
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage, _increases


def summarize(protocol, rows):
    result = root_pruning_summary(protocol, rows)
    models = {model["node"]: model for model in protocol["models"]}
    reserved_failures, increases = [], []
    for cell, row in zip(protocol["cells"], rows):
        minima = {}
        for option in models[cell["node"]]["options"]:
            requirement = tuple(option["requirements"])
            minima[requirement] = min(minima.get(requirement, float("inf")), max(1.0, option["duration"]))
        roots = len(set(minima.values()))
        for state in row["states"]:
            pointer = {"node": row["node"], "lanes": row["lanes"], **{
                key: state[key] for key in ("state_id", "queue", "source", "cohort", "panel", "seed")}}
            errors = []
            for cap, stage in state["traces"]["new"].items():
                errors.extend(f"new cap{cap}: " + message for message in audit_reserved_stage(stage, int(cap), roots))
            routed = state["routed"]
            for name, cap in (("initial_lp", protocol["initial_lp_budget"]),
                              ("final_lp", protocol["lp_node_budget"] if routed["resumed"] else protocol["initial_lp_budget"])):
                errors.extend("new routed " + name + ": " + message for message in audit_reserved_stage(routed[name], cap, roots))
            if errors:
                reserved_failures.append({**pointer, "errors": errors})
            if all("256" in state["traces"][label] for label in ("old", "new")):
                increase = _increases(state)
                if increase:
                    increases.append({**pointer, "increases": increase})
    result["reserved_budget_checks_passed"] = not reserved_failures
    result["reserved_budget_failures"] = reserved_failures
    result["work_increases"] = increases
    result["failures"].extend(reserved_failures)
    if reserved_failures:
        result["execution_checks_passed"] = False
        result["confirmation_gate_passed"] = False
        result["decision"] = "INVALID_EXECUTION"
    else:
        result["decision"] = {
            "ROOT_PRUNING_FRESH_CONFIRMATION_ADVANCE": "BRANCH_RESERVED_FRESH_CONFIRMATION_ADVANCE",
            "ROOT_PRUNING_FRESH_CONFIRMATION_HOLD": "BRANCH_RESERVED_FRESH_CONFIRMATION_HOLD",
        }.get(result["decision"], result["decision"])
    return result
