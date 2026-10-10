"""Independent fresh confirmation of certified dual reuse and reserved LP work."""
from __future__ import annotations

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import selection_score
from algorithm.experiments.branch_reserved_development_summary import _identity, _key, _increases, audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import _dual_work, audit_dual_stage
from algorithm.experiments.root_pruning_confirmation_summary import SOURCES, _cost, _work as _base_work, summarize as _base_summary


def _work(states):
    native = [state["traces"]["new"]["256"] for state in states]
    routed = [state["routed"]["final_lp"] for state in states]
    work = _base_work(states)
    work["kernel_new"]["dual_reuse"] = _dual_work(native)
    work["routed_new"]["dual_reuse"] = _dual_work(routed)
    work["executed_dual_reuse"] = _dual_work(native + routed)
    return work


def _native_cost(states):
    result = []
    for position in range(4):
        amounts = {key: {label: sum(state["traces"][label][str(state["caps"][position])]["selection"][key]
                                  for state in states) for label in ("old", "new")}
                   for key in ("lp_calls", "lp_matrix_coefficients")}
        result.append({"position": position, "states": len(states), **amounts,
                       "cost_nonincrease": all(value["new"] <= value["old"] for value in amounts.values())})
    return result


def summarize(protocol, rows):
    result = _base_summary(protocol, rows)
    models = {model["node"]: model for model in protocol["models"]}
    failures, increases, routed_losses, entries = [], [], [], []
    historical = {_key(item["node"], item["lanes"], item) for item in protocol["historical_regression_keys"]}
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    if len(historical) != len(protocol["historical_regression_keys"]) or not historical <= registered:
        failures.append({"errors": ["fixed historical keys must be unique and contained in the frozen roster"]})
    count = result["state_count"]
    expected_counts = {"expected_selector_calls": count * 2, "expected_native_advance_calls": count * 8,
                       "expected_interface_calls": count * 10, "expected_native_constructions": count * 4,
                       "expected_new_reference_calls": result["new_reference_calls"]}
    if any(protocol[key] != value for key, value in expected_counts.items()):
        failures.append({"errors": ["executed coverage differs from the registered method and native interface counts"]})
    for cell, row, cell_result in zip(protocol["cells"], rows, result["cells"]):
        point = {"node": row["node"], "lanes": row["lanes"]}
        identity = lambda state: (_identity(state), state.get("source_state_id"), tuple(state["caps"]))
        if list(map(identity, row["states"])) != list(map(identity, cell["states"])):
            failures.append({**point, "errors": ["state metadata differs from the frozen fresh confirmation roster"]})
        minima = {}
        for option in models[cell["node"]]["options"]:
            key = tuple(option["requirements"])
            minima[key] = min(minima.get(key, float("inf")), max(1.0, option["duration"]))
        roots = len(set(minima.values()))
        cell_entries = []
        for state in row["states"]:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "source", "cohort", "panel", "seed", "historical_regression")}}
            errors = []
            if state["historical_regression"] != (state["source"] == "historical_regression") or state["historical_regression"] != (
                    _key(row["node"], row["lanes"], state) in historical):
                errors.append("historical subgroup differs from its independently frozen source and keys")
            coverage = all(set(state["traces"][label]) == {str(cap) for cap in state["caps"]} for label in ("old", "new"))
            if not coverage:
                if errors:
                    failures.append({**pointer, "errors": errors})
                continue
            for label, field in (("old", "old_routed"), ("new", "routed")):
                previous = None
                for cap in state["caps"]:
                    stage = state["traces"][label][str(cap)]
                    errors.extend(f"{label} cap{cap}: " + message for message in audit_reserved_stage(stage, cap, roots))
                    if previous is not None and stage["root_pruning"] != previous["root_pruning"]:
                        errors.append(label + " resumption revisited duration roots")
                    if label == "new":
                        errors.extend(f"new cap{cap}: " + message for message in audit_dual_stage(stage, len(state["queue"]), previous))
                    previous = stage
                routed = state[field]
                for name, cap in (("initial_lp", protocol["initial_lp_budget"]),
                                  ("final_lp", protocol["lp_node_budget"] if routed["resumed"] else protocol["initial_lp_budget"])):
                    errors.extend(label + " routed " + name + ": " + message for message in audit_reserved_stage(routed[name], cap, roots))
                if label == "new":
                    errors.extend(audit_dual_stage(routed["initial_lp"], len(state["queue"])))
                    errors.extend(audit_dual_stage(routed["final_lp"], len(state["queue"]), routed["initial_lp"]))
            scores = [selection_score(selection_from_record(state[field]["selection"]), tuple(state["queue"]))
                      for field in ("old_routed", "routed")]
            if scores[1] < scores[0]:
                routed_losses.append({**pointer, "old_score": str(scores[0]), "new_score": str(scores[1])})
            routed_increase = _increases(state)
            native_increase = [item for item in _native_cost([state]) if not item["cost_nonincrease"]]
            if routed_increase or native_increase:
                increases.append({**pointer, "routed": routed_increase, "native": native_increase})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            cell_entries.append(state)
        for source in SOURCES:
            states = [state for state in cell_entries if state["source"] == source]
            group = cell_result["groups"][source]
            group["work"] = _work(states)
            group["cost_nonincrease"] = _cost(group["work"])
            group["native_per_budget"] = _native_cost(states)
            group["native_cost_nonincrease"] = all(item["cost_nonincrease"] for item in group["native_per_budget"])
    for source in SOURCES:
        states = [state for state in entries if state["source"] == source]
        group = result["groups"][source]
        group["work"] = _work(states)
        a, b = (group["work"][label] for label in ("routed_old", "routed_new"))
        group["cost_nonincrease_in_every_cell"] = all(cell["groups"][source]["cost_nonincrease"] for cell in result["cells"])
        group["native_cost_nonincrease_in_every_cell"] = all(cell["groups"][source]["native_cost_nonincrease"] for cell in result["cells"])
        group["native_per_budget"] = _native_cost(states)
        group["strict_pooled_integer_reduction"] = bool(states and b["shared_integer_states"] < a["shared_integer_states"])
        group["strict_pooled_lp_or_integer_reduction"] = group["strict_pooled_lp_reduction"] or group["strict_pooled_integer_reduction"]
        group["quality_gate_passed"] &= not any(item["source"] == source for item in result["routed_nonclosures"])
    result["failures"].extend(failures)
    valid = not result["failures"]
    fresh, hist = (result["groups"][source] for source in SOURCES[:2])
    quality = bool(fresh["state_count"] and all(group["quality_gate_passed"] for group in result["groups"].values()))
    native = all(group["native_cost_nonincrease_in_every_cell"] for group in (fresh, hist))
    fresh_cost = bool(fresh["state_count"] and fresh["cost_nonincrease_in_every_cell"] and fresh["strict_pooled_lp_or_integer_reduction"])
    hist_cost = hist["cost_nonincrease_in_every_cell"]
    cost = fresh_cost and hist_cost and native
    advance = valid and result["effective_references_complete"] and quality and result["finite_budget_gate_passed"] and not routed_losses and cost
    executed = _work(entries)
    result.update(decision="INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not result["effective_references_complete"] else
                  "DUAL_REUSE_FRESH_CONFIRMATION_ADVANCE" if advance else "DUAL_REUSE_FRESH_CONFIRMATION_HOLD",
        execution_checks_passed=valid, routed_quality_gate_passed=quality, work_gate_passed=cost, confirmation_gate_passed=advance,
        fresh_work_gate_passed=fresh_cost, historical_work_gate_passed=hist_cost, native_work_gate_passed=native,
        final_score_gate_passed=not routed_losses, routed_score_regressions=routed_losses, work_increases=increases,
        executed_selector_calls=count * 2, executed_native_advance_calls=count * 8,
        all_executed_trace_and_routed_work=executed, executed_dual_reuse_work=executed["executed_dual_reuse"],
        executed_search_work={key: sum(executed[label]["search_work"][key] for label in ("kernel_old", "kernel_new", "routed_old", "routed_new")) for key in COUNTERS})
    return result
