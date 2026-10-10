"""Reserved branch-budget development with a separate fixed historical cost gate."""
from __future__ import annotations

from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.certificate_routed_development_summary import audit_candidate
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.root_pruning_confirmation_summary import _work, _cost, _budget_summary
from algorithm.experiments.root_pruning_development_summary import audit_root_stage, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same


def audit_reserved_stage(stage, cap, roots):
    errors = audit_root_stage(stage)
    selected, work = stage["selection"], stage["root_pruning"]
    branch_lp = selected["lp_calls"] - work["roots_solved"]
    if work["roots_seen"] != roots:
        errors.append("reserved root count differs from the physical catalogue")
    if selected["lp_calls"] + work["roots_seen"] - work["roots_solved"] > cap:
        errors.append("actual LP calls plus skipped-root slots exceed the cap")
    if branch_lp > cap - roots:
        errors.append("branch LP calls exceed cap minus reserved roots")
    if cap == roots and (selected["branch_splits"] != 0 or branch_lp != 0):
        errors.append("minimum root cap must not split or solve a branch")
    return errors


def _identity(state):
    return tuple(state.get(key) for key in ("state_id", "source", "cohort", "source_index", "panel", "seed", "prior_index", "historical_regression")), tuple(state["queue"])


def _key(node, lanes, state):
    return node, lanes, tuple(state["queue"])


def _increases(state):
    old, new = _work([state])["routed_old"], _work([state])["routed_new"]
    a = {key: old["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} | {"shared_integer_states": old["shared_integer_states"]}
    b = {key: new["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} | {"shared_integer_states": new["shared_integer_states"]}
    return {key: {"old": a[key], "new": b[key], "increase": b[key] - a[key]} for key in a if b[key] > a[key]}


def summarize(protocol, rows):
    failures, nonclosures, losses, comparisons, entries, cells, work_increases, qualities = [], [], [], [], [], [], [], []
    complete = True
    models = {model["node"]: model for model in protocol["models"]}
    historical = {_key(item["node"], item["lanes"], item) for item in protocol["historical_regression_keys"]}
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    if (len(historical) != len(protocol["historical_regression_keys"]) or not historical <= registered):
        failures.append({"errors": ["fixed historical regression keys must be unique and contained in the frozen roster"]})
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        failures.append({"errors": ["all registered cells must occur once in order"]})
    for cell, row in zip(protocol["cells"], rows):
        model = models[cell["node"]]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        point = {"node": row["node"], "lanes": row["lanes"]}
        errors = []
        if (row["registered_at_utc"] != protocol["registered_at_utc"]
                or row["physical_lanes"] != model["lanes"] or row["option_count"] != len(options)):
            errors.append("cell registration or physical catalogue metadata differs")
        if list(map(_identity, row["states"])) != list(map(_identity, cell["states"])):
            errors.append("state roster differs from the frozen development roster")
        if row["new_reference_calls"] != 0 or row["new_old_routed_calls"] != len(row["states"]):
            errors.append("references must be borrowed and every old routed control independently executed")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"] for key in ("runtime_start", "runtime_end")):
            errors.append("runtime affinity or GC differs from registration")
        if errors:
            failures.append({**point, "errors": errors})
        minima = {}
        for option in options:
            key = tuple(option.requirements)
            minima[key] = min(minima.get(key, float("inf")), max(1.0, option.duration))
        thresholds, roots = sorted(set(minima.values())), len(set(minima.values()))
        cell_entries = []
        for state in row["states"]:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "source", "cohort", "historical_regression")}}
            errors, expected = [], None
            if state["historical_regression"] != (_key(row["node"], row["lanes"], state) in historical):
                errors.append("historical subgroup flag differs from its independently frozen keys")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("budget ladder differs from catalogue roots and registered caps")
            reference = selection_from_record(state["reference"])
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            metric = state["reference_metrics"]
            if (state["reference"]["lp_node_budget"] != protocol["reference_budget"] or metric["failures"]
                    or metric["reference_comparable_count"] != int(state["reference_complete"])
                    or any(metric["search_work"][key] != state["reference"][key] for key in COUNTERS)):
                errors.append("borrowed reference has invalid budget or recomputed metrics")
            checks = check_selection(reference, options, tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected)
            errors.extend("reference: " + key for key, failed in checks.items() if failed)
            complete &= expected is not None
            if not all(set(state["traces"][label]) == {str(cap) for cap in state["caps"]} for label in ("old", "new")):
                failures.append({**pointer, "errors": errors + ["all four registered snapshots are required"]})
                continue
            for label in ("old", "new"):
                trace = state["traces"][label]
                if any(stage["thresholds"] != thresholds for stage in trace.values()):
                    errors.append(label + " thresholds differ from physical catalogue strata")
                errors.extend(label + ": " + message for message in _trace_errors(trace, state, options, row["lanes"], expected, reference, label == "new"))
                if label == "new":
                    for cap in state["caps"]:
                        errors.extend(f"cap{cap}: " + message for message in audit_reserved_stage(trace[str(cap)], cap, roots))
                routed = state["old_routed"] if label == "old" else state["routed"]
                initial = trace["144"]
                audit_state = {**state, "expected_initial_lp": {key: initial[key] for key in ("selection", "preprocessing", "capacity_propagation")}}
                _, extra, quality = audit_candidate(routed, audit_state, protocol, model, row["lanes"], expected)
                errors.extend(label + " routed: " + message for message in extra)
                if not _same(routed["initial_lp"], initial):
                    errors.append(label + " routed initial differs from the independent cap144 trace")
                if routed["resumed"] and not _same(routed["final_lp"], trace["256"]):
                    errors.append(label + " routed continuation differs from the independent cap256 trace")
                if label == "new":
                    for stage, cap in ((routed["initial_lp"], 144), (routed["final_lp"], 256 if routed["resumed"] else 144)):
                        errors.extend(audit_reserved_stage(stage, cap, roots))
                    qualities.append(quality)
                if not quality:
                    nonclosures.append({**pointer, "label": label, "status": routed["status"], "reference_complete": expected is not None})
            for position, cap in enumerate(state["caps"]):
                a, b = (selection_from_record(state["traces"][label][str(cap)]["selection"]) for label in ("old", "new"))
                old_score, new_score = selection_score(a, tuple(state["queue"])), selection_score(b, tuple(state["queue"]))
                comparison = {**pointer, "budget_position": position, "lp_budget": cap, "old_lp_calls": a.lp_calls, "new_lp_calls": b.lp_calls,
                    "old_canonical_exact": a.canonical_exact, "new_canonical_exact": b.canonical_exact,
                    "old_score": str(old_score), "new_score": str(new_score), "score_regression": new_score < old_score}
                comparisons.append(comparison)
                if new_score < old_score:
                    losses.append(comparison)
            increase = _increases(state)
            if increase:
                work_increases.append({**pointer, "increases": increase})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            cell_entries.append(state)
        nonzero = _work([state for state in cell_entries if state["source"] != "new_zero"])
        hist = _work([state for state in cell_entries if state["historical_regression"]])
        cells.append({**point, "nonzero_work": nonzero, "historical_work": hist,
                      "nonzero_cost_nonincrease": _cost(nonzero), "historical_cost_nonincrease": _cost(hist)})
    nonzero_states = [state for state in entries if state["source"] != "new_zero"]
    historical_states = [state for state in entries if state["historical_regression"]]
    nonzero, hist = _work(nonzero_states), _work(historical_states)
    old_lp, new_lp = (nonzero[label]["search_work"]["lp_calls"] for label in ("routed_old", "routed_new"))
    valid, quality, finite = not failures, bool(entries and len(qualities) == len(entries) and all(qualities)), not losses
    nonzero_cost = bool(nonzero_states and new_lp < old_lp and all(cell["nonzero_cost_nonincrease"] for cell in cells))
    hist_cost = all(cell["historical_cost_nonincrease"] for cell in cells)
    advance = valid and complete and quality and finite and nonzero_cost and hist_cost
    all_states = [state for row in rows for state in row["states"]]
    return {"decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "BRANCH_RESERVED_DEVELOPMENT_ADVANCE" if advance else "BRANCH_RESERVED_DEVELOPMENT_HOLD",
        "state_count": len(all_states), "nonzero_state_count": len(nonzero_states), "historical_state_count": len(historical_states),
        "borrowed_reference_count": len(all_states), "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_reference_oracle_calls": 0, "new_old_routed_control_calls": sum(row["new_old_routed_calls"] for row in rows),
        "executed_interface_calls": len(all_states) * 10, "native_search_constructions": len(all_states) * 4, "reference_native_search_constructions": 0,
        "effective_references_complete": complete, "execution_checks_passed": valid, "routed_quality_gate_passed": quality,
        "nonzero_work_gate_passed": nonzero_cost, "historical_work_gate_passed": hist_cost,
        "work_gate_passed": nonzero_cost and hist_cost, "finite_budget_gate_passed": finite, "development_gate_passed": advance,
        "failures": failures, "routed_nonclosures": nonclosures, "finite_budget_score_regressions": losses, "work_increases": work_increases,
        "per_budget": _budget_summary(comparisons), "historical_per_budget": _budget_summary([item for item in comparisons if item["historical_regression"]]),
        "nonzero_work": nonzero, "historical_work": hist, "all_executed_trace_and_routed_work": _work(entries), "cells": cells,
        "nonzero_routed_lp_ratio": new_lp / old_lp if old_lp else None,
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in all_states),
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
