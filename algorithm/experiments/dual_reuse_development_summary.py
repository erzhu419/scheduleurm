"""Certified dual reuse against the frozen, independently rerun reserved router."""
from __future__ import annotations

from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.branch_reserved_development_summary import _identity, _key, audit_reserved_stage
from algorithm.experiments.certificate_routed_development_summary import audit_candidate
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _candidate_work
from algorithm.experiments.root_pruning_confirmation_summary import _budget_summary
from algorithm.experiments.root_pruning_development_summary import ROOT_COUNTERS, _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same

DUAL_COUNTERS = ("certificates_saved", "child_bound_evaluations", "child_bound_improvements",
                 "child_bound_prunes", "child_rhs_entries", "child_requirement_reads", "child_weight_reads")


def audit_dual_stage(stage, dimensions, previous=None):
    work = stage["dual_reuse"]
    errors = []
    if any(type(work[key]) is not int or work[key] < 0 for key in DUAL_COUNTERS):
        errors.append("dual-reuse counters must be nonnegative integers")
    if (work["certificates_saved"] != stage["selection"]["dual_certified_lp_calls"]
            or not work["child_bound_prunes"] <= work["child_bound_improvements"] <= work["child_bound_evaluations"]):
        errors.append("saved certificates or child bound outcomes differ from actual certified work")
    if (work["child_requirement_reads"] != dimensions * work["child_weight_reads"]
            or work["child_rhs_entries"] != (1 + dimensions) * work["child_bound_evaluations"] + work["child_weight_reads"]):
        errors.append("child RHS, requirement or weight reads differ from actual dual evaluations")
    if previous is not None and any(work[key] < previous["dual_reuse"][key] for key in DUAL_COUNTERS):
        errors.append("cumulative dual-reuse work decreased")
    return errors


def _dual_work(stages):
    return {key: sum(stage["dual_reuse"][key] for stage in stages) for key in DUAL_COUNTERS}


def _work(states):
    native = [state["traces"]["256"] for state in states]
    final = [state["routed"]["final_lp"] for state in states]
    return {"kernel_old_borrowed": _native_work([state["old_traces"]["256"] for state in states]),
        "kernel_new": {**_native_work(native), "dual_reuse": _dual_work(native)},
        "routed_old": _candidate_work([state["old_routed"] for state in states]),
        "routed_new": {**_candidate_work([state["routed"] for state in states]), "dual_reuse": _dual_work(final)},
        "routed_root_pruning": {label: {key: sum(state[field]["final_lp"]["root_pruning"][key] for state in states)
            for key in ROOT_COUNTERS} for label, field in (("old", "old_routed"), ("new", "routed"))},
        "executed_dual_reuse": _dual_work(native + final)}


def _cost(work):
    a, b = work["routed_old"], work["routed_new"]
    return (all(b["search_work"][key] <= a["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients"))
            and b["shared_integer_states"] <= a["shared_integer_states"])


def _native_cost(states):
    result = []
    for position in range(4):
        old = [state["old_traces"][str(state["caps"][position])]["selection"] for state in states]
        new = [state["traces"][str(state["caps"][position])]["selection"] for state in states]
        amounts = {key: {"old": sum(item[key] for item in old), "new": sum(item[key] for item in new)}
                   for key in ("lp_calls", "lp_matrix_coefficients")}
        result.append({"position": position, "states": len(states), **amounts,
                       "cost_nonincrease": all(value["new"] <= value["old"] for value in amounts.values())})
    return result


def _increases(state):
    work = _work([state])
    a, b = work["routed_old"], work["routed_new"]
    old = {key: a["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} | {"shared_integer_states": a["shared_integer_states"]}
    new = {key: b["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} | {"shared_integer_states": b["shared_integer_states"]}
    return {"routed": {key: {"old": old[key], "new": new[key]} for key in old if new[key] > old[key]},
            "native": [item for item in _native_cost([state]) if not item["cost_nonincrease"]]}


def summarize(protocol, rows):
    failures, nonclosures, losses, routed_losses, comparisons, entries, cells, increases, qualities = [], [], [], [], [], [], [], [], []
    complete = True
    models = {model["node"]: model for model in protocol["models"]}
    historical = {_key(item["node"], item["lanes"], item) for item in protocol["historical_regression_keys"]}
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    if len(historical) != len(protocol["historical_regression_keys"]) or not historical <= registered:
        failures.append({"errors": ["fixed historical keys must be unique and contained in the frozen roster"]})
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        failures.append({"errors": ["all registered cells must occur once in order"]})
    all_states = [state for row in rows for state in row["states"]]
    count = len(all_states)
    counts = {"expected_selector_calls": count * 2, "expected_native_advance_calls": count * 4,
              "expected_interface_calls": count * 6, "expected_native_constructions": count * 3}
    if any(protocol[key] != value for key, value in counts.items()):
        failures.append({"errors": ["executed coverage differs from the registered method and native interface counts"]})
    for cell, row in zip(protocol["cells"], rows):
        model = models[cell["node"]]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        point = {"node": row["node"], "lanes": row["lanes"]}
        errors = []
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options)):
            errors.append("cell registration or physical catalogue metadata differs")
        identity = lambda state: (_identity(state), state["source_state_id"], tuple(state["caps"]))
        if list(map(identity, row["states"])) != list(map(identity, cell["states"])):
            errors.append("state roster differs from the frozen development roster")
        if (row["new_reference_calls"] != 0 or row["new_reference_oracle_calls"] != 0
                or row["new_old_routed_calls"] != len(row["states"])):
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
            if not all(set(state[field]) == {str(cap) for cap in state["caps"]} for field in ("old_traces", "traces")):
                failures.append({**pointer, "errors": errors + ["all four registered snapshots are required"]})
                continue
            for label, trace, routed in (("old", state["old_traces"], state["old_routed"]), ("new", state["traces"], state["routed"])):
                errors.extend(label + ": " + message for message in _trace_errors(trace, state, options, row["lanes"], expected, reference, True))
                previous = None
                for cap in state["caps"]:
                    stage = trace[str(cap)]
                    if stage["thresholds"] != thresholds:
                        errors.append(label + " thresholds differ from physical catalogue strata")
                    errors.extend(f"{label} cap{cap}: " + message for message in audit_reserved_stage(stage, cap, roots))
                    if label == "new":
                        errors.extend(f"new cap{cap}: " + message for message in audit_dual_stage(stage, len(state["queue"]), previous))
                    previous = stage
                initial = trace["144"]
                _, extra, quality = audit_candidate(routed, {**state, "expected_initial_lp": initial}, protocol, model, row["lanes"], expected)
                errors.extend(label + " routed: " + message for message in extra)
                if not _same(routed["initial_lp"], initial) or not _same(routed["final_lp"], trace["256"] if routed["resumed"] else initial):
                    errors.append(label + " routed snapshots differ from their independent native ladder")
                if label == "old" and not _same(routed, state["expected_old_routed"]):
                    errors.append("old routed output differs from its frozen direct predecessor")
                for stage, cap in ((routed["initial_lp"], 144), (routed["final_lp"], 256 if routed["resumed"] else 144)):
                    errors.extend(label + " routed: " + message for message in audit_reserved_stage(stage, cap, roots))
                if label == "new":
                    errors.extend(audit_dual_stage(routed["initial_lp"], len(state["queue"])))
                    errors.extend(audit_dual_stage(routed["final_lp"], len(state["queue"]), routed["initial_lp"]))
                qualities.append(quality)
                if not quality:
                    nonclosures.append({**pointer, "label": label, "status": routed["status"], "reference_complete": expected is not None})
            for position, cap in enumerate(state["caps"]):
                a, b = (selection_from_record(state[field][str(cap)]["selection"]) for field in ("old_traces", "traces"))
                old_score, new_score = selection_score(a, tuple(state["queue"])), selection_score(b, tuple(state["queue"]))
                comparison = {**pointer, "budget_position": position, "lp_budget": cap, "old_lp_calls": a.lp_calls, "new_lp_calls": b.lp_calls,
                    "old_canonical_exact": a.canonical_exact, "new_canonical_exact": b.canonical_exact,
                    "old_score": str(old_score), "new_score": str(new_score), "score_regression": new_score < old_score}
                comparisons.append(comparison)
                if new_score < old_score:
                    losses.append(comparison)
            scores = [selection_score(selection_from_record(state[field]["selection"]), tuple(state["queue"])) for field in ("old_routed", "routed")]
            if scores[1] < scores[0]:
                routed_losses.append({**pointer, "old_score": str(scores[0]), "new_score": str(scores[1])})
            increase = _increases(state)
            if increase["routed"] or increase["native"]:
                increases.append({**pointer, **increase})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            cell_entries.append(state)
        regular = [state for state in cell_entries if state["source"] != "new_zero"]
        hist = [state for state in cell_entries if state["historical_regression"]]
        nonzero, historical_work = _work(regular), _work(hist)
        cells.append({**point, "nonzero_work": nonzero, "historical_work": historical_work,
            "nonzero_cost_nonincrease": _cost(nonzero), "historical_cost_nonincrease": _cost(historical_work),
            "nonzero_native_per_budget": _native_cost(regular), "historical_native_per_budget": _native_cost(hist)})
    regular = [state for state in entries if state["source"] != "new_zero"]
    hist = [state for state in entries if state["historical_regression"]]
    work, historical_work, executed = _work(regular), _work(hist), _work(entries)
    a, b = work["routed_old"], work["routed_new"]
    strict = b["search_work"]["lp_calls"] < a["search_work"]["lp_calls"] or b["shared_integer_states"] < a["shared_integer_states"]
    native_ok = all(item["cost_nonincrease"] for cell in cells for field in ("nonzero_native_per_budget", "historical_native_per_budget") for item in cell[field])
    nonzero_ok = bool(regular and strict and all(cell["nonzero_cost_nonincrease"] for cell in cells))
    hist_ok = all(cell["historical_cost_nonincrease"] for cell in cells)
    valid, quality, finite = not failures, bool(entries and len(qualities) == 2 * len(entries) and all(qualities)), not losses
    cost = nonzero_ok and hist_ok and native_ok
    advance = valid and complete and quality and finite and not routed_losses and cost
    return {"decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "DUAL_REUSE_DEVELOPMENT_ADVANCE" if advance else "DUAL_REUSE_DEVELOPMENT_HOLD",
        "state_count": count, "nonzero_state_count": len(regular), "historical_state_count": len(hist),
        "borrowed_reference_count": count, "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_reference_oracle_calls": sum(row["new_reference_oracle_calls"] for row in rows),
        "new_old_routed_control_calls": sum(row["new_old_routed_calls"] for row in rows),
        "executed_selector_calls": count * 2, "executed_native_advance_calls": sum(len(state["caps"]) for state in all_states),
        "executed_interface_calls": count * 6, "native_search_constructions": count * 3, "reference_native_search_constructions": 0,
        "effective_references_complete": complete, "execution_checks_passed": valid, "routed_quality_gate_passed": quality,
        "final_score_gate_passed": not routed_losses, "finite_budget_gate_passed": finite,
        "nonzero_work_gate_passed": nonzero_ok, "historical_work_gate_passed": hist_ok, "native_work_gate_passed": native_ok,
        "work_gate_passed": cost, "development_gate_passed": advance, "strict_pooled_lp_or_integer_reduction": bool(strict),
        "failures": failures, "routed_nonclosures": nonclosures, "finite_budget_score_regressions": losses,
        "routed_score_regressions": routed_losses, "work_increases": increases,
        "per_budget": _budget_summary(comparisons), "historical_per_budget": _budget_summary([item for item in comparisons if item["historical_regression"]]),
        "nonzero_work": work, "historical_work": historical_work, "all_executed_trace_and_routed_work": executed,
        "executed_search_work": {key: sum(executed[label]["search_work"][key] for label in ("kernel_new", "routed_old", "routed_new")) for key in COUNTERS},
        "executed_dual_reuse_work": executed["executed_dual_reuse"], "cells": cells,
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in all_states),
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
