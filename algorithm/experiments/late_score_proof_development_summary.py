"""Shared-budget late score proof with unchanged LP snapshots and ordinary routes."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.branch_reserved_development_summary import _identity, _key, audit_reserved_stage
from algorithm.experiments.certificate_routed_development_summary import audit_candidate
from algorithm.experiments.dual_reuse_development_summary import _cost, _dual_work, audit_dual_stage
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _candidate_work, _metric, _exact
from algorithm.experiments.root_pruning_development_summary import _native_work
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _audit_result as _audit_strict, _same


def audit_late_router(result, original, state, protocol, model, lanes, label, expected):
    trace = state["traces"][label]
    initial, final = original["initial_lp"], original["final_lp"]
    errors = []
    try:
        _, extra, _ = audit_candidate(original, {**state, "expected_initial_lp": trace["144"]}, protocol, model, lanes, expected)
        errors.extend("original: " + message for message in extra)
    except ValueError as error:
        errors.append("original: " + str(error))
    if not _same(initial, trace["144"]) or not _same(final, trace["256"] if original["resumed"] else trace["144"]):
        errors.append("original routed LP snapshots differ from the borrowed native evidence")
    if any(result[key] != protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")):
        errors.append("late router budgets differ from registration")
    if not _same(result["initial_lp"], initial) or not _same(result["final_lp"], final) or result["resumed"] != original["resumed"]:
        errors.append("late routing changed the original paid LP snapshots or resumption")
    late = not final["selection"]["score_exact"] and original["probe"] is None
    if not late:
        if not _same(result, original):
            errors.append("ordinary route must reproduce the complete frozen original output without a second probe")
    else:
        probe = result["probe"]
        if result["probe_stage"] != "late" or probe is None:
            errors.append("unproved final score without an earlier probe requires exactly one late probe")
        if probe is not None:
            try:
                _, extra, _ = _audit_strict(probe, state, {**protocol, "lp_budgets": {"candidate": protocol["lp_node_budget"]}},
                                           model, lanes, "candidate", expected)
                errors.extend("late probe: " + message for message in extra)
            except ValueError as error:
                errors.append("late probe: " + str(error))
            if (not _same(probe["baseline_selection"], final["selection"])
                    or not _same(probe["preprocessing"], final["preprocessing"])
                    or not _same(probe["capacity_propagation"], final["capacity_propagation"])):
                errors.append("late probe must reuse the paid final LP selection, preprocessing and capacity work")
            if not probe["score_proof"]["attempted"]:
                errors.append("late probe must attempt strict score proof on its unproved final LP snapshot")
            status = "LATE_PROBE_COMPLETE" if probe["selection"]["canonical_exact"] else "LATE_PROBE_INCOMPLETE"
            if result["status"] != status or not _same(result["selection"], probe["selection"]):
                errors.append("late returned selection or status differs from its actual integer certificate")
    budget = protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, result["final_lp"]["preprocessing"])
    if metric["failures"]:
        errors.append("returned routed action has invalid feasibility, certificate or paid work")
    return metric, errors, _exact(result["selection"], metric, expected)


def _work(states):
    result = {label: _candidate_work([state[field] for state in states])
              for label, field in (("routed_old", "old_routed"), ("routed_new", "routed"))}
    result["routed_new"]["dual_reuse"] = _dual_work([state["routed"]["final_lp"] for state in states])
    result["probe_stage_counts"] = {label: dict(Counter(state[field]["probe_stage"] for state in states))
                                    for label, field in (("old", "old_routed"), ("new", "routed"))}
    return result


def _increases(state):
    work = _work([state])
    a, b = work["routed_old"], work["routed_new"]
    old = {key: a["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} | {"shared_integer_states": a["shared_integer_states"]}
    new = {key: b["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} | {"shared_integer_states": b["shared_integer_states"]}
    return {key: {"old": old[key], "new": new[key]} for key in old if new[key] > old[key]}


def summarize(protocol, rows):
    failures, nonclosures, losses, increases, entries, cells, counterexamples, qualities = [], [], [], [], [], [], [], []
    complete = True
    models = {model["node"]: model for model in protocol["models"]}
    all_states = [state for row in rows for state in row["states"]]
    count = len(all_states)
    counts = {"expected_state_count": count, "expected_selector_calls": count * 2, "expected_interface_calls": count * 2,
              "expected_native_constructions": count * 2, "expected_native_advance_calls": 0,
              "expected_borrowed_native_snapshots": count * 8,
              "expected_nonzero_state_count": sum(any(state["queue"]) for state in all_states),
              "expected_failure_development_count": sum(state["cohort"] == "failure_development" for state in all_states)}
    if any(protocol[key] != value for key, value in counts.items()):
        failures.append({"errors": ["executed roster or interface counts differ from registration"]})
    historical = {_key(item["node"], item["lanes"], item) for item in protocol["historical_regression_keys"]}
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    if len(historical) != len(protocol["historical_regression_keys"]) or not historical <= registered:
        failures.append({"errors": ["fixed historical keys must be unique and contained in the frozen roster"]})
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        failures.append({"errors": ["all registered cells must occur once in order"]})
    for cell, row in zip(protocol["cells"], rows):
        model = models[cell["node"]]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        point = {"node": row["node"], "lanes": row["lanes"]}
        identity = lambda state: (_identity(state), tuple(state["caps"]), tuple(state[key] for key in (
            "borrowed_from", "borrowed_index", "source_state_id", "source_record_source")))
        errors = []
        if list(map(identity, row["states"])) != list(map(identity, cell["states"])):
            errors.append("state roster or borrowed source pointers differ from registration")
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options)):
            errors.append("cell registration or catalogue metadata differs")
        if row["new_reference_calls"] != 0 or row["new_reference_oracle_calls"] != 0 or row["new_old_routed_calls"] != len(row["states"]):
            errors.append("references must be borrowed and both routed arms independently executed")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"] for key in ("runtime_start", "runtime_end")):
            errors.append("runtime affinity or GC differs from registration")
        if errors:
            failures.append({**point, "errors": errors})
        minima = {}
        for option in options:
            key = tuple(option.requirements)
            minima[key] = min(minima.get(key, float("inf")), max(1.0, option.duration))
        roots = len(set(minima.values()))
        cell_entries = []
        for state in row["states"]:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "source", "cohort", "historical_regression")}}
            errors, expected = [], None
            if state["historical_regression"] != (_key(row["node"], row["lanes"], state) in historical):
                errors.append("historical subgroup differs from its independently frozen keys")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("borrowed budget ladder differs from the catalogue and registered budgets")
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
                failures.append({**pointer, "errors": errors + ["all eight borrowed native snapshots are required"]})
                continue
            state_quality = {}
            for label, field in (("old", "old_routed"), ("new", "routed")):
                routed = state[field]
                _, extra, quality = audit_late_router(routed, state["original_routed"][label], state, protocol, model, row["lanes"], label, expected)
                errors.extend(label + ": " + message for message in extra)
                for name, cap in (("initial_lp", 144), ("final_lp", 256 if routed["resumed"] else 144)):
                    errors.extend(label + " " + name + ": " + message for message in audit_reserved_stage(routed[name], cap, roots))
                if label == "new":
                    errors.extend(audit_dual_stage(routed["initial_lp"], len(state["queue"])))
                    errors.extend(audit_dual_stage(routed["final_lp"], len(state["queue"]), routed["initial_lp"]))
                state_quality[label] = quality
                qualities.append(quality)
                if not quality:
                    nonclosures.append({**pointer, "label": label, "status": routed["status"], "reference_complete": expected is not None})
            scores = [selection_score(selection_from_record(state[field]["selection"]), tuple(state["queue"])) for field in ("old_routed", "routed")]
            if scores[1] < scores[0]:
                losses.append({**pointer, "old_score": str(scores[0]), "new_score": str(scores[1])})
            increase = _increases(state)
            if increase:
                increases.append({**pointer, "increases": increase})
            if state["cohort"] == "failure_development":
                counterexamples.append({**pointer, "quality": state_quality,
                    "original_status": {label: state["original_routed"][label]["status"] for label in ("old", "new")},
                    "status": {label: state[field]["status"] for label, field in (("old", "old_routed"), ("new", "routed"))}, "work": _work([state])})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            cell_entries.append(state)
        nonzero = _work([state for state in cell_entries if any(state["queue"])])
        hist = _work([state for state in cell_entries if state["historical_regression"]])
        cells.append({**point, "nonzero_work": nonzero, "historical_work": hist,
                      "nonzero_cost_nonincrease": _cost(nonzero), "historical_cost_nonincrease": _cost(hist)})
    regular = [state for state in entries if any(state["queue"])]
    hist = [state for state in entries if state["historical_regression"]]
    work, historical_work, executed = _work(regular), _work(hist), _work(entries)
    a, b = work["routed_old"], work["routed_new"]
    strict = b["search_work"]["lp_calls"] < a["search_work"]["lp_calls"] or b["shared_integer_states"] < a["shared_integer_states"]
    nonzero_cost = bool(regular and strict and all(cell["nonzero_cost_nonincrease"] for cell in cells))
    hist_cost = all(cell["historical_cost_nonincrease"] for cell in cells)
    valid, quality = not failures, bool(entries and len(qualities) == 2 * len(entries) and all(qualities))
    advance = valid and complete and quality and not losses and nonzero_cost and hist_cost
    borrowed_native = {label: _native_work([state["traces"][label]["256"] for state in entries]) for label in ("old", "new")}
    borrowed_native["new"]["dual_reuse"] = _dual_work([state["traces"]["new"]["256"] for state in entries])
    return {"decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "LATE_SCORE_PROOF_DEVELOPMENT_ADVANCE" if advance else "LATE_SCORE_PROOF_DEVELOPMENT_HOLD",
        "state_count": count, "nonzero_state_count": len(regular), "historical_state_count": len(hist), "failure_development_count": len(counterexamples),
        "borrowed_reference_count": count, "borrowed_native_snapshots": count * 8, "borrowed_original_routed_count": count * 2,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows), "new_reference_oracle_calls": sum(row["new_reference_oracle_calls"] for row in rows),
        "new_old_routed_control_calls": sum(row["new_old_routed_calls"] for row in rows), "executed_selector_calls": count * 2,
        "executed_interface_calls": count * 2, "executed_native_advance_calls": 0, "native_search_constructions": count * 2,
        "reference_native_search_constructions": 0, "effective_references_complete": complete, "execution_checks_passed": valid,
        "routed_quality_gate_passed": quality, "final_score_gate_passed": not losses, "nonzero_work_gate_passed": nonzero_cost,
        "historical_work_gate_passed": hist_cost, "strict_pooled_lp_or_integer_reduction": bool(strict),
        "work_gate_passed": nonzero_cost and hist_cost, "development_gate_passed": advance,
        "failures": failures, "routed_nonclosures": nonclosures, "routed_score_regressions": losses, "work_increases": increases,
        "nonzero_work": work, "historical_work": historical_work, "all_executed_routed_work": executed,
        "executed_search_work": {key: sum(executed[label]["search_work"][key] for label in ("routed_old", "routed_new")) for key in COUNTERS},
        "executed_dual_reuse_work": executed["routed_new"]["dual_reuse"], "counterexample_states": counterexamples,
        "counterexample_work": _work([state for state in entries if state["cohort"] == "failure_development"]),
        "borrowed_native_work": borrowed_native, "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in all_states),
        "cells": cells, "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
