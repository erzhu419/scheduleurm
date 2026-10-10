"""Exact path preservation and complete affine certificate arithmetic fees."""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.branch_reserved_development_summary import _key, audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS, _dual_work
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.residual_capacity_timing_summary import (
    GROUPS, _audit_result as _audit_residual_result, _work as _residual_work,
)
from algorithm.experiments.axis_capacity_frontier_development_summary import audit_axis_capacity_proof
from algorithm.experiments.frontier_score_proof_confirmation_summary import audit_frontier_confirmation_router
from algorithm.experiments.residual_capacity_frontier_development_summary import (
    _merged_values, _original_proof, audit_residual_completion,
)
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same

FEE_COUNTERS = DUAL_COUNTERS[4:]
CANONICAL_COUNTERS = ("canonical_key_bound_evaluations", "canonical_tie_pruned_regions")
ALLOWED_COUNTERS = frozenset((*FEE_COUNTERS, "affine_dual"))
AFFINE_COUNTERS = (
    "certificates_prepared", "certificate_numerator_products", "price_reads", "denominator_lcm_steps",
    "integer_scalings", "coefficient_requirement_reads", "coefficient_weight_reads",
    "coefficient_integer_operations", "bound_variable_visits", "box_coordinate_reads",
    "box_coordinate_comparisons", "changed_lower_coordinates", "changed_upper_coordinates",
    "delta_integer_operations", "bound_integer_products", "bound_rational_conversions",
    "delta_coefficient_reads",
)


def normalized_result(record):
    """Retain all search and canonical work; ignore only the changed arithmetic."""
    if isinstance(record, dict):
        return {key: normalized_result(value) for key, value in record.items() if key not in ALLOWED_COUNTERS}
    if isinstance(record, (list, tuple)):
        return [normalized_result(value) for value in record]
    return record


def _counter_schema(record, *, affine=False):
    errors = []
    if isinstance(record, dict):
        if "option_indices" in record:
            if any(key not in record or type(record[key]) is not int for key in CANONICAL_COUNTERS):
                raise ValueError("selection must retain both actual canonical counters")
            if any(record[key] < 0 for key in CANONICAL_COUNTERS):
                errors.append("canonical counters must be nonnegative")
        if "dual_reuse" in record:
            work = record["dual_reuse"]
            if set(work) != set(DUAL_COUNTERS) or any(type(value) is not int for value in work.values()):
                raise ValueError("dual reuse must retain all seven actual-work counters")
            if any(value < 0 for value in work.values()):
                errors.append("dual reuse counters must be nonnegative")
            if affine:
                actual = record.get("affine_dual", {})
                if set(actual) != set(AFFINE_COUNTERS) or any(type(value) is not int for value in actual.values()):
                    raise ValueError("affine dual must retain all seventeen actual-work counters")
                if any(value < 0 for value in actual.values()):
                    errors.append("affine dual counters must be nonnegative")
        for value in record.values():
            errors.extend(_counter_schema(value, affine=affine))
    elif isinstance(record, (list, tuple)):
        for value in record:
            errors.extend(_counter_schema(value, affine=affine))
    return errors


def audit_affine_stage(stage, dimensions, previous=None):
    """Recompute fees from actual certificate sizes, child scans and changes."""
    work, dual = stage["affine_dual"], stage["dual_reuse"]
    if (set(work) != set(AFFINE_COUNTERS) or set(dual) != set(DUAL_COUNTERS)
            or any(type(value) is not int or value < 0 for value in (*work.values(), *dual.values()))):
        return ["affine and dual-reuse counters must retain all actual nonnegative integer fields"]
    errors = []
    saved, prepared = dual["certificates_saved"], work["certificates_prepared"]
    variables, visits = work["coefficient_weight_reads"], work["bound_variable_visits"]
    evaluations = dual["child_bound_evaluations"]
    groups = stage["preprocessing"]["requirement_group_count"]
    changed = work["changed_lower_coordinates"] + work["changed_upper_coordinates"]
    if (saved != stage["selection"]["dual_certified_lp_calls"] or prepared > saved
            or not dual["child_bound_prunes"] <= dual["child_bound_improvements"] <= evaluations):
        errors.append("saved certificates or child outcomes differ from the actual certified work")
    if any(dual[name] for name in FEE_COUNTERS):
        errors.append("affine updates must not charge full child RHS, requirement or weight rebuilds")
    expected = {
        "certificate_numerator_products": saved,
        "price_reads": 2 * ((1 + dimensions) * prepared + variables),
        "denominator_lcm_steps": (2 + dimensions) * prepared + variables,
        "integer_scalings": (2 + dimensions) * prepared + 2 * variables,
        "coefficient_requirement_reads": dimensions * variables,
        "coefficient_integer_operations": (2 * dimensions + 3) * variables,
        "box_coordinate_reads": 4 * visits,
        "box_coordinate_comparisons": 2 * visits,
        "delta_coefficient_reads": changed,
        "delta_integer_operations": 3 * changed,
        "bound_integer_products": 2 * evaluations,
        "bound_rational_conversions": evaluations,
    }
    if any(work[name] != count for name, count in expected.items()):
        errors.append("affine preparation or child-update fees differ from their actual operation counts")
    if (not prepared <= variables <= groups * prepared or not evaluations <= visits <= groups * evaluations
            or evaluations and not prepared or work["changed_lower_coordinates"] > visits
            or work["changed_upper_coordinates"] > visits):
        errors.append("prepared variables, child scans or changed coordinates exceed their paid boxes")
    if previous is not None and (any(work[name] < previous["affine_dual"][name] for name in AFFINE_COUNTERS)
            or any(dual[name] < previous["dual_reuse"][name] for name in DUAL_COUNTERS)):
        errors.append("cumulative affine or dual-reuse work decreased")
    return errors


def _audit_affine_result(result, state, protocol, model, lanes, label, expected):
    """Audit the real router and retained tables with the new arithmetic fees."""
    if label != "candidate":
        return _audit_residual_result(result, state, protocol, model, lanes, label, expected)
    metric, errors, quality = audit_frontier_confirmation_router(result, state, protocol, model, lanes, "new", expected)
    roots = len(result["initial_lp"]["thresholds"])
    for name, budget in (("initial_lp", protocol["initial_lp_budget"]),
                         ("final_lp", protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"])):
        errors.extend(audit_reserved_stage(result[name], budget, roots))
    errors.extend(audit_affine_stage(result["initial_lp"], len(state["queue"])))
    errors.extend(audit_affine_stage(result["final_lp"], len(state["queue"]), result["initial_lp"]))
    prior = state["prior_routed"]["new"]
    if result["probe_stage"] == "frontier_late" and result["probe"] is not None and result["frontier_evidence"] is not None:
        probe = result["probe"]
        errors.extend(audit_axis_capacity_proof(probe, result["frontier_evidence"], state["queue"], lanes))
        if not _same(_original_proof(probe["score_proof"]), prior["probe"]["score_proof"]):
            errors.append("retained tables changed the frozen axis proof arithmetic")
        capacity = probe["score_proof"]["capacity"]
        if capacity.get("merged_values_copied") != len(capacity["bounds"]) * (lanes + 1):
            errors.append("retained merged vectors must charge every copied cell")
        for bound in capacity["bounds"]:
            if not _same(bound.get("exact_slot_numerators"), _merged_values(bound["axis_service_profiles"], state["queue"], lanes)):
                errors.append("retained vector differs from the already-paid max-plus merge")
        errors.extend(audit_residual_completion(probe, prior["probe"], model, state["queue"], lanes))
    return metric, errors, quality


def _affine_work(stages):
    return {name: sum(stage["affine_dual"][name] for stage in stages) for name in AFFINE_COUNTERS}


def _work(results, label):
    work = _residual_work(results, label)
    if label == "candidate":
        work["affine_dual"] = _affine_work([result["final_lp"] for result in results])
    return work


def _canonical_work(selections):
    return {key: sum(selection[key] for selection in selections) for key in CANONICAL_COUNTERS}


def _whole_work(states, field, label):
    results = [state[field] for state in states]
    work = _residual_work(results, label) if field == "previous_dual_routed" else _work(results, label)
    return {**work,
            "canonical_work": _canonical_work([result["final_lp"]["selection"] for result in results])}


def _kernel_work(states, previous=False, branch=False):
    traces = [state["previous_dual_traces"] if previous else state["traces"]["old" if branch else "new"] for state in states]
    stages = [trace[str(state["caps"][-1])] for state, trace in zip(states, traces)]
    return {**_native_work(stages),
        **({"dual_reuse": _dual_work(stages)} if not branch else {}),
        **({"affine_dual": _affine_work(stages)} if not previous and not branch else {}),
        "canonical_work": _canonical_work([stage["selection"] for stage in stages])}


def _savings(states):
    before, after = _whole_work(states, "previous_dual_routed", "candidate"), _whole_work(states, "routed", "candidate")
    return {"state_count": len(states), "before": before, "after": after,
        "old_child_rebuild_reads": {key: before["dual_reuse"][key] for key in FEE_COUNTERS},
        "affine_dual": after["affine_dual"]}


def _pointer(row, state):
    return {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "queue": state["queue"], "timing_group": state["timing_group"]}


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, mismatches, entries = [], [], [], []
    groups = defaultdict(list)
    complete = True
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    fixed = {group: {_key(item["node"], item["lanes"], item) for item in protocol[field]}
        for group, field in (("historical_regression", "historical_regression_keys"),
                            ("trigger_control", "trigger_control_keys"), ("new_trigger", "new_trigger_keys"))}
    old_targets = {_key(item["node"], item["lanes"], item) for item in protocol["previous_timing_failure_keys"]}
    targets = {_key(item["node"], item["lanes"], item) for item in protocol["affine_focus_keys"]}
    if (len(old_targets) != 4 or len(protocol["previous_timing_failure_keys"]) != 4
            or len(targets) != 6 or len(protocol["affine_focus_keys"]) != 6
            or not old_targets <= fixed["historical_regression"] or not old_targets <= targets
            or not targets <= registered
            or any(not keys <= registered for keys in fixed.values())
            or any(_key(item["node"], item["lanes"], item) not in registered for item in protocol["ordinary_tail_keys"])):
        failures.append({"errors": ["frozen controls and six affine focus states must belong to the complete original roster"]})
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        if (len(states) != len(cell["states"]) or not states or any(not _same(
                {key: state.get(key) for key in original}, original) for state, original in zip(states, cell["states"]))):
            raise ValueError("development states and source pointers must retain the complete frozen timing roster")
        errors, point = [], {"node": row["node"], "lanes": row["lanes"]}
        count = len(states)
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options)):
            errors.append("cell registration or physical catalogue metadata differs")
        if (row["new_reference_calls"] or row["new_oracle_calls"] or row["candidate_calls"] != count
                or row["native_advance_calls"] != 4 * count or row["native_search_constructions"] != 2 * count):
            errors.append("every state requires one cold candidate and four native advances with borrowed references")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"] for key in ("runtime_start", "runtime_end")):
            errors.append("runtime affinity or GC differs from registration")
        if errors:
            failures.append({**point, "errors": errors})
        minima = {}
        for option in options:
            key = tuple(option.requirements)
            minima[key] = min(minima.get(key, float("inf")), max(1., option.duration))
        thresholds, roots = sorted(set(minima.values())), len(set(minima.values()))
        for state in states:
            pointer, errors = _pointer(row, state), []
            key = _key(row["node"], row["lanes"], state)
            group = next((name for name in ("new_trigger", "trigger_control", "historical_regression") if key in fixed[name]),
                         "zero_queue" if not any(state["queue"]) else "ordinary_development")
            if state["timing_group"] != group or state["historical_regression"] != (key in fixed["historical_regression"]):
                errors.append("development grouping differs from the independently frozen timing keys")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("native ladder differs from the physical strata and frozen LP caps")
            reference = selection_from_record(state["reference"])
            expected = None
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            if (state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]
                    or any(state["reference_metrics"]["search_work"][name] != state["reference"][name] for name in COUNTERS)):
                errors.append("borrowed independent reference has invalid budget or paid certificate")
            errors.extend("reference: " + name for name, failed in check_selection(reference, options, tuple(state["queue"]),
                row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            trace, previous_trace = state["traces"]["new"], state["previous_dual_traces"]
            if any(set(item) != {str(cap) for cap in state["caps"]} for item in (trace, previous_trace, state["traces"]["old"])):
                raise ValueError("all four current and borrowed native snapshots are required")
            schema_errors = (_counter_schema(trace, affine=True) + _counter_schema(previous_trace)
                + _counter_schema(state["routed"], affine=True) + _counter_schema(state["previous_dual_routed"]))
            errors.extend(schema_errors)
            errors.extend(_trace_errors(trace, state, options, row["lanes"], expected, reference, True))
            previous = None
            for cap in state["caps"]:
                current, saved = trace[str(cap)], previous_trace[str(cap)]
                if current["thresholds"] != thresholds:
                    errors.append(f"cap{cap}: thresholds differ from the physical catalogue")
                errors.extend(f"cap{cap}: " + message for message in audit_reserved_stage(current, cap, roots))
                if set(current.get("dual_reuse", {})) == set(DUAL_COUNTERS):
                    errors.extend(f"cap{cap}: " + message for message in audit_affine_stage(current, len(state["queue"]), previous))
                if not _same(normalized_result(current), normalized_result(saved)):
                    mismatches.append({**pointer, "phase": "native", "lp_budget": cap})
                previous = current
            current, saved = state["routed"], state["previous_dual_routed"]
            if not _same(normalized_result(current), normalized_result(saved)):
                mismatches.append({**pointer, "phase": "whole_selector"})
            if not schema_errors:
                _, extra, quality = _audit_affine_result(current, state, protocol, model, row["lanes"], "candidate", expected)
                errors.extend("candidate: " + message for message in extra)
                if not quality:
                    nonclosures.append({**pointer, "status": current["status"]})
            else:
                nonclosures.append({**pointer, "status": "INVALID_COUNTER_SCHEMA"})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append((row, state))
            groups[group].append(state)
    states = [state for _, state in entries]
    count = len(states)
    expected_counts = {"expected_state_count": count, "expected_selector_calls": count,
        "expected_native_advance_calls": 4 * count, "expected_native_constructions": 2 * count,
        "expected_interface_calls": 5 * count}
    if any(protocol[name] != actual for name, actual in expected_counts.items()):
        failures.append({"errors": ["actual cold selector and native interface counts differ from registration"]})
    if protocol["expected_group_counts"] != {group: len(groups[group]) for group in GROUPS}:
        failures.append({"errors": ["every original timing group must retain its full frozen size"]})
    before = {"whole_selector": _whole_work(states, "previous_dual_routed", "candidate"), "native": _kernel_work(states, previous=True)}
    after = {"whole_selector": _whole_work(states, "routed", "candidate"), "native": _kernel_work(states)}
    target_results = []
    for row, state in entries:
        if _key(row["node"], row["lanes"], state) in targets:
            old, new = state["previous_dual_routed"]["final_lp"], state["routed"]["final_lp"]
            target_results.append({**_pointer(row, state), "old_dual_reuse": old["dual_reuse"],
                "new_dual_reuse": new["dual_reuse"], "affine_dual": new["affine_dual"],
                "affine_update_exercised": (new["affine_dual"]["certificates_prepared"] > 0
                    and new["affine_dual"]["bound_rational_conversions"] > 0)})
    target_gate = len(target_results) == 6 and all(item["affine_update_exercised"] for item in target_results)
    valid, quality, preserved = not failures, not nonclosures, not mismatches
    advance = bool(valid and complete and quality and preserved and target_gate)
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "AFFINE_DUAL_DEVELOPMENT_ADVANCE" if advance else "AFFINE_DUAL_DEVELOPMENT_HOLD",
        "state_count": count, "execution_checks_passed": valid, "references_complete": bool(complete),
        "quality_gate_passed": bool(valid and quality), "behavior_preserved": preserved,
        "affine_work_validated": valid, "affine_updates_exercised": target_gate,
        "work_gate_passed": bool(valid and target_gate),
        "development_gate_passed": advance,
        "failures": failures, "nonclosures": nonclosures, "behavior_mismatches": mismatches,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_oracle_calls": sum(row["new_oracle_calls"] for row in rows), "borrowed_reference_count": count,
        "executed_selector_calls": sum(row["candidate_calls"] for row in rows),
        "executed_native_advance_calls": sum(row["native_advance_calls"] for row in rows),
        "native_search_constructions": sum(row["native_search_constructions"] for row in rows),
        "executed_interface_calls": sum(row["candidate_calls"] + row["native_advance_calls"] for row in rows),
        "before_work": before, "after_work": after,
        "borrowed_branch_work": {"whole_selector": _whole_work(states, "old_routed", "branch"), "native": _kernel_work(states, branch=True)},
        "executed_search_work": {name: after["native"]["search_work"][name] + after["whole_selector"]["search_work"][name] for name in COUNTERS},
        "executed_dual_reuse_work": {name: after["native"]["dual_reuse"][name] + after["whole_selector"]["dual_reuse"][name] for name in DUAL_COUNTERS},
        "executed_affine_dual_work": {name: after["native"]["affine_dual"][name]
            + after["whole_selector"]["affine_dual"][name] for name in AFFINE_COUNTERS},
        "fees_by_group": {group: _savings(groups[group]) for group in GROUPS},
        "affine_focus_states": target_results,
        "fee_interpretation": "Old full-RHS reads and new affine preparation/update operations are different units; no arithmetic-counter reduction is a wall-time result.",
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in states),
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
