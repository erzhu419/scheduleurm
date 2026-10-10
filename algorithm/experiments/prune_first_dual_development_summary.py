"""Frozen-path development audit for delaying dual work until cheap pruning."""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.branch_reserved_development_summary import _key, audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS, _dual_work, audit_dual_stage
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.residual_capacity_timing_summary import GROUPS, _audit_result, _work
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same

CHILD_COUNTERS = DUAL_COUNTERS[1:]
CANONICAL_COUNTERS = ("canonical_key_bound_evaluations", "canonical_tie_pruned_regions")
ALLOWED_COUNTERS = frozenset((*CHILD_COUNTERS, *CANONICAL_COUNTERS))


def normalized_result(record):
    """Ignore only the counters whose placement changes in this experiment."""
    if isinstance(record, dict):
        return {key: normalized_result(value) for key, value in record.items() if key not in ALLOWED_COUNTERS}
    if isinstance(record, (list, tuple)):
        return [normalized_result(value) for value in record]
    return record


def _counter_schema(record):
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
        for value in record.values():
            errors.extend(_counter_schema(value))
    elif isinstance(record, (list, tuple)):
        for value in record:
            errors.extend(_counter_schema(value))
    return errors


def _canonical_work(selections):
    return {key: sum(selection[key] for selection in selections) for key in CANONICAL_COUNTERS}


def _whole_work(states, field, label):
    results = [state[field] for state in states]
    return {**_work(results, label),
            "canonical_work": _canonical_work([result["final_lp"]["selection"] for result in results])}


def _kernel_work(states, previous=False, branch=False):
    traces = [state["previous_dual_traces"] if previous else state["traces"]["old" if branch else "new"] for state in states]
    stages = [trace[str(state["caps"][-1])] for state, trace in zip(states, traces)]
    return {**_native_work(stages),
        **({"dual_reuse": _dual_work(stages)} if not branch else {}),
        "canonical_work": _canonical_work([stage["selection"] for stage in stages])}


def _savings(states):
    before, after = _whole_work(states, "previous_dual_routed", "candidate"), _whole_work(states, "routed", "candidate")
    return {"state_count": len(states), "before": before, "after": after,
        "child_work_saved": {key: before["dual_reuse"][key] - after["dual_reuse"][key] for key in CHILD_COUNTERS}}


def _pointer(row, state):
    return {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "queue": state["queue"], "timing_group": state["timing_group"]}


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, mismatches, regressions, entries = [], [], [], [], []
    groups = defaultdict(list)
    complete = True
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    fixed = {group: {_key(item["node"], item["lanes"], item) for item in protocol[field]}
        for group, field in (("historical_regression", "historical_regression_keys"),
                            ("trigger_control", "trigger_control_keys"), ("new_trigger", "new_trigger_keys"))}
    targets = {_key(item["node"], item["lanes"], item) for item in protocol["previous_timing_failure_keys"]}
    if (len(targets) != 4 or len(protocol["previous_timing_failure_keys"]) != 4
            or not targets <= fixed["historical_regression"] or not targets <= registered
            or any(not keys <= registered for keys in fixed.values())
            or any(_key(item["node"], item["lanes"], item) not in registered for item in protocol["ordinary_tail_keys"])):
        failures.append({"errors": ["frozen controls and four timing failures must belong to the complete original roster"]})
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
            schema_errors = _counter_schema(trace) + _counter_schema(previous_trace) + _counter_schema(state["routed"]) + _counter_schema(state["previous_dual_routed"])
            errors.extend(schema_errors)
            errors.extend(_trace_errors(trace, state, options, row["lanes"], expected, reference, True))
            previous = None
            for cap in state["caps"]:
                current, saved = trace[str(cap)], previous_trace[str(cap)]
                if current["thresholds"] != thresholds:
                    errors.append(f"cap{cap}: thresholds differ from the physical catalogue")
                errors.extend(f"cap{cap}: " + message for message in audit_reserved_stage(current, cap, roots))
                if set(current.get("dual_reuse", {})) == set(DUAL_COUNTERS):
                    errors.extend(f"cap{cap}: " + message for message in audit_dual_stage(current, len(state["queue"]), previous))
                if not _same(normalized_result(current), normalized_result(saved)):
                    mismatches.append({**pointer, "phase": "native", "lp_budget": cap})
                growth = {name: {"old": saved["dual_reuse"][name], "new": current["dual_reuse"][name]}
                    for name in CHILD_COUNTERS if name in current["dual_reuse"] and current["dual_reuse"][name] > saved["dual_reuse"][name]}
                if growth:
                    regressions.append({**pointer, "phase": "native", "lp_budget": cap, "increases": growth})
                previous = current
            current, saved = state["routed"], state["previous_dual_routed"]
            if not _same(normalized_result(current), normalized_result(saved)):
                mismatches.append({**pointer, "phase": "whole_selector"})
            growth = {name: {"old": saved["final_lp"]["dual_reuse"][name], "new": current["final_lp"]["dual_reuse"][name]}
                for name in CHILD_COUNTERS if name in current["final_lp"]["dual_reuse"]
                and current["final_lp"]["dual_reuse"][name] > saved["final_lp"]["dual_reuse"][name]}
            if growth:
                regressions.append({**pointer, "phase": "whole_selector", "increases": growth})
            if not schema_errors:
                _, extra, quality = _audit_result(current, state, protocol, model, row["lanes"], "candidate", expected)
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
    strict = all(after["whole_selector"]["dual_reuse"][name] < before["whole_selector"]["dual_reuse"][name]
                 for name in ("child_bound_evaluations", "child_rhs_entries"))
    target_results = []
    for row, state in entries:
        if _key(row["node"], row["lanes"], state) in targets:
            old, new = state["previous_dual_routed"]["final_lp"]["dual_reuse"], state["routed"]["final_lp"]["dual_reuse"]
            target_results.append({**_pointer(row, state), "before": old, "after": new,
                "child_evaluations_strictly_lower": new["child_bound_evaluations"] < old["child_bound_evaluations"]})
    target_gate = bool(len(target_results) == 4 and all(item["child_evaluations_strictly_lower"] for item in target_results))
    valid, quality, preserved, nonincrease = not failures, not nonclosures, not mismatches, not regressions
    advance = bool(valid and complete and quality and preserved and nonincrease and strict and target_gate)
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "PRUNE_FIRST_DUAL_DEVELOPMENT_ADVANCE" if advance else "PRUNE_FIRST_DUAL_DEVELOPMENT_HOLD",
        "state_count": count, "execution_checks_passed": valid, "references_complete": bool(complete),
        "quality_gate_passed": bool(valid and quality), "behavior_preserved": preserved,
        "dual_work_nonincrease_passed": nonincrease, "strict_pooled_dual_reduction": strict,
        "previous_timing_failures_improved": target_gate, "development_gate_passed": advance,
        "failures": failures, "nonclosures": nonclosures, "behavior_mismatches": mismatches, "dual_work_regressions": regressions,
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
        "savings_by_group": {group: _savings(groups[group]) for group in GROUPS},
        "previous_timing_failures": target_results,
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in states),
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
