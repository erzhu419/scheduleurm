"""Path preservation and actual fees for exact zero-delta dual reuse."""
from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

from algorithm.experiments.affine_dual_development_summary import (
    AFFINE_COUNTERS, _affine_work, _canonical_work, _counter_schema, audit_affine_stage,
)
from algorithm.experiments.affine_dual_fresh_confirmation_summary import (
    _audit_confirmation_result, _work as _previous_work,
)
from algorithm.experiments.aligned_hardware_integer_capacity_development import _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.branch_reserved_development_summary import _key, audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS, _dual_work
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same

ZERO_COUNTERS = (
    "lower_flags_saved", "lower_flag_comparisons", "guard_calls", "zero_delta_skips",
    "variable_visits", "box_coordinate_reads", "box_coordinate_comparisons",
    "lower_flag_reads", "upper_price_reads", "upper_price_zero_comparisons",
)


def normalized_result(record):
    """Ignore only changed arithmetic; retain all search and canonical work."""
    if isinstance(record, dict):
        return {key: ({name: normalized_result(item) for name, item in value.items()
                       if name != "child_bound_evaluations"} if key == "dual_reuse"
                      else normalized_result(value))
                for key, value in record.items() if key not in ("affine_dual", "zero_delta")}
    if isinstance(record, (list, tuple)):
        return [normalized_result(value) for value in record]
    return record


def audit_zero_stage(stage, previous=None, frozen=None):
    work = stage.get("zero_delta", {})
    if set(work) != set(ZERO_COUNTERS) or any(type(value) is not int or value < 0 for value in work.values()):
        return ["zero-delta work requires all ten actual nonnegative integer counters"]
    errors = []
    dual, groups = stage["dual_reuse"], stage["preprocessing"]["requirement_group_count"]
    calls, skips, visits = work["guard_calls"], work["zero_delta_skips"], work["variable_visits"]
    expected = {
        "lower_flags_saved": stage["selection"]["dual_column_checks"],
        "lower_flag_comparisons": stage["selection"]["dual_column_checks"],
        "guard_calls": dual["child_bound_evaluations"] + skips,
        "box_coordinate_reads": 4 * visits, "box_coordinate_comparisons": 2 * visits,
        "upper_price_zero_comparisons": work["upper_price_reads"],
    }
    if any(work[key] != value for key, value in expected.items()):
        errors.append("saved exact flags, actual evaluations/skips or guard fees differ from paid work")
    if not (calls <= visits <= groups * calls and work["lower_flag_reads"] <= visits
            and work["upper_price_reads"] <= visits):
        errors.append("guard variable and support reads exceed actual scanned boxes")
    if frozen is not None and (calls != frozen["dual_reuse"]["child_bound_evaluations"]
            or not _same(normalized_result(stage), normalized_result(frozen))):
        errors.append("guard coverage or preserved native path differs from the frozen affine stage")
    if previous is not None and any(work[key] < previous["zero_delta"][key] for key in ZERO_COUNTERS):
        errors.append("cumulative zero-delta fees decreased")
    return errors


def _audit_zero_result(result, state, protocol, model, lanes, label, expected):
    if label == "affine":
        state = {**state, "traces": {"old": state["traces"]["old"],
                                     "new": state["previous_dual_traces"]}}
    metric, errors, quality = _audit_confirmation_result(
        result, state, protocol, model, lanes, "candidate" if label == "affine" else label, expected)
    if label != "candidate":
        return metric, errors, quality
    frozen = state["previous_dual_routed"]
    errors.extend("initial_lp: " + message for message in audit_zero_stage(
        result["initial_lp"], frozen=frozen["initial_lp"]))
    errors.extend("final_lp: " + message for message in audit_zero_stage(
        result["final_lp"], result["initial_lp"], frozen["final_lp"]))
    if not _same(normalized_result(result), normalized_result(frozen)):
        errors.append("whole selector search, proof or canonical path differs from frozen affine")
    return metric, errors, quality


def _zero_work(stages):
    return {name: sum(stage["zero_delta"][name] for stage in stages) for name in ZERO_COUNTERS}


def _work(results, label):
    work = _previous_work(results, "candidate" if label == "affine" else label)
    if label == "candidate":
        work["zero_delta"] = _zero_work([result["final_lp"] for result in results])
    return work


def _native(states, field):
    stages = [state[field][str(state["caps"][-1])] for state in states]
    return {**_native_work(stages), "dual_reuse": _dual_work(stages),
            "affine_dual": _affine_work(stages),
            "canonical_work": _canonical_work([stage["selection"] for stage in stages]),
            **({"zero_delta": _zero_work(stages)} if field == "new_native" else {})}


def _group(states):
    return {"state_count": len(states),
        "executed": {"whole_selector": _work([state["routed"] for state in states], "candidate"),
                     "native": _native(states, "new_native")},
        "borrowed": {"whole_selector": _work([state["previous_dual_routed"] for state in states], "affine"),
                     "native": _native(states, "previous_dual_traces")}}


def summarize(protocol, rows):
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    models = {model["node"]: model for model in protocol["models"]}
    fixed = {group: {_key(item["node"], item["lanes"], item) for item in protocol[field]}
             for group, field in (("historical_regression", "historical_regression_keys"),
                 ("trigger_control", "trigger_control_keys"), ("new_trigger", "new_trigger_keys"))}
    failures, nonclosures, mismatches, entries = [], [], [], []
    groups = defaultdict(list)
    complete = True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        if len(states) != len(cell["states"]) or not states or any(not _same(
                {key: state.get(key) for key in original}, original)
                for state, original in zip(states, cell["states"], strict=True)):
            raise ValueError("every frozen state and its source pointers must be retained")
        count, point = len(states), {"node": row["node"], "lanes": row["lanes"]}
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options) or row["new_reference_calls"] or row["new_oracle_calls"]
                or row["candidate_calls"] != count or row["native_advance_calls"] != 4 * count
                or row["native_search_constructions"] != 2 * count or any(
                    row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"]
                    for key in ("runtime_start", "runtime_end"))):
            failures.append({**point, "errors": ["frozen runtime, catalogue metadata or actual cold-call coverage differs"]})
        minima = {}
        for option in options:
            requirement = tuple(option.requirements)
            minima[requirement] = min(minima.get(requirement, float("inf")), max(1., option.duration))
        thresholds, roots = sorted(set(minima.values())), len(set(minima.values()))
        for state in states:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "timing_group")}}
            errors, key = [], _key(row["node"], row["lanes"], state)
            group = ("fresh_confirmation" if state["confirmation_source"] == "fresh" else
                     next((name for name in ("new_trigger", "trigger_control", "historical_regression") if key in fixed[name]),
                          "zero_queue" if not any(state["queue"]) else "ordinary_development"))
            if state["confirmation_source"] not in ("fresh", "borrowed_control") or state["timing_group"] != group:
                errors.append("all previously confirmed queues and retained control cohorts must stay separate")
            if state["confirmation_source"] == "fresh" and "prior_routed" in state:
                errors.append("previously fresh states must not invent an axis predecessor")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("native ladder differs from physical strata and fixed budgets")
            reference = selection_from_record(state["reference"])
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            expected, extra = _reference(state, protocol, len(options))
            errors.extend(extra)
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
                errors.append("borrowed reference budget or paid certificate is invalid")
            errors.extend("reference: " + name for name, failed in check_selection(
                reference, options, tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            trace, frozen = state["traces"]["new"], state["previous_dual_traces"]
            if any(set(item) != {str(cap) for cap in state["caps"]} for item in (trace, frozen, state["traces"]["old"])):
                raise ValueError("all four new and real borrowed native snapshots are required")
            errors.extend(_counter_schema(trace, affine=True) + _counter_schema(state["routed"], affine=True))
            errors.extend(_trace_errors(trace, state, options, row["lanes"], expected, reference, True))
            previous = None
            for cap in state["caps"]:
                current, saved = trace[str(cap)], frozen[str(cap)]
                if current["thresholds"] != thresholds:
                    errors.append("native thresholds differ from physical catalogue")
                errors.extend(audit_reserved_stage(current, cap, roots))
                errors.extend(audit_affine_stage(current, len(state["queue"]), previous))
                errors.extend(audit_zero_stage(current, previous, saved))
                if not _same(normalized_result(current), normalized_result(saved)):
                    mismatches.append({**pointer, "phase": "native", "lp_budget": cap})
                previous = current
            _, extra, quality = _audit_zero_result(state["routed"], state, protocol, model, row["lanes"], "candidate", expected)
            errors.extend(extra)
            if not quality:
                nonclosures.append(pointer)
            if not _same(normalized_result(state["routed"]), normalized_result(state["previous_dual_routed"])):
                mismatches.append({**pointer, "phase": "whole_selector"})
            if errors:
                failures.append({**pointer, "errors": errors})
            entry = {**state, "new_native": trace}
            entries.append((row, entry))
            groups[group].append(entry)
    states = [state for _, state in entries]
    count = len(states)
    if protocol["expected_group_counts"] != dict(Counter(state["timing_group"] for state in states)) or any(
            protocol[key] != actual for key, actual in (("expected_state_count", count),
                ("expected_selector_calls", count), ("expected_native_advance_calls", count * 4),
                ("expected_native_constructions", count * 2), ("expected_interface_calls", count * 5))):
        failures.append({"errors": ["complete cohort coverage or registered actual-call counts differ"]})
    work = _group(states)
    skips = work["executed"]["whole_selector"]["zero_delta"]["zero_delta_skips"]
    execution, behavior = not failures, not mismatches
    quality = bool(complete and not nonclosures)
    work_passed = bool(execution and skips)
    advance = bool(execution and behavior and quality and work_passed)
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "INVALID_EXECUTION" if not execution else "INCOMPLETE_REFERENCE" if not complete else
                    "ZERO_DELTA_AFFINE_DEVELOPMENT_ADVANCE" if advance else "ZERO_DELTA_AFFINE_DEVELOPMENT_HOLD",
        "state_count": count, "execution_checks_passed": execution, "references_complete": bool(complete),
        "quality_gate_passed": bool(execution and quality), "behavior_preserved": behavior,
        "work_gate_passed": work_passed, "development_gate_passed": advance,
        "failures": failures, "nonclosures": nonclosures, "behavior_mismatches": mismatches,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_oracle_calls": sum(row["new_oracle_calls"] for row in rows), "borrowed_reference_count": count,
        "executed_selector_calls": sum(row["candidate_calls"] for row in rows),
        "executed_native_advance_calls": sum(row["native_advance_calls"] for row in rows),
        "native_search_constructions": sum(row["native_search_constructions"] for row in rows),
        "executed_interface_calls": count * 5, **work,
        "groups": {group: _group(items) for group, items in groups.items()},
        "focus_states": [{"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            **_group([state])} for row, state in entries if any(_key(row["node"], row["lanes"], state) ==
                _key(item["node"], item["lanes"], item) for item in protocol["focus_keys"])],
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in states),
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True,
        "limitations": protocol["limitations"]}
