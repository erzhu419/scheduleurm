"""Preserved search paths and paid sparse-support work on frozen controls."""
from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

from algorithm.experiments.affine_dual_development_summary import (
    AFFINE_COUNTERS, _affine_work, _canonical_work, _counter_schema,
)
from algorithm.experiments.affine_dual_fresh_confirmation_summary import audit_retained_completion
from algorithm.experiments.aligned_hardware_integer_capacity_development import _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.axis_capacity_frontier_development_summary import audit_axis_capacity_proof
from algorithm.experiments.branch_reserved_development_summary import _key, audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS, _dual_work
from algorithm.experiments.frontier_score_proof_confirmation_summary import audit_frontier_confirmation_router
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same
from algorithm.experiments.zero_delta_affine_development_summary import (
    ZERO_COUNTERS, _audit_zero_result, _work as _zero_routed_work, _zero_work,
)

SPARSE_COUNTERS = (
    "branch_changes", "capacity_changes", "capacity_collection_visits", "pending_maps_created",
    "pending_entries_copied", "pending_merge_reads", "pending_merge_writes", "pending_integer_operations",
    "pending_sort_calls", "pending_sort_entries", "guard_delta_reads", "guard_delta_comparisons",
    "update_delta_reads", "update_delta_comparisons", "anchor_resets", "anchor_entries_discarded",
)


def normalized_result(record):
    if isinstance(record, dict):
        return {key: normalized_result(value) for key, value in record.items()
                if key not in ("affine_dual", "zero_delta", "sparse_delta")}
    if isinstance(record, (list, tuple)):
        return [normalized_result(value) for value in record]
    return record


def audit_sparse_stage(stage, dimensions, previous=None, frozen=None):
    schemas = (("affine_dual", AFFINE_COUNTERS), ("zero_delta", ZERO_COUNTERS),
               ("sparse_delta", SPARSE_COUNTERS), ("dual_reuse", DUAL_COUNTERS))
    if any(set(stage.get(key, {})) != set(names) or any(type(value) is not int or value < 0
            for value in stage.get(key, {}).values()) for key, names in schemas):
        return ["sparse, zero, affine and dual work must retain their actual nonnegative integer fields"]
    affine, zero, sparse, dual = (stage[key] for key in ("affine_dual", "zero_delta", "sparse_delta", "dual_reuse"))
    groups = stage["preprocessing"]["requirement_group_count"]
    saved, prepared = dual["certificates_saved"], affine["certificates_prepared"]
    variables, visits, evaluations = affine["coefficient_weight_reads"], affine["bound_variable_visits"], dual["child_bound_evaluations"]
    changed = affine["changed_lower_coordinates"] + affine["changed_upper_coordinates"]
    expected_affine = {
        "certificate_numerator_products": saved,
        "price_reads": 2 * ((1 + dimensions) * prepared + variables),
        "denominator_lcm_steps": (2 + dimensions) * prepared + variables,
        "integer_scalings": (2 + dimensions) * prepared + 2 * variables,
        "coefficient_requirement_reads": dimensions * variables,
        "coefficient_integer_operations": (2 * dimensions + 3) * variables,
        "box_coordinate_reads": 0, "box_coordinate_comparisons": 0,
        "delta_coefficient_reads": changed, "delta_integer_operations": 2 * changed,
        "bound_integer_products": 2 * evaluations, "bound_rational_conversions": evaluations,
    }
    expected_zero = {
        "lower_flags_saved": stage["selection"]["dual_column_checks"],
        "lower_flag_comparisons": stage["selection"]["dual_column_checks"],
        "guard_calls": evaluations + zero["zero_delta_skips"],
        "box_coordinate_reads": 0, "box_coordinate_comparisons": 0,
        "upper_price_zero_comparisons": zero["upper_price_reads"],
    }
    merges = sparse["branch_changes"] + sparse["capacity_changes"]
    expected_sparse = {
        "pending_merge_reads": merges + sparse["pending_sort_entries"], "pending_merge_writes": merges,
        "pending_integer_operations": 3 * merges,
        "pending_sort_calls": sparse["pending_maps_created"],
        "guard_delta_reads": 3 * zero["variable_visits"],
        "guard_delta_comparisons": 2 * zero["variable_visits"],
        "update_delta_reads": 3 * visits, "update_delta_comparisons": 2 * visits,
        "anchor_resets": saved + evaluations,
    }
    errors = []
    if any(work[name] != count for work, expected in ((affine, expected_affine), (zero, expected_zero),
            (sparse, expected_sparse)) for name, count in expected.items()):
        errors.append("sparse support, coefficient preparation, guard or cached-delta fees differ from actual work")
    if (saved != stage["selection"]["dual_certified_lp_calls"] or prepared > saved
            or not dual["child_bound_prunes"] <= dual["child_bound_improvements"] <= evaluations
            or any(dual[name] for name in DUAL_COUNTERS[4:])
            or not prepared <= variables <= groups * prepared
            or not evaluations <= visits <= groups * evaluations
            or evaluations and not prepared
            or max(affine["changed_lower_coordinates"], affine["changed_upper_coordinates"]) > visits
            or not zero["guard_calls"] <= zero["variable_visits"] <= groups * zero["guard_calls"]
            or max(zero["lower_flag_reads"], zero["upper_price_reads"]) > zero["variable_visits"]
            or not sparse["pending_sort_entries"] <= sparse["pending_entries_copied"] <= 2 * sparse["pending_sort_entries"]
            or sparse["anchor_entries_discarded"] < visits):
        errors.append("prepared certificates, support sizes or child outcomes exceed their paid work")
    if previous is not None and any(stage[key][name] < previous[key][name]
            for key, names in schemas for name in names):
        errors.append("cumulative sparse, zero, affine or dual fees decreased")
    if frozen is not None and not _same(normalized_result(stage), normalized_result(frozen)):
        errors.append("native sparse path differs from the real frozen zero-delta stage")
    return errors


def _audit_sparse_result(result, state, protocol, model, lanes, label, expected):
    if label != "candidate":
        zero_state = {**state, "traces": {"old": state["traces"]["old"], "new": state["previous_zero_traces"]},
                      "previous_dual_traces": state["original_affine_traces"],
                      "previous_dual_routed": state["original_affine_routed"]}
        return _audit_zero_result(result, zero_state, protocol, model, lanes,
                                  "candidate" if label == "zero" else label, expected)
    metric, errors, quality = audit_frontier_confirmation_router(result, state, protocol, model, lanes, "new", expected)
    frozen = state["previous_zero_routed"]
    roots = len(result["initial_lp"]["thresholds"])
    previous = None
    for name, budget in (("initial_lp", protocol["initial_lp_budget"]),
                         ("final_lp", protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"])):
        errors.extend(audit_reserved_stage(result[name], budget, roots))
        errors.extend(audit_sparse_stage(result[name], len(state["queue"]), previous, frozen[name]))
        previous = result[name]
    if result["probe_stage"] == "frontier_late" and result["probe"] is not None and result["frontier_evidence"] is not None:
        errors.extend(audit_axis_capacity_proof(result["probe"], result["frontier_evidence"], state["queue"], lanes))
        errors.extend(audit_retained_completion(result["probe"], model, state["queue"], lanes))
    if not _same(normalized_result(result), normalized_result(frozen)):
        errors.append("whole sparse selector changed frozen zero LP, proof, completion or canonical work")
    return metric, errors, quality


def _sparse_work(stages):
    return {name: sum(stage["sparse_delta"][name] for stage in stages) for name in SPARSE_COUNTERS}


def _work(results, label):
    work = _zero_routed_work(results, "candidate" if label in ("candidate", "zero") else label)
    if label == "candidate":
        work["sparse_delta"] = _sparse_work([result["final_lp"] for result in results])
    return work


def _native(states, field, sparse=False):
    stages = [state[field][str(state["caps"][-1])] for state in states]
    work = {**_native_work(stages), "dual_reuse": _dual_work(stages), "affine_dual": _affine_work(stages),
            "canonical_work": _canonical_work([stage["selection"] for stage in stages]), "zero_delta": _zero_work(stages)}
    if sparse:
        work["sparse_delta"] = _sparse_work(stages)
    return work


def _group(states):
    return {"state_count": len(states),
        "executed": {"whole_selector": _work([state["routed"] for state in states], "candidate"),
                     "native": _native(states, "new_native", True)},
        "borrowed": {"whole_selector": _work([state["previous_zero_routed"] for state in states], "zero"),
                     "native": _native(states, "previous_zero_traces")}}


def summarize(protocol, rows):
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, mismatches, entries = [], [], [], []
    groups, complete = defaultdict(list), True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        if len(states) != len(cell["states"]) or not states or any(not _same(
                {key: state.get(key) for key in original}, original)
                for state, original in zip(states, cell["states"], strict=True)):
            raise ValueError("every frozen state, lineage and source pointer must be retained")
        count, point = len(states), {"node": row["node"], "lanes": row["lanes"]}
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options) or row["new_reference_calls"] or row["new_oracle_calls"]
                or row["candidate_calls"] != count or row["native_advance_calls"] != 4 * count
                or row["native_search_constructions"] != 2 * count or any(
                    row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"]
                    for key in ("runtime_start", "runtime_end"))):
            failures.append({**point, "errors": ["registered runtime, catalogue metadata or actual call coverage differs"]})
        minima = {}
        for option in options:
            requirement = tuple(option.requirements)
            minima[requirement] = min(minima.get(requirement, float("inf")), max(1., option.duration))
        thresholds, roots = sorted(set(minima.values())), len(set(minima.values()))
        for state in states:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "timing_group")}}
            errors = []
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("native ladder differs from catalogue strata and frozen budgets")
            reference = selection_from_record(state["reference"])
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            expected, extra = _reference(state, protocol, len(options))
            errors.extend(extra)
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
                errors.append("borrowed reference budget or paid certificate is invalid")
            errors.extend("reference: " + name for name, failed in check_selection(reference, options,
                tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            trace, frozen = state["traces"]["new"], state["previous_zero_traces"]
            if any(set(item) != {str(cap) for cap in state["caps"]}
                   for item in (trace, frozen, state["traces"]["old"], state["original_affine_traces"])):
                raise ValueError("all four new and real borrowed native snapshots are required")
            errors.extend(_counter_schema(trace, affine=True) + _counter_schema(state["routed"], affine=True))
            errors.extend(_trace_errors(trace, state, options, row["lanes"], expected, reference, True))
            previous = None
            for cap in state["caps"]:
                current, saved = trace[str(cap)], frozen[str(cap)]
                if current["thresholds"] != thresholds:
                    errors.append("native thresholds differ from physical catalogue")
                errors.extend(audit_reserved_stage(current, cap, roots))
                errors.extend(audit_sparse_stage(current, len(state["queue"]), previous, saved))
                if not _same(normalized_result(current), normalized_result(saved)):
                    mismatches.append({**pointer, "phase": "native", "lp_budget": cap})
                previous = current
            _, extra, quality = _audit_sparse_result(state["routed"], state, protocol, model, row["lanes"], "candidate", expected)
            errors.extend(extra)
            if not quality:
                nonclosures.append(pointer)
            if not _same(normalized_result(state["routed"]), normalized_result(state["previous_zero_routed"])):
                mismatches.append({**pointer, "phase": "whole_selector"})
            if errors:
                failures.append({**pointer, "errors": errors})
            entry = {**state, "new_native": trace}
            entries.append((row, entry))
            groups[state["timing_group"]].append(entry)
    states, count = [state for _, state in entries], len(entries)
    if protocol["expected_group_counts"] != dict(Counter(state["timing_group"] for state in states)) or any(
            protocol[key] != actual for key, actual in (("expected_state_count", count),
                ("expected_selector_calls", count), ("expected_native_advance_calls", count * 4),
                ("expected_native_constructions", count * 2), ("expected_interface_calls", count * 5))):
        failures.append({"errors": ["full cohort coverage or actual-call counts differ"]})
    work = _group(states)
    execution, behavior, quality = not failures, not mismatches, bool(complete and not nonclosures)
    advance = bool(execution and behavior and quality)
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "SPARSE_ZERO_DELTA_AFFINE_DEVELOPMENT_ADVANCE" if advance else "SPARSE_ZERO_DELTA_AFFINE_DEVELOPMENT_HOLD",
        "state_count": count, "independent_confirmation": False, "execution_checks_passed": execution,
        "references_complete": bool(complete), "quality_gate_passed": bool(execution and quality),
        "behavior_preserved": behavior, "work_gate_passed": execution, "development_gate_passed": advance,
        "failures": failures, "nonclosures": nonclosures, "behavior_mismatches": mismatches,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_oracle_calls": sum(row["new_oracle_calls"] for row in rows), "borrowed_reference_count": count,
        "executed_selector_calls": sum(row["candidate_calls"] for row in rows),
        "executed_native_advance_calls": sum(row["native_advance_calls"] for row in rows),
        "native_search_constructions": sum(row["native_search_constructions"] for row in rows),
        "executed_interface_calls": sum(row["candidate_calls"] + row["native_advance_calls"] for row in rows), **work,
        "groups": {group: _group(items) for group, items in groups.items()},
        "focus_states": [{"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "queue": state["queue"], **_group([state])} for row, state in entries if any(
                _key(row["node"], row["lanes"], state) == _key(item["node"], item["lanes"], item)
                for item in protocol["focus_keys"])],
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in states),
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
