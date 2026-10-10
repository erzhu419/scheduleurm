"""Independent fresh-queue quality with actual affine and retained-table fees."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from types import SimpleNamespace

from algorithm.experiments.affine_dual_development_summary import (
    AFFINE_COUNTERS, _affine_work, _audit_affine_result, _canonical_work, _counter_schema,
    _work as _affine_routed_work, audit_affine_stage,
)
from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.axis_capacity_frontier_development_summary import audit_axis_capacity_proof
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import _dual_work
from algorithm.experiments.frontier_score_proof_confirmation_summary import audit_frontier_confirmation_router
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.residual_capacity_frontier_development_summary import (
    RESIDUAL_COUNTERS, _coverage, _merged_values,
)
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _audit_result as _audit_strict, _same


def audit_retained_completion(probe, model, queue, lanes):
    """Audit paid residual work without inventing an unexecuted axis predecessor."""
    proof, work = probe["score_proof"], probe["completion"]
    bounds = proof["capacity"]["bounds"]
    errors = []
    if proof["capacity"].get("merged_values_copied") != len(bounds) * (lanes + 1):
        errors.append("retained merged vectors must charge every copied cell")
    for bound in bounds:
        if not _same(bound.get("exact_slot_numerators"), _merged_values(bound["axis_service_profiles"], queue, lanes)):
            errors.append("retained vector differs from the already-paid max-plus merge")
    if "prefix_states" not in work:
        return errors
    if any(type(work.get(key)) is not int or work[key] < 0 for key in RESIDUAL_COUNTERS):
        return errors + ["residual canonical work must retain all nonnegative integer fees"]
    if (work["states_visited"] != work["prefix_states"] + work["dfs_states_visited"]
            or work["prefix_reads"] != work["prefix_states"]
            or work["states_visited"] > work["state_budget"]
            or work["state_budget"] != probe["integer_state_budget"] - proof["states_visited"]):
        errors.append("canonical prefix and DFS work must share the unchanged proof remainder")
    baseline = probe["score_certified_selection"]
    score = Fraction(sum(q * done for q, done in zip(queue, baseline["service"], strict=True))) / Fraction.from_float(baseline["duration"])
    durations = [Fraction.from_float(float(max(1., option["duration"]))) for option in model["options"]]
    requirements = [tuple(option["requirements"]) for option in model["options"]]
    thresholds = sorted({duration for duration in durations if duration <= Fraction.from_float(baseline["duration"])})
    if not proof["complete"] or not score or not bounds or work["thresholds_checked"] > len(thresholds):
        errors.append("retained canonical work requires a paid complete proof and its original duration roster")
    reads, used = _coverage(bounds, thresholds[:work["thresholds_checked"]], requirements, durations, score)
    if reads != work["coverage_coordinate_reads"] or used != work["tables_reused"]:
        errors.append("table reuse must cover all allowed requirements and charge inspected coordinates")
    full = work["tables_reused"] * (lanes + 1)
    if (work["prefix_states"] > full or work["prefix_states"] < max(0, full - lanes - 1)
            or work["prefix_states"] != full and work["status"] != "STATE_BUDGET_EXHAUSTED"):
        errors.append("prefix work must include each selected table cell except the exhausted final prefix")
    if (work["capacity_prunes"] > work["capacity_queries"] or work["capacity_queries"] > work["dfs_states_visited"]
            or work["capacity_prunes"] > work["bound_prunes"]
            or not work["tables_reused"] and (work["capacity_queries"] or work["capacity_prunes"])):
        errors.append("residual queries and strict prunes must be actual covered canonical DFS work")
    return errors


def _audit_fresh_result(result, state, protocol, model, lanes, label, expected):
    """Audit fresh records; axis-predecessor equality stays in the old regression."""
    if label == "long":
        return _audit_strict(result, state, {**protocol, "lp_budgets": {"long": protocol["long_lp_budget"]},
            "integer_state_budget": protocol["control_integer_state_budget"]}, model, lanes, label, expected)
    arm = "new" if label == "candidate" else "old"
    metric, errors, quality = audit_frontier_confirmation_router(result, state, protocol, model, lanes, arm, expected)
    roots = len(result["initial_lp"]["thresholds"])
    for name, budget in (("initial_lp", protocol["initial_lp_budget"]),
                         ("final_lp", protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"])):
        errors.extend(audit_reserved_stage(result[name], budget, roots))
    if arm == "new":
        errors.extend(audit_affine_stage(result["initial_lp"], len(state["queue"])))
        errors.extend(audit_affine_stage(result["final_lp"], len(state["queue"]), result["initial_lp"]))
    if result["probe_stage"] == "frontier_late" and result["probe"] is not None and result["frontier_evidence"] is not None:
        errors.extend(audit_axis_capacity_proof(result["probe"], result["frontier_evidence"], state["queue"], lanes))
        errors.extend(audit_retained_completion(result["probe"], model, state["queue"], lanes))
    return metric, errors, quality


def _audit_confirmation_result(result, state, protocol, model, lanes, label, expected):
    if state["confirmation_source"] == "fresh":
        return _audit_fresh_result(result, state, protocol, model, lanes, label, expected)
    return _audit_affine_result(result, state, protocol, model, lanes, label, expected)


def _work(results, label):
    work = _affine_routed_work(results, label)
    selections = [result["selection"] if label == "long" else result["final_lp"]["selection"]
                  for result in results]
    return {**work, "canonical_work": _canonical_work(selections)}


def _pointer(row, state):
    return {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "queue": state["queue"], "confirmation_source": state["confirmation_source"],
            "seed": state.get("seed"), "timing_group": state["timing_group"]}


def _paid_work(states):
    native = {label: _native_work([state["traces"][arm][str(state["caps"][-1])] for state in states])
              for label, arm in (("branch", "old"), ("candidate", "new"))}
    for label, arm in (("branch", "old"), ("candidate", "new")):
        native[label]["canonical_work"] = _canonical_work([state["traces"][arm][str(state["caps"][-1])]["selection"]
                                                          for state in states])
    native["candidate"]["dual_reuse"] = _dual_work([state["traces"]["new"][str(state["caps"][-1])] for state in states])
    native["candidate"]["affine_dual"] = _affine_work([state["traces"]["new"][str(state["caps"][-1])] for state in states])
    return {"whole_selector": {label: _work([state[field] for state in states], label)
            for label, field in (("branch", "old_routed"), ("candidate", "routed"))}, "native": native}


def summarize(protocol, rows):
    """Keep every independent queue and all borrowed controls, including failures."""
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, entries = [], [], []
    groups = defaultdict(list)
    complete = True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        if (len(states) != len(cell["states"]) or not states or any(not _same(
                {key: state.get(key) for key in original}, original)
                for state, original in zip(states, cell["states"]))):
            raise ValueError("all frozen fresh queues and borrowed control pointers must remain complete")
        fresh = sum(state["confirmation_source"] == "fresh" for state in states)
        errors, point = [], {"node": row["node"], "lanes": row["lanes"]}
        if (row["registered_at_utc"] != protocol["registered_at_utc"]
                or row["physical_lanes"] != model["lanes"] or row["option_count"] != len(options)):
            errors.append("cell registration or physical catalogue metadata differs")
        if (row["new_reference_calls"] != fresh or row["candidate_calls"] != fresh or row["branch_calls"] != fresh
                or row["native_advance_calls"] != 8 * fresh or row["native_search_constructions"] != 4 * fresh
                or row["new_oracle_calls"] != sum(state["oracle"] is not None for state in states
                                                 if state["confirmation_source"] == "fresh")):
            errors.append("only every fresh state requires its reference, conditional oracle, two native ladders and two cold routers")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"]
               for key in ("runtime_start", "runtime_end")):
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
            source = state["confirmation_source"]
            if source not in ("fresh", "borrowed_control"):
                raise ValueError("confirmation source must identify actual fresh or borrowed control records")
            if (source == "fresh") != (state["timing_group"] == "fresh_confirmation"):
                errors.append("independent fresh queues must retain their separate timing group")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("native budget ladder differs from catalogue strata or frozen caps")
            reference = selection_from_record(state["reference"])
            expected = None
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            if (state["reference"]["lp_node_budget"] != protocol["reference_budget"]
                    or state["reference_metrics"]["failures"]
                    or state["reference_metrics"]["reference_comparable_count"] != int(state["reference_complete"])
                    or any(state["reference_metrics"]["search_work"][name] != state["reference"][name] for name in COUNTERS)):
                errors.append("independent reference has invalid budget, certificate or paid work")
            errors.extend("reference: " + name for name, failed in check_selection(reference, options,
                tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            if any(set(state["traces"][arm]) != {str(cap) for cap in state["caps"]} for arm in ("old", "new")):
                raise ValueError("both independently executed or borrowed four-cap native ladders are required")
            errors.extend(_counter_schema(state["traces"]["new"], affine=True))
            errors.extend(_counter_schema(state["routed"], affine=True))
            for label, arm, field in (("branch", "old", "old_routed"), ("candidate", "new", "routed")):
                trace, previous = state["traces"][arm], None
                errors.extend(label + ": " + message for message in _trace_errors(
                    trace, state, options, row["lanes"], expected, reference, True))
                for cap in state["caps"]:
                    stage = trace[str(cap)]
                    if stage["thresholds"] != thresholds:
                        errors.append(label + " thresholds differ from physical catalogue strata")
                    errors.extend(f"{label} cap{cap}: " + message for message in audit_reserved_stage(stage, cap, roots))
                    if arm == "new":
                        errors.extend(f"candidate cap{cap}: " + message for message in audit_affine_stage(stage, len(state["queue"]), previous))
                    previous = stage
                _, extra, quality = _audit_confirmation_result(state[field], state, protocol, model, row["lanes"], label, expected)
                errors.extend(label + " routed: " + message for message in extra)
                if not quality:
                    nonclosures.append({**pointer, "label": label, "status": state[field]["status"],
                                        "reference_complete": expected is not None})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            groups[state["timing_group"]].append(state)
    fresh_states = [state for state in entries if state["confirmation_source"] == "fresh"]
    controls = [state for state in entries if state["confirmation_source"] == "borrowed_control"]
    count, fresh_count = len(entries), len(fresh_states)
    expected_counts = {"expected_state_count": count, "expected_fresh_state_count": fresh_count,
        "expected_control_state_count": len(controls), "expected_selector_calls": fresh_count * 2,
        "expected_native_advance_calls": fresh_count * 8, "expected_native_constructions": fresh_count * 4,
        "expected_interface_calls": fresh_count * 10, "expected_new_reference_calls": fresh_count}
    if any(protocol[name] != actual for name, actual in expected_counts.items()):
        failures.append({"errors": ["executed full roster or method/reference coverage differs from registration"]})
    if protocol["expected_group_counts"] != dict(Counter(state["timing_group"] for state in entries)):
        failures.append({"errors": ["fresh confirmation and every original control group must keep their complete size"]})
    executed, borrowed = _paid_work(fresh_states), _paid_work(controls)
    affine = {name: sum(executed[field]["candidate"]["affine_dual"][name]
                      for field in ("native", "whole_selector")) for name in AFFINE_COUNTERS}
    execution, quality = not failures, not nonclosures
    advance = bool(execution and complete and quality)
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "INVALID_EXECUTION" if not execution else "INCOMPLETE_REFERENCE" if not complete else
            "AFFINE_DUAL_FRESH_CONFIRMATION_ADVANCE" if advance else "AFFINE_DUAL_FRESH_CONFIRMATION_HOLD",
        "state_count": count, "fresh_state_count": fresh_count, "control_state_count": len(controls),
        "execution_checks_passed": execution, "references_complete": bool(complete),
        "quality_gate_passed": bool(execution and quality), "affine_work_validated": execution,
        "confirmation_gate_passed": advance, "failures": failures, "nonclosures": nonclosures,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_oracle_calls": sum(row["new_oracle_calls"] for row in rows), "borrowed_reference_count": len(controls),
        "executed_selector_calls": sum(row["candidate_calls"] + row["branch_calls"] for row in rows),
        "executed_native_advance_calls": sum(row["native_advance_calls"] for row in rows),
        "native_search_constructions": sum(row["native_search_constructions"] for row in rows),
        "executed_interface_calls": sum(row["candidate_calls"] + row["branch_calls"] + row["native_advance_calls"] for row in rows),
        "executed_work": executed, "borrowed_work": borrowed, "executed_affine_dual_work": affine,
        "reference_work": {"new": _combine_metrics(state["reference_metrics"] for state in fresh_states),
                           "borrowed": _combine_metrics(state["reference_metrics"] for state in controls)},
        "oracle_work": {source: {"calls": sum(state["oracle"] is not None for state in states),
            "work": {name: sum(state["oracle"]["work"][name] for state in states if state["oracle"] is not None)
                     for name in ORACLE_COUNTERS}} for source, states in (("new", fresh_states), ("borrowed", controls))},
        "groups": {name: {"state_count": len(states), "confirmation_source": states[0]["confirmation_source"],
                           "work": _paid_work(states)} for name, states in groups.items()},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
