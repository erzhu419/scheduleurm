"""Independent three-arm confirmation with actual fresh and borrowed work."""
from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

from algorithm.experiments.affine_dual_development_summary import (
    AFFINE_COUNTERS, _affine_work, _canonical_work, _counter_schema, audit_affine_stage,
)
from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import _dual_work
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same
from algorithm.experiments.zero_delta_affine_development_summary import (
    ZERO_COUNTERS, _audit_zero_result, _work, _zero_work, audit_zero_stage, normalized_result,
)

ARMS = (("branch", "old_routed"), ("affine", "previous_dual_routed"), ("candidate", "routed"))


def _trace(state, label):
    return state["previous_dual_traces"] if label == "affine" else state["traces"]["old" if label == "branch" else "new"]


def _paid_work(states):
    native = {}
    for label, _ in ARMS:
        stages = [_trace(state, label)[str(state["caps"][-1])] for state in states]
        work = {**_native_work(stages), "canonical_work": _canonical_work([stage["selection"] for stage in stages])}
        if label != "branch":
            work.update(dual_reuse=_dual_work(stages), affine_dual=_affine_work(stages))
        if label == "candidate":
            work["zero_delta"] = _zero_work(stages)
        native[label] = work
    return {"native": native, "whole_selector": {label: _work([state[field] for state in states], label)
                                                  for label, field in ARMS}}


def _pointer(row, state):
    return {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
        "queue": state["queue"], "independent_source": state["independent_source"],
        "confirmation_source": state["confirmation_source"], "timing_group": state["timing_group"],
        "seed": state.get("seed")}


def summarize(protocol, rows):
    """Audit every new and borrowed state without selecting on zero hits."""
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, mismatches, entries = [], [], [], []
    groups = defaultdict(list)
    complete = True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        if (len(states) != len(cell["states"]) or not states or any(not _same(
                {key: state.get(key) for key in original}, original)
                for state, original in zip(states, cell["states"], strict=True))):
            raise ValueError("all independent queues, borrowed metadata and source pointers must remain frozen")
        if any(state["independent_source"] not in ("fresh", "borrowed_control") for state in states):
            raise ValueError("independent_source must identify actual new or borrowed records")
        fresh = sum(state["independent_source"] == "fresh" for state in states)
        point, errors = {"node": row["node"], "lanes": row["lanes"]}, []
        if (row["registered_at_utc"] != protocol["registered_at_utc"]
                or row["physical_lanes"] != model["lanes"] or row["option_count"] != len(options)):
            errors.append("cell registration or physical catalogue metadata differs")
        if (row["new_reference_calls"] != fresh or row["new_oracle_calls"] != sum(
                state["oracle"] is not None for state in states if state["independent_source"] == "fresh")
                or any(row[field] != fresh for field in ("candidate_calls", "affine_calls", "branch_calls"))
                or row["native_advance_calls"] != 12 * fresh or row["native_search_constructions"] != 6 * fresh):
            errors.append("only independent fresh states require references, conditional oracles and three cold native/router arms")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"]
               for key in ("runtime_start", "runtime_end")):
            errors.append("runtime affinity or GC differs from registration")
        if errors:
            failures.append({**point, "errors": errors})
        minima = {}
        for option in options:
            requirement = tuple(option.requirements)
            minima[requirement] = min(minima.get(requirement, float("inf")), max(1., option.duration))
        thresholds, roots = sorted(set(minima.values())), len(set(minima.values()))
        for state in states:
            pointer, errors = _pointer(row, state), []
            source = state["independent_source"]
            if source == "fresh" and (state["timing_group"] != "independent_fresh"
                    or state["confirmation_source"] != "fresh" or "prior_routed" in state):
                errors.append("independent fresh queues require their new group and no invented axis predecessor")
            if source == "borrowed_control" and state["timing_group"] == "independent_fresh":
                errors.append("borrowed old fresh cohorts must not be counted as independently executed queues")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("native ladder differs from catalogue strata and frozen budgets")
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
                errors.append("independent or borrowed reference has invalid budget, certificate or paid work")
            errors.extend("reference: " + name for name, failed in check_selection(reference, options,
                tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            for label, field in ARMS:
                trace, previous = _trace(state, label), None
                if set(trace) != {str(cap) for cap in state["caps"]}:
                    raise ValueError("all three actual or borrowed four-cap native ladders are required")
                errors.extend(label + ": " + message for message in _counter_schema(trace, affine=label != "branch"))
                errors.extend(label + ": " + message for message in _counter_schema(state[field], affine=label != "branch"))
                errors.extend(label + ": " + message for message in _trace_errors(
                    trace, state, options, row["lanes"], expected, reference, True))
                for cap in state["caps"]:
                    stage = trace[str(cap)]
                    if stage["thresholds"] != thresholds:
                        errors.append(label + " thresholds differ from physical catalogue")
                    errors.extend(f"{label} cap{cap}: " + message for message in audit_reserved_stage(stage, cap, roots))
                    if label != "branch":
                        errors.extend(f"{label} cap{cap}: " + message for message in audit_affine_stage(
                            stage, len(state["queue"]), previous))
                    if label == "candidate":
                        frozen = state["previous_dual_traces"][str(cap)]
                        errors.extend(f"candidate cap{cap}: " + message for message in audit_zero_stage(stage, previous, frozen))
                        if not _same(normalized_result(stage), normalized_result(frozen)):
                            mismatches.append({**pointer, "phase": "native", "lp_budget": cap})
                    previous = stage
                _, extra, quality = _audit_zero_result(state[field], state, protocol, model,
                                                     row["lanes"], label, expected)
                errors.extend(label + " routed: " + message for message in extra)
                if not quality:
                    nonclosures.append({**pointer, "label": label, "status": state[field]["status"],
                                        "reference_complete": expected is not None})
            if not _same(normalized_result(state["routed"]), normalized_result(state["previous_dual_routed"])):
                mismatches.append({**pointer, "phase": "whole_selector"})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            groups[state["timing_group"]].append(state)
    fresh_states = [state for state in entries if state["independent_source"] == "fresh"]
    controls = [state for state in entries if state["independent_source"] == "borrowed_control"]
    count, fresh_count = len(entries), len(fresh_states)
    counts = {"expected_state_count": count, "expected_independent_state_count": fresh_count,
        "expected_borrowed_state_count": len(controls), "expected_selector_calls": 3 * fresh_count,
        "expected_native_advance_calls": 12 * fresh_count, "expected_native_constructions": 6 * fresh_count,
        "expected_interface_calls": 15 * fresh_count, "expected_new_reference_calls": fresh_count}
    if any(protocol[name] != actual for name, actual in counts.items()):
        failures.append({"errors": ["full roster or actual independent method/reference coverage differs"]})
    if protocol["expected_group_counts"] != dict(Counter(state["timing_group"] for state in entries)):
        failures.append({"errors": ["new independent group and every borrowed old group must retain their complete sizes"]})
    executed, borrowed = _paid_work(fresh_states), _paid_work(controls)
    execution, behavior = not failures, not mismatches
    quality = bool(complete and not nonclosures)
    work_passed = execution
    advance = bool(execution and behavior and quality and work_passed)
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_ADVANCE" if advance else "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_HOLD",
        "state_count": count, "independent_state_count": fresh_count, "borrowed_state_count": len(controls),
        "execution_checks_passed": execution, "references_complete": bool(complete),
        "quality_gate_passed": bool(execution and quality), "behavior_preserved": behavior,
        "work_gate_passed": work_passed, "confirmation_gate_passed": advance,
        "failures": failures, "nonclosures": nonclosures, "behavior_mismatches": mismatches,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_oracle_calls": sum(row["new_oracle_calls"] for row in rows), "borrowed_reference_count": len(controls),
        "executed_selector_calls": sum(sum(row[field] for field in ("candidate_calls", "affine_calls", "branch_calls")) for row in rows),
        "executed_native_advance_calls": sum(row["native_advance_calls"] for row in rows),
        "native_search_constructions": sum(row["native_search_constructions"] for row in rows),
        "executed_interface_calls": sum(sum(row[field] for field in ("candidate_calls", "affine_calls", "branch_calls", "native_advance_calls")) for row in rows),
        "executed_work": executed, "borrowed_work": borrowed,
        "executed_affine_dual_work": {name: sum(executed[field]["candidate"]["affine_dual"][name]
            for field in ("native", "whole_selector")) for name in AFFINE_COUNTERS},
        "executed_zero_delta_work": {name: sum(executed[field]["candidate"]["zero_delta"][name]
            for field in ("native", "whole_selector")) for name in ZERO_COUNTERS},
        "reference_work": {"new": _combine_metrics(state["reference_metrics"] for state in fresh_states),
                           "borrowed": _combine_metrics(state["reference_metrics"] for state in controls)},
        "oracle_work": {source: {"calls": sum(state["oracle"] is not None for state in states),
            "work": {name: sum(state["oracle"]["work"][name] for state in states if state["oracle"] is not None)
                     for name in ORACLE_COUNTERS}} for source, states in (("new", fresh_states), ("borrowed", controls))},
        "groups": {name: {"state_count": len(states), "independent_sources": dict(Counter(
            state["independent_source"] for state in states)), "work": _paid_work(states)} for name, states in groups.items()},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
