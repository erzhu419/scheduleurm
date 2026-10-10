"""Root canonical routing quality and complete paid work on frozen controls."""
from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

from algorithm.experiments.affine_dual_development_summary import _canonical_work, audit_affine_stage
from algorithm.experiments.affine_dual_fresh_confirmation_summary import audit_retained_completion
from algorithm.experiments.aligned_hardware_integer_capacity_development import _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.axis_capacity_frontier_development_summary import audit_axis_capacity_proof
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.frontier_score_proof_confirmation_summary import audit_frontier_confirmation_router
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _exact, _metric
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _audit_result as _audit_strict, _same
from algorithm.experiments.zero_delta_affine_development_summary import (
    ZERO_COUNTERS, _audit_zero_result, _native as _zero_native, _work as _zero_work, audit_zero_stage,
)

ROUTING_COUNTERS = (
    "option_visits", "requirement_coordinate_reads", "duration_reads", "duration_comparisons",
    "group_lookups", "group_writes", "threshold_value_reads", "threshold_set_insertions",
    "root_snapshot_calls", "root_snapshot_heap_entries_read", "root_snapshot_retained_entries_read",
)
LEGACY_FIELDS = (
    "selection", "initial_lp", "final_lp", "probe", "resumed", "initial_lp_budget",
    "lp_node_budget", "integer_state_budget", "status", "probe_stage", "frontier_evidence",
)


def _catalogue(model):
    minima = {}
    for option in model["options"]:
        required = tuple(option["requirements"])
        minima[required] = min(minima.get(required, float("inf")), max(1., option["duration"]))
    return len(minima), sorted(set(minima.values()))


def _control_state(state):
    return {**state, "traces": {"old": state["traces"]["old"], "new": state["previous_zero_traces"]},
            "previous_dual_traces": state["original_affine_traces"],
            "previous_dual_routed": state["original_affine_routed"]}


def audit_routing_preparation(result, model, queue):
    work = result.get("routing_preparation", {})
    if set(work) != set(ROUTING_COUNTERS) or any(type(v) is not int or v < 0 for v in work.values()):
        return ["root routing preparation requires all eleven actual nonnegative integer fees"]
    options, (groups, thresholds) = len(model["options"]), _catalogue(model)
    root = result["root_lp"]["selection"]
    expected = dict(option_visits=options, requirement_coordinate_reads=options * len(queue),
        duration_reads=options, duration_comparisons=2 * options - groups,
        group_lookups=options, group_writes=options, threshold_value_reads=groups,
        threshold_set_insertions=groups, root_snapshot_calls=1,
        root_snapshot_retained_entries_read=2 * root["frontier_regions_remaining"])
    errors = []
    if any(work[key] != value for key, value in expected.items()):
        errors.append("root grouping, duration comparisons or retained snapshot fees differ from actual work")
    heap_reads = work["root_snapshot_heap_entries_read"]
    if (heap_reads % 2 or heap_reads < work["root_snapshot_retained_entries_read"]
            or heap_reads > 2 * result["root_lp"]["root_pruning"]["roots_solved"]):
        errors.append("root snapshot must charge its two raw heap scans before any branch")
    if result["root_lp_budget"] != len(thresholds):
        errors.append("root LP budget must equal the physical catalogue's duration strata")
    return errors


def _audit_early_result(result, state, protocol, model, lanes, label, expected):
    """Audit actual root routing; other labels use their true frozen controls."""
    if label != "candidate":
        return _audit_zero_result(result, _control_state(state), protocol, model, lanes,
                                  "candidate" if label == "zero" else label, expected)
    errors = audit_routing_preparation(result, model, state["queue"])
    root, initial, final, probe = (result[name] for name in ("root_lp", "initial_lp", "final_lp", "probe"))
    roots = len(root["thresholds"])
    for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget"):
        if result[key] != protocol[key]:
            errors.append(f"early candidate {key} differs from registration")
    previous = None
    for name in ("root_lp", "initial_lp", "final_lp"):
        stage = result[name]
        budget = stage["selection"]["lp_node_budget"]
        if budget not in (roots, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
            errors.append(name + " uses an unregistered actual native cap")
            continue
        if not _same(stage, state["previous_zero_traces"][str(budget)]):
            errors.append(name + " differs from its full raw frozen native snapshot")
        errors.extend(name + ": " + message for message in audit_reserved_stage(stage, budget, roots))
        errors.extend(name + ": " + message for message in audit_affine_stage(stage, len(state["queue"]), previous))
        errors.extend(name + ": " + message for message in audit_zero_stage(stage, previous))
        previous = stage
    root_closed = root["selection"]["canonical_exact"]
    early = root["selection"]["score_exact"] and not root_closed
    if root_closed:
        if (probe is not None or result["probe_stage"] != "none" or result["resumed"]
                or result["status"] != "LP_ROOT_COMPLETE" or result["frontier_evidence"] is not None
                or not _same(initial, root) or not _same(final, root)
                or not _same(result["selection"], root["selection"])):
            errors.append("canonical-complete roots must return their own action without continuation or a probe")
    elif early:
        if probe is None or result["probe_stage"] != "root" or result["frontier_evidence"] is not None:
            errors.append("score-certified open roots require their sole retained root probe")
        if probe is not None:
            try:
                _, extra, _ = _audit_strict(probe, state,
                    {**protocol, "lp_budgets": {"candidate": roots}}, model, lanes, "candidate", expected)
                errors.extend("root probe: " + message for message in extra)
            except ValueError as error:
                errors.append("root probe: " + str(error))
            if any(not _same(probe[field], root[source]) for field, source in (
                    ("baseline_selection", "selection"), ("preprocessing", "preprocessing"),
                    ("capacity_propagation", "capacity_propagation"))):
                errors.append("root probe must use its raw paid root snapshot without another LP or preprocessing")
            if probe["selection"]["canonical_exact"]:
                if (result["resumed"] or result["status"] != "INTEGER_PROBE_COMPLETE"
                        or not _same(initial, root) or not _same(final, root)
                        or not _same(result["selection"], probe["selection"])):
                    errors.append("completed root probe must return at the actual root cap")
            else:
                if (probe["completion"]["status"] != "STATE_BUDGET_EXHAUSTED"
                        or probe["score_proof"]["states_visited"] + probe["completion"]["states_visited"]
                        != protocol["integer_state_budget"]):
                    errors.append("unsuccessful early completion must retain its full shared-budget exhaustion")
                if not _same(initial, state["previous_zero_traces"][str(protocol["initial_lp_budget"])]):
                    errors.append("exhausted root probe must continue the same native search to the initial cap")
                should_resume = not initial["selection"]["canonical_exact"]
                target = protocol["lp_node_budget"] if should_resume else protocol["initial_lp_budget"]
                status = ("LP_RESUME_COMPLETE" if final["selection"]["canonical_exact"] else "LP_RESUME_INCOMPLETE") if should_resume else "LP_INITIAL_COMPLETE"
                if (result["resumed"] != should_resume or final["selection"]["lp_node_budget"] != target
                        or result["status"] != status or not _same(result["selection"], final["selection"])):
                    errors.append("early exhaustion permits only the registered native continuation and its raw final action")
    else:
        legacy = {key: result[key] for key in LEGACY_FIELDS}
        _, extra, _ = audit_frontier_confirmation_router(legacy, state, protocol, model, lanes, "new", expected)
        errors.extend(extra)
        if not _same(legacy, state["previous_zero_routed"]):
            errors.append("untriggered early route must preserve the real frozen zero selector")
        if result["probe_stage"] == "frontier_late" and probe is not None and result["frontier_evidence"] is not None:
            errors.extend(audit_axis_capacity_proof(probe, result["frontier_evidence"], state["queue"], lanes))
            errors.extend(audit_retained_completion(probe, model, state["queue"], lanes))
    budget = final["selection"]["lp_node_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, final["preprocessing"])
    if result["selection"]["lp_node_budget"] != budget or metric["failures"]:
        errors.append("returned early action has invalid feasibility, certificate or actual LP accounting")
    return metric, errors, _exact(result["selection"], metric, expected)


def _work(results, label):
    """Count each whole call's final native work and unique retained probe once."""
    work = _zero_work(results, "candidate" if label in ("candidate", "zero") else label)
    if label == "candidate":
        work["routing_preparation"] = {name: sum(result["routing_preparation"][name] for result in results)
                                       for name in ROUTING_COUNTERS}
        work["root_lp_calls_embedded"] = sum(result["root_lp"]["selection"]["lp_calls"] for result in results)
        work["root_probe_calls"] = sum(result["probe_stage"] == "root" for result in results)
        work["root_probe_exhausted_calls"] = sum(result["probe_stage"] == "root" and
            result["probe"]["completion"]["status"] == "STATE_BUDGET_EXHAUSTED" for result in results)
    return work


def _group(states):
    branch = [state["traces"]["old"][str(state["caps"][-1])] for state in states]
    native = {label: _zero_native(states, field) for label, field in (
        ("zero", "previous_zero_traces"), ("affine", "original_affine_traces"))}
    native["zero"]["zero_delta"] = {name: sum(state["previous_zero_traces"][str(state["caps"][-1])]["zero_delta"][name]
        for state in states) for name in ZERO_COUNTERS}
    native["branch"] = {**_native_work(branch), "canonical_work": _canonical_work([stage["selection"] for stage in branch])}
    return {"state_count": len(states), "executed": {"whole_selector": _work([state["routed"] for state in states], "candidate")},
        "borrowed": {"whole_selector": {label: _work([state[field] for state in states], label)
            for label, field in (("zero", "previous_zero_routed"), ("affine", "original_affine_routed"), ("branch", "old_routed"))},
            "native": native}}


def summarize(protocol, rows):
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, entries, groups = [], [], [], defaultdict(list)
    complete = True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        if len(states) != len(cell["states"]) or not states or any(not _same(
                {key: state.get(key) for key in original}, original)
                for state, original in zip(states, cell["states"], strict=True)):
            raise ValueError("every frozen development state and borrowed source pointer must be retained")
        count, point = len(states), {"node": row["node"], "lanes": row["lanes"]}
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options) or row["new_reference_calls"] or row["new_oracle_calls"]
                or row["candidate_calls"] != count or row["native_advance_calls"]
                or row["native_search_constructions"] != count or any(
                    row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"]
                    for key in ("runtime_start", "runtime_end"))):
            failures.append({**point, "errors": ["registered runtime, model or actual cold-call coverage differs"]})
        _, thresholds = _catalogue(model)
        roots = len(thresholds)
        for state in states:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "timing_group")}}
            errors = []
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("borrowed native ladder differs from the catalogue strata and frozen caps")
            reference = selection_from_record(state["reference"])
            expected = None
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
                errors.append("borrowed reference has invalid paid budget or certificate")
            errors.extend("reference: " + name for name, failed in check_selection(reference, options,
                tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            traces = (state["traces"]["old"], state["traces"]["new"], state["previous_zero_traces"], state["original_affine_traces"])
            if any(set(trace) != {str(cap) for cap in state["caps"]} for trace in traces):
                raise ValueError("all real borrowed branch, zero and affine four-cap snapshots are required")
            if not _same(state["traces"]["new"], state["previous_zero_traces"]):
                errors.append("borrowed zero native snapshots must retain their true source instead of an executed candidate ladder")
            errors.extend(_trace_errors(state["traces"]["new"], state, options, row["lanes"], expected, reference, True))
            _, extra, quality = _audit_early_result(state["routed"], state, protocol, model, row["lanes"], "candidate", expected)
            errors.extend(extra)
            if not quality:
                nonclosures.append({**pointer, "status": state["routed"]["status"]})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append((row, state))
            groups[state["timing_group"]].append(state)
    states, count = [state for _, state in entries], len(entries)
    expected_counts = dict(expected_state_count=count, expected_selector_calls=count,
        expected_native_advance_calls=0, expected_native_constructions=count, expected_interface_calls=count)
    if protocol["expected_group_counts"] != dict(Counter(state["timing_group"] for state in states)) or any(
            protocol[key] != value for key, value in expected_counts.items()):
        failures.append({"errors": ["full frozen cohort coverage or actual-call counts differ"]})
    work = _group(states)
    execution, quality = not failures, bool(complete and not nonclosures)
    advance = bool(execution and quality)
    oracles = [state["oracle"] for state in states if state["oracle"] is not None]
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "INVALID_EXECUTION" if not execution else "INCOMPLETE_REFERENCE" if not complete else
            "EARLY_ROOT_CANONICAL_DEVELOPMENT_ADVANCE" if advance else "EARLY_ROOT_CANONICAL_DEVELOPMENT_HOLD",
        "state_count": count, "independent_confirmation": False, "execution_checks_passed": execution,
        "references_complete": bool(complete), "quality_gate_passed": bool(execution and quality),
        "work_gate_passed": execution, "development_gate_passed": advance, "failures": failures, "nonclosures": nonclosures,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_oracle_calls": sum(row["new_oracle_calls"] for row in rows), "borrowed_reference_count": count,
        "executed_selector_calls": sum(row["candidate_calls"] for row in rows),
        "executed_native_advance_calls": 0, "native_search_constructions": sum(row["native_search_constructions"] for row in rows),
        "executed_interface_calls": sum(row["candidate_calls"] + row["native_advance_calls"] for row in rows),
        "executed_routing_preparation_work": work["executed"]["whole_selector"]["routing_preparation"], **work,
        "groups": {name: _group(items) for name, items in groups.items()},
        "root_probe_states": [{"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "queue": state["queue"], "completion_status": state["routed"]["probe"]["completion"]["status"],
            "work": _work([state["routed"]], "candidate")} for row, state in entries if state["routed"]["probe_stage"] == "root"],
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in states),
        "borrowed_reference_oracle_work": {"calls": len(oracles), "work": {
            key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
