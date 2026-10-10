"""Audit the short LP prefix against the direct branch-reserved predecessor."""
from __future__ import annotations

from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.certificate_routed_development_summary import audit_candidate
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _candidate_work, _metric, _exact
from algorithm.experiments.root_pruning_development_summary import _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same, _audit_result


def _eligible(selected):
    return (selected["score_exact"] and not selected["canonical_exact"] and selected["frontier_regions_remaining"] > 0
            and selected["inherited_unsolved_regions"] == selected["frontier_regions_remaining"])


def audit_short_lookahead_router(result, state, protocol, model, lanes, expected):
    errors = []
    initial, final, prefix, probe = (result[key] for key in ("initial_lp", "final_lp", "lookahead", "probe"))
    initial_cap, total_cap = protocol["initial_lp_budget"], protocol["lp_node_budget"]
    for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget"):
        if result[key] != protocol[key]:
            errors.append(key + " differs from the registered router budget")
    if not _same(initial, state["expected_initial_lp"]):
        errors.append("initial LP differs from the independent native144 snapshot")
    limit = min(initial["selection"]["frontier_regions_remaining"], total_cap - initial_cap) if _eligible(initial["selection"]) else 0
    if len(prefix) > limit:
        errors.append("lookahead exceeds initial unresolved-frontier or total-budget allowance")
    previous = initial
    for index, stage in enumerate(prefix):
        if not _eligible(previous["selection"]):
            errors.append("lookahead continued after closure or a solved live region")
        if stage["selection"]["lp_node_budget"] != initial_cap + index + 1:
            errors.append("lookahead LP caps must increase by exactly one")
        if not _same(stage["thresholds"], initial["thresholds"]):
            errors.append("lookahead changed fixed native duration thresholds")
        previous = stage
    if len(prefix) < limit and _eligible(previous["selection"]):
        errors.append("lookahead stopped before its registered stopping condition")
    final_cap = final["selection"]["lp_node_budget"]
    if not initial_cap <= final_cap <= total_cap or result["resumed"] != (final_cap > initial_cap):
        errors.append("actual final LP cap or resumed flag is inconsistent")
    if final_cap == total_cap and not _same(final, state["expected_native_final"]):
        errors.append("full continuation differs from the independent native256 snapshot")
    stages = [initial, *prefix]
    if not _same(final, stages[-1]):
        stages.append(final)
    options = tuple(SimpleNamespace(**option) for option in model["options"])
    caps = [stage["selection"]["lp_node_budget"] for stage in stages]
    trace = {str(cap): stage for cap, stage in zip(caps, stages)}
    errors.extend(_trace_errors(trace, {**state, "caps": caps}, options, lanes, expected, selection_from_record(state["reference"]), True))
    roots = len(initial["thresholds"])
    for cap, stage in zip(caps, stages):
        errors.extend(audit_reserved_stage(stage, cap, roots))
    if not prefix:
        metric, extra, quality = audit_candidate(result, state, protocol, model, lanes, expected)
        errors.extend(extra)
        return metric, errors, quality
    current = prefix[-1]
    if current["selection"]["canonical_exact"]:
        selected, status, probe_stage = current["selection"], "LP_LOOKAHEAD_COMPLETE", "none"
        if probe is not None or not _same(final, current):
            errors.append("canonical lookahead must return immediately without an integer probe")
    else:
        probe_stage = "lookahead"
        if not current["selection"]["score_exact"] or probe is None:
            errors.append("unclosed lookahead requires one probe of its certified current score")
        if probe is not None:
            probe_protocol = {**protocol, "lp_budgets": {"candidate": current["selection"]["lp_node_budget"]}}
            try:
                _, extra, _ = _audit_result(probe, state, probe_protocol, model, lanes, "candidate", expected)
                errors.extend("probe: " + message for message in extra)
            except ValueError as error:
                errors.append("probe: " + str(error))
            if (not _same(probe["baseline_selection"], current["selection"])
                    or not _same(probe["preprocessing"], current["preprocessing"])
                    or not _same(probe["capacity_propagation"], current["capacity_propagation"])):
                errors.append("probe must reuse the paid final prefix snapshot")
            if (probe["score_proof"]["attempted"] or probe["score_proof"]["states_visited"]
                    or probe["score_proof"]["status"] != "SKIPPED_SCORE_CERTIFIED"):
                errors.append("lookahead probe must skip strict score proof")
        closed = probe is not None and probe["selection"]["canonical_exact"]
        selected = probe["selection"] if closed else final["selection"]
        status = "INTEGER_PROBE_COMPLETE" if closed else "LP_RESUME_COMPLETE" if selected["canonical_exact"] else "LP_RESUME_INCOMPLETE"
        if closed and not _same(final, current):
            errors.append("completed lookahead probe must stop at its current LP cap")
        if not closed and final_cap != total_cap:
            errors.append("exhausted lookahead probe requires full bounded continuation")
    if result["probe_stage"] != probe_stage or result["status"] != status or not _same(result["selection"], selected):
        errors.append("returned action, status or probe stage differs from its actual route")
    metric = _metric(result["selection"], state, model, lanes, final_cap, expected, final["preprocessing"])
    if metric["failures"]:
        errors.append("final selection has invalid feasibility, certificate or LP accounting")
    return metric, errors, _exact(result["selection"], metric, expected)


def _identity(state):
    return tuple(state.get(key) for key in ("state_id", "source", "cohort", "source_index", "prior_index", "source_state_id", "historical_regression")), tuple(state["queue"])


def _work(states):
    return {"old": _candidate_work([state["old_routed"] for state in states]), "new": _candidate_work([state["routed"] for state in states])}


def _amounts(work):
    return {label: {key: work[label]["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} |
            {"shared_integer_states": work[label]["shared_integer_states"]} for label in ("old", "new")}


def _cost(work):
    amounts = _amounts(work)
    return all(amounts["new"][key] <= amounts["old"][key] for key in amounts["old"])


def summarize(protocol, rows):
    failures, nonclosures, losses, increases, entries, cells, qualities = [], [], [], [], [], [], []
    models = {model["node"]: model for model in protocol["models"]}
    historical = {(item["node"], item["lanes"], tuple(item["queue"])) for item in protocol["historical_regression_keys"]}
    complete = True
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        failures.append({"errors": ["all registered development cells must occur once in order"]})
    for cell, row in zip(protocol["cells"], rows):
        model = models[cell["node"]]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        pointer = {"node": row["node"], "lanes": row["lanes"]}
        errors = []
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"] or row["option_count"] != len(options)):
            errors.append("cell registration or physical catalogue changed")
        if list(map(_identity, row["states"])) != list(map(_identity, cell["states"])):
            errors.append("state roster differs from the frozen development predecessor")
        if row["new_reference_calls"] or row["new_reference_oracle_calls"] or row["new_old_routed_calls"] != len(row["states"]):
            errors.append("references must be borrowed and both routing arms independently executed")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"] for key in ("runtime_start", "runtime_end")):
            errors.append("runtime affinity or GC differs from registration")
        if errors:
            failures.append({**pointer, "errors": errors})
        for state in row["states"]:
            point = {**pointer, **{key: state[key] for key in ("state_id", "queue", "source", "cohort", "historical_regression")}}
            errors, expected = [], None
            if state["historical_regression"] != ((row["node"], row["lanes"], tuple(state["queue"])) in historical):
                errors.append("historical subgroup flag differs from its fixed keys")
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
                errors.append("borrowed reference budget or metrics are invalid")
            checks = check_selection(selection_from_record(state["reference"]), options, tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected)
            errors.extend("reference: " + key for key, failed in checks.items() if failed)
            complete &= expected is not None
            if not _same(state["old_routed"], state["expected_old_routed"]):
                errors.append("cold direct predecessor did not reproduce its independently frozen output")
            for label, result, auditor in (("old", state["old_routed"], audit_candidate), ("new", state["routed"], audit_short_lookahead_router)):
                _, extra, quality = auditor(result, state, protocol, model, row["lanes"], expected)
                errors.extend(label + ": " + message for message in extra)
                if label == "old":
                    for name, cap in (("initial_lp", protocol["initial_lp_budget"]), ("final_lp", protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"])):
                        errors.extend(audit_reserved_stage(result[name], cap, len(result[name]["thresholds"])))
                    if result["resumed"] and not _same(result["final_lp"], state["expected_native_final"]):
                        errors.append("direct predecessor continuation differs from independent native256")
                qualities.append(quality)
                if not quality:
                    nonclosures.append({**point, "label": label, "status": result["status"], "reference_complete": expected is not None})
            a, b = (selection_score(selection_from_record(state[key]["selection"]), tuple(state["queue"])) for key in ("old_routed", "routed"))
            if b < a:
                losses.append({**point, "old_score": str(a), "new_score": str(b)})
            amounts = _amounts(_work([state]))
            changed = {key: {"old": amounts["old"][key], "new": amounts["new"][key]} for key in amounts["old"] if amounts["new"][key] > amounts["old"][key]}
            if changed:
                increases.append({**point, "increases": changed})
            if errors:
                failures.append({**point, "errors": errors})
            entries.append(state)
        nonzero = _work([state for state in row["states"] if state["source"] != "new_zero"])
        hist = _work([state for state in row["states"] if state["historical_regression"]])
        cells.append({**pointer, "nonzero_work": nonzero, "historical_work": hist,
                      "nonzero_cost_nonincrease": _cost(nonzero), "historical_cost_nonincrease": _cost(hist)})
    nonzero_states = [state for state in entries if state["source"] != "new_zero"]
    hist_states = [state for state in entries if state["historical_regression"]]
    nonzero, hist, all_work = _work(nonzero_states), _work(hist_states), _work(entries)
    cost = all(cell["nonzero_cost_nonincrease"] and cell["historical_cost_nonincrease"] for cell in cells)
    integer_gain = bool(nonzero_states and nonzero["new"]["shared_integer_states"] < nonzero["old"]["shared_integer_states"])
    valid, quality, no_loss = not failures, bool(entries and all(qualities)), not losses
    calls = len(entries) * 2
    if calls != protocol["expected_selector_calls"]:
        failures.append({"errors": ["executed independent router calls differ from registration"]})
        valid = False
    advance = valid and complete and quality and no_loss and cost and integer_gain
    advances = {"old": sum(1 + state["old_routed"]["resumed"] for state in entries),
        "new": sum(1 + len(state["routed"]["lookahead"]) + int(state["routed"]["final_lp"]["selection"]["lp_node_budget"] >
            (state["routed"]["lookahead"][-1]["selection"]["lp_node_budget"] if state["routed"]["lookahead"] else protocol["initial_lp_budget"])) for state in entries)}
    probes = {label: sum(state[key]["probe"] is not None for state in entries) for label, key in (("old", "old_routed"), ("new", "routed"))}
    return {"decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "SHORT_LOOKAHEAD_DEVELOPMENT_ADVANCE" if advance else "SHORT_LOOKAHEAD_DEVELOPMENT_HOLD",
        "state_count": len(entries), "nonzero_state_count": len(nonzero_states), "historical_state_count": len(hist_states),
        "executed_selector_calls": calls, "executed_interface_calls": calls,
        "interface_accounting": "Independent cold routing calls; nested advance_to and single _probe invocations are listed separately, not summed across interface layers. Borrowed snapshots and reference checks are not new selector calls.",
        "native_search_constructions": calls, "lp_advance_calls": advances, "integer_probe_calls": probes,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows), "new_reference_oracle_calls": sum(row["new_reference_oracle_calls"] for row in rows),
        "borrowed_reference_count": len(entries), "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in entries),
        "execution_checks_passed": valid, "effective_references_complete": complete, "routed_quality_gate_passed": quality,
        "final_score_gate_passed": no_loss, "work_nonincrease_gate_passed": cost, "strict_integer_reduction_gate_passed": integer_gain,
        "development_gate_passed": advance, "failures": failures, "routed_nonclosures": nonclosures, "final_score_losses": losses, "work_increases": increases,
        "nonzero_work": nonzero, "historical_work": hist, "all_executed_routed_work": all_work, "cells": cells,
        "lookahead": {"attempted": sum(bool(state["routed"]["lookahead"]) for state in entries),
            "canonical_successes": sum(state["routed"]["status"] == "LP_LOOKAHEAD_COMPLETE" for state in entries),
            "steps": sum(len(state["routed"]["lookahead"]) for state in entries)},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
