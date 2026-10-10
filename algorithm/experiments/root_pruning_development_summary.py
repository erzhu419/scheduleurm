"""Reference correctness, finite-budget losses, and cumulative root-pruning work."""
from __future__ import annotations

from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, reference_key, selection_score
from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key
from algorithm.experiments.resumable_probe_development_summary import _candidate_work
from algorithm.experiments.certificate_routed_development_summary import audit_candidate
from algorithm.experiments.strict_score_proof_timing_summary import _same

ROOT_COUNTERS = ("roots_seen", "roots_solved", "roots_skipped_empty", "roots_skipped_bound", "roots_skipped_canonical_tie")


def audit_root_stage(stage):
    work, selection = stage["root_pruning"], stage["selection"]
    errors = []
    if any(type(work[key]) is not int or work[key] < 0 for key in ROOT_COUNTERS):
        errors.append("root counters must be nonnegative integers")
    if work["roots_seen"] != work["roots_solved"] + work["roots_skipped_empty"] + work["roots_skipped_bound"]:
        errors.append("root counters do not partition all visited strata")
    if work["roots_skipped_canonical_tie"] > work["roots_skipped_bound"]:
        errors.append("canonical-tie skips must be a subset of bound skips")
    if (work["roots_seen"] != len(stage["thresholds"])
            or work["roots_seen"] != selection["duration_root_count"]
            or work["roots_seen"] != stage["preprocessing"]["duration_threshold_count"]
            or work["roots_solved"] > selection["lp_calls"]):
        errors.append("visited strata or solved-root LP accounting differs")
    return errors


def _native_work(stages):
    return {"searches": len(stages),
        "search_work": {key: sum(stage["selection"][key] for stage in stages) for key in COUNTERS},
        "capacity_propagation": {key: sum(stage["capacity_propagation"][key] for stage in stages)
                                 for key in (stages[0]["capacity_propagation"] if stages else ())},
        "preprocessing": {key: sum(stage["preprocessing"][key] for stage in stages)
                          for key in (stages[0]["preprocessing"] if stages else ())},
        "root_pruning": {key: sum(stage["root_pruning"][key] for stage in stages)
                         for key in ROOT_COUNTERS} if stages and "root_pruning" in stages[0] else None}


def _work(states):
    return {"kernel_old": _native_work([state["traces"]["old"]["256"] for state in states]),
        "kernel_new": _native_work([state["traces"]["new"]["256"] for state in states]),
        "routed_old_borrowed_or_zero_control": _candidate_work([state["old_routed"] for state in states]),
        "routed_new": _candidate_work([state["routed"] for state in states]),
        "routed_root_pruning": {key: sum(state["routed"]["final_lp"]["root_pruning"][key] for state in states)
                               for key in ROOT_COUNTERS}}


def _trace_errors(trace, state, options, lanes, expected, reference, new):
    errors, previous = [], None
    for cap in state["caps"]:
        stage = trace[str(cap)]
        selection = selection_from_record(stage["selection"])
        checks = check_selection(selection, options, tuple(state["queue"]), lanes, cap, expected, reference)
        errors.extend(f"cap{cap}: {key}" for key, failed in checks.items() if failed)
        if new:
            errors.extend(f"cap{cap}: {message}" for message in audit_root_stage(stage))
        if previous is not None:
            if any(stage["selection"][key] < previous["selection"][key] for key in COUNTERS):
                errors.append("cumulative search work decreased")
            if any(stage["capacity_propagation"][key] < previous["capacity_propagation"][key]
                   for key in previous["capacity_propagation"]):
                errors.append("cumulative capacity work decreased")
            if any(stage["preprocessing"][key] != previous["preprocessing"][key]
                   for key in previous["preprocessing"] if key != "lift_option_probes"):
                errors.append("catalogue preprocessing was repeated or changed")
            if new and stage["root_pruning"] != previous["root_pruning"]:
                errors.append("resumption revisited duration roots")
        previous = stage
    return errors


def summarize(protocol, rows):
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, regressions, comparisons, cell_summaries, entries = [], [], [], [], [], []
    complete = True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        if row["registered_at_utc"] != protocol["registered_at_utc"]:
            raise ValueError("cell registration changed")
        identity = lambda state: (state["state_id"], tuple(state["queue"]), state["source"])
        if list(map(identity, row["states"])) != list(map(identity, cell["states"])):
            raise ValueError("cell states differ from the frozen roster")
        model = models[row["node"]]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        for state in row["states"]:
            pointer = {"node": row["node"], "lanes": row["lanes"], "queue": state["queue"], "state_id": state["state_id"]}
            reference = selection_from_record(state["reference"])
            expected = reference_key(reference, tuple(state["queue"]), len(options))
            if expected is None and state["oracle"] is not None:
                expected = oracle_key(state["oracle"], len(options))
            complete &= expected is not None and state["effective_reference_complete"]
            errors = []
            for label in ("old", "new"):
                errors.extend(label + ": " + message for message in _trace_errors(
                    state["traces"][label], state, options, row["lanes"], expected, reference, label == "new"))
            if not _same(state["traces"]["old"]["144"], state["old_routed"]["initial_lp"]):
                errors.append("executed old kernel differs from frozen routed initial snapshot")
            independent_initial = state["traces"]["new"]["144"]
            audit_state = {**state, "expected_initial_lp": {key: independent_initial[key] for key in (
                "selection", "preprocessing", "capacity_propagation")}}
            _, routing_errors, quality = audit_candidate(state["routed"], audit_state, protocol, model, row["lanes"], expected)
            errors.extend(routing_errors)
            for stage in (state["routed"]["initial_lp"], state["routed"]["final_lp"]):
                errors.extend(audit_root_stage(stage))
            if not _same(state["routed"]["initial_lp"], independent_initial):
                errors.append("routed initial LP differs from the independent new-kernel trace")
            if state["routed"]["resumed"] and not _same(state["routed"]["final_lp"], state["traces"]["new"]["256"]):
                errors.append("routed continuation differs from the independent cumulative trace")
            if errors:
                failures.append({**pointer, "errors": errors})
            if not quality:
                nonclosures.append({**pointer, "status": state["routed"]["status"]})
            for position, cap in enumerate(state["caps"]):
                old = selection_from_record(state["traces"]["old"][str(cap)]["selection"])
                new = selection_from_record(state["traces"]["new"][str(cap)]["selection"])
                old_score, new_score = selection_score(old, tuple(state["queue"])), selection_score(new, tuple(state["queue"]))
                comparison = {**pointer, "budget_position": position, "lp_budget": cap,
                    "old_lp_calls": old.lp_calls, "new_lp_calls": new.lp_calls,
                    "old_canonical_exact": old.canonical_exact, "new_canonical_exact": new.canonical_exact,
                    "old_score": str(old_score), "new_score": str(new_score), "score_regression": new_score < old_score}
                comparisons.append(comparison)
                if comparison["score_regression"]:
                    regressions.append(comparison)
            entries.append(state)
        regular = [state for state in row["states"] if state["source"] != "new_zero"]
        work = _work(regular)
        a, b = work["routed_new"], work["routed_old_borrowed_or_zero_control"]
        cost_ok = all(a["search_work"][key] <= b["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")) and a["shared_integer_states"] <= b["shared_integer_states"]
        cell_summaries.append({"node": row["node"], "lanes": row["lanes"], "regular_states": len(regular),
            "work_gate_passed": cost_ok, "work": work})
    regular = [state for state in entries if state["source"] != "new_zero"]
    work = _work(regular)
    new_lp = work["routed_new"]["search_work"]["lp_calls"]
    old_lp = work["routed_old_borrowed_or_zero_control"]["search_work"]["lp_calls"]
    execution = not failures and complete
    quality = not nonclosures
    cost = all(cell["work_gate_passed"] for cell in cell_summaries) and new_lp < old_lp
    finite = not regressions
    advance = execution and quality and cost and finite
    return {"decision": "INVALID_EXECUTION" if not execution else "ROOT_PRUNING_DEVELOPMENT_ADVANCE" if advance else "ROOT_PRUNING_DEVELOPMENT_HOLD",
        "state_count": len(entries), "regular_state_count": len(regular), "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_old_routed_control_calls": sum(row["new_old_routed_calls"] for row in rows),
        "executed_interface_calls": sum(len(state["caps"]) * 2 + 1 for state in entries) + sum(row["new_old_routed_calls"] for row in rows),
        "native_search_constructions": len(entries) * 3 + sum(row["new_old_routed_calls"] for row in rows),
        "effective_references_complete": complete, "execution_checks_passed": execution, "routed_quality_gate_passed": quality,
        "work_gate_passed": cost, "finite_budget_gate_passed": finite, "development_gate_passed": advance,
        "failures": failures, "routed_nonclosures": nonclosures, "finite_budget_score_regressions": regressions,
        "per_budget": [{"position": position, "states": sum(item["budget_position"] == position for item in comparisons),
            "score_regressions": sum(item["budget_position"] == position and item["score_regression"] for item in comparisons),
            "old_canonical_exact": sum(item["budget_position"] == position and item["old_canonical_exact"] for item in comparisons),
            "new_canonical_exact": sum(item["budget_position"] == position and item["new_canonical_exact"] for item in comparisons)} for position in range(4)],
        "nonzero_work": work, "all_executed_trace_and_routed_work": _work(entries), "cells": cell_summaries,
        "nonzero_routed_lp_ratio": new_lp / old_lp if old_lp else None,
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True,
        "limitations": protocol["limitations"]}
