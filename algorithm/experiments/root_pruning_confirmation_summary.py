"""Fresh root-pruning confirmation; preserve finite-budget losses and old timing HOLD."""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.certificate_routed_development_summary import audit_candidate
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _candidate_work
from algorithm.experiments.root_pruning_development_summary import (
    ROOT_COUNTERS, audit_root_stage, _native_work, _trace_errors,
)
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same

SOURCES = ("fresh_confirmation", "historical_regression", "new_zero")


def _identity(state):
    return tuple(state.get(key) for key in ("state_id", "source", "cohort", "panel", "seed", "source_index", "source_state_id", "historical_trigger", "ordinary_tail")), tuple(state["queue"])


def _work(states):
    return {"kernel_old": _native_work([state["traces"]["old"]["256"] for state in states]),
        "kernel_new": _native_work([state["traces"]["new"]["256"] for state in states]),
        "routed_old": _candidate_work([state["old_routed"] for state in states]),
        "routed_new": _candidate_work([state["routed"] for state in states]),
        "routed_root_pruning": {key: sum(state["routed"]["final_lp"]["root_pruning"][key] for state in states) for key in ROOT_COUNTERS}}


def _cost(work):
    old, new = work["routed_old"], work["routed_new"]
    return all(new["search_work"][key] <= old["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")) and new["shared_integer_states"] <= old["shared_integer_states"]


def _budget_summary(comparisons):
    return [{"position": position, "states": sum(item["budget_position"] == position for item in comparisons),
        **{field: sum(item["budget_position"] == position and item[field] for item in comparisons)
           for field in ("score_regression", "old_canonical_exact", "new_canonical_exact")}}
        for position in range(4)]


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, regressions, comparisons, cells, entries = [], [], [], [], [], []
    sources, qualities = defaultdict(list), defaultdict(list)
    complete = True
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        failures.append({"errors": ["all registered cells must occur once in order"]})
    for cell, row in zip(protocol["cells"], rows):
        model = models[cell["node"]]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        pointer = {"node": row["node"], "lanes": row["lanes"]}
        errors = []
        if (row["registered_at_utc"] != protocol["registered_at_utc"]
                or row["physical_lanes"] != model["lanes"] or row["option_count"] != len(options)):
            errors.append("cell registration or physical catalogue metadata differs")
        if list(map(_identity, row["states"])) != list(map(_identity, cell["states"])):
            errors.append("state roster differs from the frozen registration")
        if (row["runtime_start"]["cpu_affinity"] != [protocol["cpu"]]
                or row["runtime_start"]["cpu_affinity"] != row["runtime_end"]["cpu_affinity"]
                or any(not row[key]["gc_enabled"] for key in ("runtime_start", "runtime_end"))):
            errors.append("runtime affinity or GC differs from the registered execution policy")
        new_refs = [state for state in row["states"] if state["source"] != "historical_regression"]
        if (row["new_old_routed_calls"] != len(row["states"]) or row["new_reference_calls"] != len(new_refs)
                or row["new_reference_oracle_calls"] != sum(state["oracle"] is not None for state in new_refs)):
            errors.append("reference or independently executed old-routed call counts differ")
        if errors:
            failures.append({**pointer, "errors": errors})
        minima = {}
        for option in options:
            key = tuple(option.requirements)
            minima[key] = min(minima.get(key, float("inf")), max(1.0, option.duration))
        roots = len(set(minima.values()))
        cell_entries = []
        for state in row["states"]:
            point = {**pointer, **{key: state[key] for key in ("state_id", "queue", "source", "cohort", "panel", "seed")}}
            errors, expected = [], None
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("budget ladder differs from catalogue roots and registered caps")
            reference = selection_from_record(state["reference"])
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            if (state["reference"]["lp_node_budget"] != protocol["reference_budget"]
                    or state["reference_metrics"]["failures"]
                    or state["reference_metrics"]["reference_comparable_count"] != int(state["reference_complete"])
                    or any(state["reference_metrics"]["search_work"][key] != state["reference"][key] for key in COUNTERS)):
                errors.append("full reference has invalid budget, metrics or feasibility")
            checks = check_selection(reference, options, tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected)
            errors.extend("reference: " + key for key, failed in checks.items() if failed)
            complete &= expected is not None
            coverage = all(set(state["traces"][label]) == {str(cap) for cap in state["caps"]} for label in ("old", "new"))
            if not coverage:
                errors.append("all and only four registered cumulative snapshots are required")
                failures.append({**point, "errors": errors})
                continue
            for label in ("old", "new"):
                if any(stage["thresholds"] != sorted(set(minima.values())) for stage in state["traces"][label].values()):
                    errors.append(label + " thresholds differ from the physical catalogue strata")
                errors.extend(label + ": " + message for message in _trace_errors(
                    state["traces"][label], state, options, row["lanes"], expected, reference, label == "new"))
                routed = state["old_routed"] if label == "old" else state["routed"]
                initial = state["traces"][label]["144"]
                audit_state = {**state, "expected_initial_lp": {key: initial[key] for key in ("selection", "preprocessing", "capacity_propagation")}}
                _, extra, quality = audit_candidate(routed, audit_state, protocol, model, row["lanes"], expected)
                errors.extend(label + " routed: " + message for message in extra)
                if not _same(routed["initial_lp"], initial):
                    errors.append(label + " routed initial differs from independently executed trace144")
                if routed["resumed"] and not _same(routed["final_lp"], state["traces"][label]["256"]):
                    errors.append(label + " routed continuation differs from independently executed trace256")
                if label == "new":
                    for stage in (routed["initial_lp"], routed["final_lp"]):
                        errors.extend(audit_root_stage(stage))
                    qualities[state["source"]].append(quality)
                if not quality:
                    nonclosures.append({**point, "label": label, "status": routed["status"], "reference_complete": expected is not None})
            for position, cap in enumerate(state["caps"]):
                old = selection_from_record(state["traces"]["old"][str(cap)]["selection"])
                new = selection_from_record(state["traces"]["new"][str(cap)]["selection"])
                a, b = selection_score(old, tuple(state["queue"])), selection_score(new, tuple(state["queue"]))
                comparison = {**point, "budget_position": position, "lp_budget": cap,
                    "old_lp_calls": old.lp_calls, "new_lp_calls": new.lp_calls,
                    "old_canonical_exact": old.canonical_exact, "new_canonical_exact": new.canonical_exact,
                    "old_score": str(a), "new_score": str(b), "score_regression": b < a}
                comparisons.append(comparison)
                if b < a:
                    regressions.append(comparison)
            if errors:
                failures.append({**point, "errors": errors})
            entries.append(state)
            cell_entries.append(state)
            sources[state["source"]].append(state)
        cell_groups = {source: {"state_count": len(group := [state for state in cell_entries if state["source"] == source]),
                               "work": _work(group)} for source in SOURCES}
        for group in cell_groups.values():
            group["cost_nonincrease"] = _cost(group["work"])
        cells.append({**pointer, "groups": cell_groups})
    groups = {}
    for source in SOURCES:
        selected = sources[source]
        work = _work(selected)
        old_lp, new_lp = (work[label]["search_work"]["lp_calls"] for label in ("routed_old", "routed_new"))
        groups[source] = {"state_count": len(selected), "work": work,
            "quality_gate_passed": len(qualities[source]) == len(selected) and all(qualities[source]),
            "finite_budget_gate_passed": not any(item["source"] == source for item in regressions),
            "cost_nonincrease_in_every_cell": all(cell["groups"][source]["cost_nonincrease"] for cell in cells),
            "strict_pooled_lp_reduction": bool(selected and new_lp < old_lp),
            "routed_lp_ratio": new_lp / old_lp if old_lp else None,
            "per_budget": _budget_summary([item for item in comparisons if item["source"] == source])}
    valid = not failures
    fresh, regression, zero = (groups[source] for source in SOURCES)
    quality = bool(fresh["state_count"] and all(group["quality_gate_passed"] for group in groups.values()))
    cost = bool(fresh["state_count"] and fresh["cost_nonincrease_in_every_cell"] and fresh["strict_pooled_lp_reduction"]
                and regression["cost_nonincrease_in_every_cell"])
    finite = not regressions
    advance = valid and complete and quality and cost and finite
    new_references = [state for state in entries if state["source"] != "historical_regression"]
    borrowed = [state for state in entries if state["source"] == "historical_regression"]
    oracles = [state["oracle"] for state in new_references if state["oracle"] is not None]
    all_states = [state for row in rows for state in row["states"]]
    return {"decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "ROOT_PRUNING_FRESH_CONFIRMATION_ADVANCE" if advance else "ROOT_PRUNING_FRESH_CONFIRMATION_HOLD",
        "state_count": len(all_states), "fresh_state_count": fresh["state_count"], "regression_state_count": regression["state_count"], "zero_state_count": zero["state_count"],
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_reference_oracle_calls": sum(row["new_reference_oracle_calls"] for row in rows),
        "borrowed_reference_count": len(borrowed), "new_old_routed_control_calls": sum(row["new_old_routed_calls"] for row in rows),
        "executed_interface_calls": len(all_states) * 10, "native_search_constructions": len(all_states) * 4,
        "reference_native_search_constructions": sum(row["new_reference_calls"] for row in rows),
        "effective_references_complete": complete, "execution_checks_passed": valid,
        "routed_quality_gate_passed": quality, "work_gate_passed": cost, "finite_budget_gate_passed": finite,
        "confirmation_gate_passed": advance, "failures": failures, "routed_nonclosures": nonclosures,
        "finite_budget_score_regressions": regressions, "per_budget": _budget_summary(comparisons),
        "groups": groups, "cells": cells, "all_executed_trace_and_routed_work": _work(entries),
        "reference_work": {"new": _combine_metrics(state["reference_metrics"] for state in new_references),
                           "borrowed": _combine_metrics(state["reference_metrics"] for state in borrowed)},
        "new_reference_oracle_work": {"calls": len(oracles), "work": {key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
