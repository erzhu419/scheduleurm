"""Fresh frontier-proof confirmation with separate historical and known-failure gates."""
from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.branch_reserved_development_summary import _identity, _key, _increases, audit_reserved_stage
from algorithm.experiments.certificate_routed_development_summary import audit_candidate
from algorithm.experiments.dual_reuse_confirmation_summary import _work as _dual_work_summary, _native_cost
from algorithm.experiments.dual_reuse_development_summary import audit_dual_stage
from algorithm.experiments.frontier_score_proof_development_summary import EXPORT_COUNTERS, audit_frontier_evidence
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _metric, _exact
from algorithm.experiments.root_pruning_confirmation_summary import _cost, _budget_summary
from algorithm.experiments.root_pruning_development_summary import _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _audit_result as _audit_strict, _same

SOURCES = ("fresh_confirmation", "historical_regression", "failure_regression", "new_zero")


def audit_frontier_confirmation_router(result, state, protocol, model, lanes, label, expected):
    trace = state["traces"][label]
    initial, final = result["initial_lp"], result["final_lp"]
    errors = []
    if not _same(initial, trace["144"]) or not _same(final, trace["256"] if result["resumed"] else trace["144"]):
        errors.append("cold routed snapshots differ from their independent native ladder")
    frontier = (not initial["selection"]["score_exact"] and not initial["selection"]["canonical_exact"]
                and not final["selection"]["score_exact"] and not final["selection"]["canonical_exact"])
    if not frontier:
        if result["frontier_evidence"] is not None:
            errors.append("ordinary route must not export a frontier or repeat an earlier probe")
        try:
            metric, extra, quality = audit_candidate(result, {**state, "expected_initial_lp": trace["144"]}, protocol, model, lanes, expected)
            errors.extend(extra)
            return metric, errors, quality
        except ValueError as error:
            errors.append(str(error))
    else:
        if any(result[key] != protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) or not result["resumed"]:
            errors.append("frontier route must use the registered initial/full LP and shared integer budgets")
        probe, evidence = result["probe"], result["frontier_evidence"]
        if result["probe_stage"] != "frontier_late" or probe is None or evidence is None:
            errors.append("unproved final score without an earlier probe requires one frontier proof and its evidence")
        if evidence is not None:
            errors.extend(audit_frontier_evidence(evidence, final, model, state["queue"], lanes))
        if probe is not None:
            try:
                _, extra, _ = _audit_strict(probe, state, {**protocol, "lp_budgets": {"candidate": protocol["lp_node_budget"]}},
                                           model, lanes, "candidate", expected)
                errors.extend("frontier probe: " + message for message in extra)
            except ValueError as error:
                errors.append("frontier probe: " + str(error))
            if (not _same(probe["baseline_selection"], final["selection"])
                    or not _same(probe["preprocessing"], final["preprocessing"])
                    or not _same(probe["capacity_propagation"], final["capacity_propagation"])):
                errors.append("frontier probe must reuse its paid final LP snapshot without new LP or preprocessing")
            if not probe["score_proof"]["attempted"]:
                errors.append("frontier proof must target its own unproved LP incumbent")
            if evidence is not None and (probe["score_proof"]["thresholds_checked"] > len(evidence["regions"])
                    or probe["score_proof"]["status"] == "COMPLETE" and probe["score_proof"]["thresholds_checked"] != len(evidence["regions"])):
                errors.append("frontier proof did not check all and only its exported strict boxes")
            status = "FRONTIER_PROBE_COMPLETE" if probe["selection"]["canonical_exact"] else "FRONTIER_PROBE_INCOMPLETE"
            if result["status"] != status or not _same(result["selection"], probe["selection"]):
                errors.append("returned frontier selection or status differs from the actual integer certificate")
    budget = protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, final["preprocessing"])
    if metric["failures"]:
        errors.append("returned frontier action has invalid feasibility, certificate or paid work")
    return metric, errors, _exact(result["selection"], metric, expected)


def _work(states):
    work = _dual_work_summary(states)
    work["frontier_export_work"] = {label: {key: sum(state[field]["frontier_evidence"]["export_work"][key]
        for state in states if state[field]["frontier_evidence"] is not None) for key in EXPORT_COUNTERS}
        for label, field in (("old", "old_routed"), ("new", "routed"))}
    work["probe_stage_counts"] = {label: dict(Counter(state[field]["probe_stage"] for state in states))
                                  for label, field in (("old", "old_routed"), ("new", "routed"))}
    return work


def _oracle_work(states):
    oracles = [state["oracle"] for state in states if state["oracle"] is not None]
    return {"calls": len(oracles), "work": {key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}}


def summarize(protocol, rows):
    failures, nonclosures, losses, comparisons, routed_losses, increases, entries, cells = [], [], [], [], [], [], [], []
    qualities = defaultdict(list)
    complete = True
    models = {model["node"]: model for model in protocol["models"]}
    all_states = [state for row in rows for state in row["states"]]
    count = len(all_states)
    expected_counts = {"expected_state_count": count,
        "expected_fresh_state_count": sum(state["source"] == "fresh_confirmation" for state in all_states),
        "expected_failure_regression_count": sum(state["source"] == "failure_regression" for state in all_states),
        "expected_selector_calls": count * 2, "expected_native_advance_calls": count * 8,
        "expected_interface_calls": count * 10, "expected_native_constructions": count * 4,
        "expected_new_reference_calls": sum(row["new_reference_calls"] for row in rows)}
    if any(protocol[key] != value for key, value in expected_counts.items()):
        failures.append({"errors": ["executed coverage differs from the registered method and reference counts"]})
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        failures.append({"errors": ["all registered cells must occur once in order"]})
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    fixed_keys = {source: {_key(item["node"], item["lanes"], item) for item in protocol[field]}
                  for source, field in (("historical_regression", "historical_regression_keys"), ("failure_regression", "failure_regression_keys"))}
    if any(not keys <= registered for keys in fixed_keys.values()):
        failures.append({"errors": ["fixed historical and known-failure keys must belong to the frozen roster"]})
    for cell, row in zip(protocol["cells"], rows):
        model = models[cell["node"]]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        point = {"node": row["node"], "lanes": row["lanes"]}
        identity = lambda state: (_identity(state), state.get("source_state_id"), state.get("source_record_source"), tuple(state["caps"]))
        errors = []
        if list(map(identity, row["states"])) != list(map(identity, cell["states"])):
            errors.append("state roster or source pointers differ from the frozen confirmation")
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options)):
            errors.append("cell registration or physical catalogue metadata differs")
        new_refs = [state for state in row["states"] if state["source"] in ("fresh_confirmation", "new_zero")]
        if (row["new_reference_calls"] != len(new_refs) or row["new_old_routed_calls"] != len(row["states"])
                or row["new_reference_oracle_calls"] != sum(state["oracle"] is not None for state in new_refs)):
            errors.append("fresh/zero reference, oracle or independently executed routed counts differ")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"] for key in ("runtime_start", "runtime_end")):
            errors.append("runtime affinity or GC differs from registration")
        if errors:
            failures.append({**point, "errors": errors})
        minima = {}
        for option in options:
            key = tuple(option.requirements)
            minima[key] = min(minima.get(key, float("inf")), max(1.0, option.duration))
        roots, thresholds = len(set(minima.values())), sorted(set(minima.values()))
        cell_entries = []
        for state in row["states"]:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "source", "cohort", "panel", "historical_regression")},
                       "seed": state.get("seed")}
            errors, expected = [], None
            key = _key(row["node"], row["lanes"], state)
            if state["historical_regression"] != (state["source"] == "historical_regression"):
                errors.append("historical flag differs from its fixed cohort")
            if any((state["source"] == source) != (key in keys) for source, keys in fixed_keys.items()):
                errors.append("historical or known-failure cohort differs from its independently frozen keys")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("budget ladder differs from catalogue roots and registered caps")
            reference = selection_from_record(state["reference"])
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
                errors.append("reference has invalid budget, certificate or recomputed paid work")
            checks = check_selection(reference, options, tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected)
            errors.extend("reference: " + key for key, failed in checks.items() if failed)
            complete &= expected is not None
            if not all(set(state["traces"][label]) == {str(cap) for cap in state["caps"]} for label in ("old", "new")):
                failures.append({**pointer, "errors": errors + ["all eight independently executed native snapshots are required"]})
                continue
            for label, field in (("old", "old_routed"), ("new", "routed")):
                trace, previous = state["traces"][label], None
                errors.extend(label + ": " + message for message in _trace_errors(trace, state, options, row["lanes"], expected, reference, True))
                for cap in state["caps"]:
                    stage = trace[str(cap)]
                    if stage["thresholds"] != thresholds:
                        errors.append(label + " thresholds differ from physical catalogue strata")
                    errors.extend(f"{label} cap{cap}: " + message for message in audit_reserved_stage(stage, cap, roots))
                    if label == "new":
                        errors.extend(f"new cap{cap}: " + message for message in audit_dual_stage(stage, len(state["queue"]), previous))
                    previous = stage
                routed = state[field]
                _, extra, quality = audit_frontier_confirmation_router(routed, state, protocol, model, row["lanes"], label, expected)
                errors.extend(label + " routed: " + message for message in extra)
                qualities[state["source"]].append(quality)
                if not quality:
                    nonclosures.append({**pointer, "label": label, "status": routed["status"], "reference_complete": expected is not None})
                for name, cap in (("initial_lp", 144), ("final_lp", 256 if routed["resumed"] else 144)):
                    errors.extend(label + " routed: " + message for message in audit_reserved_stage(routed[name], cap, roots))
                if label == "new":
                    errors.extend(audit_dual_stage(routed["initial_lp"], len(state["queue"])))
                    errors.extend(audit_dual_stage(routed["final_lp"], len(state["queue"]), routed["initial_lp"]))
            for position, cap in enumerate(state["caps"]):
                a, b = (selection_from_record(state["traces"][label][str(cap)]["selection"]) for label in ("old", "new"))
                old_score, new_score = selection_score(a, tuple(state["queue"])), selection_score(b, tuple(state["queue"]))
                item = {**pointer, "budget_position": position, "lp_budget": cap,
                    "old_score": str(old_score), "new_score": str(new_score), "score_regression": new_score < old_score,
                    "old_canonical_exact": a.canonical_exact, "new_canonical_exact": b.canonical_exact}
                comparisons.append(item)
                if new_score < old_score:
                    losses.append(item)
            scores = [selection_score(selection_from_record(state[field]["selection"]), tuple(state["queue"])) for field in ("old_routed", "routed")]
            if scores[1] < scores[0]:
                routed_losses.append({**pointer, "old_score": str(scores[0]), "new_score": str(scores[1])})
            routed_increase, native_increase = _increases(state), [item for item in _native_cost([state]) if not item["cost_nonincrease"]]
            if routed_increase or native_increase:
                increases.append({**pointer, "routed": routed_increase, "native": native_increase})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            cell_entries.append(state)
        groups = {}
        for source in SOURCES:
            selected = [state for state in cell_entries if state["source"] == source]
            work, native = _work(selected), _native_cost(selected)
            groups[source] = {"state_count": len(selected), "work": work, "routed_cost_nonincrease": _cost(work),
                              "native_per_budget": native, "native_cost_nonincrease": all(item["cost_nonincrease"] for item in native)}
        cells.append({**point, "groups": groups})
    groups = {}
    for source in SOURCES:
        selected = [state for state in entries if state["source"] == source]
        work = _work(selected)
        a, b = (work[label] for label in ("routed_old", "routed_new"))
        groups[source] = {"state_count": len(selected), "work": work,
            "quality_gate_passed": len(qualities[source]) == len(selected) * 2 and all(qualities[source]),
            "finite_budget_gate_passed": not any(item["source"] == source for item in losses), "per_budget": _budget_summary([item for item in comparisons if item["source"] == source]),
            "routed_cost_nonincrease_in_every_cell": all(cell["groups"][source]["routed_cost_nonincrease"] for cell in cells),
            "native_cost_nonincrease_in_every_cell": all(cell["groups"][source]["native_cost_nonincrease"] for cell in cells),
            "native_per_budget": _native_cost(selected), "strict_pooled_lp_or_integer_reduction": bool(selected and (
                b["search_work"]["lp_calls"] < a["search_work"]["lp_calls"] or b["shared_integer_states"] < a["shared_integer_states"]))}
    fresh, hist, failure = (groups[source] for source in SOURCES[:3])
    valid = not failures
    quality = bool(fresh["state_count"] and all(group["quality_gate_passed"] for group in groups.values()))
    costs = {source: group["routed_cost_nonincrease_in_every_cell"] and group["native_cost_nonincrease_in_every_cell"] for source, group in groups.items()}
    cost = bool(fresh["state_count"] and fresh["strict_pooled_lp_or_integer_reduction"] and all(costs[source] for source in SOURCES[:3]))
    advance = valid and complete and quality and not losses and not routed_losses and cost
    executed = _work(entries)
    new_refs = [state for state in all_states if state["source"] in ("fresh_confirmation", "new_zero")]
    borrowed = [state for state in all_states if state["source"] in ("historical_regression", "failure_regression")]
    return {"decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_ADVANCE" if advance else "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_HOLD",
        "state_count": count, "fresh_state_count": fresh["state_count"], "regression_state_count": hist["state_count"],
        "failure_regression_count": failure["state_count"], "zero_state_count": groups["new_zero"]["state_count"],
        "executed_selector_calls": count * 2, "executed_native_advance_calls": count * 8, "executed_interface_calls": count * 10,
        "native_search_constructions": count * 4, "reference_native_search_constructions": sum(row["new_reference_calls"] for row in rows),
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows), "new_reference_oracle_calls": sum(row["new_reference_oracle_calls"] for row in rows),
        "new_old_routed_control_calls": sum(row["new_old_routed_calls"] for row in rows), "borrowed_reference_count": len(borrowed),
        "effective_references_complete": complete, "execution_checks_passed": valid, "routed_quality_gate_passed": quality,
        "finite_budget_gate_passed": not losses, "final_score_gate_passed": not routed_losses, "work_gate_passed": cost,
        "fresh_work_gate_passed": costs["fresh_confirmation"] and fresh["strict_pooled_lp_or_integer_reduction"],
        "historical_work_gate_passed": costs["historical_regression"], "failure_regression_work_gate_passed": costs["failure_regression"],
        "confirmation_gate_passed": advance, "failures": failures, "routed_nonclosures": nonclosures,
        "finite_budget_score_regressions": losses, "routed_score_regressions": routed_losses, "work_increases": increases,
        "per_budget": _budget_summary(comparisons), "groups": groups, "cells": cells,
        "all_executed_trace_and_routed_work": executed,
        "executed_search_work": {key: sum(executed[label]["search_work"][key] for label in ("kernel_old", "kernel_new", "routed_old", "routed_new")) for key in COUNTERS},
        "executed_dual_reuse_work": executed["executed_dual_reuse"], "executed_frontier_export_work": executed["frontier_export_work"],
        "reference_work": {"new": _combine_metrics(state["reference_metrics"] for state in new_refs),
                           "borrowed": _combine_metrics(state["reference_metrics"] for state in borrowed)},
        "new_reference_oracle_work": _oracle_work(new_refs), "borrowed_reference_oracle_work": _oracle_work(borrowed),
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
