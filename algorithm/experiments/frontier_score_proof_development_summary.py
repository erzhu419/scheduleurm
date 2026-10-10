"""Paid-frontier score proof with explicit export work and predecessor costs."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.branch_reserved_development_summary import _identity, _key, audit_reserved_stage
from algorithm.experiments.certificate_routed_development_summary import audit_candidate
from algorithm.experiments.dual_reuse_development_summary import _cost, _dual_work, audit_dual_stage
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _candidate_work, _metric, _exact
from algorithm.experiments.root_pruning_development_summary import _native_work
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _audit_result as _audit_strict, _same

EXPORT_COUNTERS = ("heap_entries_read", "group_ids_copied", "requirement_rows_copied", "requirement_coordinates_copied",
                   "box_count_values_copied", "exact_duration_positions_copied", "representative_options_copied", "fraction_pairs_copied")


def audit_frontier_evidence(evidence, stage, model, queue, lanes):
    errors = []
    selected = stage["selection"]
    score = selection_score(selection_from_record(selected), tuple(queue))
    if Fraction(*evidence["incumbent_score_fraction"]) != score:
        errors.append("frontier proof target differs from its own paid incumbent score")
    if (evidence["root_count"] != len(stage["thresholds"])
            or evidence["live_region_count"] != selected["frontier_regions_remaining"]
            or evidence["live_tie_region_count"] != selected["canonical_tie_regions_remaining"]
            or len(evidence["regions"]) != evidence["live_region_count"] - evidence["live_tie_region_count"]
            or evidence["raw_region_count"] < evidence["live_region_count"]):
        errors.append("exported strict regions or live/root metadata differ from the paid frontier")
    groups = {}
    for index, option in enumerate(model["options"]):
        groups.setdefault(tuple(option["requirements"]), []).append(index)
    requirements = list(groups)
    durations = [min(Fraction.from_float(float(max(1.0, model["options"][index]["duration"]))) for index in members) for members in groups.values()]
    columns = positions = 0
    for region in evidence["regions"]:
        threshold, bound = Fraction(*region["threshold_fraction"]), Fraction(*region["bound_fraction"])
        if threshold not in set(durations) or bound <= score:
            errors.append("each exported box must use a catalogue stratum and an exact strict upper bound")
            continue
        representatives = {group: next(index for index in groups[requirement]
            if Fraction.from_float(float(max(1.0, model["options"][index]["duration"]))) <= threshold)
            for group, requirement in enumerate(requirements) if durations[group] <= threshold}
        eligible = sorted(representatives, key=representatives.get)
        expected_requirements = [requirements[group] for group in eligible]
        exact_positions = [position for position, group in enumerate(eligible) if durations[group] == threshold]
        if (not _same(region["eligible"], eligible) or not _same(region["requirements"], expected_requirements)
                or not _same(region["exact_duration_positions"], exact_positions)
                or not _same(region["representative_options"], [representatives[group] for group in eligible])):
            errors.append("exported groups, requirements, duration positions or representatives differ from the catalogue")
        lower, upper = region["lower"], region["upper"]
        if (len(lower) != len(eligible) or len(upper) != len(eligible)
                or any(type(value) is not int or value < 0 for value in lower + upper)
                or any(left > right or right > lanes for left, right in zip(lower, upper))):
            errors.append("exported lower and upper counts are invalid integer boxes")
        elif (sum(lower) > lanes or any(sum(count * requirement[j] for count, requirement in zip(lower, expected_requirements)) > waiting
                                       for j, waiting in enumerate(queue))):
            errors.append("exported mandatory lower counts exceed lanes or visible resources")
        columns += len(region["eligible"])
        positions += len(region["exact_duration_positions"])
    work = evidence["export_work"]
    expected_work = {"heap_entries_read": evidence["raw_region_count"], "group_ids_copied": columns,
        "requirement_rows_copied": columns, "requirement_coordinates_copied": columns * len(queue),
        "box_count_values_copied": columns * 2, "exact_duration_positions_copied": positions,
        "representative_options_copied": columns, "fraction_pairs_copied": 1 + 2 * len(evidence["regions"])}
    if any(type(work[key]) is not int or work[key] < 0 or work[key] != expected_work[key] for key in EXPORT_COUNTERS):
        errors.append("frontier export fees differ from the actual copied heap and geometry")
    return errors


def audit_frontier_router(result, original, prior, state, protocol, model, lanes, label, expected):
    trace = state["traces"][label]
    initial, final = original["initial_lp"], original["final_lp"]
    errors = []
    try:
        _, extra, _ = audit_candidate(original, {**state, "expected_initial_lp": trace["144"]}, protocol, model, lanes, expected)
        errors.extend("original: " + message for message in extra)
    except ValueError as error:
        errors.append("original: " + str(error))
    if not _same(initial, trace["144"]) or not _same(final, trace["256"] if original["resumed"] else trace["144"]):
        errors.append("original routed LP snapshots differ from the borrowed native evidence")
    if any(result[key] != protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")):
        errors.append("frontier router budgets differ from registration")
    if not _same(result["initial_lp"], initial) or not _same(result["final_lp"], final) or result["resumed"] != original["resumed"]:
        errors.append("frontier routing changed the original paid LP snapshots or resumption")
    late = not final["selection"]["score_exact"] and original["probe"] is None
    if not late:
        if result["frontier_evidence"] is not None or not _same({key: value for key, value in result.items() if key != "frontier_evidence"}, prior):
            errors.append("ordinary route must reproduce its complete preceding output with no frontier export or second probe")
    else:
        probe = result["probe"]
        evidence = result["frontier_evidence"]
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
                errors.append("frontier probe must reuse the paid final LP selection, preprocessing and capacity work")
            if not probe["score_proof"]["attempted"]:
                errors.append("frontier probe must attempt strict score proof on its unproved final LP snapshot")
            if evidence is not None and (probe["score_proof"]["thresholds_checked"] > len(evidence["regions"])
                    or probe["score_proof"]["status"] == "COMPLETE" and probe["score_proof"]["thresholds_checked"] != len(evidence["regions"])):
                errors.append("frontier proof did not check all and only its exported strict boxes")
            status = "FRONTIER_PROBE_COMPLETE" if probe["selection"]["canonical_exact"] else "FRONTIER_PROBE_INCOMPLETE"
            if result["status"] != status or not _same(result["selection"], probe["selection"]):
                errors.append("frontier returned selection or status differs from its actual integer certificate")
    budget = protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, result["final_lp"]["preprocessing"])
    if metric["failures"]:
        errors.append("returned routed action has invalid feasibility, certificate or paid work")
    return metric, errors, _exact(result["selection"], metric, expected)


def _work(states):
    result = {label: _candidate_work([state[field] for state in states])
              for label, field in (("routed_old", "old_routed"), ("routed_new", "routed"))}
    result["routed_new"]["dual_reuse"] = _dual_work([state["routed"]["final_lp"] for state in states])
    result["probe_stage_counts"] = {label: dict(Counter(state[field]["probe_stage"] for state in states))
                                    for label, field in (("old", "old_routed"), ("new", "routed"))}
    result["frontier_export_work"] = {label: {key: sum(state[field]["frontier_evidence"]["export_work"][key]
        for state in states if state[field]["frontier_evidence"] is not None) for key in EXPORT_COUNTERS}
        for label, field in (("old", "old_routed"), ("new", "routed"))}
    return result


def _increases(state):
    work = _work([state])
    a, b = work["routed_old"], work["routed_new"]
    old = {key: a["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} | {"shared_integer_states": a["shared_integer_states"]}
    new = {key: b["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients")} | {"shared_integer_states": b["shared_integer_states"]}
    return {key: {"old": old[key], "new": new[key]} for key in old if new[key] > old[key]}


def summarize(protocol, rows):
    failures, nonclosures, losses, increases, entries, cells, counterexamples, qualities, prior_regressions = [], [], [], [], [], [], [], [], []
    complete = True
    models = {model["node"]: model for model in protocol["models"]}
    all_states = [state for row in rows for state in row["states"]]
    count = len(all_states)
    counts = {"expected_state_count": count, "expected_selector_calls": count * 2, "expected_interface_calls": count * 2,
              "expected_native_constructions": count * 2, "expected_native_advance_calls": 0,
              "expected_borrowed_native_snapshots": count * 8,
              "expected_nonzero_state_count": sum(any(state["queue"]) for state in all_states),
              "expected_failure_development_count": sum(state["cohort"] == "failure_development" for state in all_states)}
    if any(protocol[key] != value for key, value in counts.items()):
        failures.append({"errors": ["executed roster or interface counts differ from registration"]})
    historical = {_key(item["node"], item["lanes"], item) for item in protocol["historical_regression_keys"]}
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    if len(historical) != len(protocol["historical_regression_keys"]) or not historical <= registered:
        failures.append({"errors": ["fixed historical keys must be unique and contained in the frozen roster"]})
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        failures.append({"errors": ["all registered cells must occur once in order"]})
    for cell, row in zip(protocol["cells"], rows):
        model = models[cell["node"]]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        point = {"node": row["node"], "lanes": row["lanes"]}
        identity = lambda state: (_identity(state), tuple(state["caps"]), tuple(state[key] for key in (
            "borrowed_from", "borrowed_index", "source_state_id", "source_record_source")))
        errors = []
        if list(map(identity, row["states"])) != list(map(identity, cell["states"])):
            errors.append("state roster or borrowed source pointers differ from registration")
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options)):
            errors.append("cell registration or catalogue metadata differs")
        if row["new_reference_calls"] != 0 or row["new_reference_oracle_calls"] != 0 or row["new_old_routed_calls"] != len(row["states"]):
            errors.append("references must be borrowed and both routed arms independently executed")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"] for key in ("runtime_start", "runtime_end")):
            errors.append("runtime affinity or GC differs from registration")
        if errors:
            failures.append({**point, "errors": errors})
        minima = {}
        for option in options:
            key = tuple(option.requirements)
            minima[key] = min(minima.get(key, float("inf")), max(1.0, option.duration))
        roots = len(set(minima.values()))
        cell_entries = []
        for state in row["states"]:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "source", "cohort", "historical_regression")}}
            errors, expected = [], None
            if state["historical_regression"] != (_key(row["node"], row["lanes"], state) in historical):
                errors.append("historical subgroup differs from its independently frozen keys")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("borrowed budget ladder differs from the catalogue and registered budgets")
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
                errors.append("borrowed reference has invalid budget or recomputed metrics")
            checks = check_selection(reference, options, tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected)
            errors.extend("reference: " + key for key, failed in checks.items() if failed)
            complete &= expected is not None
            if not all(set(state["traces"][label]) == {str(cap) for cap in state["caps"]} for label in ("old", "new")):
                failures.append({**pointer, "errors": errors + ["all eight borrowed native snapshots are required"]})
                continue
            state_quality = {}
            for label, field in (("old", "old_routed"), ("new", "routed")):
                routed = state[field]
                prior = state["prior_routed"][label]
                _, extra, quality = audit_frontier_router(routed, state["original_routed"][label], prior, state, protocol, model, row["lanes"], label, expected)
                errors.extend(label + ": " + message for message in extra)
                for name, cap in (("initial_lp", 144), ("final_lp", 256 if routed["resumed"] else 144)):
                    errors.extend(label + " " + name + ": " + message for message in audit_reserved_stage(routed[name], cap, roots))
                if label == "new":
                    errors.extend(audit_dual_stage(routed["initial_lp"], len(state["queue"])))
                    errors.extend(audit_dual_stage(routed["final_lp"], len(state["queue"]), routed["initial_lp"]))
                state_quality[label] = quality
                qualities.append(quality)
                if not quality:
                    nonclosures.append({**pointer, "label": label, "status": routed["status"], "reference_complete": expected is not None})
                a, b = (_candidate_work([result]) for result in (prior, routed))
                change = {"lp_unchanged": b["search_work"]["lp_calls"] == a["search_work"]["lp_calls"],
                          "matrix_unchanged": b["search_work"]["lp_matrix_coefficients"] == a["search_work"]["lp_matrix_coefficients"],
                          "integer_nonincrease": b["shared_integer_states"] <= a["shared_integer_states"]}
                if not all(change.values()):
                    prior_regressions.append({**pointer, "label": label, **change,
                        "prior_lp_calls": a["search_work"]["lp_calls"], "actual_lp_calls": b["search_work"]["lp_calls"],
                        "prior_matrix_coefficients": a["search_work"]["lp_matrix_coefficients"], "actual_matrix_coefficients": b["search_work"]["lp_matrix_coefficients"],
                        "prior_integer_states": a["shared_integer_states"], "actual_integer_states": b["shared_integer_states"]})
            scores = [selection_score(selection_from_record(state[field]["selection"]), tuple(state["queue"])) for field in ("old_routed", "routed")]
            if scores[1] < scores[0]:
                losses.append({**pointer, "old_score": str(scores[0]), "new_score": str(scores[1])})
            increase = _increases(state)
            if increase:
                increases.append({**pointer, "increases": increase})
            if state["cohort"] == "failure_development":
                counterexamples.append({**pointer, "quality": state_quality,
                    "original_status": {label: state["original_routed"][label]["status"] for label in ("old", "new")},
                    "prior_status": {label: state["prior_routed"][label]["status"] for label in ("old", "new")},
                    "prior_work": {label: _candidate_work([state["prior_routed"][label]]) for label in ("old", "new")},
                    "status": {label: state[field]["status"] for label, field in (("old", "old_routed"), ("new", "routed"))}, "work": _work([state])})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            cell_entries.append(state)
        nonzero = _work([state for state in cell_entries if any(state["queue"])])
        hist = _work([state for state in cell_entries if state["historical_regression"]])
        cells.append({**point, "nonzero_work": nonzero, "historical_work": hist,
                      "nonzero_cost_nonincrease": _cost(nonzero), "historical_cost_nonincrease": _cost(hist)})
    regular = [state for state in entries if any(state["queue"])]
    hist = [state for state in entries if state["historical_regression"]]
    work, historical_work, executed = _work(regular), _work(hist), _work(entries)
    a, b = work["routed_old"], work["routed_new"]
    strict = b["search_work"]["lp_calls"] < a["search_work"]["lp_calls"] or b["shared_integer_states"] < a["shared_integer_states"]
    nonzero_cost = bool(regular and strict and all(cell["nonzero_cost_nonincrease"] for cell in cells))
    hist_cost = all(cell["historical_cost_nonincrease"] for cell in cells)
    valid, quality = not failures, bool(entries and len(qualities) == 2 * len(entries) and all(qualities))
    prior_work = {label: _candidate_work([state["prior_routed"][label] for state in entries]) for label in ("old", "new")}
    prior_comparison = {label: {"lp_unchanged": executed[field]["search_work"]["lp_calls"] == prior_work[label]["search_work"]["lp_calls"],
        "matrix_unchanged": executed[field]["search_work"]["lp_matrix_coefficients"] == prior_work[label]["search_work"]["lp_matrix_coefficients"],
        "integer_nonincrease": executed[field]["shared_integer_states"] <= prior_work[label]["shared_integer_states"]}
        for label, field in (("old", "routed_old"), ("new", "routed_new"))}
    prior_ok = not prior_regressions and all(all(checks.values()) for checks in prior_comparison.values())
    failure_closure = len(counterexamples) == protocol["expected_failure_development_count"] and all(all(state["quality"].values()) for state in counterexamples)
    advance = valid and complete and quality and failure_closure and not losses and nonzero_cost and hist_cost and prior_ok
    borrowed_native = {label: _native_work([state["traces"][label]["256"] for state in entries]) for label in ("old", "new")}
    borrowed_native["new"]["dual_reuse"] = _dual_work([state["traces"]["new"]["256"] for state in entries])
    return {"decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "FRONTIER_SCORE_PROOF_DEVELOPMENT_ADVANCE" if advance else "FRONTIER_SCORE_PROOF_DEVELOPMENT_HOLD",
        "state_count": count, "nonzero_state_count": len(regular), "historical_state_count": len(hist), "failure_development_count": len(counterexamples),
        "borrowed_reference_count": count, "borrowed_native_snapshots": count * 8, "borrowed_original_routed_count": count * 2,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows), "new_reference_oracle_calls": sum(row["new_reference_oracle_calls"] for row in rows),
        "new_old_routed_control_calls": sum(row["new_old_routed_calls"] for row in rows), "executed_selector_calls": count * 2,
        "executed_interface_calls": count * 2, "executed_native_advance_calls": 0, "native_search_constructions": count * 2,
        "reference_native_search_constructions": 0, "effective_references_complete": complete, "execution_checks_passed": valid,
        "routed_quality_gate_passed": quality, "final_score_gate_passed": not losses, "nonzero_work_gate_passed": nonzero_cost,
        "historical_work_gate_passed": hist_cost, "strict_pooled_lp_or_integer_reduction": bool(strict),
        "work_gate_passed": nonzero_cost and hist_cost and prior_ok, "development_gate_passed": advance,
        "predecessor_work_gate_passed": prior_ok, "failure_development_gate_passed": failure_closure,
        "predecessor_work_regressions": prior_regressions, "borrowed_prior_routed_work": prior_work, "predecessor_work_comparison": prior_comparison,
        "failures": failures, "routed_nonclosures": nonclosures, "routed_score_regressions": losses, "work_increases": increases,
        "nonzero_work": work, "historical_work": historical_work, "all_executed_routed_work": executed,
        "executed_search_work": {key: sum(executed[label]["search_work"][key] for label in ("routed_old", "routed_new")) for key in COUNTERS},
        "executed_dual_reuse_work": executed["routed_new"]["dual_reuse"], "counterexample_states": counterexamples,
        "executed_frontier_export_work": executed["frontier_export_work"],
        "counterexample_work": _work([state for state in entries if state["cohort"] == "failure_development"]),
        "borrowed_native_work": borrowed_native, "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in all_states),
        "cells": cells, "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
