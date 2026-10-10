"""Native-only fresh trigger discovery and independent residual-table validation."""
from __future__ import annotations

from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.axis_capacity_frontier_development_summary import audit_axis_capacity_proof, _work as _axis_work
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS, _dual_work, audit_dual_stage
from algorithm.experiments.frontier_score_proof_confirmation_summary import audit_frontier_confirmation_router
from algorithm.experiments.frontier_trigger_discovery_summary import STATE_FIELDS, _runtime_errors, _thresholds
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.residual_capacity_frontier_development_summary import (
    _merged_values, _original_proof, _work as _residual_work, audit_residual_completion,
)
from algorithm.experiments.resumable_probe_development_summary import _candidate_work
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _same


def _discovery_audit(protocol, discovery):
    failures, rounds, stages, hits, screened = [], [], [], {}, {}
    nodes = [model["node"] for model in protocol["models"]]
    models = {model["node"]: model for model in protocol["models"]}
    options = {node: tuple(SimpleNamespace(**option) for option in model["options"]) for node, model in models.items()}
    thresholds = {node: _thresholds(model) for node, model in models.items()}
    lanes, block, panels = protocol["lanes"], protocol["block_size"], protocol["queue_panels"]
    queues = [tuple(state["queue"]) for panel in panels for state in panel["states"]]
    errors = _runtime_errors(discovery, protocol)
    if discovery["registered_at_utc"] != protocol["registered_at_utc"]:
        errors.append("discovery registration differs from its frozen protocol")
    if ([panel["seed"] for panel in panels] != list(protocol["panel_seeds"])
            or any(len(panel["states"]) != protocol["queues_per_panel"] for panel in panels)
            or protocol["queues_per_panel"] != block * protocol["max_rounds"]
            or protocol["expected_max_screen_calls"] != len(queues) * len(nodes)):
        errors.append("seed panels or maximum complete-round screening coverage differ")
    if len(set(queues)) != len(queues) or set(queues) & set(map(tuple, protocol["excluded_static_queues"][str(lanes)])):
        errors.append("discovery queues must be distinct and exclude all previous observed queues")
    if any(state["seed"] != panel["seed"] or state["panel"] != panel["panel"] or state["index"] != index
           or state["cohort"] != "conditional_trigger_validation"
           for panel in panels for index, state in enumerate(panel["states"])):
        errors.append("screen metadata differs from its frozen seed/queue design")
    if errors:
        failures.append({"phase": "discovery", "errors": errors})
    first_sufficient = None
    for index, record in enumerate(discovery["rounds"]):
        expected = [{**state, "node": node, "lanes": lanes, "round_index": index}
            for panel in panels for state in panel["states"][index * block:(index + 1) * block] for node in nodes]
        actual = [{key: state[key] for key in (*STATE_FIELDS, "node", "lanes", "round_index")} for state in record["states"]]
        if record["round_index"] != index or not _same(actual, expected):
            failures.append({"phase": "discovery", "round_index": index,
                "errors": ["each full round must follow the frozen seed/queue/catalogue order"]})
        for state in record["states"]:
            stage, node = state["stage"], state["node"]
            selected = selection_from_record(stage["selection"])
            checks = check_selection(selected, options[node], tuple(state["queue"]), lanes, protocol["lp_node_budget"])
            errors = [key for key, failed in checks.items() if failed]
            errors.extend(audit_reserved_stage(stage, protocol["lp_node_budget"], len(thresholds[node])))
            errors.extend(audit_dual_stage(stage, len(state["queue"])))
            if stage["thresholds"] != thresholds[node]:
                errors.append("screen strata differ from the cached catalogue")
            if errors:
                failures.append({"phase": "discovery", "node": node, "state_id": state["state_id"], "errors": errors})
            stages.append(stage)
            screened[node, state["state_id"]] = stage
            if not selected.score_exact:
                hit = hits.setdefault(state["state_id"], {**{key: state[key] for key in STATE_FIELDS}, "trigger_nodes": []})
                hit["trigger_nodes"].append(node)
        distinct, seeds = {tuple(state["queue"]) for state in hits.values()}, {state["seed"] for state in hits.values()}
        enough = len(distinct) >= protocol["min_trigger_queues"] and len(seeds) >= protocol["min_trigger_seeds"]
        if enough and first_sufficient is None:
            first_sufficient = index + 1
        rounds.append({"round_index": index, "screen_calls": len(record["states"]),
            "cumulative_trigger_queues": len(distinct), "cumulative_trigger_seeds": len(seeds), "trigger_threshold_met": enough})
    reason = "two_seed_trigger_threshold" if first_sufficient is not None else "screen_limit_reached"
    if (not discovery["rounds"] or len(discovery["rounds"]) != (first_sufficient or protocol["max_rounds"])
            or discovery["stop_reason"] != reason):
        failures.append({"phase": "discovery", "errors": ["stop only at the first sufficient full round or the registered limit"]})
    triggers = list(hits.values())
    if not _same(discovery["trigger_states"], triggers):
        failures.append({"phase": "discovery", "errors": ["all native256 score-unproved queues and their source nodes must be locked"]})
    return {"failures": failures, "rounds": rounds, "stages": stages, "screened": screened,
            "triggers": triggers, "thresholds": thresholds}


def _routed_work(states, *, prior=False):
    if prior:
        views = [{**state, "old_routed": state["prior_routed"]["old"], "routed": state["prior_routed"]["new"]} for state in states]
        work = _axis_work(views)
    else:
        work = _residual_work(states)
    return {key: work[key] for key in ("routed_old", "routed_new", "frontier_export_work", "probe_stage_counts")}


def _native(states):
    return {label: {**_native_work([state["traces"][label]["256"] for state in states]),
        **({"dual_reuse": _dual_work([state["traces"][label]["256"] for state in states])} if label == "new" else {})}
        for label in ("old", "new")}


def _oracle_work(states):
    oracles = [state["oracle"] for state in states if state["oracle"] is not None]
    return {"calls": len(oracles), "work": {key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}}


def summarize(protocol, discovery, validation, rows):
    screen = _discovery_audit(protocol, discovery)
    failures, nonclosures, prior_nonclosures, regressions, comparisons = screen["failures"], [], [], [], []
    models = {model["node"]: model for model in protocol["models"]}
    triggers, lanes = screen["triggers"], protocol["lanes"]
    errors = []
    if any(not _same(validation[key], value) for key, value in protocol["validation_template"].items()):
        errors.append("conditional validation differs from its frozen source/model/budget/reference template")
    if (validation["discovery_registered_at_utc"] != protocol["registered_at_utc"]
            or not _same(validation["trigger_states"], discovery["trigger_states"])
            or validation["screen_stop_reason"] != discovery["stop_reason"]):
        errors.append("locked validation must retain the completed discovery and every trigger queue")
    controls_by_node = {cell["node"]: cell["states"] for cell in protocol["validation_template"]["control_cells"]}
    expected_cells = [{"node": node, "lanes": lanes, "states": [
        {**state, "source": "fresh_confirmation", "historical_regression": False,
         "caps": [len(screen["thresholds"][node]), len(screen["thresholds"][node]) + 1,
                  protocol["initial_lp_budget"], protocol["lp_node_budget"]]} for state in triggers]
        + controls_by_node[node]} for node in models]
    if not _same(validation["cells"], expected_cells):
        errors.append("all locked fresh queues and all frozen controls must retain every catalogue")
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in expected_cells]:
        errors.append("all registered validation cells must occur once in order")
    if errors:
        failures.append({"phase": "validation", "errors": errors})
    states = [state for row in rows for state in row["states"]]
    fresh = [state for state in states if state["source"] == "fresh_confirmation"]
    controls = [state for state in states if state["source"] != "fresh_confirmation"]
    count, fresh_count, control_count = len(states), len(fresh), len(controls)
    counts = {"expected_state_count": count, "expected_fresh_state_count": fresh_count,
        "expected_control_state_count": control_count, "expected_prior_axis_selector_calls": 2 * fresh_count,
        "expected_residual_selector_calls": 2 * count,
        "expected_native_advance_calls": 8 * fresh_count, "expected_interface_calls": 12 * fresh_count + 2 * control_count,
        "expected_native_constructions": 6 * fresh_count + 2 * control_count,
        "expected_new_reference_calls": fresh_count, "expected_borrowed_reference_count": control_count,
        "expected_borrowed_native_snapshots": 8 * control_count, "expected_borrowed_prior_axis_count": 2 * control_count}
    if any(validation[key] != value for key, value in counts.items()):
        failures.append({"phase": "validation", "errors": ["fresh/borrowed coverage or executed method counts differ from registration"]})
    complete, qualities, cell_results = True, [], []
    for cell, row in zip(validation["cells"], rows):
        model, point = models[row["node"]], {"node": row["node"], "lanes": row["lanes"]}
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        thresholds = screen["thresholds"][row["node"]]
        row_fresh = [state for state in row["states"] if state["source"] == "fresh_confirmation"]
        errors = _runtime_errors(row, validation)
        if (row["registered_at_utc"] != validation["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options)):
            errors.append("validation registration or catalogue metadata differs")
        if (len(row["states"]) != len(cell["states"]) or any(not _same(
                {key: state.get(key) for key in registered}, registered) for state, registered in zip(row["states"], cell["states"]))):
            errors.append("validation state metadata or borrowed control pointers differ from the locked roster")
        if (row["new_reference_calls"] != len(row_fresh)
                or row["new_reference_oracle_calls"] != sum(state["oracle"] is not None for state in row_fresh)
                or row["new_prior_axis_selector_calls"] != 2 * len(row_fresh)
                or row["new_residual_selector_calls"] != 2 * len(row["states"])
                or row["executed_native_advance_calls"] != 8 * len(row_fresh)
                or row["fresh_native_search_constructions"] != 2 * len(row_fresh)
                or row["borrowed_reference_count"] != len(row["states"]) - len(row_fresh)
                or row["borrowed_native_snapshots"] != 8 * (len(row["states"]) - len(row_fresh))
                or row["borrowed_prior_axis_count"] != 2 * (len(row["states"]) - len(row_fresh))):
            errors.append("actual new reference/oracle/axis-control/residual calls differ from fresh and borrowed policy")
        if errors:
            failures.append({"phase": "validation", **point, "errors": errors})
        for state in row["states"]:
            pointer = {**point, "state_id": state["state_id"], "queue": state["queue"], "cohort": state["cohort"], "seed": state.get("seed")}
            errors, expected = [], None
            is_fresh = state["source"] == "fresh_confirmation"
            if (is_fresh and (state["cohort"] != "conditional_trigger_validation" or state["historical_regression"])
                    or tuple(state["caps"]) != (len(thresholds), len(thresholds) + 1, 144, 256)):
                errors.append("fresh conditional cohort or native ladder differs from registration")
            if is_fresh and state["previous_residual_routed"] is not None:
                errors.append("fresh validation must not borrow a previous residual outcome")
            reference = selection_from_record(state["reference"])
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, validation, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            metric = state["reference_metrics"]
            if (state["reference"]["lp_node_budget"] != validation["reference_budget"] or metric["failures"]
                    or metric["reference_comparable_count"] != int(state["reference_complete"])
                    or any(metric["search_work"][key] != state["reference"][key] for key in COUNTERS)):
                errors.append("independent reference budget, certificate or actual paid metrics differ")
            errors.extend("reference: " + key for key, failed in check_selection(reference, options, tuple(state["queue"]),
                lanes, validation["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            if not all(set(state["traces"][label]) == {str(cap) for cap in state["caps"]} for label in ("old", "new")):
                failures.append({"phase": "validation", **pointer, "errors": errors + ["all eight native snapshots are required"]})
                continue
            if is_fresh and not _same(state["traces"]["new"]["256"], screen["screened"].get((row["node"], state["state_id"]))):
                errors.append("independent fresh native256 differs from the discovery screen")
            for label, field in (("old", "old_routed"), ("new", "routed")):
                trace, previous = state["traces"][label], None
                errors.extend(label + ": " + message for message in _trace_errors(trace, state, options, lanes, expected, reference, True))
                for cap in state["caps"]:
                    stage = trace[str(cap)]
                    if stage["thresholds"] != thresholds:
                        errors.append(label + " native thresholds differ from the catalogue")
                    errors.extend(label + ": " + message for message in audit_reserved_stage(stage, cap, len(thresholds)))
                    if label == "new":
                        errors.extend(label + ": " + message for message in audit_dual_stage(stage, len(state["queue"]), previous))
                    previous = stage
                current, prior = state[field], state["prior_routed"][label]
                for role, result in (("axis_prior", prior), ("residual", current)):
                    _, extra, quality = audit_frontier_confirmation_router(result, state, validation, model, lanes, label, expected)
                    errors.extend(f"{label} {role}: " + message for message in extra)
                    if result["probe_stage"] == "frontier_late" and result["probe"] is not None and result["frontier_evidence"] is not None:
                        errors.extend(f"{label} {role}: " + message for message in audit_axis_capacity_proof(result["probe"], result["frontier_evidence"], state["queue"], lanes))
                    if role == "residual":
                        qualities.append(quality)
                        if not quality:
                            nonclosures.append({**pointer, "label": label, "status": result["status"], "reference_complete": expected is not None})
                    elif not quality:
                        prior_nonclosures.append({**pointer, "label": label, "status": result["status"]})
                if (not _same(current["initial_lp"], prior["initial_lp"]) or not _same(current["final_lp"], prior["final_lp"])
                        or current["resumed"] != prior["resumed"] or not _same(current["frontier_evidence"], prior["frontier_evidence"])):
                    errors.append(label + " residual router changed paid LP/resumption/frontier evidence from its axis predecessor")
                if prior["probe_stage"] != "frontier_late":
                    if not _same(current, prior):
                        errors.append(label + " ordinary route must reproduce its entire axis predecessor")
                elif current["probe"] is not None:
                    probe, prior_probe = current["probe"], prior["probe"]
                    capacity = probe["score_proof"]["capacity"]
                    if not _same(_original_proof(probe["score_proof"]), prior_probe["score_proof"]):
                        errors.append(label + " retained tables changed original axis/DFS proof arithmetic")
                    if capacity.get("merged_values_copied") != len(capacity["bounds"]) * (lanes + 1):
                        errors.append(label + " retained merged vectors must charge every copied value once")
                    for bound in capacity["bounds"]:
                        if not _same(bound.get("exact_slot_numerators"), _merged_values(bound["axis_service_profiles"], state["queue"], lanes)):
                            errors.append(label + " retained exact-slot vector differs from the paid max-plus merge")
                    errors.extend(label + ": " + message for message in audit_residual_completion(probe, prior_probe, model, state["queue"], lanes))
                if not is_fresh and not _same(current, state["previous_residual_routed"][label]):
                    errors.append(label + " frozen control must reproduce its complete previous residual output")
                a, b = (_candidate_work([result]) for result in (prior, current))
                if (any(a["search_work"][key] != b["search_work"][key] for key in ("lp_calls", "lp_matrix_coefficients"))
                        or b["shared_integer_states"] > a["shared_integer_states"]):
                    regressions.append({**pointer, "label": label, "prior_integer_states": a["shared_integer_states"],
                        "actual_integer_states": b["shared_integer_states"], "prior_search_work": a["search_work"], "actual_search_work": b["search_work"]})
            for cap in state["caps"]:
                scores = [selection_score(selection_from_record(state["traces"][label][str(cap)]["selection"]), tuple(state["queue"])) for label in ("old", "new")]
                comparisons.append({**pointer, "lp_budget": cap, "old_score": str(scores[0]), "new_score": str(scores[1]), "score_regression": scores[1] < scores[0]})
            if errors:
                failures.append({"phase": "validation", **pointer, "errors": errors})
        cell_results.append({**point, "state_count": len(row["states"]), "residual_work": _routed_work(row["states"]),
            "axis_prior_work": _routed_work(row["states"], prior=True)})
    reuse = [state for state in fresh if any(state[field]["probe"] is not None
        and state[field]["probe"]["completion"].get("tables_reused", 0) > 0 for field in ("old_routed", "routed"))]
    trigger_count, trigger_seeds = len({tuple(state["queue"]) for state in triggers}), len({state["seed"] for state in triggers})
    reuse_count, reuse_seeds = len({tuple(state["queue"]) for state in reuse}), len({state["seed"] for state in reuse})
    raw_sufficient = trigger_count >= protocol["min_trigger_queues"] and trigger_seeds >= protocol["min_trigger_seeds"]
    reuse_sufficient = reuse_count >= protocol["min_residual_queues"] and reuse_seeds >= protocol["min_residual_seeds"]
    valid, quality, work_gate = not failures, len(qualities) == count * 2 and all(qualities), not regressions
    advance = valid and complete and quality and work_gate and raw_sufficient and reuse_sufficient
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
        "RESIDUAL_CAPACITY_TRIGGER_CONFIRMATION_HOLD" if not quality or not work_gate else
        "TRIGGER_INSUFFICIENT" if not raw_sufficient or not reuse_sufficient else "RESIDUAL_CAPACITY_TRIGGER_CONFIRMATION_ADVANCE")
    native, borrowed_native = _native(fresh), _native(controls)
    residual, axis_fresh, axis_borrowed = _routed_work(states), _routed_work(fresh, prior=True), _routed_work(controls, prior=True)
    screen_work = {**_native_work(screen["stages"]), "dual_reuse": _dual_work(screen["stages"])}
    paid = [screen_work, *native.values(), *(axis_fresh[field] for field in ("routed_old", "routed_new")),
            *(residual[field] for field in ("routed_old", "routed_new"))]
    groups = {}
    for cohort in dict.fromkeys(state["cohort"] for state in states):
        selected = [state for state in states if state["cohort"] == cohort]
        new = [state for state in selected if state["source"] == "fresh_confirmation"]
        borrowed = [state for state in selected if state["source"] != "fresh_confirmation"]
        groups[cohort] = {"state_count": len(selected), "residual_work": _routed_work(selected),
            "fresh_axis_router_work": _routed_work(new, prior=True), "borrowed_axis_router_work": _routed_work(borrowed, prior=True),
            "fresh_native_work": _native(new), "borrowed_native_work": _native(borrowed),
            "quality_gate_passed": not any(item["cohort"] == cohort for item in nonclosures),
            "predecessor_work_gate_passed": not any(item["cohort"] == cohort for item in regressions)}
    reuse_by_arm = {}
    for label, field in (("old", "old_routed"), ("new", "routed")):
        used = [state for state in fresh if state[field]["probe"] is not None
                and state[field]["probe"]["completion"].get("tables_reused", 0) > 0]
        reuse_by_arm[label] = {"queue_count": len({tuple(state["queue"]) for state in used}),
                              "seed_count": len({state["seed"] for state in used})}
    return {"decision": decision, "cohort": "conditional_trigger_validation", "state_count": count,
        "validation_state_count": count, "fresh_state_count": fresh_count, "control_state_count": control_count,
        "execution_checks_passed": valid, "effective_references_complete": complete,
        "routed_quality_gate_passed": quality, "predecessor_work_gate_passed": work_gate,
        "raw_trigger_sufficiency_passed": raw_sufficient, "reuse_trigger_sufficiency_passed": reuse_sufficient,
        "conditional_confirmation_gate_passed": advance, "failures": failures, "routed_nonclosures": nonclosures,
        "axis_prior_nonclosures": prior_nonclosures, "predecessor_work_regressions": regressions,
        "screened_queue_count": len({tuple(state["queue"]) for record in discovery["rounds"] for state in record["states"]}),
        "screen_state_count": len(screen["stages"]), "screen_round_count": len(screen["rounds"]),
        "screen_stop_reason": discovery["stop_reason"], "rounds": screen["rounds"], "trigger_states": triggers,
        "trigger_queue_count": trigger_count, "trigger_seed_count": trigger_seeds,
        "reused_table_queue_count": reuse_count, "reused_table_seed_count": reuse_seeds, "reuse_by_arm": reuse_by_arm,
        "screen_native_advance_calls": len(screen["stages"]), "screen_native_search_constructions": len(screen["stages"]),
        "screen_new_reference_calls": 0, "screen_new_reference_oracle_calls": 0, "screen_integer_probe_calls": 0,
        "validation_selector_calls": 4 * fresh_count + 2 * control_count,
        "validation_prior_axis_selector_calls": 2 * fresh_count, "validation_residual_selector_calls": 2 * count,
        "validation_native_advance_calls": 8 * fresh_count,
        "executed_interface_calls": len(screen["stages"]) + counts["expected_interface_calls"],
        "native_search_constructions": len(screen["stages"]) + counts["expected_native_constructions"],
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_reference_oracle_calls": sum(row["new_reference_oracle_calls"] for row in rows),
        "reference_native_search_constructions": fresh_count, "borrowed_reference_count": control_count,
        "borrowed_native_snapshots": 8 * control_count, "borrowed_axis_router_count": 2 * control_count,
        "borrowed_residual_router_count": 2 * control_count, "screen_work": screen_work,
        "fresh_native_work": native, "borrowed_native_work": borrowed_native,
        "fresh_axis_router_work": axis_fresh, "borrowed_axis_router_work": axis_borrowed,
        "residual_router_work": residual, "cells": cell_results, "groups": groups,
        "native_score_comparisons": comparisons, "native_score_regressions": [item for item in comparisons if item["score_regression"]],
        "executed_search_work": {key: sum(item["search_work"][key] for item in paid) for key in COUNTERS},
        "executed_dual_reuse_work": {key: screen_work["dual_reuse"][key] + native["new"]["dual_reuse"][key]
            + axis_fresh["routed_new"]["dual_reuse"][key] + residual["routed_new"]["dual_reuse"][key] for key in DUAL_COUNTERS},
        "reference_work": {"new": _combine_metrics(state["reference_metrics"] for state in fresh),
                           "borrowed": _combine_metrics(state["reference_metrics"] for state in controls)},
        "reference_oracle_work": {"new": _oracle_work(fresh), "borrowed": _oracle_work(controls)},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
