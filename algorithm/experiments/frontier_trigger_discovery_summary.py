"""Audit frozen native-only discovery and its complete conditional validation."""
from __future__ import annotations

from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS, _dual_work, audit_dual_stage
from algorithm.experiments.frontier_score_proof_confirmation_summary import summarize as _validation_summary
from algorithm.experiments.root_pruning_development_summary import _native_work
from algorithm.experiments.strict_score_proof_timing_summary import _same

STATE_FIELDS = ("state_id", "queue", "seed", "index", "panel", "cohort")


def _thresholds(model):
    minima = {}
    for option in model["options"]:
        key = tuple(option["requirements"])
        minima[key] = min(minima.get(key, float("inf")), max(1.0, option["duration"]))
    return sorted(set(minima.values()))


def _runtime_errors(record, protocol):
    return ["runtime affinity or GC differs from registration"] if any(
        record[key]["cpu_affinity"] != [protocol["cpu"]] or not record[key]["gc_enabled"]
        for key in ("runtime_start", "runtime_end")) else []


def summarize(protocol, discovery, validation, rows):
    failures, rounds, stages, hits, screened = [], [], [], {}, {}
    nodes = [model["node"] for model in protocol["models"]]
    models = {model["node"]: model for model in protocol["models"]}
    options = {node: tuple(SimpleNamespace(**option) for option in model["options"])
               for node, model in models.items()}
    thresholds = {node: _thresholds(model) for node, model in models.items()}
    lanes, block = protocol["lanes"], protocol["block_size"]
    panels = protocol["queue_panels"]
    registered_queues = [tuple(state["queue"]) for panel in panels for state in panel["states"]]
    excluded = set(map(tuple, protocol["excluded_static_queues"][str(lanes)]))
    errors = _runtime_errors(discovery, protocol)
    if discovery["registered_at_utc"] != protocol["registered_at_utc"]:
        errors.append("discovery registration differs from its frozen protocol")
    if ([panel["seed"] for panel in panels] != list(protocol["panel_seeds"])
            or any(len(panel["states"]) != protocol["queues_per_panel"] for panel in panels)
            or protocol["queues_per_panel"] != block * protocol["max_rounds"]
            or protocol["expected_max_screen_calls"] != len(registered_queues) * len(nodes)):
        errors.append("seed panels or maximum complete-round screening coverage differ")
    if len(set(registered_queues)) != len(registered_queues) or set(registered_queues) & excluded:
        errors.append("registered discovery queues must be distinct and exclude all prior queues")
    if any(state["seed"] != panel["seed"] or state["panel"] != panel["panel"]
           or state["index"] != index or state["cohort"] != "conditional_trigger_validation"
           for panel in panels for index, state in enumerate(panel["states"])):
        errors.append("screen seed, panel, index or conditional cohort differs from its queue design")
    if errors:
        failures.append({"phase": "discovery", "errors": errors})
    first_sufficient = None
    for index, record in enumerate(discovery["rounds"]):
        expected = [{**state, "node": node, "lanes": lanes, "round_index": index}
                    for panel in panels for state in panel["states"][index * block:(index + 1) * block]
                    for node in nodes]
        actual = [{key: state[key] for key in (*STATE_FIELDS, "node", "lanes", "round_index")}
                  for state in record["states"]]
        if record["round_index"] != index or not _same(actual, expected):
            failures.append({"phase": "discovery", "round_index": index,
                             "errors": ["each full round must follow the frozen seed/queue/model order"]})
        for state in record["states"]:
            stage, node = state["stage"], state["node"]
            selected = selection_from_record(stage["selection"])
            checks = check_selection(selected, options[node], tuple(state["queue"]), lanes, protocol["lp_node_budget"])
            errors = [key for key, failed in checks.items() if failed]
            errors.extend(audit_reserved_stage(stage, protocol["lp_node_budget"], len(thresholds[node])))
            errors.extend(audit_dual_stage(stage, len(state["queue"])))
            if stage["thresholds"] != thresholds[node]:
                errors.append("screen strata differ from the cached physical catalogue")
            if errors:
                failures.append({"phase": "discovery", "node": node, "state_id": state["state_id"],
                                 "queue": state["queue"], "round_index": index, "errors": errors})
            stages.append(stage)
            screened[node, state["state_id"]] = stage
            if not selected.score_exact:
                hit = hits.setdefault(state["state_id"], {**{key: state[key] for key in STATE_FIELDS}, "trigger_nodes": []})
                hit["trigger_nodes"].append(node)
        distinct = {tuple(state["queue"]) for state in hits.values()}
        seeds = {state["seed"] for state in hits.values()}
        enough = len(distinct) >= protocol["min_trigger_queues"] and len(seeds) >= protocol["min_trigger_seeds"]
        if enough and first_sufficient is None:
            first_sufficient = index + 1
        rounds.append({"round_index": index, "screen_calls": len(record["states"]),
            "cumulative_trigger_queues": len(distinct), "cumulative_trigger_seeds": len(seeds),
            "trigger_threshold_met": enough})
    reason = "two_seed_trigger_threshold" if first_sufficient is not None else "screen_limit_reached"
    if (not discovery["rounds"] or len(discovery["rounds"]) != (first_sufficient or protocol["max_rounds"])
            or discovery["stop_reason"] != reason):
        failures.append({"phase": "discovery", "errors": ["screening must stop at the first sufficient full round or the frozen limit"]})
    triggers = list(hits.values())
    if not _same(discovery["trigger_states"], triggers):
        failures.append({"phase": "discovery", "errors": ["all and only native256 score-unproved queues and source nodes must be locked"]})
    trigger_count = len({tuple(state["queue"]) for state in triggers})
    seed_count = len({state["seed"] for state in triggers})
    sufficient = trigger_count >= protocol["min_trigger_queues"] and seed_count >= protocol["min_trigger_seeds"]

    errors = []
    if any(not _same(validation[key], value) for key, value in protocol["validation_template"].items()):
        errors.append("conditional validation differs from its frozen model, source, budget or reference template")
    if (validation["discovery_registered_at_utc"] != protocol["registered_at_utc"]
            or not _same(validation["trigger_states"], discovery["trigger_states"])
            or validation["screen_stop_reason"] != discovery["stop_reason"]):
        errors.append("locked validation is not bound to the completed discovery and its full trigger roster")
    expected_cells = [{"node": node, "lanes": lanes, "states": [
        {**state, "source": "fresh_confirmation", "historical_regression": False,
         "caps": [len(thresholds[node]), len(thresholds[node]) + 1,
                  protocol["initial_lp_budget"], protocol["lp_node_budget"]]}
        for state in triggers]} for node in nodes]
    if not _same(validation["cells"], expected_cells):
        errors.append("every discovered trigger must retain every catalogue with its original conditional metadata")
    if errors:
        failures.append({"phase": "validation", "errors": errors})
    audit = _validation_summary(validation, rows)
    for row in rows:
        for state in row["states"]:
            native = state["traces"]["new"].get(str(protocol["lp_node_budget"]))
            if not _same(native, screened.get((row["node"], state["state_id"]))):
                failures.append({"phase": "validation", "node": row["node"], "state_id": state["state_id"],
                    "queue": state["queue"], "errors": ["independent validation candidate256 differs from its discovery screen"]})
    failures.extend({"phase": "validation", **failure} for failure in audit["failures"])
    valid = not failures
    count = audit["state_count"]
    complete = audit["effective_references_complete"]
    quality = audit["routed_quality_gate_passed"] if count else True
    advance = valid and complete and quality and sufficient
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "FRONTIER_TRIGGER_VALIDATION_HOLD" if not quality else "TRIGGER_INSUFFICIENT" if not sufficient else
                "FRONTIER_TRIGGER_VALIDATION_ADVANCE")
    screen_work = {**_native_work(stages), "dual_reuse": _dual_work(stages)}
    comparison = {key: audit[key] for key in ("finite_budget_gate_passed", "final_score_gate_passed", "work_gate_passed")}
    audit.pop("decision")
    audit.pop("confirmation_gate_passed")
    return {"decision": decision, "cohort": "conditional_trigger_validation", "execution_checks_passed": valid,
        "trigger_sufficiency_passed": sufficient, "effective_references_complete": complete,
        "routed_quality_gate_passed": quality, "conditional_validation_gate_passed": advance,
        "screened_queue_count": len({tuple(state["queue"]) for record in discovery["rounds"] for state in record["states"]}),
        "screen_state_count": len(stages), "screen_round_count": len(rounds), "screen_stop_reason": discovery["stop_reason"],
        "screen_score_unproved_count": sum(not stage["selection"]["score_exact"] for stage in stages),
        "trigger_queue_count": trigger_count, "trigger_seed_count": seed_count, "trigger_states": triggers,
        "validation_state_count": count, "state_count": count, "rounds": rounds, "failures": failures,
        "screen_native_advance_calls": len(stages), "screen_native_search_constructions": len(stages),
        "screen_new_reference_calls": 0, "screen_new_reference_oracle_calls": 0, "screen_integer_probe_calls": 0,
        "validation_selector_calls": audit["executed_selector_calls"],
        "validation_native_advance_calls": audit["executed_native_advance_calls"],
        "executed_interface_calls": len(stages) + audit["executed_interface_calls"],
        "native_search_constructions": len(stages) + audit["native_search_constructions"],
        "new_reference_calls": audit["new_reference_calls"], "new_reference_oracle_calls": audit["new_reference_oracle_calls"],
        "screen_work": screen_work, "validation_work": audit["all_executed_trace_and_routed_work"],
        "executed_search_work": {key: screen_work["search_work"][key] + audit["executed_search_work"][key] for key in COUNTERS},
        "executed_dual_reuse_work": {key: screen_work["dual_reuse"][key] + audit["executed_dual_reuse_work"][key] for key in DUAL_COUNTERS},
        "executed_frontier_export_work": audit["executed_frontier_export_work"],
        "reference_work": audit["reference_work"], "reference_oracle_work": audit["new_reference_oracle_work"],
        "comparative_diagnostics": comparison, "validation_audit": audit,
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
