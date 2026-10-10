"""Whole-selector timing on the frozen residual development/confirmation roster."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations
import math
from statistics import median
from types import SimpleNamespace

from algorithm.axis_capacity_frontier_score_proof_frame_candidates import CAPACITY_COUNTERS as AXIS_COUNTERS
from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_integer_capacity_timing_summary import _latency
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.axis_capacity_frontier_development_summary import audit_axis_capacity_proof
from algorithm.experiments.branch_reserved_development_summary import _key, audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import _dual_work, audit_dual_stage
from algorithm.experiments.frontier_score_proof_confirmation_summary import audit_frontier_confirmation_router
from algorithm.experiments.frontier_score_proof_development_summary import EXPORT_COUNTERS
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.residual_capacity_frontier_development_summary import (
    RESIDUAL_COUNTERS, _merged_values, _original_proof, audit_residual_completion,
)
from algorithm.experiments.resumable_probe_development_summary import _candidate_work
from algorithm.experiments.root_pruning_development_summary import ROOT_COUNTERS
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.strict_score_proof_timing_summary import (
    _audit_result as _audit_strict, _repeat_results, _same, _work as _strict_work,
)

LABELS = ("candidate", "branch", "long")
GROUPS = ("ordinary_development", "historical_regression", "zero_queue", "trigger_control", "new_trigger")
RUNTIME_FIELDS = ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")


def _pointer(row, state):
    return {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
            "queue": state["queue"], "timing_group": state["timing_group"]}


def _latencies(states):
    if not states:
        return {"queue_count": 0, "latency": None, "comparisons": None}
    latency = {label: _latency(states, label) for label in LABELS}
    for label in LABELS:
        ordered = sorted(median(state["wall_ns"][label]) for state in states)
        latency[label]["p99_state_median_wall_ms"] = ordered[math.ceil(.99 * len(ordered)) - 1] / 1e6
    candidate = latency["candidate"]
    comparisons = {label: {
        "wall_ratio": candidate["sum_state_median_wall_ns"] / latency[label]["sum_state_median_wall_ns"],
        "cpu_ratio": (candidate["sum_state_median_cpu_ns"] / latency[label]["sum_state_median_cpu_ns"]
                      if latency[label]["sum_state_median_cpu_ns"] else None),
        "p99_state_median_ratio": candidate["p99_state_median_wall_ms"] / latency[label]["p99_state_median_wall_ms"],
        "max_paired_state_slowdown": max(median(state["wall_ns"]["candidate"]) /
                                        median(state["wall_ns"][label]) for state in states),
    } for label in LABELS[1:]}
    return {"queue_count": len(states), "latency": latency, "comparisons": comparisons}


def _audit_result(result, state, protocol, model, lanes, label, expected):
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
        errors.extend(audit_dual_stage(result["initial_lp"], len(state["queue"])))
        errors.extend(audit_dual_stage(result["final_lp"], len(state["queue"]), result["initial_lp"]))
    prior = state["prior_routed"][arm]
    if result["probe_stage"] == "frontier_late" and result["probe"] is not None and result["frontier_evidence"] is not None:
        probe = result["probe"]
        errors.extend(audit_axis_capacity_proof(probe, result["frontier_evidence"], state["queue"], lanes))
        if not _same(_original_proof(probe["score_proof"]), prior["probe"]["score_proof"]):
            errors.append("retained tables changed the frozen axis proof arithmetic")
        capacity = probe["score_proof"]["capacity"]
        if capacity.get("merged_values_copied") != len(capacity["bounds"]) * (lanes + 1):
            errors.append("retained merged vectors must charge every copied cell")
        for bound in capacity["bounds"]:
            if not _same(bound.get("exact_slot_numerators"), _merged_values(bound["axis_service_profiles"], state["queue"], lanes)):
                errors.append("retained vector differs from the already-paid max-plus merge")
        errors.extend(audit_residual_completion(probe, prior["probe"], model, state["queue"], lanes))
    return metric, errors, quality


def _work(results, label):
    if label == "long":
        return _strict_work(results)
    work = _candidate_work(results)
    probes = [result["probe"] for result in results if result["probe"] is not None]
    proofs = [probe["score_proof"] for probe in probes]
    capacities = [proof["capacity"] for proof in proofs if "capacity" in proof]
    return {**work,
        "root_pruning": {key: sum(result["final_lp"]["root_pruning"][key] for result in results) for key in ROOT_COUNTERS},
        **({"dual_reuse": _dual_work([result["final_lp"] for result in results])} if label == "candidate" else {}),
        "frontier_export": {key: sum(result["frontier_evidence"]["export_work"][key] for result in results
                                    if result["frontier_evidence"] is not None) for key in EXPORT_COUNTERS},
        "probe_stage_counts": dict(Counter(result["probe_stage"] for result in results)),
        "axis_capacity": {"attempted_calls": sum(capacity["attempted"] for capacity in capacities),
            **{key: sum(capacity[key] for capacity in capacities) for key in AXIS_COUNTERS},
            "merged_values_copied": sum(capacity.get("merged_values_copied", 0) for capacity in capacities)},
        "fallback_dfs_states": sum(proof.get("dfs_states_visited", proof["states_visited"]) for proof in proofs),
        "residual_capacity_completion": {key: sum(probe["completion"].get(key, 0) for probe in probes) for key in RESIDUAL_COUNTERS},
        "canonical_dfs_states": sum(probe["completion"].get("dfs_states_visited", probe["completion"]["states_visited"]) for probe in probes),
    }


def _group(states, repeats):
    return {**_latencies(states),
        "metrics": {label: _combine_metrics(state["metrics"][label] for state in states) for label in LABELS},
        "work": {label: _work([result for state in states for result in _repeat_results(state, label, repeats)], label)
                 for label in LABELS}}


def _orders(cell, row, repeats):
    state_orders, label_orders = cell["orders"]["state_orders"], cell["orders"]["label_orders"]
    count = len(cell["states"])
    if (len(state_orders) != repeats or any(sorted(order) != list(range(count)) for order in state_orders)
            or len(label_orders) != count or any(len(orders) != repeats or
                set(map(tuple, orders)) != set(permutations(LABELS)) for orders in label_orders)):
        raise ValueError("every state must use all six arm permutations and every sweep must cover every state")
    if not _same(row["state_orders"], state_orders):
        raise ValueError("actual sweep order differs from the frozen random order")
    for index, state in enumerate(row["states"]):
        if not _same(state["blocks"], [{"repeat": repeat, "order": label_orders[index][repeat]} for repeat in range(repeats)]):
            raise ValueError("every measured adjacent three-arm block must retain its registered order")


def summarize(protocol, rows):
    if tuple(protocol["labels"]) != LABELS or protocol["repeats"] != 6:
        raise ValueError("the experiment requires three fixed arms and all six permutations")
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered cells must occur once in order")
    repeats, models = protocol["repeats"], {model["node"]: model for model in protocol["models"]}
    fixed = {group: {_key(item["node"], item["lanes"], item) for item in protocol[field]}
        for group, field in (("historical_regression", "historical_regression_keys"),
                            ("trigger_control", "trigger_control_keys"), ("new_trigger", "new_trigger_keys"))}
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    if any(not keys <= registered for keys in fixed.values()) or any(
            _key(item["node"], item["lanes"], item) not in registered for item in protocol["ordinary_tail_keys"]):
        raise ValueError("all fixed historical, tail and trigger controls must belong to the frozen roster")
    entries, cells, failures, quality_failures = [], [], [], []
    groups = defaultdict(list)
    warmup_results = {label: [] for label in LABELS}
    stable = pinned = complete = reproduced = True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        if (len(states) != len(cell["states"]) or not states or any(not _same(
                {key: state.get(key) for key in original}, original) for state, original in zip(states, cell["states"]))):
            raise ValueError("timed states or their source pointers differ from the complete frozen roster")
        _orders(cell, row, repeats)
        point, errors = {"node": row["node"], "lanes": row["lanes"]}, []
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(model["options"])):
            errors.append("cell registration or physical catalogue metadata differs")
        pinned &= all(row[key]["cpu_affinity"] == [protocol["cpu"]] and row[key]["gc_enabled"] for key in RUNTIME_FIELDS)
        if (row["new_reference_calls"] or row["new_oracle_calls"] or row["screen_selector_calls"]
                or row["borrowed_reference_count"] != len(states)):
            errors.append("timing must borrow all references without a screening, oracle or reference selector")
        if errors:
            failures.append({**point, "errors": errors})
        keys, first_audits = [], {}
        for state_index, state in enumerate(states):
            pointer, errors = _pointer(row, state), []
            key = _key(row["node"], row["lanes"], state)
            group = next((name for name in ("new_trigger", "trigger_control", "historical_regression") if key in fixed[name]),
                         "zero_queue" if not any(state["queue"]) else "ordinary_development")
            if state["timing_group"] != group or state["historical_regression"] != (key in fixed["historical_regression"]):
                errors.append("timing group or historical flag differs from its frozen independent keys")
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            expected, extra = _reference(state, protocol, len(model["options"]))
            errors.extend(extra)
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
                errors.append("saved independent reference has invalid budget or paid certificate")
            options = tuple(SimpleNamespace(**option) for option in model["options"])
            errors.extend("reference: " + name for name, failed in check_selection(selection_from_record(state["reference"]),
                options, tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            keys.append(expected)
            complete &= expected is not None
            for label in LABELS:
                for clock in ("wall_ns", "cpu_ns"):
                    values = state[clock][label]
                    if len(values) != repeats or any(type(value) is not int or value < (1 if clock == "wall_ns" else 0) for value in values):
                        raise ValueError("every frozen state requires all wall and CPU observations")
                results = _repeat_results(state, label, repeats)
                stable &= not state["repeat_mismatches"][label]
                if label != "long":
                    reproduced &= _same(results[0], state["expected_results"][label])
                first = _audit_result(results[0], state, protocol, model, row["lanes"], label, expected)
                first_audits[state_index, label] = first
                errors.extend(f"{label} first: " + message for message in first[1])
                changed = {item["repeat_index"]: item["result"] for item in state["repeat_mismatches"][label]}
                metrics = []
                for repeat, result in enumerate(results):
                    metric, extra, quality = (_audit_result(result, state, protocol, model, row["lanes"], label, expected)
                                             if repeat in changed else first)
                    if repeat in changed:
                        errors.extend(f"{label} repeat {repeat}: " + message for message in extra)
                    metrics.append(metric)
                    if not quality:
                        quality_failures.append({**pointer, "phase": "timing", "label": label, "repeat_index": repeat})
                if not _same(_combine_metrics(metrics), {**state["metrics"][label], "operational_elapsed_s": 0.0}):
                    errors.append(label + " aggregate metrics differ from the actual repeat outputs")
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append((row, state))
            groups[state["timing_group"]].append(state)
        if (len(row["warmups"]) != len(LABELS) * protocol["warmups_per_label"] or any(
                sum(item["label"] == label for item in row["warmups"]) != protocol["warmups_per_label"] for label in LABELS)):
            raise ValueError("all and only registered warmup calls are required")
        for item in row["warmups"]:
            label, index, result = item["label"], item["state_index"], item["result"]
            if label not in LABELS or type(index) is not int or index != 0:
                raise ValueError("warmups must use each fixed arm on the first frozen state")
            state = states[index]
            metric, extra, quality = (first_audits[index, label] if _same(result, state["first_results"][label]) else
                _audit_result(result, state, protocol, model, row["lanes"], label, keys[index]))
            errors = list(extra)
            if label != "long":
                reproduced &= _same(result, state["expected_results"][label])
            stable &= _same(result, state["first_results"][label])
            if not _same(metric, {**item["metrics"], "operational_elapsed_s": 0.0}):
                errors.append("warmup metric differs from its actual saved result")
            if errors:
                failures.append({**_pointer(row, state), "phase": "warmup", "label": label, "errors": errors})
            if not quality:
                quality_failures.append({**_pointer(row, state), "phase": "warmup", "label": label})
            warmup_results[label].append(result)
        cells.append({**point, **_latencies(states),
            "groups": {group: _latencies([state for state in states if state["timing_group"] == group]) for group in GROUPS},
            **{key: row[key] for key in RUNTIME_FIELDS}})
    all_states = [state for _, state in entries]
    timed_calls = len(all_states) * repeats * len(LABELS)
    warmup_calls = sum(map(len, warmup_results.values()))
    if any(protocol[key] != count for key, count in (("expected_state_count", len(all_states)),
            ("expected_timed_calls", timed_calls), ("expected_warmup_calls", warmup_calls),
            ("expected_selector_calls", timed_calls + warmup_calls))):
        failures.append({"errors": ["timed and warmup coverage differ from frozen call accounting"]})
    if protocol["expected_group_counts"] != {group: len(groups[group]) for group in GROUPS}:
        failures.append({"errors": ["every frozen ordinary, historical, zero and trigger group must retain its full size"]})
    grouped = {group: _group(groups[group], repeats) for group in GROUPS}
    trigger = groups["trigger_control"] + groups["new_trigger"]
    trigger_group = _group(trigger, repeats)
    gates = {}
    for label in LABELS[1:]:
        ordinary, historical = (grouped[name]["comparisons"] for name in GROUPS[:2])
        combined = trigger_group["comparisons"]
        gates[label] = {
            "ordinary_pooled_nonregression": bool(ordinary and ordinary[label]["wall_ratio"] <= protocol["nonregression_limit"]),
            "ordinary_cell_nonregression": all(cell["groups"]["ordinary_development"]["comparisons"][label]["wall_ratio"] <= protocol["nonregression_limit"]
                for cell in cells if cell["groups"]["ordinary_development"]["queue_count"]),
            "ordinary_p99_nonregression": bool(ordinary and ordinary[label]["p99_state_median_ratio"] <= protocol["nonregression_limit"]),
            "historical_pooled_nonregression": bool(historical and historical[label]["wall_ratio"] <= protocol["nonregression_limit"]),
            "historical_p99_nonregression": bool(historical and historical[label]["p99_state_median_ratio"] <= protocol["nonregression_limit"]),
            "historical_every_state_nonregression": bool(historical and historical[label]["max_paired_state_slowdown"] <= protocol["nonregression_limit"]),
            "trigger_pooled_nonregression": bool(combined and combined[label]["wall_ratio"] <= protocol["nonregression_limit"]),
            "trigger_every_state_nonregression": bool(combined and combined[label]["max_paired_state_slowdown"] <= protocol["nonregression_limit"]),
        }
    gates["long"]["trigger_pooled_five_percent_gain"] = bool(trigger and trigger_group["comparisons"]["long"]["wall_ratio"] <= protocol["trigger_speedup_limit"])
    valid, quality = not failures, bool(complete and not quality_failures)
    execution = bool(valid and stable and pinned and reproduced)
    faster = all(all(values.values()) for values in gates.values())
    decision = ("INVALID_TIMING_EXECUTION" if not valid or not pinned else "INCOMPLETE_REFERENCE" if not complete else
        "RESIDUAL_CAPACITY_TIMING_QUALITY_HOLD" if not quality else "INVALID_TIMING_EXECUTION" if not stable or not reproduced else
        "RESIDUAL_CAPACITY_LOCAL_TIMING_ADVANCE" if faster else "RESIDUAL_CAPACITY_LOCAL_TIMING_HOLD")
    pooled = _group(all_states, repeats)
    warmup_work = {label: _work(warmup_results[label], label) for label in LABELS}
    active = [state for state in all_states if state["expected_results"]["candidate"]["probe"] is not None
              and state["expected_results"]["candidate"]["probe"]["completion"].get("tables_reused", 0)]
    tails = {_key(item["node"], item["lanes"], item) for item in protocol["ordinary_tail_keys"]}
    oracles = [state["oracle"] for state in all_states if state["oracle"] is not None]
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "state_count": len(all_states), "labels": list(LABELS), "repeats": repeats,
        "execution_checks_passed": execution, "certificate_checks_passed": valid, "references_complete": bool(complete),
        "quality_gate_passed": bool(valid and quality), "repeat_paths_stable": bool(stable),
        "cpu_affinity_preserved": bool(pinned), "frozen_residual_results_reproduced": bool(reproduced),
        "local_timing_gate_passed": bool(execution and quality and faster), "timing_gates": gates,
        "accounting_failures": failures, "quality_failures": quality_failures,
        "timed_selector_calls": timed_calls, "warmup_calls": warmup_calls, "executed_selector_calls": timed_calls + warmup_calls,
        "executed_reference_calls": 0, "executed_oracle_calls": 0, "screen_selector_calls": 0,
        "borrowed_reference_count": len(all_states), "borrowed_native_snapshots": len(all_states) * 8,
        "borrowed_axis_router_count": len(all_states) * 2, "borrowed_residual_router_count": len(all_states) * 2,
        **pooled, "groups": grouped, "trigger_combined": trigger_group, "cells": cells,
        "ordinary_tail": _group([state for row, state in entries if _key(row["node"], row["lanes"], state) in tails], repeats),
        "retained_table_active": _group(active, repeats),
        "retained_table_active_states": [{**_pointer(row, state), **_latencies([state])} for row, state in entries if state in active],
        "trigger_states": [{**_pointer(row, state), **_latencies([state])} for row, state in entries if state in trigger],
        "warmup_work": warmup_work,
        "executed_search_work": {key: sum(pooled["work"][label]["search_work"][key] + warmup_work[label]["search_work"][key] for label in LABELS) for key in COUNTERS},
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in all_states),
        "borrowed_reference_oracle_work": {"calls": len(oracles), "work": {key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
