"""Four-arm development timing with retained primary gates and actual guard fees."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations
import math
from statistics import median
from types import SimpleNamespace

from algorithm.experiments.affine_dual_development_summary import AFFINE_COUNTERS
from algorithm.experiments.zero_delta_affine_development_summary import (
    ZERO_COUNTERS, _audit_zero_result as _audit_result, _work,
)
from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.branch_reserved_development_summary import _key
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.aligned_hardware_integer_capacity_timing_summary import _latency
from algorithm.experiments.residual_capacity_timing_summary import (
    GROUPS as CONTROL_GROUPS, RUNTIME_FIELDS, _pointer,
)
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _repeat_results, _same


LABELS = ("candidate", "affine", "branch", "long")
ORIGINAL_LABELS = ("candidate", "branch", "long")
GROUPS = (*CONTROL_GROUPS, "fresh_confirmation")


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
        "cpu_ratio": candidate["sum_state_median_cpu_ns"] / latency[label]["sum_state_median_cpu_ns"]
                     if latency[label]["sum_state_median_cpu_ns"] else None,
        "p99_state_median_ratio": candidate["p99_state_median_wall_ms"] / latency[label]["p99_state_median_wall_ms"],
        "max_paired_state_slowdown": max(median(state["wall_ns"]["candidate"]) /
                                        median(state["wall_ns"][label]) for state in states),
    } for label in LABELS[1:]}
    return {"queue_count": len(states), "latency": latency, "comparisons": comparisons}


def _orders(cell, row, repeats):
    sweeps, blocks = cell["orders"]["state_orders"], cell["orders"]["label_orders"]
    count = len(cell["states"])
    if len(sweeps) != repeats or any(sorted(order) != list(range(count)) for order in sweeps) or len(blocks) != count:
        raise ValueError("every registered sweep must cover each state once")
    for orders in blocks:
        if (len(orders) != repeats or any(sorted(order) != sorted(LABELS) for order in orders)
                or set(tuple(label for label in order if label != "affine") for order in orders) != set(permutations(ORIGINAL_LABELS))
                or sorted(Counter(order.index("affine") for order in orders).values()) != [1, 1, 2, 2]):
            raise ValueError("retain all six original relative permutations and the registered affine insertion counts")
    if not _same(row["state_orders"], sweeps):
        raise ValueError("actual sweep order differs from the frozen order")
    for index, state in enumerate(row["states"]):
        if not _same(state["blocks"], [{"repeat": repeat, "order": blocks[index][repeat]} for repeat in range(repeats)]):
            raise ValueError("each measured adjacent four-arm block must retain its registered order")


def _group(states, repeats):
    return {**_latencies(states),
        "metrics": {label: _combine_metrics(state["metrics"][label] for state in states) for label in LABELS},
        "work": {label: _work([result for state in states for result in _repeat_results(state, label, repeats)], label)
                 for label in LABELS}}


def summarize(protocol, rows):
    if tuple(protocol["labels"]) != LABELS or protocol["repeats"] != 6:
        raise ValueError("the experiment requires four fixed arms and six registered blocks")
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
            source = state["confirmation_source"]
            group = ("fresh_confirmation" if source == "fresh" else
                next((name for name in ("new_trigger", "trigger_control", "historical_regression") if key in fixed[name]),
                     "zero_queue" if not any(state["queue"]) else "ordinary_development"))
            if source not in ("fresh", "borrowed_control") or (source == "fresh" and "prior_routed" in state):
                errors.append("fresh and borrowed control evidence must retain their registered source")
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
            ("expected_fresh_state_count", sum(state["confirmation_source"] == "fresh" for state in all_states)),
            ("expected_control_state_count", sum(state["confirmation_source"] == "borrowed_control" for state in all_states)),
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
        fresh = grouped["fresh_confirmation"]["comparisons"]
        gates[label] = {
            "fresh_pooled_nonregression": bool(fresh and fresh[label]["wall_ratio"] <= protocol["nonregression_limit"]),
            "fresh_cell_nonregression": all(cell["groups"]["fresh_confirmation"]["comparisons"][label]["wall_ratio"] <= protocol["nonregression_limit"]
                for cell in cells if cell["groups"]["fresh_confirmation"]["queue_count"]),
            "fresh_p99_nonregression": bool(fresh and fresh[label]["p99_state_median_ratio"] <= protocol["nonregression_limit"]),
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
    primary_passed = all(all(gates[label].values()) for label in ("branch", "long"))
    affine_passed = all(gates["affine"].values())
    fresh_passed = all(value for values in gates.values() for key, value in values.items() if key.startswith("fresh_"))
    control_passed = all(value for values in gates.values() for key, value in values.items() if not key.startswith("fresh_"))
    decision = ("INVALID_TIMING_EXECUTION" if not valid or not pinned else "INCOMPLETE_REFERENCE" if not complete else
        "ZERO_DELTA_AFFINE_TIMING_QUALITY_HOLD" if not quality else "INVALID_TIMING_EXECUTION" if not stable or not reproduced else
        "ZERO_DELTA_AFFINE_LOCAL_TIMING_ADVANCE" if faster else "ZERO_DELTA_AFFINE_LOCAL_TIMING_HOLD")
    pooled = _group(all_states, repeats)
    warmup_work = {label: _work(warmup_results[label], label) for label in LABELS}
    active = [state for state in all_states if state["expected_results"]["candidate"]["probe"] is not None
              and state["expected_results"]["candidate"]["probe"]["completion"].get("tables_reused", 0)]
    tails = {_key(item["node"], item["lanes"], item) for item in protocol["ordinary_tail_keys"]}
    oracles = [state["oracle"] for state in all_states if state["oracle"] is not None]
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "independent_confirmation": False, "state_count": len(all_states), "fresh_state_count": len(groups["fresh_confirmation"]),
        "control_state_count": len(all_states) - len(groups["fresh_confirmation"]), "labels": list(LABELS), "repeats": repeats,
        "execution_checks_passed": execution, "certificate_checks_passed": valid, "references_complete": bool(complete),
        "quality_gate_passed": bool(valid and quality), "repeat_paths_stable": bool(stable),
        "cpu_affinity_preserved": bool(pinned), "frozen_residual_results_reproduced": bool(reproduced),
        "local_timing_gate_passed": bool(execution and quality and faster), "timing_gates": gates,
        "primary_timing_gate_passed": bool(execution and quality and primary_passed),
        "affine_timing_gate_passed": bool(execution and quality and affine_passed),
        "origin_fresh_timing_gate_passed": bool(execution and quality and fresh_passed),
        "control_timing_gate_passed": bool(execution and quality and control_passed),
        "accounting_failures": failures, "quality_failures": quality_failures,
        "timed_selector_calls": timed_calls, "warmup_calls": warmup_calls, "executed_selector_calls": timed_calls + warmup_calls,
        "executed_reference_calls": 0, "executed_oracle_calls": 0, "screen_selector_calls": 0,
        "borrowed_reference_count": len(all_states), "borrowed_native_snapshots": len(all_states) * 12,
        "borrowed_axis_router_count": sum(state["confirmation_source"] == "borrowed_control" for state in all_states) * 2,
        "borrowed_residual_router_count": len(all_states) * 3,
        **pooled, "groups": grouped, "trigger_combined": trigger_group, "cells": cells,
        "ordinary_tail": _group([state for row, state in entries if _key(row["node"], row["lanes"], state) in tails], repeats),
        "retained_table_active": _group(active, repeats),
        "retained_table_active_states": [{**_pointer(row, state), **_latencies([state])} for row, state in entries if state in active],
        "trigger_states": [{**_pointer(row, state), **_latencies([state])} for row, state in entries if state in trigger],
        "warmup_work": warmup_work,
        "executed_affine_dual_work": {key: pooled["work"]["candidate"]["affine_dual"][key] + warmup_work["candidate"]["affine_dual"][key] for key in AFFINE_COUNTERS},
        "executed_affine_control_work": {key: pooled["work"]["affine"]["affine_dual"][key] + warmup_work["affine"]["affine_dual"][key] for key in AFFINE_COUNTERS},
        "executed_zero_delta_work": {key: pooled["work"]["candidate"]["zero_delta"][key] + warmup_work["candidate"]["zero_delta"][key] for key in ZERO_COUNTERS},
        "executed_search_work": {key: sum(pooled["work"][label]["search_work"][key] + warmup_work[label]["search_work"][key] for label in LABELS) for key in COUNTERS},
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in all_states),
        "borrowed_reference_oracle_work": {"calls": len(oracles), "work": {key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
