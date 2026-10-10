"""Independent queue confirmation and three-arm timing with separate claims."""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import median

from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    COUNTERS, _combine_metrics,
)
from algorithm.experiments.aligned_hardware_integer_capacity_timing_summary import _latency
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import audit_candidate, _candidate_work
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _pointer, _reference
from algorithm.experiments.strict_score_proof_timing_summary import (
    _audit_result as _audit_strict, _repeat_results, _same, _work as _strict_work,
)


LABELS = ("candidate", "legacy", "long")
CONTROLS = ("legacy", "long")


def result_preprocessing(result, label):
    return result["final_lp"]["preprocessing"] if label == "candidate" else result["preprocessing"]


def audit_result(result, state, protocol, model, lanes, label, expected):
    if label == "candidate":
        return audit_candidate(result, state, protocol, model, lanes, expected)
    if label not in CONTROLS:
        raise ValueError("unregistered selector label")
    budget = protocol["initial_lp_budget"] if label == "legacy" else protocol["long_lp_budget"]
    control_protocol = {**protocol, "lp_budgets": {label: budget},
                        "integer_state_budget": protocol["control_integer_state_budget"]}
    return _audit_strict(result, state, control_protocol, model, lanes, label, expected)


def _latencies(states):
    if not states:
        return {"queue_count": 0, "latency": None, "comparisons": None}
    latency = {label: _latency(states, label) for label in LABELS}
    for label in LABELS:
        ordered = sorted(median(state["wall_ns"][label]) for state in states)
        latency[label].update(p99_state_median_wall_ms=ordered[math.ceil(.99 * len(ordered)) - 1] / 1e6,
                              max_state_median_wall_ms=ordered[-1] / 1e6,
                              max_raw_wall_ms=max(max(state["wall_ns"][label]) for state in states) / 1e6)
    candidate = latency["candidate"]
    comparisons = {}
    for label in CONTROLS:
        control = latency[label]
        comparisons[label] = {
            "wall_ratio": candidate["sum_state_median_wall_ns"] / control["sum_state_median_wall_ns"],
            "cpu_ratio": (candidate["sum_state_median_cpu_ns"] / control["sum_state_median_cpu_ns"]
                          if control["sum_state_median_cpu_ns"] else None),
            "p95_state_median_ratio": candidate["p95_state_median_wall_ms"] / control["p95_state_median_wall_ms"],
            "p99_state_median_ratio": candidate["p99_state_median_wall_ms"] / control["p99_state_median_wall_ms"],
            "max_paired_state_slowdown": max(median(state["wall_ns"]["candidate"]) /
                                             median(state["wall_ns"][label]) for state in states),
        }
    return {"queue_count": len(states), "latency": latency, "comparisons": comparisons}


def _work(results, label):
    return (_candidate_work if label == "candidate" else _strict_work)(results)


def _group(entries, repeats):
    states = [state for _, state in entries]
    return {**_latencies(states),
            "metrics": {label: _combine_metrics(state["metrics"][label] for state in states) for label in LABELS},
            "work": {label: _work([result for state in states for result in _repeat_results(state, label, repeats)], label)
                     for label in LABELS}}


def _screen_group(entries):
    return {"queue_count": len(entries),
            "metrics": {label: _combine_metrics(state["screen_metrics"][label] for _, state in entries)
                        for label in ("candidate", "legacy")},
            "work": {label: _work([state["screen_results"][label] for _, state in entries], label)
                     for label in ("candidate", "legacy")}}


def _timing_indices(protocol, node, lanes, trigger_keys):
    selected, index = [], 0
    for panel in protocol["queue_panels"][str(lanes)]:
        for local, state in enumerate(panel["states"]):
            fresh = panel["cohort"] == "fresh_confirmation"
            if ((fresh and local < protocol["timing_fresh_per_panel"])
                    or (node, lanes, tuple(state["queue"])) in trigger_keys
                    or panel["cohort"] == "ordinary_tail_regression"):
                selected.append(index)
            index += 1
    return selected


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    expected_cells = {(node, lanes) for node in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(set(actual_cells)) != len(actual_cells) or set(actual_cells) != expected_cells:
        raise ValueError("all registered node/R cells must occur exactly once")
    trigger_keys = {(item["node"], item["lanes"], tuple(item["queue"]))
                    for item in protocol["historical_trigger_keys"]}
    entries, timed, cells, failures, quality_failures = [], [], [], [], []
    cohort_entries, seed_entries = defaultdict(list), defaultdict(list)
    warmup_results = {label: [] for label in LABELS}
    warmup_metrics = {label: [] for label in LABELS}
    stable = pinned = complete = True
    repeats = protocol["repeats"]
    for row in rows:
        node, lanes = row["node"], row["lanes"]
        model = models[node]
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(model["options"])):
            raise ValueError("cell registration or model metadata differs from protocol")
        roster = [(state["state_id"], panel["panel"], panel["cohort"], panel.get("seed"), tuple(state["queue"]))
                  for panel in protocol["queue_panels"][str(lanes)] for state in panel["states"]]
        actual = [(state["state_id"], state["panel"], state["cohort"], state["seed"], tuple(state["queue"]))
                  for state in row["states"]]
        if actual != roster or not actual or len({state[0] for state in actual}) != len(actual):
            raise ValueError("stored states differ from the complete registered roster")
        indices = _timing_indices(protocol, node, lanes, trigger_keys)
        if (protocol["timing_indices"][f"{node}:{lanes}"] != indices
                or [i for i, state in enumerate(row["states"]) if "first_results" in state] != indices):
            raise ValueError("all and only preregistered states must have three-arm timing")
        pinned &= all(row[key]["cpu_affinity"] == [protocol["cpu"]] for key in
                      ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end"))
        keys, cell_timed = [], []
        for index, state in enumerate(row["states"]):
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            expected, errors = _reference(state, protocol, row["option_count"])
            keys.append(expected)
            complete &= expected is not None
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
                errors.append("independent reference has invalid budget or certificate")
            if state["previous_integer_trigger"] != ((node, lanes, tuple(state["queue"])) in trigger_keys):
                errors.append("historical trigger subgroup differs from frozen keys")
            legacy = state["screen_results"]["legacy"]
            initial = {"selection": legacy["baseline_selection"], "preprocessing": legacy["preprocessing"],
                       "capacity_propagation": legacy["capacity_propagation"]}
            if not _same(initial, state["expected_initial_lp"]):
                errors.append("initial LP comparator must come from the independent precision-screen legacy call")
            for label in ("candidate", "legacy"):
                metric, extra, quality = audit_result(state["screen_results"][label], state, protocol, model, lanes, label, expected)
                errors.extend(f"screen {label}: {message}" for message in extra)
                if not _same(metric, state["screen_metrics"][label]):
                    errors.append(f"screen {label} metrics differ from actual output")
                if not quality:
                    quality_failures.append({**_pointer(row, state), "phase": "screen", "label": label})
            if index in indices:
                for label in LABELS:
                    for clock in ("wall_ns", "cpu_ns"):
                        values = state[clock][label]
                        if len(values) != repeats or any(type(v) is not int or v < (1 if clock == "wall_ns" else 0) for v in values):
                            raise ValueError("all registered wall/CPU measurements are required")
                    results = _repeat_results(state, label, repeats)
                    stable &= not state["repeat_mismatches"][label]
                    metrics = []
                    for repeat, result in enumerate(results):
                        metric, extra, quality = audit_result(result, state, protocol, model, lanes, label, expected)
                        errors.extend(f"{label} repeat {repeat}: {message}" for message in extra)
                        metrics.append(metric)
                        if label in ("candidate", "legacy") and not _same(result, state["screen_results"][label]):
                            stable = False
                        if not quality:
                            quality_failures.append({**_pointer(row, state), "phase": "timing", "label": label,
                                                     "repeat_index": repeat})
                    if not _same(_combine_metrics(metrics), state["metrics"][label]):
                        errors.append(f"{label} repeat metrics differ from actual measured calls")
                cell_timed.append((row, state))
                timed.append((row, state))
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entries.append((row, state))
            cohort_entries[state["cohort"]].append((row, state))
        if (len(row["warmups"]) != len(LABELS) * protocol["warmups_per_label"]
                or any(sum(item["label"] == label for item in row["warmups"]) != protocol["warmups_per_label"] for label in LABELS)):
            raise ValueError("warmup counts differ from registration")
        for item in row["warmups"]:
            label, index = item["label"], item["state_index"]
            if label not in LABELS or index != indices[0]:
                raise ValueError("warmups must use the first registered timing state")
            state = row["states"][index]
            metric, errors, quality = audit_result(item["result"], state, protocol, model, lanes, label, keys[index])
            if not _same(metric, item["metrics"]):
                errors.append("warmup metrics differ from actual output")
            if errors:
                failures.append({**_pointer(row, state), "phase": "warmup", "label": label, "errors": errors})
            if not quality:
                quality_failures.append({**_pointer(row, state), "phase": "warmup", "label": label})
            warmup_results[label].append(item["result"])
            warmup_metrics[label].append(metric)
        cells.append({"node": node, "lanes": lanes, "screen": _screen_group([(row, state) for state in row["states"]]),
                      "timed": _group(cell_timed, repeats),
                      "fresh_timed": _group([entry for entry in cell_timed if entry[1]["cohort"] == "fresh_confirmation"], repeats),
                      **{key: row[key] for key in ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")}})
    fresh = [entry for entry in timed if entry[1]["cohort"] == "fresh_confirmation"]
    historical = [entry for entry in timed if entry[1]["previous_integer_trigger"]]
    tail = [entry for entry in timed if entry[1]["cohort"] == "ordinary_tail_regression"]
    if {(row["node"], row["lanes"], tuple(state["queue"])) for row, state in historical} != trigger_keys:
        raise ValueError("every historical trigger key must occur in the timing roster")
    fresh_group, trigger_group, tail_group = (_group(group, repeats) for group in (fresh, historical, tail))
    for entry in fresh:
        seed_entries[entry[1]["seed"]].append(entry)
    fresh_gates = {label: {
        "pooled_no_material_regression": bool(fresh and fresh_group["comparisons"][label]["wall_ratio"] <= protocol["fresh_nonregression_limit"]),
        "no_cell_material_regression": bool(fresh and all(
            cell["fresh_timed"]["comparisons"][label]["wall_ratio"] <= protocol["fresh_nonregression_limit"]
            for cell in cells if cell["fresh_timed"]["queue_count"])),
        "p99_no_material_regression": bool(fresh and fresh_group["comparisons"][label]["p99_state_median_ratio"] <= protocol["fresh_nonregression_limit"]),
    } for label in CONTROLS}
    regression_gates = {
        "trigger_legacy_pooled_gain": bool(historical and trigger_group["comparisons"]["legacy"]["wall_ratio"] <= protocol["trigger_legacy_ratio_limit"]),
        "trigger_legacy_no_state_regression": bool(historical and trigger_group["comparisons"]["legacy"]["max_paired_state_slowdown"] <= protocol["trigger_legacy_state_limit"]),
        "trigger_long_pooled_no_regression": bool(historical and trigger_group["comparisons"]["long"]["wall_ratio"] <= protocol["trigger_long_ratio_limit"]),
        "trigger_long_p99_no_material_regression": bool(historical and trigger_group["comparisons"]["long"]["p99_state_median_ratio"] <= protocol["trigger_long_p99_limit"]),
        **{f"tail_no_state_regression_vs_{label}": bool(tail and tail_group["comparisons"][label]["max_paired_state_slowdown"] <= protocol["tail_state_limit"])
           for label in CONTROLS},
    }
    valid = not failures
    candidate_quality = bool(valid and complete and not any(item["label"] == "candidate" for item in quality_failures))
    comparison_quality = bool(valid and complete and not quality_failures)
    execution = bool(valid and stable and pinned)
    timing_passed = all(all(gate.values()) for gate in fresh_gates.values()) and all(regression_gates.values())
    advance = bool(execution and candidate_quality and comparison_quality and timing_passed)
    decision = ("INVALID_EXECUTION" if not valid or not pinned else "INCOMPLETE_REFERENCE" if not complete else
                "RESUMABLE_PROBE_CONFIRMATION_QUALITY_HOLD" if not candidate_quality or not comparison_quality else
                "INVALID_TIMING_EXECUTION" if not stable else "RESUMABLE_PROBE_CONFIRMATION_ADVANCE" if advance else
                "RESUMABLE_PROBE_CONFIRMATION_TIMING_HOLD")
    screen = _screen_group(entries)
    measured = _group(timed, repeats)
    warmup_work = {label: _work(results, label) for label, results in warmup_results.items()}
    oracles = [state["oracle"] for _, state in entries if state["oracle"] is not None]
    method_calls = 2 * len(entries) + len(LABELS) * repeats * len(timed) + sum(len(items) for items in warmup_results.values())
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "queue_count": len(entries), "screen_selector_calls": 2 * len(entries),
        "timed_queue_count": len(timed), "timed_selector_calls": len(LABELS) * repeats * len(timed),
        "warmup_calls": sum(len(items) for items in warmup_results.values()),
        "executed_selector_calls": method_calls, "executed_reference_calls": len(entries),
        "total_executed_lp_selector_calls": method_calls + len(entries), "executed_oracle_calls": len(oracles),
        "certificate_checks_passed": valid, "effective_references_complete": bool(complete),
        "candidate_quality_gate_passed": candidate_quality, "comparison_quality_gate_passed": comparison_quality,
        "execution_checks_passed": execution, "repeat_paths_stable": bool(stable), "cpu_affinity_preserved": bool(pinned),
        "confirmation_gate_passed": advance,
        "fresh_speedup_established": bool(execution and comparison_quality and fresh and all(
            fresh_group["comparisons"][label]["wall_ratio"] <= protocol["fresh_speedup_limit"] for label in CONTROLS)),
        "timing_gates": {"fresh": fresh_gates, "regression": regression_gates},
        "accounting_failures": failures, "quality_failures": quality_failures,
        "screen": screen, "timed": measured, "warmup_work": warmup_work,
        "warmup_metrics": {label: _combine_metrics(warmup_metrics[label]) for label in LABELS},
        "screen_by_cohort": {cohort: _screen_group(group) for cohort, group in cohort_entries.items()},
        "fresh_timing": fresh_group, "historical_trigger_timing": trigger_group, "ordinary_tail_timing": tail_group,
        "fresh_by_seed": [{"seed": seed, **_group(group, repeats)} for seed, group in seed_entries.items()],
        "historical_trigger_states": [{**_pointer(row, state), **_latencies([state])} for row, state in historical],
        "ordinary_tail_states": [{**_pointer(row, state), **_latencies([state])} for row, state in tail],
        "fresh_worst_paired_states": {label: [{**_pointer(row, state), **_latencies([state])} for row, state in sorted(
            fresh, key=lambda entry: median(entry[1]["wall_ns"]["candidate"]) / median(entry[1]["wall_ns"][label]), reverse=True)[:10]]
            for label in CONTROLS},
        "reference_work": _combine_metrics(state["reference_metrics"] for _, state in entries),
        "original_incomplete_reference_count": sum(not state["reference_complete"] for _, state in entries),
        "oracle_work": {key: sum(item["work"][key] for item in oracles) for key in ORACLE_COUNTERS},
        "executed_method_search_work": {key: sum(screen["work"][label]["search_work"][key] for label in ("candidate", "legacy")) + sum(
            measured["work"][label]["search_work"][key] + warmup_work[label]["search_work"][key] for label in LABELS) for key in COUNTERS},
        "cells": cells, "limitations": protocol["limitations"],
    }
