"""Matched-quality whole-selector timing, with fixed cohort and cost accounting."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from statistics import median
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    CAPACITY_COUNTERS, COUNTERS, _combine_metrics,
    _metrics, _record,
)
from algorithm.experiments.aligned_hardware_integer_capacity_timing_summary import _latency
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.score_face_completion_confirmation_summary import (
    COMPLETION_COUNTERS, _key, _pointer, _reference,
)
from algorithm.experiments.strict_score_proof_development_summary import PROOF_COUNTERS, _proof_errors
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record


LABELS = ("candidate", "control")


def _latencies(states):
    if not states:
        return {"queue_count": 0, "latency": None, "comparison": None}
    latency = {label: _latency(states, label) for label in LABELS}
    for label in LABELS:
        ordered = sorted(median(state["wall_ns"][label]) for state in states)
        latency[label].update(max_state_median_wall_ms=ordered[-1] / 1e6,
                              p99_state_median_wall_ms=ordered[math.ceil(.99 * len(ordered)) - 1] / 1e6)
    candidate, control = latency["candidate"], latency["control"]
    return {"queue_count": len(states), "latency": latency, "comparison": {
        "wall_ratio": candidate["sum_state_median_wall_ns"] / control["sum_state_median_wall_ns"],
        "cpu_ratio": (candidate["sum_state_median_cpu_ns"] / control["sum_state_median_cpu_ns"]
                      if control["sum_state_median_cpu_ns"] else None),
        "p95_state_median_ratio": candidate["p95_state_median_wall_ms"] / control["p95_state_median_wall_ms"],
        "p95_raw_call_ratio": candidate["p95_raw_call_wall_ms"] / control["p95_raw_call_wall_ms"],
        "max_paired_state_slowdown": max(median(state["wall_ns"]["candidate"]) /
                                         median(state["wall_ns"]["control"]) for state in states),
    }}


def _same(left, right):
    return json.loads(json.dumps(left)) == json.loads(json.dumps(right))


def _repeat_results(state, label, repeats):
    first = state["first_results"][label]
    overrides = state["repeat_mismatches"][label]
    indices = [item["repeat_index"] for item in overrides]
    if (len(indices) != len(set(indices)) or any(type(i) is not int or not 1 <= i < repeats for i in indices)
            or any(_same(item["result"], first) for item in overrides)):
        raise ValueError("repeat overrides must preserve actual differing outputs after repeat zero")
    changed = {item["repeat_index"]: item["result"] for item in overrides}
    return [changed.get(index, first) for index in range(repeats)]


def _audit_result(result, state, protocol, model, lanes, label, expected):
    budget = protocol["lp_budgets"][label]
    options = tuple(SimpleNamespace(**option) for option in model["options"])
    runtime_model = SimpleNamespace(options=options)
    reference = selection_from_record(state["reference"])
    records, errors = {}, []
    for name in ("selection", "baseline_selection", "score_certified_selection"):
        selected = selection_from_record(result[name])
        metrics = _metrics()
        _record(metrics, selected, runtime_model, tuple(state["queue"]), lanes, budget, expected,
                result["preprocessing"], reference)
        if selected.lp_node_budget != budget:
            errors.append(f"{name} LP budget differs from the registered arm")
        if metrics["failures"]:
            errors.append(f"{name} has invalid feasibility, certificate or LP accounting")
        records[name] = metrics
    view = {"queue": state["queue"], "baseline_source": "embedded_stage_one",
            "baseline": {"selection": result["baseline_selection"],
                         "metrics": records["baseline_selection"],
                         "capacity_propagation": result["capacity_propagation"]},
            "completed": {**result, "metrics": records["selection"]}}
    errors.extend(_proof_errors(view, protocol, model, lanes))
    if any(type(result["capacity_propagation"][key]) is not int or result["capacity_propagation"][key] < 0
           for key in CAPACITY_COUNTERS):
        errors.append("capacity work counters must be nonnegative integers")
    selected, metrics = result["selection"], records["selection"]
    quality = bool(expected is not None and selected["exact"] and selected["score_exact"]
                   and selected["canonical_exact"] and not metrics["score_mismatch_count"]
                   and not metrics["canonical_action_mismatch_count"])
    return metrics, errors, quality


def _work(results):
    return {
        "calls": len(results),
        "search_work": {key: sum(result["selection"][key] for result in results) for key in COUNTERS},
        "preprocessing": {key: sum(result["preprocessing"][key] for result in results)
                          for key in (results[0]["preprocessing"] if results else ())},
        "capacity_propagation": {key: sum(result["capacity_propagation"][key] for result in results)
                                 for key in CAPACITY_COUNTERS},
        **{stage: {
            "attempted_count": sum(result[stage]["attempted"] for result in results),
            "completed_count": sum(result[stage]["status"] == "COMPLETE" for result in results),
            "exhausted_count": sum(result[stage]["status"] == "STATE_BUDGET_EXHAUSTED" for result in results),
            "work": {key: sum(result[stage][key] for result in results) for key in counters},
        } for stage, counters in (("score_proof", PROOF_COUNTERS), ("completion", COMPLETION_COUNTERS))},
        "shared_integer_states": sum(result["score_proof"]["states_visited"] +
                                     result["completion"]["states_visited"] for result in results),
        "max_shared_integer_states": max((result["score_proof"]["states_visited"] +
                                         result["completion"]["states_visited"] for result in results), default=0),
    }


def _group(entries, repeats):
    states = [state for _, state in entries]
    return {**_latencies(states),
            "metrics": {label: _combine_metrics(state["metrics"][label] for state in states) for label in LABELS},
            "work": {label: _work([result for state in states for result in _repeat_results(state, label, repeats)])
                     for label in LABELS}}


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    expected_cells = {(node, lanes) for node in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(set(actual_cells)) != len(actual_cells) or set(actual_cells) != expected_cells:
        raise ValueError("all registered node/R cells must occur exactly once")
    repeats = protocol["repeats"]
    entries, cells, failures, quality_failures = [], [], [], []
    warmup_results = {label: [] for label in LABELS}
    warmup_metrics = {label: [] for label in LABELS}
    cohorts, seeds, cohort_cells = defaultdict(list), defaultdict(list), defaultdict(list)
    complete = stable = pinned = reproduced = True
    for row in rows:
        model = models[row["node"]]
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(model["options"])):
            raise ValueError("cell registration or model metadata differs from the protocol")
        expected = [(state["state_id"], panel["panel"], panel["cohort"], panel.get("seed"), tuple(state["queue"]))
                    for panel in protocol["queue_panels"][str(row["lanes"]) ] for state in panel["states"]]
        actual = [(state["state_id"], state["panel"], state["cohort"], state["seed"], tuple(state["queue"]))
                  for state in row["states"]]
        if actual != expected or not actual or len(set(item[0] for item in actual)) != len(actual):
            raise ValueError("timed states differ from the registered complete roster")
        pinned &= all(row[key]["cpu_affinity"] == [protocol["cpu"]] for key in
                      ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end"))
        keys = []
        for state in row["states"]:
            _check_single_call(state["reference_metrics"], state["reference"],
                               reference_complete=state["reference_complete"])
            expected_key, errors = _reference(state, protocol, row["option_count"])
            keys.append(expected_key)
            complete &= expected_key is not None
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
                errors.append("saved independent reference has invalid budget or certificate")
            reproduced &= _same(state["first_results"]["candidate"], state["expected_candidate_result"])
            previous = state["expected_candidate_result"]
            if state["previous_integer_trigger"] != bool(previous["score_proof"]["attempted"] or previous["completion"]["attempted"]):
                errors.append("integer-trigger subgroup differs from the frozen confirmation result")
            for label in LABELS:
                for clock in ("wall_ns", "cpu_ns"):
                    values = state[clock][label]
                    if len(values) != repeats or any(type(v) is not int or v < (1 if clock == "wall_ns" else 0) for v in values):
                        raise ValueError("every registered state needs all wall and CPU measurements")
                results = _repeat_results(state, label, repeats)
                stable &= not state["repeat_mismatches"][label]
                measured = _metrics()
                for repeat_index, result in enumerate(results):
                    metric, result_errors, quality = _audit_result(result, state, protocol, model, row["lanes"], label, expected_key)
                    errors.extend(f"{label} repeat {repeat_index}: {message}" for message in result_errors)
                    measured = _combine_metrics((measured, metric))
                    if not quality:
                        quality_failures.append({**_pointer(row, state), "label": label, "repeat_index": repeat_index})
                stored = {**state["metrics"][label], "operational_elapsed_s": 0.0}
                if not _same(measured, stored):
                    errors.append(f"{label} aggregate metrics do not equal its actual repeat outputs")
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entry = (row, state)
            entries.append(entry)
            cohorts[state["cohort"]].append(entry)
            seeds[state["seed"]].append(entry)
            cohort = "fresh_confirmation" if state["cohort"] == "fresh_confirmation" else "regression"
            cohort_cells[(row["node"], row["lanes"], cohort)].append(entry)
        if any(sum(warmup["label"] == label for warmup in row["warmups"]) != protocol["warmups_per_label"] for label in LABELS):
            raise ValueError("warmup calls differ from the registered count")
        for warmup in row["warmups"]:
            label, index, result = warmup["label"], warmup["state_index"], warmup["result"]
            if label not in LABELS or type(index) is not int or index != 0:
                raise ValueError("warmup must use the registered arm and first state")
            state = row["states"][index]
            metric, errors, quality = _audit_result(result, state, protocol, model, row["lanes"], label, keys[index])
            if not _same(metric, {**warmup["metrics"], "operational_elapsed_s": 0.0}):
                errors.append("warmup metrics disagree with its saved result")
            if errors:
                failures.append({**_pointer(row, state), "phase": "warmup", "label": label, "errors": errors})
            if not quality:
                quality_failures.append({**_pointer(row, state), "phase": "warmup", "label": label})
            warmup_results[label].append(result)
            warmup_metrics[label].append(metric)
        cells.append({"node": row["node"], "lanes": row["lanes"],
                      **_group([(row, state) for state in row["states"]], repeats),
                      **{key: row[key] for key in ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")}})
    fresh_entries = cohorts["fresh_confirmation"]
    regression_entries = [entry for cohort, group in cohorts.items() if cohort != "fresh_confirmation" for entry in group]
    groups = {"fresh_confirmation": _group(fresh_entries, repeats), "regression": _group(regression_entries, repeats)}
    cell_groups = [{"node": node, "lanes": lanes, "cohort": cohort, **_group(group, repeats)}
                   for (node, lanes, cohort), group in cohort_cells.items()]
    gates = {cohort: {
        "pooled_at_least_five_percent_faster": bool(group["queue_count"] and
             group["comparison"]["wall_ratio"] <= protocol["pooled_ratio_limit"]),
        "no_node_lane_cell_regression": all(cell["comparison"]["wall_ratio"] <= protocol["cell_ratio_limit"]
                                            for cell in cell_groups if cell["cohort"] == cohort),
    } for cohort, group in groups.items()}
    valid = not failures
    quality = bool(valid and complete and not quality_failures)
    execution = bool(valid and stable and pinned and reproduced)
    faster = all(all(gate.values()) for gate in gates.values())
    decision = ("INVALID_TIMING_EXECUTION" if not valid or not pinned else
                "INCOMPLETE_REFERENCE" if not complete else "STRICT_SCORE_PROOF_TIMING_QUALITY_HOLD" if not quality else
                "INVALID_TIMING_EXECUTION" if not stable or not reproduced else "STRICT_SCORE_PROOF_LOCAL_TIMING_ADVANCE" if faster else
                "STRICT_SCORE_PROOF_LOCAL_TIMING_HOLD")
    pooled = _group(entries, repeats)
    warmup_work = {label: _work(warmup_results[label]) for label in LABELS}
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "lp_budgets": protocol["lp_budgets"], "integer_state_budget": protocol["integer_state_budget"],
        "repeats": repeats, "execution_checks_passed": execution, "certificate_checks_passed": valid,
        "repeat_paths_stable": bool(stable), "cpu_affinity_preserved": bool(pinned),
        "candidate_reproduced": bool(reproduced), "references_complete": bool(complete),
        "quality_gate_passed": quality, "local_timing_gate_passed": bool(execution and quality and faster),
        "timing_gates_by_cohort": gates, "accounting_failures": failures, "quality_failures": quality_failures,
        "timed_selector_calls": repeats * len(entries) * len(LABELS),
        "warmup_calls": sum(len(values) for values in warmup_results.values()),
        "executed_selector_calls": repeats * len(entries) * len(LABELS) + sum(len(values) for values in warmup_results.values()),
        "executed_reference_calls": 0, "executed_oracle_calls": 0, "reused_reference_count": len(entries),
        "reused_original_incomplete_reference_count": sum(not state["reference_complete"] for _, state in entries),
        "reused_oracle_count": sum(state["oracle"] is not None for _, state in entries),
        "embedded_baseline_count": repeats * len(entries) * len(LABELS) + sum(len(values) for values in warmup_results.values()),
        **pooled, **groups, "cells": cells, "cohort_cells": cell_groups,
        "cohorts": {cohort: _group(group, repeats) for cohort, group in cohorts.items()},
        "by_seed": [{"seed": seed, **_group(group, repeats)} for seed, group in seeds.items()],
        "previous_integer_trigger": _group([entry for entry in entries if entry[1]["previous_integer_trigger"]], repeats),
        "previous_integer_trigger_states": [{**_pointer(row, state), **_latencies([state])}
                                             for row, state in entries if state["previous_integer_trigger"]],
        "warmup_work": warmup_work,
        "warmup_metrics": {label: _combine_metrics(warmup_metrics[label]) for label in LABELS},
        "executed_search_work": {key: sum(pooled["work"][label]["search_work"][key] +
                                          warmup_work[label]["search_work"][key] for label in LABELS) for key in COUNTERS},
        "limitations": protocol["limitations"],
    }
