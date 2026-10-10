"""Audit certificate-dependent routing and matched-budget development timing."""
from __future__ import annotations

from collections import Counter
import math
from statistics import median

from algorithm.experiments.aligned_hardware_integer_capacity_development import CAPACITY_COUNTERS, COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_integer_capacity_timing_summary import _latency
from algorithm.experiments.resumable_probe_development_summary import (
    _metric, _exact, _reference_audit, _candidate_work, audit_candidate as _audit_probe,
)
from algorithm.experiments.score_face_completion_confirmation_summary import _pointer
from algorithm.experiments.strict_score_proof_timing_summary import (
    _audit_result as _audit_strict, _repeat_results, _same, _work as _strict_work,
)

LABELS = ("candidate", "probe", "long")


def result_preprocessing(result, label):
    return result["preprocessing"] if label == "long" else result["final_lp"]["preprocessing"]


def audit_candidate(result, state, protocol, model, lanes, expected):
    initial, final, probe = result["initial_lp"], result["final_lp"], result["probe"]
    errors = []
    for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget"):
        if result[key] != protocol[key]:
            errors.append(f"candidate {key} differs from registration")
    if not _same({key: initial[key] for key in state["expected_initial_lp"]}, state["expected_initial_lp"]):
        errors.append("initial LP snapshot did not reproduce the historical selection and paid work")
    for name, stage, budget in (("initial", initial, protocol["initial_lp_budget"]),
                               ("final", final, protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"])):
        metric = _metric(stage["selection"], state, model, lanes, budget, expected, stage["preprocessing"])
        if stage["selection"]["lp_node_budget"] != budget or metric["failures"]:
            errors.append(f"{name} LP snapshot has invalid feasibility, certificate or budget accounting")
        if any(type(stage["capacity_propagation"][key]) is not int or stage["capacity_propagation"][key] < 0
               for key in CAPACITY_COUNTERS):
            errors.append(f"{name} capacity counters must be nonnegative integers")
    if (any(initial["preprocessing"][key] != final["preprocessing"][key]
            for key in initial["preprocessing"] if key != "lift_option_probes")
            or final["preprocessing"]["lift_option_probes"] < initial["preprocessing"]["lift_option_probes"]
            or not _same(initial["thresholds"], final["thresholds"])):
        errors.append("resumption changed fixed catalogue preprocessing or thresholds")
    if any(final["selection"][key] < initial["selection"][key] for key in COUNTERS):
        errors.append("final cumulative LP work lost an initial charged operation")
    if any(final["capacity_propagation"][key] < initial["capacity_propagation"][key] for key in CAPACITY_COUNTERS):
        errors.append("final cumulative capacity work lost an initial charged operation")

    if initial["selection"]["canonical_exact"]:
        stage_name, should_resume = "none", False
        selected, status = initial["selection"], "LP_INITIAL_COMPLETE"
    elif initial["selection"]["score_exact"]:
        stage_name = "initial"
        should_resume = probe is None or not probe["selection"]["canonical_exact"]
        selected = final["selection"] if should_resume else probe["selection"]
        status = ("LP_RESUME_COMPLETE" if selected["canonical_exact"] else "LP_RESUME_INCOMPLETE") if should_resume else "INTEGER_PROBE_COMPLETE"
    else:
        should_resume = True
        stage_name = "resumed" if final["selection"]["score_exact"] and not final["selection"]["canonical_exact"] else "none"
        selected = probe["selection"] if stage_name == "resumed" and probe is not None else final["selection"]
        status = ("INTEGER_PROBE_COMPLETE" if stage_name == "resumed" and selected["canonical_exact"] else
                  "LP_RESUME_COMPLETE" if selected["canonical_exact"] else "LP_RESUME_INCOMPLETE")
    if result["probe_stage"] != stage_name or (probe is None) != (stage_name == "none"):
        errors.append("probe stage differs from the initial/final score certificates")
    if result["resumed"] != should_resume:
        errors.append("resumption does not follow the registered certificate route")
    if not result["resumed"] and not _same(initial, final):
        errors.append("without resumption final LP snapshot must equal initial snapshot")
    if probe is not None:
        stage = final if stage_name == "resumed" else initial
        budget = protocol["lp_node_budget"] if stage_name == "resumed" else protocol["initial_lp_budget"]
        probe_protocol = {**protocol, "lp_budgets": {"candidate": budget}}
        _, probe_errors, _ = _audit_strict(probe, state, probe_protocol, model, lanes, "candidate", expected)
        errors.extend("probe: " + message for message in probe_errors)
        if (not probe["baseline_selection"]["score_exact"] or probe["score_proof"]["attempted"]
                or probe["score_proof"]["states_visited"] != 0
                or probe["score_proof"]["status"] != "SKIPPED_SCORE_CERTIFIED"):
            errors.append("integer probe requires a proved score and must skip strict integer proof")
        if (not _same(probe["baseline_selection"], stage["selection"])
                or not _same(probe["preprocessing"], stage["preprocessing"])
                or not _same(probe["capacity_propagation"], stage["capacity_propagation"])):
            errors.append("probe did not reuse its paid LP stage without additional preprocessing")
    if result["status"] != status or not _same(result["selection"], selected):
        errors.append("returned action or status differs from its actual LP/probe certificate")
    budget = protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, final["preprocessing"])
    if metric["failures"]:
        errors.append("final candidate has invalid feasibility, certificate or work accounting")
    return metric, errors, _exact(result["selection"], metric, expected)


def audit_result(result, state, protocol, model, lanes, label, expected):
    if label == "candidate":
        return audit_candidate(result, state, protocol, model, lanes, expected)
    if label == "probe":
        metric, errors, quality = _audit_probe(result, state, protocol, model, lanes, expected)
        if not _same(result, state["expected_probe_result"]):
            errors.append("probe control did not reproduce its frozen complete output")
        return metric, errors, quality
    if label != "long":
        raise ValueError("unregistered selector label")
    return _audit_strict(result, state, {**protocol, "lp_budgets": {label: protocol["long_lp_budget"]},
        "integer_state_budget": protocol["control_integer_state_budget"]}, model, lanes, label, expected)


def _reference_checks(state, protocol, option_count):
    expected, errors = _reference_audit(state, protocol, option_count)
    if state["score_unproved"] != (not state["expected_initial_lp"]["selection"]["score_exact"]):
        errors.append("score-unproved subgroup differs from the frozen initial certificate")
    old = state["expected_probe_result"]
    if not _same({key: old["initial_lp"][key] for key in state["expected_initial_lp"]}, state["expected_initial_lp"]):
        errors.append("saved probe control differs from the frozen initial LP snapshot")
    return expected, errors


def screen_audit(protocol, row, model):
    failures = []
    for state in row["states"]:
        expected, errors = _reference_checks(state, protocol, row["option_count"])
        metric, extra, quality = audit_candidate(state["screen_result"], state, protocol, model, row["lanes"], expected)
        errors.extend(extra)
        if not _same(metric, state["screen_metrics"]):
            errors.append("screen metric differs from the saved candidate output")
        if errors or not quality:
            failures.append({**_pointer(row, state), "errors": errors, "quality_passed": quality})
    return not failures, failures


def _work(results, label):
    work = (_strict_work if label == "long" else _candidate_work)(results)
    if label == "candidate":
        work["probe_stages"] = dict(Counter(result["probe_stage"] for result in results))
    return work


def _latencies(states):
    if not states:
        return {"queue_count": 0, "latency": None, "comparisons": None}
    latency = {label: _latency(states, label) for label in LABELS}
    for label in LABELS:
        ordered = sorted(median(state["wall_ns"][label]) for state in states)
        latency[label].update(p99_state_median_wall_ms=ordered[math.ceil(.99 * len(ordered)) - 1] / 1e6,
                              max_state_median_wall_ms=ordered[-1] / 1e6)
    comparisons = {}
    for label in ("probe", "long"):
        a, b = latency["candidate"], latency[label]
        comparisons[label] = {"wall_ratio": a["sum_state_median_wall_ns"] / b["sum_state_median_wall_ns"],
            "cpu_ratio": a["sum_state_median_cpu_ns"] / b["sum_state_median_cpu_ns"] if b["sum_state_median_cpu_ns"] else None,
            "p99_state_median_ratio": a["p99_state_median_wall_ms"] / b["p99_state_median_wall_ms"],
            "max_paired_state_slowdown": max(median(state["wall_ns"]["candidate"]) / median(state["wall_ns"][label]) for state in states)}
    return {"queue_count": len(states), "latency": latency, "comparisons": comparisons}


def _group(states, repeats):
    return {**_latencies(states),
        "metrics": {label: _combine_metrics(state["metrics"][label] for state in states) for label in LABELS},
        "work": {label: _work([result for state in states for result in _repeat_results(state, label, repeats)], label) for label in LABELS}}


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    actual = [(row["node"], row["lanes"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != {(node, lanes) for node in models for lanes in protocol["lanes"]}:
        raise ValueError("all registered cells must occur exactly once")
    entries, timed, failures, quality_failures, cells = [], [], [], [], []
    warmup_results = {label: [] for label in LABELS}
    stable = pinned = complete = True
    repeats = protocol["repeats"]
    for row in rows:
        model = models[row["node"]]
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(model["options"])):
            raise ValueError("cell registration or metadata differs from the model")
        roster = [(state["state_id"], panel["panel"], panel["cohort"], panel.get("seed"), tuple(state["queue"]))
                  for panel in protocol["queue_panels"][str(row["lanes"]) ] for state in panel["states"]]
        saved = [(state["state_id"], state["panel"], state["cohort"], state["seed"], tuple(state["queue"])) for state in row["states"]]
        if saved != roster or not roster or len(set(item[0] for item in saved)) != len(saved):
            raise ValueError("stored states differ from the full development roster")
        from algorithm.experiments.aligned_hardware_resumable_probe_development import timing_roster
        indices = protocol["timing_indices"][f"{row['node']}:{row['lanes']}"]
        historical = {"states": [{"completed": state["expected_control_result"]} for state in row["states"]]}
        if indices != timing_roster(historical, protocol["ordinary_per_cell"]) or [i for i, state in enumerate(row["states"]) if "first_results" in state] != indices:
            raise ValueError("timing roster differs from frozen triggers and ordinary states")
        pinned &= all(row[key]["cpu_affinity"] == [protocol["cpu"]] for key in
                      ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end"))
        cell_pass, cell_failures = screen_audit(protocol, row, model)
        if row["screen_quality_passed"] != cell_pass or not _same(row["screen_failures"], cell_failures):
            raise ValueError("stored screen outcome differs from actual outputs")
        keys = []
        for index, state in enumerate(row["states"]):
            expected, errors = _reference_checks(state, protocol, row["option_count"])
            keys.append(expected)
            complete &= expected is not None
            metric, extra, quality = audit_candidate(state["screen_result"], state, protocol, model, row["lanes"], expected)
            errors.extend(extra)
            if not _same(metric, state["screen_metrics"]):
                errors.append("screen metric differs from actual output")
            if not quality:
                quality_failures.append({**_pointer(row, state), "phase": "screen", "label": "candidate"})
            if index in indices:
                for label in LABELS:
                    for clock in ("wall_ns", "cpu_ns"):
                        values = state[clock][label]
                        if len(values) != repeats or any(type(v) is not int or v < (1 if clock == "wall_ns" else 0) for v in values):
                            raise ValueError("all registered wall/CPU measurements required")
                    results, metrics = _repeat_results(state, label, repeats), []
                    stable &= not state["repeat_mismatches"][label]
                    for repeat, result in enumerate(results):
                        metric, extra, quality = audit_result(result, state, protocol, model, row["lanes"], label, expected)
                        errors.extend(f"{label} repeat {repeat}: {message}" for message in extra)
                        metrics.append(metric)
                        if label == "candidate" and not _same(result, state["screen_result"]):
                            stable = False
                        if not quality:
                            quality_failures.append({**_pointer(row, state), "phase": "timing", "label": label, "repeat_index": repeat})
                    if not _same(_combine_metrics(metrics), state["metrics"][label]):
                        errors.append(f"{label} aggregate metrics differ from actual repeats")
                timed.append((row, state))
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entries.append((row, state))
        if len(row["warmups"]) != 3 * protocol["warmups_per_label"] or any(
                sum(item["label"] == label for item in row["warmups"]) != protocol["warmups_per_label"] for label in LABELS):
            raise ValueError("warmup count differs from registration")
        for item in row["warmups"]:
            label, index = item["label"], item["state_index"]
            if label not in LABELS or index != indices[0]:
                raise ValueError("warmups must use the first registered timing state")
            state = row["states"][index]
            metric, extra, quality = audit_result(item["result"], state, protocol, model, row["lanes"], label, keys[index])
            if not _same(metric, item["metrics"]):
                extra.append("warmup metric differs from saved result")
            if extra:
                failures.append({**_pointer(row, state), "phase": "warmup", "label": label, "errors": extra})
            if not quality:
                quality_failures.append({**_pointer(row, state), "phase": "warmup", "label": label})
            warmup_results[label].append(item["result"])
        cells.append({"node": row["node"], "lanes": row["lanes"], "screen_queue_count": len(row["states"]),
            "screen_quality_passed": cell_pass, **_group([row["states"][i] for i in indices], repeats),
            **{key: row[key] for key in ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")}})
    score = _group([state for _, state in timed if state["score_unproved"]], repeats)
    other = _group([state for _, state in timed if state["previous_integer_trigger"] and not state["score_unproved"]], repeats)
    trigger = _group([state for _, state in timed if state["previous_integer_trigger"]], repeats)
    ordinary = _group([state for _, state in timed if not state["previous_integer_trigger"]], repeats)
    def within(group, control, field, limit):
        return bool(group["queue_count"] and group["comparisons"][control][field] <= limit)
    gates = {
        "score_unproved_probe_gain": within(score, "probe", "wall_ratio", protocol["score_gain_limit"]),
        "score_unproved_probe_no_state_regression": within(score, "probe", "max_paired_state_slowdown", protocol["state_ratio_limit"]),
        "score_unproved_long_no_regression": within(score, "long", "wall_ratio", protocol["nonregression_limit"]),
        "other_trigger_probe_no_regression": within(other, "probe", "wall_ratio", protocol["nonregression_limit"]),
        "other_trigger_probe_no_state_regression": within(other, "probe", "max_paired_state_slowdown", protocol["state_ratio_limit"]),
        "trigger_long_no_regression": within(trigger, "long", "wall_ratio", protocol["trigger_long_ratio_limit"]),
        "trigger_long_p99_no_regression": within(trigger, "long", "p99_state_median_ratio", protocol["nonregression_limit"]),
        **{f"ordinary_{label}_no_regression": within(ordinary, label, "wall_ratio", protocol["nonregression_limit"]) for label in ("probe", "long")},
    }
    valid = not failures
    quality = bool(valid and complete and not quality_failures)
    execution = bool(valid and stable and pinned)
    advance = bool(quality and execution and all(gates.values()))
    decision = ("INVALID_EXECUTION" if not valid or not pinned else "INCOMPLETE_REFERENCE" if not complete else
        "CERTIFICATE_ROUTED_DEVELOPMENT_QUALITY_HOLD" if not quality else "INVALID_TIMING_EXECUTION" if not stable else
        "CERTIFICATE_ROUTED_DEVELOPMENT_ADVANCE" if advance else "CERTIFICATE_ROUTED_DEVELOPMENT_TIMING_HOLD")
    screen_work = _work([state["screen_result"] for _, state in entries], "candidate")
    all_timed = _group([state for _, state in timed], repeats)
    warmup_work = {label: _work(results, label) for label, results in warmup_results.items()}
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "queue_count": len(entries), "screen_selector_calls": len(entries),
        "timed_queue_count": len(timed), "timed_selector_calls": len(timed) * repeats * 3,
        "warmup_calls": sum(len(items) for items in warmup_results.values()),
        "executed_selector_calls": len(entries) + len(timed) * repeats * 3 + sum(len(items) for items in warmup_results.values()),
        "executed_reference_calls": 0, "executed_oracle_calls": 0, "reused_reference_count": len(entries),
        "reused_oracle_count": sum(state["oracle"] is not None for _, state in entries),
        "certificate_checks_passed": valid, "effective_references_complete": bool(complete),
        "quality_gate_passed": quality, "execution_checks_passed": execution, "repeat_paths_stable": bool(stable),
        "cpu_affinity_preserved": bool(pinned), "development_timing_gate_passed": advance, "timing_gates": gates,
        "accounting_failures": failures, "quality_failures": quality_failures,
        "screen_metrics": _combine_metrics(state["screen_metrics"] for _, state in entries), "screen_work": screen_work,
        "borrowed_probe_screen_work": _candidate_work([state["expected_probe_result"] for _, state in entries]),
        "timed": all_timed, "warmup_work": warmup_work, "score_unproved": score, "other_trigger": other,
        "prior_integer_trigger": trigger, "ordinary": ordinary,
        "trigger_states": [{**_pointer(row, state), **_latencies([state]), "score_unproved": state["score_unproved"],
                            "screen_status": state["screen_result"]["status"], "screen_work": _work([state["screen_result"]], "candidate")}
                           for row, state in timed if state["previous_integer_trigger"]],
        "executed_search_work": {key: screen_work["search_work"][key] + sum(all_timed["work"][label]["search_work"][key]
            + warmup_work[label]["search_work"][key] for label in LABELS) for key in COUNTERS},
        "cells": cells, "limitations": protocol["limitations"]}
