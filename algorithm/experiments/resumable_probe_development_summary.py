"""Validate one cumulative LP search and its preregistered development timing."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_integer_capacity_development import (
    CAPACITY_COUNTERS, COUNTERS, _combine_metrics, _metrics, _record,
)
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.score_face_completion_confirmation_summary import _pointer, _reference, _score
from algorithm.experiments.strict_score_proof_timing_summary import (
    LABELS, _audit_result as _audit_strict, _latencies, _repeat_results, _same, _work as _strict_work,
)


def result_preprocessing(result, label):
    return result["final_lp"]["preprocessing"] if label == "candidate" else result["preprocessing"]


def _metric(selected, state, model, lanes, budget, expected, preprocessing):
    metrics = _metrics()
    runtime_model = SimpleNamespace(options=tuple(SimpleNamespace(**option) for option in model["options"]))
    _record(metrics, selection_from_record(selected), runtime_model, tuple(state["queue"]), lanes,
            budget, expected, preprocessing, selection_from_record(state["reference"]))
    return metrics


def _exact(selected, metric, expected):
    return bool(expected is not None and selected["exact"] and selected["score_exact"]
                and selected["canonical_exact"] and not metric["score_mismatch_count"]
                and not metric["canonical_action_mismatch_count"])


def audit_candidate(result, state, protocol, model, lanes, expected):
    initial, final, probe = result["initial_lp"], result["final_lp"], result["probe"]
    errors = []
    for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget"):
        if result[key] != protocol[key]:
            errors.append(f"candidate {key} differs from registration")
    if not _same({key: initial[key] for key in state["expected_initial_lp"]}, state["expected_initial_lp"]):
        errors.append("initial LP snapshot did not reproduce the historical selection and paid work")
    initial_closed = initial["selection"]["canonical_exact"]
    if (probe is None) != initial_closed:
        errors.append("integer probe must occur exactly when initial LP is not canonical exact")
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
        errors.append("resumption changed the fixed catalogue preprocessing or thresholds")
    if any(final["selection"][key] < initial["selection"][key] for key in COUNTERS):
        errors.append("final cumulative LP work lost an initial charged operation")
    if any(final["capacity_propagation"][key] < initial["capacity_propagation"][key] for key in CAPACITY_COUNTERS):
        errors.append("final cumulative capacity work lost an initial charged operation")
    if probe is not None:
        probe_protocol = {**protocol, "lp_budgets": {"candidate": protocol["initial_lp_budget"]}}
        _, probe_errors, _ = _audit_strict(probe, state, probe_protocol, model, lanes, "candidate", expected)
        errors.extend("probe: " + message for message in probe_errors)
        if (not _same(probe["baseline_selection"], initial["selection"])
                or not _same(probe["preprocessing"], initial["preprocessing"])
                or not _same(probe["capacity_propagation"], initial["capacity_propagation"])):
            errors.append("probe must use the paid initial LP snapshot without a second LP or preprocessing")
    should_resume = probe is not None and not probe["selection"]["canonical_exact"]
    if result["resumed"] != should_resume:
        errors.append("resume flag differs from integer-probe nonclosure")
    if result["resumed"]:
        selected = dict(final["selection"])
        if probe is not None and probe["selection"]["score_exact"]:
            if _score(selected, state["queue"]) != _score(probe["selection"], state["queue"]):
                errors.append("resumed action contradicts the probe-proved optimum")
            if not selected["score_exact"]:
                selected.update(score_exact=True, upper_bound=selected["score"], gap=0., epsilon=0.)
        status = "LP_RESUME_COMPLETE" if selected["canonical_exact"] else "LP_RESUME_INCOMPLETE"
    else:
        if not _same(initial, final):
            errors.append("without resumption final LP snapshot must equal initial snapshot")
        selected = initial["selection"] if probe is None else probe["selection"]
        status = "LP_INITIAL_COMPLETE" if probe is None else "INTEGER_PROBE_COMPLETE"
    if result["status"] != status or not _same(result["selection"], selected):
        errors.append("returned action or status differs from its actual LP/probe certificate")
    budget = protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, final["preprocessing"])
    if metric["failures"]:
        errors.append("final candidate has invalid feasibility, certificate or work accounting")
    return metric, errors, _exact(result["selection"], metric, expected)


def _audit_control(result, state, protocol, model, lanes, expected):
    control_protocol = {**protocol, "lp_budgets": {"control": protocol["initial_lp_budget"]},
                        "integer_state_budget": protocol["control_integer_state_budget"]}
    metric, errors, quality = _audit_strict(result, state, control_protocol, model, lanes, "control", expected)
    if not _same(result, state["expected_control_result"]):
        errors.append("timed control did not reproduce its complete frozen strict-selector result")
    return metric, errors, quality


def _reference_audit(state, protocol, option_count):
    _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
    expected, errors = _reference(state, protocol, option_count)
    if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
        errors.append("saved reference has invalid budget or certificate")
    old = state["expected_control_result"]
    if state["previous_integer_trigger"] != bool(old["score_proof"]["attempted"] or old["completion"]["attempted"]):
        errors.append("timing subgroup differs from the frozen historical integer trigger")
    if not _same(state["expected_initial_lp"], {"selection": old["baseline_selection"],
            "preprocessing": old["preprocessing"], "capacity_propagation": old["capacity_propagation"]}):
        errors.append("historical initial snapshot differs from the frozen control")
    return expected, errors


def screen_audit(protocol, row, model):
    failures = []
    for state in row["states"]:
        expected, errors = _reference_audit(state, protocol, row["option_count"])
        metric, candidate_errors, quality = audit_candidate(state["screen_result"], state, protocol, model, row["lanes"], expected)
        errors.extend(candidate_errors)
        if not _same(metric, state["screen_metrics"]):
            errors.append("screen metric does not equal the actual saved candidate output")
        if errors or not quality:
            failures.append({**_pointer(row, state), "errors": errors, "quality_passed": quality})
    return not failures, failures


def _candidate_work(results):
    probes = [result["probe"] for result in results if result["probe"] is not None]
    probe_work = _strict_work(probes)
    return {
        "calls": len(results), "statuses": dict(Counter(result["status"] for result in results)),
        "resumed_calls": sum(result["resumed"] for result in results),
        "search_work": {key: sum(result["final_lp"]["selection"][key] for result in results) for key in COUNTERS},
        "initial_lp_calls_embedded": sum(result["initial_lp"]["selection"]["lp_calls"] for result in results),
        "additional_lp_calls": sum(result["final_lp"]["selection"]["lp_calls"] -
                                   result["initial_lp"]["selection"]["lp_calls"] for result in results),
        "preprocessing": {key: sum(result["final_lp"]["preprocessing"][key] for result in results)
                          for key in (results[0]["final_lp"]["preprocessing"] if results else ())},
        "capacity_propagation": {key: sum(result["final_lp"]["capacity_propagation"][key] for result in results)
                                 for key in CAPACITY_COUNTERS},
        **{key: probe_work[key] for key in ("score_proof", "completion", "shared_integer_states", "max_shared_integer_states")},
    }


def _group(states, repeats):
    return {**_latencies(states), "work": {
        label: (_candidate_work if label == "candidate" else _strict_work)(
            [result for state in states for result in _repeat_results(state, label, repeats)])
        for label in LABELS},
        "metrics": {label: _combine_metrics(state["metrics"][label] for state in states) for label in LABELS}}


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    expected_cells = {(node, lanes) for node in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(set(actual_cells)) != len(actual_cells) or set(actual_cells) != expected_cells:
        raise ValueError("all registered cells must occur exactly once")
    entries, timed, failures, quality_failures, cell_summaries = [], [], [], [], []
    warmup_results = {label: [] for label in LABELS}
    stable = pinned = complete = True
    repeats = protocol["repeats"]
    for row in rows:
        model = models[row["node"]]
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(model["options"])):
            raise ValueError("cell registration or metadata differs from the frozen model")
        expected_roster = [(state["state_id"], panel["panel"], panel["cohort"], panel.get("seed"), tuple(state["queue"]))
                           for panel in protocol["queue_panels"][str(row["lanes"]) ] for state in panel["states"]]
        actual_roster = [(state["state_id"], state["panel"], state["cohort"], state["seed"], tuple(state["queue"])) for state in row["states"]]
        if actual_roster != expected_roster or not actual_roster or len({state[0] for state in actual_roster}) != len(actual_roster):
            raise ValueError("stored states differ from the full registered development roster")
        from algorithm.experiments.aligned_hardware_resumable_probe_development import timing_roster
        indices = protocol["timing_indices"][f"{row['node']}:{row['lanes']}"]
        historical_view = {"states": [{"completed": state["expected_control_result"]} for state in row["states"]]}
        if indices != timing_roster(historical_view, protocol["ordinary_per_cell"]):
            raise ValueError("timing indices differ from prior triggers and first ordinary states")
        actual_timed_indices = [i for i, state in enumerate(row["states"]) if "first_results" in state]
        if actual_timed_indices != indices:
            raise ValueError("all and only preregistered states must have paired timing")
        pinned &= all(row[key]["cpu_affinity"] == [protocol["cpu"]] for key in
                      ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end"))
        cell_pass, cell_failures = screen_audit(protocol, row, model)
        if row["screen_quality_passed"] != cell_pass or not _same(row["screen_failures"], cell_failures):
            raise ValueError("stored screen outcome differs from actual saved outputs")
        keys = []
        for index, state in enumerate(row["states"]):
            expected, errors = _reference_audit(state, protocol, row["option_count"])
            keys.append(expected)
            complete &= expected is not None
            metric, extra, quality = audit_candidate(state["screen_result"], state, protocol, model, row["lanes"], expected)
            errors.extend(extra)
            if not _same(metric, state["screen_metrics"]):
                errors.append("screen metric differs from actual candidate output")
            if not quality:
                quality_failures.append({**_pointer(row, state), "phase": "screen", "label": "candidate"})
            if index in indices:
                for label in LABELS:
                    for clock in ("wall_ns", "cpu_ns"):
                        values = state[clock][label]
                        if len(values) != repeats or any(type(v) is not int or v < (1 if clock == "wall_ns" else 0) for v in values):
                            raise ValueError("all registered paired wall/CPU observations required")
                    results = _repeat_results(state, label, repeats)
                    stable &= not state["repeat_mismatches"][label]
                    metrics = []
                    for repeat, result in enumerate(results):
                        metric, extra, quality = (audit_candidate if label == "candidate" else _audit_control)(
                            result, state, protocol, model, row["lanes"], expected)
                        errors.extend(f"{label} repeat {repeat}: {message}" for message in extra)
                        metrics.append(metric)
                        if label == "candidate" and not _same(result, state["screen_result"]):
                            stable = False
                        if not quality:
                            quality_failures.append({**_pointer(row, state), "phase": "timing", "label": label, "repeat_index": repeat})
                    if not _same(_combine_metrics(metrics), state["metrics"][label]):
                        errors.append(f"{label} aggregate metrics differ from actual measured repeats")
                timed.append((row, state))
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entries.append((row, state))
        if (len(row["warmups"]) != 2 * protocol["warmups_per_label"] or
                any(sum(item["label"] == label for item in row["warmups"]) != protocol["warmups_per_label"] for label in LABELS)):
            raise ValueError("warmup count differs from registration")
        for item in row["warmups"]:
            label, index = item["label"], item["state_index"]
            if label not in LABELS or index != indices[0]:
                raise ValueError("warmups must use the first registered timing state")
            state = row["states"][index]
            metric, errors, quality = (audit_candidate if label == "candidate" else _audit_control)(
                item["result"], state, protocol, model, row["lanes"], keys[index])
            if not _same(metric, item["metrics"]):
                errors.append("warmup metric differs from the saved actual result")
            if errors:
                failures.append({**_pointer(row, state), "phase": "warmup", "label": label, "errors": errors})
            if not quality:
                quality_failures.append({**_pointer(row, state), "phase": "warmup", "label": label})
            warmup_results[label].append(item["result"])
        cell_summaries.append({"node": row["node"], "lanes": row["lanes"], "screen_queue_count": len(row["states"]),
                               "screen_quality_passed": cell_pass,
                               **_group([row["states"][i] for i in indices], repeats),
                               **{key: row[key] for key in ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")}})
    trigger = [state for _, state in timed if state["previous_integer_trigger"]]
    ordinary = [state for _, state in timed if not state["previous_integer_trigger"]]
    trigger_group, ordinary_group = _group(trigger, repeats), _group(ordinary, repeats)
    timing_gates = {
        "trigger_pooled_gain": bool(trigger and trigger_group["comparison"]["wall_ratio"] <= protocol["trigger_ratio_limit"]),
        "no_trigger_state_regression": bool(trigger and trigger_group["comparison"]["max_paired_state_slowdown"] <= protocol["trigger_state_ratio_limit"]),
        "ordinary_no_material_regression": bool(ordinary and ordinary_group["comparison"]["wall_ratio"] <= protocol["ordinary_ratio_limit"]),
    }
    valid = not failures
    quality = bool(valid and complete and not quality_failures)
    execution = bool(valid and stable and pinned)
    advance = bool(quality and execution and all(timing_gates.values()))
    decision = ("INVALID_EXECUTION" if not valid or not pinned else "INCOMPLETE_REFERENCE" if not complete else
                "RESUMABLE_PROBE_DEVELOPMENT_QUALITY_HOLD" if not quality else "INVALID_TIMING_EXECUTION" if not stable else
                "RESUMABLE_PROBE_DEVELOPMENT_ADVANCE" if advance else "RESUMABLE_PROBE_DEVELOPMENT_TIMING_HOLD")
    screen_work = _candidate_work([state["screen_result"] for _, state in entries])
    timed_work = _group([state for _, state in timed], repeats)
    warmup_work = {label: (_candidate_work if label == "candidate" else _strict_work)(results)
                   for label, results in warmup_results.items()}
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "queue_count": len(entries), "screen_selector_calls": len(entries),
        "timed_queue_count": len(timed), "timed_selector_calls": len(timed) * repeats * 2,
        "warmup_calls": sum(len(v) for v in warmup_results.values()),
        "executed_selector_calls": len(entries) + len(timed) * repeats * 2 + sum(len(v) for v in warmup_results.values()),
        "executed_reference_calls": 0, "executed_oracle_calls": 0,
        "reused_reference_count": len(entries), "reused_oracle_count": sum(state["oracle"] is not None for _, state in entries),
        "reused_original_incomplete_reference_count": sum(not state["reference_complete"] for _, state in entries),
        "certificate_checks_passed": valid, "effective_references_complete": bool(complete),
        "quality_gate_passed": quality, "execution_checks_passed": execution,
        "repeat_paths_stable": bool(stable), "cpu_affinity_preserved": bool(pinned),
        "development_timing_gate_passed": advance, "timing_gates": timing_gates,
        "accounting_failures": failures, "quality_failures": quality_failures,
        "screen_metrics": _combine_metrics(state["screen_metrics"] for _, state in entries),
        "screen_work": screen_work, "timed": timed_work, "warmup_work": warmup_work,
        "prior_integer_trigger": trigger_group, "ordinary": ordinary_group,
        "trigger_states": [{**_pointer(row, state), **_latencies([state]),
                            "previous_integer_states": state["expected_control_result"]["score_proof"]["states_visited"] + state["expected_control_result"]["completion"]["states_visited"],
                            "screen_status": state["screen_result"]["status"],
                            "screen_work": _candidate_work([state["screen_result"]])}
                           for row, state in timed if state["previous_integer_trigger"]],
        "executed_search_work": {key: screen_work["search_work"][key] + sum(
            timed_work["work"][label]["search_work"][key] + warmup_work[label]["search_work"][key] for label in LABELS) for key in COUNTERS},
        "cells": cell_summaries, "limitations": protocol["limitations"],
    }
