"""Independent queue confirmation with fixed historical routing and tail groups."""
from __future__ import annotations

from collections import defaultdict

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.certificate_routed_development_summary import (
    LABELS, audit_result, result_preprocessing, _latencies, _group as _state_group, _work,
)
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _pointer, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _repeat_results, _same

CONTROLS = ("probe", "long")


def _keys(items):
    return {(item["node"], item["lanes"], tuple(item["queue"])) for item in items}


def _group(entries, repeats):
    return _state_group([state for _, state in entries], repeats)


def _screen_group(entries):
    return {"queue_count": len(entries),
        "metrics": {label: _combine_metrics(state["screen_metrics"][label] for _, state in entries)
                    for label in ("candidate", "probe")},
        "work": {label: _work([state["screen_results"][label] for _, state in entries], label)
                 for label in ("candidate", "probe")}}


def _timing_indices(protocol, node, lanes, trigger_keys):
    selected, index = [], 0
    for panel in protocol["queue_panels"][str(lanes)]:
        for local, state in enumerate(panel["states"]):
            if ((panel["cohort"] == "fresh_confirmation" and local < protocol["timing_fresh_per_panel"])
                    or (node, lanes, tuple(state["queue"])) in trigger_keys
                    or panel["cohort"] == "ordinary_tail_regression"):
                selected.append(index)
            index += 1
    return selected


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    actual = [(row["node"], row["lanes"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != {(node, lanes) for node in models for lanes in protocol["lanes"]}:
        raise ValueError("all registered cells must occur exactly once")
    trigger_keys = _keys(protocol["historical_trigger_keys"])
    score_keys = _keys(protocol["score_unproved_keys"])
    prior_untimed_keys = _keys(protocol["previous_untimed_probe_keys"])
    if not score_keys <= trigger_keys or not prior_untimed_keys <= trigger_keys or score_keys & prior_untimed_keys:
        raise ValueError("registered regression subgroups must be disjoint subsets of known triggers")
    entries, timed, cells, failures, quality_failures = [], [], [], [], []
    cohorts, seeds = defaultdict(list), defaultdict(list)
    warmup_results, warmup_metrics = ({label: [] for label in LABELS} for _ in range(2))
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
        saved = [(state["state_id"], state["panel"], state["cohort"], state["seed"], tuple(state["queue"])) for state in row["states"]]
        if saved != roster or not roster or len({item[0] for item in saved}) != len(saved):
            raise ValueError("stored states differ from the complete registered roster")
        indices = _timing_indices(protocol, node, lanes, trigger_keys)
        if (protocol["timing_indices"][f"{node}:{lanes}"] != indices
                or [i for i, state in enumerate(row["states"]) if "first_results" in state] != indices):
            raise ValueError("all and only preregistered states must have three-arm timing")
        pinned &= all(row[key]["cpu_affinity"] == [protocol["cpu"]] for key in
                      ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end"))
        references, cell_timed = [], []
        for index, state in enumerate(row["states"]):
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            expected, errors = _reference(state, protocol, row["option_count"])
            references.append(expected)
            complete &= expected is not None
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
                errors.append("independent reference has invalid budget or certificate")
            key = node, lanes, tuple(state["queue"])
            for field, keys in (("previous_integer_trigger", trigger_keys), ("score_unproved_regression", score_keys),
                                ("previous_untimed_probe_trigger", prior_untimed_keys)):
                if state[field] != (key in keys):
                    errors.append(f"{field} differs from its frozen regression keys")
            probe = state["screen_results"]["probe"]
            initial = {key: probe["initial_lp"][key] for key in ("selection", "preprocessing", "capacity_propagation")}
            if not _same(initial, state["expected_initial_lp"]):
                errors.append("initial LP comparator must come from the independent probe screen call")
            if not _same(probe, state["expected_probe_result"]):
                errors.append("expected probe output must be the independent screen result")
            if state["score_unproved"] != (not initial["selection"]["score_exact"]):
                errors.append("current score-unproved diagnostic differs from its initial certificate")
            for label in ("candidate", "probe"):
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
                            raise ValueError("all registered wall/CPU measurements required")
                    results, metrics = _repeat_results(state, label, repeats), []
                    stable &= not state["repeat_mismatches"][label]
                    for repeat, result in enumerate(results):
                        metric, extra, quality = audit_result(result, state, protocol, model, lanes, label, expected)
                        errors.extend(f"{label} repeat {repeat}: {message}" for message in extra)
                        metrics.append(metric)
                        if label in ("candidate", "probe") and not _same(result, state["screen_results"][label]):
                            stable = False
                        if not quality:
                            quality_failures.append({**_pointer(row, state), "phase": "timing", "label": label, "repeat_index": repeat})
                    if not _same(_combine_metrics(metrics), state["metrics"][label]):
                        errors.append(f"{label} repeat metrics differ from actual measured calls")
                timed.append((row, state))
                cell_timed.append((row, state))
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entries.append((row, state))
            cohorts[state["cohort"]].append((row, state))
        if (len(row["warmups"]) != len(LABELS) * protocol["warmups_per_label"] or any(
                sum(item["label"] == label for item in row["warmups"]) != protocol["warmups_per_label"] for label in LABELS)):
            raise ValueError("warmup counts differ from registration")
        for item in row["warmups"]:
            label, index = item["label"], item["state_index"]
            if label not in LABELS or index != indices[0]:
                raise ValueError("warmups must use the first registered timing state")
            state = row["states"][index]
            metric, errors, quality = audit_result(item["result"], state, protocol, model, lanes, label, references[index])
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
    known = [entry for entry in timed if entry[1]["previous_integer_trigger"]]
    score = [entry for entry in known if entry[1]["score_unproved_regression"]]
    other = [entry for entry in known if not entry[1]["score_unproved_regression"]]
    untimed_before = [entry for entry in known if entry[1]["previous_untimed_probe_trigger"]]
    tails = [entry for entry in timed if entry[1]["cohort"] == "ordinary_tail_regression"]
    if {(row["node"], row["lanes"], tuple(state["queue"])) for row, state in known} != trigger_keys:
        raise ValueError("every fixed historical trigger must occur in timing")
    groups = {name: _group(items, repeats) for name, items in (
        ("fresh_timing", fresh), ("historical_trigger_timing", known), ("score_unproved_timing", score),
        ("other_trigger_timing", other), ("previous_untimed_probe_timing", untimed_before), ("ordinary_tail_timing", tails))}
    for entry in fresh:
        seeds[entry[1]["seed"]].append(entry)
    fresh_group = groups["fresh_timing"]
    fresh_gates = {label: {
        "pooled_no_material_regression": bool(fresh and fresh_group["comparisons"][label]["wall_ratio"] <= protocol["fresh_nonregression_limit"]),
        "no_cell_material_regression": bool(fresh and all(cell["fresh_timed"]["comparisons"][label]["wall_ratio"] <= protocol["fresh_nonregression_limit"]
                                                        for cell in cells if cell["fresh_timed"]["queue_count"])),
        "p99_no_material_regression": bool(fresh and fresh_group["comparisons"][label]["p99_state_median_ratio"] <= protocol["fresh_nonregression_limit"]),
    } for label in CONTROLS}
    def within(name, label, field, limit):
        group = groups[name]
        return bool(group["queue_count"] and group["comparisons"][label][field] <= limit)
    regression_gates = {
        "score_unproved_probe_gain": within("score_unproved_timing", "probe", "wall_ratio", protocol["score_gain_limit"]),
        "score_unproved_probe_no_state_regression": within("score_unproved_timing", "probe", "max_paired_state_slowdown", protocol["state_ratio_limit"]),
        "score_unproved_long_no_regression": within("score_unproved_timing", "long", "wall_ratio", protocol["nonregression_limit"]),
        "other_trigger_probe_no_regression": within("other_trigger_timing", "probe", "wall_ratio", protocol["nonregression_limit"]),
        "other_trigger_probe_no_state_regression": within("other_trigger_timing", "probe", "max_paired_state_slowdown", protocol["state_ratio_limit"]),
        "trigger_long_no_regression": within("historical_trigger_timing", "long", "wall_ratio", protocol["trigger_long_ratio_limit"]),
        "trigger_long_p99_no_regression": within("historical_trigger_timing", "long", "p99_state_median_ratio", protocol["nonregression_limit"]),
        **{f"tail_no_state_regression_vs_{label}": within("ordinary_tail_timing", label, "max_paired_state_slowdown", protocol["tail_state_limit"]) for label in CONTROLS},
    }
    valid = not failures
    candidate_quality = bool(valid and complete and not any(item["label"] == "candidate" for item in quality_failures))
    comparison_quality = bool(valid and complete and not quality_failures)
    execution = bool(valid and stable and pinned)
    timing_passed = all(all(gate.values()) for gate in fresh_gates.values()) and all(regression_gates.values())
    advance = bool(execution and candidate_quality and comparison_quality and timing_passed)
    decision = ("INVALID_EXECUTION" if not valid or not pinned else "INCOMPLETE_REFERENCE" if not complete else
        "CERTIFICATE_ROUTED_CONFIRMATION_QUALITY_HOLD" if not candidate_quality or not comparison_quality else
        "INVALID_TIMING_EXECUTION" if not stable else "CERTIFICATE_ROUTED_CONFIRMATION_ADVANCE" if advance else
        "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD")
    screen, measured = _screen_group(entries), _group(timed, repeats)
    warmup_work = {label: _work(results, label) for label, results in warmup_results.items()}
    oracles = [state["oracle"] for _, state in entries if state["oracle"] is not None]
    method_calls = 2 * len(entries) + len(LABELS) * repeats * len(timed) + sum(len(items) for items in warmup_results.values())
    def state_views(items):
        return [{**_pointer(row, state), **_latencies([state])} for row, state in items]
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
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
        "warmup_metrics": {label: _combine_metrics(metrics) for label, metrics in warmup_metrics.items()},
        "screen_by_cohort": {cohort: _screen_group(items) for cohort, items in cohorts.items()},
        **groups, "fresh_by_seed": {str(seed): _group(items, repeats) for seed, items in seeds.items()},
        "historical_trigger_states": state_views(known), "ordinary_tail_states": state_views(tails),
        "previous_untimed_probe_states": state_views(untimed_before),
        "fresh_worst_paired_states": {label: sorted(state_views(fresh), key=lambda item: item["comparisons"][label]["wall_ratio"], reverse=True)[:10] for label in CONTROLS},
        "reference_work": _combine_metrics(state["reference_metrics"] for _, state in entries),
        "original_incomplete_reference_count": sum(not state["reference_complete"] for _, state in entries),
        "oracle_work": {key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS},
        "executed_method_search_work": {key: sum(screen["work"][label]["search_work"][key] for label in ("candidate", "probe"))
            + sum(measured["work"][label]["search_work"][key] + warmup_work[label]["search_work"][key] for label in LABELS) for key in COUNTERS},
        "cells": cells, "limitations": protocol["limitations"]}
