"""Test integer-capacity propagation on the frozen reused development queues."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import scipy

from algorithm.experiments.aligned_hardware_fractional_shared_threshold_development import (
    COMPARISON_COUNTERS, COUNTERS, EXECUTION_SOURCES as PRIOR_EXECUTION_SOURCES,
    LANES, REFERENCE_BUDGET, ROOT, _cell_path, _combine_metrics, _metrics, _record,
    _write_new, enumerate_frames, load_catalogues, oracle, reference_key,
    select_canonical_count_space_bnb_frame,
    select_fractional_shared_threshold_canonical_requirement_dominance_frame,
    select_shared_threshold_canonical_requirement_dominance_frame, topology,
)
from algorithm.integer_capacity_shared_threshold_frame_candidates import (
    select_integer_capacity_shared_threshold_frame,
)


DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_development_20260927.json"
PRIOR_DEVELOPMENT = ROOT / "md/experiment_artifacts/aligned_hardware_fractional_shared_threshold_development_20260927_protocol.json"
CONTROLS = ("old_shared", "fractional_shared")
LABELS = ("reference", *CONTROLS, "integer_capacity")
CAPACITY_COUNTERS = ("region_visits", "requirement_coordinate_reads", "tightened_bounds", "removed_count_values")
EXECUTION_SOURCES = (*PRIOR_EXECUTION_SOURCES,
    "algorithm/integer_capacity_shared_threshold_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_integer_capacity_development.py")


def build_protocol(models):
    previous = json.loads(PRIOR_DEVELOPMENT.read_text(encoding="utf-8"))
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "integer_capacity_reused_queue_development_v1",
        "models": [model.snapshot() for model in models], "lanes": previous["lanes"],
        "queue_panels": previous["queue_panels"], "budgets": previous["budgets"],
        "reference_budget": previous["reference_budget"],
        "calls": "four independent cold calls/state: full K4096 reference, old shared K36R, fractional shared K36R, integer capacity K36R; no escalation or rescue",
        "accounting": "reference excluded; cold preprocessing and integer-capacity work recorded separately; failed LP calls charged; queues paired across nodes",
        "search_counters": COUNTERS, "capacity_counters": CAPACITY_COUNTERS,
        "oracle": previous["oracle"],
        "advance_gate": "complete references, valid certificates, integer capacity canonical exact and matching every score/action; integer capacity LP and matrix totals no greater than BOTH frozen controls in every node/R cell and both strictly lower pooled against BOTH; per-state regressions descriptive",
        "queue_source": str(PRIOR_DEVELOPMENT.relative_to(ROOT)),
        "retained_witnesses": [
            {"node": "gpu_a", "lanes": 8, "queues": [6, 6, 37]},
            {"node": "gpu_b", "lanes": 16, "queues": [35, 32, 77]},
        ],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8") for path in EXECUTION_SOURCES},
        "limitations": "Reused development queues, including failed former confirmation. Fixed cached point models and virtual identical-lane scaling; node-state comparisons are paired, not independent observations. Propagation has separately reported work. No new physical measurement, fresh confirmation, trajectory, flow-time, or total wall-time claim.",
    }))


def screen_cell(model, lanes, protocol, progress=True):
    started_cell = perf_counter()
    budget, reference_budget = protocol["budgets"][str(lanes)], protocol["reference_budget"]
    frames = enumerate_frames(model.options, lanes) if lanes == 4 else None
    panels = []
    for registered in protocol["queue_panels"][str(lanes)]:
        panel = {
            "panel": registered["panel"], "queue_count": len(registered["queues"]),
            "uncertified_reference_count": 0, "first_uncertified_reference": None,
            **{label: _metrics() for label in LABELS},
            "first_unresolved": dict.fromkeys(LABELS[1:]),
            "capacity_propagation": dict.fromkeys(CAPACITY_COUNTERS, 0),
            "cost_regression_counts": {control: dict.fromkeys(COUNTERS, 0) for control in CONTROLS},
            "worst_lp_regression": dict.fromkeys(CONTROLS), "retained_witnesses": [],
        }
        for raw_queue in registered["queues"]:
            queue = tuple(raw_queue)
            started = perf_counter()
            reference = select_canonical_count_space_bnb_frame(model.options, queue, lanes, reference_budget)
            elapsed = perf_counter() - started
            certified = reference_key(reference, queue, len(model.options))
            if certified is None:
                panel["uncertified_reference_count"] += 1
                if panel["first_uncertified_reference"] is None:
                    panel["first_uncertified_reference"] = {"queues": queue, "selection": asdict(reference)}
            expected = oracle(frames, queue) if frames is not None else certified
            _record(panel["reference"], reference, model, queue, lanes, reference_budget, expected, elapsed_s=elapsed)
            selections = {"reference": reference}
            for label, selector in (
                ("old_shared", select_shared_threshold_canonical_requirement_dominance_frame),
                ("fractional_shared", select_fractional_shared_threshold_canonical_requirement_dominance_frame),
                ("integer_capacity", select_integer_capacity_shared_threshold_frame),
            ):
                started = perf_counter()
                result = selector(model.options, queue, lanes, budget)
                elapsed, selected = perf_counter() - started, result.selection
                _record(panel[label], selected, model, queue, lanes, budget, expected,
                        asdict(result.preprocessing), reference, elapsed)
                selections[label] = selected
                if not selected.canonical_exact and panel["first_unresolved"][label] is None:
                    panel["first_unresolved"][label] = {"queues": queue, "selection": asdict(selected)}
                if label == "integer_capacity":
                    for key in CAPACITY_COUNTERS:
                        panel["capacity_propagation"][key] += getattr(result.capacity_propagation, key)
            new = selections["integer_capacity"]
            for control in CONTROLS:
                old = selections[control]
                for key in COUNTERS:
                    panel["cost_regression_counts"][control][key] += getattr(new, key) > getattr(old, key)
                previous, increase = panel["worst_lp_regression"][control], new.lp_calls - old.lp_calls
                if increase > (previous["increase"] if previous else 0):
                    panel["worst_lp_regression"][control] = {
                        "queues": queue, "increase": increase, control: old.lp_calls, "integer_capacity": new.lp_calls}
            if {"node": model.node, "lanes": lanes, "queues": list(queue)} in protocol["retained_witnesses"]:
                panel["retained_witnesses"].append({
                    "queues": queue, **{label: asdict(selected) for label, selected in selections.items()}})
        panels.append(panel)
        if progress:
            print(f"{model.node} R{lanes} {panel['panel']}: LP old/fractional/capacity="
                  f"{panel['old_shared']['search_work']['lp_calls']}/{panel['fractional_shared']['search_work']['lp_calls']}/"
                  f"{panel['integer_capacity']['search_work']['lp_calls']}; unresolved references={panel['uncertified_reference_count']}", flush=True)
    queue_count = sum(panel["queue_count"] for panel in panels)
    return {
        "node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
        "queue_count": queue_count, "option_count": len(model.options),
        "oracle_queue_count": queue_count if frames is not None else 0,
        "uncertified_reference_count": sum(panel["uncertified_reference_count"] for panel in panels),
        **{label: _combine_metrics(panel[label] for panel in panels) for label in LABELS},
        "capacity_propagation": {key: sum(panel["capacity_propagation"][key] for panel in panels) for key in CAPACITY_COUNTERS},
        "cost_regression_counts": {control: {
            key: sum(panel["cost_regression_counts"][control][key] for panel in panels) for key in COUNTERS} for control in CONTROLS},
        "panels": panels, "operational_elapsed_s": perf_counter() - started_cell,
        "runtime_versions": {"python": platform.python_version(), "scipy": scipy.__version__},
    }


def summarize(models, protocol, rows):
    expected = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual = [(row["node"], row["lanes"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("all registered node/R cells must appear exactly once")
    for row in rows:
        roster = [(panel["panel"], len(panel["queues"])) for panel in protocol["queue_panels"][str(row["lanes"])]]
        if [(panel["panel"], panel["queue_count"]) for panel in row["panels"]] != roster or row["queue_count"] != sum(count for _, count in roster):
            raise ValueError("cell panel coverage differs from registered queues")
    work = {label: {key: sum(row[label]["search_work"][key] for row in rows) for key in COUNTERS} for label in LABELS[1:]}
    valid = all(not row[label]["failures"] and row[label]["calls"] == row["queue_count"] for row in rows for label in LABELS)
    complete = not any(row["uncertified_reference_count"] for row in rows)
    actions = all(row["integer_capacity"]["canonical_exact_count"] == row["queue_count"]
                  and not row["integer_capacity"]["score_mismatch_count"]
                  and not row["integer_capacity"]["canonical_action_mismatch_count"] for row in rows)
    cost_by_control = {control: (
        all(row["integer_capacity"]["search_work"][key] <= row[control]["search_work"][key] for row in rows for key in COMPARISON_COUNTERS)
        and all(work["integer_capacity"][key] < work[control][key] for key in COMPARISON_COUNTERS)) for control in CONTROLS}
    cost = all(cost_by_control.values())
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "DEVELOPMENT_ACTION_REVIEW_REQUIRED" if not actions else
                "INTEGER_CAPACITY_DEVELOPMENT_ADVANCE" if cost else "SEARCH_COST_REVIEW_REQUIRED")
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models), "requirement_topology_count": len({topology(model) for model in models}),
        "distinct_lane_queue_count": sum(len(panel["queues"]) for panels in protocol["queue_panels"].values() for panel in panels),
        "queue_count": sum(row["queue_count"] for row in rows), "selector_calls": sum(row["queue_count"] for row in rows) * 4,
        "oracle_queue_count": sum(row["oracle_queue_count"] for row in rows),
        "certificate_checks_passed": valid, "references_complete": complete,
        "development_action_gate_passed": valid and complete and actions,
        "development_search_cost_gate_passed": cost, "search_cost_gate_by_control": cost_by_control, "decision": decision,
        "comparison_search_work": work,
        "capacity_propagation": {key: sum(row["capacity_propagation"][key] for row in rows) for key in CAPACITY_COUNTERS},
        "reference_search_work": {key: sum(row["reference"]["search_work"][key] for row in rows) for key in COUNTERS},
        "cells": rows, "limitations": protocol["limitations"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--cell", help="registered NODE:R cell")
    mode.add_argument("--merge-cells", action="store_true")
    args = parser.parse_args()
    models = load_catalogues()
    expected = build_protocol(models)
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        expected["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_new(protocol_path, expected)
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered models, execution source, or protocol changed; register a new output")
    if args.cell:
        node, lanes_text = args.cell.rsplit(":", 1)
        lanes = int(lanes_text)
        model = next(model for model in models if model.node == node)
        if lanes not in protocol["lanes"]:
            raise ValueError("unregistered lane count")
        output = _cell_path(args.output, node, lanes)
        if output.exists():
            raise FileExistsError(output)
        _write_new(output, {"protocol": protocol, "cell": screen_cell(model, lanes, protocol)})
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for model in models:
        for lanes in protocol["lanes"]:
            if args.merge_cells:
                saved = json.loads(_cell_path(args.output, model.node, lanes).read_text(encoding="utf-8"))
                if saved["protocol"] != protocol:
                    raise ValueError("cell protocol differs from registered protocol")
                rows.append(saved["cell"])
            else:
                rows.append(screen_cell(model, lanes, protocol))
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    _write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "comparison_search_work", "capacity_propagation")}))


if __name__ == "__main__":
    main()
