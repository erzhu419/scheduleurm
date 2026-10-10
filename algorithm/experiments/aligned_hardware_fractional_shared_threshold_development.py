"""Compare fractional-first branching on reused four-catalogue development queues.

The old scaled grid and the completed queue confirmation are now development
inputs. This comparison is not an independent confirmation of the new search.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import scipy

from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import (
    COUNTERS, LANES, REFERENCE_BUDGET, ROOT, _cell_path, _metrics, _record,
    _write_new, enumerate_frames, load_catalogues, oracle, reference_key,
    scaled_queues, select_canonical_count_space_bnb_frame,
    select_shared_threshold_canonical_requirement_dominance_frame, topology,
)
from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import _combine_metrics
from algorithm.fractional_shared_threshold_canonical_requirement_dominance_frame_candidates import (
    select_fractional_shared_threshold_canonical_requirement_dominance_frame,
)


DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_fractional_shared_threshold_development_20260927.json"
PRIOR_CONFIRMATION = ROOT / "md/experiment_artifacts/aligned_hardware_shared_threshold_queue_confirmation_20260927_protocol.json"
LABELS = ("reference", "old_shared", "fractional_shared")
COMPARISON_COUNTERS = ("lp_calls", "lp_matrix_coefficients")
EXECUTION_SOURCES = (
    "algorithm/canonical_count_space_bnb_frame_candidates.py",
    "algorithm/shared_threshold_canonical_requirement_dominance_frame_candidates.py",
    "algorithm/fractional_shared_threshold_canonical_requirement_dominance_frame_candidates.py",
    "algorithm/requirement_dominance_frame_candidates.py",
    "algorithm/threshold_canonical_requirement_dominance_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_physical_catalogue_models.py",
    "algorithm/experiments/aligned_hardware_physical_catalogue_screen.py",
    "algorithm/experiments/aligned_hardware_shared_threshold_scaling_screen.py",
    "algorithm/experiments/aligned_hardware_shared_threshold_queue_confirmation.py",
    "algorithm/experiments/aligned_hardware_fractional_shared_threshold_development.py",
)


def build_protocol(models):
    previous = json.loads(PRIOR_CONFIRMATION.read_text(encoding="utf-8"))
    panels = {str(lanes): [{"panel": "scaled_grid", "queues": scaled_queues(lanes)}] + [
        {"panel": f"former_confirmation_{panel['seed']}", "queues": panel["queues"]}
        for panel in previous["queue_panels"][str(lanes)]] for lanes in LANES}
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "fractional_shared_threshold_reused_queue_development_v1",
        "models": [model.snapshot() for model in models], "lanes": LANES, "queue_panels": panels,
        "reference_budget": REFERENCE_BUDGET, "budgets": {str(r): 36 * r for r in LANES},
        "calls": "three independent cold calls/state: full K4096 reference, old shared K36R, fractional shared K36R; no escalation or rescue",
        "accounting": "reference excluded; cold preprocessing separate; failed LP calls charged; queues paired across nodes",
        "search_counters": COUNTERS,
        "oracle": "R4 full-catalogue Fraction multiset enumeration; R8/R16 certified canonical full K4096; unresolved references retained",
        "advance_gate": "complete references, valid certificates, fractional shared canonical exact and matching every score/action; fractional shared LP and matrix totals no greater than old shared in every node/R cell and both strictly lower pooled; per-state regressions descriptive",
        "queue_sources": ["md/experiment_artifacts/aligned_hardware_shared_threshold_scaling_screen_development_20260927.json", str(PRIOR_CONFIRMATION.relative_to(ROOT))],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8") for path in EXECUTION_SOURCES},
        "limitations": "All queues are reused development inputs, including the failed former confirmation. Fixed cached point models and virtual identical-lane scaling; node-state comparisons are paired, not independent observations. No new physical measurement, fresh confirmation, trajectory, flow-time, or wall-time claim.",
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
            "first_unresolved_fractional": None,
            "fractional_cost_regression_counts": dict.fromkeys(COUNTERS, 0),
            "worst_fractional_lp_regression": None,
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
            selections = {}
            for label, selector in (
                ("old_shared", select_shared_threshold_canonical_requirement_dominance_frame),
                ("fractional_shared", select_fractional_shared_threshold_canonical_requirement_dominance_frame),
            ):
                started = perf_counter()
                result = selector(model.options, queue, lanes, budget)
                elapsed, selected = perf_counter() - started, result.selection
                _record(panel[label], selected, model, queue, lanes, budget, expected,
                        asdict(result.preprocessing), reference, elapsed)
                selections[label] = selected
            old, new = selections["old_shared"], selections["fractional_shared"]
            if not new.canonical_exact and panel["first_unresolved_fractional"] is None:
                panel["first_unresolved_fractional"] = {"queues": queue, "selection": asdict(new)}
            for key in COUNTERS:
                panel["fractional_cost_regression_counts"][key] += getattr(new, key) > getattr(old, key)
            previous, increase = panel["worst_fractional_lp_regression"], new.lp_calls - old.lp_calls
            if increase > (previous["increase"] if previous else 0):
                panel["worst_fractional_lp_regression"] = {"queues": queue, "increase": increase, "old_shared": old.lp_calls, "fractional_shared": new.lp_calls}
        panels.append(panel)
        if progress:
            print(f"{model.node} R{lanes} {panel['panel']}: LP old/fractional={panel['old_shared']['search_work']['lp_calls']}/{panel['fractional_shared']['search_work']['lp_calls']}; unresolved references={panel['uncertified_reference_count']}", flush=True)
    queue_count = sum(panel["queue_count"] for panel in panels)
    return {
        "node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
        "queue_count": queue_count, "option_count": len(model.options),
        "oracle_queue_count": queue_count if frames is not None else 0,
        "uncertified_reference_count": sum(panel["uncertified_reference_count"] for panel in panels),
        **{label: _combine_metrics(panel[label] for panel in panels) for label in LABELS},
        "fractional_cost_regression_counts": {key: sum(panel["fractional_cost_regression_counts"][key] for panel in panels) for key in COUNTERS},
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
    actions = all(row["fractional_shared"]["canonical_exact_count"] == row["queue_count"]
                  and not row["fractional_shared"]["score_mismatch_count"]
                  and not row["fractional_shared"]["canonical_action_mismatch_count"] for row in rows)
    cost = (all(row["fractional_shared"]["search_work"][key] <= row["old_shared"]["search_work"][key] for row in rows for key in COMPARISON_COUNTERS)
            and all(work["fractional_shared"][key] < work["old_shared"][key] for key in COMPARISON_COUNTERS))
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "DEVELOPMENT_ACTION_REVIEW_REQUIRED" if not actions else
                "FRACTIONAL_BRANCH_DEVELOPMENT_ADVANCE" if cost else "SEARCH_COST_REVIEW_REQUIRED")
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models), "requirement_topology_count": len({topology(model) for model in models}),
        "distinct_lane_queue_count": sum(len(panel["queues"]) for panels in protocol["queue_panels"].values() for panel in panels),
        "queue_count": sum(row["queue_count"] for row in rows), "selector_calls": sum(row["queue_count"] for row in rows) * 3,
        "oracle_queue_count": sum(row["oracle_queue_count"] for row in rows),
        "certificate_checks_passed": valid, "references_complete": complete,
        "development_action_gate_passed": valid and complete and actions,
        "development_search_cost_gate_passed": cost, "decision": decision,
        "comparison_search_work": work,
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
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "comparison_search_work")}))


if __name__ == "__main__":
    main()
