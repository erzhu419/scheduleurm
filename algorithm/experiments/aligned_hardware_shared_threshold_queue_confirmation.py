"""Frozen fresh-queue confirmation of the four-catalogue 36R search comparison.

Queues are paired across nodes; the 3,072 node-state comparisons are not
independent samples. Only the queue panel is new: models and algorithms stay
fixed, and larger lane counts still represent virtual identical-lane scaling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import platform
import random
from time import perf_counter

import scipy

from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import (
    COUNTERS, LANES, QUEUE_LEVELS, REFERENCE_BUDGET, ROOT, _cell_path, _metrics,
    _record, _write_new, enumerate_frames, load_catalogues, oracle, reference_key,
    scaled_queues, select_canonical_count_space_bnb_frame,
    select_shared_threshold_canonical_requirement_dominance_frame, topology,
)


DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_shared_threshold_queue_confirmation_20260927.json"
PANEL_SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
QUEUES_PER_PANEL = 64
COMPARISON_COUNTERS = ("lp_calls", "lp_matrix_coefficients")
EXECUTION_SOURCES = (
    "algorithm/canonical_count_space_bnb_frame_candidates.py",
    "algorithm/shared_threshold_canonical_requirement_dominance_frame_candidates.py",
    "algorithm/requirement_dominance_frame_candidates.py",
    "algorithm/threshold_canonical_requirement_dominance_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_shared_threshold_scaling_screen.py",
    "algorithm/experiments/aligned_hardware_physical_catalogue_screen.py",
    "algorithm/experiments/aligned_hardware_shared_threshold_queue_confirmation.py",
)


def sample_queues(lanes, panel_seed, count=QUEUES_PER_PANEL, *, excluded=()):
    """Uniform integer triples, conditioned on novelty, without replacement."""
    forbidden = set(product(QUEUE_LEVELS, repeat=3)) | set(scaled_queues(lanes))
    forbidden.update(tuple(queue) for queue in excluded)
    available = (5 * lanes + 1) ** 3 - sum(all(0 <= q <= 5 * lanes for q in queue) for queue in forbidden)
    if count < 0 or count > available:
        raise ValueError("requested panel exceeds available fresh queues")
    rng = random.Random(panel_seed + 1000 * lanes)
    queues = []
    while len(queues) < count:
        queue = tuple(rng.randint(0, 5 * lanes) for _ in range(3))
        if queue not in forbidden:
            queues.append(queue)
            forbidden.add(queue)
    return tuple(queues)


def build_protocol(models):
    panels = {}
    for lanes in LANES:
        seen, panels[str(lanes)] = set(), []
        for seed in PANEL_SEEDS:
            queues = sample_queues(lanes, seed, excluded=seen)
            panels[str(lanes)].append({"seed": seed, "queues": queues})
            seen.update(queues)
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "shared_threshold_physical_catalogue_fresh_queue_confirmation_v1",
        "models": [model.snapshot() for model in models], "lanes": LANES,
        "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL, "queue_panels": panels,
        "sampler": "Python random.Random(panel_seed + 1000*R); each coordinate randint(0,5*R); reject native/scaled development grids and earlier draws across panels within R; conditional uniform without replacement",
        "reference_budget": REFERENCE_BUDGET, "budgets": {str(r): 36 * r for r in LANES},
        "calls": "three independent cold calls/state: full K4096 reference, full K36R, shared K36R; no reference escalation or rescue",
        "accounting": "reference excluded; cold preprocessing separate; failed LP calls charged; panels paired across all four models",
        "search_counters": COUNTERS,
        "oracle": "R4 full-catalogue Fraction multiset enumeration; R8/R16 certified canonical full K4096 only; unresolved references retained",
        "advance_gate": "complete references, valid certificates, both deployment selectors canonical exact and matching every action/score; shared LP and matrix totals no greater in every node/R cell and both strictly lower pooled; per-seed costs descriptive only",
        "prior_development": "md/experiment_artifacts/aligned_hardware_shared_threshold_scaling_screen_development_20260927.json",
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8") for path in EXECUTION_SOURCES},
        "limitations": "Fresh queues conditional on a fixed sampling law, with fixed cached point models and virtual identical-lane scaling. Nodes share queues; node-state comparisons are not independent observations. No new physical measurement, joint95 transfer, trajectory, flow-time, wall-time, or low-budget deployment claim.",
    }))


def _combine_metrics(items):
    total = _metrics()
    for item in items:
        for key, value in item.items():
            if key in ("first_failure", "first_mismatch"):
                total[key] = total[key] if total[key] is not None else value
            elif key in ("search_work", "preprocessing"):
                for counter, amount in value.items():
                    total[key][counter] = total[key].get(counter, 0) + amount
            elif key.startswith("max_"):
                total[key] = max(total[key], value)
            else:
                total[key] += value
    return total


def screen_cell(model, lanes, protocol, progress=True):
    started_cell = perf_counter()
    budget, reference_budget = protocol["budgets"][str(lanes)], protocol["reference_budget"]
    frames = enumerate_frames(model.options, lanes) if lanes == 4 else None
    panels = []
    for registered in protocol["queue_panels"][str(lanes)]:
        panel = {
            "seed": registered["seed"], "queue_count": len(registered["queues"]),
            "uncertified_reference_count": 0, "first_uncertified_reference": None,
            **{label: _metrics() for label in ("reference", "full", "shared")},
            "shared_cost_regression_counts": dict.fromkeys(COUNTERS, 0),
            "worst_shared_lp_regression": None,
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
            started = perf_counter()
            full = select_canonical_count_space_bnb_frame(model.options, queue, lanes, budget)
            elapsed = perf_counter() - started
            _record(panel["full"], full, model, queue, lanes, budget, expected, reference=reference, elapsed_s=elapsed)
            started = perf_counter()
            result = select_shared_threshold_canonical_requirement_dominance_frame(model.options, queue, lanes, budget)
            elapsed, shared = perf_counter() - started, result.selection
            _record(panel["shared"], shared, model, queue, lanes, budget, expected, asdict(result.preprocessing), reference, elapsed)
            for key in COUNTERS:
                panel["shared_cost_regression_counts"][key] += getattr(shared, key) > getattr(full, key)
            previous = panel["worst_shared_lp_regression"]
            increase = shared.lp_calls - full.lp_calls
            if increase > (previous["increase"] if previous else 0):
                panel["worst_shared_lp_regression"] = {"queues": queue, "increase": increase, "full": full.lp_calls, "shared": shared.lp_calls}
        panels.append(panel)
        if progress:
            print(f"{model.node} R{lanes} seed {panel['seed']}: LP full/shared={panel['full']['search_work']['lp_calls']}/{panel['shared']['search_work']['lp_calls']}; unresolved={panel['uncertified_reference_count']}", flush=True)
    witnesses = [panel["worst_shared_lp_regression"] for panel in panels if panel["worst_shared_lp_regression"]]
    queue_count = sum(panel["queue_count"] for panel in panels)
    return {
        "node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
        "queue_count": queue_count, "option_count": len(model.options),
        "oracle_queue_count": queue_count if frames is not None else 0,
        "uncertified_reference_count": sum(panel["uncertified_reference_count"] for panel in panels),
        "first_uncertified_reference": next((panel["first_uncertified_reference"] for panel in panels if panel["first_uncertified_reference"]), None),
        **{label: _combine_metrics(panel[label] for panel in panels) for label in ("reference", "full", "shared")},
        "shared_cost_regression_counts": {key: sum(panel["shared_cost_regression_counts"][key] for panel in panels) for key in COUNTERS},
        "worst_shared_lp_regression": max(witnesses, key=lambda item: item["increase"], default=None),
        "panels": panels, "operational_elapsed_s": perf_counter() - started_cell,
        "runtime_versions": {"python": platform.python_version(), "scipy": scipy.__version__},
    }


def summarize(models, protocol, rows):
    expected = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual = [(row["node"], row["lanes"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("all registered node/R cells must appear exactly once")
    for row in rows:
        registered = protocol["queue_panels"][str(row["lanes"])]
        roster = [(panel["seed"], len(panel["queues"])) for panel in registered]
        if [(panel["seed"], panel["queue_count"]) for panel in row["panels"]] != roster or row["queue_count"] != sum(count for _, count in roster):
            raise ValueError("cell panel coverage differs from registered queues")
    work = {label: {key: sum(row[label]["search_work"][key] for row in rows) for key in COUNTERS} for label in ("full", "shared")}
    valid = all(not row[label]["failures"] and row[label]["calls"] == row["queue_count"] for row in rows for label in ("reference", "full", "shared"))
    complete = not any(row["uncertified_reference_count"] for row in rows)
    actions = all(row[label]["canonical_exact_count"] == row["queue_count"]
                  and not row[label]["score_mismatch_count"] and not row[label]["canonical_action_mismatch_count"]
                  for row in rows for label in ("full", "shared"))
    cost = (all(row["shared"]["search_work"][key] <= row["full"]["search_work"][key] for row in rows for key in COMPARISON_COUNTERS)
            and all(work["shared"][key] < work["full"][key] for key in COMPARISON_COUNTERS))
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "DEPLOYMENT_ACTION_REVIEW_REQUIRED" if not actions else
                "FRESH_QUEUE_CONFIRMATION_ADVANCE" if cost else "SEARCH_COST_REVIEW_REQUIRED")
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models), "requirement_topology_count": len({topology(model) for model in models}),
        "distinct_lane_queue_count": sum(len(panel["queues"]) for panels in protocol["queue_panels"].values() for panel in panels),
        "queue_count": sum(row["queue_count"] for row in rows), "selector_calls": sum(row["queue_count"] for row in rows) * 3,
        "oracle_queue_count": sum(row["oracle_queue_count"] for row in rows),
        "certificate_checks_passed": valid, "references_complete": complete,
        "deployment_action_gate_passed": valid and complete and actions,
        "deployment_search_cost_gate_passed": cost, "decision": decision,
        "comparison_search_work": work,
        "reference_search_work": {key: sum(row["reference"]["search_work"][key] for row in rows) for key in COUNTERS},
        "per_seed_search_work": {str(seed): {label: {key: sum(panel[label]["search_work"][key] for row in rows for panel in row["panels"] if panel["seed"] == seed) for key in COUNTERS} for label in ("full", "shared")} for seed in protocol["panel_seeds"]},
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
