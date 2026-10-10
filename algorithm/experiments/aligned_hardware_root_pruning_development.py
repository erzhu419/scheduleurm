"""Frozen work-count development of root pruning and its finite-budget behavior."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_root_pruning_development_20260928.json"
DEVELOPMENT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_development_20260927.json"
CONFIRMATION = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_confirmation_20260927.json"
SOURCES = (
    "algorithm/root_pruned_integer_capacity_frame_candidates.py",
    "algorithm/root_pruned_certificate_routed_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_root_pruning_development.py",
    "algorithm/experiments/root_pruning_development_summary.py",
)


def load_cell(path, node, lanes, registration):
    with gzip.open(_cell_path(path, node, lanes), "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    if saved["registered_at_utc"] != registration:
        raise ValueError("borrowed cell differs from its frozen registration")
    return saved["cell"]


def build_protocol(models):
    old = json.loads(DEVELOPMENT.with_name(DEVELOPMENT.stem + "_protocol.json").read_text())
    recent = json.loads(CONFIRMATION.with_name(CONFIRMATION.stem + "_protocol.json").read_text())
    old_result = json.loads(DEVELOPMENT.read_text())
    recent_result = json.loads(CONFIRMATION.read_text())
    if not all(result[key] for result in (old_result, recent_result) for key in (
            "certificate_checks_passed", "effective_references_complete", "execution_checks_passed")):
        raise ValueError("complete quality-valid frozen predecessors are required")
    if (json.loads(json.dumps([model.snapshot() for model in models])) != old["models"]
            or recent["models"] != old["models"]):
        raise ValueError("cached physical models changed")
    for path, literal in old["execution_sources"].items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
    triggers = {(key["node"], key["lanes"], tuple(key["queue"])) for key in recent["historical_trigger_keys"]}
    cells = []
    for node, lanes in old["execution_order"]:
        development = load_cell(DEVELOPMENT, node, lanes, old["registered_at_utc"])
        confirmation = load_cell(CONFIRMATION, node, lanes, recent["registered_at_utc"])
        states, seen, panel_counts = [], set(), {}
        for index, state in enumerate(development["states"]):
            cohort = state["cohort"]
            count = panel_counts.get(state["panel"], 0)
            panel_counts[state["panel"]] = count + 1
            selected = (cohort == "earlier_development" and count < 4) or cohort not in (
                "earlier_development", "reused_confirmation")
            if selected:
                states.append({"state_id": state["state_id"], "queue": state["queue"],
                    "source": "development", "source_index": index, "cohort": cohort})
                seen.add(tuple(state["queue"]))
        for index, state in enumerate(confirmation["states"]):
            queue = tuple(state["queue"])
            if ((node, lanes, queue) in triggers or state["cohort"] == "ordinary_tail_regression") and queue not in seen:
                states.append({"state_id": state["state_id"], "queue": state["queue"],
                    "source": "confirmation", "source_index": index, "cohort": "fixed_extra_regression"})
                seen.add(queue)
        zero = [0] * len(models[0].options[0].requirements)
        if tuple(zero) in seen:
            raise ValueError("zero queue already occurs in the sampled roster")
        states.append({"state_id": "zero_queue", "queue": zero, "source": "new_zero", "cohort": "zero_queue"})
        cells.append({"node": node, "lanes": lanes, "states": states})
    return {"schema_version": 1, "experiment": "root_pruning_work_development_v1",
        "models": old["models"], "cells": cells, "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_budget": 4096,
        "sources": {"development": {"path": str(DEVELOPMENT.relative_to(ROOT)), "registered_at_utc": old["registered_at_utc"]},
                    "confirmation": {"path": str(CONFIRMATION.relative_to(ROOT)), "registered_at_utc": recent["registered_at_utc"]}},
        "prior_decisions": {"development": old_result["decision"], "confirmation": recent_result["decision"]},
        "execution_sources": {**old["execution_sources"], **{path: (ROOT / path).read_text() for path in SOURCES}},
        "roster": "First four queues from each of four earlier development seed panels per R/node; every existing hard/neighborhood/known-completion development regression; all18 registered confirmation triggers and16 ordinary tails not already included; one zero queue per cell. Deduplicate by node/R/Q before observing new outputs.",
        "budget_ladder": "One old and one new resumable search per state, at duration_root_count, duration_root_count+1,144,256. Continue each frontier without restarting; charge only its final cumulative snapshot once. Independently invoke the new routed144/256/8192 selector once. Reuse old routed results and references; zero queues alone receive a new fullK4096 reference and old routed control.",
        "gate": "Valid certificates/accounting and effective references everywhere; new routed action canonical exact and matching reference everywhere. On nonzero queues: routed LP/matrix/integer totals nonincreasing in every cell with a strict pooled LP reduction. No new-kernel score regression against old at any registered ladder budget. Preserve every regression and nonclosure. Zero queues are correctness controls and cannot supply the strict work improvement.",
        "limitations": "Reused development/regression data and fixed cached catalogues with virtual lanes. Root bounds certify exact-duration strata; skipping an LP may remove shorter-duration heuristic incumbents and change finite-budget paths. No latency comparison, fresh confirmation, physical scaling or flow-time claim; previous HOLD retained."}


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen root-pruning protocol changed")


def run_cell(model, cell, protocol, runtime):
    from algorithm.resumable_integer_capacity_frame_candidates import IntegerCapacitySearch
    from algorithm.root_pruned_integer_capacity_frame_candidates import RootPrunedIntegerCapacitySearch
    from algorithm.root_pruned_certificate_routed_frame_candidates import select_root_pruned_certificate_routed_frame
    from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
    sources = {name: load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
               for name, source in protocol["sources"].items()}
    row = {"node": model.node, "lanes": cell["lanes"], "registered_at_utc": protocol["registered_at_utc"],
           "states": [], "new_reference_calls": 0, "new_old_routed_calls": 0}
    minima = {}
    for option in model.options:
        key = tuple(option.requirements)
        minima[key] = min(minima.get(key, float("inf")), max(1.0, option.duration))
    roots = len(set(minima.values()))
    caps = (roots, roots + 1, 144, 256)
    for registered in cell["states"]:
        queue, lanes = tuple(registered["queue"]), cell["lanes"]
        if registered["source"] == "new_zero":
            reference = runtime.select_canonical_count_space_bnb_frame(model.options, queue, lanes, 4096)
            old_routed = select_certificate_routed_frame(model.options, queue, lanes, 144, 256, 8192)
            borrowed = {"reference": asdict(reference), "oracle": None,
                        "effective_reference_complete": runtime.reference_key(reference, queue, len(model.options)) is not None}
            row["new_reference_calls"] += 1
            row["new_old_routed_calls"] += 1
        else:
            saved = sources[registered["source"]]["states"][registered["source_index"]]
            if (saved["state_id"], saved["queue"]) != (registered["state_id"], registered["queue"]):
                raise ValueError("borrowed state changed")
            borrowed = {key: saved[key] for key in ("reference", "oracle", "effective_reference_complete")}
            old_routed = saved["screen_result"] if registered["source"] == "development" else saved["screen_results"]["candidate"]
        traces = {}
        for label, search in (("old", IntegerCapacitySearch(model.options, queue, lanes)),
                              ("new", RootPrunedIntegerCapacitySearch(model.options, queue, lanes))):
            traces[label] = {str(cap): asdict(search.advance_to(cap)) for cap in caps}
        routed = select_root_pruned_certificate_routed_frame(model.options, queue, lanes, 144, 256, 8192)
        row["states"].append({**registered, **borrowed, "caps": caps, "traces": traces,
            "old_routed": old_routed if isinstance(old_routed, dict) else asdict(old_routed), "routed": asdict(routed)})
    return json.loads(json.dumps(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    runtime = runtime_setup()
    models = runtime.load_catalogues()
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        with protocol_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, separators=(",", ":"), allow_nan=False)
        print(json.dumps({"protocol": str(protocol_path), "states": sum(len(cell["states"]) for cell in protocol["cells"])}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
        print(f"{model.node} R{cell['lanes']}: {len(row['states'])} states complete", flush=True)
    from algorithm.experiments.root_pruning_development_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
