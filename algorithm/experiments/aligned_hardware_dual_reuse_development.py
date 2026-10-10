"""Develop certified parent-dual bounds on the frozen branch-reserved roster."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_root_pruning_development import load_cell

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_dual_reuse_development_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_development_20260928.json"
HELD_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_short_lookahead_development_20260928.json"
SOURCES = (
    "algorithm/dual_reuse_integer_capacity_frame_candidates.py",
    "algorithm/dual_reuse_certificate_routed_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_dual_reuse_development.py",
    "algorithm/experiments/dual_reuse_development_summary.py",
)
REFERENCE_FIELDS = ("reference", "reference_metrics", "reference_complete", "oracle",
                    "effective_reference_complete", "effective_reference_source")


def _caps(model):
    minima = {}
    for option in model.options:
        key = tuple(option.requirements)
        minima[key] = min(minima.get(key, float("inf")), max(1.0, option.duration))
    roots = len(set(minima.values()))
    return [roots, roots + 1, 144, 256]


def build_protocol(models):
    prior = json.loads(PRIOR_OUTPUT.read_text())
    previous = json.loads(PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json").read_text())
    held = json.loads(HELD_OUTPUT.read_text())
    held_protocol = json.loads(HELD_OUTPUT.with_name(HELD_OUTPUT.stem + "_protocol.json").read_text())
    if prior["decision"] != "BRANCH_RESERVED_DEVELOPMENT_ADVANCE" or not prior["development_gate_passed"]:
        raise ValueError("complete branch-reserved development is required")
    if held["decision"] != "SHORT_LOOKAHEAD_DEVELOPMENT_HOLD" or not all(held[key] for key in (
            "execution_checks_passed", "effective_references_complete", "routed_quality_gate_passed", "final_score_gate_passed")):
        raise ValueError("the quality-valid short-lookahead HOLD must be retained")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for old in (previous, held_protocol):
        if snapshots != old["models"]:
            raise ValueError("cached physical catalogue models changed")
        if tuple(old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("developed routing budgets must remain fixed")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = deepcopy(previous["cells"])
    for cell in cells:
        model = next(model for model in models if model.node == cell["node"])
        for index, state in enumerate(cell["states"]):
            state.update(prior_index=index, source_state_id=state["state_id"], caps=_caps(model))
    return json.loads(json.dumps({"schema_version": 1, "experiment": "dual_reuse_development_v1",
        "models": snapshots, "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": previous["reference_budget"], "oracle_state_budget": previous["oracle_state_budget"],
        "prior_source": {"path": str(PRIOR_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
        "historical_regression_keys": previous["historical_regression_keys"],
        "prior_decisions": {**held["prior_decisions"], "short_lookahead_development": held["decision"]},
        "expected_selector_calls": 1476, "expected_native_advance_calls": 2952,
        "expected_interface_calls": 4428, "expected_native_constructions": 2214,
        "execution_sources": sources,
        "roster": "Retain exactly the preceding738 development states,722 nonzero and16 zero controls, all queues/seeds and the separate34 historical keys. No new sampling, queue filtering or state replacement.",
        "reference_policy": "Borrow the six independently obtained fullK4096/reference-only oracle fields, the preceding branch-reserved four native snapshots, and its independently executed routed output. Execute a new native four-cap ladder and independent cold old/new routers for every state, including zeros. No new reference/oracle or old native ladder solves.",
        "rule": "Save the exactly repaired nonnegative feasible dual from each certified LP. Children retain the same eligible columns, matrix and objective; after integer-capacity tightening evaluate that dual with the child's shifted base objective and RHS, then take the minimum with its inherited bound and apply the existing safe score/canonical pruning. Failed LPs retain any valid parent certificate. Preserve root-slot reservation and the original certificate route144/256/8192, with at most one probe and no short lookahead. No new tuning parameter.",
        "accounting": "Every state executes four external native advance interfaces on one new search, then two cold routed selectors:4428 external interfaces and2214 native constructions. Each cumulative native256 endpoint is charged once, each routed final_lp once, and all spent integer states. Charge additional certified-dual saves, child bound/RHS evaluations, requirement/weight reads and successful pruning explicitly; they are arithmetic work, not LP calls or LP matrix coefficients. Borrowed work is separate and cannot supply this run's savings.",
        "gate": "Complete coverage, independent effective references, valid certificates/root reservation/cumulative accounting, both final canonical actions matching references, and no new service-score loss at any of M,M+1,144,256. In each nonzero cell, each native cap must have nonincreasing LP and matrix totals; routed LP/matrix/integer totals must separately be nonincreasing. Independently require routed and native-cap nonincrease in each cell's fixed historical subset. Pooled nonzero routed LP or integer work must strictly decrease. Zero controls cannot supply a gain, and integer savings cannot compensate for LP or matrix growth. Preserve every failure, loss and nonclosure; finish allcells without changing the rule, roster, budgets or gate.",
        "limitations": "Reused development queues, four cached catalogues and virtual resource counts. Extra exact arithmetic is reported; LP reduction alone is not a wall-time improvement. No new queue, physical scaling or flow-time claim. All preceding short-lookahead and timing HOLDs remain unchanged; the paper is not edited."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, roster, catalogue, references or rule changed")


def run_cell(model, cell, protocol, runtime, search_class=None, selectors=None, progress=True):
    if search_class is None:
        from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseIntegerCapacitySearch
        search_class = DualReuseIntegerCapacitySearch
    if selectors is None:
        from algorithm.branch_reserved_certificate_routed_frame_candidates import select_branch_reserved_certificate_routed_frame
        from algorithm.dual_reuse_certificate_routed_frame_candidates import select_dual_reuse_certificate_routed_frame
        selectors = {"old": select_branch_reserved_certificate_routed_frame, "new": select_dual_reuse_certificate_routed_frame}
    source = protocol["prior_source"]
    prior = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0,
        "new_reference_oracle_calls": 0, "new_old_routed_calls": 0}
    budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
    for index, registered in enumerate(cell["states"]):
        saved = prior["states"][registered["prior_index"]]
        if (saved["state_id"], saved["queue"], saved["source"], saved["caps"]) != (
                registered["source_state_id"], registered["queue"], registered["source"], registered["caps"]):
            raise ValueError("borrowed development state or budget ladder changed")
        queue, lanes = tuple(registered["queue"]), cell["lanes"]
        search = search_class(model.options, queue, lanes)
        traces = {str(cap): asdict(search.advance_to(cap)) for cap in registered["caps"]}
        old = selectors["old"](model.options, queue, lanes, *budgets)
        row["new_old_routed_calls"] += 1
        new = selectors["new"](model.options, queue, lanes, *budgets)
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "old_traces": saved["traces"]["new"], "expected_old_routed": saved["routed"],
            "traces": traces, "old_routed": asdict(old), "routed": asdict(new)})
        if progress and (index + 1 == len(cell["states"]) or registered["cohort"] != cell["states"][index + 1]["cohort"]):
            print(f"{model.node} R{lanes}: {registered['cohort']} complete ({index + 1}/{len(cell['states'])})", flush=True)
    row["runtime_end"] = runtime_provenance()
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
        runtime._write_new(protocol_path, protocol)
        print(json.dumps({"protocol": str(protocol_path), "states": sum(len(cell["states"]) for cell in protocol["cells"])}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("dual-reuse development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.dual_reuse_development_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
