"""Frozen development of a bounded all-unsolved LP prefix before a probe."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_short_lookahead_development_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_development_20260928.json"
TIMING_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_timing_20260928.json"
SOURCES = (
    "algorithm/short_lookahead_certificate_routed_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_short_lookahead_development.py",
    "algorithm/experiments/short_lookahead_development_summary.py",
)


def build_protocol(models):
    prior = json.loads(PRIOR_OUTPUT.read_text())
    source = json.loads(PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json").read_text())
    timing = json.loads(TIMING_OUTPUT.read_text())
    timing_protocol = json.loads(TIMING_OUTPUT.with_name(TIMING_OUTPUT.stem + "_protocol.json").read_text())
    if prior["decision"] != "BRANCH_RESERVED_DEVELOPMENT_ADVANCE" or not prior["development_gate_passed"]:
        raise ValueError("complete branch-reserved development is required")
    if timing["decision"] != "BRANCH_RESERVED_TIMING_HOLD" or not all(timing[key] for key in (
            "certificate_checks_passed", "effective_references_complete", "candidate_quality_gate_passed",
            "comparison_quality_gate_passed", "execution_checks_passed")):
        raise ValueError("the quality-valid branch-reserved timing HOLD must be retained")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for old in (source, timing_protocol):
        if snapshots != old["models"]:
            raise ValueError("cached physical catalogue models changed")
        if tuple(old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("developed routing budgets must remain fixed")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = deepcopy(source["cells"])
    for cell in cells:
        for index, state in enumerate(cell["states"]):
            state.update(prior_index=index, source_state_id=state["state_id"])
    return json.loads(json.dumps({"schema_version": 1, "experiment": "short_lookahead_development_v1",
        "models": snapshots, "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": source["reference_budget"], "oracle_state_budget": source["oracle_state_budget"],
        "prior_source": {"path": str(PRIOR_OUTPUT.relative_to(ROOT)), "registered_at_utc": source["registered_at_utc"]},
        "historical_regression_keys": source["historical_regression_keys"],
        "prior_decisions": {**timing["prior_decisions"], "branch_reserved_timing": timing["decision"]},
        "expected_selector_calls": 1476, "execution_sources": sources,
        "roster": "Exactly the preceding738 development states:722 nonzero states and16 zero controls, with all seeds/queues and the separate34 historical keys retained. No timing-panel queues are added, except the historical regressions already in development. No new sampling or state replacement.",
        "reference_policy": "Borrow the six independently obtained fullK4096/reference-only oracle fields from the matched branch-reserved development state, plus independently executed native144/native256 snapshots and the frozen preceding routed output. No new reference, oracle or native budget-ladder execution: the LP kernel is unchanged. Execute both routing arms independently.",
        "rule": "Only an initial proved score with an unresolved canonical action and a nonempty entirely unsolved retained frontier permits short lookahead. Increase the same search's cap byone eachstep, for at most the initial retained frontier size and never beyond256. Stop immediately on canonical closure or a surviving solved region. Probe the latest prefix once if needed; an exhausted probe is never repeated, and LP continuation remains bounded by256 with root-slot reservation. No queue-specific choice or new tuning parameter.",
        "accounting": "Two cold routing calls and two native search constructions perstate:1476 each. Count each actual LP advance interface separately, including short steps, but charge only final cumulative LP/preprocessing/capacity work once perrouting call. All spent integer states are charged. Prefix snapshots and borrowed native snapshots are embedded/borrowed views, not extra new searches. Zero new reference or oracle solves.",
        "gate": "Complete frozen coverage and effective independent references, valid certificates, structural routing, reserved budgets and cumulative accounting. Both final arms must be canonical exact and reference matching, with no final service-score loss. Relative to the direct branch-reserved predecessor, each nonzero cell and each separate historical cell must have nonincreasing LP calls, matrix coefficients and integer states; pooled nonzero integer work must strictly decrease. Zero states cannot supply gains. An integer saving cannot compensate for more LP or matrix work. Retain every loss or nonclosure and finish allcells without changing the rule, budgets, cohort or gate.",
        "limitations": "Reused development queues and selected historical regressions on four cached catalogues with virtual resource counts. No new queue, wall-time, physical scaling or flow-time claim. The native kernel and its four-budget evidence remain unchanged. Earlier root-pruning and both certificate-routed/branch-reserved timing HOLDs are retained."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, development roster, references or rule changed")


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    if selectors is None:
        from algorithm.branch_reserved_certificate_routed_frame_candidates import select_branch_reserved_certificate_routed_frame
        from algorithm.short_lookahead_certificate_routed_frame_candidates import select_short_lookahead_certificate_routed_frame
        selectors = {"old": select_branch_reserved_certificate_routed_frame, "new": select_short_lookahead_certificate_routed_frame}
    source = protocol["prior_source"]
    prior = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0,
        "new_reference_oracle_calls": 0, "new_old_routed_calls": 0}
    budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
    for index, registered in enumerate(cell["states"]):
        saved = prior["states"][registered["prior_index"]]
        if (saved["state_id"], saved["queue"], saved["source"]) != (
                registered["source_state_id"], registered["queue"], registered["source"]):
            raise ValueError("borrowed development state differs from its registered source")
        queue, lanes = tuple(registered["queue"]), cell["lanes"]
        old = selectors["old"](model.options, queue, lanes, *budgets)
        row["new_old_routed_calls"] += 1
        new = selectors["new"](model.options, queue, lanes, *budgets)
        fields = {key: saved[key] for key in ("reference", "reference_metrics", "reference_complete", "oracle",
            "effective_reference_complete", "effective_reference_source")}
        row["states"].append({**registered, **fields,
            "expected_initial_lp": saved["traces"]["new"]["144"],
            "expected_native_final": saved["traces"]["new"]["256"],
            "expected_old_routed": saved["routed"], "old_routed": asdict(old), "routed": asdict(new)})
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
        raise FileExistsError("short-lookahead development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.short_lookahead_development_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
