"""Reuse paid axis capacity in canonical completion on the frozen 746 states."""
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
from algorithm.experiments.aligned_hardware_dual_reuse_development import _caps, REFERENCE_FIELDS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_residual_capacity_frontier_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_axis_capacity_frontier_development_20260928.json"
SOURCES = (
    "algorithm/residual_capacity_frontier_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_residual_capacity_frontier_development.py",
    "algorithm/experiments/residual_capacity_frontier_development_summary.py",
)


def build_protocol(models):
    result = json.loads(PREVIOUS_OUTPUT.read_text())
    previous = json.loads(PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    if (result["decision"] != "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"
            or result["routed_quality_gate_passed"] or not all(result[key] for key in (
                "execution_checks_passed", "effective_references_complete", "predecessor_work_gate_passed"))):
        raise ValueError("the complete axis-capacity development quality HOLD is required")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    if snapshots != previous["models"] or tuple(previous[key] for key in (
            "initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
        raise ValueError("frozen cached catalogues or routing budgets changed")
    sources = dict(previous["execution_sources"])
    for path, literal in sources.items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = deepcopy(previous["cells"])
    for cell in cells:
        model = next(model for model in models if model.node == cell["node"])
        for index, state in enumerate(cell["states"]):
            if state["caps"] != _caps(model):
                raise ValueError("borrowed native budget ladder changed")
            state.update(borrowed_from="axis_capacity_development", borrowed_index=index,
                         source_state_id=state["state_id"], source_record_source=state["source"])
    count = sum(len(cell["states"]) for cell in cells)
    if count != previous["expected_state_count"] or count != result["state_count"]:
        raise ValueError("the complete prior development roster is required")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "residual_capacity_frontier_development_v1",
        "models": snapshots, "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": previous["reference_budget"], "oracle_state_budget": previous["oracle_state_budget"],
        "borrowed_sources": {"axis_capacity_development": {
            "path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]}},
        "historical_regression_keys": previous["historical_regression_keys"],
        "failure_development_keys": previous["failure_development_keys"],
        "axis_failure_development_keys": previous["axis_failure_development_keys"],
        "expected_state_count": count,
        "expected_nonzero_state_count": previous["expected_nonzero_state_count"],
        "expected_failure_development_count": previous["expected_failure_development_count"],
        "expected_axis_failure_development_count": previous["expected_axis_failure_development_count"],
        "expected_selector_calls": count * 2, "expected_interface_calls": count * 2,
        "expected_native_constructions": count * 2, "expected_native_advance_calls": 0,
        "expected_borrowed_native_snapshots": count * 8,
        "prior_decisions": {**result["prior_decisions"], "axis_capacity_development": result["decision"]},
        "execution_sources": sources,
        "roster": "Reuse the entire previous746-state axis-capacity development roster, four cached catalogues and capacities8/16/32/64. Retain34 historical regressions,16 zeros,four Q54,58,290 controls andfour Q120,123,258 failure-informed controls with their original seeds and sources. No queue or reference resampling.",
        "rule": "Keep native routing, strict proof and original canonical threshold/index/count order. Retain each already paid axis table's final exact-slot merged numerators. Canonical completion may use only a table at its exact threshold whose requirements cover all allowed original positive options. Build U[s]=max(merged[0:s+1]) under the remaining integer budget. Reject a canonical prefix only if its needed numerator exceeds U[remaining slots]; equal upper bounds still enter the original search. The full original queue and full eligible catalogue relax suffix eligibility, residual resources and duration attainment, so the table only bounds completion and supplies no action. With no usable table retain the frozen canonical path.",
        "reference_policy": "Borrow each matched six-field independent reference, eight native snapshots and two prior cold routers from the complete axis-capacity development result. Execute only two new independent cold residual-capacity routers per state. No new reference, oracle, previous router or native ladder.",
        "accounting": "1492 actual cold selectors/native constructions and5968 borrowed native snapshots. Keep all existing axis/merge/strict/canonical fees. Charge retained merge-vector copies, requirement coverage reads, prefix reads and capacity queries explicitly. Each prefix output cell consumes one additional state, including on exhaustion. The8192 shared total is score-proof axis+merge+fallback DFS plus canonical prefix+DFS states. Retaining a table never rebuilds the paid DP during canonical completion. Charge cumulative final LP work once.",
        "gate": "Complete original roster and effective independent references; both actions score/canonical exact and reference matching in every state. Reproduce same-arm paid LP snapshots, resumption and frontier evidence. Ordinary paths keep their complete prior outputs. Frontier paths without a matching completion-threshold table keep the prior selection and all existing work counters. An examined table whose requirements do not cover the original catalogue retains the frozen completion search and charges actual coverage reads. Existing score-proof states and arithmetic equal prior; retained merge vectors and safe coverage are auditable. Audit prefix plus canonical DFS with no budget reset, no omitted fee and no action supplied by a bound. Each state/arm separately has equal LP/matrix work and nonincreasing total shared integer states relative to its prior axis router. Preserve failed/exhausted attempts and every prior HOLD; no post-outcome rule, budget, roster or gate changes.",
        "limitations": "Failure-informed development on reused queues and four cached catalogues at virtual capacities. Borrowed references/native endpoints are not newly executed gains. This protocol does not establish independent confirmation, wall-time, flow-time or physical scaling. The paper remains untouched.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, roster, references, catalogue or residual-capacity rule changed")


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    if selectors is None:
        from algorithm.residual_capacity_frontier_frame_candidates import (
            select_branch_reserved_residual_capacity_frontier_frame,
            select_dual_reuse_residual_capacity_frontier_frame,
        )
        selectors = {"old": select_branch_reserved_residual_capacity_frontier_frame,
                     "new": select_dual_reuse_residual_capacity_frontier_frame}
    source = protocol["borrowed_sources"]["axis_capacity_development"]
    saved_cell = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [],
        "new_reference_calls": 0, "new_reference_oracle_calls": 0, "new_old_routed_calls": 0}
    budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
    for index, registered in enumerate(cell["states"]):
        saved = saved_cell["states"][registered["borrowed_index"]]
        if (registered["borrowed_from"] != "axis_capacity_development"
                or (saved["state_id"], saved["queue"], saved["source"], saved["caps"]) != (
                    registered["source_state_id"], registered["queue"], registered["source_record_source"], registered["caps"])):
            raise ValueError("borrowed state, source or native budget ladder changed")
        queue = tuple(registered["queue"])
        old = selectors["old"](model.options, queue, cell["lanes"], *budgets)
        row["new_old_routed_calls"] += 1
        new = selectors["new"](model.options, queue, cell["lanes"], *budgets)
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "traces": saved["traces"], "prior_routed": {"old": saved["old_routed"], "new": saved["routed"]},
            "old_routed": asdict(old), "routed": asdict(new)})
        if progress and (index + 1 == len(cell["states"]) or registered["cohort"] != cell["states"][index + 1]["cohort"]):
            print(f"{model.node} R{cell['lanes']}: {registered['cohort']} complete ({index + 1}/{len(cell['states'])})", flush=True)
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
        print(json.dumps({"protocol": str(protocol_path), "states": protocol["expected_state_count"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("residual-capacity development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.residual_capacity_frontier_development_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
