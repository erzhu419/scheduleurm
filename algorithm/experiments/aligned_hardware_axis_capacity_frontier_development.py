"""Develop shared integer axis-capacity bounds on the complete frozen roster."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_axis_capacity_frontier_development_20260928.json"
DEVELOPMENT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_score_proof_development_20260928.json"
TRIGGER_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_trigger_discovery_20260928.json"
SOURCES = (
    "algorithm/axis_capacity_frontier_score_proof_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_axis_capacity_frontier_development.py",
    "algorithm/experiments/axis_capacity_frontier_development_summary.py",
)


def build_protocol(models):
    development, trigger = (json.loads(path.read_text()) for path in (DEVELOPMENT_OUTPUT, TRIGGER_OUTPUT))
    previous = json.loads(DEVELOPMENT_OUTPUT.with_name(DEVELOPMENT_OUTPUT.stem + "_protocol.json").read_text())
    discovery = json.loads(TRIGGER_OUTPUT.with_name(TRIGGER_OUTPUT.stem + "_protocol.json").read_text())
    validation = json.loads(TRIGGER_OUTPUT.with_name(TRIGGER_OUTPUT.stem + "_validation_protocol.json").read_text())
    if development["decision"] != "FRONTIER_SCORE_PROOF_DEVELOPMENT_ADVANCE" or not all(development[key] for key in (
            "execution_checks_passed", "effective_references_complete", "development_gate_passed")):
        raise ValueError("the completed frontier-proof development advance is required")
    if (trigger["decision"] != "FRONTIER_TRIGGER_VALIDATION_HOLD" or trigger["routed_quality_gate_passed"]
            or not trigger["execution_checks_passed"] or not trigger["effective_references_complete"]):
        raise ValueError("the complete but quality-unclosed new trigger HOLD must remain recorded")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for old in (previous, validation):
        if snapshots != old["models"] or tuple(old[key] for key in (
                "initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("frozen cached catalogues or routing budgets changed")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = deepcopy(previous["cells"])
    by_cell = {(cell["node"], cell["lanes"]): cell for cell in cells}
    for cell in cells:
        for index, state in enumerate(cell["states"]):
            state.update(borrowed_from="frontier_development", borrowed_index=index,
                         source_state_id=state["state_id"], source_record_source=state["source"])
    new_failure_keys = []
    for old_cell in validation["cells"]:
        cell = by_cell[old_cell["node"], old_cell["lanes"]]
        for index, saved in enumerate(old_cell["states"]):
            state = deepcopy(saved)
            state.update(state_id="axis_failure:" + saved["state_id"],
                source="axis_failure_development", cohort="axis_failure_development", panel="axis_failure_development",
                historical_regression=False, borrowed_from="trigger_validation", borrowed_index=index,
                source_state_id=saved["state_id"], source_record_source=saved["source"])
            cell["states"].append(state)
            new_failure_keys.append({"node": cell["node"], "lanes": cell["lanes"], "queue": state["queue"]})
    for cell in cells:
        model = next(model for model in models if model.node == cell["node"])
        if any(state["caps"] != _caps(model) for state in cell["states"]):
            raise ValueError("borrowed native budget ladder changed")
    count = sum(len(cell["states"]) for cell in cells)
    return json.loads(json.dumps({"schema_version": 1, "experiment": "axis_capacity_frontier_development_v1",
        "models": snapshots, "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": previous["reference_budget"], "oracle_state_budget": previous["oracle_state_budget"],
        "borrowed_sources": {
            "frontier_development": {"path": str(DEVELOPMENT_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
            "trigger_validation": {"path": str(TRIGGER_OUTPUT.relative_to(ROOT)), "registered_at_utc": validation["registered_at_utc"]}},
        "historical_regression_keys": previous["historical_regression_keys"],
        "failure_development_keys": previous["failure_development_keys"],
        "axis_failure_development_keys": new_failure_keys,
        "expected_state_count": count,
        "expected_nonzero_state_count": sum(any(state["queue"]) for cell in cells for state in cell["states"]),
        "expected_failure_development_count": previous["expected_failure_development_count"],
        "expected_axis_failure_development_count": len(new_failure_keys),
        "expected_selector_calls": count * 2, "expected_native_advance_calls": 0,
        "expected_interface_calls": count * 2, "expected_native_constructions": count * 2,
        "expected_borrowed_native_snapshots": count * 8,
        "prior_decisions": {**trigger["prior_decisions"], "frontier_trigger_discovery": trigger["decision"]},
        "prior_discovery_registered_at_utc": discovery["registered_at_utc"], "execution_sources": sources,
        "roster": "Retain all742 frozen frontier-development states and allfour paired states for the newQ120,123,258 trigger, total746. Preserve34 historical regressions,16 zeros,four older failure controls, andfour newly failure-informed controls; no resampling or independent confirmation claim.",
        "rule": "Keep all native and ordinary routed paths. For at least two exported strict boxes with identical threshold/eligible/requirements, build one full-queue integer axis-capacity upper bound only when each positive requirement is on one coordinate. Ignore regional bounds and duration attainment to obtain a relaxation. For each axis and exact lane count use service bitsets, respecting queue capacity; merge axis maxima under the shared lane count. Only a maximum numerator below the strict integer target eliminates boxes. A DP table never supplies a strict witness. Single-box and mixed-coordinate groups use the original box search. Retain every uneliminated box in original order for the original strict proof under the remaining budget, then canonical completion only after score closure.",
        "reference_policy": "Borrow each matched six-field independent reference, eight native snapshots and both previous cold routers from its explicitly registered source. Run only two new independent cold axis-capacity routers per state. No new reference, oracle, native ladder or previous router is executed.",
        "accounting": "1492 actual cold selectors/native constructions; zero new reference/oracle/native-ladder calls;5968 borrowed native snapshots. Every axis seed and computed option/lane bitset output cell, every merge seed and computed merged lane output cell count as a visited DP state. Add actual fallback DFS states to score_proof.states_visited and allocate canonical completion only8192-totalproof remaining states. Count bitset shifts/ORs/masks, service reads, merge candidate terms, requirement reads and cached box-bound checks explicitly. Charge each cumulative final LP snapshot once; old LP/preprocessing/frontier export and dual accounting remain intact.",
        "gate": "Complete frozen roster and effective independent references; both final actions score/canonical exact and reference matching, including all old and new failure controls. Paid LP snapshots/resumption/frontier export reproduce the prior same-arm cold router and borrowed native endpoints, with no new LP or preprocessing. Ordinary paths reproduce the previous complete output. Audit DP plus fallback plus canonical shared8192 cap, all capacity coverage and actual arithmetic. Each state/arm separately has equal LP/matrix work and nonincreasing total integer states relative to its prior frontier router. Retain every failed/exhausted attempt and incomplete reference; no post-outcome rule, budget, roster or gate changes.",
        "limitations": "Failure-informed development on reused queues and four cached catalogues with virtual capacities. Borrowed native/ref evidence supplies no newly executed gain. State counts plus explicit DP arithmetic do not establish wall-time, flow-time or physical-scaling gains. All earlier HOLDs and the4150-state ADVANCE remain unchanged; the paper is not edited."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, roster, references, catalogue or capacity-bound rule changed")


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    if selectors is None:
        from algorithm.axis_capacity_frontier_score_proof_frame_candidates import (
            select_branch_reserved_axis_capacity_frontier_frame, select_dual_reuse_axis_capacity_frontier_frame)
        selectors = {"old": select_branch_reserved_axis_capacity_frontier_frame,
                     "new": select_dual_reuse_axis_capacity_frontier_frame}
    saved_cells = {name: load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
        for name, source in protocol["borrowed_sources"].items()
        if any(state["borrowed_from"] == name for state in cell["states"])}
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [],
        "new_reference_calls": 0, "new_reference_oracle_calls": 0, "new_old_routed_calls": 0}
    budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
    for index, registered in enumerate(cell["states"]):
        saved = saved_cells[registered["borrowed_from"]]["states"][registered["borrowed_index"]]
        if (saved["state_id"], saved["queue"], saved["source"], saved["caps"]) != (
                registered["source_state_id"], registered["queue"], registered["source_record_source"], registered["caps"]):
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
        raise FileExistsError("axis-capacity development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.axis_capacity_frontier_development_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
