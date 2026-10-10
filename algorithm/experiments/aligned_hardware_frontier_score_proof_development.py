"""Develop integer score proofs restricted to the final paid LP frontier."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_score_proof_development_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_late_score_proof_development_20260928.json"
SOURCES = (
    "algorithm/frontier_score_proof_certificate_routed_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_frontier_score_proof_development.py",
    "algorithm/experiments/frontier_score_proof_development_summary.py",
)


def build_protocol(models):
    prior = json.loads(PRIOR_OUTPUT.read_text())
    previous = json.loads(PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json").read_text())
    if prior["decision"] != "LATE_SCORE_PROOF_DEVELOPMENT_HOLD" or not all(prior[key] for key in (
            "execution_checks_passed", "effective_references_complete", "final_score_gate_passed", "work_gate_passed")):
        raise ValueError("the complete quality-unclosed late-proof HOLD must be retained")
    if prior["routed_quality_gate_passed"]:
        raise ValueError("the previous late-proof nonclosure must remain recorded")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    if snapshots != previous["models"]:
        raise ValueError("cached physical catalogue models changed")
    if tuple(previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
        raise ValueError("developed routing budgets must remain fixed")
    sources = {}
    for path, literal in previous["execution_sources"].items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
        sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = deepcopy(previous["cells"])
    for cell in cells:
        model = next(model for model in models if model.node == cell["node"])
        for index, state in enumerate(cell["states"]):
            state.update(borrowed_from="late_development", borrowed_index=index,
                         source_state_id=state["state_id"], source_record_source=state["source"])
            if state["caps"] != _caps(model):
                raise ValueError("borrowed native budget ladder changed")
    count = sum(len(cell["states"]) for cell in cells)
    return json.loads(json.dumps({"schema_version": 1, "experiment": "frontier_score_proof_development_v1",
        "models": snapshots, "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": previous["reference_budget"], "oracle_state_budget": previous["oracle_state_budget"],
        "borrowed_sources": {"late_development": {"path": str(PRIOR_OUTPUT.relative_to(ROOT)),
            "registered_at_utc": previous["registered_at_utc"]}},
        "historical_regression_keys": previous["historical_regression_keys"],
        "failure_development_keys": previous["failure_development_keys"],
        "expected_state_count": count,
        "expected_nonzero_state_count": sum(any(state["queue"]) for cell in cells for state in cell["states"]),
        "expected_failure_development_count": previous["expected_failure_development_count"],
        "expected_selector_calls": count * 2, "expected_native_advance_calls": 0,
        "expected_interface_calls": count * 2, "expected_native_constructions": count * 2,
        "expected_borrowed_native_snapshots": count * 8,
        "prior_decisions": {**previous["prior_decisions"], "late_score_proof_development": prior["decision"]},
        "execution_sources": sources,
        "roster": "Retain exactly the preceding742 development states,726 nonzero,34 fixed historical keys,16 zeros andfour paired failure controls. Every queue, seed, cohort and catalogue is unchanged. The counterexample remains failure-informed development; no resampling or new confirmation claim.",
        "rule": "Both routers retain initial144/cumulative256 LP and all ordinary probe paths. Only after an unproved final score with no earlier integer probe, read the suspended paid generator's exact raw heap without resuming or solving again. Copy every box whose exact bound exceeds its own exact incumbent score, including unsolved/failed-LP children, in heap order. All duration roots already initialized and completed/pruned boxes cannot contain a strict improvement. Requirement grouping preserves any strict original-catalogue improvement by its group-minimum-duration representative. In each exported box reserve all mandatory lower counts/resources, enumerate optional counts up to upper-lower with a separate memo, and require the exact duration stratum. Prove absence above floor(incumbent_score*threshold)+1 using one global8192 state budget across boxes; a witness/exhaustion stays incomplete. Only successful score proof enables the frozen original-catalogue canonical completion with the remaining8192-proof_visits budget. Raw LP snapshots and ordinary outputs remain unchanged.",
        "reference_policy": "Borrow the matched six independent reference fields, both original native ladders, both original pre-late routed outputs and both actually executed preceding late routers. No new reference, oracle or native ladder is run. Execute two independent cold frontier-proof routers per state, including zeros and failure controls.",
        "accounting": "1484 actual cold selectors and1484 native constructions, zero new native-ladder interfaces,5936 borrowed native snapshots. Charge each final cumulative LP snapshot once and all strict proof plus canonical integer visits within8192. Charge exported heap entries and copied region/requirement/count/lift fields separately from LP and integer visits. Preserve existing exact-dual arithmetic accounting. Borrowed reference/native/prior-router work is separate; it supplies no newly executed gain.",
        "gate": "Complete coverage and effective independent references; both final routed actions score/canonical exact and reference matching, including allfour failure controls. All ordinary output fields reproduce the preceding late router; final/initial paid LP snapshots and resumption remain identical for every state. Exported strict boxes count equals final live frontier minus score-equal ties and uses its own incumbent/box/threshold/representatives. Verify shared proof/canonical cap, honest outcomes and no repeated probe or extra LP. Independently require LP/matrix equality and integer nonincrease relative to each arm's preceding late router, both per-state and pooled. Between actual branch-reserved+frontier and dual+frontier, require separate LP/matrix/integer nonincrease in every nonzero cell and every historical subset, plus strict pooled nonzero LP or integer improvement. No changing rule, roster, budgets or gate after outcome access.",
        "limitations": "Reused development states including the preceding confirmation's known counterexample, four cached catalogues and virtual resource counts. This validates development closure and counted work; it is not independent confirmation, wall time, flow time or physical scaling. All previous late-proof and confirmation HOLDs remain unchanged; the paper is not edited."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, roster, catalogue, references or frontier-proof rule changed")


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    if selectors is None:
        from algorithm.frontier_score_proof_certificate_routed_frame_candidates import (
            select_branch_reserved_frontier_score_proof_frame, select_dual_reuse_frontier_score_proof_frame)
        selectors = {"old": select_branch_reserved_frontier_score_proof_frame,
                     "new": select_dual_reuse_frontier_score_proof_frame}
    source = protocol["borrowed_sources"]["late_development"]
    saved_cell = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
           "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
           "runtime_start": runtime_provenance(), "states": [],
           "new_reference_calls": 0, "new_reference_oracle_calls": 0, "new_old_routed_calls": 0}
    budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
    for index, registered in enumerate(cell["states"]):
        saved = saved_cell["states"][registered["borrowed_index"]]
        if (registered["borrowed_from"] != "late_development" or
                (saved["state_id"], saved["queue"], saved["source"], saved["caps"]) != (
                    registered["source_state_id"], registered["queue"], registered["source_record_source"], registered["caps"])):
            raise ValueError("borrowed development state or budget ladder changed")
        queue = tuple(registered["queue"])
        old = selectors["old"](model.options, queue, cell["lanes"], *budgets)
        row["new_old_routed_calls"] += 1
        new = selectors["new"](model.options, queue, cell["lanes"], *budgets)
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "traces": saved["traces"], "original_routed": saved["original_routed"],
            "prior_routed": {"old": saved["old_routed"], "new": saved["routed"]},
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
        raise FileExistsError("frontier-score-proof development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.frontier_score_proof_development_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
