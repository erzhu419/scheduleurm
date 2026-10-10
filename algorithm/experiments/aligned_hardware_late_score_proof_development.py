"""Develop a final integer score proof for both frozen certificate routers."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_late_score_proof_development_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_dual_reuse_development_20260928.json"
HELD_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_dual_reuse_confirmation_20260928.json"
FAILURE_QUEUE = [54, 58, 290]
FAILURE_LANES = 64
SOURCES = (
    "algorithm/late_score_proof_certificate_routed_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_late_score_proof_development.py",
    "algorithm/experiments/late_score_proof_development_summary.py",
)


def _protocol(path):
    return json.loads(path.with_name(path.stem + "_protocol.json").read_text())


def build_protocol(models):
    prior, held = (json.loads(path.read_text()) for path in (PRIOR_OUTPUT, HELD_OUTPUT))
    previous, confirmation = _protocol(PRIOR_OUTPUT), _protocol(HELD_OUTPUT)
    if prior["decision"] != "DUAL_REUSE_DEVELOPMENT_ADVANCE" or not prior["development_gate_passed"]:
        raise ValueError("the complete dual-reuse development must remain unchanged")
    if held["decision"] != "DUAL_REUSE_FRESH_CONFIRMATION_HOLD" or not all(held[key] for key in (
            "execution_checks_passed", "effective_references_complete", "final_score_gate_passed")):
        raise ValueError("the quality-unclosed fresh confirmation HOLD must be retained")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for old in (previous, confirmation):
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
    failure_cells = {(cell["node"], cell["lanes"]): cell for cell in confirmation["cells"]}
    failure_keys = []
    for cell in cells:
        model = next(model for model in models if model.node == cell["node"])
        for index, state in enumerate(cell["states"]):
            state.update(borrowed_from="development", borrowed_index=index,
                         source_state_id=state["state_id"], source_record_source=state["source"])
            if state["caps"] != _caps(model):
                raise ValueError("borrowed native budget ladder changed")
        if cell["lanes"] != FAILURE_LANES:
            continue
        matched = [(index, state) for index, state in enumerate(failure_cells[cell["node"], FAILURE_LANES]["states"])
                   if state["queue"] == FAILURE_QUEUE]
        if len(matched) != 1:
            raise ValueError("the counterexample must be paired once across every catalogue")
        index, saved = matched[0]
        if any(state["queue"] == FAILURE_QUEUE for state in cell["states"]):
            raise ValueError("the added counterexample cannot duplicate a development queue")
        state = deepcopy(saved)
        state.update(state_id="failure_development:" + saved["state_id"],
                     source="failure_development", cohort="failure_development",
                     historical_regression=False, borrowed_from="confirmation", borrowed_index=index,
                     source_state_id=saved["state_id"], source_record_source=saved["source"])
        if state["caps"] != _caps(model):
            raise ValueError("borrowed counterexample budget ladder changed")
        cell["states"].append(state)
        failure_keys.append({"node": cell["node"], "lanes": FAILURE_LANES, "queue": FAILURE_QUEUE})
    count = sum(len(cell["states"]) for cell in cells)
    nonzero = sum(any(state["queue"]) for cell in cells for state in cell["states"])
    return json.loads(json.dumps({"schema_version": 1, "experiment": "late_score_proof_development_v1",
        "models": snapshots, "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": previous["reference_budget"], "oracle_state_budget": previous["oracle_state_budget"],
        "borrowed_sources": {label: {"path": str(path.relative_to(ROOT)), "registered_at_utc": old["registered_at_utc"]}
            for label, path, old in (("development", PRIOR_OUTPUT, previous), ("confirmation", HELD_OUTPUT, confirmation))},
        "historical_regression_keys": previous["historical_regression_keys"],
        "failure_development_keys": failure_keys,
        "expected_state_count": count, "expected_nonzero_state_count": nonzero,
        "expected_failure_development_count": len(failure_keys),
        "expected_selector_calls": count * 2, "expected_native_advance_calls": 0,
        "expected_interface_calls": count * 2, "expected_native_constructions": count * 2,
        "expected_borrowed_native_snapshots": count * 8,
        "prior_decisions": {**previous["prior_decisions"], "dual_reuse_development": prior["decision"],
                            "dual_reuse_confirmation": held["decision"]},
        "execution_sources": sources,
        "roster": "Retain all738 preceding development states,34 historical keys and16 zero controls. Append the confirmed R64 Q(54,58,290) counterexample paired across allfour catalogues:742 states,726 nonzero. The four appended states are failure-informed development controls, not a new confirmation. No resampling, queue filtering or state replacement.",
        "rule": "Both branch-reserved and dual-reuse routers keep the original initial144/cumulative256 LP route. Initial unproved scores still advance straight to256; ordinary initial/resumed canonical probes stay unchanged. Only if final LP is score-unproved and noncanonical and no probe has been attempted, invoke the existing strict-score-proof/canonical-completion helper with the unused8192 total integer budget. The proof spends from that budget first; canonical completion receives only its remainder. At most one integer probe, no extra LP calls, and no changes to the raw paid LP snapshots. A witness, exhaustion or nonclosure remains an incomplete result.",
        "reference_policy": "Borrow the matched six independent fullK4096/reference-only oracle fields, both original routed outputs and both native four-cap traces from their actual preceding development/confirmation records. No new reference, oracle or native ladder is executed. Execute exactly two independent cold late-proof routers per state, including zero and paired failure controls.",
        "accounting": "Charge each actual routed final_lp once, including failed LP calls and root-slot reservation. Sum strict proof and canonical-completion integer visits within8192; do not add initial and final cumulative snapshots. Report proof/canonical work and successful late closure separately. Charge dual-reuse arithmetic explicitly.1484 cold selector interfaces and1484 native constructions; zero new native-ladder interfaces and5936 borrowed snapshots. Borrowed native/reference work is separate and supplies no new gain.",
        "gate": "Complete frozen coverage and effective independent references; both actual arms must be score/canonical exact and match their reference. Ordinary results must equal their original routed output; late results must retain its raw initial/final snapshots, use the final unproved score and shared integer cap without repeat or extra LP. In every nonzero cell and fixed historical subset, routed LP/matrix/integer totals separately must not increase for dual+late relative to branch-reserved+late. Strict pooled nonzero LP or integer reduction is required. All four failure controls must close and match; preserve every failure and exhaustion with no budget/rule/roster tuning.",
        "limitations": "Reused development states including a failure selected from the preceding confirmation, four cached catalogues and virtual resource counts. This tests development closure and counted work, not independent confirmation, physical scaling, flow time or wall time. The preceding DUAL_REUSE_FRESH_CONFIRMATION_HOLD remains unchanged; the paper is not edited."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, roster, catalogue, references or late-proof rule changed")


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    if selectors is None:
        from algorithm.late_score_proof_certificate_routed_frame_candidates import (
            select_branch_reserved_late_score_proof_frame, select_dual_reuse_late_score_proof_frame)
        selectors = {"old": select_branch_reserved_late_score_proof_frame,
                     "new": select_dual_reuse_late_score_proof_frame}
    borrowed = {}
    for label in {state["borrowed_from"] for state in cell["states"]}:
        source = protocol["borrowed_sources"][label]
        borrowed[label] = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
           "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
           "runtime_start": runtime_provenance(), "runtime_end": None, "states": [],
           "new_reference_calls": 0, "new_reference_oracle_calls": 0, "new_old_routed_calls": 0}
    budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
    for index, registered in enumerate(cell["states"]):
        saved = borrowed[registered["borrowed_from"]]["states"][registered["borrowed_index"]]
        if (saved["state_id"], saved["queue"], saved["source"], saved["caps"]) != (
                registered["source_state_id"], registered["queue"], registered["source_record_source"], registered["caps"]):
            raise ValueError("borrowed development state or budget ladder changed")
        traces = ({"old": saved["old_traces"], "new": saved["traces"]}
                  if registered["borrowed_from"] == "development" else saved["traces"])
        queue = tuple(registered["queue"])
        old = selectors["old"](model.options, queue, cell["lanes"], *budgets)
        row["new_old_routed_calls"] += 1
        new = selectors["new"](model.options, queue, cell["lanes"], *budgets)
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "traces": traces, "original_routed": {"old": saved["old_routed"], "new": saved["routed"]},
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
        raise FileExistsError("late-score-proof development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.late_score_proof_development_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
