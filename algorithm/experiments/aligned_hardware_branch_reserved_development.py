"""Develop root pruning with the original per-budget branch LP allowance."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_development_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_root_pruning_development_20260928.json"
CONFIRMATION_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_root_pruning_confirmation_20260928.json"
SOURCES = (
    "algorithm/branch_reserved_integer_capacity_frame_candidates.py",
    "algorithm/branch_reserved_certificate_routed_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_branch_reserved_development.py",
    "algorithm/experiments/branch_reserved_development_summary.py",
)


def build_protocol(models):
    previous = json.loads(PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json").read_text())
    prior = json.loads(PRIOR_OUTPUT.read_text())
    held = json.loads(CONFIRMATION_OUTPUT.with_name(CONFIRMATION_OUTPUT.stem + "_protocol.json").read_text())
    held_result = json.loads(CONFIRMATION_OUTPUT.read_text())
    if prior["decision"] != "ROOT_PRUNING_DEVELOPMENT_ADVANCE" or not prior["development_gate_passed"]:
        raise ValueError("completed root-pruning development is required")
    if (held_result["decision"] != "ROOT_PRUNING_FRESH_CONFIRMATION_HOLD"
            or not all(held_result[key] for key in ("execution_checks_passed", "effective_references_complete",
                                                   "routed_quality_gate_passed", "finite_budget_gate_passed"))):
        raise ValueError("the quality-valid frozen root-pruning confirmation HOLD must be retained")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    for old in (previous, held):
        if snapshots != old["models"]:
            raise ValueError("cached physical catalogue models changed")
        if tuple(old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("developed routing budgets must be retained")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
    historical = {(cell["node"], cell["lanes"], tuple(state["queue"]))
        for cell in held["cells"] for state in cell["states"] if state["source"] == "historical_regression"}
    cells, found = [], set()
    for original in previous["cells"]:
        cell = deepcopy(original)
        for index, state in enumerate(cell["states"]):
            key = cell["node"], cell["lanes"], tuple(state["queue"])
            state["prior_index"] = index
            state["historical_regression"] = key in historical
            if state["historical_regression"]:
                found.add(key)
        cells.append(cell)
    if found != historical:
        raise ValueError("all frozen historical regression keys must be in the development roster")
    return {"schema_version": 1, "experiment": "branch_reserved_root_pruning_development_v1",
        "models": snapshots, "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "prior_source": {"path": str(PRIOR_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
        "prior_confirmation_path": str(CONFIRMATION_OUTPUT.relative_to(ROOT)),
        "historical_regression_keys": [{"node": node, "lanes": lanes, "queue": list(queue)}
            for node, lanes, queue in sorted(historical)],
        "prior_decisions": {"root_pruning_development": prior["decision"],
            "root_pruning_confirmation": held_result["decision"],
            "certificate_routed_confirmation": previous["prior_decisions"]["confirmation"]},
        "execution_sources": {**previous["execution_sources"], **held["execution_sources"],
            **{path: (ROOT / path).read_text() for path in SOURCES}},
        "roster": "Exactly the prior738-state development roster, with every original queue and seed retained;722 nonzero states plus16 zero controls. Separately retain the34 historical trigger/tail keys that accompanied the preceding confirmation. No preceding fresh-confirmation queue is added to this development run.",
        "budget_rule": "Reserve one position for every duration root, even when its LP is skipped. Continue LP search only while actual_lp_calls+(roots_seen-roots_solved)<budget. The branch LP allowance is budget-root_count. Record only executed LP calls, including failures. Keep all pruned-root metadata, frontier and preprocessing on resumption; no restart and no new tuning parameter.",
        "calls": "Borrow every independent fullK4096 reference and any conditional oracle from the frozen development cells, including zero queues. Reconstruct audit metrics for these borrowed records without a new reference solve. Every state executes original and branch-reserved native searches through root_count,root_count+1,144,256, then one cold original and one cold branch-reserved routed144/256/8192 selector. Ten interface calls and four native search constructions per state; charge each cumulative256 trace endpoint once and routed final_lp once. Borrowed reference costs are reported separately from this run's execution.",
        "gate": "Complete frozen coverage, valid certificates and cumulative work, complete effective independent references, and new routed canonical action matching every reference. No score loss relative to the original kernel at any of four budgets. On all nonzero states: routed LP/matrix/integer totals nonincreasing in every cell with strict pooled LP reduction. Independently apply per-cell nonincrease to the34 fixed historical regressions so ordinary development savings cannot hide a recurrence of matrix growth. Zero controls cannot supply strict work gain. At every snapshot actualLP+reserved<=cap; root-only cap must not execute branch LPs. Finish all cells and retain every loss or nonclosure without changing the rule, budgets, seeds or gate.",
        "limitations": "Reused development and historical regression queues, fixed cached catalogues, and virtual resource counts. Reserving branch allowance may remove earlier incumbent improvements or change certificate routing; no finite-cap equality is presumed. This is development evidence with no fresh confirmation, wall-time, physical scaling or flow-time claim. Both preceding root-pruning confirmation HOLD and certificate-routed timing HOLD remain unchanged."}


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, catalogue, development roster or budget rule changed")


def run_cell(model, cell, protocol, runtime, progress=True):
    from algorithm.resumable_integer_capacity_frame_candidates import IntegerCapacitySearch
    from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
    from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
    from algorithm.branch_reserved_certificate_routed_frame_candidates import select_branch_reserved_certificate_routed_frame
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key
    source = protocol["prior_source"]
    prior = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    minima = {}
    for option in model.options:
        required = tuple(option.requirements)
        minima[required] = min(minima.get(required, float("inf")), max(1.0, option.duration))
    roots = len(set(minima.values()))
    caps = (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0,
        "new_old_routed_calls": 0}
    for index, registered in enumerate(cell["states"]):
        saved = prior["states"][registered["prior_index"]]
        if (saved["state_id"], saved["queue"], saved["source"]) != (
                registered["state_id"], registered["queue"], registered["source"]):
            raise ValueError("borrowed development reference changed")
        queue, lanes = tuple(registered["queue"]), cell["lanes"]
        reference = selection_from_record(saved["reference"])
        expected = runtime.reference_key(reference, queue, len(model.options))
        complete = expected is not None
        reference_source = "full_K4096" if complete else None
        metrics = runtime._metrics()
        runtime._record(metrics, reference, model, queue, lanes, protocol["reference_budget"], expected)
        if expected is None and saved["oracle"] is not None:
            expected = oracle_key(saved["oracle"], len(model.options))
            reference_source = "independent_score_face_oracle" if expected is not None else None
        if saved["effective_reference_complete"] != (expected is not None):
            raise ValueError("borrowed effective reference differs from its full reference and oracle")
        traces = {}
        for label, cls in (("old", IntegerCapacitySearch), ("new", BranchReservedIntegerCapacitySearch)):
            search = cls(model.options, queue, lanes)
            traces[label] = {str(cap): asdict(search.advance_to(cap)) for cap in caps}
        budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
        old = select_certificate_routed_frame(model.options, queue, lanes, *budgets)
        row["new_old_routed_calls"] += 1
        result = select_branch_reserved_certificate_routed_frame(model.options, queue, lanes, *budgets)
        row["states"].append({**registered, "reference": saved["reference"], "reference_metrics": metrics,
            "reference_complete": complete, "oracle": saved["oracle"],
            "effective_reference_complete": expected is not None, "effective_reference_source": reference_source,
            "caps": caps, "traces": traces, "old_routed": asdict(old), "routed": asdict(result)})
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
        raise FileExistsError("branch-reserved development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
        print(f"{model.node} R{cell['lanes']}: {len(row['states'])} states saved", flush=True)
    from algorithm.experiments.branch_reserved_development_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
