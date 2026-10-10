"""Replace surviving-child rational RHS rebuilding with exact affine updates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path

from algorithm.experiments.aligned_hardware_dual_reuse_development import REFERENCE_FIELDS
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_root_pruning_development import load_cell

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_prune_first_dual_timing_20260928.json"
FOCUS_QUEUES = (
    ("gpu_a", 64, (88, 8, 23)),
    ("gpu_b", 64, (122, 3, 4)),
    ("gpu_b", 64, (85, 34, 58)),
    ("gpu_b", 64, (101, 9, 59)),
    ("gpu_a", 64, (118, 119, 269)),
    ("gpu_b", 32, (49, 4, 26)),
)
SOURCES = (
    "algorithm/affine_dual_reuse_integer_capacity_frame_candidates.py",
    "algorithm/affine_dual_residual_capacity_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_affine_dual_development.py",
    "algorithm/experiments/affine_dual_development_summary.py",
)


def build_protocol(models):
    previous = json.loads(PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    result = json.loads(PREVIOUS_OUTPUT.read_text())
    if (result["decision"] != "PRUNE_FIRST_DUAL_LOCAL_TIMING_HOLD" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "repeat_paths_stable",
            "frozen_residual_results_reproduced"))):
        raise ValueError("retain the quality-valid prune-first timing HOLD")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    if previous["models"] != snapshots:
        raise ValueError("cached catalogue models changed")
    if tuple(previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
        raise ValueError("retain the original selector and native budgets")
    sources = {}
    for path, literal in previous["execution_sources"].items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
        sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells, failures = [], []
    for old in previous["cells"]:
        states = []
        for index, state in enumerate(old["states"]):
            states.append({**deepcopy(state), "affine_source_index": index,
                           "affine_source_state_id": state["state_id"]})
            if (old["node"], old["lanes"], tuple(state["queue"])) in FOCUS_QUEUES:
                failures.append({"node": old["node"], "lanes": old["lanes"], "queue": state["queue"],
                                 "state_id": state["state_id"]})
        cells.append({"node": old["node"], "lanes": old["lanes"], "states": states})
    counts = {group: sum(state["timing_group"] == group for cell in cells for state in cell["states"])
              for group in previous["expected_group_counts"]}
    count = sum(map(lambda cell: len(cell["states"]), cells))
    if count != 754 or counts != previous["expected_group_counts"] or len(failures) != 6:
        raise ValueError("retain all 754 states and the six frozen timing diagnostics")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "affine_dual_development_v1", "models": snapshots,
        "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget", "reference_budget", "oracle_state_budget",
            "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys", "expected_group_counts")},
        "expected_state_count": count, "expected_selector_calls": count,
        "expected_native_advance_calls": count * 4, "expected_native_constructions": count * 2,
        "expected_interface_calls": count * 5, "previous_timing_failure_keys": previous["previous_timing_failure_keys"], "affine_focus_keys": failures,
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
        "execution_sources": sources,
        "prior_decisions": {**result["prior_decisions"], "prune_first_dual_timing": result["decision"]},
        "rule": "Retain prune-first insertion and exactly the same feasible parent dual. Save its own exact numerator at the solved box, separately from the minimum inherited region bound. Prepare integer affine coefficients lazily at first actual reuse and share them between descendants. Scan lower/upper coordinates and update the numerator only at changes. Count certificate numerator products, lazy preparation, coordinate scans, coefficient reads, integer updates and rational conversions. Keep LP, branching, pruning and proof budgets unchanged.",
        "roster": "All754 states from the quality-valid preceding timing HOLD, including688 ordinary development,34 historical,16 zero,8 old trigger controls and8 new paired trigger states. The original four failures and two preceding prune-first timing failures are six fixed development diagnostics, not fresh confirmation.",
        "accounting": "One new cold native ladder with four advances and one independent cold whole selector perstate:3016 native advances,754 selector calls,3770 interfaces,1508 fresh search constructions. Borrow all reference/oracle, branch and preceding prune-first dual records; no new screening, reference, oracle or control execution. Count every certificate preparation, scan and update; old three RHS-reading counters report actual legacy reads (zero for affine evaluation). Keep borrowed old costs separate rather than comparing unlike operations as a speed metric.",
        "gate": "Complete roster/runtime, valid independent references and certificates, canonical/ref-matching final selections. Removing only the three legacy RHS/requirement/weight-reading counters and the new affine_dual work must leave every native snapshot and whole routed output identical to the preceding prune-first selector, including canonical and all other dual counters. Validate cumulative real affine fees and exercise actual affine updates at each of the six fixed timing diagnostics. Do not turn changed operation units or zero legacy reads into a speed claim. Finish all states and retain honest HOLD; no budget, queue, threshold or historical source changes.",
        "limitations": "Development on previously used queues and four cached catalogues at virtual capacities. Reused control/reference costs are separate. This establishes neither independent queue confirmation nor wall-time, physical scaling or flow-time benefit. Earlier timing HOLD and the paper remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered source, roster, budget or affine-dual design changed")


def run_cell(model, cell, protocol, runtime, search_type=None, selector=None, progress=True):
    if search_type is None:
        from algorithm.affine_dual_reuse_integer_capacity_frame_candidates import AffineDualReuseIntegerCapacitySearch
        search_type = AffineDualReuseIntegerCapacitySearch
    if selector is None:
        from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
        selector = select_affine_dual_residual_capacity_frontier_frame
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "native_advance_calls": 0, "native_search_constructions": 0}
    for index, registered in enumerate(cell["states"]):
        saved = borrowed["states"][registered["affine_source_index"]]
        if saved["state_id"] != registered["affine_source_state_id"] or any(
                saved.get(key) != value for key, value in registered.items() if not key.startswith("affine_source_")):
            raise ValueError("borrowed development state differs from its frozen timing source")
        queue = tuple(registered["queue"])
        search = search_type(model.options, queue, cell["lanes"])
        row["native_search_constructions"] += 1
        traces = {}
        for cap in registered["caps"]:
            traces[str(cap)] = asdict(search.advance_to(cap))
            row["native_advance_calls"] += 1
        result = selector(model.options, queue, cell["lanes"], protocol["initial_lp_budget"],
                          protocol["lp_node_budget"], protocol["integer_state_budget"])
        row["native_search_constructions"] += 1
        row["candidate_calls"] += 1
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "traces": {"old": saved["traces"]["old"], "new": traces},
            "previous_dual_traces": saved["traces"]["new"], "prior_routed": saved["prior_routed"],
            "old_routed": saved["first_results"]["branch"],
            "previous_dual_routed": saved["first_results"]["candidate"], "routed": asdict(result)})
        if progress and (index + 1 == len(cell["states"]) or registered["timing_group"] != cell["states"][index + 1]["timing_group"]):
            print(f"{model.node} R{cell['lanes']}: {registered['timing_group']} ({index + 1}/{len(cell['states'])}) complete", flush=True)
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
        raise FileExistsError("affine-dual development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.affine_dual_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
