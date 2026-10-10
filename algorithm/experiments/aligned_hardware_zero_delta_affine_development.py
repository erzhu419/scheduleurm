"""Develop an exact zero-delta shortcut on the complete prior timing HOLD."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_fresh_timing_20260928.json"
SOURCES = (
    "algorithm/zero_delta_affine_dual_reuse_integer_capacity_frame_candidates.py",
    "algorithm/zero_delta_affine_dual_residual_capacity_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_zero_delta_affine_development.py",
    "algorithm/experiments/zero_delta_affine_development_summary.py",
)


def build_protocol(models):
    previous = json.loads(PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    result = json.loads(PREVIOUS_OUTPUT.read_text())
    if result["decision"] != "AFFINE_DUAL_FRESH_TIMING_HOLD" or not all(result[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced")):
        raise ValueError("retain the quality-valid complete preceding affine timing HOLD")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    if previous["models"] != snapshots:
        raise ValueError("retain the same four cached catalogue models")
    if tuple(previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget")) != (144, 256, 8192, 4096, 5_000_000):
        raise ValueError("retain the original method and reference budgets")
    sources = dict(previous["execution_sources"])
    for path, literal in sources.items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = []
    for old in previous["cells"]:
        cells.append({"node": old["node"], "lanes": old["lanes"], "states": [
            {**deepcopy(state), "zero_delta_source_index": index,
             "zero_delta_source_state_id": state["state_id"]}
            for index, state in enumerate(old["states"])]})
    counts = {group: sum(state["timing_group"] == group for cell in cells for state in cell["states"])
              for group in previous["expected_group_counts"]}
    count = sum(len(cell["states"]) for cell in cells)
    if count != 1778 or counts != previous["expected_group_counts"] or len(cells) != 16:
        raise ValueError("retain all 1778 states, six cohorts and sixteen cells")
    focuses = [{key: item[key] for key in ("node", "lanes", "state_id", "queue", "timing_group")}
               for item in result["trigger_states"] if any(
                   comparison["wall_ratio"] > previous["nonregression_limit"]
                   for comparison in item["comparisons"].values())]
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "zero_delta_affine_development_v1", "models": snapshots,
        "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget", "long_lp_budget", "control_integer_state_budget",
            "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys",
            "expected_group_counts", "nonregression_limit", "trigger_speedup_limit")},
        "expected_state_count": count, "expected_selector_calls": count,
        "expected_native_advance_calls": count * 4, "expected_native_constructions": count * 2,
        "expected_interface_calls": count * 5, "focus_keys": focuses,
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
                             "registered_at_utc": previous["registered_at_utc"]},
        "execution_sources": sources,
        "prior_decisions": {**result["prior_decisions"], "affine_dual_fresh_timing": result["decision"]},
        "rule": "Save exact lower-zero flags in the repaired dual's existing final column checks. After inherited pruning and capacity tightening, scan every changed lower/upper against the real certificate anchor or last evaluated box. If every changed lower coefficient and upper price is exactly zero, keep the inherited bound and skip affine preparation, update and rational conversion. Otherwise run the unchanged affine evaluation. Preserve the same LP, branching, proof, canonical and budget rules; no float-zero inference, queue-specific conditions or new thresholds.",
        "roster": "All1778 states from the preceding quality-valid timing HOLD:1024 previously confirmed ordinary node-states and all754 earlier controls. Every queue is now a development/regression queue, including the earlier trigger failure; no new independent confirmation claim.",
        "accounting": "1778 new cold whole selectors plus7112 native advances from1778 independent cold ladders:8890 interfaces and3556 search constructions. Borrow all references, oracle records, branch outputs and frozen affine outputs/native ladders. No reference, oracle, screening or control execution. Charge all ten zero-delta counters and actual seventeen affine counters separately; operation-unit changes do not establish speed.",
        "gate": "Finish the complete roster with reference/certificate/canonical quality and truthful cumulative fees. Except affine_dual, zero_delta and actual child_bound_evaluations, every native snapshot and whole routed output must equal the preceding affine output, including canonical and all search/proof work. For every compared stage guard_calls equals preceding child_bound_evaluations and equals actual evaluations plus skips. At least one actual skip across the full roster is required; focus-state hits are descriptive. Preserve earlier HOLDs and stop on an honest negative result, without tuning samples, conditions or budgets.",
        "limitations": "Development on previously used queues, four cached catalogues and virtual identical lanes. References and prior controls are borrowed. No independent confirmation, timing, physical scaling or flow-time claim; paper and all earlier results remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered source, roster, budgets or zero-delta rule changed")


def run_cell(model, cell, protocol, runtime, search_type=None, selector=None, progress=True):
    if search_type is None:
        from algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates import ZeroDeltaAffineDualReuseIntegerCapacitySearch
        search_type = ZeroDeltaAffineDualReuseIntegerCapacitySearch
    if selector is None:
        from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import select_zero_delta_affine_dual_residual_capacity_frontier_frame
        selector = select_zero_delta_affine_dual_residual_capacity_frontier_frame
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "native_advance_calls": 0, "native_search_constructions": 0}
    for index, registered in enumerate(cell["states"]):
        saved = borrowed["states"][registered["zero_delta_source_index"]]
        if saved["state_id"] != registered["zero_delta_source_state_id"] or any(
                saved.get(key) != value for key, value in registered.items() if not key.startswith("zero_delta_source_")):
            raise ValueError("borrowed state differs from the frozen timing source")
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
            "previous_dual_traces": saved["traces"]["new"],
            **({"prior_routed": saved["prior_routed"]} if "prior_routed" in saved else {}),
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
        raise FileExistsError("zero-delta development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.zero_delta_affine_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
