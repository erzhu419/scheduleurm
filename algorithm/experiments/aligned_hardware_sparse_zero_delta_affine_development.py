"""Develop sparse dual-delta support on every preceding frozen queue state."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_sparse_zero_delta_affine_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_fresh_timing_20260928.json"
SOURCES = (
    "algorithm/sparse_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates.py",
    "algorithm/sparse_zero_delta_affine_dual_residual_capacity_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_sparse_zero_delta_affine_development.py",
    "algorithm/experiments/sparse_zero_delta_affine_development_summary.py",
    "algorithm/experiments/aligned_hardware_sparse_zero_delta_affine_timing.py",
    "algorithm/experiments/sparse_zero_delta_affine_timing_summary.py",
)
TIMING_PLAN = dict(order_seed=2026092898, labels=["candidate", "zero", "affine", "branch", "long"],
    repeats=6, warmups_per_label=2, expected_timed_calls=84060, expected_warmup_calls=160,
    nonregression_limit=1.05, trigger_speedup_limit=.95)
FOCUS_QUEUES = (("gpu_a", 64, (88, 8, 23)), ("gpu_a", 64, (118, 119, 269)),
                ("cpu_a", 8, (27, 15, 34)), ("gpu_a", 64, (158, 196, 145)),
                ("gpu_a", 64, (124, 10, 80)))


def build_protocol(models):
    previous = json.loads(PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    result = json.loads(PREVIOUS_OUTPUT.read_text())
    if result["decision"] != "ZERO_DELTA_AFFINE_FRESH_TIMING_HOLD" or not all(result[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced")):
        raise ValueError("retain the complete quality-valid preceding independent timing HOLD")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    if previous["models"] != snapshots:
        raise ValueError("retain the same four cached physical catalogue models")
    if tuple(previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget")) != (144, 256, 8192, 4096, 5_000_000):
        raise ValueError("retain the original routing and reference budgets")
    sources = dict(previous["execution_sources"])
    for path, literal in sources.items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = [{"node": old["node"], "lanes": old["lanes"], "states": [
        {**deepcopy(state), "sparse_delta_source_index": index,
         "sparse_delta_source_state_id": state["state_id"]}
        for index, state in enumerate(old["states"])]} for old in previous["cells"]]
    count = sum(len(cell["states"]) for cell in cells)
    counts = {name: sum(state["timing_group"] == name for cell in cells for state in cell["states"])
              for name in previous["expected_group_counts"]}
    if count != 2802 or counts != previous["expected_group_counts"] or len(cells) != 16:
        raise ValueError("retain all2802 preceding states, seven cohorts and sixteen cells")
    focuses = [{"node": cell["node"], "lanes": cell["lanes"], **{key: state[key]
        for key in ("state_id", "queue", "timing_group")}} for cell in cells for state in cell["states"]
        if (cell["node"], cell["lanes"], tuple(state["queue"])) in FOCUS_QUEUES]
    if len(focuses) != len(FOCUS_QUEUES):
        raise ValueError("retain the three failures and two observed ordinary tails as descriptive controls")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "sparse_zero_delta_affine_development_v1", "models": snapshots,
        "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget", "long_lp_budget", "control_integer_state_budget",
            "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys",
            "expected_group_counts", "nonregression_limit", "trigger_speedup_limit")},
        "expected_state_count": count, "expected_selector_calls": count,
        "expected_native_advance_calls": count * 4, "expected_native_constructions": count * 2,
        "expected_interface_calls": count * 5, "focus_keys": focuses,
        "subsequent_timing_plan": TIMING_PLAN,
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
                             "registered_at_utc": previous["registered_at_utc"]},
        "execution_sources": sources,
        "prior_decisions": {**result["prior_decisions"], "zero_delta_affine_fresh_timing": result["decision"]},
        "rule": "Collect actual changed lower/upper coordinates during branch construction and capacity tightening; merge them in position order relative to the same certificate's last real evaluation anchor. Zero guards read only that support, and affine fallback consumes the same complete support. Preserve pending changes across zero skips and failed LPs; reset on an actual evaluation or a new successful certificate. Keep every LP, bound, pruning, heap, canonical and router decision unchanged; no queue-specific rules or new thresholds.",
        "roster": "Every2802 state from the preceding quality-valid independent timing HOLD, with original metadata, seven groups and real axis predecessors preserved. All queues are now development controls; no new ordinary draw or independent confirmation claim.",
        "accounting": "2802 cold whole selectors plus11208 native advances from2802 separate cold native ladders:14010 interfaces and5604 search constructions. Borrow all2802 references, conditional oracles and genuine zero/affine/branch controls. Zero reference, oracle or screening execution. Charge support collection/merge/storage, guard reads and actual affine fallback separately; operation-unit changes do not establish speed.",
        "gate": "Complete every state with reference/certificate/canonical quality and truthful cumulative fees. Except actual affine_dual, zero_delta and sparse_delta arithmetic fees, native snapshots and whole outputs must equal the real frozen zero-delta outputs, including all dual outcomes, LP/pruning/canonical and shared proof/completion work. Work gate checks truthful accounting, with no unregistered minimum reduction. Any nonclosure or path mismatch holds timing. Subsequent timing retains every preceding gate and requires candidate/zero nonregression in the same groups; all observations retained without trimming, threshold changes or reruns.",
        "limitations": "Development on previously used queues, four cached catalogues and virtual identical lanes. References and old controls are borrowed. No independent confirmation, physical scaling, flow-time or significance claim. Retain all preceding HOLDs; paper and old algorithms/protocols/results unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered source, roster, budgets or sparse-delta rule changed")


def run_cell(model, cell, protocol, runtime, search_type=None, selector=None, progress=True):
    if search_type is None:
        from algorithm.sparse_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates import SparseZeroDeltaAffineDualReuseIntegerCapacitySearch
        search_type = SparseZeroDeltaAffineDualReuseIntegerCapacitySearch
    if selector is None:
        from algorithm.sparse_zero_delta_affine_dual_residual_capacity_frame_candidates import select_sparse_zero_delta_affine_dual_residual_capacity_frontier_frame
        selector = select_sparse_zero_delta_affine_dual_residual_capacity_frontier_frame
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "native_advance_calls": 0, "native_search_constructions": 0}
    for index, registered in enumerate(cell["states"]):
        saved = borrowed["states"][registered["sparse_delta_source_index"]]
        if saved["state_id"] != registered["sparse_delta_source_state_id"] or any(
                saved.get(key) != value for key, value in registered.items() if not key.startswith("sparse_delta_source_")):
            raise ValueError("borrowed state differs from its frozen timing source")
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
            "previous_zero_traces": saved["traces"]["new"],
            "original_affine_traces": saved["previous_dual_traces"],
            **({"prior_routed": saved["prior_routed"]} if "prior_routed" in saved else {}),
            "old_routed": saved["first_results"]["branch"],
            "original_affine_routed": saved["first_results"]["affine"],
            "previous_zero_routed": saved["first_results"]["candidate"], "routed": asdict(result)})
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
        raise FileExistsError("sparse-delta development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.sparse_zero_delta_affine_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
