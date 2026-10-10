"""Develop fused scheduler root checkpoints on the complete frozen roster."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_fused_root_dual_face_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_settled_root_dual_face_development_20260928_checked.json"
PREVIOUS_TIMING = ROOT / "md/experiment_artifacts/aligned_hardware_settled_root_dual_face_timing_20260928_checked.json"
SOURCES = (
    "algorithm/fused_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates.py",
    "algorithm/fused_root_dual_face_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_fused_root_dual_face_development.py",
    "algorithm/experiments/fused_root_dual_face_development_summary.py",
    "algorithm/experiments/aligned_hardware_fused_root_dual_face_timing.py",
    "algorithm/experiments/fused_root_dual_face_timing_summary.py",
)
TIMING_PLAN = dict(order_seed=2026092903,
    labels=["candidate", "settled", "native", "face", "early", "zero", "affine", "branch", "long"],
    repeats=6, warmups_per_label=2, expected_timed_calls=151308, expected_warmup_calls=288,
    nonregression_limit=1.05, trigger_speedup_limit=.95)


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    previous, timing = (json.loads(_protocol_path(path).read_text())
                        for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    result, clocks = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    if result["decision"] != "SETTLED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed",
            "work_gate_passed", "development_gate_passed", "source_native_behavior_preserved")):
        raise ValueError("retain the complete quality-valid checked settled-root development")
    if clocks["decision"] != "SETTLED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD" or not all(clocks[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced")):
        raise ValueError("retain the complete preceding settled-root timing HOLD")
    source = previous["summary_reanalysis"]
    raw_path = ROOT / source["data_path"]
    raw_protocol = json.loads(_protocol_path(raw_path).read_text())
    if (source["registered_at_utc"] != previous["registered_at_utc"]
            or raw_protocol["registered_at_utc"] != previous["registered_at_utc"]
            or raw_protocol["cells"] != previous["cells"]
            or source["actual_selector_calls"] != 0 or source["actual_native_advance_calls"] != 0):
        raise ValueError("checked quality must retain the actual original settled run and its registration")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for parent in (previous, timing):
        if parent["models"] != snapshots:
            raise ValueError("retain the same four cached physical catalogue models")
        if tuple(parent[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
                "reference_budget", "oracle_state_budget")) != (144, 256, 8192, 4096, 5_000_000):
            raise ValueError("retain the original routing and reference budgets")
        for path, literal in parent["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = [{"node": old["node"], "lanes": old["lanes"], "states": [
        {**deepcopy(state), "fused_root_source_index": index,
         "fused_root_source_state_id": state["state_id"]}
        for index, state in enumerate(old["states"])]} for old in previous["cells"]]
    count = sum(len(cell["states"]) for cell in cells)
    counts = {name: sum(state["timing_group"] == name for cell in cells for state in cell["states"])
              for name in previous["expected_group_counts"]}
    if count != 2802 or counts != previous["expected_group_counts"] or len(cells) != 16:
        raise ValueError("retain all2802 preceding states, seven cohorts and sixteen cells")
    focuses = deepcopy(previous["focus_keys"])
    focuses.extend({"node": cell["node"], "lanes": cell["lanes"], **{key: state[key]
        for key in ("state_id", "queue", "timing_group")}} for cell in cells for state in cell["states"]
        if cell["node"] == "node007" and cell["lanes"] == 8 and state["queue"] == [27, 15, 34])
    if len(focuses) != 11:
        raise ValueError("retain ten preceding focuses and the observed node007 historical tail")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "fused_root_dual_face_development_v1", "models": snapshots,
        "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget", "long_lp_budget", "control_integer_state_budget",
            "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys",
            "expected_group_counts", "nonregression_limit", "trigger_speedup_limit")},
        "expected_state_count": count, "expected_selector_calls": count,
        "expected_native_advance_calls": 0, "expected_native_constructions": count,
        "expected_interface_calls": count, "expected_fused_root_initialize_calls": count,
        "expected_root_settle_calls": result["executed_root_settle_calls"],
        "focus_keys": focuses, "subsequent_timing_plan": TIMING_PLAN,
        "reference_source": {"path": source["data_path"], "registered_at_utc": source["registered_at_utc"]},
        "execution_sources": sources,
        "prior_decisions": {**clocks["prior_decisions"], "settled_root_dual_face_timing": clocks["decision"],
                            "settled_root_dual_face_original_development": source["original_decision"]},
        "rule": "One cold native search initializes its actual M roots, fully materializes the immutable raw-root checkpoint, then conditionally performs exactly the prior zero-additional-LP settlement before its first generator yield. Use one raw-heap pass per snapshot to compute the same upper bound, retained count, ties and unsolved count without a retained-entry list. Reuse only immutable duration/root records; rebuild changing native work at each checkpoint. Keep all raw/settled/canonical actions, native work, child dual evidence, the sole8192 completion and same-search144/256 continuation identical to the real settled parent.",
        "roster": "All2802 preceding states, all seven original source-cohort labels and metadata, every historical/trigger/tail remain. Eleven focus pointers are descriptive; none filters the roster. No new draw.",
        "accounting": "2802 whole selectors, one native search and one fused root initialization each; no separate native experiment or new references/oracles. Borrow true settled/native/face/early/zero/affine/branch outputs and real ladders from the original saved settled run. Only root and settled snapshot scan fees change: one raw heap read per entry instead of two, and zero retained-list reads instead of two per retained entry. Snapshot calls still charge actual full construction; root and conditional settled objects are both constructed inside the whole call before return. Preserve all original phase head/prunability fees and native cumulative work, counted only once. No lazy object or deferred clock-external evidence.",
        "gate": "Every root, settled and final snapshot, action, route, LP/branch/integer count and child certificate must match the genuine settled output, except the two explicitly registered snapshot scan-fee pairs. Independently audit actual single-pass raw heap fees and zero retained-list fees; preserve the snapshot lazy-prune correction using actual raw and retained counts. Keep every reference/canonical/budget/exhaustion/no-restart gate. Any quality, path or fee failure holds timing. Freeze nine-arm timing before algorithm outcomes: all prior ordinary/cell/P99 and historical/trigger every-state1.05 gates remain, with candidate/settled added; trigger/long<=.95. No filtering, reruns or retuning.",
        "limitations": "Scheduler MaxWeight frame-selection algorithm development on previously used queues, four cached physical catalogues and virtual homogeneous lanes. This backend remains experimental while timing HOLDs remain; no production dispatch switch, new independent confirmation, physical scaling, flow-time or significance result. Preserve all earlier results, including the original audit INVALID and the checked settled timing HOLD; paper deferred.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered source, roster, budget or fused checkpoint rule changed")


def run_cell(model, cell, protocol, runtime, selector=None, progress=True):
    if selector is None:
        from algorithm.fused_root_dual_face_frame_candidates import select_fused_root_dual_face_frame
        selector = select_fused_root_dual_face_frame
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "native_advance_calls": 0, "native_search_constructions": 0,
        "fused_root_initialize_calls": 0, "root_settle_calls": 0}
    for index, registered in enumerate(cell["states"]):
        saved = borrowed["states"][registered["fused_root_source_index"]]
        if saved["state_id"] != registered["fused_root_source_state_id"] or any(
                saved.get(key) != value for key, value in registered.items() if not key.startswith("fused_root_source_")):
            raise ValueError("borrowed state differs from its frozen settled development source")
        result = selector(model.options, tuple(registered["queue"]), cell["lanes"],
                          protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
        row["native_search_constructions"] += 1
        row["fused_root_initialize_calls"] += 1
        row["candidate_calls"] += 1
        row["root_settle_calls"] += int(result.root_settle_work is not None)
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "traces": saved["traces"], "previous_zero_traces": saved["previous_zero_traces"],
            "original_affine_traces": saved["original_affine_traces"],
            **({"prior_routed": saved["prior_routed"]} if "prior_routed" in saved else {}),
            **{key: saved[key] for key in ("old_routed", "original_affine_routed", "previous_zero_routed",
                "previous_early_routed", "previous_dual_face_routed", "previous_dual_face_traces",
                "previous_native_root_routed", "previous_native_root_traces")},
            "previous_settled_root_routed": saved["routed"],
            "previous_settled_root_traces": saved["traces"]["new"], "routed": asdict(result)})
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
    protocol_path = _protocol_path(args.output)
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, protocol)
        print(json.dumps({"protocol": str(protocol_path), "states": protocol["expected_state_count"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("fused root development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.fused_root_dual_face_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
