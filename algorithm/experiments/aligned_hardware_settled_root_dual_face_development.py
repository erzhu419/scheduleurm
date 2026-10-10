"""Develop paid root settling without another LP on the frozen queue roster."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_settled_root_dual_face_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_native_root_dual_face_development_20260928.json"
PREVIOUS_TIMING = ROOT / "md/experiment_artifacts/aligned_hardware_native_root_dual_face_timing_20260928.json"
SOURCES = (
    "algorithm/settled_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates.py",
    "algorithm/settled_root_dual_face_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_settled_root_dual_face_development.py",
    "algorithm/experiments/settled_root_dual_face_development_summary.py",
    "algorithm/experiments/aligned_hardware_settled_root_dual_face_timing.py",
    "algorithm/experiments/settled_root_dual_face_timing_summary.py",
)
TIMING_PLAN = dict(order_seed=2026092902,
    labels=["candidate", "native", "face", "early", "zero", "affine", "branch", "long"],
    repeats=6, warmups_per_label=2, expected_timed_calls=134496, expected_warmup_calls=256,
    nonregression_limit=1.05, trigger_speedup_limit=.95)
FOCUS_QUEUES = (("gpu_a", 64, (88, 8, 23)), ("gpu_a", 64, (118, 119, 269)),
    ("cpu_a", 8, (27, 15, 34)), ("gpu_a", 64, (158, 196, 145)),
    ("gpu_a", 64, (124, 10, 80)), ("gpu_b", 64, (122, 235, 280)),
    ("cpu_a", 32, (96, 15, 13)), ("cpu_a", 64, (192, 192, 317)),
    ("gpu_b", 64, (120, 123, 258)), ("cpu_a", 64, (54, 58, 290)))


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    previous, timing = (json.loads(_protocol_path(path).read_text())
                        for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    result, clocks = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    if result["decision"] != "NATIVE_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed",
            "work_gate_passed", "development_gate_passed", "behavior_preserved")):
        raise ValueError("retain complete quality-valid native-root development")
    if clocks["decision"] != "NATIVE_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD" or not all(clocks[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced")):
        raise ValueError("retain the complete preceding native-root timing HOLD")
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
        {**deepcopy(state), "settled_root_source_index": index,
         "settled_root_source_state_id": state["state_id"]}
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
        raise ValueError("retain all eight focuses and both additional timing counterexamples")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "settled_root_dual_face_development_v1", "models": snapshots,
        "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget", "long_lp_budget", "control_integer_state_budget",
            "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys",
            "expected_group_counts", "nonregression_limit", "trigger_speedup_limit")},
        "expected_state_count": count, "expected_selector_calls": count,
        "expected_native_advance_calls": 0, "expected_native_constructions": count,
        "expected_interface_calls": count,
        "expected_root_settle_calls": result["executed"]["whole_selector"]["root_probe_calls"],
        "focus_keys": focuses, "subsequent_timing_plan": TIMING_PLAN,
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
                             "registered_at_utc": previous["registered_at_utc"]},
        "execution_sources": sources,
        "prior_decisions": {**clocks["prior_decisions"], "native_root_dual_face_timing": clocks["decision"]},
        "rule": "Keep one native search and its original raw M checkpoint. Only score-certified canonical-open roots first settle the same live heap without another LP: prune, split already-solved roots or evaluate solved singleton roots, stopping before the first live unsolved box needs an LP. Children remain unsolved, so this stage processes at most M paid roots without a new arbitrary cap. If canonical closes, return that actual native action without an integer probe. Otherwise use the existing exact dual-face completion on the actual remaining child boxes and settled baseline, with the same sole8192 probe. Exhaustion retains all work and continues the same native generator at144/256. No restart, new LP allowance or queue-specific condition.",
        "roster": "Every2802 preceding development state, seven original groups, all metadata and genuine controls retained. Include every historical/trigger failure, with ten frozen focus pointers. No new draw or independent cohort.",
        "accounting": "2802 cold whole selectors, each with one native search; no separate native experiment or new reference/oracle/screening calls. Borrow genuine native-root, face, early, zero, affine and branch outputs and the true native ladders. Charge final cumulative native LP/branch/capacity/dual/guard work once, the sole completion once and all actual dual-face export operations. External root scan fees remain zero. Charge extra root-settle call/head/prunability checks and its additional snapshot separately; its branch/prune/pop and before/after counts are diagnostics already covered by native work. Whole clocks include every preparation, phase, check and construction cost.",
        "gate": "Require all effective reference scores, canonical actions, exact certificate/child-box mathematics and cumulative fees. Raw root snapshot equals the actual preceding raw M; root settling changes no LP call count or certified score. Settled native closure must match the genuine zero144 prefix apart from its smaller declared LP cap. Every actual144/256 continuation must match the frozen zero native ladder. Unsettled routes retain their actual parent path. A failed sole probe returns its true settled baseline and retains all cost before continuation. Any quality, path or accounting failure holds timing. Subsequent eight-arm timing preserves every previous group/cell/P99 and historical/trigger every-state1.05 gate, trigger/long<=.95 and adds the actual preceding native control under the same rules. Freeze all six rounds before outcome access; no filtering, rerun or retuning.",
        "limitations": "Previously used development queues, four cached catalogues and virtual identical lanes; borrowed references and controls. No independent confirmation, physical scaling, flow-time, significance or adoption claim. Retain every preceding HOLD; paper and old sources/protocols/results unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered source, roster, budgets or root settling rule changed")


def run_cell(model, cell, protocol, runtime, selector=None, progress=True):
    if selector is None:
        from algorithm.settled_root_dual_face_frame_candidates import select_settled_root_dual_face_frame
        selector = select_settled_root_dual_face_frame
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "native_advance_calls": 0, "native_search_constructions": 0,
        "root_settle_calls": 0}
    for index, registered in enumerate(cell["states"]):
        saved = borrowed["states"][registered["settled_root_source_index"]]
        if saved["state_id"] != registered["settled_root_source_state_id"] or any(
                saved.get(key) != value for key, value in registered.items() if not key.startswith("settled_root_source_")):
            raise ValueError("borrowed state differs from its frozen development source")
        result = selector(model.options, tuple(registered["queue"]), cell["lanes"],
                          protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
        row["native_search_constructions"] += 1
        row["candidate_calls"] += 1
        row["root_settle_calls"] += int(result.root_settle_work is not None)
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "traces": saved["traces"], "previous_zero_traces": saved["previous_zero_traces"],
            "original_affine_traces": saved["original_affine_traces"],
            **({"prior_routed": saved["prior_routed"]} if "prior_routed" in saved else {}),
            **{key: saved[key] for key in ("old_routed", "original_affine_routed", "previous_zero_routed",
                "previous_early_routed", "previous_dual_face_routed", "previous_dual_face_traces")},
            "previous_native_root_routed": saved["routed"],
            "previous_native_root_traces": saved["traces"]["new"], "routed": asdict(result)})
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
        raise FileExistsError("settled root development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.settled_root_dual_face_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
