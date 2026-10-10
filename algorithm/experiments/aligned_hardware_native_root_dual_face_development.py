"""Remove duplicated root preparation on the complete frozen queue roster."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_native_root_dual_face_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_dual_face_root_canonical_development_20260928.json"
PREVIOUS_TIMING = ROOT / "md/experiment_artifacts/aligned_hardware_dual_face_root_canonical_timing_20260928.json"
SOURCES = (
    "algorithm/native_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates.py",
    "algorithm/native_root_dual_face_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_native_root_dual_face_development.py",
    "algorithm/experiments/native_root_dual_face_development_summary.py",
    "algorithm/experiments/aligned_hardware_native_root_dual_face_timing.py",
    "algorithm/experiments/native_root_dual_face_timing_summary.py",
)
TIMING_PLAN = dict(order_seed=2026092901,
    labels=["candidate", "face", "early", "zero", "affine", "branch", "long"],
    repeats=6, warmups_per_label=2, expected_timed_calls=117684, expected_warmup_calls=224,
    nonregression_limit=1.05, trigger_speedup_limit=.95)
FOCUS_QUEUES = (("gpu_a", 64, (88, 8, 23)), ("gpu_a", 64, (118, 119, 269)),
    ("cpu_a", 8, (27, 15, 34)), ("gpu_a", 64, (158, 196, 145)),
    ("gpu_a", 64, (124, 10, 80)), ("gpu_b", 64, (122, 235, 280)),
    ("cpu_a", 32, (96, 15, 13)), ("cpu_a", 64, (192, 192, 317)))


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    previous, timing = (json.loads(_protocol_path(path).read_text())
                        for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    result, clocks = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    if result["decision"] != "DUAL_FACE_ROOT_CANONICAL_DEVELOPMENT_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed",
            "work_gate_passed", "development_gate_passed")):
        raise ValueError("retain complete quality-valid dual-face development")
    if clocks["decision"] != "DUAL_FACE_ROOT_CANONICAL_LOCAL_TIMING_HOLD" or not all(clocks[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced")):
        raise ValueError("retain the complete preceding dual-face timing HOLD")
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
        {**deepcopy(state), "native_root_source_index": index,
         "native_root_source_state_id": state["state_id"]}
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
        raise ValueError("retain all six focuses and both preceding timing counterexamples")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "native_root_dual_face_development_v1", "models": snapshots,
        "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget", "long_lp_budget", "control_integer_state_budget",
            "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys",
            "expected_group_counts", "nonregression_limit", "trigger_speedup_limit")},
        "expected_state_count": count, "expected_selector_calls": count,
        "expected_native_advance_calls": 0, "expected_native_constructions": count,
        "expected_interface_calls": count, "focus_keys": focuses, "subsequent_timing_plan": TIMING_PLAN,
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
                             "registered_at_utc": previous["registered_at_utc"]},
        "execution_sources": sources,
        "prior_decisions": {**clocks["prior_decisions"], "dual_face_root_canonical_timing": clocks["decision"]},
        "rule": "One new static native search performs its own original grouping once and derives actual root_count M internally. advance_roots(initial144) rejects M>144 and yields the unbranched raw M snapshot. Continue the same generator at144/256 only along the frozen dual-face route, with one shared8192 integer probe. All native mathematics, dual-face export/completion, actions, raw snapshots and subsequent search work retain the preceding true dual-face results. Remove only the external catalogue scan previously used to compute M; no proxy budget, free preprocessing, restart, rescue budget or queue-specific condition.",
        "roster": "Every2802 state from the preceding complete dual-face development and timing HOLD, with all original metadata, seven groups, historical/trigger failures and actual controls retained. Previously used development states, no new draw.",
        "accounting": "2802 cold whole selectors containing exactly one native search each, no separate native experiment, zero new reference/oracle/screening calls. Borrow the genuine preceding dual-face outputs, true zero/affine/branch native ladders and whole controls, early outputs and references. Charge final cumulative native work once, the unique probe once and all actual dual-face work. The eight removed external catalogue scan counters must be zero; all three actual root snapshot counters and all native preprocessing remain charged. Whole-selector timing includes native grouping and every construction/routing cost.",
        "gate": "Audit all states against effective independent reference scores/canonical actions and exact certificates. New whole-selector output must equal the true preceding dual-face output in every field except the eight genuinely removed routing scan counters. Preserve raw roots, initial/final native snapshots, probe work, evidence and statuses. Any accounting, path or quality failure holds timing. Subsequent seven-arm timing retains every preceding group/cell/P99 and historical/trigger every-state1.05 gate, trigger/long<=.95 and adds face nonregression with the same group rules. Freeze six rounds and all tails before outcome access; no filtering, rerun or retuning.",
        "limitations": "Previously used development queues, four cached catalogues and virtual identical lanes. Borrowed references and old controls. No independent confirmation, physical scaling, flow-time, significance or adoption claim. Retain every preceding HOLD; paper and old sources/protocols/results unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered source, roster, budgets or native root rule changed")


def run_cell(model, cell, protocol, runtime, selector=None, progress=True):
    if selector is None:
        from algorithm.native_root_dual_face_frame_candidates import select_native_root_dual_face_frame
        selector = select_native_root_dual_face_frame
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "native_advance_calls": 0, "native_search_constructions": 0}
    for index, registered in enumerate(cell["states"]):
        saved = borrowed["states"][registered["native_root_source_index"]]
        if saved["state_id"] != registered["native_root_source_state_id"] or any(
                saved.get(key) != value for key, value in registered.items() if not key.startswith("native_root_source_")):
            raise ValueError("borrowed state differs from its frozen development source")
        result = selector(model.options, tuple(registered["queue"]), cell["lanes"],
                          protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
        row["native_search_constructions"] += 1
        row["candidate_calls"] += 1
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "traces": saved["traces"], "previous_zero_traces": saved["previous_zero_traces"],
            "original_affine_traces": saved["original_affine_traces"],
            **({"prior_routed": saved["prior_routed"]} if "prior_routed" in saved else {}),
            **{key: saved[key] for key in ("old_routed", "original_affine_routed", "previous_zero_routed", "previous_early_routed")},
            "previous_dual_face_routed": saved["routed"],
            "previous_dual_face_traces": saved["traces"]["new"], "routed": asdict(result)})
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
        raise FileExistsError("native root development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.native_root_dual_face_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
