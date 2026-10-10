"""Develop native own-tight root boxes without free root branching."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_direct_probe_own_tight_native_development_20260929.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_own_tight_dual_native_development_20260928.json"
PREVIOUS_TIMING = ROOT / "md/experiment_artifacts/aligned_hardware_own_tight_dual_native_timing_20260928.json"
SOURCES = (
    "algorithm/direct_probe_own_tight_native_kernel.py",
    "algorithm/direct_probe_own_tight_native_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_direct_probe_own_tight_native_development.py",
    "algorithm/experiments/direct_probe_own_tight_native_development_summary.py",
    "algorithm/experiments/aligned_hardware_direct_probe_own_tight_native_timing.py",
    "algorithm/experiments/direct_probe_own_tight_native_timing_summary.py",
)
TIMING_PLAN = dict(order_seed=2026092905,
    labels=["candidate", "own", "fused", "settled", "native", "face", "early", "zero", "affine", "branch", "long"],
    repeats=6, warmups_per_label=2, expected_timed_calls=184932, expected_warmup_calls=352,
    nonregression_limit=1.05, trigger_speedup_limit=.95)


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    previous, timing = (json.loads(_protocol_path(path).read_text())
                        for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    result, clocks = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    if result["decision"] != "OWN_TIGHT_DUAL_NATIVE_DEVELOPMENT_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "work_gate_passed",
            "development_gate_passed")):
        raise ValueError("retain the complete quality-valid own-tight parent development")
    if clocks["decision"] != "OWN_TIGHT_DUAL_NATIVE_LOCAL_TIMING_HOLD" or not all(clocks[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced")):
        raise ValueError("retain the complete preceding own-tight timing HOLD")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for parent in (previous, timing):
        if parent["models"] != snapshots:
            raise ValueError("retain the same four cached physical catalogues")
        if tuple(parent[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
                "reference_budget", "oracle_state_budget")) != (144, 256, 8192, 4096, 5_000_000):
            raise ValueError("retain all native, completion and reference budgets")
        for path, literal in parent["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    cells = [{"node": old["node"], "lanes": old["lanes"], "states": [
        {**deepcopy(state), "direct_probe_source_index": index,
         "direct_probe_source_state_id": state["state_id"]}
        for index, state in enumerate(old["states"])]} for old in previous["cells"]]
    count = sum(len(cell["states"]) for cell in cells)
    counts = {name: sum(state["timing_group"] == name for cell in cells for state in cell["states"])
              for name in previous["expected_group_counts"]}
    if count != 2802 or counts != previous["expected_group_counts"] or len(cells) != 16:
        raise ValueError("retain all2802 states, seven cohorts and sixteen configurations")
    focuses = deepcopy(previous["focus_keys"])
    focuses.extend({"node": cell["node"], "lanes": cell["lanes"], **{key: state[key]
        for key in ("state_id", "queue", "timing_group")}} for cell in cells for state in cell["states"]
        if cell["node"] == "gpu_b" and cell["lanes"] == 64 and state["queue"] == [122, 3, 4])
    if len(focuses) != 13:
        raise ValueError("retain twelve preceding focuses plus the actual R64 historical regression")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "direct_probe_own_tight_native_development_v1", "models": snapshots,
        "cells": cells, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget", "long_lp_budget", "control_integer_state_budget",
            "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys",
            "expected_group_counts", "nonregression_limit", "trigger_speedup_limit")},
        "expected_state_count": count, "expected_selector_calls": count,
        "expected_native_advance_calls": 4 * count, "expected_native_constructions": 2 * count,
        "expected_interface_calls": 5 * count, "expected_candidate_native_snapshots": 4 * count,
        "expected_root_settle_calls": result["executed_root_settle_calls"],
        "focus_keys": focuses, "subsequent_timing_plan": TIMING_PLAN,
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
                             "registered_at_utc": previous["registered_at_utc"]},
        "execution_sources": sources,
        "prior_decisions": {**clocks["prior_decisions"], "own_tight_dual_native_timing": clocks["decision"]},
        "rule": "Fully materialize the same genuine raw-M checkpoint as the own-tight parent. Keep native own-bound endpoint restrictions, cheap canonical pruning and singleton action updates. In the root-only phase stop before constructing children for a still-open solved box: retain the complete shrunk box with its original serial and valid own cache. Record stopped_for_branch separately from stopped_for_lp; root phase split_regions is zero. Ordinary native continuation still branches normally. Use the existing sole shared8192 integer probe on the live complete frontier; exhaustion continues the same generator at144/256. No threshold, coefficient, budget or state-specific exception.",
        "roster": "Retain every2802 parent queue, complete metadata, sixteen configurations and seven source labels. Thirteen focus pointers are descriptive; no filtering or new draw.",
        "accounting": "One cold whole selector and one separate cold native audit search perstate. Four actual native calls at the original caps plus the whole call give14010 interfaces and5604 constructions. Save genuine raw/settled audit checkpoints and four cumulative prefixes; count each actual search's final cumulative work once. Whole and audit event snapshot-copy fees remain separately charged. Parent own native root/traces and routed result remain genuine borrowed controls; true fused and older records retain their own original provenance and fees. Head checks include branch-position inspection and complete-box restoration. No new references, oracles, routed controls or screening in this quality stage.",
        "gate": "Require complete references, exact original canonical actions, valid own endpoint restrictions, truthful native/probe/phase fees, complete retained threshold coverage,144/256 LP limits and the sole8192 shared integer budget. Raw-M must reproduce the true parent own record including its actual33-field native fees and evidence. Compare new whole mathematics to independently executed new native prefixes while auditing the extra snapshot-copy charges separately. The root phase must not construct children, increment branch_splits or consume LP calls; stopped-for-branch must retain its complete region. Preserve all parent controls. Any quality, fee, source, coverage or nonclosure failure holds timing. Freeze the eleven-arm plan before outcomes; preserve all nine preceding controls and their gates, add true own with the identical1.05 rules, retain trigger/long<=.95. No selective repeats or retuning.",
        "limitations": "Scheduler MaxWeight frame-selection development on previously used queues, four fixed cached catalogues and virtual homogeneous lanes. Source names and repeated timing do not establish new independence. Stopping root splitting may change completion work; it is not a measured gain until the full frozen protocol passes. Preserve all prior HOLD results; paper and live dispatch remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered source, roster, budgets or direct-probe rule changed")


def run_cell(model, cell, protocol, runtime, selector=None, search_class=None, progress=True):
    if selector is None:
        from algorithm.direct_probe_own_tight_native_frame_candidates import select_direct_probe_own_tight_native_frame
        selector = select_direct_probe_own_tight_native_frame
    if search_class is None:
        from algorithm.direct_probe_own_tight_native_kernel import DirectProbeOwnTightNativeSearch
        search_class = DirectProbeOwnTightNativeSearch
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "native_advance_calls": 0, "native_search_constructions": 0,
        "root_settle_calls": 0}
    for index, registered in enumerate(cell["states"]):
        saved = borrowed["states"][registered["direct_probe_source_index"]]
        if saved["state_id"] != registered["direct_probe_source_state_id"] or any(
                saved.get(key) != value for key, value in registered.items() if not key.startswith("direct_probe_source_")):
            raise ValueError("borrowed state differs from its frozen own-tight development source")
        queue = tuple(registered["queue"])
        audit = search_class(model.options, queue, cell["lanes"])
        row["native_search_constructions"] += 1
        root, settled, phase = audit.advance_roots_and_settle(registered["caps"][0])
        row["native_advance_calls"] += 1
        native_root = {"root_lp": asdict(root), "settled_lp": asdict(settled) if settled is not None else None,
                       "root_settle_work": asdict(phase) if phase is not None else None}
        native_traces = {str(registered["caps"][0]): asdict(settled if settled is not None else root)}
        for cap in registered["caps"][1:]:
            native_traces[str(cap)] = asdict(audit.advance_to(cap))
            row["native_advance_calls"] += 1
        result = selector(model.options, queue, cell["lanes"], protocol["initial_lp_budget"],
                          protocol["lp_node_budget"], protocol["integer_state_budget"])
        row["native_search_constructions"] += 1
        row["candidate_calls"] += 1
        row["root_settle_calls"] += int(result.root_settle_work is not None)
        row["states"].append({**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
            "traces": saved["traces"], "previous_zero_traces": saved["previous_zero_traces"],
            "original_affine_traces": saved["original_affine_traces"],
            **({"prior_routed": saved["prior_routed"]} if "prior_routed" in saved else {}),
            **{key: saved[key] for key in ("old_routed", "original_affine_routed", "previous_zero_routed",
                "previous_early_routed", "previous_dual_face_routed", "previous_dual_face_traces",
                "previous_native_root_routed", "previous_native_root_traces",
                "previous_settled_root_routed", "previous_settled_root_traces",
                "previous_fused_root_routed", "previous_fused_root_traces")},
            "previous_own_tight_routed": saved["routed"],
            "previous_own_tight_native_root": saved["candidate_native_root"],
            "previous_own_tight_native_traces": saved["candidate_native_traces"],
            "candidate_native_root": native_root, "candidate_native_traces": native_traces,
            "routed": asdict(result)})
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
        print(json.dumps({"protocol": str(protocol_path), "states": protocol["expected_state_count"],
                          "interfaces": protocol["expected_interface_calls"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("direct-probe development outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.direct_probe_own_tight_native_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
