"""Independent queues for the frozen exact zero-delta affine selector."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path

from algorithm.experiments.aligned_hardware_dual_reuse_development import _caps, REFERENCE_FIELDS
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_root_pruning_development import load_cell

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_fresh_confirmation_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_timing_20260928.json"
EXCLUSION_PROTOCOLS = (
    "md/experiment_artifacts/aligned_hardware_frontier_score_proof_confirmation_20260928_protocol.json",
    "md/experiment_artifacts/aligned_hardware_residual_capacity_trigger_confirmation_20260928_protocol.json",
    "md/experiment_artifacts/aligned_hardware_affine_dual_fresh_confirmation_20260928_protocol.json",
)
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092893, 2026092894, 2026092895, 2026092896)
QUEUES_PER_PANEL = 16
SOURCES = (
    "algorithm/experiments/aligned_hardware_zero_delta_affine_fresh_confirmation.py",
    "algorithm/experiments/zero_delta_affine_fresh_confirmation_summary.py",
)


def excluded_queues(protocols):
    excluded = {str(lanes): set() for lanes in LANES}
    for old in protocols:
        for lanes, queues in old.get("excluded_static_queues", {}).items():
            excluded[str(lanes)].update(map(tuple, queues))
        for cell in old.get("cells", []):
            excluded[str(cell["lanes"])].update(tuple(state["queue"]) for state in cell["states"])
        panels = old.get("queue_panels", {})
        if isinstance(panels, list):
            panels = {str(old["lanes"]): panels}
        for lanes, roster in panels.items():
            for panel in roster:
                excluded[str(lanes)].update(tuple(state["queue"]) for state in panel["states"])
    return excluded


def fresh_design(excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues

    panels = {}
    for lanes in LANES:
        seen, panels[str(lanes)] = set(excluded[str(lanes)]), []
        for seed in PANEL_SEEDS:
            queues = sample_queues(lanes, seed, QUEUES_PER_PANEL, excluded=seen)
            panels[str(lanes)].append({"seed": seed, "panel": f"independent_{seed}", "states": [
                {"state_id": f"ordinary:{seed}:{index}", "queue": queue} for index, queue in enumerate(queues)]})
            seen.update(queues)
    return panels


def build_protocol(models):
    previous = json.loads(PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    result = json.loads(PREVIOUS_OUTPUT.read_text())
    if result["decision"] != "ZERO_DELTA_AFFINE_LOCAL_TIMING_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced", "local_timing_gate_passed")):
        raise ValueError("retain the complete quality-valid preceding four-arm development advance")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    exclusions = [json.loads((ROOT / path).read_text()) for path in EXCLUSION_PROTOCOLS]
    sources = {}
    for old in (*exclusions, previous):
        if old["models"] != snapshots or tuple(old[key] for key in (
                "initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("retain the frozen catalogue models and routing budgets")
        # Historical protocols provide exclusions; the latest protocol owns
        # the current, already repaired executable literals.
        sources.update(old["execution_sources"])
    for path, literal in sources.items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    excluded = excluded_queues((*exclusions, previous))
    panels = fresh_design(excluded)
    by_cell = {(cell["node"], cell["lanes"]): cell for cell in previous["cells"]}
    cells = []
    for model in models:
        for lanes in LANES:
            states = [{**deepcopy(state), "seed": panel["seed"], "panel": panel["panel"],
                "source": "independent_confirmation", "cohort": "independent_confirmation", "caps": _caps(model),
                "historical_regression": False, "timing_group": "independent_fresh",
                "confirmation_source": "fresh", "independent_source": "fresh"}
                for panel in panels[str(lanes)] for state in panel["states"]]
            states.extend({**deepcopy(state), "independent_source": "borrowed_control",
                "independent_source_index": index, "independent_source_state_id": state["state_id"]}
                for index, state in enumerate(by_cell[model.node, lanes]["states"]))
            cells.append({"node": model.node, "lanes": lanes, "states": states})
    fresh_count = len(models) * len(LANES) * len(PANEL_SEEDS) * QUEUES_PER_PANEL
    control_count = sum(len(cell["states"]) for cell in previous["cells"])
    if fresh_count != 1024 or control_count != 1778 or len(cells) != 16:
        raise ValueError("retain 1024 independent paired states and every1778 prior state")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "zero_delta_affine_fresh_confirmation_v1", "models": snapshots,
        "cells": cells, "lanes": LANES, "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "queue_panels": panels, "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "exclusion_protocol_paths": EXCLUSION_PROTOCOLS, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget", "reference_budget",
            "oracle_state_budget", "long_lp_budget", "control_integer_state_budget", "nonregression_limit",
            "trigger_speedup_limit", "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys")},
        "expected_state_count": fresh_count + control_count,
        "expected_independent_state_count": fresh_count, "expected_borrowed_state_count": control_count,
        "expected_group_counts": {**previous["expected_group_counts"], "independent_fresh": fresh_count},
        "expected_selector_calls": 3 * fresh_count, "expected_native_advance_calls": 12 * fresh_count,
        "expected_native_constructions": 6 * fresh_count, "expected_interface_calls": 15 * fresh_count,
        "expected_new_reference_calls": fresh_count, "execution_sources": sources,
        "subsequent_timing_plan": {
            "order_seed": 2026092897, "labels": ["candidate", "affine", "branch", "long"],
            "repeats": 6, "warmups_per_label": 2, "expected_timed_calls": 67248,
            "expected_warmup_calls": 128, "nonregression_limit": 1.05, "trigger_speedup_limit": .95,
            "gate": "On independent1024 separately, candidate/affine,branch,long pooled/cell/p99<=1.05; retain all1778 preceding primary/affine gates and trigger/long<=.95 within original groups. No new ordinary every-state gate, trimming, replacement, reruns or tuning after quality outcomes.",
            "orders": "Retain all six relative candidate/branch/long permutations; affine insertion positions2/2/1/1 in randomly assigned positions, then shuffle complete state sweeps under the fixed new seed.",
        },
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
        "prior_decisions": {**result["prior_decisions"], "zero_delta_affine_timing": result["decision"]},
        "sampler": "Unused seeds2026092893-96, each16 queues/R under the unchanged Random(seed+1000R), randint(0,5R) conditional law. Exclude the original native/scaled grids through the original sampler, every stored exclusion and registered queue from the listed protocols, all registered trigger-discovery tails, all1778 preceding states and preceding accepted draws/R. Pair256 distinct R/Q draws across four fixed catalogues; no outcome screening or historical trajectory disjointness claim.",
        "reference_policy": "Before any native ladder or router on each new state, run the independent original-catalogue fullK4096 reference. Only its score-exact/canonical-open result may trigger the independent5M score-face oracle. No method-triggered rescue or state replacement; retain incomplete references and finish all states. All1778 prior states borrow their exact existing references and results.",
        "accounting": "Fresh only:1024 independent references, at most1024 reference-triggered oracles,12288 native advances on3072 cold ladders and3072 independent cold whole selectors:6144 fresh search constructions and15360 method interfaces. Count all actual10 zero-delta and17 affine fees. All1778 prior states are borrowed controls; only the original754 have real borrowed axis predecessors. No new long, axis or screening selector in quality. Subsequent timing covers2802 states with four whole-call arms and fresh clocks.",
        "gate": "Finish all16 cells,1024 new states and1778 genuine borrowed controls. Require complete independent references, feasible exact canonical/ref-matching cold routers for branch/affine/zero, truthful cumulative native and whole-selector fees and unchanged shared proof budgets. Zero and affine native snapshots/whole outputs must preserve all paths except actual child_bound_evaluations and the separately paid affine/zero fees, with guard_calls matching the real affine evaluations. Audit fresh proof/completion directly without inventing axis predecessors. No positive-hit, work dominance or timing gate in quality; keep honest nonclosures/incompleteness and prevent timing on quality HOLD.",
        "limitations": "Independent ordinary draws under one fixed conditional law, four cached catalogues and virtual identicalR8/16/32/64.256 R/Q draws are paired across four catalogues, not1024 independent observations. Old1778 controls are previously used evidence. No quality-only speed, physical scaling, flow-time or significance claim; all earlier HOLDs and the paper remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, seeds, queues, budgets or confirmation design changed")


def _dependencies():
    from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
    from algorithm.affine_dual_reuse_integer_capacity_frame_candidates import AffineDualReuseIntegerCapacitySearch
    from algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates import ZeroDeltaAffineDualReuseIntegerCapacitySearch
    from algorithm.residual_capacity_frontier_frame_candidates import select_branch_reserved_residual_capacity_frontier_frame
    from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
    from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import select_zero_delta_affine_dual_residual_capacity_frontier_frame
    from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
    from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
    return {"searches": {"branch": BranchReservedIntegerCapacitySearch, "affine": AffineDualReuseIntegerCapacitySearch,
                         "candidate": ZeroDeltaAffineDualReuseIntegerCapacitySearch},
        "selectors": {"branch": select_branch_reserved_residual_capacity_frontier_frame,
                      "affine": select_affine_dual_residual_capacity_frontier_frame,
                      "candidate": select_zero_delta_affine_dual_residual_capacity_frontier_frame},
        "new_reference": _new_reference, "oracle": canonical_at_certified_score, "oracle_key": oracle_key}


def run_cell(model, cell, protocol, runtime, dependencies=None, progress=True):
    dependencies = _dependencies() if dependencies is None else dependencies
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "affine_calls": 0, "branch_calls": 0,
        "native_advance_calls": 0, "native_search_constructions": 0}
    for index, registered in enumerate(cell["states"]):
        if registered["independent_source"] == "borrowed_control":
            saved = borrowed["states"][registered["independent_source_index"]]
            if saved["state_id"] != registered["independent_source_state_id"] or any(
                    saved.get(key) != value for key, value in registered.items() if not key.startswith("independent_source")):
                raise ValueError("borrowed control differs from its exact frozen state")
            state = {**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
                "traces": saved["traces"], "previous_dual_traces": saved["previous_dual_traces"],
                **({"prior_routed": saved["prior_routed"]} if "prior_routed" in saved else {}),
                "old_routed": saved["first_results"]["branch"],
                "previous_dual_routed": saved["first_results"]["affine"], "routed": saved["first_results"]["candidate"]}
        else:
            queue, lanes = tuple(registered["queue"]), cell["lanes"]
            reference, metrics, complete, independent = dependencies["new_reference"](
                model, queue, lanes, protocol, runtime, dependencies["oracle"])
            expected = runtime.reference_key(reference, queue, len(model.options))
            reference_kind = "full_K4096" if expected is not None else None
            if expected is None and independent is not None:
                expected = dependencies["oracle_key"](independent, len(model.options))
                reference_kind = "independent_score_face_oracle" if expected is not None else None
            row["new_reference_calls"] += 1
            row["new_oracle_calls"] += int(independent is not None)
            traces = {}
            for label, search_type in dependencies["searches"].items():
                search = search_type(model.options, queue, lanes)
                row["native_search_constructions"] += 1
                traces[label] = {}
                for cap in registered["caps"]:
                    traces[label][str(cap)] = asdict(search.advance_to(cap))
                    row["native_advance_calls"] += 1
            results = {}
            for label, selector in dependencies["selectors"].items():
                results[label] = asdict(selector(model.options, queue, lanes, protocol["initial_lp_budget"],
                                               protocol["lp_node_budget"], protocol["integer_state_budget"]))
                row["native_search_constructions"] += 1
                row[label + "_calls"] += 1
            state = {**registered, "reference": asdict(reference), "reference_metrics": metrics,
                "reference_complete": complete, "oracle": independent, "effective_reference_complete": expected is not None,
                "effective_reference_source": reference_kind,
                "traces": {"old": traces["branch"], "new": traces["candidate"]},
                "previous_dual_traces": traces["affine"], "old_routed": results["branch"],
                "previous_dual_routed": results["affine"], "routed": results["candidate"]}
        row["states"].append(state)
        if progress and (index + 1 == len(cell["states"]) or
                registered["independent_source"] != cell["states"][index + 1]["independent_source"] or
                registered["independent_source"] == "fresh" and registered["panel"] != cell["states"][index + 1]["panel"]):
            panel = registered["panel"] if registered["independent_source"] == "fresh" else "borrowed_controls"
            print(f"{model.node} R{cell['lanes']}: {panel} complete ({index + 1}/{len(cell['states'])})", flush=True)
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
        print(json.dumps({"protocol": str(protocol_path), "independent": protocol["expected_independent_state_count"],
                          "borrowed": protocol["expected_borrowed_state_count"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("independent zero-delta confirmation outputs already exist")
    rows = []
    dependencies = _dependencies()
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime, dependencies=dependencies)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream,
                      separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.zero_delta_affine_fresh_confirmation_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
