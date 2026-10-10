"""Independent ordinary queues for the frozen affine-dual residual selector."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_fresh_confirmation_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_timing_20260928.json"
EXCLUSION_PROTOCOLS = (
    "md/experiment_artifacts/aligned_hardware_frontier_score_proof_confirmation_20260928_protocol.json",
    "md/experiment_artifacts/aligned_hardware_residual_capacity_trigger_confirmation_20260928_protocol.json",
)
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092881, 2026092882, 2026092883, 2026092884)
QUEUES_PER_PANEL = 16
SOURCES = (
    "algorithm/experiments/aligned_hardware_affine_dual_fresh_confirmation.py",
    "algorithm/experiments/affine_dual_fresh_confirmation_summary.py",
)


def fresh_design(excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues

    panels = {}
    for lanes in LANES:
        seen, panels[str(lanes)] = set(excluded[str(lanes)]), []
        for seed in PANEL_SEEDS:
            queues = sample_queues(lanes, seed, QUEUES_PER_PANEL, excluded=seen)
            panels[str(lanes)].append({"seed": seed, "panel": f"fresh_{seed}", "states": [
                {"state_id": f"ordinary:{seed}:{index}", "queue": queue} for index, queue in enumerate(queues)]})
            seen.update(queues)
    return panels


def build_protocol(models):
    previous = json.loads(PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    result = json.loads(PREVIOUS_OUTPUT.read_text())
    if result["decision"] != "AFFINE_DUAL_LOCAL_TIMING_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "repeat_paths_stable",
            "frozen_residual_results_reproduced", "local_timing_gate_passed")):
        raise ValueError("the complete frozen affine development timing must have advanced")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    exclusions = [json.loads((ROOT / path).read_text()) for path in EXCLUSION_PROTOCOLS]
    sources = {}
    # Earlier protocols supply queue exclusions; retain the latest frozen code.
    for old in (*exclusions, previous):
        if old["models"] != snapshots or tuple(old[key] for key in (
                "initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("retain the frozen catalogue models and routing budgets")
        sources.update(old["execution_sources"])
    for path, literal in sources.items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    excluded = {str(lanes): set() for lanes in LANES}
    for old in (previous, *exclusions):
        for lanes, queues in old.get("excluded_static_queues", {}).items():
            excluded[str(lanes)].update(map(tuple, queues))
        for cell in old.get("cells", []):
            excluded[str(cell["lanes"])].update(tuple(state["queue"]) for state in cell["states"])
    # The entire registered discovery panel is excluded, including its unrun tail.
    for panel in exclusions[1]["queue_panels"]:
        excluded[str(exclusions[1]["lanes"])].update(tuple(state["queue"]) for state in panel["states"])
    panels = fresh_design(excluded)
    by_cell = {(cell["node"], cell["lanes"]): cell for cell in previous["cells"]}
    cells = []
    for model in models:
        for lanes in LANES:
            states = [{**deepcopy(state), "seed": panel["seed"], "panel": panel["panel"],
                "source": "fresh_confirmation", "cohort": "fresh_confirmation", "caps": _caps(model),
                "historical_regression": False, "timing_group": "fresh_confirmation", "confirmation_source": "fresh"}
                for panel in panels[str(lanes)] for state in panel["states"]]
            states.extend({**deepcopy(state), "confirmation_source": "borrowed_control",
                "confirmation_source_index": index, "confirmation_source_state_id": state["state_id"]}
                for index, state in enumerate(by_cell[model.node, lanes]["states"]))
            cells.append({"node": model.node, "lanes": lanes, "states": states})
    fresh_count = len(models) * len(LANES) * len(PANEL_SEEDS) * QUEUES_PER_PANEL
    control_count = sum(len(cell["states"]) for cell in previous["cells"])
    if fresh_count != 1024 or control_count != 754 or len(cells) != 16:
        raise ValueError("retain 1024 fresh paired states and every754 preceding control")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "affine_dual_fresh_confirmation_v1", "models": snapshots,
        "cells": cells, "lanes": LANES, "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "queue_panels": panels, "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "exclusion_protocol_paths": EXCLUSION_PROTOCOLS, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget", "reference_budget",
            "oracle_state_budget", "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys",
            "previous_timing_failure_keys", "affine_focus_keys")},
        "expected_state_count": fresh_count + control_count, "expected_fresh_state_count": fresh_count,
        "expected_control_state_count": control_count,
        "expected_group_counts": {**previous["expected_group_counts"], "fresh_confirmation": fresh_count},
        "expected_selector_calls": 2 * fresh_count, "expected_native_advance_calls": 8 * fresh_count,
        "expected_native_constructions": 4 * fresh_count, "expected_interface_calls": 10 * fresh_count,
        "expected_new_reference_calls": fresh_count, "execution_sources": sources,
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
        "prior_decisions": {**result["prior_decisions"], "affine_dual_timing": result["decision"]},
        "sampler": "Four unused seeds2026092881-84, each16 queues/R under unchanged Random(seed+1000R), randint(0,5R) law. Exclude native/scaled grids through the original sampler, all stored earlier confirmation exclusions and queues, the full registered residual trigger-discovery panels including unrun tails, all754 old controls and preceding accepted draws/R. Pair256 distinct R/Q samples across the four frozen catalogues; no outcome screening or historical-trajectory disjointness claim.",
        "reference_policy": "Before either native ladder or router for each fresh state, run an independent original-catalogue fullK4096 reference. If and only if its score is exact and its canonical action remains open, spend at most5M visits on its independent score-face oracle. Candidate/control results never trigger rescue. Preserve incomplete references and finish all states. All754 old controls borrow only their exact frozen results and references.",
        "accounting": "Fresh only:1024 independent references, at most1024 reference-triggered oracles,8192 native advances on2048 cold ladders and2048 cold routers,4096 fresh search constructions,10240 method interfaces. Count cumulative256 native work and final router work once, with all17 affine fees. Old754 controls, their references/native traces/results and prior axis records are borrowed costs, never new execution. No additional prune-first or axis predecessor selector, no screening. Subsequent clocks include all1778 states and three arms.",
        "gate": "Finish all16 cells and all1024 fresh plus754 exact borrowed controls. Require complete independent references, valid root reservation/cumulative actual fees, correct feasible score certificates, and both fresh cold routers fully score/canonical exact and reference matching. Fresh paid frontier/retained-vector/canonical-budget certificates are audited directly; no fabricated axis predecessor. No native work dominance or speed gate in this quality phase. Honest nonclosure or reference incompleteness holds confirmation and prevents timing; retain all states without replacement or tuning.",
        "limitations": "New ordinary queue seeds under one fixed conditional sampling law, four cached catalogues and virtualR8/16/32/64. Only256 independent R/Q draws are paired across catalogues, not1024 independent observations. Quality alone proves neither timing nor physical scaling or flow-time benefit. Old controls are not new confirmation evidence. Earlier HOLD results and the paper remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, seeds, queues, budgets or confirmation design changed")


def _dependencies():
    from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
    from algorithm.affine_dual_reuse_integer_capacity_frame_candidates import AffineDualReuseIntegerCapacitySearch
    from algorithm.residual_capacity_frontier_frame_candidates import select_branch_reserved_residual_capacity_frontier_frame
    from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
    from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
    from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
    return {"searches": {"old": BranchReservedIntegerCapacitySearch, "new": AffineDualReuseIntegerCapacitySearch},
        "selectors": {"old": select_branch_reserved_residual_capacity_frontier_frame,
                      "new": select_affine_dual_residual_capacity_frontier_frame},
        "new_reference": _new_reference, "oracle": canonical_at_certified_score, "oracle_key": oracle_key}


def run_cell(model, cell, protocol, runtime, dependencies=None, progress=True):
    dependencies = _dependencies() if dependencies is None else dependencies
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "candidate_calls": 0, "branch_calls": 0, "native_advance_calls": 0, "native_search_constructions": 0}
    for index, registered in enumerate(cell["states"]):
        if registered["confirmation_source"] == "borrowed_control":
            saved = borrowed["states"][registered["confirmation_source_index"]]
            if saved["state_id"] != registered["confirmation_source_state_id"] or any(
                    saved.get(key) != value for key, value in registered.items() if not key.startswith("confirmation_source")):
                raise ValueError("borrowed control differs from its exact frozen state")
            state = {**registered, **{key: saved[key] for key in REFERENCE_FIELDS},
                "traces": saved["traces"], "prior_routed": saved["prior_routed"],
                "old_routed": saved["first_results"]["branch"], "routed": saved["first_results"]["candidate"]}
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
                row["branch_calls" if label == "old" else "candidate_calls"] += 1
            state = {**registered, "reference": asdict(reference), "reference_metrics": metrics,
                "reference_complete": complete, "oracle": independent, "effective_reference_complete": expected is not None,
                "effective_reference_source": reference_kind, "traces": traces,
                "old_routed": results["old"], "routed": results["new"]}
        row["states"].append(state)
        if progress and (index + 1 == len(cell["states"]) or
                registered["confirmation_source"] != cell["states"][index + 1]["confirmation_source"] or
                registered["confirmation_source"] == "fresh" and registered["panel"] != cell["states"][index + 1]["panel"]):
            panel = registered["panel"] if registered["confirmation_source"] == "fresh" else "borrowed_controls"
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
        print(json.dumps({"protocol": str(protocol_path), "fresh": protocol["expected_fresh_state_count"],
                          "controls": protocol["expected_control_state_count"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("fresh confirmation outputs already exist")
    rows = []
    dependencies = _dependencies()
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime, dependencies=dependencies)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream,
                      separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.affine_dual_fresh_confirmation_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
