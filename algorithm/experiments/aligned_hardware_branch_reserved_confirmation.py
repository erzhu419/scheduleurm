"""Fresh-queue confirmation of the frozen branch-reserved root-pruning rule."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_confirmation_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_root_pruning_confirmation_20260928.json"
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092821, 2026092822, 2026092823, 2026092824)
QUEUES_PER_PANEL = 64
SOURCES = (
    "algorithm/experiments/aligned_hardware_branch_reserved_confirmation.py",
    "algorithm/experiments/branch_reserved_confirmation_summary.py",
)


def fresh_design(excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues
    panels = {}
    for lanes in LANES:
        seen, panels[str(lanes)] = set(excluded[str(lanes)]), []
        for seed in PANEL_SEEDS:
            queues = sample_queues(lanes, seed, QUEUES_PER_PANEL, excluded=seen)
            panels[str(lanes)].append({"panel": f"fresh_{seed}", "seed": seed,
                "cohort": "fresh_confirmation", "states": [
                    {"state_id": f"fresh:{seed}:{index}", "queue": queue}
                    for index, queue in enumerate(queues)]})
            seen.update(queues)
    return panels


def build_protocol(models):
    prior = json.loads(PRIOR_OUTPUT.read_text())
    development = json.loads(PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json").read_text())
    previous = json.loads(PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    previous_result = json.loads(PREVIOUS_OUTPUT.read_text())
    if prior["decision"] != "BRANCH_RESERVED_DEVELOPMENT_ADVANCE" or not prior["development_gate_passed"]:
        raise ValueError("successful frozen branch-reserved development is required")
    if (previous_result["decision"] != "ROOT_PRUNING_FRESH_CONFIRMATION_HOLD" or not all(previous_result[key]
            for key in ("execution_checks_passed", "effective_references_complete", "routed_quality_gate_passed", "finite_budget_gate_passed"))):
        raise ValueError("quality-valid frozen root-pruning confirmation HOLD must be retained")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    for old in (development, previous):
        if snapshots != old["models"]:
            raise ValueError("cached physical catalogue models changed")
        if tuple(old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("confirmation must retain the developed routing budgets")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
    excluded = {str(lanes): set(map(tuple, previous["excluded_static_queues"][str(lanes)])) for lanes in LANES}
    for old in (previous, development):
        for cell in old["cells"]:
            excluded[str(cell["lanes"])].update(tuple(state["queue"]) for state in cell["states"])
    panels = fresh_design(excluded)
    previous_cells = {(cell["node"], cell["lanes"]): cell for cell in previous["cells"]}
    historical = {(item["node"], item["lanes"], tuple(item["queue"])) for item in development["historical_regression_keys"]}
    cells, retained = [], set()
    for model in models:
        for lanes in LANES:
            states = [{**deepcopy(state), "panel": panel["panel"], "cohort": panel["cohort"],
                "seed": panel["seed"], "source": "fresh_confirmation"}
                for panel in panels[str(lanes)] for state in panel["states"]]
            for index, registered in enumerate(previous_cells[model.node, lanes]["states"]):
                if registered["source"] != "historical_regression":
                    continue
                key = model.node, lanes, tuple(registered["queue"])
                state = deepcopy(registered)
                state.update(source_index=index, source_state_id=registered["state_id"])
                states.append(state)
                retained.add(key)
            states.append({"state_id": "zero_queue", "queue": [0] * len(model.options[0].requirements),
                "source": "new_zero", "panel": "zero_queue", "cohort": "zero_queue", "seed": None})
            cells.append({"node": model.node, "lanes": lanes, "states": states})
    if retained != historical:
        raise ValueError("historical regressions must retain every developed node/R/Q key")
    return json.loads(json.dumps({"schema_version": 1, "experiment": "branch_reserved_fresh_confirmation_v1",
        "models": snapshots, "lanes": LANES, "cells": cells, "queue_panels": panels,
        "cpu": min(os.sched_getaffinity(0)), "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "prior_decisions": {**development["prior_decisions"], "branch_reserved_development": prior["decision"],
            "root_pruning_confirmation": previous_result["decision"]},
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)), "prior_registered_at_utc": development["registered_at_utc"],
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
        "historical_regression_keys": development["historical_regression_keys"],
        "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "execution_sources": {**development["execution_sources"], **previous["execution_sources"],
            **{path: (ROOT / path).read_text() for path in SOURCES}},
        "sampler": "Four new seeds2026092821-24; unchanged random.Random(seed+1000R), randint(0,5R) law. Exclude the preceding root-pruning confirmation's full stored exclusion set and all its queues, including fresh2811-14, plus all branch-reserved development queues, native/scaled grids, and earlier accepted draws within R. Four64-queue panels per R, paired across four fixed catalogues;1024 distinct R/Q samples and4096 node-states. No historical-trajectory disjointness claim.",
        "roster": "All4096 new node-states, the same34 historical trigger/tail node/R/Q regressions and16 zero controls. Borrow only historical references from their matching preceding-confirmation cell indices. Fresh, historical and zero cohorts remain separate; no replacement or missing-state substitution.",
        "budget_rule": development["budget_rule"],
        "calls": "Fresh and zero states receive a new independent original-catalogue fullK4096 reference and its reference-triggered oracle before both selector arms. Historical references alone are borrowed. Each state executes an original and a branch-reserved native search through root_count,root_count+1,144,256 with one frontier per search, then one independently executed old and one new routed144/256/8192 selector. Ten interfaces and four native constructions per state; charge cumulative256 endpoints and routed final_lp once. Reference and oracle costs separately reported.",
        "reference_policy": previous["reference_policy"],
        "gate": "Complete frozen coverage and valid certificates/accounting, including actualLP+skipped_roots<=cap and no branch at root-only cap. Effective independent references complete; every new routed action canonical exact and matching its reference. No new-kernel score loss at any of four registered budgets. Fresh routed LP/matrix/integer totals nonincreasing in every cell and pooled LP strictly reduced; separately enforce per-cell nonincrease for the34 historical regressions. Zero states cannot supply work gains. Finish every cell after honest nonclosure or loss without changing the rule, budgets, seeds or gate.",
        "limitations": "New queue panels under the fixed sampling law, four cached catalogues, virtualR8/16/32/64 lanes. Outcomes are paired across catalogues, not4096 independent observations. Reserving skipped-root positions changes finite-budget paths; equality is not presumed. Reference oracle remains independent code conditional on the full reference's score proof. No wall-time, new physical scaling or flow-time claim. Preceding root-pruning confirmation HOLD and certificate-routed timing HOLD remain unchanged."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, catalogue, queues, reference rule or branch allowance changed")


def run_cell(model, cell, protocol, runtime, progress=True):
    from algorithm.resumable_integer_capacity_frame_candidates import IntegerCapacitySearch
    from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
    from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
    from algorithm.branch_reserved_certificate_routed_frame_candidates import select_branch_reserved_certificate_routed_frame
    from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
    from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    minima = {}
    for option in model.options:
        requirement = tuple(option.requirements)
        minima[requirement] = min(minima.get(requirement, float("inf")), max(1.0, option.duration))
    roots = len(set(minima.values()))
    caps = (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0,
        "new_reference_oracle_calls": 0, "new_old_routed_calls": 0}
    for index, registered in enumerate(cell["states"]):
        queue, lanes = tuple(registered["queue"]), cell["lanes"]
        if registered["source"] == "historical_regression":
            saved = borrowed["states"][registered["source_index"]]
            if (saved["state_id"], saved["queue"]) != (registered["source_state_id"], registered["queue"]):
                raise ValueError("borrowed historical reference changed")
            reference_fields = {key: saved[key] for key in ("reference", "reference_metrics", "reference_complete",
                "oracle", "effective_reference_complete", "effective_reference_source")}
        else:
            reference, metrics, complete, independent = _new_reference(
                model, queue, lanes, protocol, runtime, canonical_at_certified_score)
            expected = runtime.reference_key(reference, queue, len(model.options))
            reference_source = "full_K4096" if expected is not None else None
            if expected is None and independent is not None:
                expected = oracle_key(independent, len(model.options))
                reference_source = "independent_score_face_oracle" if expected is not None else None
            reference_fields = {"reference": asdict(reference), "reference_metrics": metrics,
                "reference_complete": complete, "oracle": independent,
                "effective_reference_complete": expected is not None, "effective_reference_source": reference_source}
            row["new_reference_calls"] += 1
            row["new_reference_oracle_calls"] += int(independent is not None)
        traces = {}
        for label, cls in (("old", IntegerCapacitySearch), ("new", BranchReservedIntegerCapacitySearch)):
            search = cls(model.options, queue, lanes)
            traces[label] = {str(cap): asdict(search.advance_to(cap)) for cap in caps}
        budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
        old = select_certificate_routed_frame(model.options, queue, lanes, *budgets)
        row["new_old_routed_calls"] += 1
        result = select_branch_reserved_certificate_routed_frame(model.options, queue, lanes, *budgets)
        row["states"].append({**registered, **reference_fields, "caps": caps, "traces": traces,
            "old_routed": asdict(old), "routed": asdict(result)})
        if progress and (index + 1 == len(cell["states"]) or registered["panel"] != cell["states"][index + 1]["panel"]):
            print(f"{model.node} R{lanes}: {registered['panel']} complete ({index + 1}/{len(cell['states'])})", flush=True)
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
        raise FileExistsError("branch-reserved confirmation outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
        print(f"{model.node} R{cell['lanes']}: {len(row['states'])} states saved", flush=True)
    from algorithm.experiments.branch_reserved_confirmation_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
