"""Fresh-queue confirmation of frozen root pruning, without timing claims."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_root_pruning_confirmation_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_root_pruning_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_confirmation_20260927.json"
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092811, 2026092812, 2026092813, 2026092814)
QUEUES_PER_PANEL = 64
SOURCES = (
    "algorithm/experiments/aligned_hardware_root_pruning_confirmation.py",
    "algorithm/experiments/root_pruning_confirmation_summary.py",
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
    if prior["decision"] != "ROOT_PRUNING_DEVELOPMENT_ADVANCE" or not prior["development_gate_passed"]:
        raise ValueError("successful frozen root-pruning development is required")
    if not all(previous_result[key] for key in (
            "certificate_checks_passed", "effective_references_complete", "execution_checks_passed")):
        raise ValueError("complete quality-valid historical references are required")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    if snapshots != development["models"] or previous["models"] != snapshots:
        raise ValueError("cached physical catalogue models changed")
    for old in (development, previous):
        if tuple(old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("confirmation must retain the developed routing budgets")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
    excluded = {str(lanes): set(map(tuple, previous["excluded_static_queues"][str(lanes)])) for lanes in LANES}
    for lanes in LANES:
        excluded[str(lanes)].update(tuple(state["queue"])
            for panel in previous["queue_panels"][str(lanes)] for state in panel["states"])
    for cell in development["cells"]:
        excluded[str(cell["lanes"])].update(tuple(state["queue"]) for state in cell["states"])
    panels = fresh_design(excluded)
    triggers = {(item["node"], item["lanes"], tuple(item["queue"])) for item in previous["historical_trigger_keys"]}
    cells, found = [], set()
    for model in models:
        for lanes in LANES:
            states = [{**deepcopy(state), "panel": panel["panel"], "cohort": panel["cohort"],
                "seed": panel["seed"], "source": "fresh_confirmation"}
                for panel in panels[str(lanes)] for state in panel["states"]]
            old_states = [(panel, state) for panel in previous["queue_panels"][str(lanes)] for state in panel["states"]]
            retained = set()
            for index, (panel, state) in enumerate(old_states):
                key = model.node, lanes, tuple(state["queue"])
                trigger, tail = key in triggers, panel["cohort"] == "ordinary_tail_regression"
                if not (trigger or tail) or key[2] in retained:
                    continue
                retained.add(key[2])
                if trigger:
                    found.add(key)
                states.append({"state_id": "regression:" + state["state_id"], "queue": state["queue"],
                    "source": "historical_regression", "source_index": index, "source_state_id": state["state_id"],
                    "panel": "historical_regression", "cohort": "historical_regression", "seed": None,
                    "historical_trigger": trigger, "ordinary_tail": tail})
            states.append({"state_id": "zero_queue", "queue": [0] * len(model.options[0].requirements),
                "source": "new_zero", "panel": "zero_queue", "cohort": "zero_queue", "seed": None})
            cells.append({"node": model.node, "lanes": lanes, "states": states})
    if found != triggers:
        raise ValueError("all frozen historical trigger keys must be retained")
    return json.loads(json.dumps({"schema_version": 1, "experiment": "root_pruning_fresh_confirmation_v1",
        "models": snapshots, "lanes": LANES, "cells": cells, "queue_panels": panels,
        "cpu": min(os.sched_getaffinity(0)),
        "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "prior_decisions": {"development": prior["decision"], "confirmation": previous_result["decision"]},
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)),
        "prior_registered_at_utc": development["registered_at_utc"],
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
        "historical_trigger_keys": previous["historical_trigger_keys"],
        "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "execution_sources": {**development["execution_sources"], **previous["execution_sources"],
            **{path: (ROOT / path).read_text() for path in SOURCES}},
        "sampler": "Four new seeds2026092811-14; unchanged random.Random(seed+1000R), randint(0,5R) law. Exclude the preceding confirmation's entire stored exclusion set and roster, all root-pruning development queues, native/scaled grids, and prior accepted draws within R. Four64-queue panels per R, paired across four cached catalogues;1024 distinct R/Q samples and4096 node-states. No disjointness claim for historical trajectories.",
        "roster": "All4096 fresh node-states plus the complete18 historical node/R/Q trigger keys,16 ordinary-tail node-states deduplicated against triggers, and16 zero controls. Fresh, historical regression, and zero cohorts remain separate; none may substitute for a missing fresh state.",
        "calls": "Fresh and zero states first receive a new independent original-catalogue fullK4096 reference and its reference-triggered oracle. Historical references alone are borrowed from their frozen confirmation cells. Every state executes two independently constructed resumable searches advanced through root_count,root_count+1,144,256 and one cold old plus one cold new routed selector at144/256/8192. Ten selector-interface calls and four native search constructions per state. Charge each trace's cumulative256 endpoint once; reference and oracle work separately.",
        "reference_policy": "Oracle only if the independent full reference has score_exact and lacks canonical_exact; use only that reference's proved score and incumbent duration with a fixed5000000-state cap, before either selector. Keep original reference nonclosure and oracle exhaustion; neither arm supplies its own reference. Unproved reference score or exhausted oracle yields INCOMPLETE_REFERENCE without budget escalation.",
        "gate": "Complete frozen coverage, valid certificates/accounting and complete effective references. New routed canonical action equals the reference for every fresh, regression, and zero state. No new-kernel score regression at any of four registered budgets. Fresh routed LP/matrix/integer totals do not increase in any cell and pooled LP calls strictly decrease; regression totals separately do not increase in any cell. Zero controls cannot supply cost improvements. Finish every cell after a nonclosure or loss; no replacement, retuning, or timing gate.",
        "limitations": "New queue panels under a fixed sampling law, four fixed cached catalogues and virtualR8/16/32/64 lanes. Catalogue outcomes are paired, not4096 independent observations. Root bounds concern exact-duration strata; finite-budget paths can change. Reference oracle is independent code conditional on the full reference's score certificate. No wall-time, new physical scaling, or flow-time claim. Earlier confirmation timing HOLD remains unchanged."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, catalogue, roster, reference rule or budgets changed")


def run_cell(model, cell, protocol, runtime, progress=True):
    from algorithm.resumable_integer_capacity_frame_candidates import IntegerCapacitySearch
    from algorithm.root_pruned_integer_capacity_frame_candidates import RootPrunedIntegerCapacitySearch
    from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
    from algorithm.root_pruned_certificate_routed_frame_candidates import select_root_pruned_certificate_routed_frame
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
                raise ValueError("borrowed reference queue changed")
            reference_fields = {key: saved[key] for key in ("reference", "reference_metrics", "reference_complete",
                "oracle", "effective_reference_complete", "effective_reference_source")}
        else:
            reference, metrics, complete, independent = _new_reference(
                model, queue, lanes, protocol, runtime, canonical_at_certified_score)
            expected = runtime.reference_key(reference, queue, len(model.options))
            source_name = "full_K4096" if expected is not None else None
            if expected is None and independent is not None:
                expected = oracle_key(independent, len(model.options))
                source_name = "independent_score_face_oracle" if expected is not None else None
            reference_fields = {"reference": asdict(reference), "reference_metrics": metrics,
                "reference_complete": complete, "oracle": independent,
                "effective_reference_complete": expected is not None, "effective_reference_source": source_name}
            row["new_reference_calls"] += 1
            row["new_reference_oracle_calls"] += int(independent is not None)
        traces = {}
        for label, cls in (("old", IntegerCapacitySearch), ("new", RootPrunedIntegerCapacitySearch)):
            search = cls(model.options, queue, lanes)
            traces[label] = {str(cap): asdict(search.advance_to(cap)) for cap in caps}
        budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
        old_routed = select_certificate_routed_frame(model.options, queue, lanes, *budgets)
        row["new_old_routed_calls"] += 1
        routed = select_root_pruned_certificate_routed_frame(model.options, queue, lanes, *budgets)
        row["states"].append({**registered, **reference_fields, "caps": caps, "traces": traces,
            "old_routed": asdict(old_routed), "routed": asdict(routed)})
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
        raise FileExistsError("confirmation outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
        print(f"{model.node} R{cell['lanes']}: {len(row['states'])} states saved", flush=True)
    from algorithm.experiments.root_pruning_confirmation_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
