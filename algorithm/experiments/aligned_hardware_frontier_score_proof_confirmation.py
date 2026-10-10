"""Confirm paid-frontier score proofs on independent frozen queue panels."""
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
from algorithm.experiments.aligned_hardware_dual_reuse_development import _caps, REFERENCE_FIELDS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_score_proof_confirmation_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_score_proof_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_dual_reuse_confirmation_20260928.json"
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092841, 2026092842, 2026092843, 2026092844)
QUEUES_PER_PANEL = 64
BORROWED_SOURCES = ("historical_regression", "failure_regression")
SOURCES = (
    "algorithm/experiments/aligned_hardware_frontier_score_proof_confirmation.py",
    "algorithm/experiments/frontier_score_proof_confirmation_summary.py",
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


def _read_protocol(path):
    return json.loads(path.with_name(path.stem + "_protocol.json").read_text())


def build_protocol(models):
    prior, previous_result = (json.loads(path.read_text()) for path in (PRIOR_OUTPUT, PREVIOUS_OUTPUT))
    development, previous = _read_protocol(PRIOR_OUTPUT), _read_protocol(PREVIOUS_OUTPUT)
    if prior["decision"] != "FRONTIER_SCORE_PROOF_DEVELOPMENT_ADVANCE" or not all(prior[key] for key in (
            "development_gate_passed", "execution_checks_passed", "effective_references_complete", "failure_development_gate_passed")):
        raise ValueError("the complete frontier-proof development advance is required")
    if previous_result["decision"] != "DUAL_REUSE_FRESH_CONFIRMATION_HOLD" or not all(previous_result[key] for key in (
            "execution_checks_passed", "effective_references_complete", "final_score_gate_passed", "work_gate_passed")):
        raise ValueError("the preceding independent confirmation HOLD must be retained")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for old in (development, previous):
        if snapshots != old["models"]:
            raise ValueError("cached physical catalogue models changed")
        if tuple(old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("confirmation must retain the developed routing budgets")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    excluded = {str(lanes): set(map(tuple, previous["excluded_static_queues"][str(lanes)])) for lanes in LANES}
    for old in (previous, development):
        for cell in old["cells"]:
            excluded[str(cell["lanes"])].update(tuple(state["queue"]) for state in cell["states"])
    panels = fresh_design(excluded)
    historical = {(item["node"], item["lanes"], tuple(item["queue"])) for item in development["historical_regression_keys"]}
    failures = {(item["node"], item["lanes"], tuple(item["queue"])) for item in development["failure_development_keys"]}
    prior_cells = {(cell["node"], cell["lanes"]): cell for cell in development["cells"]}
    cells, retained_history, retained_failures = [], set(), set()
    for model in models:
        for lanes in LANES:
            states = [{**deepcopy(state), "panel": panel["panel"], "cohort": panel["cohort"],
                "seed": panel["seed"], "source": "fresh_confirmation", "historical_regression": False,
                "caps": _caps(model)} for panel in panels[str(lanes)] for state in panel["states"]]
            for index, registered in enumerate(prior_cells[model.node, lanes]["states"]):
                key = model.node, lanes, tuple(registered["queue"])
                if key not in historical and key not in failures:
                    continue
                source = "historical_regression" if key in historical else "failure_regression"
                state = deepcopy(registered)
                state.update(source=source, cohort=source, panel=source,
                    source_index=index, source_state_id=registered["state_id"],
                    source_record_source=registered["source"], historical_regression=key in historical,
                    caps=_caps(model))
                states.append(state)
                (retained_history if key in historical else retained_failures).add(key)
            states.append({"state_id": "zero_queue", "queue": [0] * len(model.options[0].requirements),
                "source": "new_zero", "panel": "zero_queue", "cohort": "zero_queue", "seed": None,
                "historical_regression": False, "caps": _caps(model)})
            cells.append({"node": model.node, "lanes": lanes, "states": states})
    if retained_history != historical or retained_failures != failures:
        raise ValueError("all historical and paired failure regression keys must be retained")
    count = sum(len(cell["states"]) for cell in cells)
    new_reference_count = sum(state["source"] not in BORROWED_SOURCES for cell in cells for state in cell["states"])
    return json.loads(json.dumps({"schema_version": 1, "experiment": "frontier_score_proof_fresh_confirmation_v1",
        "models": snapshots, "lanes": LANES, "cells": cells, "queue_panels": panels,
        "cpu": min(os.sched_getaffinity(0)), "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": development["reference_budget"], "oracle_state_budget": development["oracle_state_budget"],
        "historical_regression_keys": development["historical_regression_keys"],
        "failure_regression_keys": development["failure_development_keys"],
        "prior_decisions": {**development["prior_decisions"], "frontier_score_proof_development": prior["decision"],
                            "dual_reuse_confirmation": previous_result["decision"]},
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)), "prior_registered_at_utc": development["registered_at_utc"],
        "reference_source": {"path": str(PRIOR_OUTPUT.relative_to(ROOT)), "registered_at_utc": development["registered_at_utc"]},
        "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "exclusion_protocol_paths": [str(path.with_name(path.stem + "_protocol.json").relative_to(ROOT))
                                     for path in (PREVIOUS_OUTPUT, PRIOR_OUTPUT)],
        "expected_state_count": count,
        "expected_fresh_state_count": sum(state["source"] == "fresh_confirmation" for cell in cells for state in cell["states"]),
        "expected_failure_regression_count": len(failures),
        "expected_selector_calls": count * 2, "expected_native_advance_calls": count * 8,
        "expected_interface_calls": count * 10, "expected_native_constructions": count * 4,
        "expected_new_reference_calls": new_reference_count, "execution_sources": sources,
        "sampler": "Four new seeds2026092841-44; unchanged Random(seed+1000R), randint(0,5R) law. Exclude the preceding dual-reuse confirmation's full stored exclusions and all4146 states, plus all742 frontier-development states. This union also retains all preceding static grids/development/timing exclusions. Reject earlier accepted draws within R. Four64-queue panels/R, paired across four catalogues:1024 distinct R/Q draws and4096 fresh node-states. No historical-trajectory disjointness claim.",
        "roster": "All4096 fresh node-states,34 original historical regressions,16 zero controls andfour separately labelled known failure regressions, total4150. Borrow only38 matched independent reference records for historical/failure controls. Preserve every seed, queue, source and state, complete allcells without tuning, replacement or deleting a failure.",
        "rule": development["rule"],
        "reference_policy": "Before any method runs on a fresh or zero state, solve an independent original-catalogue fullK4096 reference. If and only if it certifies score but not canonical completion, call its independent5M-state certified-score oracle using only that reference's score/duration. Candidate outcomes cannot trigger or rescue references. Historical/failure controls borrow their matched six independent reference fields from the completed development. Incomplete references remain recorded while all methods and remaining states execute.",
        "accounting": "Each state runs two independent native ladders through M,M+1,144,256 and two independently constructed cold frontier-proof routers144/256/8192:41500 method interfaces and16600 native constructions. Charge each native256 cumulative endpoint and each routed final_lp once, all failed LPs and all spent proof/canonical integer states. Count exact-dual arithmetic and the eight frontier-export counters explicitly.4112 new full references and any reference-only oracle work are separate;38 borrowed references cannot supply new gain.",
        "gate": "All states covered, independent effective references complete, valid native/root/dual/strict-proof/canonical certificates and shared8192 caps. Both final routers must be canonical exact and match reference. Cold paid144/final256 snapshots must equal their independent native ladder snapshots; frontier evidence must cover all exact-bound-above-own-score boxes. At each native cap no new service-score loss. In every fresh cell, native-cap LP/matrix and routed LP/matrix/integer totals separately nonincrease; strict pooled fresh LP or integer gain required. Apply native-cap and routed nonincrease and final quality independently to the34 historical andfour failure subsets in their cells. Zero controls supply no gain. Retain every witness, exhaustion, nonclosure and incomplete reference with no post-outcome change.",
        "limitations": "Independent new queue draws under the fixed law, paired across four cached catalogues with virtualR8/16/32/64.4096 node-states are1024 distinct R/Q draws, not independent catalogue replicates. Counted work and extra exact arithmetic do not establish wall-time, flow-time or physical-scaling gains. All earlier confirmation/late/timing HOLDs remain unchanged; the paper is not edited."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, queues, references, catalogue or frontier-proof rule changed")


def run_cell(model, cell, protocol, runtime, dependencies=None, progress=True):
    if dependencies is None:
        from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
        from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseIntegerCapacitySearch
        from algorithm.frontier_score_proof_certificate_routed_frame_candidates import (
            select_branch_reserved_frontier_score_proof_frame, select_dual_reuse_frontier_score_proof_frame)
        from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
        from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
        from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
        dependencies = {"searches": {"old": BranchReservedIntegerCapacitySearch, "new": DualReuseIntegerCapacitySearch},
            "selectors": {"old": select_branch_reserved_frontier_score_proof_frame, "new": select_dual_reuse_frontier_score_proof_frame},
            "new_reference": _new_reference, "oracle": canonical_at_certified_score, "oracle_key": oracle_key}
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0,
        "new_reference_oracle_calls": 0, "new_old_routed_calls": 0}
    budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
    for index, registered in enumerate(cell["states"]):
        queue, lanes = tuple(registered["queue"]), cell["lanes"]
        if registered["source"] in BORROWED_SOURCES:
            saved = borrowed["states"][registered["source_index"]]
            if (saved["state_id"], saved["queue"], saved["source"]) != (
                    registered["source_state_id"], registered["queue"], registered["source_record_source"]):
                raise ValueError("borrowed historical or failure reference changed")
            reference_fields = {key: saved[key] for key in REFERENCE_FIELDS}
        else:
            reference, metrics, complete, independent = dependencies["new_reference"](
                model, queue, lanes, protocol, runtime, dependencies["oracle"])
            expected = runtime.reference_key(reference, queue, len(model.options))
            reference_kind = "full_K4096" if expected is not None else None
            if expected is None and independent is not None:
                expected = dependencies["oracle_key"](independent, len(model.options))
                reference_kind = "independent_score_face_oracle" if expected is not None else None
            reference_fields = {"reference": asdict(reference), "reference_metrics": metrics,
                "reference_complete": complete, "oracle": independent,
                "effective_reference_complete": expected is not None, "effective_reference_source": reference_kind}
            row["new_reference_calls"] += 1
            row["new_reference_oracle_calls"] += int(independent is not None)
        traces = {}
        for label, cls in dependencies["searches"].items():
            search = cls(model.options, queue, lanes)
            traces[label] = {str(cap): asdict(search.advance_to(cap)) for cap in registered["caps"]}
        old = dependencies["selectors"]["old"](model.options, queue, lanes, *budgets)
        row["new_old_routed_calls"] += 1
        new = dependencies["selectors"]["new"](model.options, queue, lanes, *budgets)
        row["states"].append({**registered, **reference_fields, "traces": traces,
                              "old_routed": asdict(old), "routed": asdict(new)})
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
        print(json.dumps({"protocol": str(protocol_path), "states": protocol["expected_state_count"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("frontier-proof confirmation outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.frontier_score_proof_confirmation_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
