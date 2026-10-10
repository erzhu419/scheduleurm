"""Independent fresh-queue confirmation of certified parent-dual reuse."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_dual_reuse_confirmation_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_dual_reuse_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_confirmation_20260928.json"
EXCLUSION_OUTPUTS = (
    ROOT / "md/experiment_artifacts/aligned_hardware_short_lookahead_development_20260928.json",
    ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_timing_20260928.json",
)
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092831, 2026092832, 2026092833, 2026092834)
QUEUES_PER_PANEL = 64
SOURCES = (
    "algorithm/experiments/aligned_hardware_dual_reuse_confirmation.py",
    "algorithm/experiments/dual_reuse_confirmation_summary.py",
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
    if prior["decision"] != "DUAL_REUSE_DEVELOPMENT_ADVANCE" or not prior["development_gate_passed"]:
        raise ValueError("successful frozen dual-reuse development is required")
    if previous_result["decision"] != "BRANCH_RESERVED_FRESH_CONFIRMATION_ADVANCE" or not all(previous_result[key]
            for key in ("confirmation_gate_passed", "execution_checks_passed", "effective_references_complete")):
        raise ValueError("complete preceding confirmation references are required")
    exclusions = [_read_protocol(path) for path in EXCLUSION_OUTPUTS]
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for old in (development, previous, *exclusions):
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
    for old in (previous, development, *exclusions):
        for cell in old["cells"]:
            excluded[str(cell["lanes"])].update(tuple(state["queue"]) for state in cell["states"])
    panels = fresh_design(excluded)
    previous_cells = {(cell["node"], cell["lanes"]): cell for cell in previous["cells"]}
    historical = {(item["node"], item["lanes"], tuple(item["queue"])) for item in development["historical_regression_keys"]}
    cells, retained = [], set()
    for model in models:
        for lanes in LANES:
            states = [{**deepcopy(state), "panel": panel["panel"], "cohort": panel["cohort"],
                "seed": panel["seed"], "source": "fresh_confirmation", "historical_regression": False,
                "caps": _caps(model)} for panel in panels[str(lanes)] for state in panel["states"]]
            for index, registered in enumerate(previous_cells[model.node, lanes]["states"]):
                if registered["source"] != "historical_regression":
                    continue
                state = deepcopy(registered)
                state.update(source_index=index, source_state_id=registered["state_id"], historical_regression=True,
                             caps=_caps(model))
                states.append(state)
                retained.add((model.node, lanes, tuple(registered["queue"])))
            states.append({"state_id": "zero_queue", "queue": [0] * len(model.options[0].requirements),
                "source": "new_zero", "panel": "zero_queue", "cohort": "zero_queue", "seed": None,
                "historical_regression": False, "caps": _caps(model)})
            cells.append({"node": model.node, "lanes": lanes, "states": states})
    if retained != historical:
        raise ValueError("historical regressions must retain every developed node/R/Q key")
    state_count = sum(len(cell["states"]) for cell in cells)
    new_reference_count = sum(state["source"] != "historical_regression" for cell in cells for state in cell["states"])
    return json.loads(json.dumps({"schema_version": 1, "experiment": "dual_reuse_fresh_confirmation_v1",
        "models": snapshots, "lanes": LANES, "cells": cells, "queue_panels": panels,
        "cpu": min(os.sched_getaffinity(0)), "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": development["reference_budget"], "oracle_state_budget": development["oracle_state_budget"],
        "prior_decisions": {**development["prior_decisions"], "dual_reuse_development": prior["decision"],
                            "branch_reserved_confirmation": previous_result["decision"]},
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)), "prior_registered_at_utc": development["registered_at_utc"],
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "registered_at_utc": previous["registered_at_utc"]},
        "historical_regression_keys": development["historical_regression_keys"],
        "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "exclusion_protocol_paths": [str(path.with_name(path.stem + "_protocol.json").relative_to(ROOT))
                                     for path in (PREVIOUS_OUTPUT, PRIOR_OUTPUT, *EXCLUSION_OUTPUTS)],
        "expected_selector_calls": state_count * 2, "expected_native_advance_calls": state_count * 8,
        "expected_interface_calls": state_count * 10, "expected_native_constructions": state_count * 4,
        "expected_new_reference_calls": new_reference_count, "execution_sources": sources,
        "sampler": "Four new seeds2026092831-34; unchanged Random(seed+1000R), randint(0,5R) law. Exclude the preceding branch-reserved confirmation's full stored exclusion set, every state in that confirmation, dual-reuse and short-lookahead development, the branch-reserved timing roster, native/scaled grids, and earlier accepted draws within R. Four64-queue panels/R, paired across four fixed catalogues:1024 distinct R/Q samples,4096 node-states. No historical-trajectory disjointness claim.",
        "roster": "All4096 fresh node-states plus the same34 fixed historical regressions and16 zero controls. Historical references alone are borrowed from matching preceding-confirmation indices/ids/queues. Cohorts remain separate; finish every state with no replacement, missing-state substitution or tuning after outcome access.",
        "rule": development["rule"],
        "reference_policy": "Before any native or routed method execution, fresh and zero states receive an independent original-catalogue fullK4096 reference. If and only if that reference proves score but leaves canonical unresolved, run its independent certified-score oracle with5M visits, using the full reference's score and duration. Candidate results never trigger/reference/rescue the oracle. Historical records borrow only their matched six reference fields. Preserve incomplete references and continue all methods and later states.",
        "accounting": "Per state, two independently constructed native frontiers each advance through M,M+1,144,256, followed by two cold routers144/256/8192:ten external method interfaces andfour native constructions. Charge each cumulative native256 endpoint and routed final_lp once, including failed LPs and all spent integer states. Count saved duals, child upper-bound/RHS evaluations and requirement/weight reads separately. Fresh/zero reference and oracle work is separate; borrowed references are never new execution or savings.",
        "gate": "Complete coverage, valid certificates/root reservation/cumulative accounting, effective independent references complete, and both final routed actions canonical exact and reference matching. No new native service-score loss at any of four budgets. For every fresh cell and every native cap, LP and matrix totals nonincrease; routed LP/matrix/integer totals separately nonincrease, with strict pooled fresh LP or integer reduction. Independently apply the same native-cap and routed nonincrease to each cell's34-key historical subset. Zero states cannot supply gains; integer savings cannot compensate for added LP/matrix work. Preserve every loss or nonclosure and finish allcells without changing algorithm, seeds, cohort, budgets or gate.",
        "limitations": "New queue panels under a fixed sampling law on four cached catalogues and virtualR8/16/32/64. Outcomes are paired across catalogues, not4096 independent observations. Extra exact-dual arithmetic is measured as work counts; LP savings alone do not establish wall-time gains. No new physical scaling or flow-time claim. Preceding short-lookahead and timing HOLDs remain unchanged; the paper is not edited."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen sources, queues, references, catalogue or dual-reuse rule changed")


def run_cell(model, cell, protocol, runtime, dependencies=None, progress=True):
    if dependencies is None:
        from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
        from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseIntegerCapacitySearch
        from algorithm.branch_reserved_certificate_routed_frame_candidates import select_branch_reserved_certificate_routed_frame
        from algorithm.dual_reuse_certificate_routed_frame_candidates import select_dual_reuse_certificate_routed_frame
        from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
        from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
        from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
        dependencies = {"searches": {"old": BranchReservedIntegerCapacitySearch, "new": DualReuseIntegerCapacitySearch},
            "selectors": {"old": select_branch_reserved_certificate_routed_frame, "new": select_dual_reuse_certificate_routed_frame},
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
        if registered["source"] == "historical_regression":
            saved = borrowed["states"][registered["source_index"]]
            if (saved["state_id"], saved["queue"], saved["source"]) != (
                    registered["source_state_id"], registered["queue"], "historical_regression"):
                raise ValueError("borrowed historical reference changed")
            reference_fields = {key: saved[key] for key in REFERENCE_FIELDS}
        else:
            reference, metrics, complete, independent = dependencies["new_reference"](
                model, queue, lanes, protocol, runtime, dependencies["oracle"])
            expected = runtime.reference_key(reference, queue, len(model.options))
            reference_source = "full_K4096" if expected is not None else None
            if expected is None and independent is not None:
                expected = dependencies["oracle_key"](independent, len(model.options))
                reference_source = "independent_score_face_oracle" if expected is not None else None
            reference_fields = {"reference": asdict(reference), "reference_metrics": metrics,
                "reference_complete": complete, "oracle": independent,
                "effective_reference_complete": expected is not None, "effective_reference_source": reference_source}
            row["new_reference_calls"] += 1
            row["new_reference_oracle_calls"] += int(independent is not None)
        traces = {}
        for label, cls in dependencies["searches"].items():
            search = cls(model.options, queue, lanes)
            traces[label] = {str(cap): asdict(search.advance_to(cap)) for cap in registered["caps"]}
        old = dependencies["selectors"]["old"](model.options, queue, lanes, *budgets)
        row["new_old_routed_calls"] += 1
        result = dependencies["selectors"]["new"](model.options, queue, lanes, *budgets)
        row["states"].append({**registered, **reference_fields, "traces": traces,
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
        raise FileExistsError("dual-reuse confirmation outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.dual_reuse_confirmation_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
