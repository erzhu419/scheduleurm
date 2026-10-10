"""Independent queues for the unchanged exact small-dimension scheduler."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path

from algorithm.experiments.aligned_hardware_dual_reuse_development import _caps
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_root_pruning_development import load_cell

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_20260929.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_development_20260929.json"
PREVIOUS_TIMING = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_timing_20260929.json"
EXCLUSION_PROTOCOLS = (
    "md/experiment_artifacts/aligned_hardware_zero_delta_affine_fresh_confirmation_20260928_protocol.json",
)
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092911, 2026092912, 2026092913, 2026092914)
QUEUES_PER_PANEL = 16
LABELS = ("candidate", "direct", "own", "fused", "settled", "native", "face", "early", "zero", "affine", "branch", "long")
NATIVE_LABELS = ("branch", "affine", "zero", "own", "direct", "candidate")
ROOT_LABELS = ("own", "direct", "candidate")
NEW_GROUP = "new_independent_fresh"
SOURCES = (
    "algorithm/experiments/aligned_hardware_small_dimension_exact_native_confirmation.py",
    "algorithm/experiments/small_dimension_exact_native_confirmation_summary.py",
    "algorithm/experiments/aligned_hardware_small_dimension_exact_native_confirmation_timing.py",
    "algorithm/experiments/small_dimension_exact_native_confirmation_timing_summary.py",
)
TIMING_PLAN = dict(order_seed=2026092915, labels=list(LABELS), repeats=6, warmups_per_label=2,
    expected_timed_calls=275472, expected_warmup_calls=384, expected_selector_calls=275856,
    nonregression_limit=1.05, trigger_speedup_limit=.95, independent_speedup_limit=.95,
    expected_timing_gate_count=189)


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


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
    previous, timing = (json.loads(_protocol_path(path).read_text()) for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    result, clocks = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, PREVIOUS_TIMING))
    if result["decision"] != "SMALL_DIMENSION_EXACT_NATIVE_DEVELOPMENT_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "work_gate_passed",
            "development_gate_passed")):
        raise ValueError("retain the complete preceding small-dimension development")
    if clocks["decision"] != "SMALL_DIMENSION_EXACT_NATIVE_LOCAL_TIMING_ADVANCE" or not all(clocks[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced", "local_timing_gate_passed")):
        raise ValueError("complete small-dimension local timing advance is required")
    snapshots, sources = json.loads(json.dumps([model.snapshot() for model in models])), {}
    for old in (previous, timing):
        if old["models"] != snapshots or tuple(old[key] for key in (
                "initial_lp_budget", "lp_node_budget", "integer_state_budget", "reference_budget", "oracle_state_budget",
                "long_lp_budget", "control_integer_state_budget", "nonregression_limit", "trigger_speedup_limit")) != (
                    144, 256, 8192, 4096, 5_000_000, 4096, 100_000, 1.05, .95):
            raise ValueError("retain all four catalogue models, every budget and original gate")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    exclusions = [json.loads((ROOT / path).read_text()) for path in EXCLUSION_PROTOCOLS]
    if any(old["models"] != snapshots for old in exclusions):
        raise ValueError("excluded historical panels belong to the same frozen catalogue models")
    from algorithm.experiments.aligned_hardware_zero_delta_affine_fresh_confirmation import excluded_queues
    excluded = excluded_queues((*exclusions, previous, timing))
    panels = fresh_design(excluded)
    cells = []
    model_by_node = {model.node: model for model in models}
    for old in previous["cells"]:
        model, lanes = model_by_node[old["node"]], old["lanes"]
        states = [{**deepcopy(state), "seed": panel["seed"], "panel": panel["panel"],
            "source": "small_dimension_exact_native_confirmation", "cohort": NEW_GROUP, "caps": _caps(model),
            "historical_regression": False, "timing_group": NEW_GROUP, "confirmation_source": "fresh",
            "small_dimension_confirmation_source": "fresh"}
            for panel in panels[str(lanes)] for state in panel["states"]]
        states.extend({**deepcopy(state), "small_dimension_confirmation_source": "borrowed_control",
            "small_dimension_confirmation_source_index": index,
            "small_dimension_confirmation_source_state_id": state["state_id"]}
            for index, state in enumerate(old["states"]))
        cells.append({"node": model.node, "lanes": lanes, "states": states})
    count = sum(len(cell["states"]) for cell in cells)
    fresh_count = len(models) * len(LANES) * len(PANEL_SEEDS) * QUEUES_PER_PANEL
    borrowed_count = sum(len(cell["states"]) for cell in previous["cells"])
    if (count, fresh_count, borrowed_count, len(cells)) != (3826, 1024, 2802, 16):
        raise ValueError("retain every2802 old state and1024 new paired states across sixteen cells")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "small_dimension_exact_native_independent_confirmation_v1", "models": snapshots,
        "cells": cells, "lanes": LANES, "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "queue_panels": panels, "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "exclusion_protocol_paths": EXCLUSION_PROTOCOLS, "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget", "reference_budget",
            "oracle_state_budget", "long_lp_budget", "control_integer_state_budget", "nonregression_limit",
            "trigger_speedup_limit", "historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys",
            "focus_keys")},
        "expected_state_count": count, "expected_independent_state_count": fresh_count,
        "expected_borrowed_state_count": borrowed_count, "expected_fresh_state_count": fresh_count,
        "expected_control_state_count": borrowed_count,
        "expected_group_counts": {**previous["expected_group_counts"], NEW_GROUP: fresh_count},
        "expected_selector_calls": 12 * fresh_count, "expected_native_advance_calls": 24 * fresh_count,
        "expected_native_constructions": 18 * fresh_count, "expected_interface_calls": 36 * fresh_count,
        "expected_new_reference_calls": fresh_count, "expected_candidate_native_snapshots": 4 * fresh_count,
        "labels": LABELS, "native_labels": NATIVE_LABELS, "subsequent_timing_plan": TIMING_PLAN,
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
                             "registered_at_utc": previous["registered_at_utc"]},
        "execution_sources": sources,
        "prior_decisions": {**clocks["prior_decisions"], "small_dimension_exact_native_development": result["decision"],
                            "small_dimension_exact_native_timing": clocks["decision"]},
        "sampler": "Unused seeds2026092911-14, each16 queues/R, under the original Random(seed+1000R) randint(0,5R) conditional law. Exclude original native/scaled grids through the unchanged sampler, all exclusions/panels/cells of the frozen zero-delta independent protocol including registered discovery tails, every current small-dimension state and all preceding accepted draws/R. Pair256 distinct R/Q draws across four fixed catalogue models; no outcome screening, replacement or trajectory independence claim.",
        "reference_policy": "Before any method on each new state run the original-catalogue fullK4096 reference. Only its own score-exact/canonical-open result triggers the independent5M score-face oracle. A method result cannot trigger rescue. Preserve incomplete references and finish all states. All2802 preceding state references and full saved results are borrowed unchanged.",
        "accounting": "Each fresh state executes six independent cold native ladders: branch, affine, zero and genuine own/direct/candidate root-settle engines; four actual calls each atM,M+1,144,256. Then execute all twelve independent cold whole selectors, with the original long4096/100000 budget and other144/256/8192 budgets. Count24576 advances,12288 selectors,36864 method interfaces,18432 cold constructions and1024 independent references; conditional oracles are separate. Zero native ladders are genuine shared-kernel references for early/face/native/settled/fused and remain labeled as such. Whole and native cumulative work/evidence-copy fees are separate; old2802 states supply no new execution and no fabricated axis predecessors. Timing later measures275472 whole calls and384 warmups on all3826 states.",
        "gate": "Finish all sixteen cells,1024 independent states and2802 unchanged borrowed controls. Require complete independent references, all twelve fresh whole selectors feasible/score/canonical exact and reference matching, independently valid exact-LP certificates, all actual native ladders/root records/cumulative fees, complete threshold coverage, frozen logicalLP and shared integer budgets, and truthful cold-call accounting. Reuse the existing independent proof auditors without requiring fresh LP decisions to equal direct. No quality-only speed or work-dominance gate. Any nonclosure/incomplete reference/source/fee failure holds timing. Freeze twelve-arm timing before quality outcomes: retain all155 old gates, add33 new-cohort pooled/cell/p99<=1.05 gates against eleven comparators and one candidate/direct independent pooled-wall<=.95 gate,189 total. CPU and P99 remain reported. No changed rule, queue, threshold, trimming, selective repeat or tuning.",
        "limitations": "Independent ordinary queue draws under one conditional law, four unchanged cached catalogue models and virtual homogeneousR8/16/32/64. The256 R/Q draws are paired across catalogues, not1024 independent observations. The2802 borrowed controls are used development evidence. Prior HOLDs remain preserved; quality alone establishes no timing, physical scaling, flow-time or significance claim. Paper and live dispatch remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, seeds, queues, budgets or confirmation design changed")


def _dependencies():
    from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
    from algorithm.affine_dual_reuse_integer_capacity_frame_candidates import AffineDualReuseIntegerCapacitySearch
    from algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates import ZeroDeltaAffineDualReuseIntegerCapacitySearch
    from algorithm.own_tight_dual_native_kernel import OwnTightDualNativeSearch
    from algorithm.direct_probe_own_tight_native_kernel import DirectProbeOwnTightNativeSearch
    from algorithm.small_dimension_exact_native_kernel import SmallDimensionExactNativeSearch
    from algorithm.experiments.aligned_hardware_small_dimension_exact_native_timing import _selectors
    from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
    from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
    return {"searches": dict(zip(NATIVE_LABELS, (BranchReservedIntegerCapacitySearch,
        AffineDualReuseIntegerCapacitySearch, ZeroDeltaAffineDualReuseIntegerCapacitySearch,
        OwnTightDualNativeSearch, DirectProbeOwnTightNativeSearch, SmallDimensionExactNativeSearch), strict=True)),
        "selectors": _selectors(), "new_reference": _new_reference,
        "oracle": canonical_at_certified_score, "oracle_key": oracle_key}


def run_cell(model, cell, protocol, runtime, dependencies=None, progress=True):
    dependencies = _dependencies() if dependencies is None else dependencies
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "new_reference_calls": 0, "new_oracle_calls": 0,
        "selector_calls": dict.fromkeys(LABELS, 0), "native_advance_calls": 0, "native_search_constructions": 0,
        "root_settle_calls": dict.fromkeys(ROOT_LABELS, 0)}
    for index, registered in enumerate(cell["states"]):
        if registered["small_dimension_confirmation_source"] == "borrowed_control":
            saved = borrowed["states"][registered["small_dimension_confirmation_source_index"]]
            if (saved["state_id"] != registered["small_dimension_confirmation_source_state_id"] or any(
                    saved.get(key) != value for key, value in registered.items()
                    if not key.startswith("small_dimension_confirmation_source"))):
                raise ValueError("borrowed control differs from its complete frozen state")
            state = {**deepcopy(saved), **registered}
        else:
            queue, lanes = tuple(registered["queue"]), cell["lanes"]
            reference, metrics, complete, independent = dependencies["new_reference"](
                model, queue, lanes, protocol, runtime, dependencies["oracle"])
            expected = runtime.reference_key(reference, queue, len(model.options))
            kind = "full_K4096" if expected is not None else None
            if expected is None and independent is not None:
                expected = dependencies["oracle_key"](independent, len(model.options))
                kind = "independent_score_face_oracle" if expected is not None else None
            row["new_reference_calls"] += 1
            row["new_oracle_calls"] += int(independent is not None)
            traces, native_roots = {}, {}
            for label in NATIVE_LABELS:
                search = dependencies["searches"][label](model.options, queue, lanes)
                row["native_search_constructions"] += 1
                if label in ROOT_LABELS:
                    root, settled, phase = search.advance_roots_and_settle(registered["caps"][0])
                    native_roots[label] = {"root_lp": asdict(root),
                        "settled_lp": asdict(settled) if settled is not None else None,
                        "root_settle_work": asdict(phase) if phase is not None else None}
                    traces[label] = {str(registered["caps"][0]): asdict(settled if settled is not None else root)}
                    row["native_advance_calls"] += 1
                    caps = registered["caps"][1:]
                else:
                    traces[label], caps = {}, registered["caps"]
                for cap in caps:
                    traces[label][str(cap)] = asdict(search.advance_to(cap))
                    row["native_advance_calls"] += 1
            results = {}
            for label in LABELS:
                budgets = ((protocol["long_lp_budget"], protocol["control_integer_state_budget"]) if label == "long" else
                    (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"]))
                results[label] = asdict(dependencies["selectors"][label](model.options, queue, lanes, *budgets))
                row["selector_calls"][label] += 1
                row["native_search_constructions"] += 1
                if label in ROOT_LABELS:
                    row["root_settle_calls"][label] += int(results[label]["root_settle_work"] is not None)
            state = {**registered, "reference": asdict(reference), "reference_metrics": metrics,
                "reference_complete": complete, "oracle": independent, "effective_reference_complete": expected is not None,
                "effective_reference_source": kind, "traces": {"old": traces["branch"], "new": traces["zero"]},
                "original_affine_traces": traces["affine"], "previous_zero_traces": traces["zero"],
                "old_routed": results["branch"], "original_affine_routed": results["affine"],
                "previous_zero_routed": results["zero"], "previous_early_routed": results["early"],
                "previous_dual_face_routed": results["face"], "previous_dual_face_traces": traces["zero"],
                "previous_native_root_routed": results["native"], "previous_native_root_traces": traces["zero"],
                "previous_settled_root_routed": results["settled"], "previous_settled_root_traces": traces["zero"],
                "previous_fused_root_routed": results["fused"], "previous_fused_root_traces": traces["zero"],
                "previous_own_tight_routed": results["own"], "previous_own_tight_native_root": native_roots["own"],
                "previous_own_tight_native_traces": traces["own"], "previous_direct_probe_routed": results["direct"],
                "previous_direct_probe_native_root": native_roots["direct"], "previous_direct_probe_native_traces": traces["direct"],
                "candidate_native_root": native_roots["candidate"], "candidate_native_traces": traces["candidate"],
                "routed": results["candidate"], "long_routed": results["long"]}
        row["states"].append(state)
        if progress and (index + 1 == len(cell["states"]) or
                registered["small_dimension_confirmation_source"] != cell["states"][index + 1]["small_dimension_confirmation_source"] or
                registered["small_dimension_confirmation_source"] == "fresh" and registered["panel"] != cell["states"][index + 1]["panel"]):
            panel = registered["panel"] if registered["small_dimension_confirmation_source"] == "fresh" else "borrowed_controls"
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
    protocol_path = _protocol_path(args.output)
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, protocol)
        print(json.dumps({"protocol": str(protocol_path), "independent": protocol["expected_independent_state_count"],
                          "borrowed": protocol["expected_borrowed_state_count"], "interfaces": protocol["expected_interface_calls"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("small-dimension independent confirmation outputs already exist")
    rows = []
    dependencies = _dependencies()
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime, dependencies=dependencies)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream,
                      separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.small_dimension_exact_native_confirmation_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_interface_calls")}))


if __name__ == "__main__":
    main()
