"""Native-only fresh trigger discovery followed by residual-capacity confirmation."""
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
from algorithm.experiments.aligned_hardware_frontier_trigger_discovery import run_discovery

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_residual_capacity_trigger_confirmation_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_residual_capacity_frontier_development_20260928.json"
DISCOVERY_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_trigger_discovery_20260928.json"
CONFIRMATION_PROTOCOL = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_score_proof_confirmation_20260928_protocol.json"
LANES = 64
PANEL_SEEDS = (2026092861, 2026092862, 2026092863, 2026092864)
QUEUES_PER_PANEL, BLOCK_SIZE, MAX_ROUNDS = 1024, 64, 16
MIN_TRIGGER_QUEUES = MIN_TRIGGER_SEEDS = 2
SOURCES = (
    "algorithm/experiments/aligned_hardware_residual_capacity_trigger_confirmation.py",
    "algorithm/experiments/residual_capacity_trigger_confirmation_summary.py",
)


def fresh_design(excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues
    seen, panels = set(excluded), []
    for seed in PANEL_SEEDS:
        queues = sample_queues(LANES, seed, QUEUES_PER_PANEL, excluded=seen)
        panels.append({"seed": seed, "panel": f"trigger_{seed}", "states": [
            {"state_id": f"trigger:{seed}:{index}", "queue": queue, "seed": seed,
             "index": index, "panel": f"trigger_{seed}", "cohort": "conditional_trigger_validation"}
            for index, queue in enumerate(queues)]})
        seen.update(queues)
    return panels


def build_protocol(models):
    previous, earlier = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, DISCOVERY_OUTPUT))
    previous_path = PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json")
    discovery_path = DISCOVERY_OUTPUT.with_name(DISCOVERY_OUTPUT.stem + "_protocol.json")
    old, screened, confirmed = (json.loads(path.read_text()) for path in (
        previous_path, discovery_path, CONFIRMATION_PROTOCOL))
    if previous["decision"] != "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE" or not all(previous[key] for key in (
            "execution_checks_passed", "effective_references_complete", "development_gate_passed")):
        raise ValueError("the completed residual-capacity development advance is required")
    if earlier["decision"] != "FRONTIER_TRIGGER_VALIDATION_HOLD" or not earlier["execution_checks_passed"]:
        raise ValueError("the previous independent trigger HOLD must remain recorded")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for prior in (old, screened):
        if snapshots != prior["models"] or tuple(prior[key] for key in (
                "initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("frozen cached catalogues or routing budgets changed")
        for path, literal in prior["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    excluded = {tuple(queue) for protocol in (screened, confirmed)
                for queues in protocol["excluded_static_queues"].values() for queue in queues}
    for prior in (old, confirmed):
        excluded.update(tuple(state["queue"]) for cell in prior["cells"] for state in cell["states"])
    excluded.update(tuple(state["queue"]) for panel in screened["queue_panels"] for state in panel["states"])
    panels = fresh_design(excluded)
    fixed = {(item["node"], item["lanes"], tuple(item["queue"])) for key in (
        "failure_development_keys", "axis_failure_development_keys") for item in old[key]}
    controls, retained = [], set()
    for model in models:
        states = []
        for cell in old["cells"]:
            if (cell["node"], cell["lanes"]) != (model.node, LANES):
                continue
            for index, saved in enumerate(cell["states"]):
                key = (model.node, LANES, tuple(saved["queue"]))
                if key not in fixed:
                    continue
                state = deepcopy(saved)
                state.update(borrowed_from="residual_capacity_development", borrowed_index=index,
                    source_state_id=saved["state_id"], source_record_source=saved["source"])
                if state["caps"] != _caps(model):
                    raise ValueError("borrowed native budget ladder changed")
                states.append(state)
                retained.add(key)
        controls.append({"node": model.node, "lanes": LANES, "states": states})
    if retained != fixed:
        raise ValueError("all previous paired failure controls must be retained")
    decisions = {**previous["prior_decisions"], "residual_capacity_development": previous["decision"]}
    limitations = ("Fresh conditional triggers under the unchanged R64 queue law, paired across four cached catalogues. "
        "Known failure controls are reused evidence. Conditional successes do not establish wall-time, flow-time, "
        "a population trigger rate or physical scaling. All earlier HOLDs remain recorded.")
    template = {"schema_version": 1, "experiment": "residual_capacity_trigger_validation_v1",
        "models": snapshots, "lanes": [LANES], "cpu": min(os.sched_getaffinity(0)),
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": old["reference_budget"], "oracle_state_budget": old["oracle_state_budget"],
        "control_cells": controls, "failure_development_keys": old["failure_development_keys"],
        "axis_failure_development_keys": old["axis_failure_development_keys"],
        "reference_source": {"path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "registered_at_utc": old["registered_at_utc"]},
        "prior_decisions": decisions, "execution_sources": sources, "limitations": limitations,
        "reference_policy": "Lock every trigger across all catalogues and all eight fixed controls before any reference or selector. Each fresh state receives an independent original-catalogue K4096 reference before its methods. Only that reference's certified score may request a 5M-state canonical oracle. Controls borrow six reference fields, eight native snapshots and prior axis outputs from the completed residual development; incomplete references stay visible.",
        "accounting": "Fresh states execute two native ladders M,M+1,144,256, two cold axis-capacity predecessor routers and two cold residual-capacity routers. Charge cumulative native256 and each final routed LP once; every proof, prefix, canonical, dual and export fee remains recorded. Controls execute only the two residual routers; their references/native/predecessor outputs are borrowed, with saved previous residual outputs retained for reproduction. References and native-only discovery are separate executed work.",
        "gate": "Every locked state and reference is present, paid native and routed snapshots agree, all strict/axis/prefix/canonical fees and shared8192 budgets are valid. Both residual arms must finish score and canonical certificates matching independent reference. For each state/arm LP/matrix work equals its axis predecessor and total integer states do not increase. All controls reproduce their saved residual results. Raw trigger and residual table activation sufficiency require two distinct fresh queues from two seeds; known controls cannot satisfy either. Honest method/work failures take priority over insufficiency. No rule, budget, gate or roster changes after outcomes.",
    }
    return json.loads(json.dumps({"schema_version": 1, "experiment": "residual_capacity_trigger_confirmation_v1",
        "models": snapshots, "lanes": LANES, "cpu": template["cpu"], "panel_seeds": PANEL_SEEDS,
        "queues_per_panel": QUEUES_PER_PANEL, "block_size": BLOCK_SIZE, "max_rounds": MAX_ROUNDS,
        "min_trigger_queues": MIN_TRIGGER_QUEUES, "min_trigger_seeds": MIN_TRIGGER_SEEDS,
        "min_residual_queues": 2, "min_residual_seeds": 2, "queue_panels": panels,
        "excluded_static_queues": {str(LANES): sorted(excluded)},
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "expected_max_screen_calls": len(models) * len(PANEL_SEEDS) * QUEUES_PER_PANEL,
        "validation_template": template, "execution_sources": sources, "prior_decisions": decisions,
        "exclusion_protocol_paths": [str(path.relative_to(ROOT)) for path in (
            previous_path, discovery_path, CONFIRMATION_PROTOCOL)],
        "sampler": "New seeds2026092861-64; unchanged Random(seed+1000R)/randint(0,5R) law atR64. Exclude all stored confirmation/static queues across capacities, all preceding discovery panels and all residual-development states. Freeze1024 distinct queues per seed, rejecting prior accepted draws across seeds.",
        "discovery_rule": "Use the frozen native-only run_discovery:64 draws per seed per whole round, every queue across all four catalogues, each one cold DualReuseIntegerCapacitySearch.advance_to(256). A queue triggers only if a native endpoint has score_exact=False. No reference, oracle, integer probe, predecessor or residual result enters screening. Stop at first complete round with two distinct triggers from two seeds, else all16 rounds. Keep every stopping-round trigger.",
        "validation_rule": "Lock all discovered triggers across every catalogue plus all eight paired failure controls before reference access. Complete every state even when references or methods do not close. Residual activation is measured only after validation; insufficient activation never extends discovery or changes the sampler.",
        "gate": template["gate"], "limitations": limitations}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen discovery sources, queues, catalogue or stopping rule changed")


def validation_protocol(protocol, discovery, models):
    validation = deepcopy(protocol["validation_template"])
    validation.update(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        discovery_registered_at_utc=protocol["registered_at_utc"],
        trigger_states=deepcopy(discovery["trigger_states"]), screen_stop_reason=discovery["stop_reason"])
    controls = {cell["node"]: cell for cell in validation["control_cells"]}
    validation["cells"] = [{"node": model.node, "lanes": protocol["lanes"], "states": [
        {**deepcopy(state), "source": "fresh_confirmation", "historical_regression": False, "caps": _caps(model)}
        for state in discovery["trigger_states"]] + deepcopy(controls[model.node]["states"])} for model in models]
    fresh = len(models) * len(discovery["trigger_states"])
    control = sum(len(cell["states"]) for cell in validation["control_cells"])
    validation.update(expected_state_count=fresh + control, expected_fresh_state_count=fresh,
        expected_control_state_count=control, expected_new_reference_calls=fresh,
        expected_borrowed_reference_count=control, expected_borrowed_native_snapshots=control * 8,
        expected_borrowed_prior_axis_count=control * 2, expected_prior_axis_selector_calls=fresh * 2,
        expected_residual_selector_calls=(fresh + control) * 2, expected_native_advance_calls=fresh * 8,
        expected_interface_calls=fresh * 12 + control * 2, expected_native_constructions=fresh * 6 + control * 2)
    return json.loads(json.dumps(validation))


def _dependencies():
    from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
    from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseIntegerCapacitySearch
    from algorithm.axis_capacity_frontier_score_proof_frame_candidates import (
        select_branch_reserved_axis_capacity_frontier_frame, select_dual_reuse_axis_capacity_frontier_frame)
    from algorithm.residual_capacity_frontier_frame_candidates import (
        select_branch_reserved_residual_capacity_frontier_frame, select_dual_reuse_residual_capacity_frontier_frame)
    from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
    from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
    return {"searches": {"old": BranchReservedIntegerCapacitySearch, "new": DualReuseIntegerCapacitySearch},
        "prior_selectors": {"old": select_branch_reserved_axis_capacity_frontier_frame,
                            "new": select_dual_reuse_axis_capacity_frontier_frame},
        "selectors": {"old": select_branch_reserved_residual_capacity_frontier_frame,
                      "new": select_dual_reuse_residual_capacity_frontier_frame},
        "new_reference": _new_reference, "oracle": canonical_at_certified_score, "oracle_key": oracle_key}


def run_cell(model, cell, protocol, runtime, dependencies=None, progress=True):
    deps = _dependencies() if dependencies is None else dependencies
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [],
        **dict.fromkeys(("new_reference_calls", "new_reference_oracle_calls", "borrowed_reference_count",
            "borrowed_native_snapshots", "borrowed_prior_axis_count", "new_prior_axis_selector_calls",
            "new_residual_selector_calls", "executed_native_advance_calls", "fresh_native_search_constructions"), 0)}
    budgets = (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
    for index, registered in enumerate(cell["states"]):
        queue, lanes = tuple(registered["queue"]), cell["lanes"]
        previous_residual = None
        if registered["source"] == "fresh_confirmation":
            reference, metrics, complete, independent = deps["new_reference"](
                model, queue, lanes, protocol, runtime, deps["oracle"])
            expected = runtime.reference_key(reference, queue, len(model.options))
            kind = "full_K4096" if expected is not None else None
            if expected is None and independent is not None:
                expected = deps["oracle_key"](independent, len(model.options))
                kind = "independent_score_face_oracle" if expected is not None else None
            reference_fields = {"reference": asdict(reference), "reference_metrics": metrics,
                "reference_complete": complete, "oracle": independent,
                "effective_reference_complete": expected is not None, "effective_reference_source": kind}
            row["new_reference_calls"] += 1
            row["new_reference_oracle_calls"] += int(independent is not None)
            traces = {}
            for label, cls in deps["searches"].items():
                search = cls(model.options, queue, lanes)
                row["fresh_native_search_constructions"] += 1
                traces[label] = {}
                for cap in registered["caps"]:
                    traces[label][str(cap)] = asdict(search.advance_to(cap))
                    row["executed_native_advance_calls"] += 1
            prior_routed = {label: asdict(selector(model.options, queue, lanes, *budgets))
                            for label, selector in deps["prior_selectors"].items()}
            row["new_prior_axis_selector_calls"] += 2
        else:
            saved = borrowed["states"][registered["borrowed_index"]]
            if (registered["borrowed_from"] != "residual_capacity_development"
                    or (saved["state_id"], saved["queue"], saved["source"], saved["caps"]) != (
                        registered["source_state_id"], registered["queue"], registered["source_record_source"], registered["caps"])):
                raise ValueError("borrowed control state, source or native ladder changed")
            reference_fields = {key: saved[key] for key in REFERENCE_FIELDS}
            traces, prior_routed = saved["traces"], saved["prior_routed"]
            previous_residual = {"old": saved["old_routed"], "new": saved["routed"]}
            row["borrowed_reference_count"] += 1
            row["borrowed_native_snapshots"] += 8
            row["borrowed_prior_axis_count"] += 2
        old = deps["selectors"]["old"](model.options, queue, lanes, *budgets)
        row["new_residual_selector_calls"] += 1
        new = deps["selectors"]["new"](model.options, queue, lanes, *budgets)
        row["new_residual_selector_calls"] += 1
        row["states"].append({**registered, **reference_fields, "traces": traces,
            "prior_routed": prior_routed, "previous_residual_routed": previous_residual,
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
        print(json.dumps({"protocol": str(protocol_path), "max_screen_calls": protocol["expected_max_screen_calls"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if list(args.output.parent.glob(args.output.stem + "*.json*")) != [protocol_path]:
        raise FileExistsError("residual trigger confirmation outputs already exist")
    paths = []

    def save_round(record):
        path = args.output.with_name(args.output.stem + f"_round_{record['round_index'] + 1:02d}.json.gz")
        with gzip.open(path, "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "round": record}, stream,
                      separators=(",", ":"), allow_nan=False)
        paths.append(str(path))

    discovery = run_discovery(models, protocol, on_round=save_round)
    discovery_path = args.output.with_name(args.output.stem + "_discovery.json")
    runtime._write_new(discovery_path, {**{key: value for key, value in discovery.items() if key != "rounds"}, "round_paths": paths})
    validation = validation_protocol(protocol, discovery, models)
    validation_path = args.output.with_name(args.output.stem + "_validation_protocol.json")
    runtime._write_new(validation_path, validation)
    print(f"locked {len(discovery['trigger_states'])} fresh triggers and {validation['expected_control_state_count']} controls; beginning independent validation", flush=True)
    rows = []
    for model, cell in zip(models, validation["cells"]):
        row = run_cell(model, cell, validation, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": validation["registered_at_utc"], "cell": row}, stream,
                      separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.residual_capacity_trigger_confirmation_summary import summarize
    summary = summarize(protocol, discovery, validation, rows)
    summary.update(protocol_path=str(protocol_path), discovery_path=str(discovery_path), validation_protocol_path=str(validation_path))
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "screened_queue_count", "trigger_queue_count", "validation_state_count")}))


if __name__ == "__main__":
    main()
