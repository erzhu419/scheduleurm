"""Common computation-window replay with paired actual native solve costs."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_provenance, runtime_setup


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_common_cadence_development_20260929.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_decision_delay_development_20260929.json"
LABELS = ("candidate", "direct")
COMPUTATION_WINDOW_S = 1.0
SOURCES = (
    "algorithm/experiments/aligned_hardware_small_dimension_exact_native_common_cadence.py",
    "algorithm/experiments/small_dimension_exact_native_common_cadence_replay.py",
    "algorithm/experiments/small_dimension_exact_native_common_cadence_summary.py",
)


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    result = json.loads(PREVIOUS_OUTPUT.read_text())
    if not all(result[key] for key in ("execution_checks_passed", "runtime_checks_passed",
            "certificate_checks_passed", "references_complete", "canonical_action_gate_passed",
            "zero_trajectory_identity_passed", "integration_gate_passed")):
        raise ValueError("the complete preceding native replay integration must pass")
    previous = json.loads(_protocol_path(PREVIOUS_OUTPUT).read_text())
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("retain the preceding cached complete-frame models")
    sources = {}
    for path, literal in previous["execution_sources"].items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen preceding source changed: {path}")
        sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    frozen = {key: previous[key] for key in ("models", "lanes", "loads", "arrival_seeds",
        "arrival_horizon", "observation_end", "initial_lp_budget", "lp_node_budget",
        "integer_state_budget", "reference_budget", "oracle_state_budget", "budgets",
        "capacities", "tapes", "timing_orders", "order_seed", "execution_order",
        "float_relative_tolerance")}
    if tuple(previous["labels"]) != LABELS or (len(models), len(frozen["tapes"]), previous["repeats"]) != (4, 24, 6):
        raise ValueError("retain all24 original tapes, both native controllers and six balanced rounds")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "small_dimension_exact_native_common_computation_window_development_v1",
        **frozen, "labels": LABELS, "repeats": 6, "computation_window_s": COMPUTATION_WINDOW_S,
        "maximum_pooled_wall_ratio": .95, "maximum_group_wall_ratio": 1.05,
        "cpu": min(os.sched_getaffinity(0)), "expected_tape_count": 24,
        "expected_reference_trajectory_count": 24, "expected_measured_trajectory_count": 288,
        "expected_trajectory_count": 312, "expected_cost_group_count": 8, "expected_speed_gate_count": 9,
        "previous_protocol_path": str(_protocol_path(PREVIOUS_OUTPUT).relative_to(ROOT)),
        "previous_result_path": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
        "prior_decisions": {**result["prior_decisions"], "small_dimension_exact_native_decision_delay": result["decision"]},
        "execution_sources": sources,
        "design": "Use all24 previously frozen R8/cache/load/seed arrival tapes and service models, controller144/256/8192 limits and original six AB/BA orders (each three times). One fixed-window fullK4096 normative trajectory and twelve real measured native trajectories per tape,312 total. No new seeds, scenario replacement, warmups, samples, resampling or repeat selection.",
        "window_origin": "Fix1.0 second before outcomes, directly from both original controllers' default idle_duration=1.0. This is a new development computation-window condition, not an existing physical scheduler deadline. Every frame receives this window; an idle frame therefore has a2-second cycle. Do not scan window sizes to obtain a favorable result.",
        "clock": "Freeze solve-start released FIFO Q, execute the genuine cold whole selector and measure perf_counter_ns wall plus process_time_ns CPU. Dispatch at start+max(1s,observed_wall); wait padding=max(1s-wall,0), overrun=max(wall-1s,0). Window arrivals cannot enter the current frame. Preserve paid terminal decisions and scheduled service completions beyondT. No real sleep, truncation, synchronized repair or catch-up after an overrun; any overrun makes the final identity/cost adoption decision HOLD after the whole batch.",
        "reference": "Before each tape's twelve measurements run one actual cold fullK4096 normative trajectory with dispatch=start+1s. Its actual computation cost is recorded separately, but does not constrain this offline normative clock or enter the deployable-controller deadline gate. After all six measured rounds audit every original result using the preceding actual-prefix native auditors; share only offline reference results within the tape, beginning with the actual normative reference calls. Only a reference's own score-exact/canonical-open result may invoke the unchanged5M score-face oracle; retain incomplete references and never rescue a policy.",
        "measurement": "Single fresh serial process, registered CPU pinned before numerical imports, BLAS1 and GC enabled. Timers surround each complete cold selector call, including its actual native/LP/snapshot/root/probe work. All bookkeeping, audit and I/O are outside those timers and virtual costs. Record wall,CPU,padding andoverrun separately; a fixed1s window is never called measured execution time. Retain every observation and all full whole-fee families.",
        "integration_gate": "Require all registered cells/tapes/orders/calls, causal FIFO/resource/job/clock accounting, runtime conditions, valid certificates and complete canonical references on each actual Q. Every candidate/direct decision canonical exact and reference matching. Any actual solve longer than1s fails the no-overrun gate while its real trajectory remains intact. Compare all twelve measured outputs with the same fixed-window normative trajectory: solve-start,dispatch,Q,action,service,duration,execution/cancellation status,all job start/completion and H/T snapshots exactly equal. No identity to earlier zero-delay/free-clock paths is requested.",
        "cost_gate": "Average total actual wall and CPU over six repeats within each tape before summing across tapes. Freeze nine wall gates: pooled candidate/direct<=.95 and all eight node/load groups<=1.05. CPU and P99 whole-call latency are descriptive. Empty/zero-cost groups cannot pass. A cost advance requires integration, zero overruns, all trajectory identities and all nine speed gates; no flow-improvement threshold.",
        "cohort": "At H/T retain all released-job restricted means and censored tails, plus one shared completed-ID intersection across all13 outputs per tape. Flow outcomes must coincide with the new fixed-window normative trajectory when identity passes; their equality is not a flow-time improvement.",
        "limitations": "Development diagnosis on reused arrivals, four deterministic cached point-model catalogues and virtualR8.1s is an added computation window, not a demonstrated deployment cadence. Six timing repeats and shared arrival seeds/catalogues are paired. Normative full-reference costs do not establish its deployability. Preserve every historical HOLD, especially free-clock end-to-end utility HOLD. No independent flow-time, physical cluster, energy or statistical-significance claim.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, tapes, budgets, computation window or cost gates changed")


def _dependencies():
    from algorithm.experiments.small_dimension_exact_native_decision_delay_audit import ReplayReferenceAudit, selectors
    from algorithm.experiments.small_dimension_exact_native_common_cadence_replay import (
        simulate_common_cadence_trajectory, audit_common_cadence_trajectory,
    )
    return {"simulate": simulate_common_cadence_trajectory, "audit": audit_common_cadence_trajectory,
            "reference": ReplayReferenceAudit, "selectors": selectors, "provenance": runtime_provenance}


def run_cell(model, lanes, protocol, dependencies=None, progress=True):
    dependencies = _dependencies() if dependencies is None else dependencies
    native_model, started = replace(model, lanes=lanes), perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "runtime_start": dependencies["provenance"](), "runs": []}
    selectors = dependencies["selectors"](protocol)
    for tape in protocol["tapes"]:
        if (tape["node"], tape["lanes"]) != (model.node, lanes):
            continue
        run = {key: tape[key] for key in ("tape_id", "load", "seed")}
        run["measured"] = []
        baseline, original = dependencies["simulate"](
            native_model, tape["arrivals"], protocol["arrival_horizon"], protocol["observation_end"],
            protocol["reference_budget"], selectors["full_reference"],
            protocol["computation_window_s"], policy="full_reference")
        run["reference"] = baseline
        pending = [(baseline, original)]
        for repeat, order in enumerate(protocol["timing_orders"][tape["tape_id"]]):
            block = {"repeat": repeat, "order": order, "policies": {}}
            for label in order:
                result, original = dependencies["simulate"](
                    native_model, tape["arrivals"], protocol["arrival_horizon"], protocol["observation_end"],
                    protocol["lp_node_budget"], selectors[label], protocol["computation_window_s"], policy=label)
                block["policies"][label] = result
                pending.append((result, original))
            run["measured"].append(block)
        reference = dependencies["reference"](native_model, protocol)
        for result, original in pending:
            dependencies["audit"](native_model, result, original, reference, protocol)
        run["reference_audit"] = reference.summary()
        row["runs"].append(run)
        if progress:
            print(f"{model.node} R{lanes}: load={run['load']} seed={run['seed']} fixed reference + 6 paired rounds audited", flush=True)
    row["runtime_end"] = dependencies["provenance"]()
    row["operational_elapsed_s"] = perf_counter() - started
    return row


def _cell_path(output, node, lanes):
    return output.with_name(f"{output.stem}_cell_{node}_r{lanes}.json.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--merge-cells", action="store_true")
    args = parser.parse_args()
    runtime = runtime_setup()
    models = runtime.load_catalogues()
    protocol_path = _protocol_path(args.output)
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, protocol)
        print(protocol_path, flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.run:
        for node, lanes in protocol["execution_order"]:
            if _cell_path(args.output, node, lanes).exists():
                raise FileExistsError(_cell_path(args.output, node, lanes))
    rows = []
    for node, lanes in protocol["execution_order"]:
        path = _cell_path(args.output, node, lanes)
        if args.merge_cells:
            with gzip.open(path, "rt") as source:
                saved = json.load(source)
            if saved["registered_at_utc"] != protocol["registered_at_utc"]:
                raise ValueError("cell registration differs from protocol")
            row = saved["cell"]
        else:
            row = run_cell(next(model for model in models if model.node == node), lanes, protocol)
            with gzip.open(path, "xt") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.small_dimension_exact_native_common_cadence_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "decision", "integration_gate_passed", "no_overrun_gate_passed", "trajectory_identity_passed", "timing_gate_passed")}), flush=True)


if __name__ == "__main__":
    main()
