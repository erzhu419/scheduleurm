"""Serial development replay with each actual selector latency charged once."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
from itertools import permutations
import json
import os
from pathlib import Path
import random
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_decision_delay_development_20260927.json"
TRAJECTORY_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_trajectory_20260927.json"
TRAJECTORY_PROTOCOL = TRAJECTORY_OUTPUT.with_name(TRAJECTORY_OUTPUT.stem + "_protocol.json")
LABELS = ("old_shared", "fractional_shared", "integer_capacity")
ZERO_LABELS = ("full_reference", *LABELS)
LANES = (8,)
LOADS = (.80, .95)
ARRIVAL_SEEDS = (41, 42, 43)
ARRIVAL_HORIZON = 600.0
OBSERVATION_END = 1200.0
REPEATS = 6
ORDER_SEED = 2026092715
EXTRA_SOURCES = (
    "algorithm/experiments/integer_capacity_decision_delay_replay.py",
    "algorithm/experiments/integer_capacity_decision_delay_summary.py",
    "algorithm/experiments/aligned_hardware_integer_capacity_decision_delay.py",
)


def require_trajectory_integration():
    previous = json.loads(TRAJECTORY_OUTPUT.read_text(encoding="utf-8"))
    if not previous["integration_gate_passed"]:
        raise ValueError("prior zero-delay trajectory integration must pass")
    return {key: previous[key] for key in ("decision", "integration_gate_passed", "utility_gate_passed", "flow_gain_established")}


def build_protocol(models):
    from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
    prior_result = require_trajectory_integration()
    previous = json.loads(TRAJECTORY_PROTOCOL.read_text(encoding="utf-8"))
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen trajectory source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen trajectory catalogues changed")
    capacities = [entry for entry in previous["capacities"]
                  if entry["lanes"] in LANES and entry["load"] in LOADS]
    tapes = []
    for model in models:
        for lanes in LANES:
            for load in LOADS:
                capacity = next(entry["certificate"] for entry in capacities
                                if (entry["node"], entry["lanes"], entry["load"]) == (model.node, lanes, load))
                for seed in ARRIVAL_SEEDS:
                    tapes.append({"tape_id": f"{model.node}:R{lanes}:load{load:g}:seed{seed}",
                                  "node": model.node, "lanes": lanes, "load": load, "seed": seed,
                                  "arrivals": poisson_arrivals(capacity["arrival_rates"], ARRIVAL_HORIZON, seed)})
    orders, rng = {}, random.Random(ORDER_SEED)
    for tape in tapes:
        sequence = list(permutations(LABELS))
        rng.shuffle(sequence)
        orders[tape["tape_id"]] = sequence
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "integer_capacity_actual_decision_delay_development_v1",
        "models": previous["models"], "lanes": LANES, "loads": LOADS, "arrival_seeds": ARRIVAL_SEEDS,
        "arrival_horizon": ARRIVAL_HORIZON, "observation_end": OBSERVATION_END,
        "labels": LABELS, "zero_labels": ZERO_LABELS, "repeats": REPEATS,
        "budgets": {str(lanes): 36 * lanes for lanes in LANES}, "reference_budget": 4096,
        "capacities": capacities, "tapes": tapes, "selector_permutations": list(permutations(LABELS)),
        "order_seed": ORDER_SEED, "timing_orders": orders,
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "cpu": min(os.sched_getaffinity(0)), "prior_result": prior_result,
        "prior_trajectory": str(TRAJECTORY_OUTPUT.relative_to(ROOT)),
        "design": "Development seeds 41/42/43, all four catalogues at the middle virtual parallelism R8, both previous loads. Per tape: four zero-delay policies once, then six measured blocks covering every permutation of the three deployable selectors. 24 tapes, 96 zero and 432 measured trajectories. Repeats are within-tape timing repeats, not independent seeds.",
        "calls": "All cells and trajectories serial in one fresh CPU-pinned process, with BLAS thread environment fixed before numerical imports. Each policy invokes its actual cold selector at every decision; zero-delay calls first also warm numerical libraries. No candidate caching, timing resampling, warm starts, escalation, or rescue. Retain every observed wall duration including outliers.",
        "clock": "Freeze released FIFO queue at decision start. perf_counter_ns wraps only the complete selector call including preprocessing and returned object construction. Apply that observed wall duration once at scale 1 before dispatch. Arrivals during the solve wait for the next decision. Every idle decision pays its call and then the original 1-second idle frame. Dispatch at or after T is cancelled but its paid decision and queued jobs remain recorded; decisions and executed frames counted separately.",
        "reference_audit": "Share only offline full-K4096 reference results within a tape. Zero-delay policies use the established audited engine; after ALL six measured blocks, audit every actual solve-start Q including any cancelled final decision. Reference work, validation, recording and simulator bookkeeping are outside selector timers and virtual time.",
        "cohort": "At each H/T cutoff use one identical completed job-ID intersection across all 22 outputs on the tape. Also retain all-job restricted mean and unfinished tails. Average measured repeats within each seed before averaging seeds; sum seed-level mean unfinished counts. Retain per-repeat diagnostics.",
        "integration_gate": "All registered coverage and job/clock/accounting invariants, valid certificates and complete references; measured and zero-delay integer selections canonical exact and matching audited optimum. Full trajectory identity is required only for zero-delay integer versus zero-delay full K4096.",
        "utility_gate": "Measured integer must not increase unfinished counts, all-job restricted mean, common-cohort mean or common-cohort p90 against BOTH measured controls in ANY node/load/cutoff group. Empty common cohorts cannot pass. All metrics aggregate repeats within seed before seeds. Relative float tolerance 1e-9 only; zero-delay comparisons and delay-minus-zero increments remain diagnostics.",
        "flow_gain_gate": "Integration and every measured utility gate pass, and pooled per-tape common-cohort mean at T is at least 1% lower than BOTH controls. Faster solving alone is not flow-time gain.",
        "float_relative_tolerance": 1e-9, "minimum_pooled_flow_improvement": .01,
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Development screen, not independent confirmation. Cached deterministic physical catalogues with virtual R8 lanes and synthetic arrivals; observed selector latency comes from one local execution host, not those physical nodes. Only selector computation is charged; dispatcher/network/process-start costs are absent. Three arrival seeds and six within-tape repeats do not establish physical end-to-end superiority. Prior zero-delay utility HOLD remains unchanged.",
    }))


def verify_protocol(protocol, models):
    expected = build_protocol(models)
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources, catalogues, tapes, orders or delay design changed")


def decision_delay_cell(model, lanes, protocol, runtime, progress=True):
    from algorithm.experiments.integer_capacity_trajectory_replay import ReferenceAudit, simulate_trajectory
    from algorithm.experiments.integer_capacity_decision_delay_replay import simulate_delayed_trajectory, audit_delayed_trajectory
    started = perf_counter()
    model_at_lanes = replace(model, lanes=lanes)
    selectors = dict(zip(ZERO_LABELS, (
        runtime.select_canonical_count_space_bnb_frame,
        runtime.select_shared_threshold_canonical_requirement_dominance_frame,
        runtime.select_fractional_shared_threshold_canonical_requirement_dominance_frame,
        runtime.select_integer_capacity_shared_threshold_frame), strict=True))
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "option_count": len(model.options), "runs": [], "runtime_start": runtime_provenance()}
    for registered in protocol["tapes"]:
        if (registered["node"], registered["lanes"]) != (model.node, lanes):
            continue
        tape = tuple(tuple(event) for event in registered["arrivals"])
        audit = ReferenceAudit(model_at_lanes, protocol["reference_budget"])
        run = {key: registered[key] for key in ("tape_id", "load", "seed")}
        run.update(arrival_count=len(tape), zero={}, measured=[])
        for label in protocol["zero_labels"]:
            budget = protocol["reference_budget"] if label == "full_reference" else protocol["budgets"][str(lanes)]
            run["zero"][label] = simulate_trajectory(
                model_at_lanes, tape, protocol["arrival_horizon"], protocol["observation_end"],
                budget, selectors[label], reference_cache=audit, policy=label)
        pending = []
        for repeat, order in enumerate(protocol["timing_orders"][registered["tape_id"]]):
            block = {"repeat": repeat, "order": order, "policies": {}}
            for label in order:
                result, selections = simulate_delayed_trajectory(
                    model_at_lanes, tape, protocol["arrival_horizon"], protocol["observation_end"],
                    protocol["budgets"][str(lanes)], selectors[label], policy=label)
                block["policies"][label] = result
                pending.append((result, selections))
            run["measured"].append(block)
        for result, selections in pending:
            audit_delayed_trajectory(model_at_lanes, result, selections, audit)
        run["reference_audit"] = audit.summary()
        row["runs"].append(run)
        if progress:
            print(f"{model.node} R{lanes}: load={run['load']} seed={run['seed']} zero + 6 measured blocks audited", flush=True)
    row["runtime_end"] = runtime_provenance()
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
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, protocol)
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
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
            with gzip.open(path, "rt", encoding="utf-8") as source:
                saved = json.load(source)
            if saved["registered_at_utc"] != protocol["registered_at_utc"]:
                raise ValueError("cell registration differs from protocol")
            row = saved["cell"]
        else:
            model = next(model for model in models if model.node == node)
            row = decision_delay_cell(model, lanes, protocol, runtime)
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.integer_capacity_decision_delay_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "integration_gate_passed", "utility_gate_passed", "flow_gain_established")}))


if __name__ == "__main__":
    main()
