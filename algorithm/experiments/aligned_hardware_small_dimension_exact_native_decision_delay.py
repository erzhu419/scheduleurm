"""Main-scenario development replay with each whole native solve charged once."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import (
    runtime_provenance, runtime_setup,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_decision_delay_development_20260929.json"
QUALITY_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_20260929.json"
TIMING_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_timing_20260929.json"
TAPE_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_decision_delay_development_20260927.json"
LABELS = ("candidate", "direct")
ZERO_LABELS = ("full_reference", *LABELS)
LANES, LOADS, ARRIVAL_SEEDS = (8,), (.80, .95), (41, 42, 43)
ARRIVAL_HORIZON, OBSERVATION_END = 600.0, 1200.0
REPEATS, ORDER_SEED = 6, 2026092916
SOURCES = (
    "algorithm/experiments/aligned_hardware_small_dimension_exact_native_decision_delay.py",
    "algorithm/experiments/small_dimension_exact_native_decision_delay_audit.py",
    "algorithm/experiments/small_dimension_exact_native_decision_delay_summary.py",
    "algorithm/experiments/integer_capacity_decision_delay_replay.py",
    "algorithm/experiments/integer_capacity_trajectory_replay.py",
    "algorithm/experiments/integer_capacity_trajectory_summary.py",
    "algorithm/experiments/aligned_hardware_integer_capacity_timing.py",
)


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def timing_orders(tape_id):
    orders = [list(LABELS), list(reversed(LABELS))] * (REPEATS // 2)
    random.Random(f"{ORDER_SEED}:{tape_id}").shuffle(orders)
    return orders


def build_protocol(models):
    quality, timing = (json.loads(path.read_text()) for path in (QUALITY_OUTPUT, TIMING_OUTPUT))
    required = (
        (quality, "SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_ADVANCE", (
            "execution_checks_passed", "references_complete", "quality_gate_passed",
            "work_gate_passed", "confirmation_gate_passed")),
        (timing, "SMALL_DIMENSION_EXACT_NATIVE_INDEPENDENT_TIMING_ADVANCE", (
            "execution_checks_passed", "certificate_checks_passed", "references_complete",
            "quality_gate_passed", "repeat_paths_stable", "cpu_affinity_preserved",
            "local_timing_gate_passed", "new_independent_timing_gate_passed",
            "independent_gain_gate_passed")),
    )
    for result, decision, flags in required:
        if result["decision"] != decision or not all(result[key] for key in flags):
            raise ValueError("complete independent native quality and timing advances are required")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    previous = json.loads(_protocol_path(QUALITY_OUTPUT).read_text())
    old = json.loads(_protocol_path(TAPE_OUTPUT).read_text())
    if snapshots != previous["models"] or snapshots != old["models"]:
        raise ValueError("retain the four original cached complete-frame catalogues")
    if (len(models), tuple(old["lanes"]), tuple(old["loads"]), tuple(old["arrival_seeds"]),
            old["arrival_horizon"], old["observation_end"]) != (
                4, LANES, LOADS, ARRIVAL_SEEDS, ARRIVAL_HORIZON, OBSERVATION_END):
        raise ValueError("reuse the original four-catalogue R8 development arrival design")
    if tuple(previous[key] for key in ("initial_lp_budget", "lp_node_budget",
            "integer_state_budget", "reference_budget", "oracle_state_budget")) != (
                144, 256, 8192, 4096, 5_000_000):
        raise ValueError("retain the independently confirmed whole-controller budgets")
    sources = {}
    for registered in (previous, old):
        for path, literal in registered["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen execution source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    tapes = old["tapes"]
    expected = {(model.node, lanes, load, seed)
                for model in models for lanes in LANES for load in LOADS for seed in ARRIVAL_SEEDS}
    actual = [(tape["node"], tape["lanes"], tape["load"], tape["seed"]) for tape in tapes]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("every original node/R/load/seed tape must appear once")
    prior = json.loads(TAPE_OUTPUT.read_text())
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "small_dimension_exact_native_actual_decision_delay_development_v1",
        "models": snapshots, "lanes": LANES, "loads": LOADS, "arrival_seeds": ARRIVAL_SEEDS,
        "arrival_horizon": ARRIVAL_HORIZON, "observation_end": OBSERVATION_END,
        "labels": LABELS, "zero_labels": ZERO_LABELS, "repeats": REPEATS,
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget",
            "integer_state_budget", "reference_budget", "oracle_state_budget")},
        "budgets": {str(lanes): previous["lp_node_budget"] for lanes in LANES},
        "capacities": old["capacities"], "tapes": tapes,
        "tape_source": str(_protocol_path(TAPE_OUTPUT).relative_to(ROOT)),
        "timing_orders": {tape["tape_id"]: timing_orders(tape["tape_id"]) for tape in tapes},
        "order_seed": ORDER_SEED, "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "cpu": min(os.sched_getaffinity(0)), "expected_tape_count": 24,
        "expected_zero_trajectory_count": 72, "expected_measured_trajectory_count": 288,
        "expected_trajectory_count": 360, "expected_group_count": 16,
        "float_relative_tolerance": 1e-9, "minimum_pooled_flow_improvement": .01,
        "prior_decisions": {**timing["prior_decisions"],
            "small_dimension_exact_native_confirmation": quality["decision"],
            "small_dimension_exact_native_confirmation_timing": timing["decision"],
            "integer_capacity_decision_delay": prior["decision"]},
        "execution_sources": sources,
        "design": "Development integration on all24 original tapes: four cached catalogues, virtualR8, loads0.80/0.95, previously used seeds41/42/43, H600/T1200. Each tape executes fullK4096 and candidate/direct zero-delay trajectories, then six paired measured rounds with both whole-call orders exactly three times;360 trajectories. Repeats and catalogue copies are paired, not new independent arrivals.",
        "calls": "One serial fresh process pinned to the registered CPU before numerical imports, BLAS thread environment1 and GC enabled. Every decision invokes a genuinely cold complete candidate/direct controller at144/256/8192, including repeated and idle Q. Three actual zero-delay policies warm libraries before measurements. No policy-result cache, resampling, outlier deletion, escalation, rescue or rerun.",
        "clock": "perf_counter_ns encloses the whole selector including bound-argument invocation, native construction, exact/generic LP, snapshots, root phase and the unique integer probe. Pay observed wall seconds once at scale1 before dispatch. Only solve-start released FIFO Q enters the decision; later arrivals wait. Paid decisions at/afterT cancel dispatch and remain recorded; service completion offsets crossingT remain unfinished at the cutoff. Validation, bookkeeping and reference work are outside timers and simulated time.",
        "reference_audit": "Only after all six measured rounds, audit zero and measured decisions using the actual original-catalogue K4096 reference. Actual zero full-reference calls may populate that offline reference cache; only reference's own score-exact/canonical-open result can invoke the independent5M score-face oracle. Preserve incomplete references, uncensored selections and cancelled decisions; cache references only within each tape. Reference/audit/oracle costs are separate from whole policy fees.",
        "accounting": "Retain actual controller144/256/8192 limits and actual raw/settled/initial/final LP snapshot caps without relabeling. Audit each actual whole prefix and certificate; reuse full native, exact-LP, root-phase, dual-face and shared probe fee accounting. No extra independent native ladders are run or claimed. Retain each measured call's untrimmed delay and every job outcome.",
        "cohort": "At each H/T cutoff share one identical completed job-ID intersection across all15 outputs on the tape; keep restricted mean over every released job and all unfinished tails. Average six measured repeats within each arrival seed before averaging seeds; sum seed-level mean unfinished counts. Completed-only means are descriptive.",
        "integration_gate": "Complete coverage, invariant and clock/job/budget accounting, CPU/thread provenance, valid whole certificates and complete references for every decision including idle/cancelled. Candidate/direct canonical exact and canonical-reference matching on their own actual Q. Only zero-delay trajectories must have identical frames/actions/jobs to fullK4096; measured paths may diverge when their clocks cross different arrival events.",
        "utility_gate": "Across all16 node/load/H,T groups, measured candidate must not increase unfinished counts, all-job restricted mean, common-cohort mean or common-cohort p90 against direct. Empty common cohorts fail. Average repeats within seed first; compare with1e-9 relative float tolerance only. Speed alone cannot pass this gate.",
        "flow_gain_gate": "Integration and every utility gate must pass, and pooled per-tape common-cohort mean atT must be at least1% lower than direct. Zero-delay and per-trajectory delay increments remain diagnostics; historical utility/timing HOLDs remain unchanged.",
        "limitations": "First main-scenario development integration on already used arrival tapes, deterministic cached service point models and virtual lanes. Selector latency measured on one local CPU is not physical node latency. Only selector computation is charged, without network, launch or dispatcher costs. No independent flow generalization, physical scaling or statistical-significance claim.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, budgets, tapes, orders or decision-clock design changed")


def _dependencies():
    from algorithm.experiments.integer_capacity_decision_delay_replay import simulate_delayed_trajectory
    from algorithm.experiments.small_dimension_exact_native_decision_delay_audit import (
        ReplayReferenceAudit, audit_trajectory, selectors,
    )
    return {"simulate": simulate_delayed_trajectory, "audit": audit_trajectory,
            "reference": ReplayReferenceAudit, "selectors": selectors, "provenance": runtime_provenance}


def run_cell(model, lanes, protocol, dependencies=None, progress=True):
    dependencies = _dependencies() if dependencies is None else dependencies
    native_model, started = replace(model, lanes=lanes), perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "runtime_start": dependencies["provenance"](), "runs": []}
    selected_by_label = dependencies["selectors"](protocol)
    for registered in protocol["tapes"]:
        if (registered["node"], registered["lanes"]) != (model.node, lanes):
            continue
        run = {key: registered[key] for key in ("tape_id", "load", "seed")}
        run.update(zero={}, measured=[])
        pending = []
        for label in protocol["zero_labels"]:
            budget = protocol["reference_budget"] if label == "full_reference" else protocol["lp_node_budget"]
            result, original = dependencies["simulate"](
                native_model, registered["arrivals"], protocol["arrival_horizon"],
                protocol["observation_end"], budget, selected_by_label[label],
                policy=label, clock=lambda: 0)
            run["zero"][label] = result
            pending.append((result, original))
        for repeat, order in enumerate(protocol["timing_orders"][registered["tape_id"]]):
            block = {"repeat": repeat, "order": order, "policies": {}}
            for label in order:
                result, original = dependencies["simulate"](
                    native_model, registered["arrivals"], protocol["arrival_horizon"],
                    protocol["observation_end"], protocol["lp_node_budget"],
                    selected_by_label[label], policy=label)
                block["policies"][label] = result
                pending.append((result, original))
            run["measured"].append(block)
        reference = dependencies["reference"](native_model, protocol)
        for result, original in pending:
            dependencies["audit"](native_model, result, original, reference, protocol)
        run["reference_audit"] = reference.summary()
        row["runs"].append(run)
        if progress:
            print(f"{model.node} R{lanes}: load={run['load']} seed={run['seed']} 3 zero + 6 paired measured rounds audited", flush=True)
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
            model = next(model for model in models if model.node == node)
            row = run_cell(model, lanes, protocol)
            with gzip.open(path, "xt") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.small_dimension_exact_native_decision_delay_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "decision", "integration_gate_passed", "utility_gate_passed", "flow_gain_established")}), flush=True)


if __name__ == "__main__":
    main()
