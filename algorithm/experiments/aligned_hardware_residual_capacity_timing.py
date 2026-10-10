"""Matched-quality whole-selector timing on all retained residual-capacity states."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from itertools import permutations
import json
import os
from pathlib import Path
import random
from time import perf_counter, perf_counter_ns, process_time_ns

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_root_pruning_development import load_cell
from algorithm.experiments.aligned_hardware_dual_reuse_development import REFERENCE_FIELDS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_residual_capacity_timing_20260928.json"
DEVELOPMENT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_residual_capacity_frontier_development_20260928.json"
CONFIRMATION_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_residual_capacity_trigger_confirmation_20260928.json"
TAIL_PROTOCOL = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_timing_20260928_protocol.json"
LABELS = ("candidate", "branch", "long")
REPEATS = 6
WARMUPS_PER_LABEL = 2
ORDER_SEED = 2026092871
SOURCES = (
    "algorithm/experiments/aligned_hardware_residual_capacity_timing.py",
    "algorithm/experiments/residual_capacity_timing_summary.py",
)


def timing_orders(node, lanes, count):
    rng = random.Random(f"{ORDER_SEED}:{node}:{lanes}")
    label_orders = []
    for _ in range(count):
        orders = [list(order) for order in permutations(LABELS)]
        rng.shuffle(orders)
        label_orders.append(orders)
    state_orders = []
    for _ in range(REPEATS):
        order = list(range(count))
        rng.shuffle(order)
        state_orders.append(order)
    return {"label_orders": label_orders, "state_orders": state_orders}


def _keys(items):
    return {(item["node"], item["lanes"], tuple(item["queue"])) for item in items}


def build_protocol(models):
    development = json.loads(DEVELOPMENT_OUTPUT.with_name(DEVELOPMENT_OUTPUT.stem + "_protocol.json").read_text())
    confirmation = json.loads(CONFIRMATION_OUTPUT.with_name(CONFIRMATION_OUTPUT.stem + "_validation_protocol.json").read_text())
    dev_result, conf_result = (json.loads(path.read_text()) for path in (DEVELOPMENT_OUTPUT, CONFIRMATION_OUTPUT))
    if (dev_result["decision"] != "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE"
            or not dev_result["development_gate_passed"]
            or conf_result["decision"] != "RESIDUAL_CAPACITY_TRIGGER_CONFIRMATION_ADVANCE"
            or not conf_result["conditional_confirmation_gate_passed"]):
        raise ValueError("complete residual development and new-trigger confirmation are required")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for previous in (development, confirmation):
        if previous["models"] != snapshots:
            raise ValueError("cached catalogue models differ from the successful residual runs")
        if tuple(previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("retain the confirmed residual routing budgets")
        for path, literal in previous["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    historical = _keys(development["historical_regression_keys"])
    tails = json.loads(TAIL_PROTOCOL.read_text())["ordinary_tail_keys"]
    if len(historical) != 34 or len(_keys(tails)) != 16 or not _keys(tails) <= historical:
        raise ValueError("retain all 34 historical regressions and their 16 fixed tails")
    cells = []
    trigger_control_keys, new_trigger_keys = [], []
    for old in development["cells"]:
        cell = {"node": old["node"], "lanes": old["lanes"], "states": []}
        for index, state in enumerate(old["states"]):
            key = old["node"], old["lanes"], tuple(state["queue"])
            group = ("historical_regression" if key in historical else
                     "zero_queue" if state["cohort"] == "zero_queue" else
                     "trigger_control" if state["cohort"] in ("failure_development", "axis_failure_development") else
                     "ordinary_development")
            cell["states"].append({**deepcopy(state), "timing_source": "residual_development",
                "timing_source_index": index, "timing_source_state_id": state["state_id"], "timing_group": group})
            if group == "trigger_control":
                trigger_control_keys.append({"node": old["node"], "lanes": old["lanes"], "queue": state["queue"]})
        cells.append(cell)
    by_cell = {(cell["node"], cell["lanes"]): cell for cell in cells}
    for source in confirmation["cells"]:
        for index, state in enumerate(source["states"]):
            if state["source"] != "fresh_confirmation":
                continue
            cell = by_cell[source["node"], source["lanes"]]
            cell["states"].append({**deepcopy(state), "timing_source": "trigger_confirmation",
                "timing_source_index": index, "timing_source_state_id": state["state_id"], "timing_group": "new_trigger"})
            new_trigger_keys.append({"node": source["node"], "lanes": source["lanes"], "queue": state["queue"]})
    actual_historical = {(cell["node"], cell["lanes"], tuple(state["queue"]))
        for cell in cells for state in cell["states"] if state["timing_group"] == "historical_regression"}
    counts = {group: sum(state["timing_group"] == group for cell in cells for state in cell["states"])
              for group in ("ordinary_development", "historical_regression", "zero_queue", "trigger_control", "new_trigger")}
    if counts != dict(ordinary_development=688, historical_regression=34, zero_queue=16, trigger_control=8, new_trigger=8) or actual_historical != historical:
        raise ValueError("timing must retain the complete 746-state roster and all eight new paired states")
    for cell in cells:
        if len({tuple(state["queue"]) for state in cell["states"]}) != len(cell["states"]):
            raise ValueError("timing must not duplicate borrowed controls or queue states")
        cell["orders"] = timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
    state_count = sum(counts.values())
    timed_calls = state_count * REPEATS * len(LABELS)
    warmup_calls = len(cells) * WARMUPS_PER_LABEL * len(LABELS)
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "residual_capacity_equal_quality_timing_v1",
        "models": snapshots, "cells": cells, "labels": LABELS, "repeats": REPEATS,
        "warmups_per_label": WARMUPS_PER_LABEL, "order_seed": ORDER_SEED, "cpu": min(os.sched_getaffinity(0)),
        **{key: development[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget", "reference_budget", "oracle_state_budget")},
        "long_lp_budget": 4096, "control_integer_state_budget": 100_000,
        "nonregression_limit": 1.05, "trigger_speedup_limit": .95,
        "historical_regression_keys": development["historical_regression_keys"], "ordinary_tail_keys": tails,
        "trigger_control_keys": trigger_control_keys, "new_trigger_keys": new_trigger_keys,
        "expected_state_count": state_count, "expected_group_counts": counts,
        "expected_timed_calls": timed_calls, "expected_warmup_calls": warmup_calls,
        "expected_selector_calls": timed_calls + warmup_calls,
        "reference_sources": {
            "residual_development": {"path": str(DEVELOPMENT_OUTPUT.relative_to(ROOT)), "registered_at_utc": development["registered_at_utc"]},
            "trigger_confirmation": {"path": str(CONFIRMATION_OUTPUT.relative_to(ROOT)), "registered_at_utc": confirmation["registered_at_utc"]}},
        "execution_sources": sources,
        "prior_decisions": {**conf_result["prior_decisions"], "residual_capacity_trigger_confirmation": conf_result["decision"]},
        "comparison": "Candidate is dual-reuse residual-capacity routing; branch is branch-reserved residual-capacity routing, both144/256/shared8192. Long is the frozen strict-score-proof selector with LP4096/shared100000. All three must finish score and canonical proofs and match independently borrowed references. The incomplete axis predecessor is not a timing control.",
        "roster": "All746 residual-development states plus all8 new-trigger paired states, without repeating their8 old controls. Preserve every34 historical key,16 fixed tails,16 zero states and all four observed trigger queues across four catalogues. No new sampling, screening, references or oracle calls.",
        "measurement": "Six whole-selector cold calls per arm/state, all six three-arm permutations once perstate, eachlabel eachposition twice and everydirected adjacentpair twice. Independently shuffle stateorder eachsweep. Two untimed warmups perarm/cell at its firststate. Wall/CPU clocks enclose only the entire selector; finish all three calls before validation/asdict/I/O. Retain every observation and report eachstate's six-repeat median, pooled ratios, p95/p99/max and slow states. No manual GC, warm-start searches, trimming or timing reruns.",
        "runtime_policy": "One host and serial cells, minimum available CPU pinned before numerical imports, BLAS/OMP environment1, automatic GC enabled. Record actual affinity, library pools and native threads; fresh search instances do not imply cold hardware caches.",
        "accounting": "13572 timed and96 warmup calls, separately reported; whole costs include native LP/preprocessing/root work, frontier export, proof DP, retained-vector copies, canonical prefixes/DFS, queries and dual reuse. References/native traces/axis outputs are borrowed and never charged as newly executed work. Finish the complete roster after honest nonclosure.",
        "gate": "Require complete measurements, reference/certificate validity, score/canonical quality for all arms and warmups, both residual arms reproducing frozen complete outputs, all repeated outputs stable, fixed orders and CPU affinity. Against both branch and long, ordinary-development pooled/eachcell/p99<=1.05; historical34 pooled/p99/everypairedstate<=1.05; all16 trigger-paired states pooled/everypairedstate<=1.05. Additionally candidate/long pooled for all16 trigger-paired states<=.95. Zero timings are descriptive while zero quality is mandatory. ADVANCE needs all gates; retain every slow state and earlier HOLD without changing budgets, cohorts, thresholds or orders.",
        "limitations": "Local descriptive timing reuses development and conditional-confirmation states, four cached physical catalogues and virtual R8/16/32/64. Paired repeats are dependent. This is not independent ordinary-seed speed confirmation, physical scaling or flow-time benefit. The long baseline compares complete implementations and does not isolate the residual table's causal contribution. Five-percent thresholds are operational, not significance tests. The paper remains untouched.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, roster, budgets or timing design changed")


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key
    from algorithm.experiments.certificate_routed_development_summary import result_preprocessing
    from algorithm.experiments.strict_score_proof_timing_summary import _same
    if selectors is None:
        from algorithm.residual_capacity_frontier_frame_candidates import (
            select_branch_reserved_residual_capacity_frontier_frame, select_dual_reuse_residual_capacity_frontier_frame)
        from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
        selectors = {"candidate": select_dual_reuse_residual_capacity_frontier_frame,
                     "branch": select_branch_reserved_residual_capacity_frontier_frame,
                     "long": select_strict_score_proof_canonical_completion_frame}
    borrowed = {}
    for name in {state["timing_source"] for state in cell["states"]}:
        source = protocol["reference_sources"][name]
        borrowed[name] = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    started = perf_counter()
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "warmups": [], "state_orders": [],
        "new_reference_calls": 0, "new_oracle_calls": 0, "screen_selector_calls": 0,
        "borrowed_reference_count": len(cell["states"])}
    references, expected_keys = [], []
    for registered in cell["states"]:
        saved = borrowed[registered["timing_source"]]["states"][registered["timing_source_index"]]
        if saved["state_id"] != registered["timing_source_state_id"] or any(
                saved.get(key) != value for key, value in registered.items() if not key.startswith("timing_")):
            raise ValueError("borrowed timing state differs from its frozen source")
        fields = {key: saved[key] for key in REFERENCE_FIELDS}
        reference = selection_from_record(fields["reference"])
        expected = runtime.reference_key(reference, tuple(registered["queue"]), len(model.options))
        if expected is None and fields["oracle"] is not None:
            expected = oracle_key(fields["oracle"], len(model.options))
        references.append(reference)
        expected_keys.append(expected)
        row["states"].append({**registered, **fields, "traces": saved["traces"], "prior_routed": saved["prior_routed"],
            "expected_results": {"candidate": saved["routed"], "branch": saved["old_routed"]},
            "first_results": {}, "repeat_mismatches": {label: [] for label in LABELS},
            "metrics": {label: runtime._metrics() for label in LABELS}, "blocks": [],
            "wall_ns": {label: [None] * REPEATS for label in LABELS}, "cpu_ns": {label: [None] * REPEATS for label in LABELS}})

    def call(label, index):
        budgets = ((protocol["long_lp_budget"], protocol["control_integer_state_budget"]) if label == "long" else
                   (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"]))
        return selectors[label](model.options, tuple(row["states"][index]["queue"]), cell["lanes"], *budgets)

    def record(metric, literal, label, index):
        state = row["states"][index]
        runtime._record(metric, selection_from_record(literal["selection"]), model, tuple(state["queue"]), cell["lanes"],
            literal["selection"]["lp_node_budget"], expected_keys[index],
            result_preprocessing(literal, "long" if label == "long" else "candidate"), references[index])

    row["runtime_before_warmup"] = runtime_provenance()
    for _ in range(WARMUPS_PER_LABEL):
        for label in LABELS:
            literal = asdict(call(label, 0))
            metric = runtime._metrics()
            record(metric, literal, label, 0)
            row["warmups"].append({"label": label, "state_index": 0, "result": literal, "metrics": metric})
    row["runtime_after_warmup"] = runtime_provenance()
    for repeat, order in enumerate(cell["orders"]["state_orders"]):
        row["state_orders"].append([])
        for index in order:
            row["state_orders"][-1].append(index)
            state, outputs = row["states"][index], {}
            labels = cell["orders"]["label_orders"][index][repeat]
            state["blocks"].append({"repeat": repeat, "order": labels})
            for label in labels:
                cpu_start, wall_start = process_time_ns(), perf_counter_ns()
                result = call(label, index)
                state["wall_ns"][label][repeat] = perf_counter_ns() - wall_start
                state["cpu_ns"][label][repeat] = process_time_ns() - cpu_start
                outputs[label] = result
            for label, result in outputs.items():
                literal = asdict(result)
                record(state["metrics"][label], literal, label, index)
                if repeat == 0:
                    state["first_results"][label] = literal
                elif not _same(literal, state["first_results"][label]):
                    state["repeat_mismatches"][label].append({"repeat_index": repeat, "result": literal})
        if progress:
            print(f"{model.node} R{cell['lanes']}: three-arm timing {repeat + 1}/{REPEATS} complete", flush=True)
    row["runtime_end"] = runtime_provenance()
    row["operational_elapsed_s"] = perf_counter() - started
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
        print(json.dumps({"protocol": str(protocol_path), "states": protocol["expected_state_count"], "calls": protocol["expected_selector_calls"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("residual-capacity timing outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.residual_capacity_timing_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
