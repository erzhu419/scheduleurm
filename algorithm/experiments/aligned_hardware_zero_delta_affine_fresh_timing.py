"""Independent four-arm confirmation with every preceding regression state retained."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_fresh_timing_20260928.json"
CONFIRMATION_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_fresh_confirmation_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_timing_20260928.json"
LABELS, REPEATS, WARMUPS_PER_LABEL = ("candidate", "affine", "branch", "long"), 6, 2
ORIGINAL_LABELS = ("candidate", "branch", "long")
ORDER_SEED = 2026092897
CONTROL_COUNTS = dict(ordinary_development=688, historical_regression=34,
                      zero_queue=16, trigger_control=8, new_trigger=8)
SOURCES = ("algorithm/experiments/aligned_hardware_zero_delta_affine_fresh_timing.py",
           "algorithm/experiments/zero_delta_affine_fresh_timing_summary.py")


def timing_orders(node, lanes, count):
    rng = random.Random(f"{ORDER_SEED}:{node}:{lanes}")
    label_orders = []
    for _ in range(count):
        orders = [list(order) for order in permutations(ORIGINAL_LABELS)]
        rng.shuffle(orders)
        positions = list(range(4))
        rng.shuffle(positions)
        positions.extend(rng.sample(range(4), 2))
        rng.shuffle(positions)
        for order, position in zip(orders, positions, strict=True):
            order.insert(position, "affine")
        label_orders.append(orders)
    state_orders = []
    for _ in range(REPEATS):
        order = list(range(count))
        rng.shuffle(order)
        state_orders.append(order)
    return {"label_orders": label_orders, "state_orders": state_orders}


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    old, quality = (json.loads(_protocol_path(path).read_text())
                    for path in (PREVIOUS_OUTPUT, CONFIRMATION_OUTPUT))
    previous, result = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, CONFIRMATION_OUTPUT))
    if previous["decision"] != "ZERO_DELTA_AFFINE_LOCAL_TIMING_ADVANCE" or not all(previous[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "frozen_residual_results_reproduced", "local_timing_gate_passed")):
        raise ValueError("the complete preceding zero-delta timing advance is required")
    if result["decision"] != "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "confirmation_gate_passed",
            "behavior_preserved", "work_gate_passed")):
        raise ValueError("complete independent zero-delta confirmation is required before timing")
    plan = quality["subsequent_timing_plan"]
    if (plan["order_seed"], tuple(plan["labels"]), plan["repeats"], plan["warmups_per_label"],
            plan["expected_timed_calls"], plan["expected_warmup_calls"], plan["nonregression_limit"],
            plan["trigger_speedup_limit"]) != (ORDER_SEED, LABELS, REPEATS, WARMUPS_PER_LABEL,
                                               67248, 128, 1.05, .95):
        raise ValueError("retain the timing plan frozen before independent quality outcomes")
    snapshots, sources = json.loads(json.dumps([model.snapshot() for model in models])), {}
    for source in (old, quality):
        if source["models"] != snapshots:
            raise ValueError("retain the frozen physical catalogue models")
        if tuple(source[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
                "reference_budget", "oracle_state_budget")) != (144, 256, 8192, 4096, 5_000_000):
            raise ValueError("retain the frozen routing and reference budgets")
        for path, literal in source["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    prior_counts = {**CONTROL_COUNTS, "fresh_confirmation": 1024}
    if old["expected_group_counts"] != prior_counts or old["expected_state_count"] != 1778 or tuple(
            old[key] for key in ("labels", "repeats", "warmups_per_label", "nonregression_limit",
                "trigger_speedup_limit", "long_lp_budget", "control_integer_state_budget")) != (
                    list(LABELS), 6, 2, 1.05, .95, 4096, 100_000):
        raise ValueError("retain all1778 preceding states and every four-arm gate")
    if tuple(quality[key] for key in ("expected_state_count", "expected_independent_state_count",
            "expected_borrowed_state_count")) != (2802, 1024, 1778):
        raise ValueError("retain every independent queue and all preceding controls")
    if len(old["cells"]) != 16 or [(cell["node"], cell["lanes"]) for cell in quality["cells"]] != [
            (cell["node"], cell["lanes"]) for cell in old["cells"]]:
        raise ValueError("all 16 frozen cells must occur in order")
    cells, counts = deepcopy(quality["cells"]), {**prior_counts, "independent_fresh": 1024}
    actual = dict.fromkeys(counts, 0)
    for cell, previous_cell in zip(cells, old["cells"], strict=True):
        fresh = [state for state in cell["states"] if state["independent_source"] == "fresh"]
        controls = [state for state in cell["states"] if state["independent_source"] == "borrowed_control"]
        if len(fresh) != 64 or len(controls) != len(previous_cell["states"]) or len(fresh) + len(controls) != len(cell["states"]):
            raise ValueError("each cell must retain all64 independent states and complete prior controls")
        for state, original in zip(controls, previous_cell["states"], strict=True):
            if {key: value for key, value in state.items() if not key.startswith("independent_source")} != original:
                raise ValueError("borrowed controls must preserve all preceding metadata and lineage")
        if len({tuple(state["queue"]) for state in cell["states"]}) != len(cell["states"]):
            raise ValueError("independent and control queue states must remain distinct")
        for index, state in enumerate(cell["states"]):
            if state["independent_source"] == "fresh" and (
                    state["timing_group"] != "independent_fresh" or state["confirmation_source"] != "fresh"):
                raise ValueError("new independent queues must remain a separate endpoint")
            actual[state["timing_group"]] += 1
            state.update(timing_source="zero_delta_fresh_quality", timing_source_index=index,
                         timing_source_state_id=state["state_id"])
        cell["orders"] = timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
    if actual != counts:
        raise ValueError("retain all independent and previous timing cohorts")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    protocol = deepcopy(old)
    protocol.pop("registered_at_utc", None)
    protocol.pop("expected_fresh_state_count", None)
    protocol.pop("expected_control_state_count", None)
    protocol.update(experiment="zero_delta_affine_independent_four_arm_timing_v1", cells=cells,
        cpu=min(os.sched_getaffinity(0)), order_seed=ORDER_SEED, execution_sources=sources,
        expected_state_count=2802, expected_independent_state_count=1024, expected_borrowed_state_count=1778,
        expected_group_counts=counts, expected_timed_calls=67248, expected_warmup_calls=128,
        expected_selector_calls=67376,
        reference_sources={"zero_delta_fresh_quality": {
            "path": str(CONFIRMATION_OUTPUT.relative_to(ROOT)), "registered_at_utc": quality["registered_at_utc"]}},
        prior_decisions={**quality["prior_decisions"], "zero_delta_affine_timing": previous["decision"],
                         "zero_delta_affine_fresh_confirmation": result["decision"]},
        roster="All1024 independently drawn ordinary node-states, paired256 R/Q draws across four fixed catalogues, plus every1778 preceding regression state. Preserve all original six control groups and lineage; the earlier1024 fresh_confirmation states are now ordinary regression controls. New independent_fresh outcomes are a separate endpoint.",
        measurement="Six adjacent four-arm cold blocks perstate under frozen seed2026092897. Removing affine leaves the six original candidate/branch/long permutations once each. Affine insertion counts are2,2,1,1 across four positions; no all24 or exact four-position balance claim. Independently shuffle state sweeps; two untimed warmups perarm/cell. Entire-selector clocks include all work, ending before serialization or validation. Retain every observation and every failure without trimming or reruns.",
        accounting="67248 timed plus128 warmup calls. Borrow all2802 independent reference records and expected real candidate/affine/branch outputs and twelve native snapshots from the completed quality phase. Only the original754 controls have two genuine axis predecessors; new independent states and prior ordinary controls have no fabricated prior. All actual zero guards, flags, affine preparation/updates, LP, DP, export, retained copies, proof/prefix/canonical states and queries are charged by arm. Zero new references, oracles or screening in timing.",
        gate="Every state and warmup must pass reference, certificate, canonical/action, path/fee, repeat, frozen-output, order and affinity requirements. Separately on1024 new independent queues, candidate/affine, candidate/branch and candidate/long pooled, each cell and pooled p99 state-median ratios must be<=1.05. Reapply all original1778 primary and affine gates within their unchanged six groups, including trigger/long<=.95. No new ordinary every-state gate. Zero timing remains descriptive while zero quality is mandatory. ADVANCE requires every independent and retained-control gate.",
        limitations="Only the1024 new states, representing256 distinct paired R/Q draws, are independent ordinary-queue confirmation under the fixed law. All1778 earlier states are reused controls. Six four-arm repeats and catalogue copies are dependent; catalogues and virtual identical lanes remain fixed. Retain earlier HOLDs and observed mean/tail counterevidence. No new physical scaling, flow-time or significance claim; paper unchanged.")
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, roster, budgets or independent four-arm timing design changed")


def _selectors():
    from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import select_zero_delta_affine_dual_residual_capacity_frontier_frame
    from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
    from algorithm.residual_capacity_frontier_frame_candidates import select_branch_reserved_residual_capacity_frontier_frame
    from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
    return {"candidate": select_zero_delta_affine_dual_residual_capacity_frontier_frame,
            "affine": select_affine_dual_residual_capacity_frontier_frame,
            "branch": select_branch_reserved_residual_capacity_frontier_frame,
            "long": select_strict_score_proof_canonical_completion_frame}



def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key
    from algorithm.experiments.certificate_routed_development_summary import result_preprocessing
    from algorithm.experiments.strict_score_proof_timing_summary import _same
    if selectors is None:
        selectors = _selectors()
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
        row["states"].append({**registered, **fields, "traces": saved["traces"],
            **({"prior_routed": saved["prior_routed"]} if saved["confirmation_source"] == "borrowed_control" else {}),
            "previous_dual_traces": saved["previous_dual_traces"],
            "previous_dual_routed": saved["previous_dual_routed"],
            "expected_results": {"candidate": saved["routed"], "affine": saved["previous_dual_routed"],
                                 "branch": saved["old_routed"]},
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
            print(f"{model.node} R{cell['lanes']}: four-arm timing {repeat + 1}/{REPEATS} complete", flush=True)
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
        raise FileExistsError("independent zero-delta timing outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.zero_delta_affine_fresh_timing_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
