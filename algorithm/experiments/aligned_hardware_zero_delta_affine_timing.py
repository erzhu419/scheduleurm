"""Four-arm timing of exact zero-coefficient affine dual updates."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_timing_20260928.json"
DEVELOPMENT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_zero_delta_affine_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_fresh_timing_20260928.json"
LABELS, REPEATS, WARMUPS_PER_LABEL = ("candidate", "affine", "branch", "long"), 6, 2
ORIGINAL_LABELS = ("candidate", "branch", "long")
ORDER_SEED = 2026092892
CONTROL_COUNTS = dict(ordinary_development=688, historical_regression=34,
                      zero_queue=16, trigger_control=8, new_trigger=8)
SOURCES = ("algorithm/experiments/aligned_hardware_zero_delta_affine_timing.py",
           "algorithm/experiments/zero_delta_affine_timing_summary.py")


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
    old, development = (json.loads(_protocol_path(path).read_text())
                        for path in (PREVIOUS_OUTPUT, DEVELOPMENT_OUTPUT))
    previous, result = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, DEVELOPMENT_OUTPUT))
    if previous["decision"] != "AFFINE_DUAL_FRESH_TIMING_HOLD" or not all(previous[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "frozen_residual_results_reproduced")):
        raise ValueError("retain the complete quality-valid prior affine timing HOLD")
    if result["decision"] != "ZERO_DELTA_AFFINE_DEVELOPMENT_ADVANCE" or not all(result[key] for key in (
            "development_gate_passed", "execution_checks_passed", "references_complete",
            "quality_gate_passed", "behavior_preserved", "work_gate_passed")):
        raise ValueError("complete zero-delta affine development is required before timing")
    snapshots, sources = json.loads(json.dumps([model.snapshot() for model in models])), {}
    for source in (old, development):
        if source["models"] != snapshots:
            raise ValueError("retain the frozen physical catalogue models")
        if tuple(source[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
                "reference_budget", "oracle_state_budget")) != (144, 256, 8192, 4096, 5_000_000):
            raise ValueError("retain the frozen routing and reference budgets")
        for path, literal in source["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    counts = {**CONTROL_COUNTS, "fresh_confirmation": 1024}
    for source in (old, development):
        if source["expected_group_counts"] != counts or source["expected_state_count"] != 1778:
            raise ValueError("retain the entire 1778-state development roster and every group")
    if tuple(old[key] for key in ("expected_fresh_state_count", "expected_control_state_count",
            "nonregression_limit", "trigger_speedup_limit", "long_lp_budget", "control_integer_state_budget")) != (
                1024, 754, 1.05, .95, 4096, 100_000):
        raise ValueError("retain every original timing gate and comparison budget")
    if len(old["cells"]) != 16 or [(cell["node"], cell["lanes"]) for cell in development["cells"]] != [
            (cell["node"], cell["lanes"]) for cell in old["cells"]]:
        raise ValueError("all 16 frozen cells must occur in order")
    cells = deepcopy(development["cells"])
    actual = dict.fromkeys(counts, 0)
    for cell, previous_cell in zip(cells, old["cells"], strict=True):
        if len(cell["states"]) != len(previous_cell["states"]):
            raise ValueError("preserve every old state before timing")
        for index, (state, original) in enumerate(zip(cell["states"], previous_cell["states"], strict=True)):
            added = {key: state[key] for key in ("zero_delta_source_index", "zero_delta_source_state_id")}
            if {key: value for key, value in state.items() if key not in added} != original:
                raise ValueError("development metadata differs from the complete preceding timing roster")
            actual[state["timing_group"]] += 1
            state.update(timing_source="zero_delta_development", timing_source_index=index,
                         timing_source_state_id=state["state_id"])
        cell["orders"] = timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
    if actual != counts:
        raise ValueError("every registered development group must retain its complete size")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    protocol = deepcopy(old)
    protocol.pop("registered_at_utc", None)
    protocol.update(experiment="zero_delta_affine_four_arm_development_timing_v1", cells=cells,
        cpu=min(os.sched_getaffinity(0)), labels=LABELS, repeats=REPEATS,
        warmups_per_label=WARMUPS_PER_LABEL, order_seed=ORDER_SEED, execution_sources=sources,
        expected_timed_calls=42672, expected_warmup_calls=128, expected_selector_calls=42800,
        reference_sources={"zero_delta_development": {
            "path": str(DEVELOPMENT_OUTPUT.relative_to(ROOT)), "registered_at_utc": development["registered_at_utc"]}},
        prior_decisions={**development["prior_decisions"], "affine_dual_fresh_timing": previous["decision"],
                         "zero_delta_affine_development": result["decision"]},
        comparison="Candidate is the exact zero-coefficient affine shortcut; affine is the frozen preceding implementation. Branch and long retain the original residual/strict selectors and144/256/shared8192 or4096/shared100000 budgets. Every arm has newly measured whole-call clocks and must reproduce complete reference-matching certified actions. No historical clocks serve as controls.",
        roster="All1778 previously inspected states are reused development evidence. Preserve their original1024 former-fresh and754 older-control labels and fixed regression keys. No new queues, selection, screening or independently sampled confirmation.",
        measurement="Six adjacent four-arm cold blocks perstate. Removing affine retains every original candidate/branch/long relative permutation exactly once. Affine is inserted at each position once, with two independently sampled distinct positions repeated; position counts are2,2,1,1 in random assignment. This is not all24 permutations or exact four-position balance. Shuffle blocks and state sweeps using frozen seed2026092892. Two untimed warmups perarm/cell. Whole-call wall/CPU clocks end before any serialization or validation; retain every observation without trimming or reruns.",
        accounting="42672 timed plus128 warmup calls. Borrow all reference/oracle records,12 native snapshots and three real residual outputs perstate from development; only754 older controls have genuine borrowed axis predecessors. Record each arm's complete actual work, including guard checks, zero skips, LP flags, affine preparation/deltas, copies, DP, export, canonical prefix/DFS and queries. No new reference or screening calls.",
        gate="All execution, references, certificates, canonical quality, expected-output reproduction and repeat/affinity/order requirements hold for all four arms and warmups. Candidate/branch and candidate/long retain all former-fresh separated ordinary gates and every original older-control gate, including trigger/long<=.95. Candidate/affine additionally must pass<=1.05 pooled/cell/p99 on each ordinary group, pooled/p99/every-state on34 historical states, and pooled/every-state on16 trigger states. Zero timing remains descriptive with mandatory quality. ADVANCE requires every unchanged primary gate and all affine comparison gates.",
        limitations="All1778 states now support development regression only, including the previously fresh cohort. Four cached catalogues and virtual identical lanes remain fixed; paired catalogue copies and repeats are dependent. Six four-arm rounds retain the three-arm relative orders but change workload cadence. No independent confirmation, physical scaling, flow-time or significance claim. All prior HOLDs remain recorded and the paper remains untouched.")
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, roster, budgets or four-arm timing design changed")


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
        raise FileExistsError("zero-delta affine timing outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.zero_delta_affine_timing_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
