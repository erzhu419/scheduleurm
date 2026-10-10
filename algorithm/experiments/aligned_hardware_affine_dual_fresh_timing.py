"""Time independently drawn ordinary queues alongside all frozen controls."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_fresh_timing_20260928.json"
CONFIRMATION_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_fresh_confirmation_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_timing_20260928.json"
LABELS, REPEATS, WARMUPS_PER_LABEL = ("candidate", "branch", "long"), 6, 2
ORDER_SEED = 2026092891
CONTROL_COUNTS = dict(ordinary_development=688, historical_regression=34,
                      zero_queue=16, trigger_control=8, new_trigger=8)
SOURCES = ("algorithm/experiments/aligned_hardware_affine_dual_fresh_timing.py",
           "algorithm/experiments/affine_dual_fresh_timing_summary.py")


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


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    old, quality = (json.loads(_protocol_path(path).read_text())
                    for path in (PREVIOUS_OUTPUT, CONFIRMATION_OUTPUT))
    old_result, result = (json.loads(path.read_text()) for path in (PREVIOUS_OUTPUT, CONFIRMATION_OUTPUT))
    if old_result["decision"] != "AFFINE_DUAL_LOCAL_TIMING_ADVANCE" or not all(old_result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "frozen_residual_results_reproduced", "local_timing_gate_passed")):
        raise ValueError("the complete prior affine timing advance is required")
    if result["decision"] != "AFFINE_DUAL_FRESH_CONFIRMATION_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "confirmation_gate_passed")):
        raise ValueError("complete independent fresh confirmation is required before timing")
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
    if (old["expected_group_counts"] != CONTROL_COUNTS or old["expected_state_count"] != 754
            or tuple(old[key] for key in ("nonregression_limit", "trigger_speedup_limit",
                "long_lp_budget", "control_integer_state_budget")) != (1.05, .95, 4096, 100_000)):
        raise ValueError("retain the complete prior 754 controls and every original timing gate")
    if tuple(quality[key] for key in ("expected_state_count", "expected_fresh_state_count",
            "expected_control_state_count")) != (1778, 1024, 754):
        raise ValueError("retain every independent fresh state and every old control")
    if [(cell["node"], cell["lanes"]) for cell in quality["cells"]] != [
            (cell["node"], cell["lanes"]) for cell in old["cells"]] or len(old["cells"]) != 16:
        raise ValueError("all 16 original cells must occur in order")
    cells = deepcopy(quality["cells"])
    counts = {**CONTROL_COUNTS, "fresh_confirmation": 1024}
    actual = dict.fromkeys(counts, 0)
    for cell, prior in zip(cells, old["cells"], strict=True):
        fresh = [state for state in cell["states"] if state["confirmation_source"] == "fresh"]
        controls = [state for state in cell["states"] if state["confirmation_source"] == "borrowed_control"]
        if len(fresh) != 64 or len(controls) != len(prior["states"]) or len(fresh) + len(controls) != len(cell["states"]):
            raise ValueError("each cell must retain its 64 fresh states and complete controls")
        for state, previous in zip(controls, prior["states"], strict=True):
            if {key: value for key, value in state.items() if not key.startswith("confirmation_")} != previous:
                raise ValueError("borrowed control metadata differs from the preceding affine roster")
        if len({tuple(state["queue"]) for state in cell["states"]}) != len(cell["states"]):
            raise ValueError("fresh and control queues must remain distinct within each cell")
        for index, state in enumerate(cell["states"]):
            group = "fresh_confirmation" if state["confirmation_source"] == "fresh" else state["timing_group"]
            if state["timing_group"] != group:
                raise ValueError("fresh states must remain a separate timing endpoint")
            actual[group] += 1
            state.update(timing_source="fresh_confirmation_quality", timing_source_index=index,
                         timing_source_state_id=state["state_id"])
        cell["orders"] = timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
    if actual != counts:
        raise ValueError("every frozen timing cohort must retain its complete size")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    protocol = deepcopy(old)
    protocol.pop("registered_at_utc", None)
    protocol.update(experiment="affine_dual_fresh_equal_quality_timing_v1", cells=cells,
        cpu=min(os.sched_getaffinity(0)), labels=LABELS, repeats=REPEATS,
        warmups_per_label=WARMUPS_PER_LABEL, order_seed=ORDER_SEED,
        execution_sources=sources, expected_state_count=1778, expected_fresh_state_count=1024,
        expected_control_state_count=754, expected_group_counts=counts,
        expected_timed_calls=32004, expected_warmup_calls=96, expected_selector_calls=32100,
        reference_sources={"fresh_confirmation_quality": {
            "path": str(CONFIRMATION_OUTPUT.relative_to(ROOT)), "registered_at_utc": quality["registered_at_utc"]}},
        prior_decisions={**quality["prior_decisions"], "affine_dual_timing": old_result["decision"],
                         "affine_dual_fresh_confirmation": result["decision"]},
        roster="All 1024 independent ordinary node-states plus the complete prior 754 controls. Fresh queues use four new seeds and 16 queues per seed/R, paired across four cached catalogues; 256 distinct R/Q draws. Every cell uses new fixed six-permutation orders; no state replacement or outcome-dependent extension.",
        accounting="32004 timed plus96 warmup selector calls. References, oracle records, both native ladders and expected residual outputs are borrowed from the complete quality phase. Only the754 reused controls have two genuine borrowed axis predecessors. Charge all actual whole-selector work, including affine preparation, updates, DP, export, retained copies, prefixes and canonical DFS. No screening or new references in timing.",
        gate="All reference, certificate, canonical quality, output reproduction, stable-repeat, order and affinity requirements hold on every state and warmup. For the1024 fresh states separately, candidate/branch and candidate/long pooled, each node/R cell and pooled p99 state-median ratios must be<=1.05. Reapply all original ordinary,34 historical and16 trigger timing gates only to the754 controls, including trigger/long<=.95. Zero timing remains descriptive and zero quality mandatory. ADVANCE requires every fresh and retained-control gate; no changed thresholds, cohorts, budgets, orders, trimming or reruns.",
        limitations="Independent queue draws under the unchanged conditional law on four cached catalogues and virtual identical lanes. Paired catalogue copies and six timing repeats are dependent. This is local ordinary-queue confirmation, with separate reused regression controls; no new physical scaling, flow-time or statistical significance claim. All earlier HOLDs remain recorded and the paper remains untouched.")
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, roster, budgets or timing design changed")


def _selectors():
    from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
    from algorithm.residual_capacity_frontier_frame_candidates import select_branch_reserved_residual_capacity_frontier_frame
    from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
    return {"candidate": select_affine_dual_residual_capacity_frontier_frame,
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
        raise FileExistsError("affine fresh timing outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.affine_dual_fresh_timing_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
