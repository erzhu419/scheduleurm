"""Eleven-arm direct-root-probe timing with every genuine prior control retained."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_direct_probe_own_tight_native_timing_20260929.json"
DEVELOPMENT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_direct_probe_own_tight_native_development_20260929.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_own_tight_dual_native_timing_20260928.json"
PREVIOUS_DEVELOPMENT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_own_tight_dual_native_development_20260928.json"
LABELS, REPEATS, WARMUPS_PER_LABEL = ("candidate", "own", "fused", "settled", "native", "face", "early", "zero", "affine", "branch", "long"), 6, 2
ORIGINAL_LABELS = ("candidate", "branch", "long")
ORDER_SEED = 2026092905
CONTROL_COUNTS = dict(ordinary_development=688, historical_regression=34,
                      zero_queue=16, trigger_control=8, new_trigger=8)
SOURCES = ("algorithm/experiments/aligned_hardware_direct_probe_own_tight_native_timing.py",
           "algorithm/experiments/direct_probe_own_tight_native_timing_summary.py")


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
        positions = list(range(5))
        rng.shuffle(positions)
        positions.append(rng.randrange(5))
        rng.shuffle(positions)
        for order, position in zip(orders, positions, strict=True):
            order.insert(position, "zero")
        positions = list(range(6))
        rng.shuffle(positions)
        for order, position in zip(orders, positions, strict=True):
            order.insert(position, "early")
        for order, position in zip(orders, rng.sample(range(7), 6), strict=True):
            order.insert(position, "face")
        for order, position in zip(orders, rng.sample(range(8), 6), strict=True):
            order.insert(position, "native")
        for order, position in zip(orders, rng.sample(range(9), 6), strict=True):
            order.insert(position, "settled")
        for order, position in zip(orders, rng.sample(range(10), 6), strict=True):
            order.insert(position, "fused")
        for order, position in zip(orders, rng.sample(range(11), 6), strict=True):
            order.insert(position, "own")
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
    old, parent_development, development = (json.loads(_protocol_path(path).read_text())
                        for path in (PREVIOUS_OUTPUT, PREVIOUS_DEVELOPMENT_OUTPUT, DEVELOPMENT_OUTPUT))
    previous, parent_result, result = (json.loads(path.read_text())
                        for path in (PREVIOUS_OUTPUT, PREVIOUS_DEVELOPMENT_OUTPUT, DEVELOPMENT_OUTPUT))
    if previous["decision"] != "OWN_TIGHT_DUAL_NATIVE_LOCAL_TIMING_HOLD" or not all(previous[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "frozen_residual_results_reproduced")):
        raise ValueError("retain the complete quality-valid preceding own-tight native timing HOLD")
    if parent_result["decision"] != "OWN_TIGHT_DUAL_NATIVE_DEVELOPMENT_ADVANCE" or not all(parent_result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "development_gate_passed", "work_gate_passed")):
        raise ValueError("retain the actual complete parent own-tight native development")
    if result["decision"] != "DIRECT_PROBE_OWN_TIGHT_NATIVE_DEVELOPMENT_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "development_gate_passed",
            "work_gate_passed")):
        raise ValueError("complete direct-root-probe development is required before timing")
    if development["prior_decisions"]["own_tight_dual_native_timing"] != previous["decision"]:
        raise ValueError("new development must retain the actual preceding own-tight native timing HOLD")
    plan = development["subsequent_timing_plan"]
    if (plan["order_seed"], tuple(plan["labels"]), plan["repeats"], plan["warmups_per_label"],
            plan["expected_timed_calls"], plan["expected_warmup_calls"], plan["nonregression_limit"],
            plan["trigger_speedup_limit"]) != (ORDER_SEED, LABELS, REPEATS, WARMUPS_PER_LABEL,
                                               184932, 352, 1.05, .95):
        raise ValueError("retain the timing plan frozen before direct-root-probe development outcomes")
    snapshots, sources = json.loads(json.dumps([model.snapshot() for model in models])), {}
    for source in (old, parent_development, development):
        if source["models"] != snapshots:
            raise ValueError("retain the frozen physical catalogue models")
        if tuple(source[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
                "reference_budget", "oracle_state_budget")) != (144, 256, 8192, 4096, 5_000_000):
            raise ValueError("retain the frozen routing and reference budgets")
        for path, literal in source["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    counts = {**CONTROL_COUNTS, "fresh_confirmation": 1024, "independent_fresh": 1024}
    if any(source["expected_state_count"] != 2802 or source["expected_group_counts"] != counts
           for source in (old, parent_development, development)) or tuple(old[key] for key in (
            "nonregression_limit", "trigger_speedup_limit", "long_lp_budget", "control_integer_state_budget")) != (
                1.05, .95, 4096, 100_000):
        raise ValueError("retain all2802 preceding states and every original gate")
    if len(old["cells"]) != 16 or any([(cell["node"], cell["lanes"]) for cell in source["cells"]] != [
            (cell["node"], cell["lanes"]) for cell in old["cells"]] for source in (parent_development, development)):
        raise ValueError("all 16 frozen cells must occur in order")
    cells, actual = deepcopy(development["cells"]), dict.fromkeys(counts, 0)
    for cell, previous_cell, parent_cell in zip(cells, old["cells"], parent_development["cells"], strict=True):
        if len(cell["states"]) != len(previous_cell["states"]) or len(cell["states"]) != len(parent_cell["states"]):
            raise ValueError("preserve every preceding state before timing")
        for index, (state, original, timed) in enumerate(zip(cell["states"], parent_cell["states"], previous_cell["states"], strict=True)):
            if {key: value for key, value in original.items() if key not in (
                    "timing_source", "timing_source_index", "timing_source_state_id")} != {
                    key: value for key, value in timed.items() if key not in (
                    "timing_source", "timing_source_index", "timing_source_state_id")}:
                raise ValueError("parent development and timing must retain the same original state lineage")
            if {key: value for key, value in state.items() if key not in (
                    "direct_probe_source_index", "direct_probe_source_state_id")} != original:
                raise ValueError("development must preserve the complete preceding state metadata")
            actual[state["timing_group"]] += 1
            state.update(timing_source="direct_probe_development", timing_source_index=index,
                         timing_source_state_id=state["state_id"])
        cell["orders"] = timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
    if actual != counts:
        raise ValueError("retain every preceding timing cohort separately")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    protocol = deepcopy(old)
    for key in ("registered_at_utc", "expected_independent_state_count", "expected_borrowed_state_count",
                "expected_fresh_state_count", "expected_control_state_count"):
        protocol.pop(key, None)
    protocol.update(experiment="direct_probe_own_tight_native_eleven_arm_development_timing_v1", cells=cells,
        focus_keys=deepcopy(development["focus_keys"]),
        labels=LABELS, repeats=REPEATS, warmups_per_label=WARMUPS_PER_LABEL,
        cpu=min(os.sched_getaffinity(0)), order_seed=ORDER_SEED, execution_sources=sources,
        expected_timed_calls=184932, expected_warmup_calls=352, expected_selector_calls=185284,
        reference_sources={"direct_probe_development": {
            "path": str(DEVELOPMENT_OUTPUT.relative_to(ROOT)), "registered_at_utc": development["registered_at_utc"]}},
        prior_decisions={**development["prior_decisions"], "own_tight_dual_native_timing": previous["decision"],
                         "direct_probe_own_tight_native_development": result["decision"]},
        comparison="Candidate retains the own-tight native engine and offers the single root completion probe before splitting the next solved region. Own reproduces the genuine preceding own-tight whole router, including its actual old root phase and native prefixes; fused, settled, native, face, early, zero, affine, branch and long retain their real implementations and budgets. Candidate reproduces its new development output and receives the independent direct-probe auditor. All eleven whole-call clocks are newly measured. Retain the preceding own-tight timing HOLD, original invalid audit artifact and all earlier results.",
        roster="All2802 previously used states are development controls. Preserve the seven original timing_group labels, metadata and fixed regression keys without creating a new independent cohort or merging the three ordinary groups. Zero queue latency remains descriptive.",
        measurement="Six adjacent eleven-arm cold blocks perstate under frozen seed2026092905. Generate the preceding ten-arm law with the candidate role occupied by direct-probe: six candidate/branch/long relative permutations; affine insertion counts2,2,1,1; zero insertion counts2,1,1,1,1; early at all six positions once; face at six different positions out of seven; native at six different positions out of eight; settled at six different positions out of nine; fused at six different positions out of ten. Then insert own at six different randomly ordered positions out of eleven. Removing own retains that ten-arm law. This does not assert all39916800 permutations or full eleven-position balance. Shuffle state sweeps independently; two untimed warmups perarm/cell. Whole-call clocks include native construction, cached own-value preparation, coefficient checks and domain contraction, root decisions, advancement, every required snapshot or evidence object, the direct-probe root phase, LP, dual-face export and the unique shared completion operation; clocks end before serialization or validation. Retain every observation without trimming or reruns.",
        accounting="184932 timed plus352 warmup selectors. Borrow reference/oracle records, the new direct-probe expectation and genuine preceding own expectation with their separate actual raw/settled/phase root records and four-cap native prefixes, and every earlier routed control. Borrow20 native snapshots perstate:12 from the genuine zero/branch/original-affine ladders,4 from the previous own native engine and4 from the new candidate native engine. Each raw/settled/phase triple is one root record, not three new snapshots or executed advances. The first new cap stores the settled checkpoint when present and otherwise the raw checkpoint; it does not replace separate genuine raw-root evidence. Run no separate native engine, reference, oracle or screening during timing. Only the original754 states have genuine axis predecessors. Charge final cumulative native work including actual own-tight elimination once, the unique retained probe once including failed completion, plus every actual root-phase, snapshot, dual-face export and domain operation. Old own and every other control retain their genuine prior fees and old fifteen-field phase diagnostics; only candidate has the new stopped_for_branch diagnostic, which adds no fee.",
        gate="Require every reference/certificate/action/canonical/path/fee/repeat/frozen-output/order/affinity check on all states and warmups. Against fused, settled, native, face, early, zero, affine, branch and long preserve every original2802-state gate: each of independent_fresh, fresh_confirmation and ordinary_development separately pooled/cell/p99<=1.05;34 historical pooled/p99/every-state<=1.05;16 trigger states pooled/every-state<=1.05 and trigger/long<=.95. The newly measured own control satisfies the same1.05 gates without an extra trigger gain. Zero queue latency remains descriptive with mandatory quality. No new ordinary every-state gate and no deletion of failed tails or failed comparators.",
        limitations="All2802 queues have been used previously and this is development regression only. Seven source-cohort labels preserve the evidence history and do not establish new independence. Catalogue copies and timing repeats are dependent; four cached catalogues and virtual lanes remain fixed. Preserve every earlier HOLD and mean/tail counterexample. No new physical scaling, flow-time or significance claim; paper unchanged.")
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, roster, budgets or direct-probe eleven-arm timing design changed")


def _selectors():
    from algorithm.direct_probe_own_tight_native_frame_candidates import select_direct_probe_own_tight_native_frame
    from algorithm.own_tight_dual_native_frame_candidates import select_own_tight_dual_native_frame
    from algorithm.fused_root_dual_face_frame_candidates import select_fused_root_dual_face_frame
    from algorithm.settled_root_dual_face_frame_candidates import select_settled_root_dual_face_frame
    from algorithm.native_root_dual_face_frame_candidates import select_native_root_dual_face_frame
    from algorithm.dual_face_root_canonical_frame_candidates import select_dual_face_root_canonical_frame
    from algorithm.early_root_canonical_frame_candidates import select_early_root_canonical_frame
    from algorithm.zero_delta_affine_dual_residual_capacity_frame_candidates import select_zero_delta_affine_dual_residual_capacity_frontier_frame
    from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
    from algorithm.residual_capacity_frontier_frame_candidates import select_branch_reserved_residual_capacity_frontier_frame
    from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
    return {"candidate": select_direct_probe_own_tight_native_frame,
            "own": select_own_tight_dual_native_frame,
            "fused": select_fused_root_dual_face_frame,
            "settled": select_settled_root_dual_face_frame,
            "native": select_native_root_dual_face_frame,
            "face": select_dual_face_root_canonical_frame,
            "early": select_early_root_canonical_frame,
            "zero": select_zero_delta_affine_dual_residual_capacity_frontier_frame,
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
            **({"prior_routed": saved["prior_routed"]} if "prior_routed" in saved else {}),
            "previous_zero_traces": saved["previous_zero_traces"],
            "previous_zero_routed": saved["previous_zero_routed"],
            "original_affine_traces": saved["original_affine_traces"],
            "original_affine_routed": saved["original_affine_routed"],
            "previous_early_routed": saved["previous_early_routed"],
            "previous_dual_face_routed": saved["previous_dual_face_routed"],
            "previous_dual_face_traces": saved["previous_dual_face_traces"],
            "previous_native_root_routed": saved["previous_native_root_routed"],
            "previous_native_root_traces": saved["previous_native_root_traces"],
            "previous_settled_root_routed": saved["previous_settled_root_routed"],
            "previous_settled_root_traces": saved["previous_settled_root_traces"],
            "previous_fused_root_routed": saved["previous_fused_root_routed"],
            "previous_fused_root_traces": saved["previous_fused_root_traces"],
            "previous_own_tight_routed": saved["previous_own_tight_routed"],
            "previous_own_tight_native_root": saved["previous_own_tight_native_root"],
            "previous_own_tight_native_traces": saved["previous_own_tight_native_traces"],
            "candidate_native_root": saved["candidate_native_root"],
            "candidate_native_traces": saved["candidate_native_traces"],
            "expected_results": {"candidate": saved["routed"], "own": saved["previous_own_tight_routed"], "fused": saved["previous_fused_root_routed"], "settled": saved["previous_settled_root_routed"], "native": saved["previous_native_root_routed"], "face": saved["previous_dual_face_routed"], "early": saved["previous_early_routed"], "zero": saved["previous_zero_routed"],
                                 "affine": saved["original_affine_routed"], "branch": saved["old_routed"]},
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
            print(f"{model.node} R{cell['lanes']}: eleven-arm timing {repeat + 1}/{REPEATS} complete", flush=True)
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
        raise FileExistsError("direct-probe timing outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.direct_probe_own_tight_native_timing_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
