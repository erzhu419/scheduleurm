"""Frozen exact-LP twelve-arm timing on new queues and every retained control."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_timing_20260929.json"
CONFIRMATION_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_20260929.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_timing_20260929.json"
LABELS, REPEATS, WARMUPS_PER_LABEL = ("candidate", "direct", "own", "fused", "settled", "native", "face", "early", "zero", "affine", "branch", "long"), 6, 2
ORIGINAL_LABELS = ("direct", "branch", "long")
ORDER_SEED = 2026092915
CONTROL_COUNTS = dict(ordinary_development=688, historical_regression=34,
                      zero_queue=16, trigger_control=8, new_trigger=8, fresh_confirmation=1024, independent_fresh=1024)
SOURCES = ("algorithm/experiments/aligned_hardware_small_dimension_exact_native_confirmation_timing.py",
           "algorithm/experiments/small_dimension_exact_native_confirmation_timing_summary.py")


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
        for order, position in zip(orders, rng.sample(range(12), 6), strict=True):
            order.insert(position, "candidate")
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
    if previous["decision"] != "SMALL_DIMENSION_EXACT_NATIVE_LOCAL_TIMING_ADVANCE" or not all(previous[key] for key in (
            "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
            "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced", "local_timing_gate_passed")):
        raise ValueError("the complete prior small-dimension timing advance is required")
    if result["decision"] != "SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_ADVANCE" or not all(result[key] for key in (
            "execution_checks_passed", "references_complete", "quality_gate_passed", "confirmation_gate_passed", "work_gate_passed")):
        raise ValueError("complete independent exact-LP quality confirmation is required before timing")
    if quality["prior_decisions"]["small_dimension_exact_native_timing"] != previous["decision"]:
        raise ValueError("quality confirmation must preserve the actual preceding timing advance")
    plan = quality["subsequent_timing_plan"]
    if (plan["order_seed"], tuple(plan["labels"]), plan["repeats"], plan["warmups_per_label"],
            plan["expected_timed_calls"], plan["expected_warmup_calls"], plan["expected_selector_calls"],
            plan["nonregression_limit"], plan["trigger_speedup_limit"], plan["independent_speedup_limit"],
            plan["expected_timing_gate_count"]) != (ORDER_SEED, LABELS, REPEATS, WARMUPS_PER_LABEL,
                                             275472, 384, 275856, 1.05, .95, .95, 189):
        raise ValueError("retain the full timing plan frozen before new independent quality outcomes")
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
    counts = {**CONTROL_COUNTS, "new_independent_fresh": 1024}
    if (old["expected_state_count"] != 2802 or old["expected_group_counts"] != CONTROL_COUNTS
            or tuple(old[key] for key in (
            "nonregression_limit", "trigger_speedup_limit", "long_lp_budget", "control_integer_state_budget")) != (
                1.05, .95, 4096, 100_000)):
        raise ValueError("retain all2802 preceding states and every original gate")
    if (tuple(quality[key] for key in ("expected_state_count", "expected_fresh_state_count", "expected_control_state_count"))
            != (3826, 1024, 2802) or quality["expected_group_counts"] != counts):
        raise ValueError("retain every independent queue and all complete preceding controls")
    if len(old["cells"]) != 16 or [(cell["node"], cell["lanes"]) for cell in quality["cells"]] != [
            (cell["node"], cell["lanes"]) for cell in old["cells"]]:
        raise ValueError("all 16 frozen cells must occur in order")
    cells, actual = deepcopy(quality["cells"]), dict.fromkeys(counts, 0)
    for cell, previous_cell in zip(cells, old["cells"], strict=True):
        fresh = [state for state in cell["states"] if state["small_dimension_confirmation_source"] == "fresh"]
        controls = [state for state in cell["states"] if state["small_dimension_confirmation_source"] == "borrowed_control"]
        if len(fresh) != 64 or len(controls) != len(previous_cell["states"]) or len(fresh) + len(controls) != len(cell["states"]):
            raise ValueError("each cell must preserve64 new queues and every complete preceding control")
        for state, original in zip(controls, previous_cell["states"], strict=True):
            pointers = ("timing_source", "timing_source_index", "timing_source_state_id")
            if {key: value for key, value in state.items() if not key.startswith("small_dimension_confirmation_") and key not in pointers} != {
                    key: value for key, value in original.items() if key not in pointers}:
                raise ValueError("borrowed control metadata differs from its actual preceding timing roster")
        if len({tuple(state["queue"]) for state in cell["states"]}) != len(cell["states"]):
            raise ValueError("new queues and retained controls must remain distinct within each configuration")
        for index, state in enumerate(cell["states"]):
            if (state["timing_group"] == "new_independent_fresh") != (state["small_dimension_confirmation_source"] == "fresh"):
                raise ValueError("the new independent queues must remain a separate endpoint")
            actual[state["timing_group"]] += 1
            state.update(timing_source="small_dimension_confirmation", timing_source_index=index,
                         timing_source_state_id=state["state_id"])
        cell["orders"] = timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
    if actual != counts:
        raise ValueError("retain the new independent endpoint and every preceding timing cohort separately")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    protocol = deepcopy(old)
    for key in ("registered_at_utc", "expected_independent_state_count", "expected_borrowed_state_count",
                "expected_fresh_state_count", "expected_control_state_count"):
        protocol.pop(key, None)
    protocol.update(experiment="small_dimension_exact_native_independent_twelve_arm_timing_v1", cells=cells,
        focus_keys=deepcopy(quality["focus_keys"]),
        labels=LABELS, repeats=REPEATS, warmups_per_label=WARMUPS_PER_LABEL,
        cpu=min(os.sched_getaffinity(0)), order_seed=ORDER_SEED, execution_sources=sources,
        expected_state_count=3826, expected_fresh_state_count=1024, expected_control_state_count=2802,
        expected_group_counts=counts, expected_timed_calls=275472, expected_warmup_calls=384,
        expected_selector_calls=275856, expected_timing_gate_count=189, independent_speedup_limit=.95,
        reference_sources={"small_dimension_confirmation": {
            "path": str(CONFIRMATION_OUTPUT.relative_to(ROOT)), "registered_at_utc": quality["registered_at_utc"]}},
        prior_decisions={**quality["prior_decisions"], "small_dimension_exact_native_timing": previous["decision"],
                         "small_dimension_exact_native_confirmation": result["decision"]},
        comparison="The frozen exact low-dimensional LP selector and the eleven genuine preceding comparators retain their real implementations and budgets. Candidate and every non-long control reproduce the complete saved quality results and receive their independent action/certificate/fee auditors. All twelve whole-call clocks are newly measured. Preserve the prior timing advance, all earlier HOLDs and the invalid historical audit artifact.",
        roster="All1024 new ordinary node-states plus every2802 previously used control. Four new panel seeds2026092911..14 each contribute16 queues perR, paired over four unchanged catalogues. Keep the256 distinctR/queue draws, new_independent_fresh endpoint and all seven old groups separate; preserve all fixed regression pointers and original metadata. Zero queue latency remains descriptive.",
        measurement="Six adjacent twelve-arm cold blocks perstate under seed2026092915. Preserve the six direct/branch/long relative permutations and all nine insertion laws: affine2,2,1,1; zero2,1,1,1,1; early all six positions; face/native/settled/fused/own/candidate each six distinct insertion positions at their original stage. Shuffle state sweeps independently; two untimed warmups perarm/cell. Whole-call clocks include construction, exact or generic LP solves, primal/dual calculations, own checks, domain contraction, root decisions, all actual snapshots/evidence copies, root phase, LP, dual-face exports and the sole shared completion operation. Clocks end before serialization/validation. Retain every observation without trimming, screening or reruns.",
        accounting="275472 timed plus384 warmup selectors. Borrow all reference/oracle records and real quality control expectations. Borrow24 native snapshots perstate:zero/branch/affine each4, own/direct/candidate each4 plus their separate actual raw/settled/phase root triples. No separate native engine, reference, oracle or screening runs during timing. Only controls with genuine saved axis predecessors retain them. Charge final cumulative native work once and the sole probe once, including failed completion; preserve all actual phase, snapshot, export, exact-LP and domain-operation fees.",
        gate="Keep all155 original timing gates on the2802 controls: each old ordinary group separately pooled/cell/p99<=1.05 versus every comparator;34 historical pooled/p99/every-state<=1.05;16 trigger pooled/every-state<=1.05 and trigger/long<=.95. On the1024 new states add pooled/cell/p99<=1.05 for each of eleven comparators, and candidate/direct pooled wall<=.95 to confirm a positive benefit. All189 gates and every reference/action/canonical/certificate/fee/reproduction/repeat/order/affinity check must pass. Zero timing is descriptive with mandatory quality. No new ordinary every-state gate or removal of tails, comparators or failed observations.",
        limitations="Independent new queue draws under the unchanged conditional sampling law; four paired catalogue copies and six timing repeats are dependent. The2802 old states are regression controls. This bounded local timing confirmation does not establish new physical scaling, flow-time gains or significance. Earlier HOLDs remain recorded; paper and live dispatch remain unchanged.")
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, roster, budgets or independent exact-LP timing design changed")


def _selectors():
    from algorithm.small_dimension_exact_native_frame_candidates import select_small_dimension_exact_native_frame
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
    return {"candidate": select_small_dimension_exact_native_frame,
            "direct": select_direct_probe_own_tight_native_frame,
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
            "previous_direct_probe_routed": saved["previous_direct_probe_routed"],
            "previous_direct_probe_native_root": saved["previous_direct_probe_native_root"],
            "previous_direct_probe_native_traces": saved["previous_direct_probe_native_traces"],
            "candidate_native_root": saved["candidate_native_root"],
            "candidate_native_traces": saved["candidate_native_traces"],
            "expected_results": {"candidate": saved["routed"], "direct": saved["previous_direct_probe_routed"], "own": saved["previous_own_tight_routed"], "fused": saved["previous_fused_root_routed"], "settled": saved["previous_settled_root_routed"], "native": saved["previous_native_root_routed"], "face": saved["previous_dual_face_routed"], "early": saved["previous_early_routed"], "zero": saved["previous_zero_routed"],
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
            print(f"{model.node} R{cell['lanes']}: twelve-arm timing {repeat + 1}/{REPEATS} complete", flush=True)
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
        raise FileExistsError("independent exact-LP timing outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.small_dimension_exact_native_confirmation_timing_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
