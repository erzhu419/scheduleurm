"""Balanced whole-selector timing on the frozen branch-reserved queue panels."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
from time import perf_counter, perf_counter_ns, process_time_ns

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_root_pruning_development import load_cell

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_timing_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_branch_reserved_confirmation_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_confirmation_20260927.json"
LABELS = ("candidate", "routed", "probe", "long")
REPEATS = 8
WARMUPS_PER_LABEL = 2
TIMING_FRESH_PER_PANEL = 16
ORDER_SEED = 2026092829
SOURCES = (
    "algorithm/experiments/aligned_hardware_branch_reserved_timing.py",
    "algorithm/experiments/branch_reserved_timing_summary.py",
)


def timing_orders(node, lanes, count):
    rng = random.Random(f"{ORDER_SEED}:{node}:{lanes}")
    # Williams orders balance positions and every directed adjacent pair.
    designs = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    label_orders = []
    for _ in range(count):
        orders = [[LABELS[index] for index in order] for order in designs] * 2
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
    prior = json.loads(PRIOR_OUTPUT.read_text())
    source = json.loads(PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json").read_text())
    previous = json.loads(PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    previous_result = json.loads(PREVIOUS_OUTPUT.read_text())
    if prior["decision"] != "BRANCH_RESERVED_FRESH_CONFIRMATION_ADVANCE" or not prior["confirmation_gate_passed"]:
        raise ValueError("successful frozen branch-reserved work confirmation is required")
    if previous_result["decision"] != "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD" or not all(
            previous_result[key] for key in ("certificate_checks_passed", "effective_references_complete",
                "candidate_quality_gate_passed", "comparison_quality_gate_passed", "execution_checks_passed")):
        raise ValueError("the quality-valid preceding timing HOLD must be retained")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for old in (source, previous):
        if snapshots != old["models"]:
            raise ValueError("cached physical catalogue models changed")
        if tuple(old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
            raise ValueError("developed routing budgets must remain fixed")
        for path, literal in old["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    trigger = _keys(previous["historical_trigger_keys"])
    score = _keys(previous["score_unproved_keys"])
    untimed = _keys(previous["previous_untimed_probe_keys"])
    fresh_ids = {lanes: {state["state_id"] for panel in source["queue_panels"][str(lanes)]
        for state in panel["states"][:TIMING_FRESH_PER_PANEL]} for lanes in source["lanes"]}
    cells, retained, tails = [], set(), []
    for cell in source["cells"]:
        states = []
        for index, state in enumerate(cell["states"]):
            if not (state["source"] == "historical_regression"
                    or state["source"] == "fresh_confirmation" and state["state_id"] in fresh_ids[cell["lanes"]]):
                continue
            key = cell["node"], cell["lanes"], tuple(state["queue"])
            registered = deepcopy(state)
            registered.update(source_index=index, source_state_id=state["state_id"],
                previous_integer_trigger=key in trigger, score_unproved_regression=key in score,
                previous_untimed_probe_trigger=key in untimed)
            if state["source"] == "historical_regression":
                retained.add(key)
                registered["cohort"] = "historical_trigger" if key in trigger else "ordinary_tail_regression"
                if key not in trigger:
                    tails.append({"node": cell["node"], "lanes": cell["lanes"], "queue": state["queue"]})
            states.append(registered)
        cells.append({"node": cell["node"], "lanes": cell["lanes"], "states": states,
                      "orders": timing_orders(cell["node"], cell["lanes"], len(states))})
    if retained != _keys(source["historical_regression_keys"]) or not score <= trigger <= retained or not untimed <= trigger:
        raise ValueError("all historical keys and their original subgroups must remain fixed")
    if sum(len(cell["states"]) for cell in cells) != 1058 or (len(trigger), len(score), len(tails)) != (18, 6, 16):
        raise ValueError("timing must retain 1024 panel states and all 34 historical regressions")
    limits = {key: previous[key] for key in ("fresh_nonregression_limit", "fresh_speedup_limit",
        "score_gain_limit", "state_ratio_limit", "nonregression_limit", "trigger_long_ratio_limit", "tail_state_limit")}
    return json.loads(json.dumps({"schema_version": 1, "experiment": "branch_reserved_whole_selector_timing_v1",
        "models": snapshots, "lanes": source["lanes"], "cells": cells,
        "execution_order": [[cell["node"], cell["lanes"]] for cell in cells],
        "labels": LABELS, "repeats": REPEATS, "warmups_per_label": WARMUPS_PER_LABEL,
        "timing_fresh_per_panel": TIMING_FRESH_PER_PANEL, "order_seed": ORDER_SEED,
        "cpu": min(os.sched_getaffinity(0)),
        **{key: previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "long_lp_budget", "control_integer_state_budget", "reference_budget", "oracle_state_budget")},
        **limits, "direct_predecessor_limit": previous["nonregression_limit"],
        "historical_trigger_keys": previous["historical_trigger_keys"], "score_unproved_keys": previous["score_unproved_keys"],
        "previous_untimed_probe_keys": previous["previous_untimed_probe_keys"], "ordinary_tail_keys": tails,
        "reference_source": {"path": str(PRIOR_OUTPUT.relative_to(ROOT)), "registered_at_utc": source["registered_at_utc"]},
        "prior_decisions": {**prior["prior_decisions"], "branch_reserved_confirmation": prior["decision"]},
        "expected_timed_calls": 33856, "expected_warmup_calls": 128, "expected_selector_calls": 33984,
        "execution_sources": sources,
        "roster": "First16 of64 queues in each of the four previously locked seed2821-24 panels/R/node (1024 states), plus the identical18 historical trigger and16 ordinary-tail keys. All1058 states timed; no zeros. These queues already supported work confirmation; this is a new timing measurement, not another independent queue confirmation. Fixed score-unproved6 and other12 labels remain historical, regardless of new certificates.",
        "measurement": "Four fresh selector calls per adjacent block: branch-reserved candidate, original certificate-routed, short resumable-probe, long strict. Eight sweeps; Williams orders ABDC,BCAD,CDBA,DACB each twice, preshuffled perstate, and stateorder shuffled persweep. Each arm occupies everyposition twice and every directed adjacentpair twice. Two warmups perarm/cell at firststate. Entireselector wall/CPU intervals, all four calls before serialization/validation, all repeats retained. No manual GC, instrumentation or waits.",
        "reference_policy": "Borrow all six fullK4096/reference-only oracle fields from the matched preceding work-confirmation cell before measured calls. Match candidate and original-routed outputs to their independently executed frozen outputs and native144 initial snapshots. Probe/long must pass their own certificates and independent reference checks. Zero new screening, reference and oracle calls; all borrowed costs separately identified, never charged as new work.",
        "accounting": "33856 timed and128 warmup selector calls; final cumulative LP/preprocessing/capacity charged once and every spent integer state retained. Timed metrics exclude warmups; warmups charged separately. Report all four implementations, borrowed reference/oracle costs, and zero new reference/oracle/screen counts.",
        "runtime_policy": "One host, serial cells, pin minimum available CPU before numerical imports, BLAS/OMP environment1, normal automatic GC. Record actual affinity, GC and threadpools/native thread count. Independent fresh search state does not mean cold hardware caches.",
        "gate": "Complete roster and valid borrowed references, all four arms canonical exact and reference matching, frozen-output consistency, repeat consistency and valid reserved budgets/accounting. Preserve all original probe/long fresh,score-unproved,other-trigger,trigger and individual-tail timing thresholds. Separately require candidate/routed fresh pooled,eachcell,p99<=1.05; historical34 pooled,p99,everypairedstate<=1.05. Do not require the .95 score-unproved/probe improvement against routed. ADVANCE requires every gate. Panel speedup requires pooled<=.95 against allthree controls. Finish every cell; no rerun, trimming, replacement, altered budget/order/threshold or post-outcome cohort changes.",
        "limitations": "New local timing of previously used conditional queue panels and fixed regressions on four cached catalogues and virtualR8/16/32/64. Catalogue pairs and eight repeats are dependent. Borrowed references remain independent of the timing arms. Descriptive paired timing establishes no statistical significance, physical scaling or flow-time benefit. Prior root-pruning and certificate-routed timing HOLDs remain unchanged; A/A diagnostics do not cancel them."}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered source, roster, references, budgets or timing design changed")


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key
    from algorithm.experiments.certificate_routed_development_summary import result_preprocessing
    from algorithm.experiments.strict_score_proof_timing_summary import _same
    if selectors is None:
        from algorithm.branch_reserved_certificate_routed_frame_candidates import select_branch_reserved_certificate_routed_frame
        from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
        from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
        from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
        selectors = {"candidate": select_branch_reserved_certificate_routed_frame, "routed": select_certificate_routed_frame,
                     "probe": select_resumable_probe_frame, "long": select_strict_score_proof_canonical_completion_frame}
    source = protocol["reference_source"]
    borrowed = load_cell(ROOT / source["path"], model.node, cell["lanes"], source["registered_at_utc"])
    started = perf_counter()
    row = {"node": model.node, "lanes": cell["lanes"], "physical_lanes": model.lanes,
        "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "states": [], "warmups": [], "state_orders": [],
        "new_reference_calls": 0, "new_oracle_calls": 0, "screen_selector_calls": 0}
    references, expected_keys = [], []
    for registered in cell["states"]:
        saved = borrowed["states"][registered["source_index"]]
        if (saved["state_id"], saved["queue"]) != (registered["source_state_id"], registered["queue"]):
            raise ValueError("borrowed timing state differs from its frozen source")
        fields = {key: saved[key] for key in ("reference", "reference_metrics", "reference_complete", "oracle",
            "effective_reference_complete", "effective_reference_source")}
        reference = selection_from_record(fields["reference"])
        expected = runtime.reference_key(reference, tuple(registered["queue"]), len(model.options))
        if expected is None and fields["oracle"] is not None:
            expected = oracle_key(fields["oracle"], len(model.options))
        references.append(reference)
        expected_keys.append(expected)
        row["states"].append({**registered, **fields,
            "expected_initial_lp": {key: saved["traces"]["old"]["144"][key] for key in ("selection", "preprocessing", "capacity_propagation")},
            "expected_reserved_initial_lp": {key: saved["traces"]["new"]["144"][key] for key in ("selection", "preprocessing", "capacity_propagation")},
            "expected_results": {"candidate": saved["routed"], "routed": saved["old_routed"]},
            "first_results": {}, "repeat_mismatches": {label: [] for label in LABELS},
            "metrics": {label: runtime._metrics() for label in LABELS}, "blocks": [],
            "wall_ns": {label: [None] * REPEATS for label in LABELS}, "cpu_ns": {label: [None] * REPEATS for label in LABELS}})
    row["borrowed_reference_count"] = len(row["states"])

    def call(label, index):
        budgets = ((protocol["long_lp_budget"], protocol["control_integer_state_budget"]) if label == "long" else
            (protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"]))
        return selectors[label](model.options, tuple(row["states"][index]["queue"]), cell["lanes"], *budgets)

    def record(metric, literal, label, index):
        state = row["states"][index]
        runtime._record(metric, selection_from_record(literal["selection"]), model, tuple(state["queue"]), cell["lanes"],
            literal["selection"]["lp_node_budget"], expected_keys[index], result_preprocessing(literal, label), references[index])

    row["runtime_before_warmup"] = runtime_provenance()
    for _ in range(WARMUPS_PER_LABEL):
        for label in LABELS:
            literal = asdict(call(label, 0))
            metrics = runtime._metrics()
            record(metrics, literal, label, 0)
            row["warmups"].append({"label": label, "state_index": 0, "result": literal, "metrics": metrics})
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
        print(json.dumps({"protocol": str(protocol_path), "timed_states": sum(len(cell["states"]) for cell in protocol["cells"])}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists() for cell in protocol["cells"]):
        raise FileExistsError("branch-reserved timing outputs already exist")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.branch_reserved_timing_summary import summarize
    summary = summarize(protocol, rows)
    summary["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
