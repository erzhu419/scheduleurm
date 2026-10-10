"""Replay the frozen timing workload with two same-implementation sentinels."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
from time import perf_counter, perf_counter_ns, process_time_ns

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_cadence_diagnosis_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_confirmation_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
LABELS = ("candidate", "probe", "long")
TARGETS = (("node007", 64, (88, 19, 118)), ("node007", 64, (184, 78, 235)))
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_certificate_routed_cadence_diagnosis.py",
    "algorithm/experiments/certificate_routed_cadence_diagnosis_summary.py",
)


def same_result(actual, expected):
    """JSON lists and dataclass tuples represent the same saved sequences."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_result(actual[key], expected[key]) for key in actual)
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(same_result(left, right) for left, right in zip(actual, expected))
    return actual == expected


def build_protocol(models):
    prior = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    result = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if not same_result([model.snapshot() for model in models], prior["models"]):
        raise ValueError("frozen physical catalogue models changed")
    quality = all(result[key] for key in ("certificate_checks_passed", "effective_references_complete",
        "candidate_quality_gate_passed", "comparison_quality_gate_passed", "execution_checks_passed"))
    if result["decision"] != "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD" or not quality:
        raise ValueError("cadence diagnosis requires the quality-valid frozen confirmation HOLD")
    sources = {path: (ROOT / path).read_text(encoding="utf-8") for path in (*prior["execution_sources"], *EXTRA_SOURCES)}
    if any(sources[path] != value for path, value in prior["execution_sources"].items()):
        raise ValueError("frozen confirmation source changed")
    cells = []
    for node, lanes in prior["execution_order"]:
        with gzip.open(_cell_path(PRIOR_OUTPUT, node, lanes), "rt", encoding="utf-8") as stream:
            saved = json.load(stream)
        row = saved["cell"]
        roster = [(state["state_id"], tuple(state["queue"]))
                  for panel in prior["queue_panels"][str(lanes)] for state in panel["states"]]
        if (saved["registered_at_utc"] != prior["registered_at_utc"]
                or (row["node"], row["lanes"]) != (node, lanes)
                or [(state["state_id"], tuple(state["queue"])) for state in row["states"]] != roster):
            raise ValueError("frozen timing cell registration or roster changed")
        states = []
        key = f"{node}:{lanes}"
        for index in prior["timing_indices"][key]:
            old = row["states"][index]
            target = (node, lanes, tuple(old["queue"])) in TARGETS
            states.append({"state_id": old["state_id"], "queue": old["queue"], "target": target,
                "reference": old["reference"], "oracle": old["oracle"],
                "effective_reference_complete": old["effective_reference_complete"],
                "expected_results": old["first_results"],
                **({"prior_wall_ns": old["wall_ns"], "prior_cpu_ns": old["cpu_ns"]} if target else {})})
        cells.append({"node": node, "lanes": lanes, "states": states, "orders": prior["timing_orders"][key]})
    if len(cells) != 16 or sum(len(cell["states"]) for cell in cells) != 1058:
        raise ValueError("cadence replay must retain all sixteen cells and 1058 timed states")
    if sum(state["target"] for cell in cells for state in cell["states"]) != 2:
        raise ValueError("both fixed tail sentinels must occur once")
    return {
        "schema_version": 1, "experiment": "certificate_routed_cadence_diagnosis_v1",
        "models": prior["models"], "cells": cells, "execution_order": prior["execution_order"],
        "labels": list(LABELS), "repeats": prior["repeats"], "warmups_per_label": prior["warmups_per_label"],
        "cpu": min(os.sched_getaffinity(0)),
        **{key: prior[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
                                      "long_lp_budget", "control_integer_state_budget")},
        "state_ratio_limit": 1.05, "prior_registered_at_utc": prior["registered_at_utc"],
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)), "prior_decision": result["decision"],
        "prior_quality_passed": quality, "expected_selector_calls": 19140,
        "expected_timed_calls": 19044, "expected_warmup_calls": 96,
        "expected_implementation_calls": {"candidate": 6404, "probe": 6368, "long": 6368},
        "execution_sources": sources,
        "measurement": "One timing-only replay of the original16 cells and1058 timed states, preserving cell order, state orders and all six per-state role permutations. Two warmups perrole on each cell's first timed state, with original repeat-then-role order and immediate serialization/validation. Timed roles are adjacent before serialization/validation. At only the two fixed node007 R64 tail sentinels every role invokes the identical routed callable at144/256/8192; background roles retain candidate,probe,long implementations and budgets. Save every whole-call wall/CPU interval and cell-relative block start; no manual GC, instrumentation or waits.",
        "accounting": "19044 timed plus96 warmup selectors. Attribute work by actual implementation:6404 routed,6368 probe,6368 long; cumulative LP totals charged once and all spent integer states retained. Reuse frozen independent reference and oracle records; zero new screening, reference or oracle calls. Any mismatch is saved without replacement.",
        "runtime_policy": "Pin minimum available CPU before numerical imports, BLAS/OMP environment1 and normal automatic GC; record actual affinity and native thread counts. Reproduce workload ordering rather than forcing historical wall-clock gaps; report observed sentinel gaps.",
        "endpoint": "Only the two same-implementation sentinel states are diagnostic endpoints. Report fixed candidate/probe and candidate/long median ratios against unchanged1.05, symmetric max/min and CPU ratios. Background calls reconstruct cadence and do not become a new algorithm comparison. Preserve the prior TIMING_HOLD for every outcome. This is the final timing-diagnostic pass: no rerun, expansion or threshold adjustment based on its result.",
        "limitations": "One dependent six-repeat window at each of two previously inspected states, on one host and cached catalogues. Replaying timing workload omits original screens and references and cannot reproduce original host history or exact delays. Equality checks and saved block metadata add small between-block work. A threshold crossing demonstrates a possible same-implementation discrepancy under this replay; its absence does not establish universal five-percent resolution or a false-positive rate. No independent speedup, physical scaling or flow-time claim.",
    }


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered cadence protocol changed")


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    if selectors is None:
        from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
        from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
        from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
        selectors = {"candidate": select_certificate_routed_frame, "probe": select_resumable_probe_frame,
                     "long": select_strict_score_proof_canonical_completion_frame}
    from algorithm.experiments.resumable_probe_development_summary import _candidate_work
    from algorithm.experiments.strict_score_proof_timing_summary import _work as strict_work
    started, origin = perf_counter(), perf_counter_ns()
    lanes = cell["lanes"]
    row = {"node": model.node, "lanes": lanes, "registered_at_utc": protocol["registered_at_utc"],
           "runtime_start": runtime_provenance(), "states": []}
    references, expected_keys = [], []
    results = {label: [] for label in LABELS}
    for state in cell["states"]:
        reference = selection_from_record(state["reference"])
        expected = runtime.reference_key(reference, tuple(state["queue"]), len(model.options))
        if expected is None and state["oracle"] is not None:
            expected = oracle_key(state["oracle"], len(model.options))
        references.append(reference)
        expected_keys.append(expected)
        row["states"].append({key: state[key] for key in ("state_id", "queue", "target")})
        row["states"][-1].update(calls=[], blocks=[], first_results={}, mismatches=[],
                                metrics={label: runtime._metrics() for label in LABELS})

    def invoke(index, role):
        state = cell["states"][index]
        implementation = "candidate" if state["target"] else role
        args = (model.options, tuple(state["queue"]), lanes)
        budgets = ((protocol["long_lp_budget"], protocol["control_integer_state_budget"])
                   if implementation == "long" else (protocol["initial_lp_budget"],
                       protocol["lp_node_budget"], protocol["integer_state_budget"]))
        return selectors[implementation](*args, *budgets)

    def record(index, role, kind, repeat, result, measured):
        state, saved = cell["states"][index], row["states"][index]
        implementation = "candidate" if state["target"] else role
        literal = asdict(result)
        prep = literal["preprocessing"] if implementation == "long" else literal["final_lp"]["preprocessing"]
        runtime._record(saved["metrics"][role], selection_from_record(literal["selection"]), model,
            tuple(state["queue"]), lanes, literal["selection"]["lp_node_budget"], expected_keys[index],
            prep, references[index])
        matches = same_result(literal, state["expected_results"][implementation])
        pointer = {"kind": kind, "repeat": repeat, "role": role, "implementation": implementation}
        saved["calls"].append({**pointer, **measured, "matches_frozen": matches})
        saved["first_results"].setdefault(role, literal)
        if not matches:
            saved["mismatches"].append({**pointer, "result": literal})
        results[implementation].append(literal)

    def block(index, kind, repeat, order):
        outputs, measured = {}, {}
        row["states"][index]["blocks"].append({"kind": kind, "repeat": repeat, "order": order,
                                               "start_ns": perf_counter_ns() - origin})
        for role in order:
            cpu_start, wall_start = process_time_ns(), perf_counter_ns()
            outputs[role] = invoke(index, role)
            measured[role] = {"wall_ns": perf_counter_ns() - wall_start,
                              "cpu_ns": process_time_ns() - cpu_start}
        for role, result in outputs.items():
            record(index, role, kind, repeat, result, measured[role])

    row["runtime_before_warmup"] = runtime_provenance()
    for repeat in range(protocol["warmups_per_label"]):
        for role in LABELS:
            block(0, "warmup", repeat, [role])
    row["runtime_after_warmup"] = runtime_provenance()
    for repeat, state_order in enumerate(cell["orders"]["state_orders"]):
        for index in state_order:
            block(index, "timed", repeat, cell["orders"]["label_orders"][index][repeat])
        if progress:
            print(f"{model.node} R{lanes}: cadence timing {repeat + 1}/{protocol['repeats']} complete", flush=True)
    row["runtime_end"] = runtime_provenance()
    row["work"] = {label: (strict_work if label == "long" else _candidate_work)(values)
                   for label, values in results.items()}
    row["elapsed_s"] = perf_counter() - started
    return row


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
        with protocol_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, separators=(",", ":"), allow_nan=False)
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, *key).exists() for key in protocol["execution_order"]):
        raise FileExistsError("cadence output already exists; this diagnostic is not rerun")
    rows = []
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime)
        row = json.loads(json.dumps(row, separators=(",", ":"), allow_nan=False))
        path = _cell_path(args.output, cell["node"], cell["lanes"])
        with gzip.open(path, "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream,
                      separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.certificate_routed_cadence_diagnosis_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
