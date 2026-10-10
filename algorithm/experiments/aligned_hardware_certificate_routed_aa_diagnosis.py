"""Frozen same-implementation timing windows and separate phase attribution."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from itertools import permutations
import json
import os
from pathlib import Path
import random
from time import perf_counter, perf_counter_ns, process_time_ns
from unittest.mock import patch

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_resumable_probe_diagnosis import Instrumentation, validate_result

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_aa_diagnosis_20260928.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_confirmation_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
LABELS, PHASE_LABELS = ("a", "b", "c"), ("candidate", "probe")
WINDOWS, REPEATS, WARMUPS, PHASE_REPEATS, PHASE_WARMUPS = 12, 6, 2, 6, 2
ORDER_SEED = 2026092891
CASES = (("tail_88", "node007", 64, (88, 19, 118)),
         ("tail_184", "node007", 64, (184, 78, 235)))
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_certificate_routed_aa_diagnosis.py",
    "algorithm/experiments/certificate_routed_aa_diagnosis_summary.py",
    "algorithm/experiments/aligned_hardware_resumable_probe_diagnosis.py",
)


def balanced_orders(count, labels, repeats, seed):
    """Each state visits each label permutation equally often."""
    possibilities = list(permutations(labels))
    if repeats % len(possibilities):
        raise ValueError("repeats must cover every label permutation equally")
    rng = random.Random(seed)
    orders = []
    for _ in range(count):
        values = possibilities * (repeats // len(possibilities))
        rng.shuffle(values)
        orders.append([list(value) for value in values])
    return {"state_orders": [rng.sample(range(count), count) for _ in range(repeats)],
            "label_orders": orders}


def aa_orders(count=2, windows=WINDOWS, repeats=REPEATS, seed=ORDER_SEED):
    designs = [balanced_orders(count, LABELS, repeats, seed + window) for window in range(windows)]
    return {"state_orders": [design["state_orders"] for design in designs],
            "label_orders": [[design["label_orders"][index] for design in designs] for index in range(count)]}


def warmup_orders(count, labels, repeats, seed):
    rng = random.Random(seed)
    return {"state_orders": [rng.sample(range(count), count) for _ in range(repeats)],
            "label_orders": [[rng.sample(labels, len(labels)) for _ in range(repeats)] for _ in range(count)]}


def build_protocol(models):
    prior = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    result = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if json.loads(json.dumps([model.snapshot() for model in models])) != prior["models"]:
        raise ValueError("frozen physical catalogue models changed")
    if result["decision"] != "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD":
        raise ValueError("this diagnosis preserves the completed routed confirmation HOLD")
    # Retain the existing execution source map and add only this diagnosis.
    sources = {path: (ROOT / path).read_text(encoding="utf-8")
               for path in (*prior["execution_sources"], *EXTRA_SOURCES)}
    if any(sources[path] != literal for path, literal in prior["execution_sources"].items()):
        raise ValueError("the diagnostic selectors must retain their frozen confirmation sources")
    node, lanes = CASES[0][1:3]
    path = PRIOR_OUTPUT.with_name(f"{PRIOR_OUTPUT.stem}_cell_{node}_r{lanes}.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    if saved["registered_at_utc"] != prior["registered_at_utc"]:
        raise ValueError("prior cell registration changed")
    states = []
    for name, node, lanes, queue in CASES:
        matches = [state for state in saved["cell"]["states"] if tuple(state["queue"]) == queue]
        if len(matches) != 1:
            raise ValueError("diagnostic state requires one frozen screen record")
        old = matches[0]
        states.append({"name": name, "node": node, "lanes": lanes, "queue": queue,
            "state_id": old["state_id"],
            "expected_results": {label: old["screen_results"][label] for label in PHASE_LABELS},
            "screen_metrics": {label: old["screen_metrics"][label] for label in PHASE_LABELS},
            **{key: old[key] for key in ("reference", "reference_metrics", "oracle",
                                       "effective_reference_complete", "effective_reference_source")}})
    protocol = {"schema_version": 1, "experiment": "certificate_routed_aa_development_diagnosis_v1",
        "models": prior["models"], "states": states, "labels": LABELS, "phase_labels": PHASE_LABELS,
        "windows": WINDOWS, "repeats": REPEATS, "warmups": WARMUPS,
        "phase_repeats": PHASE_REPEATS, "phase_warmups": PHASE_WARMUPS, "order_seed": ORDER_SEED,
        "cpu": min(os.sched_getaffinity(0)), "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "state_ratio_limit": 1.05,
        "prior_registered_at_utc": prior["registered_at_utc"], "prior_decision": result["decision"],
        "prior_quality_passed": all(result[key] for key in ("certificate_checks_passed",
            "effective_references_complete", "candidate_quality_gate_passed",
            "comparison_quality_gate_passed", "execution_checks_passed")),
        "aa_orders": aa_orders(),
        "aa_warmup_orders": warmup_orders(2, LABELS, WARMUPS, ORDER_SEED + 100),
        "phase_orders": balanced_orders(2, PHASE_LABELS, PHASE_REPEATS, ORDER_SEED + 200),
        "phase_warmup_orders": warmup_orders(2, PHASE_LABELS, PHASE_WARMUPS, ORDER_SEED + 300),
        "execution_sources": sources,
        "measurement": "First run two warmup sweeps then 12 consecutive six-repeat A/A windows, with no window-level rewarming. All a,b,c labels use the identical routed callable and parameters. Each state/window visits all six label permutations once; shuffle states each sweep. Fix a as pseudo-candidate and b,c as controls; separately count a/b and a/c median-wall ratios above 1.05. Also report symmetric max/min spread. Afterwards install phase instrumentation, perform two warmup sweeps and six two-arm sweeps with candidate/probe order balanced three each way. Whole-call wall and CPU times include the returned result. Serialize and validate only after every adjacent block. No manual collection; automatic GC remains enabled throughout. Retain every call and mismatch.",
        "accounting": "432 A/A timed plus12 A/A warmup calls,24 phase timed plus8 phase warmup calls:476 selectors, candidate460 and probe16. Charge cumulative final LP and spent integer work once for every actual call. Reuse two frozen independent reference records, with zero new reference or oracle calls.",
        "runtime_policy": "One serial process pinned before numerical imports; same normal GC and BLAS environment as confirmation. Record actual runtime affinity, thread pools and thread counts. The uninstrumented A/A stage precedes the separate observer-affected phase stage.",
        "limitations": "Two selected previously inspected tail states,24 dependent state-windows on one host. The two-state short cadence has much shorter between-round intervals than the original1058-state confirmation workload; it cannot estimate that workload's false-positive rate. This descriptive diagnostic neither estimates an independent false-positive rate nor proves universal five-percent resolution, including when there are zero threshold crossings. Instrumented phase results are not confirmation timings; LP less linprog and whole-call less primary phases are remainders. The previous TIMING_HOLD remains unchanged regardless of this outcome; no reruns or retuning to pass it."}
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered A/A diagnostic protocol changed")


class RoutedInstrumentation(Instrumentation):
    """The routed module holds its own imported probe callable."""
    def __enter__(self):
        super().__enter__()
        import algorithm.certificate_routed_frame_candidates as routed
        self.stack.enter_context(patch.object(routed, "_probe", self._wrap(routed._probe, "probe", True)))
        return self


def aa_selectors(selector):
    return {label: selector for label in LABELS}


def run_block(order, selectors, instrument=None):
    """Time adjacent invocations without serializing, validating or forcing GC."""
    outputs, measurements = {}, {}
    for label in order:
        if instrument is None:
            cpu, wall = process_time_ns(), perf_counter_ns()
            outputs[label] = selectors[label]()
            measurements[label] = {"wall_ns": perf_counter_ns() - wall,
                                   "cpu_ns": process_time_ns() - cpu}
        else:
            outputs[label], measurements[label] = instrument.capture(label, selectors[label])
    return outputs, measurements


def run_diagnosis(protocol, models, progress=True):
    from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
    from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
    from algorithm.experiments.resumable_probe_development_summary import _candidate_work
    started = perf_counter()
    rows = [{"name": state["name"], "calls": [], "blocks": [], "first_results": {}, "mismatches": []}
            for state in protocol["states"]]
    results = {label: [] for label in PHASE_LABELS}
    record = {"runtime_start": runtime_provenance(), "states": rows}

    def sweep(stage, kind, window, repeat, state_order, label_orders, instrument=None):
        for index in state_order:
            state, row = protocol["states"][index], rows[index]
            model = next(model for model in models if model.node == state["node"])
            args = (model.options, tuple(state["queue"]), state["lanes"],
                    protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
            candidate = lambda: select_certificate_routed_frame(*args)
            selectors = aa_selectors(candidate) if stage == "aa" else {
                "candidate": candidate, "probe": lambda: select_resumable_probe_frame(*args)}
            order = label_orders[index][repeat]
            outputs, measurements = run_block(order, selectors, instrument)
            pointer = {"stage": stage, "kind": kind, "window": window, "repeat": repeat}
            row["blocks"].append({**pointer, "order": order})
            for label, result in outputs.items():
                implementation = "candidate" if stage == "aa" else label
                literal = json.loads(json.dumps(asdict(result)))
                valid = validate_result(literal, state["expected_results"][implementation],
                                        state["effective_reference_complete"], state["screen_metrics"][implementation])
                call_pointer = {**pointer, "label": label}
                row["calls"].append({**call_pointer, **measurements[label], "matches_frozen": valid})
                row["first_results"].setdefault(implementation, literal)
                if not valid:
                    row["mismatches"].append({**call_pointer, "implementation": implementation, "result": literal})
                results[implementation].append(literal)

    design = protocol["aa_warmup_orders"]
    for repeat in range(protocol["warmups"]):
        sweep("aa", "warmup", -1, repeat, design["state_orders"][repeat], design["label_orders"])
    for window in range(protocol["windows"]):
        design = protocol["aa_orders"]
        labels = [state[window] for state in design["label_orders"]]
        for repeat in range(protocol["repeats"]):
            sweep("aa", "timed", window, repeat, design["state_orders"][window][repeat], labels)
        if progress:
            print(f"A/A window {window + 1}/{protocol['windows']} complete", flush=True)
    with RoutedInstrumentation() as instrument:
        for kind, count, design in (("warmup", protocol["phase_warmups"], protocol["phase_warmup_orders"]),
                                    ("timed", protocol["phase_repeats"], protocol["phase_orders"])):
            for repeat in range(count):
                sweep("phase", kind, -1, repeat, design["state_orders"][repeat], design["label_orders"], instrument)
            if progress:
                print(f"phase {kind} {count}/{count} complete", flush=True)
    record["runtime_end"] = runtime_provenance()
    record["work"] = {label: _candidate_work(values) for label, values in results.items()}
    record["elapsed_s"] = perf_counter() - started
    return record


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
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_protocol(protocol, models)
    raw_path = args.output.with_name(args.output.stem + "_raw.json.gz")
    if args.output.exists() or raw_path.exists():
        raise FileExistsError("diagnostic output already exists")
    record = run_diagnosis(protocol, models)
    with gzip.open(raw_path, "xt", encoding="utf-8") as target:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], **record}, target,
                  separators=(",", ":"), allow_nan=False)
    from algorithm.experiments.certificate_routed_aa_diagnosis_summary import summarize
    summary = summarize(protocol, record)
    summary.update(protocol_path=str(protocol_path), raw_path=str(raw_path))
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
