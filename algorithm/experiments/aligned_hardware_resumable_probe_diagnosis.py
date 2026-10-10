"""Instrumented development attribution on seven already-inspected queue states."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import gzip
import json
import os
from pathlib import Path
import random
from statistics import median
from time import perf_counter_ns, process_time_ns
from unittest.mock import patch

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_resumable_probe_diagnosis_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_resumable_probe_confirmation_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
SOURCE = "algorithm/experiments/aligned_hardware_resumable_probe_diagnosis.py"
LABELS, CONDITIONS = ("candidate", "legacy"), ("enabled", "disabled")
REPEATS, WARMUPS, ORDER_SEED = 12, 2, 2026092798
CASES = (
    ("failed_39", "gpu_a", 64, (39, 3, 313)),
    ("failed_88", "gpu_a", 64, (88, 8, 23)),
    ("probe_control", "gpu_a", 64, (81, 31, 48)),
    ("resume_control", "gpu_a", 64, (153, 147, 290)),
    ("score_proof_control", "gpu_a", 32, (85, 86, 146)),
    ("ordinary_tail", "gpu_b", 32, (96, 15, 13)),
    ("fresh_worst", "gpu_b", 64, (184, 78, 235)),
)


def timing_orders(count=7, repeats=REPEATS, seed=ORDER_SEED):
    if repeats % 2:
        raise ValueError("balanced timing requires an even repeat count")
    rng = random.Random(seed)
    def balanced(values):
        orders = [list(values), list(reversed(values))] * (repeats // 2)
        rng.shuffle(orders)
        return orders
    conditions = [balanced(CONDITIONS) for _ in range(count)]
    labels = [{condition: balanced(LABELS) for condition in CONDITIONS} for _ in range(count)]
    states = [rng.sample(range(count), count) for _ in range(repeats)]
    return {"state_orders": states, "condition_orders": conditions, "label_orders": labels}


def build_protocol(models):
    prior = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    result = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    for path, literal in prior["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != literal:
            raise ValueError(f"frozen source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != prior["models"]:
        raise ValueError("frozen physical catalogues changed")
    states, cells = [], {}
    for name, node, lanes, queue in CASES:
        key = (node, lanes)
        if key not in cells:
            path = PRIOR_OUTPUT.with_name(f"{PRIOR_OUTPUT.stem}_cell_{node}_r{lanes}.json.gz")
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                saved = json.load(stream)
            if saved["registered_at_utc"] != prior["registered_at_utc"]:
                raise ValueError("prior cell registration changed")
            cells[key] = saved["cell"]
        matches = [state for state in cells[key]["states"] if tuple(state["queue"]) == queue]
        if len(matches) != 1:
            raise ValueError("diagnostic state must have one frozen screen record")
        old = matches[0]
        states.append({"name": name, "node": node, "lanes": lanes, "queue": queue,
            "state_id": old["state_id"], "expected_results": old["screen_results"],
            "screen_metrics": old["screen_metrics"], "reference": old["reference"],
            "reference_metrics": old["reference_metrics"], "oracle": old["oracle"],
            "effective_reference_complete": old["effective_reference_complete"],
            "effective_reference_source": old["effective_reference_source"]})
    protocol = {"schema_version": 1, "experiment": "resumable_probe_development_diagnosis_v1",
        "models": prior["models"], "states": states, "labels": LABELS, "conditions": CONDITIONS,
        "repeats": REPEATS, "warmups_per_label_condition_state": WARMUPS, "order_seed": ORDER_SEED,
        "cpu": min(os.sched_getaffinity(0)), "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "control_integer_state_budget": 100000,
        "prior_registered_at_utc": prior["registered_at_utc"], "prior_decision": result["decision"],
        "prior_quality_passed": all(result[key] for key in ("effective_references_complete",
            "candidate_quality_gate_passed", "comparison_quality_gate_passed", "execution_checks_passed")),
        "timing_orders": timing_orders(), "warmup_orders": timing_orders(repeats=WARMUPS, seed=ORDER_SEED + 1),
        "execution_sources": {**prior["execution_sources"], SOURCE: (ROOT / SOURCE).read_text(encoding="utf-8")},
        "measurement": "Every adjacent two-arm block receives one separately charged gc.collect. Set automatic GC enabled/disabled for the block and restore its original setting. Patches are installed before timing. Whole-call and primary phase wall/process CPU are inclusive; linprog and probe subphases are nested diagnostics, never added twice. Serialize and compare complete results only after both calls. Record all 336 timed and 56 warmup calls and every mismatch.",
        "limitations": "Seven selected previously inspected states; instrumented development attribution, not repeated confirmation. Explicit collection and disabled GC change runtime conditions and their cost is excluded from selectors but separately charged. Phase instrumentation adds overhead; residual includes wrapper construction, cleanup and measurement bookkeeping, not pure teardown. Initial LP less nested linprog is a remainder, not a strict Python profile. Prior independent references are reused with zero new reference/oracle calls. No ADVANCE gate or claim that the prior TIMING_HOLD is overturned."}
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered diagnostic protocol changed")


class Instrumentation:
    """Patch once outside intervals; primary phases partition, nested phases do not."""
    def __enter__(self):
        import algorithm.resumable_probe_frame_candidates as candidate
        import algorithm.resumable_integer_capacity_frame_candidates as resumable
        import algorithm.strict_score_proof_canonical_completion_frame_candidates as legacy
        import algorithm.integer_capacity_shared_threshold_frame_candidates as original
        self.active, self.phase_stack, self.gc_started, self.stack = None, [], {}, ExitStack()
        targets = [(resumable.IntegerCapacitySearch, "advance_to", "advance", True),
            (candidate, "_probe", "probe", True),
            (legacy, "select_integer_capacity_shared_threshold_frame", "initial_lp", True),
            (candidate, "_prove_no_strict_improvement", "proof", False),
            (candidate, "_complete_canonical", "completion", False),
            (legacy, "_prove_no_strict_improvement", "proof", True),
            (legacy, "_complete_canonical", "completion", True),
            (resumable, "linprog", "linprog", False), (original, "linprog", "linprog", False)]
        try:
            for owner, attr, name, primary in targets:
                self.stack.enter_context(patch.object(owner, attr, self._wrap(getattr(owner, attr), name, primary)))
            gc.callbacks.append(self._gc_callback)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *error):
        gc.callbacks.remove(self._gc_callback)
        self.active = None
        return self.stack.__exit__(*error)

    def _wrap(self, function, name, primary):
        def measured(*args, **kwargs):
            if self.active is None:
                return function(*args, **kwargs)
            phase = ("initial_lp" if args[0]._budget is None else "resume_lp") if name == "advance" else name
            self.phase_stack.append(phase)
            cpu, wall = process_time_ns(), perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                wall, cpu = perf_counter_ns() - wall, process_time_ns() - cpu
                self.phase_stack.pop()
                entry = self.active["phases"].setdefault(phase, {"wall_ns": 0, "cpu_ns": 0, "calls": 0})
                entry["wall_ns"] += wall
                entry["cpu_ns"] += cpu
                entry["calls"] += 1
                if name == "linprog" and self.phase_stack:
                    nested = self.active["phases"].setdefault(self.phase_stack[-1] + "_linprog",
                                                            {"wall_ns": 0, "cpu_ns": 0, "calls": 0})
                    nested["wall_ns"] += wall
                    nested["cpu_ns"] += cpu
                    nested["calls"] += 1
                if primary:
                    self.active["primary_phase_wall_ns"] += wall
                    self.active["primary_phase_cpu_ns"] += cpu
        return measured

    def _gc_callback(self, event, info):
        if self.active is None:
            return
        generation = info["generation"]
        if event == "start":
            self.gc_started[generation] = (perf_counter_ns(), process_time_ns(), tuple(self.phase_stack))
        else:
            wall, cpu, phases = self.gc_started.pop(generation)
            self.active["gc_events"].append({"generation": generation,
                "wall_ns": perf_counter_ns() - wall, "cpu_ns": process_time_ns() - cpu,
                "phase": phases[-1] if phases else "residual", "phase_stack": phases,
                "collected": info["collected"], "uncollectable": info["uncollectable"]})

    def capture(self, label, function):
        measurement = {"label": label, "phases": {}, "gc_events": [],
                       "primary_phase_wall_ns": 0, "primary_phase_cpu_ns": 0}
        self.active = measurement
        cpu, wall = process_time_ns(), perf_counter_ns()
        try:
            result = function()
        finally:
            measurement["wall_ns"], measurement["cpu_ns"] = perf_counter_ns() - wall, process_time_ns() - cpu
            self.active = None
        for clock in ("wall_ns", "cpu_ns"):
            measurement["residual_" + clock] = measurement[clock] - measurement["primary_phase_" + clock]
        return result, measurement


def run_pair(instrument, condition, order, selectors):
    original_gc = gc.isenabled()
    outputs, measurements = {}, {}
    try:
        (gc.enable if condition == "enabled" else gc.disable)()
        cpu, wall = process_time_ns(), perf_counter_ns()
        collected = gc.collect()
        collection = {"wall_ns": perf_counter_ns() - wall, "cpu_ns": process_time_ns() - cpu,
                      "collected": collected}
        for label in order:
            outputs[label], measurements[label] = instrument.capture(label, selectors[label])
    finally:
        (gc.enable if original_gc else gc.disable)()
    return outputs, {"condition": condition, "order": order, "collection": collection, "measurements": measurements}


def validate_result(literal, expected, reference_complete, metrics):
    return bool(reference_complete and literal == expected and all(literal["selection"][key]
        for key in ("exact", "score_exact", "canonical_exact")) and not metrics["failures"]
        and not metrics["score_mismatch_count"] and not metrics["canonical_action_mismatch_count"])


def run_diagnosis(protocol, models, progress=True):
    from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
    from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
    from algorithm.experiments.resumable_probe_development_summary import _candidate_work, _strict_work
    rows = [{"name": state["name"], "calls": [], "pairs": [], "first_results": {}, "mismatches": []}
            for state in protocol["states"]]
    results = {label: [] for label in LABELS}
    record = {"runtime_start": runtime_provenance(), "states": rows}
    with Instrumentation() as instrument:
        for kind, count, design in (("warmup", protocol["warmups_per_label_condition_state"], protocol["warmup_orders"]),
                                    ("timed", protocol["repeats"], protocol["timing_orders"])):
            for repeat in range(count):
                for index in design["state_orders"][repeat]:
                    state, row = protocol["states"][index], rows[index]
                    model = next(model for model in models if model.node == state["node"])
                    args = model.options, tuple(state["queue"]), state["lanes"]
                    selectors = {"candidate": lambda: select_resumable_probe_frame(*args, 144, 256, 8192),
                                 "legacy": lambda: select_strict_score_proof_canonical_completion_frame(*args, 144, 100000)}
                    for condition in design["condition_orders"][index][repeat]:
                        outputs, pair = run_pair(instrument, condition, design["label_orders"][index][condition][repeat], selectors)
                        row["pairs"].append({"kind": kind, "repeat": repeat, **pair})
                        for label, result in outputs.items():
                            literal = json.loads(json.dumps(asdict(result)))
                            valid = validate_result(literal, state["expected_results"][label],
                                state["effective_reference_complete"], state["screen_metrics"][label])
                            pointer = {"kind": kind, "repeat": repeat, "condition": condition, "label": label}
                            row["calls"].append({**pointer, **pair["measurements"][label], "matches_frozen": valid})
                            row["first_results"].setdefault(label, literal)
                            if not valid:
                                row["mismatches"].append({**pointer, "result": literal})
                            results[label].append(literal)
                if progress:
                    print(f"diagnosis {kind} {repeat + 1}/{count} complete", flush=True)
    record["runtime_end"] = runtime_provenance()
    record["work"] = {"candidate": _candidate_work(results["candidate"]), "legacy": _strict_work(results["legacy"])}
    return record


def summarize(protocol, records):
    summaries = []
    for state, row in zip(protocol["states"], records["states"], strict=True):
        conditions = {}
        for condition in CONDITIONS:
            arms = {}
            for label in LABELS:
                calls = [call for call in row["calls"] if call["kind"] == "timed" and call["condition"] == condition and call["label"] == label]
                names = sorted({name for call in calls for name in call["phases"]})
                arms[label] = {"calls": len(calls), **{key: median(call[key] for call in calls) for key in
                    ("wall_ns", "cpu_ns", "residual_wall_ns", "residual_cpu_ns")},
                    "phases": {name: {clock: median(call["phases"].get(name, {}).get(clock, 0) for call in calls)
                        for clock in ("wall_ns", "cpu_ns")} for name in names},
                    "initial_lp_less_linprog_wall_ns": median(call["phases"].get("initial_lp", {}).get("wall_ns", 0)
                        - call["phases"].get("initial_lp_linprog", {}).get("wall_ns", 0) for call in calls),
                    "gc_events": sum(len(call["gc_events"]) for call in calls),
                    "gc_pause_wall_ns": sum(event["wall_ns"] for call in calls for event in call["gc_events"])}
                arms[label]["gc_pause_fraction_of_whole"] = arms[label]["gc_pause_wall_ns"] / sum(call["wall_ns"] for call in calls)
                for phase in arms[label]["phases"].values():
                    phase.update(wall_ms=phase["wall_ns"] / 1e6, cpu_ms=phase["cpu_ns"] / 1e6)
            conditions[condition] = {"arms": arms, "candidate_legacy_wall_ratio": arms["candidate"]["wall_ns"] / arms["legacy"]["wall_ns"],
                                    "candidate_legacy_cpu_ratio": arms["candidate"]["cpu_ns"] / arms["legacy"]["cpu_ns"]}
            for phase in ("initial_lp", "linprog"):
                denominator = arms["legacy"]["phases"].get(phase, {}).get("wall_ns", 0)
                conditions[condition][f"candidate_legacy_{phase}_wall_ratio"] = (
                    arms["candidate"]["phases"].get(phase, {}).get("wall_ns", 0) / denominator if denominator else None)
        summaries.append({key: state[key] for key in ("name", "node", "lanes", "queue")} | {"conditions": conditions})
    calls = [call for row in records["states"] for call in row["calls"]]
    collections = [pair["collection"] for row in records["states"] for pair in row["pairs"]]
    valid = (protocol["prior_quality_passed"] and all(call["matches_frozen"] for call in calls)
             and all(state["effective_reference_complete"] for state in protocol["states"])
             and records["runtime_start"]["cpu_affinity"] == records["runtime_end"]["cpu_affinity"] == [protocol["cpu"]]
             and len(calls) == len(protocol["states"]) * len(CONDITIONS) * len(LABELS) * (protocol["repeats"] + protocol["warmups_per_label_condition_state"]))
    return {"decision": "DEVELOPMENT_DIAGNOSIS_COMPLETE" if valid else "INVALID_EXECUTION",
        "execution_checks_passed": valid, "executed_selector_calls": len(calls),
        "timed_calls": sum(call["kind"] == "timed" for call in calls), "warmup_calls": sum(call["kind"] == "warmup" for call in calls),
        "executed_reference_calls": 0, "executed_oracle_calls": 0, "borrowed_reference_count": len(protocol["states"]),
        "explicit_collection": {"calls": len(collections), **{key: sum(item[key] for item in collections) for key in ("wall_ns", "cpu_ns", "collected")}},
        "states": summaries, "work": records["work"], "limitations": protocol["limitations"]}


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
    records = run_diagnosis(protocol, models)
    with gzip.open(raw_path, "xt", encoding="utf-8") as target:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], **records}, target, separators=(",", ":"), allow_nan=False)
    summary = summarize(protocol, records)
    summary.update(protocol_path=str(protocol_path), raw_path=str(raw_path))
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
