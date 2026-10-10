"""Serial paired whole-selector latency on the successful confirmation queues.

Only standard-library imports occur before runtime_setup pins one CPU and sets
BLAS environment variables. Four catalogues run on this one execution host.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from itertools import permutations
import json
import os
from pathlib import Path
import platform
import random
import sys
from time import perf_counter, perf_counter_ns, process_time_ns


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_timing_20260927.json"
CONFIRMATION_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_queue_confirmation_20260927.json"
CONFIRMATION_PROTOCOL = CONFIRMATION_OUTPUT.with_name(CONFIRMATION_OUTPUT.stem + "_protocol.json")
LABELS = ("old_shared", "fractional_shared", "integer_capacity")
REPEATS = 6
ORDER_SEED = 2026092709
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_integer_capacity_timing.py",
    "algorithm/experiments/aligned_hardware_integer_capacity_timing_summary.py",
)


def runtime_setup():
    """Configure the supported Linux host before importing numerical libraries."""
    if "numpy" in sys.modules or "scipy" in sys.modules:
        raise RuntimeError("timing requires a fresh process before NumPy/SciPy imports")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for name in THREAD_ENV:
        os.environ[name] = "1"
    gc.enable()
    from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
    return runtime


def runtime_provenance():
    import scipy
    from threadpoolctl import threadpool_info
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    return {
        "hostname": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "scipy": scipy.__version__,
        "cpu_model": next(line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                          if line.startswith("model name")),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
        "threadpool_info": threadpool_info(),
        "native_thread_count": int(next(line.split(":", 1)[1] for line in status.splitlines()
                                        if line.startswith("Threads:"))),
        "gc_enabled": gc.isenabled(), "gc_threshold": gc.get_threshold(),
    }


def state_roster(protocol, lanes):
    return [{"panel": panel["panel"], "seed": panel["seed"], "queues": queue}
            for panel in protocol["queue_panels"][str(lanes)] for queue in panel["queues"]]


def timing_orders(node, lanes, count):
    rng = random.Random(f"{ORDER_SEED}:{node}:{lanes}")
    permutation_indices = []
    for _ in range(count):
        order = list(range(REPEATS))
        rng.shuffle(order)
        permutation_indices.append(order)
    state_orders = []
    for _ in range(REPEATS):
        order = list(range(count))
        rng.shuffle(order)
        state_orders.append(order)
    return {"state_orders": state_orders, "permutation_indices": permutation_indices}


def require_confirmation_advance():
    result = json.loads(CONFIRMATION_OUTPUT.read_text(encoding="utf-8"))
    flags = ("certificate_checks_passed", "references_complete", "confirmation_action_gate_passed",
             "confirmation_search_cost_gate_passed")
    if result["decision"] != "INTEGER_CAPACITY_FRESH_QUEUE_CONFIRMATION_ADVANCE" or not all(result[key] for key in flags):
        raise ValueError("successful fixed-budget confirmation required before timing registration")


def build_protocol(models):
    previous = json.loads(CONFIRMATION_PROTOCOL.read_text(encoding="utf-8"))
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"successful confirmation source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("successful confirmation models changed")
    protocol = {
        "schema_version": 1, "experiment": "integer_capacity_whole_selector_timing_v1",
        "models": previous["models"], "lanes": previous["lanes"],
        "queue_panels": previous["queue_panels"], "panel_seeds": previous["panel_seeds"],
        "queue_source": str(CONFIRMATION_PROTOCOL.relative_to(ROOT)),
        "prior_confirmation": str(CONFIRMATION_OUTPUT.relative_to(ROOT)),
        "labels": LABELS, "repeats": REPEATS, "order_seed": ORDER_SEED,
        "cpu": min(os.sched_getaffinity(0)), "pooled_ratio_limit": 0.95, "cell_ratio_limit": 1.0,
        "budgets": previous["budgets"], "reference_budget": previous["reference_budget"],
        "selector_permutations": list(permutations(LABELS)),
        "execution_order": [[model.node, lanes] for model in models for lanes in previous["lanes"]],
        "calls": "one untimed full K4096 reference/state; six cold timed calls per selector/state at 36R; two untimed warmups per selector/cell; all twelve cells serial in one process; no escalation",
        "measurement": "perf_counter_ns surrounds each entire cold selector call, including preprocessing, capacity propagation, and returned object construction; process_time_ns secondary; no oracle, validation, or I/O inside interval; three methods adjacent then validated; GC enabled; retain every raw observation",
        "runtime_policy": "pin minimum available CPU before NumPy/SciPy imports; set OMP/OPENBLAS/MKL/BLIS/NUMEXPR_NUM_THREADS=1; record actual affinity, threadpools, and native task counts; no assertion of single-threaded HiGHS execution",
        "reference_policy": previous["oracle"],
        "validity": "references complete, all certificates valid, integer capacity exact and matching every reference on every repeat; all repeated dataclass results equal their first repeat; inexact controls retained",
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "p95_state_median": "descriptive only",
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Latency reuses the successful queue confirmation roster and fixed cached point models. All four catalogues run on one local host; virtual lanes are not new physical nodes. Six timing repeats and catalogue comparisons are paired, not independent queue samples. No new queue generalization or trajectory/flow-time claim.",
    }
    protocol["timing_orders"] = {
        f"{model.node}:{lanes}": timing_orders(model.node, lanes, len(state_roster(protocol, lanes)))
        for model in models for lanes in protocol["lanes"]}
    return json.loads(json.dumps(protocol))


def timing_cell(model, lanes, protocol, runtime, progress=True):
    started = perf_counter()
    budget, reference_budget = protocol["budgets"][str(lanes)], protocol["reference_budget"]
    selectors = dict(zip(LABELS, (
        runtime.select_shared_threshold_canonical_requirement_dominance_frame,
        runtime.select_fractional_shared_threshold_canonical_requirement_dominance_frame,
        runtime.select_integer_capacity_shared_threshold_frame), strict=True))
    states = [{**state, "wall_ns": {label: [None] * REPEATS for label in LABELS},
               "cpu_ns": {label: [None] * REPEATS for label in LABELS}, "search_work": {}}
              for state in state_roster(protocol, lanes)]
    correctness = {label: runtime._metrics() for label in ("reference", *LABELS)}
    for label in LABELS:
        correctness[label].update(deterministic_repeat_mismatch_count=0, first_repeat_mismatch=None)
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "option_count": len(model.options), "queue_count": len(states), "states": states,
           "correctness": correctness, "uncertified_reference_count": 0,
           "first_uncertified_reference": None, "warmup_calls_by_label": dict.fromkeys(LABELS, 0),
           "runtime_start": runtime_provenance()}
    frames = runtime.enumerate_frames(model.options, lanes) if lanes == 4 else None
    references, expected_keys = [], []
    for state in states:
        queue = tuple(state["queues"])
        reference = runtime.select_canonical_count_space_bnb_frame(model.options, queue, lanes, reference_budget)
        certified = runtime.reference_key(reference, queue, len(model.options))
        expected = runtime.oracle(frames, queue) if frames is not None else certified
        runtime._record(correctness["reference"], reference, model, queue, lanes, reference_budget, expected)
        if certified is None:
            row["uncertified_reference_count"] += 1
            if row["first_uncertified_reference"] is None:
                row["first_uncertified_reference"] = {"queues": queue, "selection": asdict(reference)}
        references.append(reference)
        expected_keys.append(expected)
    row["runtime_before_warmup"] = runtime_provenance()
    for _ in range(2):
        for label, selector in selectors.items():
            selector(model.options, tuple(states[0]["queues"]), lanes, budget)
            row["warmup_calls_by_label"][label] += 1
    row["runtime_after_warmup"] = runtime_provenance()
    first_results = [{} for _ in states]
    orders = protocol["timing_orders"][f"{model.node}:{lanes}"]
    for repeat, state_order in enumerate(orders["state_orders"]):
        for index in state_order:
            state, queue = states[index], tuple(states[index]["queues"])
            permutation = orders["permutation_indices"][index][repeat]
            labels = protocol["selector_permutations"][permutation]
            results = {}
            for label in labels:
                selector = selectors[label]
                cpu_start = process_time_ns()
                wall_start = perf_counter_ns()
                result = selector(model.options, queue, lanes, budget)
                wall_elapsed = perf_counter_ns() - wall_start
                cpu_elapsed = process_time_ns() - cpu_start
                results[label] = result
                state["wall_ns"][label][repeat] = wall_elapsed
                state["cpu_ns"][label][repeat] = cpu_elapsed
            # Check the entire adjacent three-method block only after it ends.
            for label, result in results.items():
                selected, literal = result.selection, asdict(result)
                runtime._record(correctness[label], selected, model, queue, lanes, budget,
                                expected_keys[index], asdict(result.preprocessing), references[index])
                if repeat == 0:
                    first_results[index][label] = literal
                    state["search_work"][label] = {key: getattr(selected, key) for key in protocol["search_counters"]}
                    if label == "integer_capacity":
                        state["capacity_propagation"] = asdict(result.capacity_propagation)
                elif literal != first_results[index][label]:
                    correctness[label]["deterministic_repeat_mismatch_count"] += 1
                    if correctness[label]["first_repeat_mismatch"] is None:
                        correctness[label]["first_repeat_mismatch"] = {"queues": queue, "repeat": repeat,
                            "first_result": first_results[index][label], "current_result": literal}
        if progress:
            print(f"{model.node} R{lanes}: timed sweep {repeat + 1}/{REPEATS} complete", flush=True)
    row["runtime_end"] = runtime_provenance()
    row["operational_elapsed_s"] = perf_counter() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true", help="all cells serial in one fresh process")
    mode.add_argument("--merge-cells", action="store_true")
    args = parser.parse_args()
    runtime = runtime_setup()
    models = runtime.load_catalogues()
    expected = build_protocol(models)
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        require_confirmation_advance()
        expected["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, expected)
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources, models, queues, or timing design changed")
    if args.output.exists():
        raise FileExistsError(args.output)
    cells = [(next(model for model in models if model.node == node), lanes)
             for node, lanes in protocol["execution_order"]]
    if args.run:
        for model, lanes in cells:
            path = runtime._cell_path(args.output, model.node, lanes)
            if path.exists():
                raise FileExistsError(path)
    rows = []
    for model, lanes in cells:
        path = runtime._cell_path(args.output, model.node, lanes)
        if args.merge_cells:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["protocol"] != protocol:
                raise ValueError("cell protocol differs from registered protocol")
            row = saved["cell"]
        else:
            row = timing_cell(model, lanes, protocol, runtime)
            runtime._write_new(path, {"protocol": protocol, "cell": row})
        rows.append(row)
    from algorithm.experiments.aligned_hardware_integer_capacity_timing_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count")}))


if __name__ == "__main__":
    main()
