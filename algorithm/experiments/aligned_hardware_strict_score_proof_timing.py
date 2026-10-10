"""Paired whole-call timing of early integer proof versus longer LP search."""
from __future__ import annotations

import argparse
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
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_strict_score_proof_timing_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_strict_score_proof_confirmation_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
LABELS = ("candidate", "control")
LP_BUDGETS = {"candidate": 144, "control": 4096}
INTEGER_STATE_BUDGET = 100_000
REPEATS = 6
WARMUPS_PER_LABEL = 2
ORDER_SEED = 2026092781
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_strict_score_proof_timing.py",
    "algorithm/experiments/strict_score_proof_timing_summary.py",
)


def state_roster(protocol, lanes):
    return [{"state_id": state["state_id"], "panel": panel["panel"], "cohort": panel["cohort"],
             "seed": panel.get("seed"), "queue": state["queue"]}
            for panel in protocol["queue_panels"][str(lanes)] for state in panel["states"]]


def timing_orders(node, lanes, count, repeats=REPEATS):
    if repeats < 2 or repeats % 2:
        raise ValueError("balanced paired timing requires positive even repeats")
    rng = random.Random(f"{ORDER_SEED}:{node}:{lanes}")
    label_orders = []
    for _ in range(count):
        order = [0, 1] * (repeats // 2)
        rng.shuffle(order)
        label_orders.append([list(LABELS if value == 0 else reversed(LABELS)) for value in order])
    state_orders = []
    for _ in range(repeats):
        order = list(range(count))
        rng.shuffle(order)
        state_orders.append(order)
    return {"state_orders": state_orders, "label_orders": label_orders}


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if (prior["decision"] != "STRICT_SCORE_PROOF_CONFIRMATION_ADVANCE"
            or not all(prior[key] for key in ("certificate_checks_passed", "effective_references_complete",
                                              "fresh_confirmation_gate_passed", "regression_gate_passed"))):
        raise ValueError("successful shared-budget confirmation required before timing")
    if (previous["budget"], previous["integer_state_budget"]) != (LP_BUDGETS["candidate"], INTEGER_STATE_BUDGET):
        raise ValueError("retain the confirmed candidate budget pair")
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen confirmation source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("confirmed catalogue models changed")
    protocol = {
        "schema_version": 1, "experiment": "strict_score_proof_equal_quality_timing_v1",
        "models": previous["models"], "lanes": previous["lanes"],
        "queue_panels": previous["queue_panels"], "labels": LABELS,
        "lp_budgets": LP_BUDGETS, "integer_state_budget": INTEGER_STATE_BUDGET,
        "reference_budget": previous["reference_budget"], "oracle_state_budget": previous["oracle_state_budget"],
        "repeats": REPEATS, "warmups_per_label": WARMUPS_PER_LABEL, "order_seed": ORDER_SEED,
        "cpu": min(os.sched_getaffinity(0)), "pooled_ratio_limit": 0.95, "cell_ratio_limit": 1.0,
        "execution_order": previous["execution_order"],
        "prior_registered_at_utc": previous["registered_at_utc"],
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)), "prior_decision": prior["decision"],
        "comparison": "Both arms call the same frozen strict-score-proof selector. Candidate LP cap144 and control LP cap4096 each have the same shared100000 integer-state cap. This isolates when to switch from native integer-capacity LP search to integer proof/completion. Ordinary already-closed paths coincide; no claim of comparison against every exact method.",
        "calls": "Reuse all4552 frozen full references and5 conditional oracle records. Execute six cold calls per arm per state, plus two untimed warmups per arm per node/R cell at its first registered state. No new references, oracle calls, warm starts or queue-specific caches. Baseline/intermediate certificates remain embedded views, not extra calls. Keep all calls after honest nonclosure; no cap changes or sample replacement.",
        "measurement": "perf_counter_ns surrounds the entire selector including LP, preprocessing, strict proof, canonical search and result construction; process_time_ns secondary. No validation, serialization, reference lookup or I/O inside either interval. Validate both outputs only after the adjacent pair. Per state, three AB and three BA orders are shuffled before timing; shuffle state order independently each sweep. Retain all observations without trimming; summarize each state's six repeats by median.",
        "runtime_policy": "Pin minimum available CPU before importing NumPy/SciPy; BLAS/OMP environment1, GC enabled. Save actual CPU affinity, library pools and native thread count; do not assume HiGHS has one native thread. Run all cells serially in one process on this host.",
        "reference_policy": "Use only prior independently generated full-reference or conditional-oracle keys. Retain original reference nonclosure. Candidate first result must reproduce its confirmed whole result; all subsequent whole results per arm must match their first repeat. No timed arm supplies its own comparison reference.",
        "quality_gate": "Complete roster, all wall/CPU measurements, budgets/accounting valid, saved effective references complete, both arms and every warmup canonical exact and exactly reference-matching, deterministic repeated results, candidate confirmation reproduction and preserved affinity. Honest nonclosure in either arm is QUALITY_HOLD; invalid certificates/accounting or unstable paths invalidate timing. No incomplete arm can establish an equal-quality speed gain.",
        "timing_gate": "Report fresh confirmation and regression cohorts separately. Within each, require ratio of sums of per-state median walltimes candidate/control<=0.95 and every nonempty node/R cohort-cell ratio<=1.0. Overall timing ADVANCE requires both cohort gates plus equal-quality validation. Preserve per-state slowdowns, p95/p99/max tails and the separately reported14 prior integer-trigger states; never select a favorable post-timing subset.",
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Local descriptive timing reuses the confirmed roster and fixed cached catalogues. Paired repeats and catalogue states are not independent samples; no significance claim, new queue confirmation, new physical hardware scaling or flow-time gain. Fresh queues previously triggered no strict-score-proof searches; expensive proof cases remain selected regressions. A five-percent gate is an operational threshold, not a confidence interval.",
    }
    protocol["timing_orders"] = {f"{model.node}:{lanes}": timing_orders(model.node, lanes, len(state_roster(protocol, lanes)))
                                 for model in models for lanes in previous["lanes"]}
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered models, sources, budget pair or timing design changed")


def load_prior_cell(protocol, node, lanes):
    with gzip.open(_cell_path(ROOT / protocol["prior_result_path"], node, lanes), "rt", encoding="utf-8") as f:
        saved = json.load(f)
    if saved["registered_at_utc"] != protocol["prior_registered_at_utc"]:
        raise ValueError("reused reference cell registration differs from confirmation")
    row = saved["cell"]
    signature = lambda state: (state["state_id"], state["panel"], state["cohort"], tuple(state["queue"]))
    if ((row["node"], row["lanes"]) != (node, lanes)
            or list(map(signature, row["states"])) != list(map(signature, state_roster(protocol, lanes)))):
        raise ValueError("reused reference cell differs from registered timing roster")
    return row


def confirmed_result(state):
    record = state["completed"]
    return {**{key: record[key] for key in (
        "selection", "baseline_selection", "score_certified_selection", "capacity_propagation",
        "score_proof", "completion", "integer_state_budget")},
        "preprocessing": record["metrics"]["preprocessing"]}


def timing_cell(model, lanes, protocol, runtime, prior_cell, selector=None, progress=True):
    if selector is None:
        from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
        selector = select_strict_score_proof_canonical_completion_frame
    started = perf_counter()
    repeats, labels = protocol["repeats"], tuple(protocol["labels"])
    saved = {state["state_id"]: state for state in prior_cell["states"]}
    states, references, expected_keys = [], [], []
    for registered in state_roster(protocol, lanes):
        old = saved[registered["state_id"]]
        if tuple(old["queue"]) != tuple(registered["queue"]):
            raise ValueError("reference queue differs from timing state")
        reference = selection_from_record(old["reference"])
        expected = runtime.reference_key(reference, tuple(registered["queue"]), len(model.options))
        if expected is None and old["oracle"] is not None:
            expected = oracle_key(old["oracle"], len(model.options))
        references.append(reference)
        expected_keys.append(expected)
        states.append({**registered,
            **{key: old[key] for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                                       "effective_reference_complete", "effective_reference_source")},
            "previous_integer_trigger": bool(old["completed"]["score_proof"]["attempted"]
                                             or old["completed"]["completion"]["attempted"]),
            "expected_candidate_result": confirmed_result(old),
            "first_results": {}, "repeat_mismatches": {label: [] for label in labels},
            "metrics": {label: runtime._metrics() for label in labels},
            "wall_ns": {label: [None] * repeats for label in labels},
            "cpu_ns": {label: [None] * repeats for label in labels},
        })
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
           "states": states, "warmups": [], "runtime_start": runtime_provenance()}
    row["runtime_before_warmup"] = runtime_provenance()
    for _ in range(protocol["warmups_per_label"]):
        for label in labels:
            result = selector(model.options, tuple(states[0]["queue"]), lanes,
                              protocol["lp_budgets"][label], protocol["integer_state_budget"])
            metrics = runtime._metrics()
            runtime._record(metrics, result.selection, model, tuple(states[0]["queue"]), lanes,
                            protocol["lp_budgets"][label], expected_keys[0], asdict(result.preprocessing), references[0])
            row["warmups"].append({"label": label, "state_index": 0, "result": asdict(result), "metrics": metrics})
    row["runtime_after_warmup"] = runtime_provenance()
    orders = protocol["timing_orders"][f"{model.node}:{lanes}"]
    for repeat, state_order in enumerate(orders["state_orders"]):
        for index in state_order:
            state, queue = states[index], tuple(states[index]["queue"])
            results = {}
            for label in orders["label_orders"][index][repeat]:
                cpu_start = process_time_ns()
                wall_start = perf_counter_ns()
                result = selector(model.options, queue, lanes, protocol["lp_budgets"][label], protocol["integer_state_budget"])
                wall_elapsed = perf_counter_ns() - wall_start
                cpu_elapsed = process_time_ns() - cpu_start
                results[label] = result
                state["wall_ns"][label][repeat] = wall_elapsed
                state["cpu_ns"][label][repeat] = cpu_elapsed
            # Validation and serialization follow the entire adjacent pair.
            for label, result in results.items():
                literal = asdict(result)
                runtime._record(state["metrics"][label], result.selection, model, queue, lanes,
                                protocol["lp_budgets"][label], expected_keys[index],
                                literal["preprocessing"], references[index])
                if repeat == 0:
                    state["first_results"][label] = literal
                elif literal != state["first_results"][label]:
                    state["repeat_mismatches"][label].append({"repeat_index": repeat, "result": literal})
        if progress:
            print(f"{model.node} R{lanes}: timed sweep {repeat + 1}/{repeats} complete", flush=True)
    row["runtime_end"] = runtime_provenance()
    row["operational_elapsed_s"] = perf_counter() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--merge-cells", action="store_true")
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
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.run:
        for node, lanes in protocol["execution_order"]:
            if _cell_path(args.output, node, lanes).exists():
                raise FileExistsError(_cell_path(args.output, node, lanes))
    rows = []
    for node, lanes in protocol["execution_order"]:
        path = _cell_path(args.output, node, lanes)
        if args.merge_cells:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                saved = json.load(source)
            if saved["registered_at_utc"] != protocol["registered_at_utc"]:
                raise ValueError("cell registration differs from protocol")
            row = saved["cell"]
        else:
            model = next(model for model in models if model.node == node)
            row = timing_cell(model, lanes, protocol, runtime, load_prior_cell(protocol, node, lanes))
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.strict_score_proof_timing_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "timed_selector_calls")}))


if __name__ == "__main__":
    main()
