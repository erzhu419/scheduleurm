"""Frozen development of a short integer probe followed by resumed LP search."""
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
from algorithm.experiments.aligned_hardware_strict_score_proof_timing import confirmed_result, state_roster, timing_orders


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_resumable_probe_development_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_strict_score_proof_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
LABELS = ("candidate", "control")
INITIAL_LP_BUDGET = 144
LP_NODE_BUDGET = 256
INTEGER_STATE_BUDGET = 8192
CONTROL_INTEGER_STATE_BUDGET = 100000
REPEATS = 6
WARMUPS_PER_LABEL = 2
ORDINARY_PER_CELL = 8
EXTRA_SOURCES = (
    "algorithm/resumable_integer_capacity_frame_candidates.py",
    "algorithm/resumable_probe_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_strict_score_proof_timing.py",
    "algorithm/experiments/strict_score_proof_timing_summary.py",
    "algorithm/experiments/aligned_hardware_resumable_probe_development.py",
    "algorithm/experiments/resumable_probe_development_summary.py",
)


def previous_initial(old):
    result = confirmed_result(old)
    return {"selection": result["baseline_selection"], "preprocessing": result["preprocessing"],
            "capacity_propagation": result["capacity_propagation"]}


def timing_roster(prior_cell, ordinary_per_cell=ORDINARY_PER_CELL):
    indices, ordinary = [], 0
    for index, state in enumerate(prior_cell["states"]):
        previous = state["completed"]
        trigger = bool(previous["score_proof"]["attempted"] or previous["completion"]["attempted"])
        if trigger or ordinary < ordinary_per_cell:
            indices.append(index)
            ordinary += not trigger
    return indices


def load_prior_cell(protocol, node, lanes):
    with gzip.open(_cell_path(ROOT / protocol["prior_result_path"], node, lanes), "rt", encoding="utf-8") as source:
        saved = json.load(source)
    if saved["registered_at_utc"] != protocol["prior_registered_at_utc"]:
        raise ValueError("prior cell registration differs from frozen development")
    row = saved["cell"]
    signature = lambda state: (state["state_id"], state["panel"], state["cohort"], tuple(state["queue"]))
    if ((row["node"], row["lanes"]) != (node, lanes)
            or list(map(signature, row["states"])) != list(map(signature, state_roster(protocol, lanes)))):
        raise ValueError("prior cell differs from the complete registered development roster")
    return row


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if (prior["decision"] != "STRICT_SCORE_PROOF_DEVELOPMENT_ADVANCE"
            or not prior["certificate_checks_passed"] or not prior["effective_references_complete"]):
        raise ValueError("successful frozen strict-proof development required")
    if (previous["budget"], previous["integer_state_budget"]) != (INITIAL_LP_BUDGET, CONTROL_INTEGER_STATE_BUDGET):
        raise ValueError("historical control budget pair changed")
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen physical catalogue changed")
    protocol = {
        "schema_version": 1, "experiment": "resumable_integer_probe_development_v1",
        "models": previous["models"], "lanes": previous["lanes"], "queue_panels": previous["queue_panels"],
        "initial_lp_budget": INITIAL_LP_BUDGET, "lp_node_budget": LP_NODE_BUDGET,
        "integer_state_budget": INTEGER_STATE_BUDGET, "control_integer_state_budget": CONTROL_INTEGER_STATE_BUDGET,
        "reference_budget": previous["reference_budget"], "oracle_state_budget": previous["oracle_state_budget"],
        "repeats": REPEATS, "warmups_per_label": WARMUPS_PER_LABEL, "ordinary_per_cell": ORDINARY_PER_CELL,
        "labels": LABELS, "cpu": min(os.sched_getaffinity(0)), "execution_order": previous["execution_order"],
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)), "prior_registered_at_utc": previous["registered_at_utc"],
        "trigger_ratio_limit": .90, "trigger_state_ratio_limit": 1.05, "ordinary_ratio_limit": 1.05,
        "roster": "Entire8620-state strict-score-proof development roster; subsequent fresh confirmation panels excluded. Reuse every full-reference and conditional-oracle record with zero new reference or oracle calls.",
        "candidate": "One persistent LP search: initial cap144, then shared8192-state strict proof/canonical probe if needed. On unresolved probe resume the same LP frontier to cumulative cap256. No second LP start or fallback second probe. FinalLP accounting is cumulative and counted once.",
        "screen": "One cold candidate call per registered state. Reproduce prior initial LP selection/preprocessing/capacity counters and validate final canonical exactness against frozen effective reference. Preserve every output. Finish each cell screen before its timing phase; retain all preregistered timing calls even on quality failures and mark the overall result HOLD or INVALID.",
        "timing": "Preselect all previously integer-triggered states and first8 ordinary states per node/R cell in registered order. Six cold whole-selector calls per arm/state; three AB and three BA orders shuffled per state, shuffled state order per sweep. Two untimed warmups per arm/cell at first selected timing state. Candidate compared with frozen strict selector LP144/integer100000. Keep every repeat and differing output; validate/serialize only after the adjacent pair. No trimming.",
        "measurement": "perf_counter_ns and process_time_ns surround entire selector only, including preprocessing and returned objects. Pin minimum available CPU before NumPy/SciPy import; retain actual runtime thread/affinity metadata. BLAS/OMP environment1 does not establish native HiGHS single threading.",
        "gate": "All8620 candidate outcomes and every measured/warmup result must be canonical exact and reference matching, initial LP reproduced, valid budgets/cumulative accounting, all timed paths stable, timed controls reproduce history. Trigger-subset ratio sums per-state median candidate/control<=0.90 and every triggered state ratio<=1.05; ordinary-subset pooledratio<=1.05. These are descriptive development targets, not confirmation or confidence intervals.",
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Post-failure development on reused queues and four fixed cached catalogues with virtual identical lanes. Timing subset is selected before new outcomes from historical triggers. Paired repeats are not independent samples; no new queue confirmation, physical scaling, significance or flow-time gain claim. Preserve old HOLD results.",
    }
    protocol["timing_indices"], protocol["timing_orders"] = {}, {}
    for model in models:
        for lanes in protocol["lanes"]:
            key = f"{model.node}:{lanes}"
            indices = timing_roster(load_prior_cell(protocol, model.node, lanes))
            protocol["timing_indices"][key] = indices
            protocol["timing_orders"][key] = timing_orders(model.node, lanes, len(indices), REPEATS)
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, rosters, budgets or timing design changed")


def screen_and_time_cell(model, lanes, protocol, runtime, prior_cell,
                         candidate_selector=None, control_selector=None, progress=True):
    if candidate_selector is None:
        from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
        candidate_selector = select_resumable_probe_frame
    if control_selector is None:
        from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
        control_selector = select_strict_score_proof_canonical_completion_frame
    from algorithm.experiments.resumable_probe_development_summary import audit_candidate, screen_audit, result_preprocessing
    started = perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes, "option_count": len(model.options),
           "registered_at_utc": protocol["registered_at_utc"], "states": [], "warmups": [],
           "runtime_start": runtime_provenance()}
    saved = {state["state_id"]: state for state in prior_cell["states"]}
    references, expected_keys = [], []

    def call(label, queue):
        if label == "candidate":
            return candidate_selector(model.options, queue, lanes, protocol["initial_lp_budget"],
                                      protocol["lp_node_budget"], protocol["integer_state_budget"])
        return control_selector(model.options, queue, lanes, protocol["initial_lp_budget"], protocol["control_integer_state_budget"])

    for registered in state_roster(protocol, lanes):
        old, queue = saved[registered["state_id"]], tuple(registered["queue"])
        reference = selection_from_record(old["reference"])
        expected = runtime.reference_key(reference, queue, len(model.options))
        if expected is None and old["oracle"] is not None:
            expected = oracle_key(old["oracle"], len(model.options))
        result = asdict(call("candidate", queue))
        state = {**registered, **{key: old[key] for key in (
            "reference", "reference_metrics", "reference_complete", "oracle", "effective_reference_complete", "effective_reference_source")},
            "expected_initial_lp": previous_initial(old), "expected_control_result": confirmed_result(old),
            "previous_integer_trigger": bool(old["completed"]["score_proof"]["attempted"] or old["completed"]["completion"]["attempted"]),
            "screen_result": result}
        state["screen_metrics"], _, _ = audit_candidate(result, state, protocol, model.snapshot(), lanes, expected)
        row["states"].append(state)
        references.append(reference)
        expected_keys.append(expected)
    row["screen_quality_passed"], row["screen_failures"] = screen_audit(protocol, row, model.snapshot())
    if progress:
        print(f"{model.node} R{lanes}: screen {len(row['states'])} states; quality={row['screen_quality_passed']}", flush=True)
    key = f"{model.node}:{lanes}"
    indices = protocol["timing_indices"][key]
    repeats = protocol["repeats"]
    for index in indices:
        row["states"][index].update(first_results={}, repeat_mismatches={label: [] for label in LABELS},
            metrics={label: runtime._metrics() for label in LABELS},
            wall_ns={label: [None] * repeats for label in LABELS}, cpu_ns={label: [None] * repeats for label in LABELS})
    row["runtime_before_warmup"] = runtime_provenance()
    index = indices[0]
    for _ in range(protocol["warmups_per_label"]):
        for label in LABELS:
            state = row["states"][index]
            literal = asdict(call(label, tuple(state["queue"])))
            metrics = runtime._metrics()
            runtime._record(metrics, selection_from_record(literal["selection"]), model, tuple(state["queue"]), lanes,
                            literal["selection"]["lp_node_budget"], expected_keys[index],
                            result_preprocessing(literal, label), references[index])
            row["warmups"].append({"label": label, "state_index": index, "result": literal, "metrics": metrics})
    row["runtime_after_warmup"] = runtime_provenance()
    orders = protocol["timing_orders"][key]
    for repeat, order in enumerate(orders["state_orders"]):
        for position in order:
            index = indices[position]
            state, outputs = row["states"][index], {}
            queue = tuple(state["queue"])
            for label in orders["label_orders"][position][repeat]:
                cpu_start = process_time_ns()
                wall_start = perf_counter_ns()
                result = call(label, queue)
                wall_elapsed = perf_counter_ns() - wall_start
                cpu_elapsed = process_time_ns() - cpu_start
                outputs[label] = result
                state["wall_ns"][label][repeat], state["cpu_ns"][label][repeat] = wall_elapsed, cpu_elapsed
            for label, result in outputs.items():
                literal = asdict(result)
                runtime._record(state["metrics"][label], result.selection, model, queue, lanes,
                                literal["selection"]["lp_node_budget"], expected_keys[index],
                                result_preprocessing(literal, label), references[index])
                if repeat == 0:
                    state["first_results"][label] = literal
                elif literal != state["first_results"][label]:
                    state["repeat_mismatches"][label].append({"repeat_index": repeat, "result": literal})
        if progress:
            print(f"{model.node} R{lanes}: paired timing {repeat + 1}/{repeats} complete", flush=True)
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
            row = screen_and_time_cell(model, lanes, protocol, runtime, load_prior_cell(protocol, node, lanes))
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, target,
                          separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.resumable_probe_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
