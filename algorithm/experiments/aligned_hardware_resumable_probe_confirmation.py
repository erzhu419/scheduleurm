"""Fresh queue confirmation and three-arm whole-call timing of resumable probes."""
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
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key
from algorithm.experiments.aligned_hardware_score_face_completion_confirmation import (
    static_exclusions, STATIC_PROTOCOLS, FRONTIER_PROTOCOL,
)
from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
from algorithm.experiments.aligned_hardware_strict_score_proof_timing import state_roster


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_resumable_probe_confirmation_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_resumable_probe_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
STRICT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_strict_score_proof_confirmation_20260927.json"
STRICT_PROTOCOL = STRICT_OUTPUT.with_name(STRICT_OUTPUT.stem + "_protocol.json")
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092791, 2026092792, 2026092793, 2026092794)
QUEUES_PER_PANEL = 64
TIMING_FRESH_PER_PANEL = 16
LABELS = ("candidate", "legacy", "long")
INITIAL_LP_BUDGET = 144
LP_NODE_BUDGET = 256
INTEGER_STATE_BUDGET = 8192
CONTROL_INTEGER_STATE_BUDGET = 100000
LONG_LP_BUDGET = 4096
REFERENCE_BUDGET = 4096
ORACLE_STATE_BUDGET = 5000000
REPEATS = 6
WARMUPS_PER_LABEL = 2
ORDER_SEED = 2026092799
EXPECTED_TRIGGER_COUNT = 14
PRIOR_FRESH_TRIGGERS = {32: ((61, 0, 35),), 64: ((88, 8, 23),)}
ORDINARY_TAIL_QUEUES = {32: ((96, 15, 13),)}
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_strict_score_proof_confirmation.py",
    "algorithm/experiments/strict_score_proof_confirmation_summary.py",
    "algorithm/experiments/aligned_hardware_resumable_probe_confirmation.py",
    "algorithm/experiments/resumable_probe_confirmation_summary.py",
)


def queue_design(previous_panels, excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues
    panels = {}
    for lanes in LANES:
        seen, panels[str(lanes)] = set(excluded[str(lanes)]), []
        for seed in PANEL_SEEDS:
            queues = sample_queues(lanes, seed, QUEUES_PER_PANEL, excluded=seen)
            panels[str(lanes)].append({"panel": f"fresh_{seed}", "seed": seed, "cohort": "fresh_confirmation",
                "states": [{"state_id": f"fresh:{seed}:{index}", "queue": queue} for index, queue in enumerate(queues)]})
            seen.update(queues)
        retained = set()
        for old in previous_panels[str(lanes)]:
            if old["cohort"] == "fresh_confirmation":
                continue
            panel = deepcopy(old)
            panel["states"] = [state for state in panel["states"] if tuple(state["queue"]) not in retained]
            retained.update(tuple(state["queue"]) for state in panel["states"])
            if panel["states"]:
                panels[str(lanes)].append(panel)
        for cohort, additions in (("prior_fresh_trigger_regression", PRIOR_FRESH_TRIGGERS),
                                  ("ordinary_tail_regression", ORDINARY_TAIL_QUEUES)):
            queues = [queue for queue in additions.get(lanes, ()) if tuple(queue) not in retained]
            retained.update(map(tuple, queues))
            if queues:
                panels[str(lanes)].append({"panel": cohort, "seed": None, "cohort": cohort,
                    "states": [{"state_id": cohort + ":" + ":".join(map(str, queue)), "queue": queue} for queue in queues]})
    return panels


def historical_trigger_keys(previous):
    keys = []
    for node, lanes in previous["execution_order"]:
        with gzip.open(_cell_path(STRICT_OUTPUT, node, lanes), "rt", encoding="utf-8") as source:
            saved = json.load(source)
        if saved["registered_at_utc"] != previous["registered_at_utc"]:
            raise ValueError("historical trigger cell has changed registration")
        row = saved["cell"]
        signature = lambda state: (state["state_id"], state["panel"], state["cohort"], tuple(state["queue"]))
        if ((row["node"], row["lanes"]) != (node, lanes)
                or list(map(signature, row["states"])) != list(map(signature, state_roster(previous, lanes)))):
            raise ValueError("historical trigger cell differs from its frozen strict-confirmation roster")
        for state in row["states"]:
            record = state["completed"]
            if record["score_proof"]["attempted"] or record["completion"]["attempted"]:
                keys.append({"node": node, "lanes": lanes, "queue": state["queue"]})
    if len(keys) != EXPECTED_TRIGGER_COUNT or len({(key["node"], key["lanes"], tuple(key["queue"])) for key in keys}) != len(keys):
        raise ValueError("historical integer trigger subgroup must remain the complete frozen set")
    return keys


def timing_indices(protocol, node, lanes):
    trigger = {(key["node"], key["lanes"], tuple(key["queue"])) for key in protocol["historical_trigger_keys"]}
    indices, index = [], 0
    for panel in protocol["queue_panels"][str(lanes)]:
        for position, state in enumerate(panel["states"]):
            if ((panel["cohort"] == "fresh_confirmation" and position < protocol["timing_fresh_per_panel"])
                    or panel["cohort"] == "ordinary_tail_regression"
                    or (node, lanes, tuple(state["queue"])) in trigger):
                indices.append(index)
            index += 1
    return indices


def timing_orders(node, lanes, count):
    rng = random.Random(f"{ORDER_SEED}:{node}:{lanes}")
    label_orders = []
    for _ in range(count):
        orders = list(permutations(LABELS))
        rng.shuffle(orders)
        label_orders.append(orders)
    state_orders = []
    for _ in range(REPEATS):
        order = list(range(count))
        rng.shuffle(order)
        state_orders.append(order)
    return {"label_orders": label_orders, "state_orders": state_orders}


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    strict = json.loads(STRICT_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if (prior["decision"] != "RESUMABLE_PROBE_DEVELOPMENT_ADVANCE" or not all(prior[key] for key in (
            "certificate_checks_passed", "effective_references_complete", "quality_gate_passed",
            "execution_checks_passed", "development_timing_gate_passed"))):
        raise ValueError("successful frozen resumable-probe development required before confirmation")
    if tuple(previous[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget", "control_integer_state_budget")) != (
            INITIAL_LP_BUDGET, LP_NODE_BUDGET, INTEGER_STATE_BUDGET, CONTROL_INTEGER_STATE_BUDGET):
        raise ValueError("confirmation must retain developed LP and integer budgets")
    for old in (previous, strict):
        for path, frozen in old["execution_sources"].items():
            if (ROOT / path).read_text(encoding="utf-8") != frozen:
                raise ValueError(f"frozen source changed: {path}")
        if json.loads(json.dumps([model.snapshot() for model in models])) != old["models"]:
            raise ValueError("frozen physical catalogue models changed")
    excluded = static_exclusions(previous)
    for lanes in LANES:
        excluded[str(lanes)].update(tuple(state["queue"]) for panel in strict["queue_panels"][str(lanes)] for state in panel["states"])
        excluded[str(lanes)].update(PRIOR_FRESH_TRIGGERS.get(lanes, ()))
        excluded[str(lanes)].update(ORDINARY_TAIL_QUEUES.get(lanes, ()))
    protocol = {
        "schema_version": 1, "experiment": "resumable_probe_fresh_confirmation_v1", "models": previous["models"],
        "lanes": LANES, "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "initial_lp_budget": INITIAL_LP_BUDGET, "lp_node_budget": LP_NODE_BUDGET,
        "integer_state_budget": INTEGER_STATE_BUDGET, "control_integer_state_budget": CONTROL_INTEGER_STATE_BUDGET,
        "long_lp_budget": LONG_LP_BUDGET, "reference_budget": REFERENCE_BUDGET, "oracle_state_budget": ORACLE_STATE_BUDGET,
        "labels": LABELS, "repeats": REPEATS, "warmups_per_label": WARMUPS_PER_LABEL,
        "timing_fresh_per_panel": TIMING_FRESH_PER_PANEL, "order_seed": ORDER_SEED, "cpu": min(os.sched_getaffinity(0)),
        "queue_panels": queue_design(strict["queue_panels"], excluded),
        "historical_trigger_keys": historical_trigger_keys(strict),
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "prior_registered_at_utc": previous["registered_at_utc"], "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)),
        "prior_decision": prior["decision"], "strict_registered_at_utc": strict["registered_at_utc"],
        "strict_result_path": str(STRICT_OUTPUT.relative_to(ROOT)),
        "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "excluded_protocol_paths": [str(path.relative_to(ROOT)) for path in (*STATIC_PROTOCOLS, FRONTIER_PROTOCOL, PRIOR_PROTOCOL, STRICT_PROTOCOL)],
        "fresh_nonregression_limit": 1.05, "trigger_legacy_ratio_limit": .90, "trigger_legacy_state_limit": 1.05,
        "trigger_long_ratio_limit": 1., "trigger_long_p99_limit": 1.05, "tail_state_limit": 1.05,
        "fresh_speedup_limit": .95,
        "sampler": "Four new seeds2026092791-94; each draws64 unique queues/R using unchanged sample_queues law randint(0,5R) and seed+1000R. Exclude frozen static/grid/development queues and every latest strict-confirmation queue, plus registered regression additions; reject earlier accepted draws. Queues paired across four fixed catalogues; no historical trajectory disjointness claim.",
        "regression_roster": "Retain every nonfresh strict-confirmation panel (456 node-states), append both formerly fresh integer-trigger queues atR32/R64 across all four catalogues (8), and known ordinary tailQ(96,15,13)R32 across allfour(4). Deduplicate regression queues byR. The exact14 historical node/R/Q integer-trigger keys are a separate frozen subgroup, not all paired copies of those queues.",
        "calls": "Every state receives a new cold original-catalogue fullK4096 reference and any reference-only oracle before an independently executed legacy LP144/integer100000 call and a cold candidate144->256/integer8192 call. Candidate initial snapshot is compared with that independently executed legacy baseline. Finish every screen, warmup and timed call after honest nonclosure; no replacement, retuning or reference reuse.",
        "reference_policy": "Before either screen arm, invoke original-option canonical oracle only when the full reference has score_exact but not canonical_exact. Use only the full reference's proved score and incumbent duration with fixed5000000-state cap. Preserve original nonclosure; unproved reference score or exhausted oracle remains INCOMPLETE_REFERENCE. No candidate output supplies its own reference.",
        "timing": "Preselect first16 of64 fresh queues perseed/R/node(1024), exact14 historical integer-trigger states, and all4 ordinary-tail states, giving1042. Six cold repeats of candidate,legacy,long; perstate every3-arm permutation exactlyonce shuffled, stateorder shuffled eachsweep. Long is frozen strict selector LP4096/integer100000. Two untimedwarmups perarm/cell at firsttimingstate(96). Perfcounter/processCPU intervals enclose entireselector; serialize and validate only after entire adjacent3-arm block. Retain all repeats and mismatching outputs; no trimming.",
        "accounting": "Final candidate LP/preprocessing/capacity totals are cumulative and charged once; initial snapshots and reused probe preprocessing are embedded views. Charge all probe states including exhausted work and every independently executed legacy,long,reference,warmup call. Oracles have separately reported calls/work. No uncharged cold restarts, second probes, or reference-score inputs to candidate.",
        "runtime_policy": "One host, serial cells, pin minimum available CPU before NumPy/SciPy import. BLAS/OMP environment1, GC enabled; record actual affinity, threadpools and native threadcount without assuming HiGHS singlethreading. Cold calls mean fresh search state; earlier references and calls warm the shared runtime, so this is not a cold-cache measurement.",
        "gate": "Complete roster and effective independent references; every executed candidate,legacy,long outcome canonical exact and reference matching, stable repeats, reproduced screen outputs and initial LP snapshots, valid budgets/accounting. Fresh timing pooled/cell/p99 ratios<=1.05 againstboth controls. Historicaltrigger ratio<=0.90 and everytriggerstate<=1.05 againstlegacy; ratio<=1.0 andp99<=1.05 againstlong. Everytailstate<=1.05 againstboth. ADVANCE means independent quality and targeted benefit with fresh nonregression. A separate fresh_speedup flag requires pooled<=0.95 againstboth; no fresh speed claim follows from nonregression alone.",
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *strict["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "New conditional queue samples on four fixed cached catalogues and virtual identical lanes. Queue pairs and repeated timings are not independent samples. The measured fresh subset is predetermined; the historicaltrigger and ordinarytail subgroups remain selected regression evidence. Local descriptive timing gives no significance, universalbudget, newphysicalscaling orflow-time claim. Earlier negative results retained.",
    }
    protocol["timing_indices"] = {f"{model.node}:{lanes}": timing_indices(protocol, model.node, lanes) for model in models for lanes in LANES}
    protocol["timing_orders"] = {key: timing_orders(key.rsplit(":", 1)[0], int(key.rsplit(":", 1)[1]), len(indices))
                                 for key, indices in protocol["timing_indices"].items()}
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, models, independent queues, budgets or timing design changed")


def screen_and_time_cell(model, lanes, protocol, runtime, candidate_selector=None, legacy_selector=None,
                         long_selector=None, reference_oracle=None, progress=True):
    if candidate_selector is None:
        from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
        candidate_selector = select_resumable_probe_frame
    if legacy_selector is None or long_selector is None:
        from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
        legacy_selector = legacy_selector or select_strict_score_proof_canonical_completion_frame
        long_selector = long_selector or select_strict_score_proof_canonical_completion_frame
    if reference_oracle is None:
        from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
        reference_oracle = canonical_at_certified_score
    started = perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes, "option_count": len(model.options),
           "registered_at_utc": protocol["registered_at_utc"], "states": [], "warmups": [], "runtime_start": runtime_provenance()}
    references, expected_keys = [], []
    triggers = {(key["node"], key["lanes"], tuple(key["queue"])) for key in protocol["historical_trigger_keys"]}

    def call(label, queue):
        if label == "candidate":
            return candidate_selector(model.options, queue, lanes, protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
        selector = legacy_selector if label == "legacy" else long_selector
        budget = protocol["initial_lp_budget"] if label == "legacy" else protocol["long_lp_budget"]
        return selector(model.options, queue, lanes, budget, protocol["control_integer_state_budget"])

    def record(metrics, literal, label, index):
        prep = literal["final_lp"]["preprocessing"] if label == "candidate" else literal["preprocessing"]
        runtime._record(metrics, selection_from_record(literal["selection"]), model, tuple(row["states"][index]["queue"]), lanes,
                        literal["selection"]["lp_node_budget"], expected_keys[index], prep, references[index])

    for registered in state_roster(protocol, lanes):
        queue = tuple(registered["queue"])
        reference, reference_metrics, reference_complete, independent = _new_reference(model, queue, lanes, protocol, runtime, reference_oracle)
        expected = runtime.reference_key(reference, queue, len(model.options))
        source = "full_K4096" if expected is not None else None
        if expected is None and independent is not None:
            expected = oracle_key(independent, len(model.options))
            source = "independent_score_face_oracle" if expected is not None else None
        legacy = asdict(call("legacy", queue))
        candidate = asdict(call("candidate", queue))
        state = {**registered, "reference": asdict(reference), "reference_metrics": reference_metrics,
            "reference_complete": reference_complete, "oracle": independent,
            "effective_reference_complete": expected is not None, "effective_reference_source": source,
            "expected_initial_lp": {"selection": legacy["baseline_selection"], "preprocessing": legacy["preprocessing"],
                                    "capacity_propagation": legacy["capacity_propagation"]},
            "screen_results": {"candidate": candidate, "legacy": legacy},
            "screen_metrics": {label: runtime._metrics() for label in ("candidate", "legacy")},
            "previous_integer_trigger": (model.node, lanes, queue) in triggers}
        references.append(reference)
        expected_keys.append(expected)
        row["states"].append(state)
        for label in ("candidate", "legacy"):
            record(state["screen_metrics"][label], state["screen_results"][label], label, len(row["states"]) - 1)
    if progress:
        print(f"{model.node} R{lanes}: independent screen {len(row['states'])} states complete", flush=True)
    key, repeats = f"{model.node}:{lanes}", protocol["repeats"]
    indices = protocol["timing_indices"][key]
    for index in indices:
        row["states"][index].update(first_results={}, repeat_mismatches={label: [] for label in LABELS},
            metrics={label: runtime._metrics() for label in LABELS},
            wall_ns={label: [None] * repeats for label in LABELS}, cpu_ns={label: [None] * repeats for label in LABELS})
    row["runtime_before_warmup"] = runtime_provenance()
    index = indices[0]
    for _ in range(protocol["warmups_per_label"]):
        for label in LABELS:
            literal = asdict(call(label, tuple(row["states"][index]["queue"])))
            metrics = runtime._metrics()
            record(metrics, literal, label, index)
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
                record(state["metrics"][label], literal, label, index)
                if repeat == 0:
                    state["first_results"][label] = literal
                elif literal != state["first_results"][label]:
                    state["repeat_mismatches"][label].append({"repeat_index": repeat, "result": literal})
        if progress:
            print(f"{model.node} R{lanes}: three-arm timing {repeat + 1}/{repeats} complete", flush=True)
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
            row = screen_and_time_cell(model, lanes, protocol, runtime)
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, target,
                          separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.resumable_probe_confirmation_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
