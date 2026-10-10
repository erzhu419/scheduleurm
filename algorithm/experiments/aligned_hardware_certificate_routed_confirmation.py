"""Fresh queue confirmation of certificate-routed LP continuation and probes."""
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
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_confirmation_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_certificate_routed_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_resumable_probe_confirmation_20260927.json"
PREVIOUS_PROTOCOL = PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json")
STRICT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_strict_score_proof_confirmation_20260927.json"
STRICT_PROTOCOL = STRICT_OUTPUT.with_name(STRICT_OUTPUT.stem + "_protocol.json")
LANES = (8, 16, 32, 64)
PANEL_SEEDS = (2026092801, 2026092802, 2026092803, 2026092804)
QUEUES_PER_PANEL = 64
TIMING_FRESH_PER_PANEL = 16
LABELS = ("candidate", "probe", "long")
INITIAL_LP_BUDGET = 144
LP_NODE_BUDGET = 256
INTEGER_STATE_BUDGET = 8192
CONTROL_INTEGER_STATE_BUDGET = 100000
LONG_LP_BUDGET = 4096
REFERENCE_BUDGET = 4096
ORACLE_STATE_BUDGET = 5000000
REPEATS = 6
WARMUPS_PER_LABEL = 2
ORDER_SEED = 2026092809
EXPECTED_TRIGGER_COUNT = 18
EXPECTED_SCORE_UNPROVED_COUNT = 6
PREVIOUS_UNTIMED_PROBE_QUEUES = {32: ((64, 1, 18), (49, 4, 26)), 64: ((122, 3, 4), (129, 2, 46))}
PREVIOUS_UNTIMED_PROBE_NODE = "gpu_b"
ORDINARY_TAIL_QUEUES = {8: ((27, 15, 34),), 32: ((96, 15, 13),), 64: ((88, 19, 118), (184, 78, 235))}
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_certificate_routed_confirmation.py",
    "algorithm/experiments/certificate_routed_confirmation_summary.py",
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
        for cohort, additions in (("previously_untimed_probe_regression", PREVIOUS_UNTIMED_PROBE_QUEUES),
                                  ("ordinary_tail_regression", ORDINARY_TAIL_QUEUES)):
            queues = [queue for queue in additions.get(lanes, ()) if tuple(queue) not in retained]
            retained.update(map(tuple, queues))
            if queues:
                matching = next((panel for panel in panels[str(lanes)] if panel["panel"] == cohort), None)
                states = [{"state_id": cohort + ":" + ":".join(map(str, queue)), "queue": queue} for queue in queues]
                if matching is None:
                    panels[str(lanes)].append({"panel": cohort, "seed": None, "cohort": cohort, "states": states})
                else:
                    matching["states"].extend(states)
    return panels


def registered_subgroups(previous):
    keys = deepcopy(previous["historical_trigger_keys"])
    additions = [{"node": PREVIOUS_UNTIMED_PROBE_NODE, "lanes": lanes, "queue": list(queue)}
                 for lanes, queues in PREVIOUS_UNTIMED_PROBE_QUEUES.items() for queue in queues]
    keys.extend(additions)
    trigger_keys = {(key["node"], key["lanes"], tuple(key["queue"])) for key in keys}
    score_keys, found = [], set()
    for node, lanes in previous["execution_order"]:
        with gzip.open(_cell_path(PREVIOUS_OUTPUT, node, lanes), "rt", encoding="utf-8") as source:
            saved = json.load(source)
        if saved["registered_at_utc"] != previous["registered_at_utc"]:
            raise ValueError("historical trigger cell has changed registration")
        row = saved["cell"]
        signature = lambda state: (state["state_id"], state["panel"], state["cohort"], tuple(state["queue"]))
        if ((row["node"], row["lanes"]) != (node, lanes)
                or list(map(signature, row["states"])) != list(map(signature, state_roster(previous, lanes)))):
            raise ValueError("historical trigger cell differs from its frozen probe-confirmation roster")
        for state in row["states"]:
            key = (node, lanes, tuple(state["queue"]))
            if key in trigger_keys:
                found.add(key)
                record = state["screen_results"]["candidate"]
                if record["initial_lp"]["selection"]["canonical_exact"]:
                    raise ValueError("registered historical trigger has a complete initial LP")
                if not record["initial_lp"]["selection"]["score_exact"]:
                    score_keys.append({"node": node, "lanes": lanes, "queue": state["queue"]})
    if len(keys) != EXPECTED_TRIGGER_COUNT or len(trigger_keys) != len(keys) or found != trigger_keys:
        raise ValueError("historical integer trigger subgroup must remain the complete frozen set")
    if len(score_keys) != EXPECTED_SCORE_UNPROVED_COUNT:
        raise ValueError("historical score-unproved subgroup must remain the complete frozen set")
    return keys, score_keys, additions


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
    prior_protocol = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS_PROTOCOL.read_text(encoding="utf-8"))
    strict = json.loads(STRICT_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    previous_result = json.loads(PREVIOUS_OUTPUT.read_text(encoding="utf-8"))
    if (prior["decision"] != "CERTIFICATE_ROUTED_DEVELOPMENT_ADVANCE" or not all(prior[key] for key in (
            "certificate_checks_passed", "effective_references_complete", "quality_gate_passed",
            "execution_checks_passed", "development_timing_gate_passed"))):
        raise ValueError("successful frozen certificate-routed development required before confirmation")
    if not all(previous_result[key] for key in (
            "certificate_checks_passed", "effective_references_complete", "candidate_quality_gate_passed",
            "comparison_quality_gate_passed", "execution_checks_passed")):
        raise ValueError("complete quality-valid previous probe confirmation required; its timing HOLD is retained")
    for old in (prior_protocol, previous):
        if tuple(old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget", "control_integer_state_budget")) != (
                INITIAL_LP_BUDGET, LP_NODE_BUDGET, INTEGER_STATE_BUDGET, CONTROL_INTEGER_STATE_BUDGET):
            raise ValueError("confirmation must retain developed LP and integer budgets")
    for old in (prior_protocol, previous, strict):
        for path, frozen in old["execution_sources"].items():
            if (ROOT / path).read_text(encoding="utf-8") != frozen:
                raise ValueError(f"frozen source changed: {path}")
        if json.loads(json.dumps([model.snapshot() for model in models])) != old["models"]:
            raise ValueError("frozen physical catalogue models changed")
    excluded = static_exclusions(prior_protocol)
    for lanes in LANES:
        for old in (strict, previous):
            excluded[str(lanes)].update(tuple(state["queue"]) for panel in old["queue_panels"][str(lanes)] for state in panel["states"])
        excluded[str(lanes)].update(PREVIOUS_UNTIMED_PROBE_QUEUES.get(lanes, ()))
        excluded[str(lanes)].update(ORDINARY_TAIL_QUEUES.get(lanes, ()))
    triggers, score_keys, untimed_keys = registered_subgroups(previous)
    protocol = {
        "schema_version": 1, "experiment": "certificate_routed_fresh_confirmation_v1", "models": previous["models"],
        "lanes": LANES, "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "initial_lp_budget": INITIAL_LP_BUDGET, "lp_node_budget": LP_NODE_BUDGET,
        "integer_state_budget": INTEGER_STATE_BUDGET, "control_integer_state_budget": CONTROL_INTEGER_STATE_BUDGET,
        "long_lp_budget": LONG_LP_BUDGET, "reference_budget": REFERENCE_BUDGET, "oracle_state_budget": ORACLE_STATE_BUDGET,
        "labels": LABELS, "repeats": REPEATS, "warmups_per_label": WARMUPS_PER_LABEL,
        "timing_fresh_per_panel": TIMING_FRESH_PER_PANEL, "order_seed": ORDER_SEED, "cpu": min(os.sched_getaffinity(0)),
        "queue_panels": queue_design(previous["queue_panels"], excluded),
        "historical_trigger_keys": triggers, "score_unproved_keys": score_keys,
        "previous_untimed_probe_keys": untimed_keys,
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "prior_registered_at_utc": prior_protocol["registered_at_utc"], "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)),
        "prior_decision": prior["decision"], "previous_registered_at_utc": previous["registered_at_utc"],
        "previous_result_path": str(PREVIOUS_OUTPUT.relative_to(ROOT)), "previous_decision": previous_result["decision"],
        "strict_registered_at_utc": strict["registered_at_utc"], "strict_result_path": str(STRICT_OUTPUT.relative_to(ROOT)),
        "excluded_static_queues": {key: sorted(queues) for key, queues in excluded.items()},
        "excluded_protocol_paths": [str(path.relative_to(ROOT)) for path in (
            *STATIC_PROTOCOLS, FRONTIER_PROTOCOL, PRIOR_PROTOCOL, PREVIOUS_PROTOCOL, STRICT_PROTOCOL)],
        "fresh_nonregression_limit": 1.05, "fresh_speedup_limit": .95,
        "score_gain_limit": .95, "state_ratio_limit": 1.05, "nonregression_limit": 1.05,
        "trigger_long_ratio_limit": 1., "tail_state_limit": 1.05,
        "sampler": "Four new seeds2026092801-04; each draws64 unique queues/R using unchanged sample_queues law randint(0,5R) and seed+1000R. Exclude frozen static/grid/development queues, all strict/probe-confirmation queues and registered regressions; reject earlier accepted draws. Queues paired across four fixed catalogues; no historical trajectory disjointness claim.",
        "regression_roster": "Retain every nonfresh prior probe-confirmation panel (468 node-states), append four previously untimed probe queues across allfour catalogues(16), and three ordinary tail queues across allfour(12). Deduplicate regression queues byR; total496 regressions. Exact18 historical node/R/Q triggers comprise14 prior timing triggers plusfour previously untimed gpu2 states. Freeze six score-unproved keys and four previously untimed keys separately; paired catalogue copies are not trigger keys.",
        "calls": "Every state receives a new original-catalogue fullK4096 reference and any reference-only oracle before an independently executed probe144/256/8192 and candidate144/256/8192. Candidate initial LP is compared with that independent probe initial snapshot. Retain all screens,warmups,timed calls and nonclosures without replacement or retuning.",
        "reference_policy": "Before either screen arm, invoke original-option canonical oracle only when the full reference has score_exact but not canonical_exact. Use only the reference's proved score and incumbent duration with fixed5000000-state cap. Preserve original nonclosure; unproved score or exhausted oracle remains INCOMPLETE_REFERENCE. Candidate never supplies its own reference.",
        "timing": "Preselect first16 of64 fresh queues perseed/R/node(1024), exact18 historical trigger states and all16 ordinary-tail states:1058 states. Six fresh selector invocations of candidate,probe,long; perstate every3-arm permutation exactlyonce shuffled, stateorder shuffled eachsweep. Long is frozen strict LP4096/integer100000. Two untimedwarmups perarm/cell at firsttimingstate(96). Entireselector wall/CPU intervals; serialization and validation after adjacent3-arm block. Retain all repeats and mismatches without trimming.",
        "accounting": "Candidate/probe final LP/preprocessing/capacity totals are cumulative and charged once; initial snapshots and reused probe preprocessing are embedded views. Charge all spent integer states, independent controls,references,warmups. Oracles separately reported. No uncharged restarts or candidate reference-score inputs.",
        "runtime_policy": "One host, serial cells, pin minimum available CPU before NumPy/SciPy import. BLAS/OMP environment1, GC enabled; record actual affinity, threadpools and native threadcount without assuming HiGHS singlethreading. New calls have fresh search state, not a cold-cache guarantee.",
        "gate": "Complete roster and effective independent references; all candidate,probe,long outcomes canonical exact and reference matching; repeat/screen/initial snapshot consistency and valid budgets/accounting. Fresh pooled/cell/p99 ratios<=1.05 againstboth. Six score-unproved states: pooled<=.95 and every state<=1.05 againstprobe, pooled<=1.05 againstlong. Other12 historical triggers: pooled and every state<=1.05 againstprobe. All18 triggers: pooled<=1 andp99<=1.05 againstlong. Everytailstate<=1.05 againstboth. ADVANCE is independent quality/fresh nonregression plus revalidated known-regression benefit. Fresh speedup requires pooled<=.95 againstboth separately.",
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*prior_protocol["execution_sources"], *previous["execution_sources"], *strict["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "New conditional queue samples on four fixed cached catalogues and virtual identical lanes. Queue pairs and timing repeats are dependent. Fresh timing subset fixed before outcomes; trigger and tail groups remain selected regression evidence. Descriptive local timing gives no significance, universalbudget, newphysicalscaling orflow-time claim. Previous HOLD results retained.",
    }
    protocol["timing_indices"] = {f"{model.node}:{lanes}": timing_indices(protocol, model.node, lanes) for model in models for lanes in LANES}
    protocol["timing_orders"] = {key: timing_orders(key.rsplit(":", 1)[0], int(key.rsplit(":", 1)[1]), len(indices))
                                 for key, indices in protocol["timing_indices"].items()}
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, models, independent queues, budgets or timing design changed")


def screen_and_time_cell(model, lanes, protocol, runtime, candidate_selector=None, probe_selector=None,
                         long_selector=None, reference_oracle=None, progress=True):
    if candidate_selector is None:
        from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
        candidate_selector = select_certificate_routed_frame
    if probe_selector is None:
        from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
        probe_selector = select_resumable_probe_frame
    if long_selector is None:
        from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
        long_selector = select_strict_score_proof_canonical_completion_frame
    if reference_oracle is None:
        from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
        reference_oracle = canonical_at_certified_score
    started = perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes, "option_count": len(model.options),
           "registered_at_utc": protocol["registered_at_utc"], "states": [], "warmups": [], "runtime_start": runtime_provenance()}
    references, expected_keys = [], []
    triggers = {(key["node"], key["lanes"], tuple(key["queue"])) for key in protocol["historical_trigger_keys"]}
    score_keys = {(key["node"], key["lanes"], tuple(key["queue"])) for key in protocol["score_unproved_keys"]}
    untimed_keys = {(key["node"], key["lanes"], tuple(key["queue"])) for key in protocol["previous_untimed_probe_keys"]}

    def call(label, queue):
        if label in ("candidate", "probe"):
            selector = candidate_selector if label == "candidate" else probe_selector
            return selector(model.options, queue, lanes, protocol["initial_lp_budget"], protocol["lp_node_budget"], protocol["integer_state_budget"])
        return long_selector(model.options, queue, lanes, protocol["long_lp_budget"], protocol["control_integer_state_budget"])

    def record(metrics, literal, label, index):
        prep = literal["preprocessing"] if label == "long" else literal["final_lp"]["preprocessing"]
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
        probe = asdict(call("probe", queue))
        candidate = asdict(call("candidate", queue))
        state = {**registered, "reference": asdict(reference), "reference_metrics": reference_metrics,
            "reference_complete": reference_complete, "oracle": independent,
            "effective_reference_complete": expected is not None, "effective_reference_source": source,
            "expected_initial_lp": {key: probe["initial_lp"][key] for key in ("selection", "preprocessing", "capacity_propagation")},
            "expected_probe_result": probe,
            "screen_results": {"candidate": candidate, "probe": probe},
            "screen_metrics": {label: runtime._metrics() for label in ("candidate", "probe")},
            "previous_integer_trigger": (model.node, lanes, queue) in triggers,
            "score_unproved": not probe["initial_lp"]["selection"]["score_exact"],
            "score_unproved_regression": (model.node, lanes, queue) in score_keys,
            "previous_untimed_probe_trigger": (model.node, lanes, queue) in untimed_keys}
        references.append(reference)
        expected_keys.append(expected)
        row["states"].append(state)
        for label in ("candidate", "probe"):
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
    from algorithm.experiments.certificate_routed_confirmation_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
