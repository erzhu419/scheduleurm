"""Confirm the frozen shared-budget score proof on new queues and retained hard cases."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from itertools import product
import json
from pathlib import Path
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
from algorithm.experiments.aligned_hardware_score_face_completion_confirmation import (
    static_exclusions, STATIC_PROTOCOLS, FRONTIER_PROTOCOL,
)
from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_strict_score_proof_confirmation_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_strict_score_proof_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
LANES = (8, 16, 32, 64)
BUDGET = 144
INTEGER_STATE_BUDGET = 100_000
REFERENCE_BUDGET = 4096
ORACLE_STATE_BUDGET = 5_000_000
PANEL_SEEDS = (2026092771, 2026092772, 2026092773, 2026092774)
QUEUES_PER_PANEL = 64
WITNESS_QUEUE = (86, 87, 146)
KNOWN_COMPLETION_QUEUES = (
    (153, 147, 290), (81, 31, 48), (101, 9, 59),
    (39, 3, 313), (145, 1, 67), (85, 34, 58),
)
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_strict_score_proof_confirmation.py",
    "algorithm/experiments/strict_score_proof_confirmation_summary.py",
)


def queue_design(previous_panels, excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues
    panels = {}
    for lanes in LANES:
        seen, panels[str(lanes)] = set(excluded[str(lanes)]), []
        for seed in PANEL_SEEDS:
            queues = sample_queues(lanes, seed, QUEUES_PER_PANEL, excluded=seen)
            panels[str(lanes)].append({
                "panel": f"fresh_{seed}", "seed": seed, "cohort": "fresh_confirmation",
                "states": [{"state_id": f"fresh:{seed}:{index}", "queue": queue}
                           for index, queue in enumerate(queues)],
            })
            seen.update(queues)
        for old in previous_panels[str(lanes)]:
            if old["cohort"] == "old_regression":
                panels[str(lanes)].append({
                    "panel": "regression:" + old["panel"], "seed": old.get("seed"),
                    "cohort": "old_regression", "states": [
                        {"state_id": "regression:" + state["state_id"], "queue": state["queue"]}
                        for state in old["states"]],
                })
        if lanes == 32:
            neighbors = [tuple(q + delta for q, delta in zip(WITNESS_QUEUE, offsets))
                         for offsets in product((-1, 0, 1), repeat=3)]
            panels[str(lanes)].append({
                "panel": "score_proof_neighbors", "seed": None, "cohort": "score_proof_regression",
                "states": [{"state_id": "score:" + ":".join(map(str, queue)), "queue": queue}
                           for queue in neighbors],
            })
        if lanes == 64:
            panels[str(lanes)].append({
                "panel": "known_completion", "seed": None, "cohort": "canonical_regression",
                "states": [{"state_id": f"canonical:{index}", "queue": queue}
                           for index, queue in enumerate(KNOWN_COMPLETION_QUEUES)],
            })
    return panels


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if (prior["decision"] != "STRICT_SCORE_PROOF_DEVELOPMENT_ADVANCE"
            or not all(prior[key] for key in ("certificate_checks_passed", "effective_references_complete",
                                              "development_action_gate_passed"))):
        raise ValueError("successful strict-score-proof development is required before confirmation")
    if (previous["budget"], previous["integer_state_budget"]) != (BUDGET, INTEGER_STATE_BUDGET):
        raise ValueError("confirmation must retain the developed shared budget pair")
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen development source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen catalogue models changed")
    excluded = static_exclusions(previous)
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "strict_score_proof_fresh_confirmation_v1",
        "models": previous["models"], "lanes": LANES,
        "budget": BUDGET, "integer_state_budget": INTEGER_STATE_BUDGET,
        "reference_budget": REFERENCE_BUDGET, "oracle_state_budget": ORACLE_STATE_BUDGET,
        "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "queue_panels": queue_design(previous["queue_panels"], excluded),
        "excluded_static_queues": {lanes: sorted(queues) for lanes, queues in excluded.items()},
        "excluded_protocol_paths": [str(path.relative_to(ROOT))
                                    for path in (*STATIC_PROTOCOLS, FRONTIER_PROTOCOL, PRIOR_PROTOCOL)],
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "prior_registered_at_utc": previous["registered_at_utc"],
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)), "prior_decision": prior["decision"],
        "sampler": "For each R use random.Random(panel_seed+1000*R), each coordinate randint(0,5R). Reject the four listed static protocol rosters (including all8620 strict-proof development states), native/scaled grids and earlier accepted draws at that R. Four new panels of64 distinct queues per R, shared across four catalogues, without cross-R coupling. This is conditional sampling without replacement, not a claim of disjointness from all historical trajectory states.",
        "regression_roster": "Keep324 node-states from the three old radius-one panels; the full27-queue R32 score-proof neighborhood across four catalogues adds108; six known R64 canonical-completion queues across four catalogues add24. These456 regression states cover every previously triggered integer proof/completion queue and are separate from4096 fresh node-states.",
        "calls": "For every fresh and regression state execute one cold fullK4096 reference, any reference-only oracle, then one cold strict-proof selector. No reference reuse or warm starts. Baseline and intermediate score certificate are embedded views of the same candidate call, not additional cold calls. Complete all frozen states despite honest nonclosure or mismatch, without replacing seeds or retuning either budget.",
        "trigger": previous["trigger"],
        "reference_policy": "Before the candidate, run the independent original-option oracle only if the fullK4096 reference has score_exact and not canonical_exact. Use that reference's own exact score and incumbent duration, with a fixed5000000-state cap. Preserve original nonclosure. An exhausted oracle or unproved full-reference score remains INCOMPLETE_REFERENCE; candidate outputs never supply reference proofs or actions.",
        "main_gate": "Complete registered coverage and valid references, selections and accounting. Require LP<=144 and proof states plus canonical states<=100000 per candidate; preserve honest failure statuses. Every fresh final output must have a canonical certificate and exact reference score/duration/original-index agreement; apply the same gate separately to all three regression cohorts. Overall ADVANCE requires both fresh and regression gates and all effective references complete.",
        "accounting": "Count two cold LP selectors per state, with reference-oracle work separate. Candidate work retains original LP/preprocessing/capacity counters and separately charges strict proof and canonical integer work, including failed searches. The integer cap is shared, not100000 per stage. No speed gate; individual elapsed times are operational diagnostics only.",
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Only queue panels are new under a fixed sampling law; algorithms, budgets and four cached physical catalogues are fixed. R8/16/32/64 are virtual identical lanes. Queues are paired across catalogues, not4096 independent observations. Conditional canonical oracles use separate implementations with shared enumeration mathematics and depend on the full reference's score proof. This does not establish a universal budget, new physical scaling, runtime or flow-time gain. Previous HOLD results remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, models, queues, reference rule or shared budget changed")


def screen_cell(model, lanes, protocol, runtime, selector=None, reference_oracle=None, progress=True):
    if selector is None:
        from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
        selector = select_strict_score_proof_canonical_completion_frame
    if reference_oracle is None:
        from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
        reference_oracle = canonical_at_certified_score
    started = perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
           "states": [], "runtime_start": runtime_provenance()}
    for panel in protocol["queue_panels"][str(lanes)]:
        for registered in panel["states"]:
            queue = tuple(registered["queue"])
            reference, reference_metrics, reference_complete, independent = _new_reference(
                model, queue, lanes, protocol, runtime, reference_oracle)
            expected = runtime.reference_key(reference, queue, len(model.options))
            source = "full_K4096" if expected is not None else None
            if expected is None and independent is not None:
                expected = oracle_key(independent, len(model.options))
                source = "independent_score_face_oracle" if expected is not None else None
            call_started = perf_counter()
            result = selector(model.options, queue, lanes, protocol["budget"], protocol["integer_state_budget"])
            elapsed = perf_counter() - call_started
            prep, capacity = asdict(result.preprocessing), asdict(result.capacity_propagation)
            metrics = {}
            for label, selected in (("baseline", result.baseline_selection),
                                    ("score_certified", result.score_certified_selection), ("completed", result.selection)):
                metrics[label] = runtime._metrics()
                runtime._record(metrics[label], selected, model, queue, lanes, protocol["budget"], expected,
                                prep, reference, elapsed if label == "completed" else 0.0)
            row["states"].append({
                **registered, "panel": panel["panel"], "cohort": panel["cohort"],
                "reference": asdict(reference), "reference_metrics": reference_metrics,
                "reference_complete": reference_complete, "oracle": independent,
                "effective_reference_complete": expected is not None, "effective_reference_source": source,
                "baseline_source": "embedded_stage_one",
                "baseline": {"selection": asdict(result.baseline_selection), "metrics": metrics["baseline"],
                             "capacity_propagation": capacity},
                "score_certified_metrics": metrics["score_certified"],
                "completed": {"selection": asdict(result.selection), "baseline_selection": asdict(result.baseline_selection),
                              "score_certified_selection": asdict(result.score_certified_selection),
                              "integer_state_budget": result.integer_state_budget,
                              "metrics": metrics["completed"], "capacity_propagation": capacity,
                              "score_proof": asdict(result.score_proof), "completion": asdict(result.completion)},
            })
        if progress:
            print(f"{model.node} R{lanes}: {panel['panel']} {len(panel['states'])} states complete", flush=True)
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
            row = screen_cell(model, lanes, protocol, runtime)
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.strict_score_proof_confirmation_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
