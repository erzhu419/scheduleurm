"""Fresh-queue confirmation with frozen LP and canonical-completion budgets."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_score_face_completion_confirmation_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_score_face_completion_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
STATIC_PROTOCOLS = (
    ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_development_20260927_protocol.json",
    ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_queue_confirmation_20260927_protocol.json",
)
FRONTIER_PROTOCOL = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_budget_frontier_development_20260927_protocol.json"
LANES = (8, 16, 32, 64)
BUDGET = 144
CANONICAL_STATE_BUDGET = 100_000
REFERENCE_BUDGET = 4096
ORACLE_STATE_BUDGET = 5_000_000
PANEL_SEEDS = (2026092761, 2026092762, 2026092763, 2026092764)
QUEUES_PER_PANEL = 64
KNOWN_COMPLETION_QUEUES = ((153, 147, 290), (81, 31, 48), (101, 9, 59))
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_score_face_completion_confirmation.py",
    "algorithm/experiments/score_face_completion_confirmation_summary.py",
)


def static_exclusions(previous):
    excluded = {str(lanes): {tuple(state["queue"])
                            for panel in previous["queue_panels"][str(lanes)]
                            for state in panel["states"]} for lanes in LANES}
    for path in STATIC_PROTOCOLS:
        old = json.loads(path.read_text(encoding="utf-8"))
        for lanes in LANES:
            excluded[str(lanes)].update(tuple(queue)
                                       for panel in old["queue_panels"].get(str(lanes), [])
                                       for queue in panel["queues"])
    frontier = json.loads(FRONTIER_PROTOCOL.read_text(encoding="utf-8"))
    for lanes in LANES:
        excluded[str(lanes)].update(tuple(state["queue"])
                                   for panel in frontier["queue_panels"][str(lanes)]
                                   for state in panel["states"])
    return excluded


def queue_design(previous_panels, excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues
    panels = {}
    for lanes in LANES:
        seen = set(excluded[str(lanes)])
        panels[str(lanes)] = []
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
                panels[str(lanes)].append(deepcopy(old))
        if lanes == 64:
            panels[str(lanes)].append({
                "panel": "known_completion", "seed": None, "cohort": "known_completion_regression",
                "states": [{"state_id": f"completion:{index}", "queue": queue}
                           for index, queue in enumerate(KNOWN_COMPLETION_QUEUES)],
            })
    return panels


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if prior["decision"] != "SCORE_FACE_COMPLETION_DEVELOPMENT_ADVANCE":
        raise ValueError("completed score-face development is required before confirmation")
    if (previous["budget"], previous["canonical_state_budget"]) != (BUDGET, CANONICAL_STATE_BUDGET):
        raise ValueError("confirmation must retain the developed budget pair")
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen development source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen catalogue models changed")
    excluded = static_exclusions(previous)
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "score_face_completion_fresh_confirmation_v1",
        "models": previous["models"], "lanes": LANES,
        "budget": BUDGET, "canonical_state_budget": CANONICAL_STATE_BUDGET,
        "reference_budget": REFERENCE_BUDGET, "oracle_state_budget": ORACLE_STATE_BUDGET,
        "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "queue_panels": queue_design(previous["queue_panels"], excluded),
        "excluded_static_queues": {lanes: sorted(queues) for lanes, queues in excluded.items()},
        "excluded_protocol_paths": [str(path.relative_to(ROOT))
                                    for path in (*STATIC_PROTOCOLS, FRONTIER_PROTOCOL, PRIOR_PROTOCOL)],
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "prior_registered_at_utc": previous["registered_at_utc"],
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)), "prior_decision": prior["decision"],
        "sampler": "For each R use random.Random(panel_seed+1000*R), each coordinate randint(0,5R). Reject the four listed static protocol rosters, native/scaled grids and earlier accepted draws at that R. Four new panels of64 distinct queues per R, paired across four nodes, with no cross-R coupling. No claim of disjointness from all historical trajectory states.",
        "regression_roster": "Retain all324 node-states from the three old radius-one hard panels; add the three known completion queues atR64 across all four catalogues (12 node-states). These336 states are separate regression data and cannot substitute for the4096 fresh node-states.",
        "calls": "For each state run one cold full K4096 reference, its conditionally triggered independent oracle, then one cold two-stage selector. The original integer-capacity stage-one result is the embedded baseline; it is neither a second cold call nor an independent timing baseline. Two LP selector calls per state; report reference-oracle calls and work separately. Finish all panels after honest nonclosure or mismatch without replacing seeds or retuning caps.",
        "trigger": previous["trigger"],
        "reference_policy": "Run the independent original-option integer oracle only when the full K4096 reference has score_exact and not canonical_exact, before the candidate call. Use only that full reference's proved exact score and incumbent duration, with a fixed5000000-state cap. Preserve the original reference nonclosure. A completed oracle can provide a conditional canonical reference; an unproved full-reference score or exhausted oracle remains INCOMPLETE_REFERENCE. Candidate output never supplies or changes its own reference.",
        "main_gate": "Complete registered coverage, valid feasibility/certificate/accounting records, LP<=144 and integer states<=100000 separately, all effective references complete. Every fresh final output must be canonical exact and match reference score/duration/original indices; apply the same gate separately to both regression cohorts. Overall advance requires both fresh and regression gates. Failed gates remain HOLD, incomplete references remain incomplete, and invalid execution is distinct.",
        "accounting": "The method budget is the pair(144 LP calls,100000 new integer states). Preserve stage-one LP/preprocessing/capacity work and report extra integer search work separately, including exhausted work. Original full-reference and conditional oracle costs are reference costs. No speed gate; individual elapsed times are operational diagnostics only.",
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Only queue panels are new under a fixed sampling law; algorithms, budgets and four cached physical catalogues remain fixed. R8/16/32/64 are virtual identical lanes. Queues are paired across catalogues, not4096 independent observations. Oracle and completion are separate implementations sharing integer enumeration mathematics; the oracle relies on the full reference's score certificate. This is correctness confirmation, not a minimum/universal budget, new physical scaling, runtime or flow-time benefit claim. Previous failed confirmation and flow utility HOLD remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, models, queues, reference rule or budget pair changed")


def screen_cell(model, lanes, protocol, runtime, selector=None, reference_oracle=None, progress=True):
    from algorithm.experiments.integer_capacity_canonical_oracle import (
        canonical_at_certified_score, certified_score_from_selection,
    )
    if selector is None:
        from algorithm.score_face_canonical_completion_frame_candidates import select_score_face_canonical_completion_frame
        selector = select_score_face_canonical_completion_frame
    if reference_oracle is None:
        reference_oracle = canonical_at_certified_score
    started = perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "option_count": len(model.options), "registered_at_utc": protocol["registered_at_utc"],
           "states": [], "runtime_start": runtime_provenance()}
    for panel in protocol["queue_panels"][str(lanes)]:
        for registered in panel["states"]:
            queue = tuple(registered["queue"])
            call_started = perf_counter()
            reference = runtime.select_canonical_count_space_bnb_frame(
                model.options, queue, lanes, protocol["reference_budget"])
            reference_elapsed = perf_counter() - call_started
            expected = runtime.reference_key(reference, queue, len(model.options))
            original_complete = expected is not None
            source = "full_K4096" if original_complete else None
            reference_metrics = runtime._metrics()
            runtime._record(reference_metrics, reference, model, queue, lanes,
                            protocol["reference_budget"], expected, elapsed_s=reference_elapsed)
            independent, oracle_elapsed = None, 0.0
            if reference.score_exact and not reference.canonical_exact:
                score = certified_score_from_selection(model.options, queue, lanes, asdict(reference))
                call_started = perf_counter()
                independent = reference_oracle(model.options, queue, lanes, score,
                    state_visit_budget=protocol["oracle_state_budget"], maximum_duration=reference.duration)
                oracle_elapsed = perf_counter() - call_started
                expected = oracle_key(independent, len(model.options))
                source = "independent_score_face_oracle" if expected is not None else None
            call_started = perf_counter()
            result = selector(model.options, queue, lanes, protocol["budget"], protocol["canonical_state_budget"])
            elapsed = perf_counter() - call_started
            preprocessing, capacity = asdict(result.preprocessing), asdict(result.capacity_propagation)
            baseline_metrics, metrics = runtime._metrics(), runtime._metrics()
            runtime._record(baseline_metrics, result.baseline_selection, model, queue, lanes,
                            protocol["budget"], expected, preprocessing, reference)
            runtime._record(metrics, result.selection, model, queue, lanes, protocol["budget"], expected,
                            preprocessing, reference, elapsed)
            row["states"].append({
                **registered, "panel": panel["panel"], "cohort": panel["cohort"],
                "reference": asdict(reference), "reference_metrics": reference_metrics,
                "reference_elapsed_s": reference_elapsed, "reference_complete": original_complete,
                "oracle": independent, "oracle_elapsed_s": oracle_elapsed,
                "effective_reference_complete": expected is not None, "effective_reference_source": source,
                "baseline_source": "embedded_stage_one",
                "baseline": {"selection": asdict(result.baseline_selection), "metrics": baseline_metrics,
                             "capacity_propagation": capacity},
                "completed": {"selection": asdict(result.selection), "metrics": metrics,
                              "baseline_selection": asdict(result.baseline_selection),
                              "capacity_propagation": capacity, "completion": asdict(result.completion)},
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
    from algorithm.experiments.score_face_completion_confirmation_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
