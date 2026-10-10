"""Test separately budgeted canonical completion after a score-certified K144 search."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_score_face_completion_development_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_queue_aware_canonical_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
PRIOR_ORACLES = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_oracles.json")
BUDGET = 144
CANONICAL_STATE_BUDGET = 100_000
EXTRA_SOURCES = (
    "algorithm/score_face_canonical_completion_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_score_face_completion_development.py",
    "algorithm/experiments/score_face_completion_development_summary.py",
)


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if not all(prior[key] for key in ("certificate_checks_passed", "effective_references_complete", "independent_oracles_complete")):
        raise ValueError("valid completed development references are required")
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen predecessor source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen catalogue models changed")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "score_face_completion_development_v1",
        "models": previous["models"], "lanes": previous["lanes"],
        "queue_panels": previous["queue_panels"], "budget": BUDGET,
        "canonical_state_budget": CANONICAL_STATE_BUDGET,
        "reference_budget": previous["reference_budget"], "oracle_state_budget": previous["oracle_state_budget"],
        "oracle_witnesses": previous["oracle_witnesses"],
        "execution_order": previous["execution_order"],
        "prior_registered_at_utc": previous["registered_at_utc"],
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)),
        "prior_oracles_path": str(PRIOR_ORACLES.relative_to(ROOT)), "prior_decision": prior["decision"],
        "calls": "One cold two-stage selector per registered state. Stage1 is the original integer-capacity K144 selector, without the unsuccessful queue-aware bound. Only a proved optimal score with an unfinished canonical certificate triggers a separately capped100000-state original-option integer search. Count all stage1 LP work and all stage2 search work. No LP escalation, external reference action input or warm start from prior results.",
        "trigger": "The stage2 trigger depends solely on stage1 score_exact and not canonical_exact. A proved canonical stage1 action or an unproved score skips stage2 with zero integer search work. Integer budget exhaustion preserves the stage1 action and certificate flags. A completed integer search certifies the earliest-duration, smallest original-index action at the same exact score.",
        "reference_policy": "Reuse all frozen full K4096 reference records and the three prior independent-implementation canonical references. Keep original full-reference nonclosure visible. No reference or oracle is rerun; no candidate output can supply its own comparison reference. The production integer search does not import the experimental oracle. These are reused development references, not a new independent scientific confirmation.",
        "baseline_gate": "Every new stage1 selection, LP/search counters, preprocessing and capacity-propagation work must exactly reproduce the frozen original integer-capacity record. Retain stage1 frontier state in baseline_selection; a completed stage2 discharges the final unresolved-state fields without inventing LP visits or LP prunes.",
        "main_gate": "Complete coverage; all references and outputs valid; all effective references complete; stage1 reproduced on every state; LP calls<=144 and integer states<=100000 separately on every call; final canonical certificate and exact score/duration/original-index agreement on every state. Any failure remains HOLD or INVALID; no budget or seed replacement.",
        "accounting": "The budget is the pair (144 LP calls,100000 new integer states), not144 LP calls alone. Completion state/branch/memo/prune/resource-read counters reported separately, including failed work. Historical predecessor/reference/oracle calls are reused, not newly executed. Single-call elapsed times are diagnostic only; no total speed or cost-reduction gate.",
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "All4420 states are reused development/regression data; the100000-state cap is chosen after the prior three-witness diagnosis. Fixed cached physical catalogues and virtual identical lanes. Score-face references rely on an external proved score and share mathematical enumeration principles with the separately implemented completion stage. Prior failed confirmation and flow-time utility HOLD remain unchanged; no fresh confirmation, physical scaling or runtime/flow benefit claim.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, models, roster or either budget changed")


def load_oracles(protocol):
    saved = json.loads(PRIOR_ORACLES.read_text(encoding="utf-8"))
    if saved["registered_at_utc"] != protocol["prior_registered_at_utc"]:
        raise ValueError("frozen oracle registration differs from the source protocol")
    return saved["oracles"]


def load_prior_cell(protocol, node, lanes):
    with gzip.open(_cell_path(PRIOR_OUTPUT, node, lanes), "rt", encoding="utf-8") as source:
        saved = json.load(source)
    if saved["registered_at_utc"] != protocol["prior_registered_at_utc"]:
        raise ValueError("historical cell registration differs from the source protocol")
    row = saved["cell"]
    expected = [(state["state_id"], panel["panel"], tuple(state["queue"]))
                for panel in protocol["queue_panels"][str(lanes)] for state in panel["states"]]
    actual = [(state["state_id"], state["panel"], tuple(state["queue"])) for state in row["states"]]
    if (row["node"], row["lanes"]) != (node, lanes) or actual != expected:
        raise ValueError("historical cell does not match the development roster")
    return row


def screen_cell(model, lanes, protocol, previous, oracles, runtime, selector=None, progress=True):
    if selector is None:
        from algorithm.score_face_canonical_completion_frame_candidates import select_score_face_canonical_completion_frame
        selector = select_score_face_canonical_completion_frame
    started = perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "option_count": len(model.options), "states": [], "runtime_start": runtime_provenance()}
    saved_states = {state["state_id"]: state for state in previous["states"]}
    oracle_lookup = {(item["node"], item["lanes"], item["state_id"]): item["result"] for item in oracles}
    for panel in protocol["queue_panels"][str(lanes)]:
        for registered in panel["states"]:
            state = saved_states[registered["state_id"]]
            queue = tuple(registered["queue"])
            reference = selection_from_record(state["reference"])
            expected = runtime.reference_key(reference, queue, len(model.options))
            source = "full_K4096" if expected is not None else None
            independent = oracle_lookup.get((model.node, lanes, registered["state_id"]))
            supplemental = oracle_key(independent, len(model.options)) if independent is not None else None
            contradiction = bool(expected is not None and supplemental is not None and expected != supplemental)
            if expected is None and supplemental is not None:
                expected, source = supplemental, "independent_score_face_oracle"
            call_started = perf_counter()
            result = selector(model.options, queue, lanes, protocol["budget"], protocol["canonical_state_budget"])
            elapsed = perf_counter() - call_started
            preprocessing = asdict(result.preprocessing)
            baseline_metrics = runtime._metrics()
            runtime._record(baseline_metrics, result.baseline_selection, model, queue, lanes,
                            protocol["budget"], expected, preprocessing, reference)
            metrics = runtime._metrics()
            runtime._record(metrics, result.selection, model, queue, lanes, protocol["budget"], expected,
                            preprocessing, reference, elapsed)
            legacy = state["legacy"]
            reproduced = (result.baseline_selection == selection_from_record(legacy["selection"])
                          and preprocessing == legacy["metrics"]["preprocessing"]
                          and asdict(result.capacity_propagation) == legacy["capacity_propagation"])
            row["states"].append({
                **registered, "panel": panel["panel"], "cohort": panel["cohort"],
                "reference": state["reference"], "reference_metrics": state["reference_metrics"],
                "reference_complete": state["reference_complete"],
                "effective_reference_complete": expected is not None, "effective_reference_source": source,
                "oracle_crosscheck_failed": contradiction, "legacy": legacy,
                "baseline_metrics": baseline_metrics, "baseline_reproduced": reproduced,
                "completed": {"selection": asdict(result.selection), "metrics": metrics,
                              "baseline_selection": asdict(result.baseline_selection),
                              "capacity_propagation": asdict(result.capacity_propagation),
                              "completion": asdict(result.completion)},
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
    oracles = load_oracles(protocol)
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
            row = screen_cell(model, lanes, protocol, load_prior_cell(protocol, node, lanes), oracles, runtime)
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.score_face_completion_development_summary import summarize
    result = summarize(models, protocol, rows, oracles)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
