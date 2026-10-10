"""Develop a shared-budget integer score proof on retained failures and neighbors."""
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
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record, oracle_key


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_strict_score_proof_development_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_score_face_completion_confirmation_20260927.json"
EARLIER_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_score_face_completion_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
EARLIER_PROTOCOL = EARLIER_OUTPUT.with_name(EARLIER_OUTPUT.stem + "_protocol.json")
BUDGET = 144
INTEGER_STATE_BUDGET = 100_000
WITNESS_QUEUE = (86, 87, 146)
EXTRA_SOURCES = (
    "algorithm/strict_score_proof_canonical_completion_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_strict_score_proof_development.py",
    "algorithm/experiments/strict_score_proof_development_summary.py",
)


def queue_design(current, earlier):
    panels = {}
    for lanes in current["lanes"]:
        key, seen = str(lanes), set()
        panels[key] = []
        for source, previous in (("current", current), ("earlier", earlier)):
            for old in previous["queue_panels"][key]:
                states = []
                for state in old["states"]:
                    queue = tuple(state["queue"])
                    if queue in seen:
                        continue
                    seen.add(queue)
                    states.append({"state_id": f"{source}:{state['state_id']}", "queue": queue,
                                   "source": source, "source_state_id": state["state_id"]})
                if states:
                    cohort = ("reused_confirmation" if old["cohort"] == "fresh_confirmation"
                              else old["cohort"]) if source == "current" else "earlier_development"
                    panels[key].append({"panel": f"{source}:{old['panel']}", "cohort": cohort,
                                        "seed": old.get("seed"), "states": states})
        if lanes == 32:
            neighbors = [tuple(q + delta for q, delta in zip(WITNESS_QUEUE, offsets))
                         for offsets in product((-1, 0, 1), repeat=3)]
            panels[key].append({"panel": "score_failure_neighbors", "cohort": "new_neighborhood",
                "seed": None, "states": [
                    {"state_id": "neighbor:" + ":".join(map(str, queue)), "queue": queue,
                     "source": "new_neighborhood", "source_state_id": None}
                    for queue in neighbors if queue not in seen]})
    return panels


def build_protocol(models):
    current, earlier = (json.loads(path.read_text(encoding="utf-8"))
                        for path in (PRIOR_PROTOCOL, EARLIER_PROTOCOL))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in (PRIOR_OUTPUT, EARLIER_OUTPUT)]
    if not all(result["certificate_checks_passed"] and result["effective_references_complete"] for result in results):
        raise ValueError("valid prior execution and complete effective references are required")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    if snapshots != current["models"] or snapshots != earlier["models"]:
        raise ValueError("frozen catalogue models changed")
    if any((prior["budget"], prior["canonical_state_budget"]) != (BUDGET, INTEGER_STATE_BUDGET)
           for prior in (current, earlier)):
        raise ValueError("retain the existing LP/integer budget pair")
    for path, frozen in current["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen source changed: {path}")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "strict_score_proof_development_v1",
        "models": snapshots, "lanes": current["lanes"], "budget": BUDGET,
        "integer_state_budget": INTEGER_STATE_BUDGET,
        "reference_budget": current["reference_budget"], "oracle_state_budget": current["oracle_state_budget"],
        "queue_panels": queue_design(current, earlier), "execution_order": current["execution_order"],
        "prior_sources": {label: {"result_path": str(path.relative_to(ROOT)),
                                    "registered_at_utc": protocol["registered_at_utc"], "decision": result["decision"]}
                          for label, path, protocol, result in zip(
                              ("current", "earlier"), (PRIOR_OUTPUT, EARLIER_OUTPUT), (current, earlier), results)},
        "roster": "Deduplicate the two existing rosters byR/Q, prefer the latest confirmation record on overlaps, and pair queues across catalogues. Add all not-already-present radius-one neighbors of(86,87,146) atR32. Existing confirmation queues become development data; no new confirmation claim.",
        "calls": "One cold new selector per state. Reuse existing reference/oracle records; only new neighbors receive one fullK4096 reference, followed before the candidate by an independent oracle capped at5000000 states if score_exact and not canonical_exact. Stageone is executed once inside the new selector; embedded baseline is not another call.",
        "trigger": "If stageone is canonical exact, skip integer work. If only score exact, retain the existing canonical completion. Otherwise test for strictly better integer actions at every effective durationT using target floor(own incumbent score*T)+1. Only exhaustive absence at all thresholds proves the score. A better witness or exhausted proof preserves the original baseline action and flags. After proof, canonical completion uses only the remaining integer state budget; if it exhausts, retain the proved score but not a canonical certificate.",
        "accounting": "Per-call shared cap: proof states plus canonical states<=100000; LP calls<=144. Report all proof and canonical counters separately, including failed work. No LP escalation, reference score inputs, uncharged oracle calls or selector restarts. Individual walltimes diagnostic only.",
        "reproduction": "Every reused state's original LP selection, preprocessing and capacity work must reproduce its saved embedded baseline. When that baseline already proved the score, final selection and canonical work must also reproduce the frozen previous completion. New neighbor baselines have no independent historical comparison.",
        "main_gate": "Complete coverage and effective references, valid certificates and separate/shared budget accounting, required historical reproduction, every final action canonical exact and exactly matching reference score/duration/original indices. Any honest nonclosure is HOLD; incomplete references and invalid execution remain distinct. Complete the frozen roster without retuning or deleting failures.",
        "search_counters": current["search_counters"], "capacity_counters": current["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*current["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Post-failure development with reused panels and a chosen local neighborhood. Four fixed cached physical catalogues and virtual identical lanes. No new confirmation, physical scaling, speed or flow-time benefit claim; previous HOLD results retained. Conditional reference oracles share integer enumeration principles but are separate implementations and use only full-reference certified scores.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, models, development roster or shared budget changed")


def load_prior_states(protocol, node, lanes):
    sources = {}
    for label in ("current", "earlier"):
        info = protocol["prior_sources"][label]
        with gzip.open(_cell_path(ROOT / info["result_path"], node, lanes), "rt", encoding="utf-8") as f:
            saved = json.load(f)
        if saved["registered_at_utc"] != info["registered_at_utc"]:
            raise ValueError("reused cell differs from registered predecessor")
        sources[label] = {state["state_id"]: state for state in saved["cell"]["states"]}
    return sources


def _new_reference(model, queue, lanes, protocol, runtime, reference_oracle):
    from algorithm.experiments.integer_capacity_canonical_oracle import certified_score_from_selection
    started = perf_counter()
    reference = runtime.select_canonical_count_space_bnb_frame(model.options, queue, lanes, protocol["reference_budget"])
    elapsed = perf_counter() - started
    expected = runtime.reference_key(reference, queue, len(model.options))
    metrics = runtime._metrics()
    runtime._record(metrics, reference, model, queue, lanes, protocol["reference_budget"], expected, elapsed_s=elapsed)
    independent = None
    if reference.score_exact and not reference.canonical_exact:
        score = certified_score_from_selection(model.options, queue, lanes, asdict(reference))
        independent = reference_oracle(model.options, queue, lanes, score,
            state_visit_budget=protocol["oracle_state_budget"], maximum_duration=reference.duration)
    return reference, metrics, expected is not None, independent


def screen_cell(model, lanes, protocol, sources, runtime, selector=None, reference_oracle=None, progress=True):
    from algorithm.experiments.score_face_completion_development_summary import _selection_equal
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
            reused = registered["source"] != "new_neighborhood"
            previous = None
            if reused:
                old = sources[registered["source"]][registered["source_state_id"]]
                if tuple(old["queue"]) != queue:
                    raise ValueError("reused queue differs from registered queue")
                reference = selection_from_record(old["reference"])
                reference_metrics, reference_complete = old["reference_metrics"], old["reference_complete"]
                independent = old.get("oracle")
                if not reference_complete and registered["source"] == "earlier":
                    raise ValueError("earlier incomplete references must use their current overlapping record")
                previous = {"baseline_selection": old["completed"]["baseline_selection"],
                            "selection": old["completed"]["selection"], "completion": old["completed"]["completion"],
                            "preprocessing": old["completed"]["metrics"]["preprocessing"],
                            "capacity_propagation": old["completed"]["capacity_propagation"]}
            else:
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
            reproduced = None if previous is None else (
                _selection_equal(asdict(result.baseline_selection), previous["baseline_selection"])
                and prep == previous["preprocessing"] and capacity == previous["capacity_propagation"])
            completion_reproduced = None
            if previous is not None and previous["baseline_selection"]["score_exact"]:
                completion_reproduced = (_selection_equal(asdict(result.selection), previous["selection"])
                                        and json.loads(json.dumps(asdict(result.completion))) == previous["completion"])
            row["states"].append({
                **registered, "panel": panel["panel"], "cohort": panel["cohort"],
                "reference": asdict(reference), "reference_metrics": reference_metrics,
                "reference_complete": reference_complete, "oracle": independent, "reference_reused": reused,
                "effective_reference_complete": expected is not None, "effective_reference_source": source,
                "previous": previous, "baseline_reproduced": reproduced,
                "completion_reproduced": completion_reproduced,
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
            row = screen_cell(model, lanes, protocol, load_prior_states(protocol, node, lanes), runtime)
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.strict_score_proof_development_summary import summarize
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
