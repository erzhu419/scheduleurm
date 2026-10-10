"""Develop joint-queue canonical bounds on the failed K144 confirmation roster."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_queue_aware_canonical_development_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_k144_confirmation_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
BUDGET = 144
ORACLE_STATE_BUDGET = 5_000_000
ORACLE_WITNESSES = (
    {"name": "r64_tie_unclosed", "node": "gpu_a", "lanes": 64,
     "queue": (153, 147, 290), "state_id": "fresh:2026092751:63"},
    {"name": "r64_canonical_mismatch", "node": "gpu_a", "lanes": 64,
     "queue": (81, 31, 48), "state_id": "fresh:2026092752:63"},
    {"name": "r64_reference_unclosed", "node": "gpu_b", "lanes": 64,
     "queue": (101, 9, 59), "state_id": "fresh:2026092754:18"},
)
EXTRA_SOURCES = (
    "algorithm/queue_aware_canonical_integer_capacity_frame_candidates.py",
    "algorithm/experiments/integer_capacity_canonical_oracle.py",
    "algorithm/experiments/aligned_hardware_queue_aware_canonical_development.py",
    "algorithm/experiments/queue_aware_canonical_development_summary.py",
)


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if not prior["certificate_checks_passed"]:
        raise ValueError("the prior recorded calls must be valid before reusing their references")
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen predecessor source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen catalogue models changed")
    panels = deepcopy(previous["queue_panels"])
    for registered in panels.values():
        for panel in registered:
            panel["cohort"] = "reused_confirmation" if panel["cohort"] == "fresh" else "old_regression"
            for state in panel["states"]:
                state["cohort"] = panel["cohort"]
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "queue_aware_canonical_development_v1",
        "models": previous["models"], "lanes": previous["lanes"],
        "queue_panels": panels, "labels": ["legacy", "queue_aware"], "budget": BUDGET,
        "reference_budget": previous["reference_budget"],
        "execution_order": previous["execution_order"],
        "prior_registered_at_utc": previous["registered_at_utc"],
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)),
        "prior_decision": prior["decision"],
        "oracle_witnesses": ORACLE_WITNESSES, "oracle_state_budget": ORACLE_STATE_BUDGET,
        "calls": "One new cold queue-aware selector call atK144 per prior state. Reuse the frozen predecessor K144 action/work and full K4096 reference records; neither is rerun or charged as a new execution. The three registered failed states receive separate exact original-option count searches, each capped at5 million new DFS states. No reference or candidate LP budget escalation.",
        "reference_policy": "Keep original reference completeness unchanged. Independently search the canonical action on the score-optimal face, conditional on the full reference's valid score_exact certificate, using exact Fraction duration/score arithmetic. Require every registered oracle complete and consistent with any already-complete full reference. This new experiment may use a completed independent canonical reference; no historical confirmation verdict is rewritten and a candidate's own certificate never supplies its reference.",
        "main_gate": "All recorded and new selections valid; all three independent oracles complete without contradiction; effective references complete for all states; queue-aware canonical certificate and exact score/duration/original-index agreement on every state. Any honest candidate nonclosure or mismatch remains ACTION_HOLD.",
        "cost_gate": "New LP and matrix totals no greater than the reused predecessor in every node/R cell, and both strictly lower pooled. Retain per-state regressions. Charge the new joint-queue-bound integer work separately; single-call elapsed times are diagnostic and are not compared with historical wall times.",
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "All4420 states are reused development/regression evidence, including the failed confirmation; none is fresh confirmation of the new selector. Four fixed cached point catalogues with virtual identical lanes. Oracle canonical exactness is conditional on the independently recorded full-reference score certificate. Original INCOMPLETE_REFERENCE and trajectory/decision-delay utility HOLD remain unchanged; no latency or flow-time improvement claim.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, models, development roster or budgets changed")


def selection_from_record(payload):
    from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
    return CanonicalCountSpaceBnbSelection(**{
        key: tuple(value) if key in ("option_counts", "option_indices", "service") else value
        for key, value in payload.items()
    })


def load_prior_cell(protocol, node, lanes):
    with gzip.open(_cell_path(PRIOR_OUTPUT, node, lanes), "rt", encoding="utf-8") as source:
        saved = json.load(source)
    if saved["registered_at_utc"] != protocol["prior_registered_at_utc"]:
        raise ValueError("historical cell registration differs from the frozen source protocol")
    row = saved["cell"]
    expected = [(state["state_id"], panel["panel"], tuple(state["queue"]))
                for panel in protocol["queue_panels"][str(lanes)] for state in panel["states"]]
    actual = [(state["state_id"], state["panel"], tuple(state["queue"])) for state in row["states"]]
    if (row["node"], row["lanes"]) != (node, lanes) or actual != expected:
        raise ValueError("historical cell does not match the registered development roster")
    return row


def run_oracles(models, protocol, prior_rows):
    from algorithm.experiments.integer_capacity_canonical_oracle import (
        canonical_at_certified_score, certified_score_from_selection,
    )
    results = []
    for witness in protocol["oracle_witnesses"]:
        model = next(model for model in models if model.node == witness["node"])
        row = next(row for row in prior_rows if (row["node"], row["lanes"]) == (witness["node"], witness["lanes"]))
        state = next(state for state in row["states"] if state["state_id"] == witness["state_id"])
        if tuple(state["queue"]) != tuple(witness["queue"]):
            raise ValueError("independent oracle witness queue differs from its frozen record")
        reference = selection_from_record(state["reference"])
        score = certified_score_from_selection(model.options, state["queue"], witness["lanes"], state["reference"])
        started = perf_counter()
        result = canonical_at_certified_score(
            model.options, state["queue"], witness["lanes"], score,
            state_visit_budget=protocol["oracle_state_budget"], maximum_duration=reference.duration,
        )
        results.append({**witness, "result": result, "operational_elapsed_s": perf_counter() - started})
        print(f"oracle {witness['name']}: {result['status']}", flush=True)
    return results


def oracle_key(result, option_count):
    if not (result["complete"] and result["canonical_exact"]):
        return None
    return (-Fraction(*result["score_fraction"]), Fraction(*result["duration_fraction"]),
            tuple(option_count if index is None else index for index in result["option_indices"]))


def screen_cell(model, lanes, protocol, previous, oracles, runtime, selector=None, progress=True):
    if selector is None:
        from algorithm.queue_aware_canonical_integer_capacity_frame_candidates import select_queue_aware_canonical_integer_capacity_frame
        selector = select_queue_aware_canonical_integer_capacity_frame
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
            legacy = deepcopy(state["budgets"][str(BUDGET)]["integer_capacity"])
            preprocessing = legacy["metrics"]["preprocessing"]
            legacy["metrics"] = runtime._metrics()
            runtime._record(legacy["metrics"], selection_from_record(legacy["selection"]), model, queue, lanes,
                            BUDGET, expected, preprocessing, reference)
            call_started = perf_counter()
            result = selector(model.options, queue, lanes, BUDGET)
            elapsed = perf_counter() - call_started
            metrics = runtime._metrics()
            runtime._record(metrics, result.selection, model, queue, lanes, BUDGET, expected,
                            asdict(result.preprocessing), reference, elapsed)
            row["states"].append({
                **registered, "panel": panel["panel"], "cohort": panel["cohort"],
                "reference": state["reference"], "reference_metrics": state["reference_metrics"],
                "reference_complete": state["reference_complete"],
                "effective_reference_complete": expected is not None,
                "effective_reference_source": source, "oracle_crosscheck_failed": contradiction,
                "legacy": legacy,
                "queue_aware": {"selection": asdict(result.selection), "metrics": metrics,
                                "capacity_propagation": asdict(result.capacity_propagation),
                                "canonical_queue_bound": asdict(result.canonical_queue_bound)},
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
    oracle_path = args.output.with_name(args.output.stem + "_oracles.json")
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
        for path in [oracle_path, *(_cell_path(args.output, node, lanes) for node, lanes in protocol["execution_order"])]:
            if path.exists():
                raise FileExistsError(path)
        prior_rows = [load_prior_cell(protocol, node, lanes) for node, lanes in protocol["execution_order"]]
        oracles = run_oracles(models, protocol, prior_rows)
        runtime._write_new(oracle_path, {"registered_at_utc": protocol["registered_at_utc"], "oracles": oracles})
    else:
        saved = json.loads(oracle_path.read_text(encoding="utf-8"))
        if saved["registered_at_utc"] != protocol["registered_at_utc"]:
            raise ValueError("oracle registration differs from protocol")
        oracles = saved["oracles"]
    rows = []
    for index, (node, lanes) in enumerate(protocol["execution_order"]):
        path = _cell_path(args.output, node, lanes)
        if args.merge_cells:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                saved = json.load(source)
            if saved["registered_at_utc"] != protocol["registered_at_utc"]:
                raise ValueError("cell registration differs from protocol")
            row = saved["cell"]
        else:
            model = next(model for model in models if model.node == node)
            row = screen_cell(model, lanes, protocol, prior_rows[index], oracles, runtime)
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.queue_aware_canonical_development_summary import summarize
    result = summarize(models, protocol, rows, oracles)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
