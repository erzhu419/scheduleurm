"""Native-lane development screen of the threshold canonical quotient.

The full catalogue's exact multiset oracle fixes both support and action order.
All four cached point models and the queue panel are registered before running.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
import json
from pathlib import Path

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_physical_catalogue_models import (
    DEFAULT_CACHE, DEFAULT_LOADED_LEDGER, load_catalogues,
)
from algorithm.experiments.aligned_hardware_physical_catalogue_screen import (
    enumerate_frames, oracle, topology,
)
from algorithm.threshold_canonical_requirement_dominance_frame_candidates import (
    select_threshold_canonical_requirement_dominance_frame,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_threshold_canonical_catalogue_screen_development_20260927.json"
QUEUE_LEVELS = (0, 1, 3, 4, 10)
LP_BUDGET = 4096
COUNTERS = ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations",
            "duration_root_count", "dual_column_checks")


def build_protocol(models):
    return json.loads(json.dumps({
        "schema_version": 1,
        "experiment": "threshold_canonical_physical_catalogue_development_v1",
        "role": "development; includes the known reversed-dominance diagnostic",
        "sources": [str(path.relative_to(ROOT)) for path in (DEFAULT_CACHE, DEFAULT_LOADED_LEDGER)],
        "models": [model.snapshot() for model in models],
        "queue_levels": QUEUE_LEVELS,
        "lanes": "native model lanes; separate named R2 diagnostic",
        "lp_node_budget": LP_BUDGET,
        "search_counters": COUNTERS,
        "preprocessing": "reported separately; cold setup charged on every selector call",
        "oracle": "full action multisets; exact binary-float fractions; score then duration then original indices, idle last",
        "known_diagnostic": {"node": "node007", "queues": [1, 0, 4], "lanes": 2,
                             "expected_full_option_indices": [0, 8]},
        "advance_gate": "all full and quotient score/duration/action/count/service/exact-flag checks pass on at least two distinct requirement topologies with repeated requirements",
        "limitations": "Cached phase point models only. No joint95 transfer, trajectory, flow-time, wall-time, or independent confirmation claim. Native lane counts only in the main panel.",
    }))


def _check_selection(selected, expected, options, queue):
    indices = tuple(None if i == len(options) else i for i in expected[2])
    counts = tuple(expected[2].count(i) for i in range(len(options)))
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    score = Fraction(sum(q * s for q, s in zip(queue, selected.service, strict=True))) / Fraction.from_float(selected.duration)
    return {
        "score_mismatch": score != -expected[0],
        "duration_mismatch": Fraction.from_float(selected.duration) != expected[1],
        "action_mismatch": selected.option_indices != indices,
        "count_mismatch": selected.option_counts != counts,
        "service_mismatch": selected.service != service,
        "uncertified": not (selected.exact and selected.score_exact and selected.canonical_exact),
    }


def screen_catalogue(model, queues, lanes=None):
    lanes = model.lanes if lanes is None else lanes
    frames = enumerate_frames(model.options, lanes)
    row = {
        "node": model.node, "lanes": lanes, "queue_count": len(queues),
        "option_count": len(model.options),
        "requirement_group_count": len({option.requirements for option in model.options}),
        "full_search_work": dict.fromkeys(COUNTERS, 0),
        "threshold_search_work": dict.fromkeys(COUNTERS, 0),
        "preprocessing_total": {}, "preprocessing_first_call": None,
        "full_failures": 0, "threshold_failures": 0, "first_failure": None,
    }
    for queue in queues:
        expected = oracle(frames, queue)
        full = select_canonical_count_space_bnb_frame(model.options, queue, lanes, LP_BUDGET)
        result = select_threshold_canonical_requirement_dominance_frame(
            model.options, queue, lanes, LP_BUDGET)
        preprocessing = asdict(result.preprocessing)
        if row["preprocessing_first_call"] is None:
            row["preprocessing_first_call"] = preprocessing
        for key, value in preprocessing.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row["preprocessing_total"][key] = row["preprocessing_total"].get(key, 0) + value
        for label, selected in (("full", full), ("threshold", result.selection)):
            checks = _check_selection(selected, expected, model.options, queue)
            if any(checks.values()):
                row[f"{label}_failures"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "queues": queue, "selector": label, "checks": checks,
                        "expected_indices": tuple(None if i == len(model.options) else i for i in expected[2]),
                        "actual_indices": selected.option_indices,
                        "expected_score_fraction": str(-expected[0]),
                    }
            for counter in COUNTERS:
                row[f"{label}_search_work"][counter] += getattr(selected, counter)
    return row


def run_screen(models, protocol):
    queues = tuple(product(protocol["queue_levels"], repeat=3))
    rows = []
    for model in models:
        rows.append(screen_catalogue(model, queues))
        print(f"{model.node}: {len(queues)} queues completed", flush=True)
    diagnostic = protocol["known_diagnostic"]
    model = next(model for model in models if model.node == diagnostic["node"])
    witness = screen_catalogue(model, (tuple(diagnostic["queues"]),), diagnostic["lanes"])
    expected = oracle(enumerate_frames(model.options, diagnostic["lanes"]), diagnostic["queues"])
    witness["queues"] = diagnostic["queues"]
    witness["oracle_option_indices"] = expected[2]
    witness["oracle_score_fraction"] = str(-expected[0])
    valid = (not any(row["full_failures"] or row["threshold_failures"] for row in (*rows, witness))
             and list(expected[2]) == diagnostic["expected_full_option_indices"])
    eligible = {topology(model) for model, row in zip(models, rows, strict=True)
                if row["requirement_group_count"] < row["option_count"]}
    return {
        "schema_version": 1, "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models),
        "requirement_topology_count": len({topology(model) for model in models}),
        "quotient_requirement_topology_count": len(eligible),
        "queue_count": len(queues) * len(models), "diagnostic_queue_count": 1,
        "selector_calls": 2 * (len(queues) * len(models) + 1),
        "oracle_checks_passed": valid,
        "decision": ("INVALID_EXECUTION" if not valid else
                     "MULTI_CATALOGUE_SCALING_ADVANCE" if len(eligible) >= 2 else
                     "INSUFFICIENT_DISTINCT_CATALOGUES"),
        "catalogues": rows, "known_r2_diagnostic": witness,
        "limitations": protocol["limitations"],
    }


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    models = load_catalogues()
    expected = build_protocol(models)
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        expected["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_new(protocol_path, expected)
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered models or protocol changed; preserve the old run and register a new output")
    result = run_screen(models, protocol)
    resolved = protocol_path.resolve()
    result["protocol_path"] = str(resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved)
    _write_new(args.output, result)
    print(json.dumps({key: result[key] for key in
                      ("decision", "queue_count", "selector_calls", "requirement_topology_count")}))


if __name__ == "__main__":
    main()
