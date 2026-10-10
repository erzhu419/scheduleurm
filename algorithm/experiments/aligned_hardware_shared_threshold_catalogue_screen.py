"""Compare shared and independent threshold search on the frozen native panel."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.experiments.aligned_hardware_physical_catalogue_models import (
    DEFAULT_CACHE, DEFAULT_LOADED_LEDGER, load_catalogues,
)
from algorithm.experiments.aligned_hardware_physical_catalogue_screen import enumerate_frames, oracle, topology
from algorithm.experiments.aligned_hardware_threshold_canonical_catalogue_screen import _check_selection
from algorithm.shared_threshold_canonical_requirement_dominance_frame_candidates import (
    select_shared_threshold_canonical_requirement_dominance_frame,
)
from algorithm.threshold_canonical_requirement_dominance_frame_candidates import (
    select_threshold_canonical_requirement_dominance_frame,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_shared_threshold_catalogue_screen_development_20260927.json"
QUEUE_LEVELS = (0, 1, 3, 4, 10)
LP_BUDGET = 4096
COUNTERS = ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations", "duration_root_count",
            "dual_column_checks", "branch_splits", "bound_pruned_regions")
LABELS = ("full", "independent", "shared")


def build_protocol(models):
    return json.loads(json.dumps({
        "schema_version": 1,
        "experiment": "shared_threshold_physical_catalogue_development_v1",
        "role": "same-panel development comparison after canonical restoration",
        "sources": [str(path.relative_to(ROOT)) for path in (DEFAULT_CACHE, DEFAULT_LOADED_LEDGER)],
        "models": [model.snapshot() for model in models],
        "queue_levels": QUEUE_LEVELS, "lp_node_budget": LP_BUDGET,
        "lanes": "native model lanes; separate named R2 diagnostic",
        "selectors": LABELS, "search_counters": COUNTERS,
        "accounting": "cold catalogue setup/lift work separate from search; failed LP calls count toward budget",
        "oracle": "full catalogue Fraction multiset oracle; score, duration, original indices, counts, service, exact flags",
        "known_diagnostic": {"node": "node007", "queues": [1, 0, 4], "lanes": 2,
                             "expected_full_option_indices": [0, 8]},
        "advance_gate": "all oracle and budget checks pass; shared aggregate LP calls and matrix coefficients do not exceed full baseline for any node, with strict pooled reductions in both",
        "limitations": "Repeated cached point-model development panel, not independent confirmation. No joint95, trajectory, flow-time, or wall-time claim. Larger lane counts are not run here.",
    }))


def screen_catalogue(model, queues, lanes=None):
    lanes = model.lanes if lanes is None else lanes
    frames = enumerate_frames(model.options, lanes)
    row = {
        "node": model.node, "lanes": lanes, "queue_count": len(queues),
        "failures": dict.fromkeys(LABELS, 0), "first_failure": None,
        "search_work": {label: dict.fromkeys(COUNTERS, 0) for label in LABELS},
        "preprocessing_total": {label: {} for label in LABELS[1:]},
        "shared_vs_full_lp_regression_count": 0, "first_lp_regression": None,
    }
    for queue in queues:
        expected = oracle(frames, queue)
        full = select_canonical_count_space_bnb_frame(model.options, queue, lanes, LP_BUDGET)
        independent = select_threshold_canonical_requirement_dominance_frame(model.options, queue, lanes, LP_BUDGET)
        shared = select_shared_threshold_canonical_requirement_dominance_frame(model.options, queue, lanes, LP_BUDGET)
        for label, result in (("independent", independent), ("shared", shared)):
            for key, value in asdict(result.preprocessing).items():
                row["preprocessing_total"][label][key] = row["preprocessing_total"][label].get(key, 0) + value
        for label, selected in zip(LABELS, (full, independent.selection, shared.selection), strict=True):
            checks = _check_selection(selected, expected, model.options, queue)
            checks["budget_exceeded"] = selected.lp_calls > LP_BUDGET
            checks["budget_mismatch"] = selected.lp_node_budget != LP_BUDGET
            if any(checks.values()):
                row["failures"][label] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {"queues": queue, "selector": label, "checks": checks,
                                            "actual_indices": selected.option_indices,
                                            "expected_indices": expected[2]}
            for counter in COUNTERS:
                row["search_work"][label][counter] += getattr(selected, counter)
        if shared.selection.lp_calls > full.lp_calls:
            row["shared_vs_full_lp_regression_count"] += 1
            if row["first_lp_regression"] is None:
                row["first_lp_regression"] = {"queues": queue, "full_lp_calls": full.lp_calls,
                                              "shared_lp_calls": shared.selection.lp_calls}
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
    valid = not any(any(row["failures"].values()) for row in (*rows, witness))
    valid = valid and list(expected[2]) == diagnostic["expected_full_option_indices"]
    totals = {label: {counter: sum(row["search_work"][label][counter] for row in rows)
                      for counter in COUNTERS} for label in LABELS}
    comparable = ("lp_calls", "lp_matrix_coefficients")
    cost_gate = (all(row["search_work"]["shared"][key] <= row["search_work"]["full"][key]
                     for row in rows for key in comparable)
                 and all(totals["shared"][key] < totals["full"][key] for key in comparable))
    return {
        "schema_version": 1, "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models),
        "requirement_topology_count": len({topology(model) for model in models}),
        "queue_count": len(queues) * len(models), "diagnostic_queue_count": 1,
        "selector_calls": len(LABELS) * (len(queues) * len(models) + 1),
        "oracle_and_budget_checks_passed": valid, "search_cost_gate_passed": cost_gate,
        "decision": ("INVALID_EXECUTION" if not valid else
                     "SHARED_SEARCH_SCALING_ADVANCE" if cost_gate else "SEARCH_COST_REVIEW_REQUIRED"),
        "main_panel_search_work": totals, "catalogues": rows, "known_r2_diagnostic": witness,
        "limitations": protocol["limitations"],
    }


def _write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
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
        raise ValueError("registered models or protocol changed; register a new output")
    result = run_screen(models, protocol)
    resolved = protocol_path.resolve()
    result["protocol_path"] = str(resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved)
    _write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "main_panel_search_work")}))


if __name__ == "__main__":
    main()
