"""Frozen cold-budget, four-catalogue R4/R8/R16 development screen.

R4 has an independent exact multiset oracle. Larger lane counts use the full
catalogue's K4096 certificate, with unresolved references retained explicitly.
The lane scaling reuses measured point-model options; it is not a new physical
multi-GPU measurement or a trajectory experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
from time import perf_counter

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.experiments.aligned_hardware_physical_catalogue_models import (
    DEFAULT_CACHE, DEFAULT_LOADED_LEDGER, load_catalogues,
)
from algorithm.experiments.aligned_hardware_physical_catalogue_screen import enumerate_frames, oracle, topology
from algorithm.shared_threshold_canonical_requirement_dominance_frame_candidates import (
    select_shared_threshold_canonical_requirement_dominance_frame,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_shared_threshold_scaling_screen_development_20260927.json"
QUEUE_LEVELS = (0, 1, 3, 4, 10)
LANES = (4, 8, 16)
REFERENCE_BUDGET = 4096
CONTROL_BUDGET = 9
COUNTERS = ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations", "duration_root_count",
            "dual_column_checks", "branch_splits", "bound_pruned_regions",
            "safe_fallback_lp_calls", "residual_fill_option_probes")


def scaled_queues(lanes, levels=QUEUE_LEVELS):
    return tuple(tuple(value * (lanes // 2) for value in queue) for queue in product(levels, repeat=3))


def build_protocol(models):
    return json.loads(json.dumps({
        "schema_version": 1,
        "experiment": "shared_threshold_physical_catalogue_scaling_development_v1",
        "sources": [str(path.relative_to(ROOT)) for path in (DEFAULT_CACHE, DEFAULT_LOADED_LEDGER)],
        "models": [model.snapshot() for model in models],
        "lanes": LANES, "base_queue_levels": QUEUE_LEVELS,
        "queue_scaling": "multiply every base coordinate by R//2; all 125 queues at each R",
        "reference_budget": REFERENCE_BUDGET,
        "budgets": {str(lanes): {"control": CONTROL_BUDGET, "deployment": 36 * lanes} for lanes in LANES},
        "calls": "five independent cold calls per state: full reference, full/shared K9, full/shared K36R",
        "accounting": "reference excluded from comparison; cold preprocessing separate; failed LP calls charged; no counter subtraction across budgets",
        "search_counters": COUNTERS,
        "oracle": "R4 full-catalogue Fraction multiset enumeration; R8/R16 exact full K4096 reference only when all exact flags hold; unresolved references retained without budget escalation",
        "validity": "all outputs feasible in original catalogue; reported score equals exact binary-float-duration score; bounds cover certified optimum; score_exact and canonical_exact claims independently checked",
        "advance_gate": "complete certified references; both selectors canonical_exact at deployment cap and actions/scores match every state; shared deployment LP and matrix work no greater than full in each node/R cell, with both strictly lower pooled",
        "limitations": "Cached point-model development, with virtual identical-lane scaling. No new physical measurement, joint95 transfer, independent confirmation, trajectory, flow-time, or wall-time claim. K9 is a low-budget control, not a deployment correctness gate.",
    }))


def selection_score(selected, queue):
    return Fraction(sum(q * s for q, s in zip(queue, selected.service, strict=True))) / Fraction.from_float(selected.duration)


def reference_key(selected, queue, option_count):
    if not (selected.exact and selected.score_exact and selected.canonical_exact):
        return None
    return (-selection_score(selected, queue), Fraction.from_float(selected.duration),
            tuple(option_count if index is None else index for index in selected.option_indices))


def check_selection(selected, options, queue, lanes, budget, expected=None, reference=None):
    """Audit certificates without treating an uncertified action as exact.

    A certified exact score publishes its nearest float, so reconstruct that
    bound from the selected action. An inexact bound must cover the exact
    optimum as its published binary-float fraction, without rounding tolerance.
"""
    counts = selected.option_counts
    indices = tuple(index for index, count in enumerate(counts) for _ in range(count))
    indices += (None,) * (lanes - len(indices))
    service = tuple(sum(count * option.requirements[c] for count, option in zip(counts, options, strict=True))
                    for c in range(len(queue)))
    duration = max((1.0, *(float(option.duration) for count, option in zip(counts, options, strict=True) if count)))
    score = selection_score(selected, queue)
    upper = score if selected.score_exact else Fraction.from_float(selected.upper_bound)
    checks = {
        "invalid_counts": any(count < 0 or int(count) != count for count in counts) or sum(counts) > lanes,
        "index_count_mismatch": selected.option_indices != indices,
        "service_count_mismatch": selected.service != service,
        "infeasible_service": any(done > waiting for done, waiting in zip(service, queue, strict=True)),
        "duration_count_mismatch": selected.duration != duration,
        "reported_score_mismatch": selected.score != float(score),
        "bound_below_incumbent": upper < score,
        "gap_mismatch": selected.gap != max(0.0, selected.upper_bound - selected.score),
        "budget_exceeded": selected.lp_calls > budget,
        "budget_mismatch": selected.lp_node_budget != budget,
        "inconsistent_exact_flags": selected.exact != selected.canonical_exact or (selected.canonical_exact and not selected.score_exact),
        "exact_score_with_gap": selected.score_exact and (selected.gap != 0.0 or selected.upper_bound != selected.score),
    }
    if expected is not None:
        optimum = -expected[0]
        actual_key = (Fraction.from_float(selected.duration),
                      tuple(len(options) if index is None else index for index in selected.option_indices))
        checks.update({
            "score_above_optimum": score > optimum,
            "bound_below_optimum": upper < optimum,
            "false_score_exact": selected.score_exact and score != optimum,
            "false_canonical_exact": selected.canonical_exact and (score != optimum or actual_key != expected[1:]),
        })
    elif reference is not None:
        reference_score = selection_score(reference, queue)
        reference_upper = reference_score if reference.score_exact else Fraction.from_float(reference.upper_bound)
        checks.update({
            "bound_below_reference_incumbent": upper < reference_score,
            "score_above_reference_bound": score > reference_upper,
            "false_score_exact": reference.score_exact and selected.score_exact and score != reference_score,
        })
    return checks


def _metrics():
    return {
        "calls": 0, "failures": 0, "first_failure": None,
        "score_exact_count": 0, "canonical_exact_count": 0,
        "cap_reached_count": 0, "uncertified_cap_reached_count": 0,
        "reference_comparable_count": 0, "score_mismatch_count": 0,
        "score_reference_comparable_count": 0,
        "canonical_action_mismatch_count": 0, "first_mismatch": None,
        "max_relative_score_deficit": 0.0, "relative_score_deficit_sum": 0.0,
        "gap_sum": 0.0, "max_gap": 0.0,
        "search_work": dict.fromkeys(COUNTERS, 0), "preprocessing": {},
        "operational_elapsed_s": 0.0,
    }


def _record(metrics, selected, model, queue, lanes, budget, expected, preprocessing=None, reference=None, elapsed_s=0.0):
    checks = check_selection(selected, model.options, queue, lanes, budget, expected, reference)
    metrics["calls"] += 1
    metrics["operational_elapsed_s"] += elapsed_s
    metrics["failures"] += bool(any(checks.values()))
    if any(checks.values()) and metrics["first_failure"] is None:
        metrics["first_failure"] = {"queues": queue, "checks": {key: value for key, value in checks.items() if value},
                                    "selection": asdict(selected)}
    metrics["score_exact_count"] += selected.score_exact
    metrics["canonical_exact_count"] += selected.canonical_exact
    metrics["cap_reached_count"] += selected.lp_calls == budget
    metrics["uncertified_cap_reached_count"] += selected.lp_calls == budget and not selected.canonical_exact
    metrics["gap_sum"] += selected.gap
    metrics["max_gap"] = max(metrics["max_gap"], selected.gap)
    for counter in COUNTERS:
        metrics["search_work"][counter] += getattr(selected, counter)
    for key, value in (preprocessing or {}).items():
        metrics["preprocessing"][key] = metrics["preprocessing"].get(key, 0) + value
    optimum = -expected[0] if expected is not None else (
        selection_score(reference, queue) if reference is not None and reference.score_exact else None)
    if optimum is None:
        return
    metrics["score_reference_comparable_count"] += 1
    metrics["reference_comparable_count"] += expected is not None
    score = selection_score(selected, queue)
    indices = tuple(len(model.options) if index is None else index for index in selected.option_indices)
    mismatch = (score != optimum, expected is not None and (Fraction.from_float(selected.duration), indices) != expected[1:])
    metrics["score_mismatch_count"] += mismatch[0]
    metrics["canonical_action_mismatch_count"] += mismatch[1]
    deficit = float((optimum - score) / optimum) if optimum else 0.0
    metrics["relative_score_deficit_sum"] += deficit
    metrics["max_relative_score_deficit"] = max(metrics["max_relative_score_deficit"], deficit)
    if any(mismatch) and metrics["first_mismatch"] is None:
        metrics["first_mismatch"] = {
            "queues": queue, "indices": selected.option_indices,
            "expected_indices": expected[2] if expected is not None else None, "score_fraction": str(score),
            "expected_score_fraction": str(optimum), "score_exact": selected.score_exact,
            "canonical_exact": selected.canonical_exact,
        }


def screen_cell(model, lanes, protocol, progress=True):
    cell_started = perf_counter()
    budgets = protocol["budgets"][str(lanes)]
    queues = scaled_queues(lanes, protocol["base_queue_levels"])
    frames = enumerate_frames(model.options, lanes) if lanes == 4 else None
    row = {
        "node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
        "queue_count": len(queues), "option_count": len(model.options),
        "requirement_group_count": len({option.requirements for option in model.options}),
        "oracle_queue_count": len(queues) if frames is not None else 0,
        "uncertified_reference_count": 0, "first_uncertified_reference": None,
        "reference": _metrics(),
        "budgets": {name: {
            "lp_budget": budget, "full": _metrics(), "shared": _metrics(),
            "shared_cost_regression_counts": dict.fromkeys(COUNTERS, 0),
            "first_shared_cost_regression": None,
        } for name, budget in budgets.items()},
    }
    for queue in queues:
        started = perf_counter()
        full_reference = select_canonical_count_space_bnb_frame(model.options, queue, lanes, protocol["reference_budget"])
        elapsed = perf_counter() - started
        certified = reference_key(full_reference, queue, len(model.options))
        if certified is None:
            row["uncertified_reference_count"] += 1
            if row["first_uncertified_reference"] is None:
                row["first_uncertified_reference"] = {"queues": queue, "selection": asdict(full_reference)}
        expected = oracle(frames, queue) if frames is not None else certified
        _record(row["reference"], full_reference, model, queue, lanes, protocol["reference_budget"], expected, elapsed_s=elapsed)
        for name, budget in budgets.items():
            cell = row["budgets"][name]
            started = perf_counter()
            full = select_canonical_count_space_bnb_frame(model.options, queue, lanes, budget)
            full_elapsed = perf_counter() - started
            started = perf_counter()
            shared_result = select_shared_threshold_canonical_requirement_dominance_frame(model.options, queue, lanes, budget)
            shared_elapsed = perf_counter() - started
            shared = shared_result.selection
            _record(cell["full"], full, model, queue, lanes, budget, expected, reference=full_reference, elapsed_s=full_elapsed)
            _record(cell["shared"], shared, model, queue, lanes, budget, expected, asdict(shared_result.preprocessing), full_reference, shared_elapsed)
            regressions = {key: getattr(shared, key) > getattr(full, key) for key in COUNTERS}
            for key, regression in regressions.items():
                cell["shared_cost_regression_counts"][key] += regression
            if any(regressions.values()) and cell["first_shared_cost_regression"] is None:
                cell["first_shared_cost_regression"] = {
                    "queues": queue,
                    "full": {key: getattr(full, key) for key, value in regressions.items() if value},
                    "shared": {key: getattr(shared, key) for key, value in regressions.items() if value},
                }
    row["operational_elapsed_s"] = perf_counter() - cell_started
    if progress:
        print(f"{model.node} R{lanes}: {len(queues)} queues; reference unresolved={row['uncertified_reference_count']}; deployment LP full/shared={row['budgets']['deployment']['full']['search_work']['lp_calls']}/{row['budgets']['deployment']['shared']['search_work']['lp_calls']}", flush=True)
    return row


def summarize(models, protocol, rows):
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        raise ValueError("all registered node/R cells must appear exactly once")
    work = {name: {label: {counter: sum(row["budgets"][name][label]["search_work"][counter] for row in rows)
                         for counter in COUNTERS} for label in ("full", "shared")}
            for name in ("control", "deployment")}
    valid = not any(row["reference"]["failures"] or any(
        row["budgets"][name][label]["failures"] for name in ("control", "deployment") for label in ("full", "shared")) for row in rows)
    complete = not any(row["uncertified_reference_count"] for row in rows)
    actions = all(row["budgets"]["deployment"][label]["canonical_exact_count"] == row["queue_count"]
                  and not row["budgets"]["deployment"][label]["score_mismatch_count"]
                  and not row["budgets"]["deployment"][label]["canonical_action_mismatch_count"]
                  for row in rows for label in ("full", "shared"))
    comparison_counters = ("lp_calls", "lp_matrix_coefficients")
    cost = (all(row["budgets"]["deployment"]["shared"]["search_work"][key] <= row["budgets"]["deployment"]["full"]["search_work"][key]
                for row in rows for key in comparison_counters)
            and all(work["deployment"]["shared"][key] < work["deployment"]["full"][key] for key in comparison_counters))
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "DEPLOYMENT_ACTION_REVIEW_REQUIRED" if not actions else
                "MULTI_CATALOGUE_SCALING_ADVANCE" if cost else "SEARCH_COST_REVIEW_REQUIRED")
    return {
        "schema_version": 1, "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models),
        "requirement_topology_count": len({topology(model) for model in models}),
        "queue_count": sum(row["queue_count"] for row in rows),
        "selector_calls": sum(row["queue_count"] for row in rows) * 5,
        "oracle_queue_count": sum(row["oracle_queue_count"] for row in rows),
        "certificate_checks_passed": valid, "references_complete": complete,
        "deployment_action_gate_passed": complete and actions,
        "deployment_search_cost_gate_passed": cost,
        "decision": decision, "comparison_search_work": work,
        "reference_search_work": {key: sum(row["reference"]["search_work"][key] for row in rows) for key in COUNTERS},
        "cells": rows, "limitations": protocol["limitations"],
    }


def run_screen(models, protocol):
    return summarize(models, protocol, [screen_cell(model, lanes, protocol)
                                       for model in models for lanes in protocol["lanes"]])


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _cell_path(output, node, lanes):
    return output.with_name(f"{output.stem}_cell_{node}_r{lanes}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--cell", help="one registered NODE:R cell, suitable for separate worker processes")
    mode.add_argument("--merge-cells", action="store_true")
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
    if args.cell:
        node, lanes_text = args.cell.rsplit(":", 1)
        lanes = int(lanes_text)
        model = next(model for model in models if model.node == node)
        if lanes not in protocol["lanes"]:
            raise ValueError("unregistered lane count")
        output = _cell_path(args.output, node, lanes)
        if output.exists():
            raise FileExistsError(output)
        _write_new(output, {"protocol": protocol, "cell": screen_cell(model, lanes, protocol)})
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.merge_cells:
        rows = []
        for model in models:
            for lanes in protocol["lanes"]:
                saved = json.loads(_cell_path(args.output, model.node, lanes).read_text(encoding="utf-8"))
                if saved["protocol"] != protocol:
                    raise ValueError("cell protocol differs from registered protocol")
                rows.append(saved["cell"])
        result = summarize(models, protocol, rows)
    else:
        result = run_screen(models, protocol)
    result["protocol_path"] = str(protocol_path.resolve())
    _write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "comparison_search_work")}))


if __name__ == "__main__":
    main()
