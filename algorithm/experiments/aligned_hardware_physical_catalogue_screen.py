"""Development gate for real node-local catalogues, with an exact small oracle.

This diagnostic preserves the measured action order, including catalogues
whose shortest requirement representative occurs after a removed action.
It uses cached complete-frame point models, not joint95 physical envelopes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations_with_replacement, product
import json
from pathlib import Path

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.canonical_count_space_bnb_requirement_dominance_frame_candidates import (
    prepare_canonical_count_space_bnb_requirement_dominance_catalogue,
)
from algorithm.experiments.aligned_hardware_physical_catalogue_models import (
    DEFAULT_CACHE, DEFAULT_LOADED_LEDGER, load_catalogues,
)
from algorithm.requirement_dominance_frame_candidates import (
    reduce_requirement_dominated_options,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_physical_catalogue_screen_development_20260927.json"
QUEUE_LEVELS = (0, 1, 3, 4, 10)
LP_BUDGET = 4096
COUNTERS = ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations")


def _json(value):
    return json.loads(json.dumps(value))


def build_protocol(models):
    return _json({
        "schema_version": 1,
        "experiment": "physical_catalogue_canonical_dominance_development_v1",
        "role": "diagnostic development; known reversed-dominance case included",
        "sources": [str(path.relative_to(ROOT)) for path in (DEFAULT_CACHE, DEFAULT_LOADED_LEDGER)],
        "models": [model.snapshot() for model in models],
        "queue_levels": QUEUE_LEVELS,
        "lanes": "native model lanes; plus named R2 diagnostic",
        "lp_node_budget": LP_BUDGET,
        "oracle": "all unordered lane multisets; Fraction.from_float durations; score then duration then original indices, idle last",
        "known_diagnostic": {"node": "node007", "queues": [1, 0, 4], "lanes": 2},
        "requirement_topology": "ordered (loaded, requirements), excluding node labels and durations",
        "advance_gate": "at least two requirement topologies have a nonempty canonically liftable quotient and all oracle checks pass",
        "limitations": "Cached phase point models only. No joint95 transfer, trajectory, flow-time, wall-time, or independent confirmation claim. No R8/R16 scaling before the advance gate.",
    })


def enumerate_frames(options, lanes):
    """Independent oracle: idle is the last index; no B&B helper is reused."""
    idle = len(options)
    frames = []
    for indices in combinations_with_replacement(range(idle + 1), lanes):
        service = tuple(sum(options[i].requirements[c] for i in indices if i != idle)
                        for c in range(len(options[0].requirements)))
        duration = max((Fraction(1), *(Fraction.from_float(float(options[i].duration))
                                     for i in indices if i != idle)))
        frames.append((service, duration, indices))
    return frames


def oracle(frames, queues):
    best = None
    for service, duration, indices in frames:
        if any(done > waiting for done, waiting in zip(service, queues, strict=True)):
            continue
        score = Fraction(sum(waiting * done for waiting, done in
                             zip(queues, service, strict=True))) / duration
        key = (-score, duration, indices)
        if best is None or key < best:
            best = key
    return best


def _indices(indices, idle):
    return tuple(None if i == idle else i for i in indices)


def _witness(model, reduced, queues, lanes, full, quotient):
    full_indices = _indices(full[2], len(model.options))
    mapped = tuple(None if i == len(reduced.options) else reduced.retained_original_indices[i]
                   for i in quotient[2])
    return {
        "queues": list(queues), "lanes": lanes,
        "full_option_indices": full_indices, "reduced_mapped_option_indices": mapped,
        "full_action_ids": [None if i is None else model.options[i].action_id for i in full_indices],
        "reduced_action_ids": [None if i is None else model.options[i].action_id for i in mapped],
        "full_score": float(-full[0]), "reduced_score": float(-quotient[0]),
        "full_score_fraction": str(-full[0]), "reduced_score_fraction": str(-quotient[0]),
        "full_duration": float(full[1]), "reduced_duration": float(quotient[1]),
    }


def screen_catalogue(model, queues):
    reduced = reduce_requirement_dominated_options(model.options)
    try:
        prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(model.options)
        lift_valid = prepared.certificate_lift_proof.full_catalogue_canonical_order_valid
        rejection = None
    except ValueError as error:
        lift_valid, rejection = False, str(error)
    full_frames = enumerate_frames(model.options, model.lanes)
    reduced_frames = enumerate_frames(reduced.options, model.lanes)
    row = {
        "node": model.node, "lanes": model.lanes,
        "original_option_count": len(model.options), "retained_option_count": len(reduced.options),
        "retained_original_indices": reduced.retained_original_indices,
        "removed_original_indices": reduced.removed_original_indices,
        "canonical_lift_valid": lift_valid, "canonical_lift_rejection": rejection,
        "queue_count": len(queues), "support_mismatches": 0, "canonical_action_mismatches": 0,
        "selector_oracle_failures": [], "first_canonical_mismatch": None,
        "full_search_work": dict.fromkeys(COUNTERS, 0),
        "reduced_search_work": dict.fromkeys(COUNTERS, 0),
        "preprocessing": {name: getattr(reduced, name) for name in (
            "reduction_option_probes", "reduction_requirement_coordinate_reads", "reduction_duration_comparisons")},
    }
    for queue in queues:
        full, quotient = oracle(full_frames, queue), oracle(reduced_frames, queue)
        row["support_mismatches"] += full[0] != quotient[0]
        witness = _witness(model, reduced, queue, model.lanes, full, quotient)
        if witness["full_option_indices"] != witness["reduced_mapped_option_indices"]:
            row["canonical_action_mismatches"] += 1
            if row["first_canonical_mismatch"] is None:
                row["first_canonical_mismatch"] = witness
        for label, options, expected in (("full", model.options, full), ("reduced", reduced.options, quotient)):
            selected = select_canonical_count_space_bnb_frame(options, queue, model.lanes, LP_BUDGET)
            actual_score = Fraction(sum(q * s for q, s in zip(queue, selected.service, strict=True))) / Fraction.from_float(selected.duration)
            if (not selected.score_exact or not selected.canonical_exact
                    or actual_score != -expected[0]
                    or selected.option_indices != _indices(expected[2], len(options))):
                row["selector_oracle_failures"].append({"queues": queue, "catalogue": label})
            for counter in COUNTERS:
                row[f"{label}_search_work"][counter] += getattr(selected, counter)
    return row


def topology(model):
    return tuple((option.loaded, option.requirements) for option in model.options)


def run_screen(models, protocol):
    queues = tuple(product(protocol["queue_levels"], repeat=3))
    rows = []
    for model in models:
        rows.append(screen_catalogue(model, queues))
        print(f"{model.node}: {len(queues)} queues completed", flush=True)
    model = next(model for model in models if model.node == "node007")
    reduced = reduce_requirement_dominated_options(model.options)
    diagnostic = protocol["known_diagnostic"]
    full = oracle(enumerate_frames(model.options, diagnostic["lanes"]), diagnostic["queues"])
    quotient = oracle(enumerate_frames(reduced.options, diagnostic["lanes"]), diagnostic["queues"])
    valid = all(not row["support_mismatches"] and not row["selector_oracle_failures"]
                and (not row["canonical_lift_valid"] or not row["canonical_action_mismatches"])
                for row in rows)
    eligible = {topology(model) for model, row in zip(models, rows, strict=True)
                if row["canonical_lift_valid"] and row["removed_original_indices"]}
    return {
        "schema_version": 1, "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"],
        "hardware_catalogue_count": len(models),
        "requirement_topology_count": len({topology(model) for model in models}),
        "canonically_liftable_requirement_topology_count": len(eligible),
        "queue_count": len(queues) * len(models),
        "oracle_checks_passed": valid,
        "decision": ("INVALID_EXECUTION" if not valid else
                     "MULTI_CATALOGUE_SCALING_ADVANCE" if len(eligible) >= 2 else
                     "CANONICAL_RESTORATION_REQUIRED"),
        "catalogues": rows,
        "known_r2_diagnostic": _witness(model, reduced, diagnostic["queues"], diagnostic["lanes"], full, quotient),
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
    resolved_protocol = protocol_path.resolve()
    result["protocol_path"] = str(resolved_protocol.relative_to(ROOT)
                                  if resolved_protocol.is_relative_to(ROOT) else resolved_protocol)
    _write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "requirement_topology_count")}))


if __name__ == "__main__":
    main()
