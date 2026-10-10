"""Frozen new-arrival confirmation of the unchanged common-window selectors."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random

from algorithm.experiments import aligned_hardware_small_dimension_exact_native_common_cadence as development
from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_OUTPUT = development.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_common_cadence_confirmation_20260929.json"
ARRIVAL_SEEDS = tuple(range(13051, 13059))
ORDER_SEED = 2026092917
SOURCE = "algorithm/experiments/aligned_hardware_small_dimension_exact_native_common_cadence_confirmation.py"


def build_protocol(models):
    """Replace arrivals and paired orders; keep the registered development method."""
    from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals

    result = json.loads(PREVIOUS_OUTPUT.read_text())
    if (result["decision"] != "SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_TIMING_ADVANCE"
            or not all(result[key] for key in ("integration_gate_passed", "runtime_checks_passed",
                "no_overrun_gate_passed", "trajectory_identity_passed", "timing_gate_passed",
                "common_cadence_gate_passed"))):
        raise ValueError("the complete common-cadence development advance is required")
    previous = json.loads(development._protocol_path(PREVIOUS_OUTPUT).read_text())
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("retain the same cached complete-frame catalogues")
    for path, literal in previous["execution_sources"].items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen development source changed: {path}")
    if set(ARRIVAL_SEEDS).intersection(previous["arrival_seeds"]):
        raise ValueError("confirmation arrival seeds must be new")

    tapes, orders = [], {}
    for model in models:
        for lanes in previous["lanes"]:
            for load in previous["loads"]:
                rates = next(entry["certificate"]["arrival_rates"] for entry in previous["capacities"]
                             if (entry["node"], entry["lanes"], entry["load"]) == (model.node, lanes, load))
                for seed in ARRIVAL_SEEDS:
                    tape_id = f"{model.node}:R{lanes}:load{load:g}:seed{seed}"
                    tapes.append({"tape_id": tape_id, "node": model.node, "lanes": lanes,
                                  "load": load, "seed": seed,
                                  "arrivals": poisson_arrivals(rates, previous["arrival_horizon"], seed)})
                    sequence = [list(development.LABELS), list(reversed(development.LABELS))] * 3
                    random.Random(f"{ORDER_SEED}:{tape_id}").shuffle(sequence)
                    orders[tape_id] = sequence

    protocol = {key: value for key, value in previous.items() if key != "registered_at_utc"}
    protocol.update(
        experiment="small_dimension_exact_native_common_computation_window_independent_arrival_confirmation_v1",
        arrival_seeds=ARRIVAL_SEEDS, tapes=tapes, timing_orders=orders, order_seed=ORDER_SEED,
        expected_tape_count=len(tapes), expected_reference_trajectory_count=len(tapes),
        expected_measured_trajectory_count=len(tapes) * 12, expected_trajectory_count=len(tapes) * 13,
        previous_protocol_path=str(development._protocol_path(PREVIOUS_OUTPUT).relative_to(ROOT)),
        previous_result_path=str(PREVIOUS_OUTPUT.relative_to(ROOT)),
        prior_decisions={**previous["prior_decisions"], "small_dimension_exact_native_common_cadence_development": result["decision"]},
        execution_sources={**previous["execution_sources"], SOURCE: (ROOT / SOURCE).read_text()},
        design="Independent arrival confirmation with eight new seeds13051-13058, four unchanged cached catalogues, virtualR8 and loads0.80/0.95:64 literal tapes and832 trajectories. Retain H600/T1200, computation window1s, controller144/256/8192, reference4096/oracle5M and all nine original wall gates. Each tape has one normative reference and six paired measured rounds, AB/BA each three times, registered using new order seed2026092917. No method/window/budget/gate changes, additional warmups, outcome-based tape selection or timing repeats.",
        arrival_independence="Before authoring this confirmation, the eight seeds were absent from the eleven existing arrival-seed protocol rosters and from algorithm/test/Markdown sources. Generate new literal Poisson tapes using unchanged random.Random('aligned-frame:{seed}:{coordinate}') and the development capacity arrival_rates. Reuse each seed across catalogue/load pairs; those pairs and six timing repeats are not independent arrival observations. Existing static queue-state confirmation is separate evidence.",
        limitations="Independent new-arrival confirmation only on the same four deterministic cached point-model catalogues, virtualR8 and one local CPU. Preserve the development1s experimental window; it is not a demonstrated deployment deadline. Loads still reference original service capacity, not window-adjusted capacity. Normative full-reference cost is recorded but its window is offline. Retain every historical HOLD, including free-clock end-to-end utility HOLD and development P99 regression. No independent physical catalogue, physical cluster, flow-time improvement, energy or statistical-significance claim.",
    )
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered confirmation sources, arrivals, orders or gates changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    runtime = runtime_setup()
    models = runtime.load_catalogues()
    protocol_path = development._protocol_path(args.output)
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, protocol)
        print(protocol_path, flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists():
        raise FileExistsError(args.output)
    for node, lanes in protocol["execution_order"]:
        if development._cell_path(args.output, node, lanes).exists():
            raise FileExistsError(development._cell_path(args.output, node, lanes))
    rows = []
    for node, lanes in protocol["execution_order"]:
        model = next(model for model in models if model.node == node)
        row = development.run_cell(model, lanes, protocol)
        with gzip.open(development._cell_path(args.output, node, lanes), "xt") as target:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                      target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.small_dimension_exact_native_common_cadence_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "experiment", "decision", "integration_gate_passed", "no_overrun_gate_passed",
        "trajectory_identity_passed", "timing_gate_passed")}), flush=True)


if __name__ == "__main__":
    main()
