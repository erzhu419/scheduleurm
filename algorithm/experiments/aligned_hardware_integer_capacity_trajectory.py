"""Frozen, paired finite-horizon trajectory screen of the timed count selectors.

Register literal arrival tapes before running any policy.  Each policy performs
its own cold selector calls; only the separate exact-reference audit is cached.
The virtual clock uses measured-model service offsets and excludes solver time.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import scipy

from algorithm.experiments import aligned_hardware_integer_capacity_development as kernels
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate
from algorithm.experiments.integer_capacity_trajectory_replay import ReferenceAudit, simulate_trajectory
from algorithm.experiments.integer_capacity_trajectory_summary import summarize


ROOT = kernels.ROOT
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_trajectory_20260927.json"
TIMING_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_timing_20260927.json"
TIMING_PROTOCOL = TIMING_OUTPUT.with_name(TIMING_OUTPUT.stem + "_protocol.json")
LANES = (4, 8, 16)
LOADS = (0.80, 0.95)
ARRIVAL_SEEDS = (2026092711, 2026092712, 2026092713, 2026092714)
ARRIVAL_HORIZON = 600.0
OBSERVATION_END = 1200.0
REFERENCE_BUDGET = 4096
LABELS = ("full_reference", "old_shared", "fractional_shared", "integer_capacity")
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_frame_experiment.py",
    "algorithm/experiments/aligned_hardware_physical_catalogue_models.py",
    "algorithm/experiments/integer_capacity_trajectory_replay.py",
    "algorithm/experiments/integer_capacity_trajectory_summary.py",
    "algorithm/experiments/aligned_hardware_integer_capacity_trajectory.py",
)


def require_timing_advance():
    result = json.loads(TIMING_OUTPUT.read_text(encoding="utf-8"))
    flags = ("execution_checks_passed", "certificate_checks_passed", "references_complete",
             "action_gate_passed", "local_timing_gate_passed")
    if result["decision"] != "INTEGER_CAPACITY_LOCAL_TIMING_ADVANCE" or not all(result[key] for key in flags):
        raise ValueError("successful whole-selector timing required before trajectory registration")


def build_protocol(models):
    require_timing_advance()
    previous = json.loads(TIMING_PROTOCOL.read_text(encoding="utf-8"))
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"successful timing source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("successful timing catalogues changed")
    capacities, tapes = [], []
    for model in models:
        for lanes in LANES:
            virtual_model = replace(model, lanes=lanes)
            for load in LOADS:
                certificate = capacity_certificate(virtual_model, load)
                capacities.append({"node": model.node, "lanes": lanes, "load": load,
                                   "certificate": certificate})
                for seed in ARRIVAL_SEEDS:
                    tapes.append({
                        "tape_id": f"{model.node}:R{lanes}:load{load:g}:seed{seed}",
                        "node": model.node, "lanes": lanes, "load": load, "seed": seed,
                        "arrivals": poisson_arrivals(certificate["arrival_rates"], ARRIVAL_HORIZON, seed),
                    })
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "integer_capacity_paired_finite_trajectory_v1",
        "models": previous["models"], "lanes": LANES, "loads": LOADS,
        "arrival_seeds": ARRIVAL_SEEDS, "arrival_horizon": ARRIVAL_HORIZON,
        "observation_end": OBSERVATION_END, "labels": LABELS,
        "budgets": {str(lanes): 36 * lanes for lanes in LANES},
        "reference_budget": REFERENCE_BUDGET, "capacities": capacities, "tapes": tapes,
        "prior_timing": str(TIMING_OUTPUT.relative_to(ROOT)),
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "arrivals": "Independent class Poisson streams via random.Random('aligned-frame:{seed}:{coordinate}'); literal tapes registered at full-catalogue capacity rates separately for every node/R/load; identical tape for all four policies; seeds shared across configurations are paired, not independent catalogue observations.",
        "calls": "Each trajectory uses actual cold selector calls at every frame, including one-second idle frames: full reference K4096 and three candidate/control policies K36R. No warm starts, candidate-result caching, budget escalation, or rescue.",
        "reference_audit": "Certify every visited candidate queue using the original full K4096 selector; reuse previously audited queues within one tape, including the full-reference policy's actual calls. Audit calls and work reported separately from each policy's paid work. Incomplete references remain incomplete.",
        "clock": "Only released FIFO jobs enter selection. Complete-frame starts are synchronous; original per-job completion offsets apply. Solver wall time is excluded from virtual time. Stop starting frames at T, retain scheduled completions beyond T as in-flight and unfinished.",
        "snapshots": "At H and T, report completed/queued/in-flight/unfinished counts and restricted mean flow time over every released job. Preserve all per-job start/completion outcomes. Completed-only mean/p90 are descriptive; comparison uses the identical completed-ID intersection across all four policies separately at each cutoff.",
        "integration_gate": "Complete references, valid selection certificates and physical/accounting invariants for all policies, integer-capacity canonical exact on every frame and matching the audited optimum, and integer-capacity frame/action/job trajectory identical to full K4096 on every tape.",
        "utility_gate": "Against BOTH old-shared and fractional-shared controls, integer-capacity per-tape restricted mean and common-completed-cohort mean/p90 averaged across seeds must not increase in ANY node/R/load group at either H or T; unfinished counts summed across seeds must not increase. Float comparisons use relative tolerance 1e-9 only; integer counts exact. Empty common cohorts cannot establish utility nonregression.",
        "flow_gain_gate": "Integration and all utility nonregression gates pass, with pooled per-tape common-completed-cohort mean at T at least 1% lower than BOTH controls. Exact trajectory preservation can pass without flow gain.",
        "float_relative_tolerance": 1e-9, "minimum_pooled_flow_improvement": 0.01,
        "search_counters": previous["search_counters"],
        "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "A registered 600-second arrival/1200-second observation screen, not a reproduction of the earlier 20000-second protocol. Four cached deterministic complete-frame point catalogues and virtual identical-lane scaling; no new physical execution or joint95 guarantee. Four arrival seeds and catalogue/load comparisons are paired. Solver overhead is excluded from the model clock; no physical wall-clock flow-time claim.",
    }))


def verify_protocol(protocol, models):
    expected = build_protocol(models)
    actual = {key: value for key, value in protocol.items() if key != "registered_at_utc"}
    if actual != expected:
        raise ValueError("registered sources, models, capacity rates, tapes, or trajectory design changed")


def trajectory_cell(model, lanes, protocol, progress=True):
    started = perf_counter()
    virtual_model = replace(model, lanes=lanes)
    selectors = {
        "full_reference": kernels.select_canonical_count_space_bnb_frame,
        "old_shared": kernels.select_shared_threshold_canonical_requirement_dominance_frame,
        "fractional_shared": kernels.select_fractional_shared_threshold_canonical_requirement_dominance_frame,
        "integer_capacity": kernels.select_integer_capacity_shared_threshold_frame,
    }
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "option_count": len(model.options), "runs": [],
           "runtime": {"python": platform.python_version(), "scipy": scipy.__version__}}
    for registered in protocol["tapes"]:
        if (registered["node"], registered["lanes"]) != (model.node, lanes):
            continue
        audit = ReferenceAudit(virtual_model, budget=protocol["reference_budget"])
        run = {key: registered[key] for key in ("tape_id", "load", "seed")}
        run.update(arrival_count=len(registered["arrivals"]), policies={})
        tape = tuple(tuple(arrival) for arrival in registered["arrivals"])
        for label in protocol["labels"]:
            budget = (protocol["reference_budget"] if label == "full_reference"
                      else protocol["budgets"][str(lanes)])
            run["policies"][label] = simulate_trajectory(
                virtual_model, tape, protocol["arrival_horizon"], protocol["observation_end"],
                budget, selectors[label], reference_cache=audit, policy=label,
            )
        run["reference_audit"] = audit.summary()
        row["runs"].append(run)
        if progress:
            print(f"{model.node} R{lanes}: load={run['load']} seed={run['seed']} four trajectories complete", flush=True)
    row["operational_elapsed_s"] = perf_counter() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--cell", help="registered NODE:R cell")
    mode.add_argument("--merge-cells", action="store_true")
    args = parser.parse_args()
    models = kernels.load_catalogues()
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        kernels._write_new(protocol_path, protocol)
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_protocol(protocol, models)
    if args.cell:
        node, lanes_text = args.cell.rsplit(":", 1)
        lanes = int(lanes_text)
        if [node, lanes] not in protocol["execution_order"]:
            raise ValueError("unregistered node/lane cell")
        model = next(model for model in models if model.node == node)
        output = kernels._cell_path(args.output, node, lanes)
        if output.exists():
            raise FileExistsError(output)
        kernels._write_new(output, {"protocol": protocol, "cell": trajectory_cell(model, lanes, protocol)})
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for node, lanes in protocol["execution_order"]:
        saved = json.loads(kernels._cell_path(args.output, node, lanes).read_text(encoding="utf-8"))
        if saved["protocol"] != protocol:
            raise ValueError("cell protocol differs from registered protocol")
        rows.append(saved["cell"])
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    kernels._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "integration_gate_passed", "flow_gain_established")}))


if __name__ == "__main__":
    main()
