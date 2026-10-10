"""Frozen held-out arrival confirmation of the FIFO-age constrained rollout."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.experiments.aligned_hardware_age_tail_constrained_lane_rollout import run_tape
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize
from algorithm.experiments.lane_refill_performance_replay import simulate_lane_refill_trajectory
from algorithm.lane_refill_simple_controls import select_single_lane_control


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = ROOT / "md/experiment_artifacts/aligned_hardware_age_tail_constrained_lane_rollout_development_20260930"
DEFAULT = ROOT / "md/experiment_artifacts/aligned_hardware_age_tail_constrained_confirmation_20260930"
SOURCE = "algorithm/experiments/aligned_hardware_age_tail_constrained_confirmation.py"
ARRIVAL_SEEDS = tuple(range(1001, 1013))
DEVELOPMENT_PASS = "AGE_TAIL_CONSTRAINED_LANE_ROLLOUT_PERFORMANCE_PASS"
STATISTICAL_SCOPE = (
    "Independent arrival confirmation conditional on the registered fixed planning stream. "
    "Twelve held-out actual arrival seeds each provide eight paired catalogue/load cells; "
    "the 96 tapes are not 96 independent replicates. Seed aggregates are descriptive."
)


def build_protocol(development, development_result):
    if (development_result["decision"] != DEVELOPMENT_PASS
            or not development_result["execution_checks_passed"]
            or not development_result["performance_gain_established"]):
        raise ValueError("a valid developmental performance PASS is required before confirmation")
    if development_result["registered_at_utc"] != development["registered_at_utc"]:
        raise ValueError("developmental PASS must belong to the frozen development protocol")
    for path, source in development["execution_sources"].items():
        if (ROOT / path).read_text() != source:
            raise ValueError(f"frozen development execution dependency changed: {path}")
    tapes = []
    for model in development["models"]:
        for load in development["loads"]:
            rates = next(row["arrival_rates"] for row in development["capacities"]
                         if row["node"] == model["node"] and row["load"] == load)
            for seed in ARRIVAL_SEEDS:
                tapes.append({"tape_id": f"{model['node']}:R{model['lanes']}:load{load}:seed{seed}",
                              "node": model["node"], "lanes": model["lanes"], "load": load,
                              "seed": seed, "arrivals": poisson_arrivals(rates, development["arrival_horizon"], seed)})
    body = {
        "experiment": "age_tail_constrained_lane_rollout_independent_confirmation_v1",
        "evidence_type": "independent_confirmation",
        "candidate_method": development["candidate_method"],
        "development_protocol": str(DEVELOPMENT.with_name(DEVELOPMENT.name + "_protocol.json").relative_to(ROOT)),
        "development_result": str(DEVELOPMENT.with_suffix(".json").relative_to(ROOT)),
        "development_registered_at_utc": development["registered_at_utc"],
        "development_decision": development_result["decision"],
        **{key: development[key] for key in ("models", "capacities", "loads", "arrival_horizon",
            "observation_end", "execution_order", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "labels", "method", "information", "budget")},
        "arrival_seeds": ARRIVAL_SEEDS, "tapes": tapes,
        "primary": "Mean of the96 paired-tape T all-released-job restricted mean flows improves at least1% versus ordinary MaxWeight; each load mean does not worsen. All16 node/load/H,T groups retain common-ID mean/P90 and unfinished counts; P90 and unfinished allow at most5% group regression, zero unfinished control requires zero. Pooled H/T unfinished counts cannot increase. Preserve every seed/group regression.",
        "statistics": STATISTICAL_SCOPE + " No parameter, population, cutoff or threshold changes after outcome access.",
        "validation": "On every one of96 literal tapes, first execute the unchanged original exact single-lane MaxWeight engine as a reference, then use the frozen development run_tape to validate the prepared age-engine MaxWeight action/start/Q/lane/service/duration and every job timestamp against that reference. Retain the frozen per-frame age/risk/work checks. Only the two primary policies enter common cohorts and performance gates; the96 shadow references are separately charged.",
        "trajectory_accounting": {"primary_policies": 2, "primary_trajectories": 192,
                                  "reference_trajectories": 96, "total_executed_trajectories": 288},
        "limitations": "Deterministic catalogue-reported2/4 cached independent-lane replay with known arrival rates and H;12 held-out arrival seeds, conditional on the fixed registered planning streams. A finite-window sample second-moment constraint is a tail proxy, not an actual P90 bound, physical execution result or asynchronous stability proof. Existing development and confirmation regressions are retained; wall time is not the scheduling endpoint.",
        "execution_sources": {**development["execution_sources"], SOURCE: (ROOT / SOURCE).read_text()},
    }
    return json.loads(json.dumps(body))


def validate_registered_protocol(protocol, development, development_result):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(development, development_result):
        raise ValueError("registered confirmation method, rates, models, seeds, tapes, gates or sources changed")


def run_confirmation_tape(model, tape, protocol):
    shadow, originals = simulate_lane_refill_trajectory(
        model, tape["arrivals"], protocol["arrival_horizon"], protocol["observation_end"],
        0, select_single_lane_control, policy="maxweight", dispatch_rule="single")
    # The existing replay identity compares JSON-normalized old frame fields.
    row, policies = run_tape(model, tape, protocol, json.loads(json.dumps(shadow)))
    row["reference_work"] = {
        "selector_calls": len(originals),
        "option_probes": sum(selection.option_probes for selection in originals),
        "candidate_evaluations": sum(selection.candidate_evaluations for selection in originals),
    }
    return row, policies


def confirmation_summary(rows, protocol):
    result = summarize(rows, protocol)
    suffix = ("REPLAY_FAIL" if not result["execution_checks_passed"] else
              "PASS" if result["performance_gain_established"] else "HOLD")
    result["decision"] = "AGE_TAIL_CONSTRAINED_LANE_ROLLOUT_CONFIRMATION_" + suffix
    result["evidence_type"] = "independent_confirmation"
    result["candidate_method"] = protocol["candidate_method"]
    result["statistical_scope"] = STATISTICAL_SCOPE
    result["development_decision"] = protocol["development_decision"]
    result["development_registered_at_utc"] = protocol["development_registered_at_utc"]
    result["reference_trajectories"] = len(rows)
    result["total_executed_trajectories"] = result["trajectories"] + len(rows)
    work = Counter()
    for row in rows:
        work.update(row["reference_work"])
    result["reference_totals_work"] = dict(work)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    development = json.loads(DEVELOPMENT.with_name(DEVELOPMENT.name + "_protocol.json").read_text())
    development_result = json.loads(DEVELOPMENT.with_suffix(".json").read_text())
    protocol_path = args.output.with_name(args.output.name + "_protocol.json")
    if args.register:
        with protocol_path.open("x") as out:
            json.dump({**build_protocol(development, development_result), "registered_at_utc": datetime.now(timezone.utc).isoformat()},
                      out, separators=(",", ":"), allow_nan=False)
        print(protocol_path, flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    validate_registered_protocol(protocol, development, development_result)
    output = args.output.with_suffix(".json")
    trajectories = args.output.with_name(args.output.name + "_trajectories.json.gz")
    if output.exists() or trajectories.exists():
        raise FileExistsError("preserve existing frozen confirmation results")
    models = {model.node: model for model in load_models(protocol["models"])}
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_confirmation_tape(models[tape["node"]], tape, protocol)
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(f"{tape['tape_id']}: confirmation pair and shadow complete; baseline={row['baseline_reproduction_passed']} age/risk/work={row['rollout_checks_passed']}", flush=True)
    with gzip.open(trajectories, "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "evidence_type": protocol["evidence_type"],
                   "candidate_method": protocol["candidate_method"], "runs": runs},
                  out, separators=(",", ":"), allow_nan=False)
    result = confirmation_summary(rows, protocol)
    with output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "execution_checks_passed", "performance_gain_established", "pooled")}), flush=True)


if __name__ == "__main__":
    main()
