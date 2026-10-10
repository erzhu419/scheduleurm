"""Fresh-arrival confirmation of the frozen background-competition root rule."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.experiments.aligned_hardware_background_competition_continuous import (
    BASE as DEVELOPMENT, LABELS, path, run_tape,
)
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize
from algorithm.experiments.lane_refill_performance_replay import simulate_lane_refill_trajectory
from algorithm.lane_refill_simple_controls import select_single_lane_control


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_background_competition_continuous_confirmation_20260930"
SOURCE = "algorithm/experiments/aligned_hardware_background_competition_confirmation.py"
ARRIVAL_SEEDS = tuple(range(100001, 100013))
STATISTICAL_SCOPE = (
    "Twelve fresh actual arrival seeds, each paired across eight catalogue/load cells, "
    "conditional on the unchanged fixed planning streams. The 96 tapes are not "
    "96 independent replicates. Seed aggregates are descriptive; all regressions remain recorded."
)


def build_protocol():
    development = json.loads(path(DEVELOPMENT, "_protocol.json").read_text())
    result = json.loads(DEVELOPMENT.with_suffix(".json").read_text())
    if (result["decision"] != "BACKGROUND_COMPETITION_CONTINUOUS_DEVELOPMENT_PASS"
            or not result["execution_checks_passed"] or not result["performance_gain_established"]
            or result["registered_at_utc"] != development["registered_at_utc"]):
        raise ValueError("matching frozen development execution and performance PASS required")
    for name, source in development["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen development source changed: " + name)
    tapes = []
    for model in development["models"]:
        for load in development["loads"]:
            rates = next(row["arrival_rates"] for row in development["capacities"]
                         if row["node"] == model["node"] and row["load"] == load)
            for seed in ARRIVAL_SEEDS:
                tapes.append({
                    "tape_id": f"{model['node']}:R{model['lanes']}:load{load}:seed{seed}",
                    "node": model["node"], "lanes": model["lanes"], "load": load,
                    "seed": seed, "arrivals": poisson_arrivals(
                        rates, development["environment_arrival_end"], seed)})
    body = {
        "experiment": "background_competition_continuous_confirmation_v1",
        "phase": "confirmation", "evidence_type": "independent_confirmation",
        "development_protocol": str(path(DEVELOPMENT, "_protocol.json").relative_to(ROOT)),
        "development_result": str(DEVELOPMENT.with_suffix(".json").relative_to(ROOT)),
        "development_registered_at_utc": development["registered_at_utc"],
        "development_decision": result["decision"],
        **{key: development[key] for key in (
            "models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "labels", "method", "budget",
            "environment_arrival_end", "observation_end", "birth_cohort", "arrival_horizon",
            "cutoff_semantics", "primary", "information", "limitations")},
        "arrival_seeds": ARRIVAL_SEEDS, "tapes": tapes,
        "statistics": STATISTICAL_SCOPE,
        "validation": "On each fresh literal tape, execute the original exact single-lane MW engine first as a shadow reference, then reuse the unchanged development run_tape to compare the observed-count engine's MW actions, queues, lanes, service, duration and job timestamps. Retain all frozen rollout and actual-work checks. Only the two primary policies enter cohort metrics and performance gates; charge the 96 shadow trajectories separately.",
        "trajectory_accounting": {"primary_policies": 2, "primary_trajectories": 192,
                                  "reference_trajectories": 96, "total_executed_trajectories": 288},
        "execution_sources": {**development["execution_sources"], SOURCE: (ROOT / SOURCE).read_text()},
    }
    return json.loads(json.dumps(body))


def run_confirmation_tape(model, tape, protocol):
    shadow, originals = simulate_lane_refill_trajectory(
        model, tape["arrivals"], protocol["environment_arrival_end"],
        protocol["observation_end"], 0, select_single_lane_control,
        policy="maxweight", dispatch_rule="single")
    row, policies = run_tape(model, tape, protocol, json.loads(json.dumps(shadow)))
    row["reference_work"] = {
        "selector_calls": len(originals),
        "option_probes": sum(item.option_probes for item in originals),
        "candidate_evaluations": sum(item.candidate_evaluations for item in originals),
    }
    return row, policies


def confirmation_summary(rows, protocol):
    result = summarize(rows, protocol)
    status = ("REPLAY_FAIL" if not result["execution_checks_passed"] else
              "PASS" if result["performance_gain_established"] else "HOLD")
    result["decision"] = "BACKGROUND_COMPETITION_CONTINUOUS_CONFIRMATION_" + status
    result["phase"] = "confirmation"
    result["evidence_type"] = "independent_confirmation"
    result["statistical_scope"] = protocol["statistics"]
    result["development_protocol"] = protocol["development_protocol"]
    result["development_result"] = protocol["development_result"]
    result["development_registered_at_utc"] = protocol["development_registered_at_utc"]
    result["development_decision"] = protocol["development_decision"]
    result["birth_cohort"] = protocol["birth_cohort"]
    result["cutoff_semantics"] = protocol["cutoff_semantics"]
    result["reference_trajectories"] = len(rows)
    result["total_executed_trajectories"] = result["trajectories"] + len(rows)
    work = Counter()
    for row in rows:
        work.update(row["reference_work"])
    result["reference_totals_work"] = dict(work)
    result["middle_queue_area_s"] = {
        label: sum(row["middle_queue_area_s"][label] for row in rows) for label in LABELS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    expected = build_protocol()
    if args.register:
        with path(BASE, "_protocol.json").open("x") as out:
            json.dump({**expected, "registered_at_utc": datetime.now(timezone.utc).isoformat()},
                      out, separators=(",", ":"), allow_nan=False)
        print(path(BASE, "_protocol.json"), flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered confirmation sources, parameters or literal tapes changed")
    if BASE.with_suffix(".json").exists() or path(BASE, "_trajectories.json.gz").exists():
        raise FileExistsError("preserve existing frozen confirmation results")
    models = {model.node: model for model in load_models(protocol["models"])}
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_confirmation_tape(models[tape["node"]], tape, protocol)
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(tape["tape_id"], "pair and shadow complete",
              row["baseline_reproduction_passed"], row["rollout_checks_passed"], flush=True)
    result = confirmation_summary(rows, protocol)
    with gzip.open(path(BASE, "_trajectories.json.gz"), "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"],
                   "evidence_type": protocol["evidence_type"], "runs": runs},
                  out, separators=(",", ":"), allow_nan=False)
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in (
        "decision", "execution_checks_passed", "performance_gain_established", "pooled")}), flush=True)


if __name__ == "__main__":
    main()
