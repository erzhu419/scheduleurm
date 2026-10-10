"""Frozen two-frame arrival-window comparison on the original development tapes."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.two_frame_window_continuous_lane_rollout import TwoFrameWindowContinuousLaneRollout
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_age_tail_constrained_lane_rollout import rollout_checks as age_checks
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_continuous_arrival_age_tail_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_two_frame_window_continuous_development_20260930"
LABELS = ("maxweight", "arrival_rollout")
SOURCES = ("algorithm/two_frame_window_continuous_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_two_frame_window_continuous.py")


def path(prefix, suffix):
    return prefix.with_name(prefix.name + suffix)


def build_protocol():
    old = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    if json.loads(PREVIOUS.with_suffix(".json").read_text())["decision"] != "CONTINUOUS_ARRIVAL_AGE_TAIL_DEVELOPMENT_HOLD":
        raise ValueError("original continuous-arrival HOLD must remain recorded")
    for name, source in old["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    return {
        "experiment": "two_frame_window_continuous_development_v1", "phase": "development",
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        **{key: old[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "tapes", "arrival_seeds",
            "environment_arrival_end", "observation_end", "birth_cohort", "arrival_horizon",
            "cutoff_semantics", "primary", "information", "limitations")},
        "labels": list(LABELS),
        "method": "Change only the original continuous-arrival planning window from tau_max to 2*tau_max. Physical durations and completion offsets remain catalogue values. Keep cumulative observed births/now, eight shared independent populations, all feasible roots, exact FIFO S1/S2 and MW continuation, same queue envelope, tie rules and whole-decision MW fallback if any path exceeds128 events. No nested redecision, new class constraint, slack, true arrival-rate or observation-end input, or window parameter sweep.",
        "budget": "For m options, R lanes and eligible N<=R*B, at most8*m*(N+128-1+R*(ceil(2*tau_max)+1)) continuation calls, each scanning m original options, plus2*m root probes. Flow evaluations<=8*m*(N+128). At most eight sampled paths, each stops at129 events on overflow; sampling draws/events/rate reads, preprocessing and actual engine context extraction all charged. The longer polling window is charged explicitly; no computational delay is added to the service clock.",
        "statistics": "Three reused development seeds41..43, eight paired catalogue/load cells per seed. No new independent confirmation or 24-tape independence claim.",
        "execution_sources": {**old["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def check_rollout(result, model, protocol, planner):
    if not age_checks(result, model, protocol):
        return False
    for frame in result["frames"]:
        info, now, counts = frame["rollout"], frame["start"], tuple(frame["observed_arrival_counts"])
        if (tuple(info["observed_arrival_counts"]) != counts or info["rate_exposure_s"] != now
                or tuple(info["estimated_arrival_rates"]) != tuple(n / now if now else 0. for n in counts)
                or info["rate_estimation_coordinate_reads"] != len(counts)
                or info["rate_estimation_divisions"] != (len(counts) if now else 0)):
            return False
        if info["eligible"] and info["window_s"] != 2 * planner.maximum_duration:
            return False
    return True


def run_tape(model, tape, protocol, frozen_baseline):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {"maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
                    protocol["planning_samples"], protocol["planning_event_cap"]),
                "arrival_rollout": TwoFrameWindowContinuousLaneRollout(model.options, seed,
                    protocol["planning_samples"], protocol["planning_event_cap"])}
    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"],
        planners[label].maxweight if label == "maxweight" else planners[label], policy=label)
        for label in LABELS}
    row = {**{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": replay_matches(policies["maxweight"], frozen_baseline),
        "rollout_checks_passed": check_rollout(policies["arrival_rollout"], model, protocol, planners["arrival_rollout"]),
        "snapshots": {key: cohort_metrics(policies, protocol[key], protocol["birth_cohort"])
                      for key in ("arrival_horizon", "observation_end")},
        "work": {label: work_totals(result, planners[label], protocol["arrival_horizon"])
                 for label, result in policies.items()},
        "middle_queue_area_s": {label: queue_area(result) for label, result in policies.items()}}
    return row, policies


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
        raise ValueError("registered sources, window, budgets or literal tapes changed")
    if BASE.with_suffix(".json").exists() or path(BASE, "_trajectories.json.gz").exists():
        raise FileExistsError("preserve existing results")
    with gzip.open(ROOT / protocol["previous_trajectories"], "rt") as source:
        old = {run["tape_id"]: run["policies"]["maxweight"] for run in json.load(source)["runs"]}
    models = {model.node: model for model in load_models(protocol["models"])}
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_tape(models[tape["node"]], tape, protocol, old[tape["tape_id"]])
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(tape["tape_id"], "complete", row["baseline_reproduction_passed"], row["rollout_checks_passed"], flush=True)
    result = summarize(rows, protocol)
    status = "REPLAY_FAIL" if not result["execution_checks_passed"] else "PASS" if result["performance_gain_established"] else "HOLD"
    result["decision"] = "TWO_FRAME_WINDOW_CONTINUOUS_DEVELOPMENT_" + status
    result["birth_cohort"], result["cutoff_semantics"] = protocol["birth_cohort"], protocol["cutoff_semantics"]
    result["total_executed_trajectories"] = 2 * len(rows)
    result["middle_queue_area_s"] = {label: sum(row["middle_queue_area_s"][label] for row in rows) for label in LABELS}
    previous = json.loads(PREVIOUS.with_suffix(".json").read_text())
    result["previous_candidate_comparison"] = {"scope": "Descriptive reused-tape comparison, not independent confirmation.",
        "decision": previous["decision"], "pooled": previous["pooled"]}
    with gzip.open(path(BASE, "_trajectories.json.gz"), "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "runs": runs}, out, separators=(",", ":"), allow_nan=False)
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "pooled", "failed_secondary_groups")}), flush=True)


if __name__ == "__main__":
    main()
