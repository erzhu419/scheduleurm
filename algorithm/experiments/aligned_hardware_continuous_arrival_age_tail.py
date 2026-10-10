"""Frozen continuous-arrival age-tail performance, with no policy stop-time input."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.aligned_hardware_age_tail_constrained_lane_rollout import rollout_checks as age_checks
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import common_metrics, load_models, replay_matches
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.lane_refill_performance_replay import simulate_lane_refill_trajectory
from algorithm.lane_refill_simple_controls import select_single_lane_control
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_observed_rate_age_tail_confirmation_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_continuous_arrival_age_tail"
LABELS = ("maxweight", "arrival_rollout")
SEEDS = {"development": (41, 42, 43), "confirmation": tuple(range(100001, 100013))}
SOURCES = ("algorithm/continuous_arrival_age_tail_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_continuous_arrival_age_tail.py")


def base(phase):
    return BASE.with_name(BASE.name + "_" + phase + "_20260930")


def path(prefix, suffix):
    return prefix.with_name(prefix.name + suffix)


def build_protocol(phase):
    old = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    for name, source in old["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    previous_result = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if previous_result["decision"] != "OBSERVED_RATE_AGE_TAIL_CONFIRMATION_PASS":
        raise ValueError("previous observed-rate confirmation must pass")
    if phase == "confirmation":
        development = json.loads(base("development").with_suffix(".json").read_text())
        frozen = json.loads(path(base("development"), "_protocol.json").read_text())
        if (development["decision"] != "CONTINUOUS_ARRIVAL_AGE_TAIL_DEVELOPMENT_PASS"
                or development["registered_at_utc"] != frozen["registered_at_utc"]):
            raise ValueError("matching continuous-arrival development PASS required")
        for name, source in frozen["execution_sources"].items():
            if (ROOT / name).read_text() != source:
                raise ValueError("development source changed: " + name)
    tapes = []
    for model in old["models"]:
        for load in old["loads"]:
            rates = next(row["arrival_rates"] for row in old["capacities"]
                         if row["node"] == model["node"] and row["load"] == load)
            for seed in SEEDS[phase]:
                tapes.append({"tape_id": f"{model['node']}:R{model['lanes']}:load{load}:seed{seed}",
                              "node": model["node"], "lanes": model["lanes"], "load": load,
                              "seed": seed, "arrivals": poisson_arrivals(rates, 1200., seed)})
    return json.loads(json.dumps({
        "experiment": "continuous_arrival_age_tail_" + phase + "_v1", "phase": phase,
        **{key: old[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
                                    "planning_event_cap", "planning_seed_rule", "clock", "budget")},
        "arrival_seeds": SEEDS[phase], "tapes": tapes,
        "environment_arrival_end": 1200., "observation_end": 1200.,
        "birth_cohort": [300., 900.], "arrival_horizon": 900.,
        "cutoff_semantics": "arrival_horizon is a summary schema alias for cohort checkpoint900; environment arrivals continue until evaluation1200. Neither timestamp enters either selector.",
        "method": "Frozen FIFO age S1/S2 rule, eight samples, cap128 and whole-decision overflow fallback; cumulative observed births/now, zero exposure zero rate; full tau_max planning window at every eligible decision, with no public end-time input or parameter tuning.",
        "information": "Policy constructor receives only catalogue, independent planning seed and budgets. Context contains now, lane availability, Q, FIFO ages and causally released counts. Environment rates and literal future arrivals stay in the harness.",
        "primary": "Equal-tape mean restricted flow of all births in closed interval[300,900], censored at1200: >=1% reduction vs MW, no load mean regression. Original5% group P90/unfinished tolerance at900 and1200, zero baseline tail requires zero; pooled cohort unfinished cannot increase. Queue area[300,900] uses all jobs and is descriptive, not a new gate.",
        "statistics": "Three reused development arrival seeds; confirmation uses twelve fresh seeds100001..100012, eight paired catalogue/load cells per seed, conditional on fixed planning streams; no96-tape independence claim.",
        "limitations": "Fixed-rate Poisson, deterministic cached independent lanes; no drift, actual P90 bound or physical execution claim. Computation excluded from service clock. Per-class and per-seed regressions retained.",
        "execution_sources": {**old["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}},
    }))


def cohort_metrics(policies, cutoff, birth_cohort):
    lower, upper = birth_cohort
    filtered = {label: {**result, "jobs": [job for job in result["jobs"]
                if lower <= job["arrival"] <= upper]} for label, result in policies.items()}
    return common_metrics(filtered, cutoff)


def queue_area(result, lower=300., upper=900.):
    return sum(max(0., min(job["start"] if job["start"] is not None else upper, upper)
                   - max(job["arrival"], lower)) for job in result["jobs"])


def check_rollout(result, model, protocol, planner):
    if not age_checks(result, model, protocol):
        return False
    for frame in result["frames"]:
        info, counts, now = frame["rollout"], tuple(frame["observed_arrival_counts"]), frame["start"]
        if (tuple(info["observed_arrival_counts"]) != counts or info["rate_exposure_s"] != now
                or tuple(info["estimated_arrival_rates"]) != tuple(n / now if now else 0. for n in counts)
                or info["rate_estimation_coordinate_reads"] != len(counts)
                or info["rate_estimation_divisions"] != (len(counts) if now else 0)):
            return False
        if info["eligible"] and info["window_s"] != planner.maximum_duration:
            return False
    return True


def run_tape(model, tape, protocol):
    planners = {label: ContinuousArrivalAgeTailLaneRollout(model.options,
                f"arrival-plan-v1:{model.node}:{tape['load']}:online",
                protocol["planning_samples"], protocol["planning_event_cap"]) for label in LABELS}
    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
                protocol["environment_arrival_end"], protocol["observation_end"],
                planners[label].maxweight if label == "maxweight" else planners[label], policy=label)
                for label in LABELS}
    shadow, originals = simulate_lane_refill_trajectory(model, tape["arrivals"],
                protocol["environment_arrival_end"], protocol["observation_end"], 0,
                select_single_lane_control, policy="maxweight", dispatch_rule="single")
    row = {**{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": replay_matches(policies["maxweight"], json.loads(json.dumps(shadow))),
        "rollout_checks_passed": check_rollout(policies["arrival_rollout"], model, protocol, planners["arrival_rollout"]),
        "snapshots": {key: cohort_metrics(policies, protocol[key], protocol["birth_cohort"])
                      for key in ("arrival_horizon", "observation_end")},
        "work": {label: work_totals(result, planners[label], protocol["arrival_horizon"])
                 for label, result in policies.items()},
        "middle_queue_area_s": {label: queue_area(result) for label, result in policies.items()},
        "reference_option_probes": sum(item.option_probes for item in originals)}
    return row, policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(SEEDS), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prefix = base(args.phase)
    expected = build_protocol(args.phase)
    if args.register:
        with path(prefix, "_protocol.json").open("x") as out:
            json.dump({**expected, "registered_at_utc": datetime.now(timezone.utc).isoformat()}, out,
                      separators=(",", ":"), allow_nan=False)
        print(path(prefix, "_protocol.json"), flush=True)
        return
    protocol = json.loads(path(prefix, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources, rules or literal tapes changed")
    if prefix.with_suffix(".json").exists() or path(prefix, "_trajectories.json.gz").exists():
        raise FileExistsError("preserve existing results")
    models = {model.node: model for model in load_models(protocol["models"])}
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_tape(models[tape["node"]], tape, protocol)
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(tape["tape_id"], "complete", row["baseline_reproduction_passed"], row["rollout_checks_passed"], flush=True)
    result = summarize(rows, protocol)
    status = "REPLAY_FAIL" if not result["execution_checks_passed"] else "PASS" if result["performance_gain_established"] else "HOLD"
    result["decision"] = "CONTINUOUS_ARRIVAL_AGE_TAIL_" + args.phase.upper() + "_" + status
    result["statistical_scope"] = protocol["statistics"]
    result["birth_cohort"] = protocol["birth_cohort"]
    result["cutoff_semantics"] = protocol["cutoff_semantics"]
    result["reference_trajectories"] = len(rows)
    result["total_executed_trajectories"] = 3 * len(rows)
    result["reference_option_probes"] = sum(row["reference_option_probes"] for row in rows)
    result["middle_queue_area_s"] = {label: sum(row["middle_queue_area_s"][label] for row in rows) for label in LABELS}
    with gzip.open(path(prefix, "_trajectories.json.gz"), "xt") as out:
        json.dump({"runs": runs}, out, separators=(",", ":"), allow_nan=False)
    with prefix.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "pooled", "failed_secondary_groups")}), flush=True)


if __name__ == "__main__":
    main()
