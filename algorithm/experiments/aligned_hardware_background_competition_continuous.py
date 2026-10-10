"""Frozen tagged-population rollout with continuing background competition."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from math import ceil
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_continuous_arrival_age_tail_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_background_competition_continuous_development_20260930"
LABELS = ("maxweight", "arrival_rollout")
SOURCES = ("algorithm/background_competition_continuous_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_background_competition_continuous.py")


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
        "experiment": "background_competition_continuous_development_v1", "phase": "development",
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        **{key: old[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "tapes", "arrival_seeds",
            "environment_arrival_end", "observation_end", "birth_cohort", "arrival_horizon",
            "cutoff_semantics", "primary", "information", "limitations")},
        "labels": list(LABELS),
        "method": "Keep the original one-tau_max tagged objective population: visible unadmitted FIFO jobs plus the same eight sampled arrival prefixes. Continue independent per-class Poisson background arrivals after tau_max on a shared chronological lazy suffix per sample, with a separate seed per decision/sample and the same fixed causal count/now rates. Background jobs compete under original MW but never enter tagged S1/S2. Stop a root only after all tagged births are released and admitted, retaining their complete catalogue completion offsets and every lane commitment. Restore original aggregate S2<=MW, minimize tagged S1, MW ties and strict improvement. Combined tagged-future plus background cap128: any root/sample requiring event129 causes whole-decision MW fallback with partial actual work retained. No real future, announced stop or class coefficients enter the policy.",
        "budget": "For m options, R lanes and eligible N<=R*B, at most8*m*(N+128-1+R*(ceil(tau_max)+1)) continuation calls, each scanning m options, plus2*m root probes. Every nonidle call consumes at least one of N+128 actual jobs; idle occurs only before the last tagged birth<=tau_max. All admitted-job/tag-flag reads<=8*m*(N+128), and tagged flow evaluations plus background admissions equal actual job reads. Suffix generation is shared across roots and counted once per sample; boundary/overflow draws, all unique events, rate reads and seed initializations are charged, including rejected partial scoring. Catalogue preparation and engine context extraction remain charged. Computation does not delay service.",
        "statistics": "Three reused development seeds41..43, eight paired catalogue/load cells per seed. No new independent confirmation or 24-tape independence claim.",
        "execution_sources": {**old["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


EXTRA_WORK = ("background_planning_events", "background_planning_random_draws",
              "background_planning_rate_reads", "background_seed_initializations",
              "admitted_job_reads", "tag_flag_reads", "background_admissions",
              "background_rollout_overflow")


def check_rollout(result, model, protocol, planner):
    m, r, cap, samples = len(model.options), model.lanes, protocol["planning_event_cap"], protocol["planning_samples"]
    for frame in result["frames"]:
        info, now, counts = frame["rollout"], frame["start"], tuple(frame["observed_arrival_counts"])
        n, ages = sum(frame["queue"]), frame["waiting_ages"]
        if (tuple(map(len, ages)) != tuple(frame["queue"]) or any(
                any(age < 0 for age in group) or list(group) != sorted(group, reverse=True) for group in ages)
                or info["queue_limit"] != r * planner.batch_width
                or info["eligible"] != (0 < n <= info["queue_limit"])):
            return False
        if (tuple(info["observed_arrival_counts"]) != counts or info["rate_exposure_s"] != now
                or tuple(info["estimated_arrival_rates"]) != tuple(value / now if now else 0. for value in counts)
                or info["rate_estimation_coordinate_reads"] != len(counts)
                or info["rate_estimation_divisions"] != (len(counts) if now else 0)):
            return False
        k = info["candidate_count"]
        if (not 0 <= k <= m or info["rollout_option_probes"] != m * info["rollout_calls"]
                or info["planning_events"] != info["tagged_planning_events"] + info["background_planning_events"]
                or info["admitted_job_reads"] != info["flow_evaluations"] + info["background_admissions"]
                or info["tag_flag_reads"] != info["admitted_job_reads"]
                or info["background_seed_initializations"] > info["sample_count"]):
            return False
        rescheduled = info["background_planning_events"] - int(info["background_rollout_overflow"])
        positive_rates = sum(value > 0 for value in info["estimated_arrival_rates"])
        if (info["background_planning_rate_reads"] != len(counts) * info["background_seed_initializations"] + rescheduled
                or info["background_planning_random_draws"] != positive_rates * info["background_seed_initializations"] + rescheduled):
            return False
        calls_bound = samples * k * (n + cap - 1 + r * (ceil(planner.maximum_duration) + 1))
        if (info["rollout_calls"] > calls_bound or info["admitted_job_reads"] > samples * k * (n + cap)
                or info["flow_evaluations"] > k * (samples * n + info["tagged_planning_events"])):
            return False
        if not info["eligible"]:
            if info["root_option_probes"] != m or any(info[key] for key in EXTRA_WORK) or info["changed"] or k:
                return False
            continue
        if info["window_s"] != planner.maximum_duration or not 1 <= info["sample_count"] <= samples:
            return False
        if (info["planning_events"] > info["sample_count"] * (cap + 1)
                or info["background_planning_random_draws"] > info["planning_random_draws"]
                or info["background_planning_rate_reads"] > info["planning_rate_reads"]):
            return False
        if info["planning_overflow"]:
            if (info["changed"] or info["baseline_predicted_completion_s"] is not None
                    or info["chosen_predicted_completion_s"] is not None
                    or info["risk_feasible_candidates"] or info["risk_filtered_candidates"]):
                return False
            if info["background_rollout_overflow"]:
                if info["root_option_probes"] != 2 * m or not k or info["sample_count"] != samples:
                    return False
            elif info["root_option_probes"] != m or info["rollout_calls"] or k:
                return False
            continue
        if (info["background_rollout_overflow"] or info["root_option_probes"] != 2 * m
                or info["sample_count"] != samples or not 1 <= info["risk_feasible_candidates"] <= k
                or info["flow_evaluations"] != k * (samples * n + info["tagged_planning_events"])
                or info["risk_filtered_candidates"] + info["risk_feasible_candidates"] != k
                or info["chosen_predicted_completion_s"] > info["baseline_predicted_completion_s"]
                or info["chosen_predicted_squared_flow_s2"] > info["baseline_predicted_squared_flow_s2"]):
            return False
        if info["changed"] and not info["chosen_predicted_completion_s"] < info["baseline_predicted_completion_s"]:
            return False
    return True


def background_work(result, planner, horizon):
    total = work_totals(result, planner, horizon)
    for key in EXTRA_WORK:
        total[key] = sum(frame["rollout"].get(key, 0) for frame in result["frames"])
    total["tagged_planning_events"] = sum(frame["rollout"].get("tagged_planning_events", 0)
                                        for frame in result["frames"])
    return total


def run_tape(model, tape, protocol, frozen_baseline):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {"maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
                    protocol["planning_samples"], protocol["planning_event_cap"]),
                "arrival_rollout": BackgroundCompetitionContinuousLaneRollout(model.options, seed,
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
        "work": {label: background_work(result, planners[label], protocol["arrival_horizon"])
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
    result["decision"] = "BACKGROUND_COMPETITION_CONTINUOUS_DEVELOPMENT_" + status
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
