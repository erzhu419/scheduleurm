"""Frozen class-normalized second-moment gate on the original development tapes."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.normalized_class_tail_continuous_lane_rollout import NormalizedClassTailContinuousLaneRollout
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_arrival_aware_lane_rollout import rollout_checks as population_checks
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_continuous_arrival_age_tail_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_normalized_class_tail_continuous_development_20260930"
LABELS = ("maxweight", "arrival_rollout")
CLASS_SOURCE = "algorithm/class_tail_continuous_lane_rollout.py"
SOURCES = (CLASS_SOURCE, "algorithm/normalized_class_tail_continuous_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_normalized_class_tail_continuous.py")
EXTRA_WORK = ("class_squared_accumulations", "class_risk_comparisons", "normalization_divisions",
              "normalization_additions", "normalization_comparisons", "normalization_coordinate_reads")


def path(prefix, suffix):
    return prefix.with_name(prefix.name + suffix)


def build_protocol():
    old = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    if json.loads(PREVIOUS.with_suffix(".json").read_text())["decision"] != "CONTINUOUS_ARRIVAL_AGE_TAIL_DEVELOPMENT_HOLD":
        raise ValueError("original continuous-arrival HOLD must remain recorded")
    for name, source in old["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    class_protocol = ROOT / "md/experiment_artifacts/aligned_hardware_class_tail_continuous_development_20260930_protocol.json"
    if (ROOT / CLASS_SOURCE).read_text() != json.loads(class_protocol.read_text())["execution_sources"][CLASS_SOURCE]:
        raise ValueError("frozen per-class scorer changed")
    return {
        "experiment": "normalized_class_tail_continuous_development_v1", "phase": "development",
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        **{key: old[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "tapes", "arrival_seeds",
            "environment_arrival_end", "observation_end", "birth_cohort", "arrival_horizon",
            "cutoff_semantics", "primary", "information", "limitations")},
        "labels": list(LABELS),
        "method": "Replace only the original aggregate S2<=MW constraint with exact sum_c[S2_c(root)/S2_c(MW)]<=A, A=number of positive MW class totals across the same eight populations. Cached completion offsets are positive, so zero totals mean absent classes and are excluded. This allows class compensation and raw aggregate S2 growth; neither the old aggregate nor componentwise class constraint is additionally imposed. Minimize unchanged global S1, retain MW on ties, require strict S1 improvement. One tau_max, cumulative observed births/now, eight shared independent populations, cap128 whole-decision MW overflow fallback, same queue envelope and MW continuation. Risk population contains visible unadmitted and sampled future jobs; already admitted jobs are excluded by definition while all lane commitments remain fixed. No real future or class coefficients enter the policy.",
        "budget": "Original continuation bound8*m*(N+128-1+R*(ceil(tau_max)+1)) with m probes per call and2*m root probes, flow evaluations<=8*m*(N+128). Additional class accumulations=flow evaluations+k*8*C, where k is actual feasible roots and C is class dimension. Normalization charges C+k*A coordinate reads, k*A exact divisions/additions, k comparisons; k<=m,A<=C. No nested samples or continuations. Sampling, catalogue preparation and actual context extraction remain charged; computing does not delay service.",
        "statistics": "Three reused development seeds41..43, eight paired catalogue/load cells per seed. No new independent confirmation or 24-tape independence claim.",
        "execution_sources": {**old["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def check_rollout(result, model, protocol, planner):
    if not population_checks(result, model, protocol):
        return False
    for frame in result["frames"]:
        info, now, counts = frame["rollout"], frame["start"], tuple(frame["observed_arrival_counts"])
        n, dimension = sum(frame["queue"]), len(counts)
        ages = frame["waiting_ages"]
        if tuple(map(len, ages)) != tuple(frame["queue"]) or any(
                any(age < 0 for age in group) or list(group) != sorted(group, reverse=True) for group in ages):
            return False
        if (tuple(info["observed_arrival_counts"]) != counts or info["rate_exposure_s"] != now
                or tuple(info["estimated_arrival_rates"]) != tuple(value / now if now else 0. for value in counts)
                or info["rate_estimation_coordinate_reads"] != dimension
                or info["rate_estimation_divisions"] != (dimension if now else 0)):
            return False
        if info["eligible"] and info["window_s"] != planner.maximum_duration:
            return False
        if not info["eligible"] or info["planning_overflow"]:
            if any(info[key] for key in EXTRA_WORK) or info["active_risk_classes"]:
                return False
            continue
        roots, samples, active = info["candidate_count"], info["sample_count"], info["active_risk_classes"]
        if (len(info["baseline_class_squared_flow_s2"]) != dimension
                or len(info["chosen_class_squared_flow_s2"]) != dimension
                or active != sum(value > 0 for value in info["baseline_class_squared_flow_s2"])
                or not 1 <= active <= dimension
                or info["baseline_normalized_risk"] != active
                or info["chosen_normalized_risk"] > active):
            return False
        if (info["class_squared_accumulations"] != info["flow_evaluations"] + roots * samples * dimension
                or info["normalization_coordinate_reads"] != dimension + roots * active
                or info["normalization_divisions"] != roots * active
                or info["normalization_additions"] != roots * active
                or info["normalization_comparisons"] != roots or info["class_risk_comparisons"]):
            return False
        if (info["flow_evaluations"] > roots * samples * (n + protocol["planning_event_cap"])
                or not 1 <= info["risk_feasible_candidates"] <= roots
                or info["risk_feasible_candidates"] + info["risk_filtered_candidates"] != roots):
            return False
    return True


def normalized_work(result, planner, horizon):
    total = work_totals(result, planner, horizon)
    for key in EXTRA_WORK:
        total[key] = sum(frame["rollout"].get(key, 0) for frame in result["frames"])
    total["raw_s2_increase_decisions"] = sum(frame["rollout"]["changed"] and
        frame["rollout"]["chosen_predicted_squared_flow_s2"] > frame["rollout"]["baseline_predicted_squared_flow_s2"]
        for frame in result["frames"])
    return total


def run_tape(model, tape, protocol, frozen_baseline):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {"maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
                    protocol["planning_samples"], protocol["planning_event_cap"]),
                "arrival_rollout": NormalizedClassTailContinuousLaneRollout(model.options, seed,
                    protocol["planning_samples"], protocol["planning_event_cap"])}
    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"],
        planners[label].maxweight if label == "maxweight" else planners[label], policy=label)
        for label in LABELS}
    return {**{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": replay_matches(policies["maxweight"], frozen_baseline),
        "rollout_checks_passed": check_rollout(policies["arrival_rollout"], model, protocol, planners["arrival_rollout"]),
        "snapshots": {key: cohort_metrics(policies, protocol[key], protocol["birth_cohort"])
                      for key in ("arrival_horizon", "observation_end")},
        "work": {label: normalized_work(result, planners[label], protocol["arrival_horizon"])
                 for label, result in policies.items()},
        "middle_queue_area_s": {label: queue_area(result) for label, result in policies.items()}}, policies


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
        raise ValueError("registered sources, normalized gate, budgets or literal tapes changed")
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
    result["decision"] = "NORMALIZED_CLASS_TAIL_CONTINUOUS_DEVELOPMENT_" + status
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
