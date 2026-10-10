"""Frozen per-class once-BG S2 protection relative to the ordinary BG incumbent."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.class_protected_bg_incumbent_background_redecision_lane_rollout import (
    ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout, CLASS_PROTECTION_WORK,
)
from algorithm.experiments.aligned_hardware_bg_incumbent_background_redecision_development import (
    build_protocol as incumbent_protocol, check_rollout as incumbent_checks,
    incumbent_work, PREVIOUS, LABELS, path,
)
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_class_protected_bg_incumbent_development_20260930"
SOURCES = ("algorithm/class_protected_bg_incumbent_background_redecision_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_class_protected_bg_incumbent_development.py")


def build_protocol():
    protocol = incumbent_protocol()
    protocol.update(
        experiment="class_protected_bg_incumbent_development_v1",
        method="Share the original eight tagged-prefix samples across both stages. First choose the ordinary BG incumbent with its original aggregate S2<=MW-root gate and S1 rule. In the same once-BG-then-MW continuation model, decompose the exact existing target population's squared flow by class: current visible FIFO jobs and sampled prefix births through tau_max; later background jobs compete but do not contribute target cost. Retain the old aggregate MW-feasible set, then require once-BG aggregate S2<=the same-state BG incumbent and every class S2<=that incumbent in the same model. Minimize aggregate S1, with BG-first ties and strict improvement, otherwise retain BG. Prefix/original-stage overflow uses MW; new-stage overflow retains verified BG with partial fees. Actual execution freely replans. No class weights, MW-relative class constraint, outcome blacklist, real future or successor commitment.",
        budget=protocol["budget"].replace("Four independent CPU workers execute the24 tapes.",
            "Eight independent CPU workers execute the24 tapes.") +
            " Class decomposition adds k*d zero initializations and one exact squared-flow accumulation per tagged admission, reusing the aggregate square. After the aggregate gate, class checks short-circuit on a violated coordinate; charge each actual comparison and its two value reads, plus2*d reads for the reported incumbent/selected class vectors. Partial accumulations survive new-stage overflow. There are no additional samples, continuation calls or random draws.",
        hypothesis="Same-model per-class protection relative to ordinary BG may prevent cross-class cost transfers while retaining aggregate S1 gain; test all unchanged MW/BG-relative service gates on all24 development tapes. This is a new hypothesis after the failed class<=MW branch and does not guarantee actual trajectory or per-class tail improvement.",
        class_outcomes="Descriptive per-class paired cohort metrics at900/1200 against both native MW and cached confirmed ordinary BG, across every class/catalogue/load/seed. No actual outcome enters the selector or changes acceptance gates.",
        execution_sources={**protocol["execution_sources"],
                           **{name: (ROOT / name).read_text() for name in SOURCES}},
    )
    return protocol


def check_rollout(result, model, protocol, planner):
    if not incumbent_checks(result, model, protocol, planner):
        return False
    dimension = len(model.workloads)
    for frame in result["frames"]:
        info = frame["rollout"]
        k = info["candidate_count"]
        if (any(info[key] < 0 for key in CLASS_PROTECTION_WORK)
                or info["class_squared_initializations"] != k * dimension
                or info["class_squared_accumulations"] != info["flow_evaluations"]):
            return False
        baseline = info["baseline_predicted_class_squared_flow_s2"]
        chosen = info["chosen_predicted_class_squared_flow_s2"]
        aggregate = info["aggregate_risk_feasible_candidates"]
        if not info["eligible"] or info["planning_overflow"]:
            if (baseline or chosen or aggregate or info["class_risk_filtered_candidates"]
                    or info["class_squared_score_reads"] or info["class_risk_comparisons"]):
                return False
            continue
        feasible = info["risk_feasible_candidates"]
        comparisons = info["class_risk_comparisons"]
        if (len(baseline) != dimension or len(chosen) != dimension
                or any(value > control for value, control in zip(chosen, baseline, strict=True))
                or not 1 <= feasible <= aggregate <= info["incumbent_risk_feasible_candidates"]
                or info["class_risk_filtered_candidates"] != aggregate - feasible
                or not dimension * feasible + aggregate - feasible <= comparisons <= dimension * aggregate
                or info["class_squared_score_reads"] != 2 * comparisons + 2 * dimension):
            return False
    return True


def class_work(result, planner, horizon):
    total = incumbent_work(result, planner, horizon)
    for key in (*CLASS_PROTECTION_WORK, "aggregate_risk_feasible_candidates", "class_risk_filtered_candidates"):
        total[key] = sum(frame["rollout"].get(key, 0) for frame in result["frames"])
    return total


def run_tape(model, tape, protocol, frozen_policies, frozen_checks):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {
        "maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"]),
        "arrival_rollout": ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"])}

    def select_candidate(*context):
        decision = planners["arrival_rollout"](*context)
        counter = planners["arrival_rollout"].decision_counter
        if counter % 100 == 0:
            print(tape["tape_id"], "candidate dispatch", counter,
                  "service time", context[3], flush=True)
        return decision

    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"],
        planners[label].maxweight if label == "maxweight" else select_candidate, policy=label)
        for label in LABELS}
    reference = {"maxweight": frozen_policies["arrival_rollout"], "arrival_rollout": policies["arrival_rollout"]}
    class_snapshots = []
    for coordinate, name in enumerate(model.workloads):
        filtered = {label: {**result, "jobs": [job for job in result["jobs"]
                    if job["coordinate"] == coordinate]} for label, result in policies.items()}
        class_reference = {"maxweight": {**frozen_policies["arrival_rollout"], "jobs": [
            job for job in frozen_policies["arrival_rollout"]["jobs"] if job["coordinate"] == coordinate]},
            "arrival_rollout": filtered["arrival_rollout"]}
        class_snapshots.append({"class": name, "coordinate": coordinate,
            "snapshots": {key: cohort_metrics(filtered, protocol[key], protocol["birth_cohort"])
                          for key in ("arrival_horizon", "observation_end")},
            "reference_snapshots": {key: cohort_metrics(class_reference, protocol[key], protocol["birth_cohort"])
                                    for key in ("arrival_horizon", "observation_end")}})
    row = {
        **{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": replay_matches(policies["maxweight"], frozen_policies["maxweight"]),
        "rollout_checks_passed": check_rollout(policies["arrival_rollout"], model, protocol, planners["arrival_rollout"]),
        "cached_reference_checks_passed": frozen_checks["baseline_reproduction_passed"] and frozen_checks["rollout_checks_passed"],
        "snapshots": {key: cohort_metrics(policies, protocol[key], protocol["birth_cohort"])
                      for key in ("arrival_horizon", "observation_end")},
        "reference_snapshots": {key: cohort_metrics(reference, protocol[key], protocol["birth_cohort"])
                                for key in ("arrival_horizon", "observation_end")},
        "class_snapshots": class_snapshots,
        "work": {label: class_work(result, planners[label], protocol["arrival_horizon"])
                 for label, result in policies.items()},
        "middle_queue_area_s": {label: queue_area(result) for label, result in policies.items()},
        "reference_middle_queue_area_s": queue_area(frozen_policies["arrival_rollout"])}
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
        raise ValueError("registered sources, budgets, comparisons or literal tapes changed")
    if BASE.with_suffix(".json").exists() or path(BASE, "_trajectories.json.gz").exists():
        raise FileExistsError("preserve existing results")
    previous = json.loads(PREVIOUS.with_suffix(".json").read_text())
    frozen_checks = {row["tape_id"]: row for row in previous["paired_runs"]}
    with gzip.open(ROOT / protocol["previous_trajectories"], "rt") as source:
        frozen = {run["tape_id"]: run["policies"] for run in json.load(source)["runs"]}
    models = {model.node: model for model in load_models(protocol["models"])}
    completed = {}
    with ProcessPoolExecutor(max_workers=8) as workers:
        futures = {workers.submit(run_tape, models[tape["node"]], tape, protocol,
                    frozen[tape["tape_id"]], frozen_checks[tape["tape_id"]]): tape["tape_id"]
                   for tape in protocol["tapes"]}
        for future in as_completed(futures):
            row, policies = future.result()
            completed[row["tape_id"]] = (row, policies)
            print(row["tape_id"], "complete", row["baseline_reproduction_passed"],
                  row["rollout_checks_passed"], flush=True)
    rows = [completed[tape["tape_id"]][0] for tape in protocol["tapes"]]
    runs = [{"tape_id": tape["tape_id"], "policies": completed[tape["tape_id"]][1]}
            for tape in protocol["tapes"]]
    result = summarize(rows, protocol)
    reference_rows = [{**row, "snapshots": row["reference_snapshots"],
        "baseline_reproduction_passed": row["baseline_reproduction_passed"] and row["cached_reference_checks_passed"],
        "work": {"maxweight": {}, "arrival_rollout": row["work"]["arrival_rollout"]},
        "middle_queue_area_s": {"maxweight": row["reference_middle_queue_area_s"],
                                "arrival_rollout": row["middle_queue_area_s"]["arrival_rollout"]}}
        for row in rows]
    reference = summarize(reference_rows, protocol)
    reference.update(policy_roles={"maxweight": "cached_confirmed_background_method_on_development_tapes",
                                   "arrival_rollout": "class_protected_BG_incumbent_candidate"},
        cached_reference_reused=True, reference_rerun=False,
        baseline_check_scope="Frozen BG passing checks plus newly executed native MW identity; no new BG reproduction.",
        work_scope="Candidate work repeats the same executed candidate accounting; cached baseline has no new execution work.")
    executed = result["execution_checks_passed"] and reference["execution_checks_passed"]
    passed = result["performance_gain_established"] and reference["performance_gain_established"]
    result["performance_vs_maxweight_established"] = result["performance_gain_established"]
    result["performance_gain_established"], result["execution_checks_passed"] = passed, executed
    result["decision"] = "CLASS_PROTECTED_BG_INCUMBENT_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD")
    result["reference_comparison"] = reference
    result["birth_cohort"], result["cutoff_semantics"] = protocol["birth_cohort"], protocol["cutoff_semantics"]
    result["total_executed_trajectories"], result["reused_reference_trajectories"] = 2 * len(rows), len(rows)
    result["historical_reference_work_location"] = protocol["previous_result"]
    result["middle_queue_area_s"] = {label: sum(row["middle_queue_area_s"][label] for row in rows) for label in LABELS}
    result["reference_middle_queue_area_s"] = sum(row["reference_middle_queue_area_s"] for row in rows)
    with gzip.open(path(BASE, "_trajectories.json.gz"), "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "runs": runs}, out, separators=(",", ":"), allow_nan=False)
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "pooled": result["pooled"],
        "reference_flow_improvement": reference["pooled_T_flow_improvement"],
        "reference_failed_secondary_groups": reference["failed_secondary_groups"]}), flush=True)


if __name__ == "__main__":
    main()
