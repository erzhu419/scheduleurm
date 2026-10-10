"""Frozen pathwise outer S2 gate with the same once-BG continuation budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.pathwise_s2_background_redecision_lane_rollout import (
    PathwiseS2BackgroundRedecisionLaneRollout, PATHWISE_WORK,
)
from algorithm.experiments.aligned_hardware_one_step_background_redecision_development import (
    build_protocol as one_step_protocol, check_rollout as check_redecision,
    redecision_work, PREVIOUS, LABELS, path,
)
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_pathwise_s2_background_redecision_development_20260930"
AGGREGATE_RESULT = ROOT / "md/experiment_artifacts/aligned_hardware_one_step_background_redecision_development_20260930.json"
SOURCES = ("algorithm/pathwise_s2_background_redecision_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_pathwise_s2_background_redecision_development.py")


def build_protocol():
    protocol = one_step_protocol()
    protocol.update(
        experiment="pathwise_s2_background_redecision_development_v1",
        aggregate_candidate_result=str(AGGREGATE_RESULT.relative_to(ROOT)),
        method="Keep the once-BG continuation scorer, inner ordinary BG, all causal inputs, shared outer suffixes and actual free replanning unchanged. Retain every root/sample tagged S2. A root is permitted only if its S2 is no greater than the same sample's MW-root S2 on every outer sample. Minimize aggregate tagged S1 with MW ties and strict improvement as before. No class weights, new rollout or outcome filter. The same-score aggregate-rule choice and exclusions are descriptive mechanism evidence, not extra performance gates.",
        budget=protocol["budget"] + " Replace the old outer aggregate risk gate's k comparisons with at most8*k sample comparisons using short-circuit rejection; do not charge both as the operative gate. Additional work stores at most8*k outer sample S2 values. Record actual sample writes, tuple-freezing reads and two reads per sample comparison, including completed and current partial samples before outer overflow. Separately record the old aggregate diagnostic's k risk comparisons, k+1 squared-cost reads, permitted-root ordering, strict-improvement comparison and actual-root comparison; no new simulation or random draw. Four independent CPU workers execute tapes; computation remains outside the service clock.",
        hypothesis="Test whether excluding roots whose aggregate S2 passes by compensating a worse outer sample improves development flow and unfinished tails. A trajectory improvement alone does not identify the cause: per-sample S2 still pools classes, inner BG keeps its aggregate gate, and the remaining continuation uses MW.",
        execution_sources={**protocol["execution_sources"],
                           **{name: (ROOT / name).read_text() for name in SOURCES}},
    )
    return protocol


def check_rollout(result, model, protocol, planner):
    if not check_redecision(result, model, protocol, planner):
        return False
    for frame in result["frames"]:
        info = frame["rollout"]
        if any(info[key] < 0 for key in PATHWISE_WORK):
            return False
        writes = info["pathwise_sample_squared_flow_writes"]
        comparisons = info["pathwise_sample_risk_comparisons"]
        k, samples = info["candidate_count"], info["sample_count"]
        if (writes > k * samples
                or info["pathwise_sample_squared_flow_reads"] != writes + 2 * comparisons):
            return False
        if not info["eligible"] or info["planning_overflow"]:
            if any(info[key] for key in PATHWISE_WORK
                   if key not in {"pathwise_sample_squared_flow_writes",
                                  "pathwise_sample_squared_flow_reads"}):
                return False
            continue
        feasible = info["risk_feasible_candidates"]
        aggregate_feasible = info["pathwise_aggregate_feasible_candidates"]
        if (writes != k * samples
                or not feasible * samples + k - feasible <= comparisons <= k * samples
                or not feasible <= aggregate_feasible <= k
                or info["pathwise_rejected_aggregate_feasible_candidates"] != aggregate_feasible - feasible
                or info["pathwise_aggregate_risk_comparisons"] != k
                or info["pathwise_aggregate_squared_flow_reads"] != k + 1
                or info["pathwise_aggregate_ordering_comparisons"] != aggregate_feasible - 1
                or info["pathwise_aggregate_improvement_comparisons"] != 1
                or info["pathwise_root_change_comparisons"] != 1
                or info["pathwise_changed_aggregate_root"] not in (0, 1)
                or info["pathwise_changed_aggregate_root"] > int(aggregate_feasible > feasible)):
            return False
    return True


def pathwise_work(result, planner, horizon):
    total = redecision_work(result, planner, horizon)
    for key in PATHWISE_WORK:
        total[key] = sum(frame["rollout"].get(key, 0) for frame in result["frames"])
    total["risk_gate_comparisons"] = total["pathwise_sample_risk_comparisons"]
    return total


def run_tape(model, tape, protocol, frozen_policies, frozen_checks):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {
        "maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"]),
        "arrival_rollout": PathwiseS2BackgroundRedecisionLaneRollout(model.options, seed,
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
    row = {
        **{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": replay_matches(policies["maxweight"], frozen_policies["maxweight"]),
        "rollout_checks_passed": check_rollout(policies["arrival_rollout"], model, protocol, planners["arrival_rollout"]),
        "cached_reference_checks_passed": frozen_checks["baseline_reproduction_passed"] and frozen_checks["rollout_checks_passed"],
        "snapshots": {key: cohort_metrics(policies, protocol[key], protocol["birth_cohort"])
                      for key in ("arrival_horizon", "observation_end")},
        "reference_snapshots": {key: cohort_metrics(reference, protocol[key], protocol["birth_cohort"])
                                for key in ("arrival_horizon", "observation_end")},
        "work": {label: pathwise_work(result, planners[label], protocol["arrival_horizon"])
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
    with ProcessPoolExecutor(max_workers=4) as workers:
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
                                   "arrival_rollout": "pathwise_S2_once_BG_redecision_candidate"},
        cached_reference_reused=True, reference_rerun=False,
        baseline_check_scope="Frozen BG passing checks plus newly executed native MW identity; no new BG reproduction.",
        work_scope="Candidate work repeats the same executed candidate accounting; cached baseline has no new execution work.")
    executed = result["execution_checks_passed"] and reference["execution_checks_passed"]
    passed = result["performance_gain_established"] and reference["performance_gain_established"]
    result["performance_vs_maxweight_established"] = result["performance_gain_established"]
    result["performance_gain_established"], result["execution_checks_passed"] = passed, executed
    result["decision"] = "PATHWISE_S2_BACKGROUND_REDECISION_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD")
    result["reference_comparison"] = reference
    aggregate = json.loads(AGGREGATE_RESULT.read_text())
    result["aggregate_candidate_comparison"] = {
        "scope": "Descriptive reused-development comparison, not an additional gate or independent confirmation.",
        "result": protocol["aggregate_candidate_result"], "decision": aggregate["decision"],
        "pooled": aggregate["reference_comparison"]["pooled"],
    }
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
