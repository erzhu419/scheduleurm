"""Frozen once-BG-redecision root scoring against MW and confirmed BG."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from math import ceil
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.one_step_background_redecision_lane_rollout import (
    OneStepBackgroundRedecisionLaneRollout, RECOURSE_WORK,
)
from algorithm.experiments.aligned_hardware_background_competition_continuous import (
    path, LABELS, background_work, check_rollout as check_outer_rollout,
)
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_background_competition_continuous_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_one_step_background_redecision_development_20260930"
SOURCES = ("algorithm/one_step_background_redecision_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_one_step_background_redecision_development.py")


def build_protocol():
    old = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    previous = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if (previous["decision"] != "BACKGROUND_COMPETITION_CONTINUOUS_DEVELOPMENT_PASS"
            or not previous["execution_checks_passed"] or not previous["performance_gain_established"]
            or previous["registered_at_utc"] != old["registered_at_utc"]):
        raise ValueError("matching frozen passing BG development reference required")
    for name, source in old["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    return {
        "experiment": "one_step_background_redecision_development_v1", "phase": "development",
        "previous_protocol": str(path(PREVIOUS, "_protocol.json").relative_to(ROOT)),
        "previous_result": str(PREVIOUS.with_suffix(".json").relative_to(ROOT)),
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        **{key: old[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "tapes", "arrival_seeds",
            "environment_arrival_end", "observation_end", "birth_cohort", "arrival_horizon",
            "cutoff_semantics", "information", "limitations", "statistics")},
        "labels": list(LABELS),
        "primary": "Require the unchanged MW-relative overall mean gain of at least1%, load mean nonregression, sixteen group5% common-P90/unfinished and pooled unfinished gates. Require identical gates against cached current BG development trajectories, including at least1% further overall mean reduction. Per-class outcomes remain descriptive; no post-outcome class gate.",
        "method": "Keep all feasible roots, original eight one-tau_max tagged populations and shared continuing background suffixes. For each root/sample, replace only the first nonempty virtual continuation with one ordinary frozen BG decision using that dispatch's current FIFO ages, full lane reservations and cumulative released counts divided by absolute time. Later virtual continuation is MW. The inner BG sees only released births and independently samples its own future; no unobserved outer realized suffix enters it. Restore the outer post-prefix RNG state before each inner call and use the same planning seed with counter=outercounter+1+prior virtual idle calls. Inner randomness never advances outer online RNG/counter. Inner overflow uses its ordinary local MW fallback; outer prefix/background overflow retains whole-decision MW fallback. Root rule remains aggregate tagged S2<=same-state MW-root under this same once-BG continuation, minimum S1, MW ties and strict improvement. Actual calls freely repeat root planning with no successor commitment or saved action. No real future, evaluation time, load, class weights or outcome filter enters the selector.",
        "budget": "Fixed outer and inner samples8 and eventcap128; at mostone inner BG decision per outer root/sample. For m options, k<=m roots, R lanes, G=R*B, L=R*(ceil(tau_max)+1), outer calls<=8*k*(N+128-1+L), each m probes, with2*m root probes. Inner calls<=8*k; each eligible inner BG has Ninner<=G, at most8*m*(G+128-1+L) continuation calls and2*m root probes (ineligible/prefix overflow uses m). Thus total nested continuation calls<=64*k*m*(G+128-1+L). Outer admitted/tag reads<=8*k*(N+128); per eligible inner<=8*m*(G+128). Every inner prefix+suffix stops at combined event129 with local MW and retains partial work. Outer and recourse fees are separate and both counted, including rate/context extraction, all RNG state copies/restores, counter reads/writes, risk comparisons and the one reusable ordinary BG constructor's preprocessing. Computation does not delay service.",
        "reference_scope": "Only24 literal development tapes41..43 and stored passing BG trajectories. No fresh confirmation result is read. Execute native MW and the new candidate per tape,48 new trajectories; verify native MW identity and cached BG passing checks. Historical BG work stays in previous_result.",
        "model_limitation": "One ordinary BG redecision approximates continued online replanning; subsequent virtual decisions remain MW, while actual candidate continues to replan. No exact value improvement or closure of the full online policy is claimed.",
        "execution_sources": {**old["execution_sources"],
                              **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def check_rollout(result, model, protocol, planner):
    if not check_outer_rollout(result, model, protocol, planner):
        return False
    m, r, dimension = len(model.options), model.lanes, len(model.workloads)
    samples, cap = protocol["planning_samples"], protocol["planning_event_cap"]
    limit = r * planner.batch_width
    rng_words = len(planner.rng.getstate()[1])
    continuation_bound = samples * m * (limit + cap - 1 + r * (ceil(planner.maximum_duration) + 1))
    for frame in result["frames"]:
        info = frame["rollout"]
        calls, eligible = info["recourse_calls"], info["recourse_eligible_calls"]
        incumbent, _ = planner._best_index(tuple(frame["queue"]))
        actual, = frame["option_indices"]
        if (info["changed"] != (actual != incumbent)
                or any(info[key] < 0 for key in RECOURSE_WORK)
                or info["recourse_root_count_reads"] != dimension
                or info["recourse_rng_words_copied"] != rng_words * int(info["candidate_count"] > 0)
                or info["recourse_rng_words_restored"] != rng_words * calls
                or info["recourse_counter_reads"] != int(info["candidate_count"] > 0)
                or info["recourse_counter_writes"] != calls
                or info["recourse_context_coordinate_reads"] != dimension * calls
                or info["recourse_context_lane_reads"] != r * calls
                or not 0 <= eligible <= calls <= samples * info["candidate_count"]
                or info["recourse_changed_calls"] > eligible
                or not 0 <= info["recourse_background_overflows"]
                   <= info["recourse_planning_overflows"] <= eligible
                or info["recourse_rollout_option_probes"] != m * info["recourse_rollout_calls"]
                or info["recourse_rollout_calls"] > eligible * continuation_bound
                or not m * calls <= info["recourse_root_option_probes"] <= 2 * m * calls
                or info["recourse_rate_estimation_coordinate_reads"] != dimension * calls
                or info["recourse_rate_estimation_divisions"] > dimension * calls
                or info["recourse_planning_events"] != info["recourse_tagged_planning_events"]
                   + info["recourse_background_planning_events"]
                or info["recourse_planning_events"] > samples * eligible * (cap + 1)
                or info["recourse_admitted_job_reads"] != info["recourse_flow_evaluations"]
                   + info["recourse_background_admissions"]
                or info["recourse_tag_flag_reads"] != info["recourse_admitted_job_reads"]
                or info["recourse_admitted_job_reads"] > samples * m * eligible * (limit + cap)
                or info["recourse_background_seed_initializations"] > samples * eligible):
            return False
        if info["recourse_background_planning_rate_reads"] != (
                dimension * info["recourse_background_seed_initializations"]
                + info["recourse_background_planning_events"] - info["recourse_background_overflows"]):
            return False
    return True


def redecision_work(result, planner, horizon):
    total = background_work(result, planner, horizon)
    for key in RECOURSE_WORK:
        total[key] = sum(frame["rollout"].get(key, 0) for frame in result["frames"])
    total["controller_constructors"] = 1
    total["recourse_controller_constructors"] = int(hasattr(planner, "inner_planner"))
    completed = [frame["rollout"] for frame in result["frames"]
                 if frame["rollout"]["eligible"] and not frame["rollout"]["planning_overflow"]]
    total["risk_gate_comparisons"] = sum(info["candidate_count"] for info in completed)
    total["ordering_candidate_comparisons"] = sum(
        info["candidate_count"] + info["risk_feasible_candidates"] - 2 for info in completed)
    total["strict_improvement_comparisons"] = len(completed)
    total["total_option_probes"] += total["recourse_root_option_probes"] + total["recourse_rollout_option_probes"]
    total["total_option_probes"] += sum(total.get("preprocessing_recourse_" + key, 0)
        for key in ("option_probes", "age_profile_option_probes", "rate_dimension_option_probes"))
    total["total_rollout_calls"] = total["rollout_calls"] + total["recourse_rollout_calls"]
    total["total_planning_events"] = total["planning_events"] + total["recourse_planning_events"]
    return total


def run_tape(model, tape, protocol, frozen_policies, frozen_checks):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {
        "maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"]),
        "arrival_rollout": OneStepBackgroundRedecisionLaneRollout(model.options, seed,
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
        "work": {label: redecision_work(result, planners[label], protocol["arrival_horizon"])
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
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_tape(models[tape["node"]], tape, protocol,
                                 frozen[tape["tape_id"]], frozen_checks[tape["tape_id"]])
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(tape["tape_id"], "complete", row["baseline_reproduction_passed"], row["rollout_checks_passed"], flush=True)
    result = summarize(rows, protocol)
    reference_rows = [{**row, "snapshots": row["reference_snapshots"],
        "baseline_reproduction_passed": row["baseline_reproduction_passed"] and row["cached_reference_checks_passed"],
        "work": {"maxweight": {}, "arrival_rollout": row["work"]["arrival_rollout"]},
        "middle_queue_area_s": {"maxweight": row["reference_middle_queue_area_s"],
                                "arrival_rollout": row["middle_queue_area_s"]["arrival_rollout"]}}
        for row in rows]
    reference = summarize(reference_rows, protocol)
    reference.update(policy_roles={"maxweight": "cached_confirmed_background_method_on_development_tapes",
                                   "arrival_rollout": "one_step_BG_redecision_candidate"},
        cached_reference_reused=True, reference_rerun=False,
        baseline_check_scope="Frozen BG passing checks plus newly executed native MW identity; no new BG reproduction.",
        work_scope="Candidate work repeats the same executed candidate accounting; cached baseline has no new execution work.")
    executed = result["execution_checks_passed"] and reference["execution_checks_passed"]
    passed = result["performance_gain_established"] and reference["performance_gain_established"]
    result["performance_vs_maxweight_established"] = result["performance_gain_established"]
    result["performance_gain_established"] = passed
    result["execution_checks_passed"] = executed
    result["decision"] = "ONE_STEP_BACKGROUND_REDECISION_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD")
    result["reference_comparison"] = reference
    result["birth_cohort"], result["cutoff_semantics"] = protocol["birth_cohort"], protocol["cutoff_semantics"]
    result["total_executed_trajectories"] = 2 * len(rows)
    result["reused_reference_trajectories"] = len(rows)
    result["historical_reference_work_location"] = protocol["previous_result"]
    result["middle_queue_area_s"] = {label: sum(row["middle_queue_area_s"][label] for row in rows) for label in LABELS}
    result["reference_middle_queue_area_s"] = sum(row["reference_middle_queue_area_s"] for row in rows)
    with gzip.open(path(BASE, "_trajectories.json.gz"), "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "runs": runs}, out, separators=(",", ":"), allow_nan=False)
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "pooled": result["pooled"],
        "failed_secondary_groups": result["failed_secondary_groups"],
        "reference_flow_improvement": reference["pooled_T_flow_improvement"],
        "reference_failed_secondary_groups": reference["failed_secondary_groups"]}), flush=True)


if __name__ == "__main__":
    main()
