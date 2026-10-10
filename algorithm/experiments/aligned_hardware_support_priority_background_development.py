"""Frozen support-priority successor development against MW and confirmed BG."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from math import ceil
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.support_priority_background_lane_rollout import SupportPriorityBackgroundLaneRollout
from algorithm.experiments.aligned_hardware_background_competition_continuous import (
    path, LABELS, EXTRA_WORK, background_work,
)
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_background_competition_continuous_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_support_priority_background_development_20260930"
SOURCES = ("algorithm/deferred_incumbent_background_lane_rollout.py",
           "algorithm/support_priority_background_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_support_priority_background_development.py")
NEW_WORK = ("plan_candidate_count", "plan_changed", "successor_call", "successor_executed",
            "successor_changed", "successor_feasibility_coordinate_reads",
            "successor_envelope_checks", "successor_feasibility_checks",
            "successor_override_calls", "successor_envelope_passed", "successor_feasibility_passed",
            "successor_batch_changed", "successor_support_coordinate_reads", "successor_support_comparisons")
PREDICTIONS = ("baseline_predicted_completion_s", "chosen_predicted_completion_s",
               "baseline_predicted_squared_flow_s2", "chosen_predicted_squared_flow_s2")


def build_protocol():
    old = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    previous = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if (previous["decision"] != "BACKGROUND_COMPETITION_CONTINUOUS_DEVELOPMENT_PASS"
            or not previous["execution_checks_passed"] or not previous["performance_gain_established"]):
        raise ValueError("retain the passing background-competition development reference")
    for name, source in old["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    return {
        "experiment": "support_priority_background_development_v1", "phase": "development",
        "previous_protocol": str(path(PREVIOUS, "_protocol.json").relative_to(ROOT)),
        "previous_result": str(PREVIOUS.with_suffix(".json").relative_to(ROOT)),
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        **{key: old[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "tapes", "arrival_seeds",
            "environment_arrival_end", "observation_end", "birth_cohort", "arrival_horizon",
            "cutoff_semantics", "information", "limitations")},
        "labels": list(LABELS),
        "primary": "Retain all original MW-relative1% overall mean gain, load mean nonregression, sixteen group5% P90/unfinished and pooled unfinished gates. Also require the identical gates against the cached development trajectories of the current confirmed background method, including at least1% further overall mean reduction. Class outcomes remain descriptive; no post-outcome class gate.",
        "method": "Score two plans per feasible root on the same eight one-tau_max tagged populations and shared continuing background births: ordinary MW continuation, or priority for the current MW service support S on the next nonempty dispatch followed by ordinary MW. A support is the nonempty set of classes served by the saved current-MW option; planned_successor_index and pending_successor_index identify that support, not a fixed future option. One catalogue scan computes both global MW and the exact best feasible option whose nonempty service support is a subset of S, using current queue weights and native duration/index ties. Empty calls retain S; the first nonempty call consumes it, honoring the dynamic family best only inside the unchanged N<=R*B envelope and otherwise falling back to global MW. Virtual planning and actual execution use the same scan helper and guard rule. Retain aggregate tagged S2<=the same-state MW ordinary plan, global tagged S1 minimization, ordinary-control ties and strict plan improvement. Actual pending calls neither resample nor veto. No weights, new threshold, node/action blacklist, evaluation time, load or real future enter the selector.",
        "budget": "For k<=m feasible roots, score exactly2*k plans; at most8*2*k*(N+128-1+R*(ceil(tau_max)+1)) continuation calls, each scanning all m options once, plus2*m root probes. Admitted/tag reads<=8*2*k*(N+128), and tagged flow evaluations<=2*k*(8*N+tagged planning events). Shared suffix generation counts once per sample. Priority-envelope checks<=8*k, and each in-envelope family check reads D saved-support coordinates plus D coordinates for every globally feasible option, bounded by D*(m+1) reads per check; comparisons count eligible nonempty support-subset options, bounded by m per check. Legacy fixed-option feasibility reads are zero; all new support reads/comparisons and virtual/actual overrides are charged. Whole-decision overflow retains partial actual work. Actual pending calls use exactly m root probes for a single joint global/family scan, plus their actual envelope/family work. Computation does not delay service.",
        "reference_scope": "Reuse only the24 literal development tapes41..43 and their stored BG trajectories. The BG method has separate prior confirmation, but no fresh confirmation trajectory enters this experiment. Cached BG is not rerun; integrity uses its frozen passing checks and native MW identity. Two trajectories per tape execute now: native MW and the new candidate. Historical BG work stays in previous_result.",
        "statistics": old["statistics"],
        "execution_sources": {**old["execution_sources"],
                              **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def check_rollout(result, model, protocol, planner):
    m, r = len(model.options), model.lanes
    cap, samples, dimension = protocol["planning_event_cap"], protocol["planning_samples"], len(model.workloads)
    pending = None
    for frame in result["frames"]:
        info, now = frame["rollout"], frame["start"]
        queue, ages, counts = tuple(frame["queue"]), frame["waiting_ages"], tuple(frame["observed_arrival_counts"])
        n = sum(queue)
        if (tuple(map(len, ages)) != queue or any(
                any(age < 0 for age in group) or list(group) != sorted(group, reverse=True) for group in ages)
                or info["queue_limit"] != r * planner.batch_width
                or tuple(info["observed_arrival_counts"]) != counts or info["rate_exposure_s"] != now
                or tuple(info["estimated_arrival_rates"]) != tuple(value / now if now else 0. for value in counts)
                or info["rate_estimation_coordinate_reads"] != dimension
                or info["rate_estimation_divisions"] != (dimension if now else 0)):
            return False
        incumbent, _ = planner._best_index(queue)
        actual, = frame["option_indices"]
        if info["successor_call"] != (pending is not None):
            return False
        if pending is not None:
            envelope = n <= info["queue_limit"] if n else None
            scan = planner.scan_support_options(queue, pending if envelope else None)
            feasible = scan.support_index is not None if envelope else None
            executed = feasible is True
            dynamic = scan.support_index if executed else None
            if (info["eligible"] or info["sample_count"] or info["window_s"] or info["candidate_count"]
                    or info["plan_candidate_count"] or info["root_option_probes"] != m
                    or info["changed"] or info["plan_changed"] or any(info[key] for key in EXTRA_WORK)
                    or info["rollout_calls"] or info["rollout_option_probes"] or info["planning_events"]
                    or info["planning_random_draws"] or info["planning_rate_reads"] or info["waiting_age_reads"]
                    or info["flow_evaluations"] or info["risk_filtered_candidates"] or info["risk_feasible_candidates"]
                    or info["planning_overflow"] or any(info[key] is not None for key in PREDICTIONS)
                    or info["planned_successor_index"] != pending
                    or info["successor_envelope_checks"] != int(n > 0)
                    or info["successor_feasibility_checks"] != int(envelope is True)
                    or info["successor_feasibility_coordinate_reads"]
                    or info["successor_support_coordinate_reads"] != scan.support_coordinate_reads
                    or info["successor_support_comparisons"] != scan.support_comparisons
                    or info["successor_override_calls"] != int(executed)
                    or info["successor_envelope_passed"] != envelope
                    or info["successor_feasibility_passed"] != feasible
                    or info["successor_executed"] != executed
                    or info["successor_changed"] != (executed and dynamic != incumbent)
                    or info["successor_option_index"] != dynamic
                    or info["successor_batch_changed"] != (executed and dynamic != pending)
                    or actual != (dynamic if executed else incumbent)):
                return False
            if n:
                pending = None
            continue
        if (info["eligible"] != (0 < n <= info["queue_limit"]) or info["successor_executed"]
                or info["successor_changed"] or info["successor_envelope_passed"] is not None
                or info["successor_feasibility_passed"] is not None or info["successor_batch_changed"]
                or info["successor_option_index"] is not None):
            return False
        k, plans = info["candidate_count"], info["plan_candidate_count"]
        if (not 0 <= k <= m or plans != 2 * k
                or info["rollout_option_probes"] != m * info["rollout_calls"]
                or info["planning_events"] != info["tagged_planning_events"] + info["background_planning_events"]
                or info["admitted_job_reads"] != info["flow_evaluations"] + info["background_admissions"]
                or info["tag_flag_reads"] != info["admitted_job_reads"]
                or info["background_seed_initializations"] > info["sample_count"]):
            return False
        rescheduled = info["background_planning_events"] - int(info["background_rollout_overflow"])
        positive_rates = sum(value > 0 for value in info["estimated_arrival_rates"])
        calls_bound = samples * plans * (n + cap - 1 + r * (ceil(planner.maximum_duration) + 1))
        if (info["background_planning_rate_reads"] != dimension * info["background_seed_initializations"] + rescheduled
                or info["background_planning_random_draws"] != positive_rates * info["background_seed_initializations"] + rescheduled
                or info["rollout_calls"] > calls_bound
                or info["admitted_job_reads"] > samples * plans * (n + cap)
                or info["flow_evaluations"] > plans * (samples * n + info["tagged_planning_events"])
                or not 0 <= info["successor_override_calls"] <= info["successor_feasibility_checks"]
                   <= info["successor_envelope_checks"] <= samples * k
                or info["successor_feasibility_coordinate_reads"]
                or not dimension * info["successor_feasibility_checks"] <= info["successor_support_coordinate_reads"]
                   <= dimension * (m + 1) * info["successor_feasibility_checks"]
                or not info["successor_override_calls"] <= info["successor_support_comparisons"]
                   <= m * info["successor_feasibility_checks"]):
            return False
        if not info["eligible"]:
            if (info["root_option_probes"] != m or any(info[key] for key in EXTRA_WORK)
                    or any(info[key] for key in NEW_WORK) or info["changed"] or k
                    or info["planned_successor_index"] is not None or actual != incumbent):
                return False
            continue
        if (info["window_s"] != planner.maximum_duration or not 1 <= info["sample_count"] <= samples
                or info["planning_events"] > info["sample_count"] * (cap + 1)
                or info["background_planning_random_draws"] > info["planning_random_draws"]
                or info["background_planning_rate_reads"] > info["planning_rate_reads"]):
            return False
        if info["planning_overflow"]:
            if (info["changed"] or info["plan_changed"] or actual != incumbent
                    or info["planned_successor_index"] is not None
                    or any(info[key] is not None for key in PREDICTIONS)
                    or info["risk_feasible_candidates"] or info["risk_filtered_candidates"]):
                return False
            if info["background_rollout_overflow"]:
                if info["root_option_probes"] != 2 * m or not k or info["sample_count"] != samples:
                    return False
            elif info["root_option_probes"] != m or info["rollout_calls"] or k:
                return False
            continue
        if (info["background_rollout_overflow"] or info["root_option_probes"] != 2 * m
                or info["sample_count"] != samples or not 1 <= info["risk_feasible_candidates"] <= plans
                or info["flow_evaluations"] != plans * (samples * n + info["tagged_planning_events"])
                or info["risk_filtered_candidates"] + info["risk_feasible_candidates"] != plans
                or info["chosen_predicted_completion_s"] > info["baseline_predicted_completion_s"]
                or info["chosen_predicted_squared_flow_s2"] > info["baseline_predicted_squared_flow_s2"]
                or info["changed"] != (info["plan_changed"] and actual != incumbent)):
            return False
        if info["plan_changed"]:
            if not info["chosen_predicted_completion_s"] < info["baseline_predicted_completion_s"]:
                return False
            successor = info["planned_successor_index"]
            if successor is not None and successor != incumbent:
                return False
            pending = successor
        elif info["planned_successor_index"] is not None or actual != incumbent:
            return False
    return pending == planner.pending_successor_index


def deferred_work(result, planner, horizon):
    total = background_work(result, planner, horizon)
    for key in NEW_WORK:
        total[key] = sum(frame["rollout"].get(key, 0) or 0 for frame in result["frames"])
    return total


def run_tape(model, tape, protocol, frozen_policies, frozen_checks):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {
        "maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"]),
        "arrival_rollout": SupportPriorityBackgroundLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"])}
    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"],
        planners[label].maxweight if label == "maxweight" else planners[label], policy=label)
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
        "work": {label: deferred_work(result, planners[label], protocol["arrival_horizon"])
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
                                   "arrival_rollout": "support_priority_candidate"},
        cached_reference_reused=True, reference_rerun=False,
        baseline_check_scope="Frozen BG passing checks plus newly executed native MW identity; no new BG reproduction.",
        work_scope="Candidate work repeats the same executed candidate accounting; cached baseline has no new execution work.")
    executed = result["execution_checks_passed"] and reference["execution_checks_passed"]
    passed = result["performance_gain_established"] and reference["performance_gain_established"]
    result["performance_vs_maxweight_established"] = result["performance_gain_established"]
    result["performance_gain_established"] = passed
    result["execution_checks_passed"] = executed
    result["decision"] = "SUPPORT_PRIORITY_BACKGROUND_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD")
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
