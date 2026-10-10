"""Remove only the second S2 gate from the frozen two-BG development policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from math import ceil
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.bg_incumbent_background_redecision_lane_rollout import INCUMBENT_WORK
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import (
    TwoStepBGIncumbentOriginalRiskLaneRollout,
)
from algorithm.experiments.aligned_hardware_bg_incumbent_background_redecision_development import incumbent_work
from algorithm.experiments.aligned_hardware_two_step_bg_incumbent_development import (
    BASE as PARENT, build_protocol as two_step_protocol,
)
from algorithm.one_step_background_redecision_lane_rollout import RECOURSE_WORK
from algorithm.experiments.aligned_hardware_one_step_background_redecision_development import (
    PREVIOUS, LABELS, path,
)
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_two_step_bg_incumbent_original_risk_development_20261001"
SOURCES = ("algorithm/two_step_bg_incumbent_original_risk_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_two_step_bg_incumbent_original_risk_development.py")


def build_protocol():
    protocol = two_step_protocol()
    frozen = json.loads(path(PARENT, "_protocol.json").read_text())
    result = json.loads(PARENT.with_suffix(".json").read_text())
    if ({key: value for key, value in frozen.items() if key != "registered_at_utc"} != protocol
            or result["registered_at_utc"] != frozen["registered_at_utc"]
            or result["decision"] != "TWO_STEP_BG_INCUMBENT_DEVELOPMENT_HOLD"
            or not result["execution_checks_passed"]):
        raise ValueError("unchanged execution-valid two-BG development parent required")
    protocol.update(
        experiment="two_step_bg_incumbent_original_risk_development_v1",
        ablation_parent_protocol=str(path(PARENT, "_protocol.json").relative_to(ROOT)),
        ablation_parent_result=str(PARENT.with_suffix(".json").relative_to(ROOT)),
        method="Keep the frozen fixed-two-BG scorer, shared eight outer tagged-prefix populations, causal cumulative counts, post-prefix RNG restoration before the first virtual call, continued inner RNG/counter at the second, and all local/outer overflow handling. Keep ordinary BG's original S2<=MW-root feasible set and its verified incumbent. Remove only the outer second-stage two-BG S2<=BG-incumbent comparison and filter. Among original-risk-feasible roots minimize two-BG aggregate S1 with BG-first ties and strict S1 improvement; otherwise retain BG. Report actual two-BG S2 even if it increases relative to BG. Original-stage applied S2 remains<=MW. Inner ordinary BG still uses its original native risk gate. No new continuation call, coefficient, class constraint, future information, outcome filtering or successor commitment.",
        budget=protocol["budget"] + " Outer second-stage membership checks remain k; second-stage S2 comparisons are zero. S1 ordering comparisons remain k+p-2, where p is the original ordinary-BG risk-feasible count; strict improvement comparison remains one. No new policy scoring work.",
        hypothesis="The second outer S2 gate rejected some two-BG S1-optimal original actions in the frozen same-state diagnostic. Removing only that gate may improve service performance, but its conditional effects were mixed. Execute all24 development tapes and keep every original MW/BG-relative mean, load, P90 and unfinished gate unchanged.",
        model_limitation="Two ordinary BG virtual redecisions followed by MW still approximate actual ongoing replanning. Removing the second S2 gate does not establish better true ranking or actual tails. The original aggregate S2 gate remains; every applied two-model S2 value is reported without imposing a second gate.",
        execution_sources={**protocol["execution_sources"],
                           **{name: (ROOT / name).read_text() for name in SOURCES}},
    )
    return protocol


def check_rollout(result, model, protocol, planner):
    m, r, dimension = len(model.options), model.lanes, len(model.workloads)
    samples, cap = protocol["planning_samples"], protocol["planning_event_cap"]
    limit, rng_words = r * planner.batch_width, len(planner.rng.getstate()[1])
    slack = r * (ceil(planner.maximum_duration) + 1)
    inner_bound = samples * m * (limit + cap - 1 + slack)
    for frame in result["frames"]:
        info, now = frame["rollout"], frame["start"]
        counts, queue = tuple(frame["observed_arrival_counts"]), tuple(frame["queue"])
        n, ages = sum(queue), frame["waiting_ages"]
        mw, _ = planner._best_index(queue)
        actual, = frame["option_indices"]
        bg = info["incumbent_root_index"]
        k, old_k = info["candidate_count"], info["incumbent_candidate_count"]
        if (tuple(map(len, ages)) != queue or any(
                any(age < 0 for age in group) or list(group) != sorted(group, reverse=True) for group in ages)
                or info["queue_limit"] != limit or info["eligible"] != (0 < n <= limit)
                or tuple(info["observed_arrival_counts"]) != counts
                or tuple(info["estimated_arrival_rates"]) != tuple(v / now if now else 0. for v in counts)
                or info["rate_exposure_s"] != now
                or info["rate_estimation_coordinate_reads"] != dimension
                or info["rate_estimation_divisions"] != (dimension if now else 0)
                or info["incumbent_maxweight_root_index"] != mw
                or info["changed"] != (actual != mw)
                or info["incumbent_changed"] != (bg is not None and bg != mw)
                or info["improved_incumbent"] != (bg is not None and actual != bg)
                or not 0 <= k <= old_k <= m
                or info["root_option_probes"] != m * (1 + int(k > 0))
                or info["incumbent_root_option_probes"] != m * int(old_k > 0)
                or info["planning_events"] != info["tagged_planning_events"] + info["background_planning_events"]
                or any(info[key] < 0 for key in (*INCUMBENT_WORK, *RECOURSE_WORK, *TWO_STEP_WORK))):
            return False
        positive_rates = sum(v > 0 for v in info["estimated_arrival_rates"])
        for prefix, roots, overflow in (("", k, info["background_rollout_overflow"]),
                ("incumbent_", old_k, info["incumbent_background_rollout_overflow"])):
            calls = info[prefix + "rollout_calls"]
            events = info[prefix + "background_planning_events"]
            seeds = info[prefix + "background_seed_initializations"]
            if (info[prefix + "rollout_option_probes"] != m * calls
                    or calls > samples * roots * (n + cap - 1 + slack)
                    or seeds > info["sample_count"]
                    or info[prefix + "background_planning_rate_reads"] != dimension * seeds + events - int(overflow)
                    or info[prefix + "background_planning_random_draws"] != positive_rates * seeds + events - int(overflow)
                    or info["tagged_planning_events"] + events > info["sample_count"] * (cap + 1)
                    or info[prefix + "admitted_job_reads"] != info[prefix + "flow_evaluations"] + info[prefix + "background_admissions"]
                    or info[prefix + "tag_flag_reads"] != info[prefix + "admitted_job_reads"]
                    or info[prefix + "admitted_job_reads"] > samples * roots * (n + cap)
                    or info[prefix + "flow_evaluations"] > roots * (samples * n + info["tagged_planning_events"])):
                return False
            if roots and not overflow and info[prefix + "flow_evaluations"] != roots * (
                    samples * n + info["tagged_planning_events"]):
                return False
        calls, eligible = info["recourse_calls"], info["recourse_eligible_calls"]
        first, second = info["recourse_first_calls"], info["recourse_second_calls"]
        idle_updates = info["recourse_virtual_idle_counter_updates"]
        if (info["recourse_root_count_reads"] != dimension
                or info["recourse_rng_words_copied"] != rng_words * int(k > 0)
                or info["recourse_counter_reads"] != int(k > 0) + idle_updates
                or info["recourse_rng_words_restored"] != rng_words * first
                or info["recourse_counter_writes"] != first + idle_updates
                or calls != first + second
                or not 0 <= second <= first <= samples * k
                or idle_updates > info["rollout_idle_calls"]
                or info["recourse_context_coordinate_reads"] != dimension * calls
                or info["recourse_context_lane_reads"] != r * calls
                or not 0 <= eligible <= calls <= 2 * samples * k
                or info["recourse_changed_calls"] > eligible
                or not 0 <= info["recourse_background_overflows"] <= info["recourse_planning_overflows"] <= eligible
                or info["recourse_root_option_probes"] < m * calls
                or info["recourse_root_option_probes"] > 2 * m * calls
                or info["recourse_rollout_option_probes"] != m * info["recourse_rollout_calls"]
                or info["recourse_rollout_calls"] > eligible * inner_bound
                or info["recourse_rate_estimation_coordinate_reads"] != dimension * calls
                or info["recourse_rate_estimation_divisions"] > dimension * calls
                or info["recourse_planning_events"] != info["recourse_tagged_planning_events"] + info["recourse_background_planning_events"]
                or info["recourse_planning_events"] > samples * eligible * (cap + 1)
                or info["recourse_admitted_job_reads"] != info["recourse_flow_evaluations"] + info["recourse_background_admissions"]
                or info["recourse_tag_flag_reads"] != info["recourse_admitted_job_reads"]
                or info["recourse_admitted_job_reads"] > samples * m * eligible * (limit + cap)
                or info["recourse_background_seed_initializations"] > samples * eligible
                or info["recourse_background_planning_rate_reads"] != dimension * info["recourse_background_seed_initializations"]
                   + info["recourse_background_planning_events"] - info["recourse_background_overflows"]):
            return False
        if not info["eligible"]:
            if bg is not None or old_k or k or info["planning_events"] or info["changed"]:
                return False
            continue
        if info["window_s"] != planner.maximum_duration or not 1 <= info["sample_count"] <= samples:
            return False
        if bg is None:
            if (not info["planning_overflow"] or k or actual != mw
                    or any(info[key] for key in ("incumbent_risk_comparisons", "incumbent_ordering_comparisons",
                        "incumbent_strict_improvement_comparisons", "incumbent_membership_checks",
                        "incumbent_lookahead_risk_comparisons", "incumbent_lookahead_ordering_comparisons",
                        "incumbent_lookahead_strict_improvement_comparisons"))):
                return False
            continue
        old_p = info["incumbent_risk_feasible_candidates"]
        if (not 1 <= old_p <= old_k or k != old_k or info["sample_count"] != samples
                or info["incumbent_risk_filtered_candidates"] + old_p != old_k
                or info["incumbent_risk_comparisons"] != old_k
                or info["incumbent_ordering_comparisons"] != old_k + old_p - 2
                or info["incumbent_strict_improvement_comparisons"] != 1
                or info["incumbent_predicted_completion_s"] > info["incumbent_mw_predicted_completion_s"]
                or info["incumbent_predicted_squared_flow_s2"] > info["incumbent_mw_predicted_squared_flow_s2"]
                or info["incumbent_actual_predicted_squared_flow_s2"] > info["incumbent_mw_predicted_squared_flow_s2"]
                or info["incumbent_background_rollout_overflow"]):
            return False
        if info["planning_overflow"]:
            if (not info["background_rollout_overflow"] or actual != bg
                    or info["baseline_predicted_completion_s"] is not None
                    or info["chosen_predicted_completion_s"] is not None
                    or info["risk_feasible_candidates"] or info["risk_filtered_candidates"]
                    or any(info[key] for key in ("incumbent_membership_checks", "incumbent_lookahead_risk_comparisons",
                        "incumbent_lookahead_ordering_comparisons", "incumbent_lookahead_strict_improvement_comparisons"))):
                return False
            continue
        p = info["risk_feasible_candidates"]
        if (p != old_p or p + info["risk_filtered_candidates"] != k
                or info["incumbent_membership_checks"] != k
                or info["incumbent_lookahead_risk_comparisons"] != 0
                or info["incumbent_lookahead_ordering_comparisons"] != k + p - 2
                or info["incumbent_lookahead_strict_improvement_comparisons"] != 1
                or info["chosen_predicted_completion_s"] > info["baseline_predicted_completion_s"]
                or (not info["improved_incumbent"]
                    and (info["chosen_predicted_completion_s"] != info["baseline_predicted_completion_s"]
                         or info["chosen_predicted_squared_flow_s2"] != info["baseline_predicted_squared_flow_s2"]))):
            return False
    return True


def two_step_work(result, planner, horizon):
    total = incumbent_work(result, planner, horizon)
    for key in TWO_STEP_WORK:
        total[key] = sum(frame["rollout"].get(key, 0) for frame in result["frames"])
    return total


def run_tape(model, tape, protocol, frozen_policies, frozen_checks):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {
        "maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"]),
        "arrival_rollout": TwoStepBGIncumbentOriginalRiskLaneRollout(model.options, seed,
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
        "removed_gate_diagnostics": {
            "completed_second_stages": sum(frame["rollout"]["candidate_count"] > 0
                and not frame["rollout"]["planning_overflow"]
                for frame in policies["arrival_rollout"]["frames"]),
            "applied_two_model_S2_above_BG": sum(
                frame["rollout"]["chosen_predicted_squared_flow_s2"]
                > frame["rollout"]["baseline_predicted_squared_flow_s2"]
                for frame in policies["arrival_rollout"]["frames"]
                if frame["rollout"]["candidate_count"] > 0
                and not frame["rollout"]["planning_overflow"]),
        },
        "work": {label: two_step_work(result, planners[label], protocol["arrival_horizon"])
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
                                   "arrival_rollout": "two_BG_incumbent_original_risk_candidate"},
        cached_reference_reused=True, reference_rerun=False,
        baseline_check_scope="Frozen BG passing checks plus newly executed native MW identity; no new BG reproduction.",
        work_scope="Candidate work repeats the same executed candidate accounting; cached baseline has no new execution work.")
    executed = result["execution_checks_passed"] and reference["execution_checks_passed"]
    passed = result["performance_gain_established"] and reference["performance_gain_established"]
    result["performance_vs_maxweight_established"] = result["performance_gain_established"]
    result["performance_gain_established"], result["execution_checks_passed"] = passed, executed
    result["decision"] = "TWO_STEP_BG_INCUMBENT_ORIGINAL_RISK_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD")
    result["reference_comparison"] = reference
    result["removed_gate_diagnostics"] = {
        key: sum(row["removed_gate_diagnostics"][key] for row in rows)
        for key in rows[0]["removed_gate_diagnostics"]}
    result["ablation_parent_result"] = protocol["ablation_parent_result"]
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
