"""Two virtual frozen-current decisions against unchanged MW and ordinary-BG gates."""
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
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import (
    TwoStepCurrentPolicyBGIncumbentLaneRollout, INNER_CURRENT_WORK, INNER_CURRENT_FALLBACKS,
)
from algorithm.experiments.aligned_hardware_bg_incumbent_background_redecision_development import incumbent_work
from algorithm.experiments.aligned_hardware_two_step_bg_incumbent_original_risk_development import (
    BASE as PARENT, build_protocol as parent_protocol,
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
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_two_step_current_policy_bg_incumbent_development_20261001"
SOURCES = ("algorithm/two_step_current_policy_bg_incumbent_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_two_step_current_policy_bg_incumbent_development.py")


def build_protocol():
    protocol = parent_protocol()
    frozen = json.loads(path(PARENT, "_protocol.json").read_text())
    result = json.loads(PARENT.with_suffix(".json").read_text())
    if ({key: value for key, value in frozen.items() if key != "registered_at_utc"} != protocol
            or result["registered_at_utc"] != frozen["registered_at_utc"]
            or result["decision"] != "TWO_STEP_BG_INCUMBENT_ORIGINAL_RISK_DEVELOPMENT_HOLD"
            or not result["execution_checks_passed"]):
        raise ValueError("unchanged execution-valid frozen current-policy parent required")
    protocol.update(
        experiment="two_step_current_policy_bg_incumbent_development_v1",
        continuation_parent_protocol=str(path(PARENT, "_protocol.json").relative_to(ROOT)),
        continuation_parent_result=str(PARENT.with_suffix(".json").relative_to(ROOT)),
        continuation_parent_trajectories=str(path(PARENT, "_trajectories.json.gz").relative_to(ROOT)),
        method="Replace only the two nonempty outer virtual ordinary-BG calls with the frozen current policy, TwoStepBGIncumbentOriginalRiskLaneRollout. That inner policy retains its own two ordinary-BG calls then MW and never invokes this new candidate. Preserve shared eight outer tagged-prefix samples, causal released counts/FIFO ages/full lane reservations, post-prefix RNG restoration before the first call and stateful continuation at the second. Later outer virtual calls remain MW. Keep the ordinary-BG incumbent and original S2<=MW feasible roots; choose minimum new-continuation S1 with BG-first ties and strict improvement, without a second outer S2 gate. Report actual new S2. Original outer-stage overflow uses MW; new outer-stage overflow retains BG. Each virtual current call retains its native local fallback: prefix/original-stage overflow uses MW; second-stage overflow retains the verified BG incumbent. Actual calls freely replan with no successor commitment.",
        budget="Frozen outer and every native nested sample count8 and cap128. For m options, k<=m roots, N<=R*B, G=R*B, L=R*(ceil(tau_max)+1), each outer stage has calls<=8*k*(N+128-1+L), with at most16*k virtual current calls. Each eligible virtual current call has two stages each with calls<=8*m*(G+128-1+L), plus at most16*m native BG calls; each eligible deepest BG has calls<=8*m*(G+128-1+L). Retain every main/incumbent/recourse/TWO primitive fee and partial overflow. Outer recourse_* main fields charge inner-current main work; inner_current_work charges its native incumbent and BG recourse work once. Legacy extra outer recourse risk/ordering/strict comparison fields are zero: actual comparisons reside in the native current ledger. Exactly three planners are constructed: new outer, reusable frozen current and its reusable BG; preprocessing is charged once under base, recourse_ and recourse_recourse_. Computation does not delay service. Eight workers execute24 native MW+24 candidate trajectories;24 cached ordinary-BG and24 frozen current-parent trajectories are reused locally.",
        hypothesis="The fixed22-anchor crossing found real root-ranking dependence on the continuing controller. Replacing only the first two virtual policies with the frozen current controller may improve actual service. Preserve every original MW/BG-relative overall/load mean, group common-P90/unfinished and pooled unfinished gate. Parent increments remain descriptive, with no added adoption gate.",
        model_limitation="Two calls of a finite frozen current policy approximate ongoing replanning, not an exact or recursively evaluated new-policy future. Conditional interactions did not establish overall gain; sampled S1 improvement and original S2 feasibility do not guarantee true trajectory improvement.",
        parent_comparison_scope="Reuse all24 execution-valid original-risk current-parent trajectories. Recompute direct paired cohort metrics at900/1200, class metrics and queue area. Each candidate/reference pair uses its own shared completed-ID set; do not subtract different-set P90 contrasts. Parent is not rerun; confirmation is untouched and service gates are unchanged.",
        execution_sources={**protocol["execution_sources"],
                           **{name: (ROOT / name).read_text() for name in SOURCES}},
    )
    return protocol


def check_current_recourse(info, m, r, dimension, samples, cap, limit, slack, rng_words):
    work, fallbacks = info["inner_current_work"], info["inner_current_fallbacks"]
    if (set(work) != set(INNER_CURRENT_WORK) or set(fallbacks) != set(INNER_CURRENT_FALLBACKS)
            or any(value < 0 for value in (*work.values(), *fallbacks.values()))):
        return False
    calls, eligible = info["recourse_calls"], info["recourse_eligible_calls"]
    first, second = info["recourse_first_calls"], info["recourse_second_calls"]
    idle = info["recourse_virtual_idle_counter_updates"]
    original_completed = eligible - fallbacks["prefix_overflows"] - fallbacks["original_stage_overflows"]
    new_completed = original_completed - fallbacks["second_stage_overflows"]
    bound = samples * m * (limit + cap - 1 + slack)
    if (not 0 <= new_completed <= original_completed <= eligible <= calls
            or calls != first + second or not 0 <= second <= first <= samples * info["candidate_count"]
            or idle > info["rollout_idle_calls"]
            or info["recourse_root_count_reads"] != dimension
            or info["recourse_rng_words_copied"] != rng_words * int(info["candidate_count"] > 0)
            or info["recourse_counter_reads"] != int(info["candidate_count"] > 0) + idle
            or info["recourse_rng_words_restored"] != rng_words * first
            or info["recourse_counter_writes"] != first + idle
            or info["recourse_context_coordinate_reads"] != dimension * calls
            or info["recourse_context_lane_reads"] != r * calls
            or info["recourse_changed_calls"] > eligible
            or sum(fallbacks[key] for key in INNER_CURRENT_FALLBACKS[:3]) != info["recourse_planning_overflows"]
            or fallbacks["MW_fallbacks"] != fallbacks["prefix_overflows"] + fallbacks["original_stage_overflows"]
            or fallbacks["BG_fallbacks"] != fallbacks["second_stage_overflows"]
            or info["recourse_background_overflows"] != fallbacks["second_stage_overflows"]
            or info["recourse_root_option_probes"] != m * (calls + original_completed)
            or info["recourse_rollout_option_probes"] != m * info["recourse_rollout_calls"]
            or info["recourse_rollout_calls"] > original_completed * bound
            or info["recourse_rate_estimation_coordinate_reads"] != dimension * calls
            or info["recourse_rate_estimation_divisions"] > dimension * calls
            or info["recourse_planning_events"] != info["recourse_tagged_planning_events"] + info["recourse_background_planning_events"]
            or info["recourse_planning_events"] > samples * eligible * (cap + 1)
            or info["recourse_admitted_job_reads"] != info["recourse_flow_evaluations"] + info["recourse_background_admissions"]
            or info["recourse_tag_flag_reads"] != info["recourse_admitted_job_reads"]
            or info["recourse_admitted_job_reads"] > samples * m * original_completed * (limit + cap)
            or info["recourse_background_seed_initializations"] > samples * original_completed
            or info["recourse_background_planning_rate_reads"] != dimension * info["recourse_background_seed_initializations"]
               + info["recourse_background_planning_events"] - fallbacks["second_stage_overflows"]
            or any(info[key] for key in ("recourse_risk_comparisons", "recourse_ordering_comparisons",
                                         "recourse_strict_improvement_comparisons"))):
        return False
    original_scored = eligible - fallbacks["prefix_overflows"]
    if (work["incumbent_root_option_probes"] != m * original_scored
            or work["incumbent_rollout_option_probes"] != m * work["incumbent_rollout_calls"]
            or work["incumbent_rollout_calls"] > original_scored * bound
            or work["incumbent_background_seed_initializations"] > samples * original_scored
            or work["incumbent_background_planning_rate_reads"] != dimension * work["incumbent_background_seed_initializations"]
               + work["incumbent_background_planning_events"] - fallbacks["original_stage_overflows"]
            or work["incumbent_admitted_job_reads"] != work["incumbent_flow_evaluations"] + work["incumbent_background_admissions"]
            or work["incumbent_tag_flag_reads"] != work["incumbent_admitted_job_reads"]
            or work["incumbent_admitted_job_reads"] > samples * m * original_scored * (limit + cap)
            or info["recourse_flow_evaluations"] > work["incumbent_flow_evaluations"]
            or work["incumbent_strict_improvement_comparisons"] != original_completed
            or not original_completed <= work["incumbent_risk_comparisons"] <= m * original_completed
            or not work["incumbent_risk_comparisons"] - original_completed
               <= work["incumbent_ordering_comparisons"]
               <= work["incumbent_risk_comparisons"] + (m - 2) * original_completed
            or not new_completed <= work["incumbent_membership_checks"] <= m * new_completed
            or work["incumbent_lookahead_risk_comparisons"] != 0
            or work["incumbent_lookahead_ordering_comparisons"] != work["incumbent_membership_checks"]
               + info["recourse_risk_feasible_candidates"] - 2 * new_completed
            or work["incumbent_lookahead_strict_improvement_comparisons"] != new_completed):
        return False
    nested_calls, nested_eligible = work["recourse_calls"], work["recourse_eligible_calls"]
    nested_first, nested_second = work["recourse_first_calls"], work["recourse_second_calls"]
    nested_idle = work["recourse_virtual_idle_counter_updates"]
    nested_completed = nested_eligible - work["recourse_planning_overflows"]
    if (work["recourse_root_count_reads"] != dimension * calls
            or work["recourse_rng_words_copied"] != rng_words * original_completed
            or work["recourse_counter_reads"] != original_completed + nested_idle
            or work["recourse_rng_words_restored"] != rng_words * nested_first
            or work["recourse_counter_writes"] != nested_first + nested_idle
            or nested_calls != nested_first + nested_second
            or not 0 <= nested_second <= nested_first <= samples * m * original_completed
            or nested_idle > info["recourse_rollout_idle_calls"]
            or work["recourse_context_coordinate_reads"] != dimension * nested_calls
            or work["recourse_context_lane_reads"] != r * nested_calls
            or not 0 <= nested_eligible <= nested_calls <= 2 * samples * m * original_completed
            or work["recourse_changed_calls"] > nested_eligible
            or not 0 <= work["recourse_background_overflows"] <= work["recourse_planning_overflows"] <= nested_eligible
            or not m * nested_calls <= work["recourse_root_option_probes"] <= 2 * m * nested_calls
            or work["recourse_rollout_option_probes"] != m * work["recourse_rollout_calls"]
            or work["recourse_rollout_calls"] > nested_eligible * bound
            or work["recourse_rate_estimation_coordinate_reads"] != dimension * nested_calls
            or work["recourse_rate_estimation_divisions"] > dimension * nested_calls
            or work["recourse_planning_events"] != work["recourse_tagged_planning_events"] + work["recourse_background_planning_events"]
            or work["recourse_planning_events"] > samples * nested_eligible * (cap + 1)
            or work["recourse_admitted_job_reads"] != work["recourse_flow_evaluations"] + work["recourse_background_admissions"]
            or work["recourse_tag_flag_reads"] != work["recourse_admitted_job_reads"]
            or work["recourse_admitted_job_reads"] > samples * m * nested_eligible * (limit + cap)
            or work["recourse_background_seed_initializations"] > samples * nested_eligible
            or work["recourse_background_planning_rate_reads"] != dimension * work["recourse_background_seed_initializations"]
               + work["recourse_background_planning_events"] - work["recourse_background_overflows"]
            or not nested_completed <= work["recourse_risk_comparisons"] <= m * nested_completed
            or work["recourse_ordering_comparisons"] != work["recourse_risk_comparisons"]
               + work["recourse_risk_feasible_candidates"] - 2 * nested_completed
            or work["recourse_strict_improvement_comparisons"] != nested_completed):
        return False
    return True


def check_rollout(result, model, protocol, planner):
    m, r, dimension = len(model.options), model.lanes, len(model.workloads)
    samples, cap = protocol["planning_samples"], protocol["planning_event_cap"]
    limit, rng_words = r * planner.batch_width, len(planner.rng.getstate()[1])
    slack = r * (ceil(planner.maximum_duration) + 1)
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
        if not check_current_recourse(info, m, r, dimension, samples, cap,
                                      limit, slack, rng_words):
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
    for key in INNER_CURRENT_WORK:
        total["inner_current_" + key] = sum(
            frame["rollout"].get("inner_current_work", {}).get(key, 0) for frame in result["frames"])
    for key in INNER_CURRENT_FALLBACKS:
        total["inner_current_" + key] = sum(
            frame["rollout"].get("inner_current_fallbacks", {}).get(key, 0) for frame in result["frames"])
    total["recourse_controller_constructors"] = 2 * int(hasattr(planner, "inner_planner"))
    total["total_controller_constructors"] = total["controller_constructors"] + total["recourse_controller_constructors"]
    total["total_option_probes"] += sum(total["inner_current_" + key] for key in (
        "incumbent_root_option_probes", "incumbent_rollout_option_probes",
        "recourse_root_option_probes", "recourse_rollout_option_probes"))
    total["total_option_probes"] += sum(total.get("preprocessing_recourse_recourse_" + key, 0)
        for key in ("option_probes", "age_profile_option_probes", "rate_dimension_option_probes"))
    total["total_rollout_calls"] += (total["inner_current_incumbent_rollout_calls"]
                                    + total["inner_current_recourse_rollout_calls"])
    total["total_planning_events"] += (total["inner_current_incumbent_background_planning_events"]
                                      + total["inner_current_recourse_planning_events"])
    for target, keys in (
            ("risk_gate_comparisons", ("incumbent_risk_comparisons", "incumbent_lookahead_risk_comparisons", "recourse_risk_comparisons")),
            ("ordering_candidate_comparisons", ("incumbent_ordering_comparisons", "incumbent_lookahead_ordering_comparisons", "recourse_ordering_comparisons")),
            ("strict_improvement_comparisons", ("incumbent_strict_improvement_comparisons", "incumbent_lookahead_strict_improvement_comparisons", "recourse_strict_improvement_comparisons"))):
        total[target] += sum(total["inner_current_" + key] for key in keys)
    return total


def run_tape(model, tape, protocol, frozen_policies, frozen_checks,
             parent_policies, parent_checks):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {
        "maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"]),
        "arrival_rollout": TwoStepCurrentPolicyBGIncumbentLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"])}

    def select_candidate(*context):
        decision = planners["arrival_rollout"](*context)
        counter = planners["arrival_rollout"].decision_counter
        if counter % 20 == 0:
            print(tape["tape_id"], "candidate dispatch", counter,
                  "service time", context[3], flush=True)
        return decision

    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"],
        planners[label].maxweight if label == "maxweight" else select_candidate, policy=label)
        for label in LABELS}
    reference = {"maxweight": frozen_policies["arrival_rollout"], "arrival_rollout": policies["arrival_rollout"]}
    parent_reference = {"maxweight": parent_policies["arrival_rollout"],
                        "arrival_rollout": policies["arrival_rollout"]}
    class_snapshots = []
    for coordinate, name in enumerate(model.workloads):
        filtered = {label: {**result, "jobs": [job for job in result["jobs"]
                    if job["coordinate"] == coordinate]} for label, result in policies.items()}
        class_reference = {"maxweight": {**frozen_policies["arrival_rollout"], "jobs": [
            job for job in frozen_policies["arrival_rollout"]["jobs"] if job["coordinate"] == coordinate]},
            "arrival_rollout": filtered["arrival_rollout"]}
        class_parent = {"maxweight": {**parent_policies["arrival_rollout"], "jobs": [
            job for job in parent_policies["arrival_rollout"]["jobs"] if job["coordinate"] == coordinate]},
            "arrival_rollout": filtered["arrival_rollout"]}
        class_snapshots.append({"class": name, "coordinate": coordinate,
            "snapshots": {key: cohort_metrics(filtered, protocol[key], protocol["birth_cohort"])
                          for key in ("arrival_horizon", "observation_end")},
            "reference_snapshots": {key: cohort_metrics(class_reference, protocol[key], protocol["birth_cohort"])
                                    for key in ("arrival_horizon", "observation_end")},
            "parent_snapshots": {key: cohort_metrics(class_parent, protocol[key], protocol["birth_cohort"])
                                 for key in ("arrival_horizon", "observation_end")}})
    row = {
        **{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": replay_matches(policies["maxweight"], frozen_policies["maxweight"]),
        "rollout_checks_passed": check_rollout(policies["arrival_rollout"], model, protocol, planners["arrival_rollout"]),
        "cached_reference_checks_passed": frozen_checks["baseline_reproduction_passed"] and frozen_checks["rollout_checks_passed"],
        "cached_parent_checks_passed": parent_checks["baseline_reproduction_passed"]
            and parent_checks["rollout_checks_passed"],
        "snapshots": {key: cohort_metrics(policies, protocol[key], protocol["birth_cohort"])
                      for key in ("arrival_horizon", "observation_end")},
        "reference_snapshots": {key: cohort_metrics(reference, protocol[key], protocol["birth_cohort"])
                                for key in ("arrival_horizon", "observation_end")},
        "parent_snapshots": {key: cohort_metrics(parent_reference, protocol[key], protocol["birth_cohort"])
                             for key in ("arrival_horizon", "observation_end")},
        "class_snapshots": class_snapshots,
        "current_continuation_diagnostics": {
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
        "reference_middle_queue_area_s": queue_area(frozen_policies["arrival_rollout"]),
        "parent_middle_queue_area_s": queue_area(parent_policies["arrival_rollout"])}
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
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != json.loads(
            json.dumps(expected, allow_nan=False)):
        raise ValueError("registered sources, budgets, comparisons or literal tapes changed")
    if BASE.with_suffix(".json").exists() or path(BASE, "_trajectories.json.gz").exists():
        raise FileExistsError("preserve existing results")
    previous = json.loads(PREVIOUS.with_suffix(".json").read_text())
    frozen_checks = {row["tape_id"]: row for row in previous["paired_runs"]}
    with gzip.open(ROOT / protocol["previous_trajectories"], "rt") as source:
        frozen = {run["tape_id"]: run["policies"] for run in json.load(source)["runs"]}
    parent_result = json.loads((ROOT / protocol["continuation_parent_result"]).read_text())
    parent_checks = {row["tape_id"]: row for row in parent_result["paired_runs"]}
    with gzip.open(ROOT / protocol["continuation_parent_trajectories"], "rt") as source:
        parent = {run["tape_id"]: run["policies"] for run in json.load(source)["runs"]}
    models = {model.node: model for model in load_models(protocol["models"])}
    completed = {}
    with ProcessPoolExecutor(max_workers=8) as workers:
        futures = {workers.submit(run_tape, models[tape["node"]], tape, protocol,
                    frozen[tape["tape_id"]], frozen_checks[tape["tape_id"]],
                    parent[tape["tape_id"]], parent_checks[tape["tape_id"]]): tape["tape_id"]
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
                                   "arrival_rollout": "two_current_policy_BG_incumbent_candidate"},
        cached_reference_reused=True, reference_rerun=False,
        baseline_check_scope="Frozen BG passing checks plus newly executed native MW identity; no new BG reproduction.",
        work_scope="Candidate work repeats the same executed candidate accounting; cached baseline has no new execution work.")
    parent_rows = [{**row, "snapshots": row["parent_snapshots"],
        "baseline_reproduction_passed": row["baseline_reproduction_passed"] and row["cached_parent_checks_passed"],
        "work": {"maxweight": {}, "arrival_rollout": row["work"]["arrival_rollout"]},
        "middle_queue_area_s": {"maxweight": row["parent_middle_queue_area_s"],
                                "arrival_rollout": row["middle_queue_area_s"]["arrival_rollout"]}}
        for row in rows]
    parent_comparison = summarize(parent_rows, protocol)
    parent_comparison.update(
        policy_roles={"maxweight": "cached_frozen_original_risk_current_parent",
                      "arrival_rollout": "two_current_policy_BG_incumbent_candidate"},
        cached_reference_reused=True, reference_rerun=False, performance_gate_applied=False,
        baseline_check_scope="Frozen parent execution checks plus newly executed native MW identity; no parent rerun.",
        work_scope="Same executed candidate work repeated for comparison; cached parent has no new work.")
    executed = (result["execution_checks_passed"] and reference["execution_checks_passed"]
                and parent_comparison["execution_checks_passed"])
    passed = result["performance_gain_established"] and reference["performance_gain_established"]
    result["performance_vs_maxweight_established"] = result["performance_gain_established"]
    result["performance_gain_established"], result["execution_checks_passed"] = passed, executed
    result["decision"] = "TWO_STEP_CURRENT_POLICY_BG_INCUMBENT_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD")
    result["reference_comparison"] = reference
    result["parent_comparison"] = parent_comparison
    result["current_continuation_diagnostics"] = {
        key: sum(row["current_continuation_diagnostics"][key] for row in rows)
        for key in rows[0]["current_continuation_diagnostics"]}
    result["continuation_parent_result"] = protocol["continuation_parent_result"]
    result["birth_cohort"], result["cutoff_semantics"] = protocol["birth_cohort"], protocol["cutoff_semantics"]
    result["total_executed_trajectories"], result["reused_reference_trajectories"] = 2 * len(rows), len(rows)
    result["historical_reference_work_location"] = protocol["previous_result"]
    result["middle_queue_area_s"] = {label: sum(row["middle_queue_area_s"][label] for row in rows) for label in LABELS}
    result["reference_middle_queue_area_s"] = sum(row["reference_middle_queue_area_s"] for row in rows)
    result["parent_middle_queue_area_s"] = sum(row["parent_middle_queue_area_s"] for row in rows)
    result["reused_parent_trajectories"] = len(rows)
    with gzip.open(path(BASE, "_trajectories.json.gz"), "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "runs": runs}, out, separators=(",", ":"), allow_nan=False)
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "pooled": result["pooled"],
        "reference_flow_improvement": reference["pooled_T_flow_improvement"],
        "reference_failed_secondary_groups": reference["failed_secondary_groups"],
        "parent_flow_improvement": parent_comparison["pooled_T_flow_improvement"]}), flush=True)


if __name__ == "__main__":
    main()
