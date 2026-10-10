"""Offline first-root swaps under BG versus ordinary-MW continuation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_background_competition_continuous import path
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_background_competition_continuous_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_background_root_counterfactual_20260930"
SOURCE = "algorithm/experiments/aligned_hardware_background_root_counterfactual.py"
HELPER = "algorithm/experiments/single_action_counterfactual_replay.py"
FEES = (
    "root_option_probes", "rollout_calls", "rollout_option_probes", "planning_events",
    "planning_random_draws", "planning_rate_reads", "rollout_idle_calls", "waiting_age_reads",
    "flow_evaluations", "rate_estimation_coordinate_reads", "rate_estimation_divisions",
    "tagged_planning_events", "background_planning_events", "background_planning_random_draws",
    "background_planning_rate_reads", "background_seed_initializations", "admitted_job_reads",
    "tag_flag_reads", "background_admissions",
)
STATUS = ("eligible", "changed", "planning_overflow", "background_rollout_overflow",
          "risk_changed_unconstrained_root", "candidate_count", "risk_filtered_candidates",
          "risk_feasible_candidates")


def saved_runs():
    with gzip.open(path(PREVIOUS, "_trajectories.json.gz"), "rt") as source:
        return {run["tape_id"]: run["policies"]["arrival_rollout"]
                for run in json.load(source)["runs"]}


def registered_targets(tapes, factual):
    targets = []
    for tape in tapes:
        for index, frame in enumerate(factual[tape["tape_id"]]["frames"]):
            if frame["rollout"]["changed"] and 300 <= frame["start"] < 900:
                targets.append({"tape_id": tape["tape_id"], "frame_index": index,
                                "start": frame["start"], "factual_option_index": frame["option_indices"][0]})
                break
    return targets


def build_protocol():
    prior = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    result = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if (result["decision"] != "BACKGROUND_COMPETITION_CONTINUOUS_DEVELOPMENT_PASS"
            or not result["execution_checks_passed"] or not result["performance_gain_established"]
            or result["registered_at_utc"] != prior["registered_at_utc"]):
        raise ValueError("matching frozen BG development PASS required")
    for name, source in prior["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    checks = {row["tape_id"]: {key: row[key] for key in (
        "baseline_reproduction_passed", "rollout_checks_passed")} for row in result["paired_runs"]}
    if any(not all(checks[tape["tape_id"]].values()) for tape in prior["tapes"]):
        raise ValueError("every cached BG tape must retain its passing execution checks")
    targets = registered_targets(prior["tapes"], saved_runs())
    return {
        "experiment": "background_root_continuation_counterfactual_v1",
        "previous_protocol": str(path(PREVIOUS, "_protocol.json").relative_to(ROOT)),
        "previous_result": str(PREVIOUS.with_suffix(".json").relative_to(ROOT)),
        "previous_registered_at_utc": prior["registered_at_utc"], "previous_decision": result["decision"],
        "cached_execution_checks": checks,
        **{key: prior[key] for key in (
            "models", "tapes", "arrival_seeds", "birth_cohort", "arrival_horizon", "observation_end",
            "environment_arrival_end", "planning_samples", "planning_event_cap", "planning_seed_rule",
            "clock", "cutoff_semantics", "budget")},
        "targets": targets, "active_tape_ids": [target["tape_id"] for target in targets],
        "expected_executed_trajectories": 5 * len(targets),
        "method": "Register the first factual BG changed root in[300,900) on each of the24 development tapes, with no outcome filtering. Execute only target-bearing tapes: one full factual BG replay, one null BG suffix, MW-first/BG-suffix, original-BG-first/MW-suffix and MW-first/MW-suffix. All suffixes restore the same actual FIFO/jobs/busy lanes/cumulative-count checkpoint and pre-call prefix RNG plus decision_counter. Every first suffix call consumes the original BG proposal; a forced first action adds same-state MW. MW continuation later uses only plain MW and never advances planner RNG/counter artificially.",
        "accounting": "Five actual controllers per active tape; zero-target tapes remain registered and are not rerun. Charge each actual BG proposal's complete planning/background/tag/admission/rate/root/continuation and risk work, each extra MW scan, constructor/context/checkpoint/RNG/counter capture/restore work. Proposal status counts remain separate. Engine selector costs are not added again. Save only the paired outcome JSON, not large suffix trajectories.",
        "outcomes": "Replacement minus original root under each continuation: birth-cohort[300,900] restricted mean at1200, same-pair completed-ID common P90,900/1200 unfinished counts and all-job waiting area[300,900]. For each of three classes preserve restricted mean, same-class common-completed-ID mean/P90 and both tails. Retain all effect signs and mean-ranking crosses without adding intervention effects.",
        "scope": "Offline conditional causal root-by-continuation diagnostic on reused development seeds41..43. Real future births stay in the harness. No deployable oracle, independent confirmation, additive population-effect estimate or exact forecast-error decomposition is claimed.",
        "execution_sources": {**prior["execution_sources"],
                              **{name: (ROOT / name).read_text() for name in (HELPER, SOURCE)}},
    }


def make_planner(model, tape, protocol):
    return BackgroundCompetitionContinuousLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])


def restore_planner_state(planner, state):
    planner.rng.setstate(state["rng_state"])
    planner.decision_counter = state["decision_counter"]
    return {"checkpoint_rng_words_restored": len(state["rng_state"][1]),
            "checkpoint_decision_counter_writes": 1}


class RootCounterfactualSelector:
    def __init__(self, planner, capture=(), force_first=False, continuation="BG"):
        self.planner, self.capture = planner, set(capture)
        self.force_first, self.continuation = force_first, continuation
        self.calls, self.states, self.work, self.status = 0, {}, Counter(), Counter()
        self.first_proposal = self.intervention = None

    def __call__(self, *context):
        if self.calls in self.capture:
            state = {"rng_state": self.planner.rng.getstate(),
                     "decision_counter": self.planner.decision_counter}
            self.states[self.calls] = state
            self.work["checkpoint_rng_words_copied"] += len(state["rng_state"][1])
            self.work["checkpoint_decision_counter_reads"] += 1
        if self.calls and self.continuation == "MW":
            selected = self.planner.maxweight(*context)
            self.work["root_option_probes"] += selected.root_option_probes
            self.status["MW_continuation_calls"] += 1
        else:
            proposed = self.planner(*context)
            self.work.update({key: getattr(proposed, key) for key in FEES})
            self.status["BG_proposal_calls"] += 1
            self.status.update({"proposal_" + key: getattr(proposed, key) for key in STATUS})
            if proposed.eligible and not proposed.planning_overflow:
                self.work["risk_gate_comparisons"] += proposed.candidate_count
                self.work["ordering_candidate_comparisons"] += (
                    proposed.candidate_count - 1 + proposed.risk_feasible_candidates - 1)
                self.work["strict_improvement_comparisons"] += 1
            selected = proposed
            if self.calls == 0:
                before, after = proposed.baseline_predicted_completion_s, proposed.chosen_predicted_completion_s
                self.first_proposal = {"original": proposed.selection.option_indices[0],
                    "changed": proposed.changed,
                    "predicted_S1_advantage": before - after if before is not None and after is not None else None}
                if self.force_first:
                    control = self.planner.maxweight(*context)
                    self.work["root_option_probes"] += control.root_option_probes
                    self.work["override_calls"] += 1
                    replacement = control.selection.option_indices[0]
                    if not proposed.changed or replacement == self.first_proposal["original"]:
                        raise ValueError("forced first root must be a changed factual BG action")
                    self.intervention = {**self.first_proposal, "replacement": replacement,
                                         "actual_applied_option_index": replacement}
                    selected = control
        self.calls += 1
        return selected


def collect_work(selector, result):
    work = Counter(selector.work)
    work["controller_constructors"] += 1
    work.update({"preprocessing_" + key: value for key, value in selector.planner.preprocessing_work.items()})
    for key in ("age_context_job_reads", "observed_arrival_counter_updates",
                "observed_arrival_context_coordinate_reads"):
        work[key] += result["rollout_work"][key]
    return work


def paired_outcomes(factual, replacement, protocol, workloads):
    policies = {"factual": factual, "replacement": replacement}
    lower, upper = protocol["birth_cohort"]
    cohort = {label: [job for job in result["jobs"] if lower <= job["arrival"] <= upper]
              for label, result in policies.items()}
    snapshots = {}
    for key in ("arrival_horizon", "observation_end"):
        cutoff = protocol[key]
        snapshots[key] = cohort_metrics(policies, cutoff, protocol["birth_cohort"])
        done = {label: {(job["coordinate"], job["index"]): job for job in jobs
                       if job["completion"] is not None and job["completion"] <= cutoff}
                for label, jobs in cohort.items()}
        common = done["factual"].keys() & done["replacement"].keys()
        for coordinate in range(len(workloads)):
            class_ids = sorted(identity for identity in common if identity[0] == coordinate)
            for label in policies:
                group = trajectory_snapshot([done[label][identity] for identity in class_ids], cutoff, len(workloads))
                snapshots[key][label]["per_coordinate"][coordinate].update(
                    common_class_jobs=len(class_ids), common_class_mean_flow_s=group["completed_mean_flow_s"],
                    common_class_p90_flow_s=group["completed_p90_flow_s"])
    def delta(cutoff, metric, coordinate=None):
        groups = snapshots[cutoff]
        values = [groups[label] if coordinate is None else groups[label]["per_coordinate"][coordinate]
                  for label in ("factual", "replacement")]
        return (values[1][metric] - values[0][metric]
                if all(value[metric] is not None for value in values) else None)
    areas = {label: queue_area(result) for label, result in policies.items()}
    return {"snapshots": snapshots, "middle_queue_area_s": areas,
        "delta_T_mean": delta("observation_end", "restricted_mean_flow_s"),
        "delta_T_common_p90": delta("observation_end", "common_p90_flow_s"),
        "delta_900_unfinished": delta("arrival_horizon", "unfinished_jobs"),
        "delta_1200_unfinished": delta("observation_end", "unfinished_jobs"),
        "delta_middle_queue_area_s": areas["replacement"] - areas["factual"],
        "per_coordinate_deltas": [{"coordinate": coordinate, "workload": workload,
            "delta_T_mean": delta("observation_end", "restricted_mean_flow_s", coordinate),
            "delta_T_common_mean": delta("observation_end", "common_class_mean_flow_s", coordinate),
            "delta_T_common_p90": delta("observation_end", "common_class_p90_flow_s", coordinate),
            "delta_900_unfinished": delta("arrival_horizon", "unfinished_jobs", coordinate),
            "delta_1200_unfinished": delta("observation_end", "unfinished_jobs", coordinate)}
            for coordinate, workload in enumerate(workloads)]}


def sign_counts(values):
    return {"helped": sum(value is not None and value < 0 for value in values),
            "unchanged": sum(value == 0 for value in values),
            "worsened": sum(value is not None and value > 0 for value in values),
            "unavailable": sum(value is None for value in values)}


def summarize_effects(records):
    metrics = ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished",
               "delta_1200_unfinished", "delta_middle_queue_area_s")
    results = {mode: {metric: sign_counts([record[mode][metric] for record in records]) for metric in metrics}
               for mode in ("BG_continuation", "MW_continuation")}
    results["mean_ranking_flips"] = sum(record["mean_ranking_flipped"] for record in records)
    results["mean_crosses"] = dict(Counter(
        ("BG_help_MW_loss" if bg < 0 < mw else "BG_loss_MW_help" if mw < 0 < bg else
         "both_help" if bg < 0 and mw < 0 else "both_loss" if bg > 0 and mw > 0 else "contains_zero")
        for record in records for bg, mw in [(record["BG_continuation"]["delta_T_mean"],
                                            record["MW_continuation"]["delta_T_mean"])]))
    return results


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
        print("registered", len(expected["targets"]), "first roots; executions", expected["expected_executed_trajectories"], flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources, cached checks, targets or tapes changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve existing diagnostic outcome")
    frozen = saved_runs()
    models = {model.node: model for model in load_models(protocol["models"])}
    records, controls, work, status = [], [], Counter(), Counter()
    targets = {target["tape_id"]: target for target in protocol["targets"]}
    for tape in protocol["tapes"]:
        if tape["tape_id"] not in targets:
            continue
        target, model = targets[tape["tape_id"]], models[tape["node"]]
        index = target["frame_index"]
        capture = RootCounterfactualSelector(make_planner(model, tape, protocol), capture=(index,))
        factual = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
            protocol["environment_arrival_end"], protocol["observation_end"], capture, policy="arrival_rollout")
        if not replay_matches(factual, frozen[tape["tape_id"]]):
            raise ValueError("full factual BG jobs/actions/clock differ from cache")
        work.update(collect_work(capture, factual))
        status.update(capture.status)
        checkpoint = prepare_checkpoint(factual, index)
        work.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
        arms, selectors = {}, {}
        for name, forced, continuation in (("null", False, "BG"), ("replacement_BG", True, "BG"),
                                            ("original_MW", False, "MW"), ("replacement_MW", True, "MW")):
            selector = RootCounterfactualSelector(make_planner(model, tape, protocol),
                                                  force_first=forced, continuation=continuation)
            work.update(restore_planner_state(selector.planner, capture.states[index]))
            result = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
                protocol["observation_end"], selector, checkpoint, policy=name)
            if selector.first_proposal["original"] != target["factual_option_index"]:
                raise ValueError("restored first proposal differs from its registered root")
            work.update(collect_work(selector, result))
            status.update(selector.status)
            if name == "null":
                combined = {**result, "frames": factual["frames"][:index] + result["frames"]}
                if not replay_matches(combined, frozen[tape["tape_id"]]):
                    raise ValueError("null BG RNG/counter/jobs/actions/clock reproduction failed")
                controls.append({"tape_id": tape["tape_id"], "null_reproduction_passed": True})
            arms[name], selectors[name] = result, selector
        if selectors["replacement_BG"].intervention["replacement"] != selectors["replacement_MW"].intervention["replacement"]:
            raise ValueError("the same-state replacement root must match across continuations")
        bg = paired_outcomes(factual, arms["replacement_BG"], protocol, model.workloads)
        mw = paired_outcomes(arms["original_MW"], arms["replacement_MW"], protocol, model.workloads)
        intervention = selectors["replacement_BG"].intervention
        records.append({**target, **intervention, "node": tape["node"], "load": tape["load"], "seed": tape["seed"],
            "original_action": model.options[intervention["original"]].action_id,
            "replacement_action": model.options[intervention["replacement"]].action_id,
            "decision_counter": capture.states[index]["decision_counter"],
            "context": {key: factual["frames"][index][key] for key in (
                "queue", "waiting_ages", "available_before", "observed_arrival_counts", "lane_ids")},
            "BG_continuation": bg, "MW_continuation": mw,
            "mean_ranking_flipped": bg["delta_T_mean"] * mw["delta_T_mean"] < 0})
        print(tape["tape_id"], "five controllers complete", flush=True)
    if len(records) != len(protocol["targets"]) or len(controls) != len(records):
        raise ValueError("every registered first target and null control required")
    result = {"decision": "BACKGROUND_ROOT_COUNTERFACTUAL_COMPLETE",
        "registered_at_utc": protocol["registered_at_utc"], "scope": protocol["scope"],
        "registered_tapes": len(protocol["tapes"]), "targets": len(records),
        "factual_replays": len(records), "null_controls": controls,
        "total_executed_trajectories": 5 * len(records), "summary": summarize_effects(records),
        "records": records, "actual_work": dict(work), "status_counts": dict(status)}
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "targets", "total_executed_trajectories", "summary")}), flush=True)


if __name__ == "__main__":
    main()
