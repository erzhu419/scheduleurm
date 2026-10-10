"""Offline paired effects of replacing one committed successor by same-state MW."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.deferred_incumbent_background_lane_rollout import DeferredIncumbentBackgroundLaneRollout
from algorithm.experiments.aligned_hardware_background_competition_continuous import path
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_deferred_incumbent_background_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_single_successor_counterfactual_20260930"
SOURCE = "algorithm/experiments/aligned_hardware_single_successor_counterfactual.py"
HELPER = "algorithm/experiments/single_action_counterfactual_replay.py"
FEES = (
    "root_option_probes", "rollout_calls", "rollout_option_probes", "planning_events",
    "planning_random_draws", "planning_rate_reads", "rollout_idle_calls", "waiting_age_reads",
    "flow_evaluations", "rate_estimation_coordinate_reads", "rate_estimation_divisions",
    "tagged_planning_events", "background_planning_events", "background_planning_random_draws",
    "background_planning_rate_reads", "background_seed_initializations", "admitted_job_reads",
    "tag_flag_reads", "background_admissions", "successor_feasibility_coordinate_reads",
    "successor_envelope_checks", "successor_feasibility_checks", "successor_override_calls",
)
STATUS = (
    "eligible", "changed", "plan_changed", "planning_overflow", "background_rollout_overflow",
    "successor_call", "successor_executed", "successor_changed", "risk_changed_unconstrained_root",
    "successor_envelope_passed", "successor_feasibility_passed",
    "candidate_count", "plan_candidate_count", "risk_filtered_candidates", "risk_feasible_candidates",
)
WEAK_WORKLOADS = ("gpu_cnn_torch_resnet50", "gpu_llm_distilgpt2")


def saved_runs():
    with gzip.open(path(PREVIOUS, "_trajectories.json.gz"), "rt") as source:
        return {run["tape_id"]: run["policies"]["arrival_rollout"]
                for run in json.load(source)["runs"]}


def registered_targets(tapes, factual):
    return [{"tape_id": tape["tape_id"], "frame_index": index, "start": frame["start"],
             "factual_option_index": frame["option_indices"][0]}
            for tape in tapes for index, frame in enumerate(factual[tape["tape_id"]]["frames"])
            if frame["rollout"]["successor_changed"] and 300 <= frame["start"] < 900]


def build_protocol():
    prior = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    result = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if (not result["execution_checks_passed"]
            or result["registered_at_utc"] != prior["registered_at_utc"]):
        raise ValueError("matching complete deferred-incumbent development required")
    for name, source in prior["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    targets = registered_targets(prior["tapes"], saved_runs())
    active = [tape["tape_id"] for tape in prior["tapes"]
              if any(target["tape_id"] == tape["tape_id"] for target in targets)]
    return {
        "experiment": "single_successor_closed_loop_counterfactual_v1",
        "previous_protocol": str(path(PREVIOUS, "_protocol.json").relative_to(ROOT)),
        "previous_result": str(PREVIOUS.with_suffix(".json").relative_to(ROOT)),
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        "previous_registered_at_utc": prior["registered_at_utc"], "previous_decision": result["decision"],
        **{key: prior[key] for key in (
            "models", "tapes", "arrival_seeds", "birth_cohort", "arrival_horizon", "observation_end",
            "environment_arrival_end", "planning_samples", "planning_event_cap", "planning_seed_rule",
            "clock", "cutoff_semantics", "budget")},
        "targets": targets, "active_tape_ids": active,
        "weak_workloads": list(WEAK_WORKLOADS),
        "method": "Register every stored factual successor_changed action in[300,900), without filtering by counterfactual outcome. Replay each target-bearing tape once, capturing pre-call prefix RNG, decision_counter and pending_successor_index. Restore all three and the original physical/FIFO/count checkpoint. Consume the actual proposal normally, replace only the first suffix action with same-state MW, then continue the unchanged deferred-incumbent policy to1200 on identical actual arrivals. One first-target null suffix per active tape must exactly reproduce factual jobs/actions/clock. Zero-target tapes remain registered but are not re-executed.",
        "accounting": "One full factual replay and one null suffix per active tape, plus one forced suffix per target. Charge every proposal's root/rollout/planning/background/tag/admission/rate/guard/override operations and the extra MW scan; constructor, engine context, checkpoint and RNG/scalar capture/restore work are separate. Proposal status counts and actually applied successor status counts are separate from operation counters; never add engine selector counters again.",
        "outcomes": "Replacement minus factual: all-cohort restricted mean and common-completed-ID P90 at1200; unfinished counts at900 and1200. Keep all three class restricted means, separately completed-population P90s and both tails. Global/class positive, zero and negative effects remain recorded. CNN/LLM help counts are descriptive; interventions are dependent and their effects cannot be added.",
        "scope": "Offline conditional causal mechanism diagnostic on reused development seeds41..43; not a deployable future-aware rule, independent-seed confirmation or additive population-effect estimate. Real future births remain solely in the harness.",
        "execution_sources": {**prior["execution_sources"],
                              **{name: (ROOT / name).read_text() for name in (HELPER, SOURCE)}},
    }


def make_planner(model, tape, protocol):
    return DeferredIncumbentBackgroundLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])


def restore_planner_state(planner, state):
    planner.rng.setstate(state["rng_state"])
    planner.decision_counter = state["decision_counter"]
    planner.pending_successor_index = state["pending_successor_index"]
    return {"checkpoint_rng_words_restored": len(state["rng_state"][1]),
            "checkpoint_planner_state_fields_restored": 2}


class SuccessorCounterfactualSelector:
    def __init__(self, planner, capture=(), force_first=False):
        self.planner, self.capture, self.force_first = planner, set(capture), force_first
        self.calls, self.states, self.work, self.status, self.intervention = 0, {}, Counter(), Counter(), None

    def __call__(self, *context):
        if self.calls in self.capture:
            state = {"rng_state": self.planner.rng.getstate(),
                     "decision_counter": self.planner.decision_counter,
                     "pending_successor_index": self.planner.pending_successor_index}
            self.states[self.calls] = state
            self.work["checkpoint_rng_words_copied"] += len(state["rng_state"][1])
            self.work["checkpoint_planner_state_fields_copied"] += 2
        proposed = self.planner(*context)
        self.work.update({key: getattr(proposed, key) for key in FEES})
        self.status["proposal_calls"] += 1
        self.status.update({"proposal_" + key: getattr(proposed, key) or 0 for key in STATUS})
        selected = proposed
        if self.calls == 0 and self.force_first:
            control = self.planner.maxweight(*context)
            self.work["root_option_probes"] += control.root_option_probes
            self.work["override_calls"] += 1
            original, replacement = proposed.selection.option_indices[0], control.selection.option_indices[0]
            if not proposed.successor_changed or not proposed.successor_call or original == replacement:
                raise ValueError("forced target must be an actual non-MW committed successor")
            self.intervention = {"original": original, "replacement": replacement,
                                 "proposed_successor_index": proposed.planned_successor_index,
                                 "actual_applied_option_index": replacement}
            selected = control
        self.status["applied_successor_executed"] += selected.successor_executed
        self.status["applied_successor_changed"] += selected.successor_changed
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


def paired_outcomes(factual, replacement, protocol):
    snapshots = {key: cohort_metrics({"factual": factual, "replacement": replacement},
                 protocol[key], protocol["birth_cohort"])
                 for key in ("arrival_horizon", "observation_end")}
    def delta(cutoff, metric, coordinate=None):
        groups = snapshots[cutoff]
        values = [groups[label] if coordinate is None else groups[label]["per_coordinate"][coordinate]
                  for label in ("factual", "replacement")]
        return (values[1][metric] - values[0][metric]
                if all(value[metric] is not None for value in values) else None)
    return {"snapshots": snapshots,
        "delta_T_mean": delta("observation_end", "restricted_mean_flow_s"),
        "delta_T_p90": delta("observation_end", "common_p90_flow_s"),
        "delta_900_unfinished": delta("arrival_horizon", "unfinished_jobs"),
        "delta_1200_unfinished": delta("observation_end", "unfinished_jobs"),
        "per_coordinate_deltas": [{"coordinate": coordinate,
            "delta_T_mean": delta("observation_end", "restricted_mean_flow_s", coordinate),
            "delta_T_completed_p90": delta("observation_end", "completed_p90_flow_s", coordinate),
            "delta_900_unfinished": delta("arrival_horizon", "unfinished_jobs", coordinate),
            "delta_1200_unfinished": delta("observation_end", "unfinished_jobs", coordinate)}
            for coordinate in range(len(snapshots["observation_end"]["factual"]["per_coordinate"]))]}


def effect_counts(records):
    metrics = ("delta_T_mean", "delta_T_p90", "delta_900_unfinished", "delta_1200_unfinished")
    signs = {metric: {name: sum(record[metric] is not None and compare(record[metric]) for record in records)
             for name, compare in (("helped", lambda value: value < 0),
                                   ("unchanged", lambda value: value == 0),
                                   ("worsened", lambda value: value > 0))} for metric in metrics}
    weak = {workload: sum(next(item["delta_T_mean"] for item in record["per_coordinate_deltas"]
            if record["workloads"][item["coordinate"]] == workload) < 0 for record in records)
            for workload in WEAK_WORKLOADS}
    return {"targets": len(records), "effects": signs, "weak_class_mean_helped_by_MW": weak}


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
        print("registered", len(expected["targets"]), "targets on", len(expected["active_tape_ids"]), "tapes", flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources, checkpoints, targets or literal tapes changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve existing diagnostic outcome")
    factual_runs = saved_runs()
    models = {model.node: model for model in load_models(protocol["models"])}
    records, controls, work, status, groups = [], [], Counter(), Counter(), defaultdict(list)
    for tape in protocol["tapes"]:
        targets = [target for target in protocol["targets"] if target["tape_id"] == tape["tape_id"]]
        if not targets:
            continue
        model = models[tape["node"]]
        capture = SuccessorCounterfactualSelector(make_planner(model, tape, protocol),
                                                  capture=[target["frame_index"] for target in targets])
        factual = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
            protocol["environment_arrival_end"], protocol["observation_end"], capture, policy="arrival_rollout")
        if not replay_matches(factual, factual_runs[tape["tape_id"]]):
            raise ValueError("factual actions/jobs/clock reproduction failed")
        work.update(collect_work(capture, factual))
        status.update(capture.status)
        for number, target in enumerate(targets):
            index = target["frame_index"]
            checkpoint = prepare_checkpoint(factual, index)
            work.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
            state = capture.states[index]
            if number == 0:
                null = SuccessorCounterfactualSelector(make_planner(model, tape, protocol))
                work.update(restore_planner_state(null.planner, state))
                resumed = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
                    protocol["observation_end"], null, checkpoint, policy="null")
                combined = {**resumed, "frames": factual["frames"][:index] + resumed["frames"]}
                if not replay_matches(combined, factual_runs[tape["tape_id"]]):
                    raise ValueError("null suffix must reproduce RNG/counter/pending and actual clock/jobs/actions")
                controls.append({"tape_id": tape["tape_id"], "frame_index": index, "null_reproduction_passed": True})
                work.update(collect_work(null, resumed))
                status.update(null.status)
            branch = SuccessorCounterfactualSelector(make_planner(model, tape, protocol), force_first=True)
            work.update(restore_planner_state(branch.planner, state))
            replacement = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
                protocol["observation_end"], branch, checkpoint, policy="replacement")
            if branch.intervention["original"] != target["factual_option_index"]:
                raise ValueError("restored proposal differs from the factual target")
            work.update(collect_work(branch, replacement))
            status.update(branch.status)
            record = {**target, **branch.intervention, **paired_outcomes(factual, replacement, protocol),
                "node": tape["node"], "load": tape["load"], "seed": tape["seed"], "workloads": list(model.workloads),
                "original_action": model.options[branch.intervention["original"]].action_id,
                "replacement_action": model.options[branch.intervention["replacement"]].action_id,
                "planner_state": {"decision_counter": state["decision_counter"],
                                  "pending_successor_index": state["pending_successor_index"]},
                "context": {key: factual["frames"][index][key] for key in (
                    "queue", "waiting_ages", "available_before", "observed_arrival_counts", "lane_ids")}}
            records.append(record)
            groups[tape["node"], tape["load"]].append(record)
            print(tape["tape_id"], index, "paired successor substitution complete", flush=True)
    if len(records) != len(protocol["targets"]) or len(controls) != len(protocol["active_tape_ids"]):
        raise ValueError("all registered targets and active-tape null controls required")
    result = {"decision": "SINGLE_SUCCESSOR_DIAGNOSTIC_COMPLETE",
        "registered_at_utc": protocol["registered_at_utc"], "scope": protocol["scope"],
        "previous_decision": protocol["previous_decision"], "registered_tapes": len(protocol["tapes"]),
        "factual_replays": len(controls), "null_controls": controls, "interventions": len(records),
        "total_executed_trajectories": 2 * len(controls) + len(records),
        "summary": effect_counts(records),
        "by_node_load": [{"node": node, "load": load, **effect_counts(samples)}
                         for (node, load), samples in sorted(groups.items())],
        "records": records, "actual_work": dict(work), "proposal_and_applied_status_counts": dict(status)}
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "summary": result["summary"],
                      "total_executed_trajectories": result["total_executed_trajectories"]}), flush=True)


if __name__ == "__main__":
    main()
