"""Offline paired closed-loop effects of one MW substitution; never an online oracle."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_continuous_arrival_age_tail_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_single_action_counterfactual_20260930"
SOURCES = ("algorithm/experiments/single_action_counterfactual_replay.py",
           "algorithm/experiments/aligned_hardware_single_action_counterfactual.py")
FEES = ("root_option_probes", "rollout_option_probes", "rollout_calls", "planning_events",
        "planning_random_draws", "planning_rate_reads", "flow_evaluations", "waiting_age_reads")


def path(prefix, suffix):
    return prefix.with_name(prefix.name + suffix)


def saved_runs():
    with gzip.open(path(PREVIOUS, "_trajectories.json.gz"), "rt") as source:
        return {run["tape_id"]: run["policies"]["arrival_rollout"] for run in json.load(source)["runs"]}


def build_protocol():
    prior = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    for name, source in prior["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    factual = saved_runs()
    tapes = [t for t in prior["tapes"] if t["node"] == "gpu_a" and t["load"] == .8]
    targets = [{"tape_id": tape["tape_id"], "frame_index": i, "start": frame["start"],
                "factual_option_index": frame["option_indices"][0]}
               for tape in tapes for i, frame in enumerate(factual[tape["tape_id"]]["frames"])
               if frame["rollout"]["changed"] and 300 <= frame["start"] < 900]
    return {"experiment": "single_action_closed_loop_diagnostic_v1", "models": prior["models"],
        "tapes": tapes, "targets": targets, "birth_cohort": prior["birth_cohort"],
        "arrival_horizon": prior["arrival_horizon"], "observation_end": prior["observation_end"],
        "environment_arrival_end": prior["environment_arrival_end"],
        "planning_samples": prior["planning_samples"], "planning_event_cap": prior["planning_event_cap"],
        "method": "All factual changed roots in[300,900) on the three gpu_a .8 development tapes. Replay each factual policy once and capture pre-target planner RNG. Fork actual pre-action FIFO/observed counts/busy reservations, consume the same original proposal draws, replace exactly the first suffix action by same-state ordinary MW, then continue the unchanged original aggregate-S2 policy until1200. Actual arrivals are identical. One null suffix per tape at its first target verifies clock/RNG reproduction. No outcome-based target filtering or combination of action effects.",
        "accounting": "Three full factual replays, three null suffix controls, all64 single-action suffixes; reused prefixes counted only in factual runs. Charge actual proposal+override scans, sampling/continuation/per-job costs, each constructor preprocessing, context reads, checkpoint reconstruction and RNG copying/restoring. No equal-cumulative-work claim.",
        "outcomes": "Paired all-cohort restricted mean at1200, common completed-ID P90, and unfinished at900/1200; separate LLM restricted mean/unfinished. Retain every positive, zero and negative delta. Also record whether one replacement alone repairs the old35->42 group count under the original5% bound while mean does not worsen.",
        "scope": "Retrospective mechanism diagnosis, three reused seeds; 64 interventions are dependent, not64 independent samples. Effects are not additive. Future births are harness-only offline counterfactual information and never policy input; this is not a deployable rule or independent confirmation.",
        "execution_sources": {**prior["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}}}


class PairedSelector:
    def __init__(self, planner, capture=(), force_first=False):
        self.planner, self.capture, self.force_first = planner, set(capture), force_first
        self.calls, self.states, self.work, self.intervention = 0, {}, Counter(), None

    def __call__(self, *context):
        if self.calls in self.capture:
            state = self.planner.rng.getstate()
            self.states[self.calls] = state
            self.work["checkpoint_rng_words_copied"] += len(state[1])
        proposed = self.planner(*context)
        self.work.update({key: getattr(proposed, key) for key in FEES})
        self.work["proposal_calls"] += 1
        if self.calls == 0 and self.force_first:
            control = self.planner.maxweight(*context)
            self.work["root_option_probes"] += control.root_option_probes
            self.work["override_calls"] += 1
            self.intervention = {"original": proposed.selection.option_indices[0],
                                 "replacement": control.selection.option_indices[0],
                                 "predicted_S1_advantage": proposed.baseline_predicted_completion_s - proposed.chosen_predicted_completion_s}
            selected = control
        else:
            selected = proposed
        self.calls += 1
        return selected


def make_planner(model, tape, protocol):
    return ContinuousArrivalAgeTailLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])


def collect_work(selector, result):
    work = Counter(selector.work)
    work.update({"preprocessing_" + key: value for key, value in selector.planner.preprocessing_work.items()})
    for key in ("age_context_job_reads", "observed_arrival_counter_updates", "observed_arrival_context_coordinate_reads"):
        work[key] += result["rollout_work"][key]
    return work


def paired_outcomes(factual, replacement, protocol, llm_coordinate):
    snapshots = {key: cohort_metrics({"factual": factual, "replacement": replacement},
                 protocol[key], protocol["birth_cohort"]) for key in ("arrival_horizon", "observation_end")}
    end = snapshots["observation_end"]
    return {"snapshots": snapshots,
        "delta_T_mean": end["replacement"]["restricted_mean_flow_s"] - end["factual"]["restricted_mean_flow_s"],
        "delta_T_p90": end["replacement"]["common_p90_flow_s"] - end["factual"]["common_p90_flow_s"],
        "delta_900_unfinished": snapshots["arrival_horizon"]["replacement"]["unfinished_jobs"] - snapshots["arrival_horizon"]["factual"]["unfinished_jobs"],
        "delta_LLM_T_mean": end["replacement"]["per_coordinate"][llm_coordinate]["restricted_mean_flow_s"] - end["factual"]["per_coordinate"][llm_coordinate]["restricted_mean_flow_s"],
        "delta_LLM_900_unfinished": snapshots["arrival_horizon"]["replacement"]["per_coordinate"][llm_coordinate]["unfinished_jobs"] - snapshots["arrival_horizon"]["factual"]["per_coordinate"][llm_coordinate]["unfinished_jobs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    expected = build_protocol()
    if args.register:
        with path(BASE, "_protocol.json").open("x") as out:
            json.dump({**expected, "registered_at_utc": datetime.now(timezone.utc).isoformat()}, out, separators=(",", ":"), allow_nan=False)
        print("registered", len(expected["targets"]), "interventions", flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered targets, sources or rules changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve existing diagnostic")
    models = {model.node: model for model in load_models(protocol["models"])}
    old = saved_runs()
    records, controls, work = [], [], Counter()
    for tape in protocol["tapes"]:
        model = models[tape["node"]]
        targets = [t for t in protocol["targets"] if t["tape_id"] == tape["tape_id"]]
        selector = PairedSelector(make_planner(model, tape, protocol), capture=[t["frame_index"] for t in targets])
        factual = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
            protocol["environment_arrival_end"], protocol["observation_end"], selector, policy="arrival_rollout")
        if not replay_matches(factual, old[tape["tape_id"]]):
            raise ValueError("factual replay changed")
        work.update(collect_work(selector, factual))
        for number, target in enumerate(targets):
            index = target["frame_index"]
            checkpoint = prepare_checkpoint(factual, index)
            work.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
            if number == 0:
                null = PairedSelector(make_planner(model, tape, protocol))
                null.planner.rng.setstate(selector.states[index])
                work["checkpoint_rng_words_restored"] += len(selector.states[index][1])
                resumed = resume_single_action_trajectory(model, protocol["environment_arrival_end"], protocol["observation_end"], null, checkpoint, policy="null")
                combined = {**resumed, "frames": factual["frames"][:index] + resumed["frames"]}
                if not replay_matches(combined, old[tape["tape_id"]]):
                    raise ValueError("null checkpoint clock/RNG continuation changed")
                controls.append({"tape_id": tape["tape_id"], "frame_index": index, "null_reproduction_passed": True})
                work.update(collect_work(null, resumed))
            branch = PairedSelector(make_planner(model, tape, protocol), force_first=True)
            branch.planner.rng.setstate(selector.states[index])
            work["checkpoint_rng_words_restored"] += len(selector.states[index][1])
            replacement = resume_single_action_trajectory(model, protocol["environment_arrival_end"], protocol["observation_end"], branch, checkpoint, policy="replacement")
            if branch.intervention["original"] != target["factual_option_index"] or branch.intervention["original"] == branch.intervention["replacement"]:
                raise ValueError("target must be exactly one changed factual proposal")
            work.update(collect_work(branch, replacement))
            outcomes = paired_outcomes(factual, replacement, protocol, model.workloads.index("gpu_llm_distilgpt2"))
            records.append({**target, **branch.intervention, **outcomes,
                "original_action": model.options[branch.intervention["original"]].action_id,
                "replacement_action": model.options[branch.intervention["replacement"]].action_id,
                "context": {key: factual["frames"][index][key] for key in ("queue", "waiting_ages", "available_before", "observed_arrival_counts")}})
            print(tape["tape_id"], index, "paired intervention complete", flush=True)
    if len(records) != len(protocol["targets"]) or len(controls) != len(protocol["tapes"]):
        raise ValueError("all registered interventions and controls required")
    result = {"decision": "SINGLE_ACTION_DIAGNOSTIC_COMPLETE", "registered_at_utc": protocol["registered_at_utc"],
        "scope": protocol["scope"], "factual_replays": len(protocol["tapes"]), "null_controls": controls,
        "interventions": len(records), "records": records, "actual_work": dict(work),
        "mean_helped_by_MW": sum(r["delta_T_mean"] < 0 for r in records),
        "LLM_mean_helped_by_MW": sum(r["delta_LLM_T_mean"] < 0 for r in records),
        "both_means_helped_by_MW": sum(r["delta_T_mean"] < 0 and r["delta_LLM_T_mean"] < 0 for r in records),
        "single_repair_without_mean_loss": sum(r["delta_900_unfinished"] <= -6 and r["delta_T_mean"] <= 0 for r in records)}
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "interventions", "mean_helped_by_MW", "LLM_mean_helped_by_MW", "both_means_helped_by_MW", "single_repair_without_mean_loss")}), flush=True)


if __name__ == "__main__":
    main()
