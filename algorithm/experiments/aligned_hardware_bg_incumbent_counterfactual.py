"""All improved BG-incumbent roots, individually restored to BG on development tapes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.experiments.aligned_hardware_background_competition_continuous import path
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes, sign_counts
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.bg_incumbent_counterfactual_replay import (
    BGIncumbentCounterfactualSelector, collect_work, make_planner, registered_targets, restore_planner_state,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_bg_incumbent_background_redecision_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_bg_incumbent_counterfactual_20260930"
SOURCES = (
    "algorithm/experiments/single_action_counterfactual_replay.py",
    "algorithm/experiments/aligned_hardware_background_root_counterfactual.py",
    "algorithm/experiments/bg_incumbent_counterfactual_replay.py",
    "algorithm/experiments/aligned_hardware_bg_incumbent_counterfactual.py",
)
METRICS = ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished",
           "delta_1200_unfinished", "delta_middle_queue_area_s")
CONTEXT = ("start", "queue", "waiting_ages", "available_before", "observed_arrival_counts", "lane_ids")


def saved_runs():
    with gzip.open(path(PREVIOUS, "_trajectories.json.gz"), "rt") as source:
        return {run["tape_id"]: run["policies"]["arrival_rollout"]
                for run in json.load(source)["runs"]}


def build_protocol():
    prior = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    result = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if (result["decision"] != "BG_INCUMBENT_BACKGROUND_REDECISION_DEVELOPMENT_HOLD"
            or not result["execution_checks_passed"]
            or result["registered_at_utc"] != prior["registered_at_utc"]):
        raise ValueError("matching execution-valid incumbent development HOLD required")
    for name, source in prior["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    checks = {row["tape_id"]: {key: row[key] for key in (
        "baseline_reproduction_passed", "rollout_checks_passed", "cached_reference_checks_passed")}
        for row in result["paired_runs"]}
    if any(not all(checks[tape["tape_id"]].values()) for tape in prior["tapes"]):
        raise ValueError("every cached incumbent tape requires passing execution checks")
    targets = registered_targets(prior["tapes"], saved_runs())
    active = [tape["tape_id"] for tape in prior["tapes"]
              if any(target["tape_id"] == tape["tape_id"] for target in targets)]
    return {
        "experiment": "all_bg_incumbent_improvement_counterfactual_v1",
        "previous_protocol": str(path(PREVIOUS, "_protocol.json").relative_to(ROOT)),
        "previous_result": str(PREVIOUS.with_suffix(".json").relative_to(ROOT)),
        "previous_registered_at_utc": prior["registered_at_utc"],
        "previous_decision": result["decision"], "cached_execution_checks": checks,
        **{key: prior[key] for key in (
            "models", "tapes", "arrival_seeds", "birth_cohort", "arrival_horizon", "observation_end",
            "environment_arrival_end", "planning_samples", "planning_event_cap", "planning_seed_rule",
            "clock", "cutoff_semantics", "information")},
        "targets": targets, "active_tape_ids": active,
        "null_targets": [next(target for target in targets if target["tape_id"] == tape_id)
                         for tape_id in active],
        "expected_executed_trajectories": len(prior["tapes"]) + len(active) + len(targets),
        "workers": 8,
        "method": "Register every cached improved_incumbent call at all times and on all24 tapes, with no outcome filter. Reproduce all24 full candidate trajectories and capture outer prefix RNG plus decision_counter before every target. Resume one earliest-target null per active tape and require exact full jobs/actions/clock reproduction. At each registered target restore the same factual FIFO/jobs/full busy commitments/released counts and planner state; compute the entire original current-candidate proposal, replace only its first action with its already evaluated BG incumbent, then freely replan with the current candidate at every later call. Require every first proposal and physical context to match the cached factual call. No independent BG prefix sampling or MW continuation arm.",
        "budget": prior["budget"] + " Counterfactual harness executes eight tape workers. Every suffix preserves outer/inner8 samples and cap128. Charge all original proposal main/incumbent/recourse fees even when overridden, plus one extra m-option MW certificate scan per forced root, all constructors/preprocessing/context/checkpoint/state capture and restore work. Extra override probes are separate; engine selector aliases are not added again. Cached baseline outcomes are reused, not rerun per target. Save only controls, paired outcomes and action/context descriptors, not suffix trajectories.",
        "outcomes": "BG replacement minus factual candidate: closed birth-cohort[300,900] restricted mean at1200, same-pair completed-ID common P90,900/1200 unfinished counts and all-job waiting area[300,900]. Preserve all class restricted means, same-class common-ID mean/P90 and tails. Negative delta means restoring BG helped (the proposed replacement was harmful under this suffix); positive means the proposed replacement helped. Effects of different targets are dependent and cannot be added.",
        "strata": {"time": ["pre300", "middle300_900", "late900plus"],
                   "node_load": [{"node": model["node"], "load": load}
                                 for model in prior["models"] for load in prior["loads"]],
                   "seeds": prior["arrival_seeds"]},
        "scope": "Conditional single-action diagnostic on reused development seeds41..43. Retain prior HOLD. No policy-performance PASS, fresh confirmation, independent320-sample inference, additive intervention benefit or physical-run claim.",
        "execution_sources": {**prior["execution_sources"],
                              **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def time_group(start):
    return "pre300" if start < 300 else "middle300_900" if start < 900 else "late900plus"


def summarize_effects(records):
    summary = {"targets": len(records)}
    for metric in METRICS:
        values = [record["outcomes"][metric] for record in records]
        available = [value for value in values if value is not None]
        summary[metric] = {**sign_counts(values), "minimum": min(available, default=None),
                           "maximum": max(available, default=None)}
    summary["restore_BG_helps_mean_and_900_tail"] = sum(
        row["outcomes"]["delta_T_mean"] < 0 and row["outcomes"]["delta_900_unfinished"] < 0
        for row in records)
    summary["proposal_helps_mean_but_hurts_any_class_900_tail"] = sum(
        row["outcomes"]["delta_T_mean"] > 0 and any(
            item["delta_900_unfinished"] < 0 for item in row["outcomes"]["per_coordinate_deltas"])
        for row in records)
    return summary


def action_descriptor(option):
    return {"action_id": option.action_id, "requirements": option.requirements,
            "duration": option.duration, "completion_offsets": option.completion_offsets}


def run_tape(model, tape, protocol, frozen, targets):
    work, status, records, controls = Counter(), Counter(), [], []
    capture = BGIncumbentCounterfactualSelector(make_planner(model, tape, protocol),
                                               capture=[target["frame_index"] for target in targets])
    factual = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"], capture, policy="arrival_rollout")
    if (not replay_matches(factual, frozen)
            or not all(same(new["rollout"], old["rollout"])
                       for new, old in zip(factual["frames"], frozen["frames"]))):
        raise ValueError("full factual jobs/actions/clock/proposals differ: " + tape["tape_id"])
    work.update(collect_work(capture, factual))
    status.update(capture.status)
    print(tape["tape_id"], "factual replay matched; targets", len(targets), flush=True)
    for number, target in enumerate(targets):
        index = target["frame_index"]
        frame = factual["frames"][index]
        checkpoint = prepare_checkpoint(factual, index)
        work.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
        if number == 0:
            null = BGIncumbentCounterfactualSelector(make_planner(model, tape, protocol))
            work.update(restore_planner_state(null.planner, capture.states[index]))
            resumed = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
                protocol["observation_end"], null, checkpoint, policy="null")
            combined = {**resumed, "frames": factual["frames"][:index] + resumed["frames"]}
            if (not replay_matches(combined, frozen)
                    or null.planner.rng.getstate() != capture.planner.rng.getstate()
                    or null.planner.decision_counter != capture.planner.decision_counter):
                raise ValueError("null jobs/actions/clock/RNG/counter differ: " + tape["tape_id"])
            work.update(collect_work(null, resumed))
            status.update(null.status)
            controls.append({"tape_id": tape["tape_id"], "frame_index": index,
                             "null_reproduction_passed": True})
        selector = BGIncumbentCounterfactualSelector(make_planner(model, tape, protocol), force_first=True)
        work.update(restore_planner_state(selector.planner, capture.states[index]))
        replacement = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
            protocol["observation_end"], selector, checkpoint, policy="BG_first_candidate_suffix")
        proposal, intervention = selector.first_proposal, selector.intervention
        if (proposal["original"] != target["factual_option_index"]
                or proposal["replacement"] != target["replacement_option_index"]
                or intervention["maxweight_option_index"] != target["maxweight_option_index"]
                or not same(proposal["rollout"], frame["rollout"])
                or not all(same(replacement["frames"][0][key], frame[key]) for key in CONTEXT)
                or replacement["frames"][0]["option_indices"][0] != target["replacement_option_index"]):
            raise ValueError("restored first proposal/context/applied BG differs: " + str(target))
        work.update(collect_work(selector, replacement))
        status.update(selector.status)
        original_option = model.options[intervention["original"]]
        bg_option = model.options[intervention["replacement"]]
        records.append({**target, **intervention,
            **{key: tape[key] for key in ("node", "load", "seed")},
            "time_group": time_group(target["start"]),
            "decision_counter": capture.states[index]["decision_counter"],
            "first_proposal_context_match_passed": True,
            "original_action": action_descriptor(original_option),
            "replacement_action": action_descriptor(bg_option),
            "same_support": tuple(bool(q) for q in original_option.requirements)
                            == tuple(bool(q) for q in bg_option.requirements),
            "context": {key: frame[key] for key in CONTEXT},
            "outcomes": paired_outcomes(factual, replacement, protocol, model.workloads)})
        print(tape["tape_id"], "forced BG", number + 1, "/", len(targets),
              "frame", index, "mean delta", records[-1]["outcomes"]["delta_T_mean"], flush=True)
    return {"tape_id": tape["tape_id"], "factual_reproduction_passed": True,
            "records": records, "controls": controls, "work": dict(work), "status": dict(status)}


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
        print("registered", len(expected["targets"]), "roots; executions",
              expected["expected_executed_trajectories"], flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources, cached checks, targets or tapes changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve existing diagnostic outcome")
    frozen = saved_runs()
    models = {model.node: model for model in load_models(protocol["models"])}
    work, status, records, controls, factual_checks = Counter(), Counter(), [], [], []
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = [pool.submit(run_tape, models[tape["node"]], tape, protocol, frozen[tape["tape_id"]],
            [target for target in protocol["targets"] if target["tape_id"] == tape["tape_id"]])
            for tape in protocol["tapes"]]
        for future in as_completed(futures):
            row = future.result()
            work.update(row["work"])
            status.update(row["status"])
            records.extend(row["records"])
            controls.extend(row["controls"])
            factual_checks.append({key: row[key] for key in ("tape_id", "factual_reproduction_passed")})
            print("tape complete", row["tape_id"], "total roots", len(records), flush=True)
    order = {(target["tape_id"], target["frame_index"]): index
             for index, target in enumerate(protocol["targets"])}
    records.sort(key=lambda row: order[(row["tape_id"], row["frame_index"])])
    if len(records) != len(protocol["targets"]) or len(controls) != len(protocol["active_tape_ids"]):
        raise ValueError("all registered roots and null controls required")
    strata = {
        "time": [{"time_group": group, **summarize_effects([row for row in records if row["time_group"] == group])}
                 for group in protocol["strata"]["time"]],
        "node_load": [{**cell, **summarize_effects([row for row in records
                      if row["node"] == cell["node"] and row["load"] == cell["load"]])}
                      for cell in protocol["strata"]["node_load"]],
        "seed": [{"seed": seed, **summarize_effects([row for row in records if row["seed"] == seed])}
                 for seed in protocol["strata"]["seeds"]],
    }
    work["total_option_probes"] = sum(work[key] for key in (
        "root_option_probes", "rollout_option_probes", "incumbent_root_option_probes",
        "incumbent_rollout_option_probes", "recourse_root_option_probes",
        "recourse_rollout_option_probes", "override_root_option_probes",
        "preprocessing_option_probes", "preprocessing_age_profile_option_probes",
        "preprocessing_rate_dimension_option_probes", "preprocessing_recourse_option_probes",
        "preprocessing_recourse_age_profile_option_probes",
        "preprocessing_recourse_rate_dimension_option_probes"))
    result = {"decision": "BG_INCUMBENT_COUNTERFACTUAL_COMPLETE",
        "registered_at_utc": protocol["registered_at_utc"], "scope": protocol["scope"],
        "registered_tapes": len(protocol["tapes"]), "targets": len(records),
        "factual_replays": factual_checks, "null_controls": controls,
        "execution_checks_passed": True,
        "total_executed_trajectories": len(factual_checks) + len(controls) + len(records),
        "summary": summarize_effects(records), "strata": strata, "records": records,
        "actual_work": dict(work), "status_counts": dict(status)}
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in (
        "decision", "targets", "total_executed_trajectories", "summary")}), flush=True)


if __name__ == "__main__":
    main()
