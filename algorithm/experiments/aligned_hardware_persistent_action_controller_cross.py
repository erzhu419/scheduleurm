"""Complete first-action by continuation crossings at three actual prefixes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import pickle
import socket
from pathlib import Path
from time import perf_counter
import json

from algorithm.parallel_root_lane_rollout import ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout as Planner
from algorithm.experiments import aligned_hardware_persistent_parent_action_replay as parent
from algorithm.experiments import aligned_hardware_persistent_current_policy_development as development
from algorithm.experiments.aligned_hardware_persistent_first_divergence import CONTEXT
from algorithm.experiments.aligned_hardware_persistent_continuation_cross import native_first_frame
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_two_step_current_policy_bg_incumbent_development import check_rollout
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK
from algorithm.experiments.persistent_parent_incumbent_scores import last_touched_frames
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree, collect_work
from algorithm.experiments.persistent_action_controller_cross_replay import FiniteActionCrossSelector
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import reconstruct_prefix_states, restore_planner_state
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory

ROOT, ARTIFACTS = parent.ROOT, parent.ARTIFACTS
STEM = "aligned_hardware_persistent_action_controller_cross_20261002"
SOURCES = ("algorithm/experiments/aligned_hardware_persistent_action_controller_cross.py",
    "algorithm/experiments/persistent_action_controller_cross_replay.py")
read, dump, require, same, births = parent.read, parent.dump, parent.require, parent.same, parent.births
METRICS = ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished", "delta_1200_unfinished")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read_jobs(file, epoch, tape_id):
    with gzip.open(file, "rt") as source:
        saved = json.load(source)
    require(saved["registered_at_utc"] == epoch and saved["tape_id"] == tape_id,
        "cached intervention jobs identity differs")
    return {key: saved[key] for key in ("jobs", "snapshots")}


def prepare_finite_prefix(planner, factual, anchor, work):
    """Reproduce finite children from their own last-touch native proposals."""
    frames = factual["frames"]
    last = last_touched_frames(frames, anchor)
    wanted = {anchor, *(index for index in last if index is not None)}
    states = reconstruct_prefix_states(planner, {**factual, "frames": frames[:anchor + 1]}, wanted, work)
    for index in sorted(wanted - {anchor}):
        frame = frames[index]
        work.update(restore_planner_state(planner, states[index]))
        decision = planner(planner.options, tuple(frame["queue"]), tuple(frame["available_before"]),
            frame["start"], frame["lane_ids"][0], tuple(map(tuple, frame["waiting_ages"])),
            tuple(frame["observed_arrival_counts"]))
        require(same(native_first_frame(frame, decision), frame),
            "finite last-touched native proposal/fees did not reproduce")
        work.update({"finite_prefix_" + name: getattr(decision, name) for name in FEES + TWO_STEP_WORK})
        work.update({"finite_prefix_inner_current_" + name: value for group in
            (decision.inner_current_work, decision.inner_current_fallbacks) for name, value in group.items()})
        work["finite_prefix_native_proposals"] += 1
        print(f"finite_prefix_verified frame={index} last_touched={last}", flush=True)
    work.update(restore_planner_state(planner, states[anchor]))
    previous = frames[anchor - 1]
    planner.arrival_rates = tuple(previous["rollout"]["estimated_arrival_rates"])
    planner._outer_observed_counts = tuple(previous["observed_arrival_counts"])
    return last, capture_planner_tree(planner, work)


def register(directory, workers):
    prior = parent.load_protocol(directory)
    completed = read(parent.path(directory, "_summary.json"))
    original = read(development.path(directory, "_protocol.json"))
    require(completed["registered_at_utc"] == prior["registered_at_utc"]
        and completed["execution_checks_passed"] and completed["completed_suffixes"] == 3
        and original["registered_at_utc"] == prior["full24_registered_at_utc"]
        and original["planning_samples"] == 8 and original["planning_event_cap"] == 128,
        "completed three true-prefix Persistent pairs and unchanged full24 source required")
    targets = []
    for index, target in enumerate(prior["targets"]):
        with Path(target["fixture_file"]).open("rb") as source:
            native = pickle.load(source)
        reference = original["references"][target["source_index"]]
        doc = read(reference["row_path"])
        require(reference["cached_checks_passed"] and doc["tape_id"] == target["tape_id"]
            and doc["trajectory_file"] == reference["trajectory_file"]
            and doc["scientific_registered_at_utc"] == original["finite_scientific_registered_at_utc"]
            and doc["distribution_registered_at_utc"] == reference["distribution_registered_at_utc"],
            "certified finite factual identity differs")
        with gzip.open(reference["trajectory_file"], "rt") as source:
            saved = json.load(source)
        require(saved["registered_at_utc"] == original["finite_scientific_registered_at_utc"],
            "finite factual epoch differs")
        finite = next(run["policies"]["arrival_rollout"] for run in saved["runs"]
            if run["tape_id"] == target["tape_id"])
        anchor = target["frame_index"]
        checkpoint = prepare_checkpoint(finite, anchor)
        frame = finite["frames"][anchor]
        require(frame["option_indices"] == [target["parent_root"]]
            and all(same(frame[key], native["native_frame"][key]) for key in CONTEXT)
            and same(checkpoint, native["checkpoint"]),
            "finite and Persistent first-divergence physical checkpoints must coincide")
        row = read(parent.path(directory, f"_shard_{index:02d}.json"))
        require(row["registered_at_utc"] == prior["registered_at_utc"]
            and row["execution_checks_passed"] and row["first_native_prefix_verified"]
            and row["tape_id"] == target["tape_id"], "prior forced-parent suffix differs")
        forced = read_jobs(row["jobs_file"], prior["registered_at_utc"], target["tape_id"])
        require(births(forced) == births(finite) == births(checkpoint), "all cached canonical births differ")
        fixture_file = path(directory, f"_fixture_{index:02d}.pkl")
        with fixture_file.open("xb") as output:
            pickle.dump(dict(model=native["model"], seed=native["seed"], checkpoint=checkpoint,
                finite_prefix={"jobs": finite["jobs"], "frames": finite["frames"][:anchor + 1]},
                finite_frame=frame, cached_arms={"parent_finite": {key: finite[key] for key in ("jobs", "snapshots")},
                    "native_persistent": {"jobs": native["factual_jobs"], "snapshots": native["factual_snapshots"]},
                    "parent_persistent": forced}, previous_persistent_outcomes=row["outcomes"]), output)
        targets.append({**target, "fixture_file": str(fixture_file), "finite_reference": reference})
    protocol = {key: prior[key] for key in ("planning_samples", "planning_event_cap", "birth_cohort",
        "arrival_horizon", "observation_end", "environment_arrival_end", "clock", "full24_decision")}
    protocol.update(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers, targets=targets,
        parent_replay_registered_at_utc=prior["registered_at_utc"], full24_registered_at_utc=prior["full24_registered_at_utc"],
        finite_scientific_registered_at_utc=original["finite_scientific_registered_at_utc"],
        expected_new_suffixes=3, cached_complete_arms=9,
        method="At the same three physical first-divergence states complete parent/native first action by finite/Persistent continuation. Reuse nine completed arms. Restore each finite controller's own causal prefix, replay at most two last-touched complete finite proposals with exact native decision/fee checks, then consume its anchor native proposal, force only the registered Persistent first root, and freely continue finite. No Persistent child states transplanted into finite.",
        interpretation="Conditional development interaction diagnostic, not a full-policy gain or unbiased forecast-error test. Root effects are parent-minus-native within each continuation; controller effects are Persistent-minus-finite at a fixed first root. Common P90 differences across pairs require all four arms to finish the identical complete birth cohort by1200. Preserve class and900/1200 tail negatives; no threshold/anchor scan, new gate, confirmation or paper edit.",
        execution_sources={**prior["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}})
    dump(path(directory, "_protocol.json"), protocol)
    print(f"registered_at_utc={protocol['registered_at_utc']} new_suffixes=3 cached_arms=9 workers={workers}", flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_shard(index, directory):
    protocol = load_protocol(directory)
    target = protocol["targets"][index]
    with Path(target["fixture_file"]).open("rb") as source:
        fixture = pickle.load(source)
    model, work = fixture["model"], Counter(target["preparation_work"])
    started = perf_counter()
    with Planner(model.options, fixture["seed"], 8, 128, workers=protocol["workers"]) as planner:
        last, before = prepare_finite_prefix(planner, fixture["finite_prefix"], target["frame_index"], work)
        prefix_physical_work = planner.reuse_work
        selector = FiniteActionCrossSelector(planner, root=target["native_root"])

        def select(*context):
            selected = selector(*context)
            if selector.calls == 1:
                require(same(native_first_frame(fixture["finite_frame"], selector.first_native_decision), fixture["finite_frame"])
                    and all(same(context[j], fixture["finite_frame"][key]) for j, key in
                        ((1, "queue"), (2, "available_before"), (3, "start"), (5, "waiting_ages"), (6, "observed_arrival_counts")))
                    and context[4] == fixture["finite_frame"]["lane_ids"][0],
                    "finite anchor proposal/context/fees differ")
                print(f"{target['tape_id']} finite_prefix_verified parent={target['parent_root']} forced_native={target['native_root']}", flush=True)
            if selector.calls % 20 == 0:
                print(f"{target['tape_id']} finite suffix dispatch={selector.calls} service_time={context[3]}", flush=True)
            return selected

        result = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
            protocol["observation_end"], select, fixture["checkpoint"], policy="forced_native_then_finite")
        require(selector.work["override_calls"] == 1
            and result["frames"][0]["option_indices"] == (target["native_root"],)
            and selector.status["finite_proposal_calls"] == len(result["frames"])
            and planner.decision_counter == target["frame_index"] + len(result["frames"])
            and births(result) == births(fixture["checkpoint"]), "single native intervention/free finite suffix/births differ")
        require(all(old == new for old, new in zip(fixture["checkpoint"]["jobs"], result["jobs"], strict=True)
            if old["start"] is not None), "committed prefix jobs changed")
        checked = {**result, "frames": [native_first_frame(result["frames"][0], selector.first_native_decision)] + result["frames"][1:]}
        require(check_rollout(checked, model, protocol, planner), "finite first proposal/free suffix integrity failed")
        work.update(collect_work(selector, result))
        final = capture_planner_tree(planner, work)
        row = dict(index=index, registered_at_utc=protocol["registered_at_utc"], tape_id=target["tape_id"],
            execution_checks_passed=True, first_finite_prefix_verified=True, last_touched_frames=last,
            calls=selector.calls, first_proposal=selector.first_proposal, intervention=selector.intervention,
            work=dict(work), prefix_physical_work=prefix_physical_work, total_physical_work=planner.reuse_work,
            before_counters=[node["decision_counter"] for node in before["planners"]],
            final_counters=[node["decision_counter"] for node in final["planners"]],
            execution_host=socket.gethostname(), wall_s=perf_counter() - started)
    jobs_file = path(directory, f"_shard_{index:02d}_jobs.json.gz")
    with jobs_file.open("xb") as raw, gzip.open(raw, "wt") as output:
        json.dump(dict(registered_at_utc=protocol["registered_at_utc"], tape_id=target["tape_id"],
            jobs=result["jobs"], snapshots=result["snapshots"]), output, separators=(",", ":"), allow_nan=False)
    row["jobs_file"] = str(jobs_file)
    dump(path(directory, f"_shard_{index:02d}.json"), row)
    print(f"complete index={index} finite_native_calls={selector.calls}", flush=True)


def merge(directory):
    protocol = load_protocol(directory)
    records = []
    for index, target in enumerate(protocol["targets"]):
        row = read(path(directory, f"_shard_{index:02d}.json"))
        require(row["index"] == index and row["registered_at_utc"] == protocol["registered_at_utc"]
            and row["execution_checks_passed"] and row["first_finite_prefix_verified"]
            and row["tape_id"] == target["tape_id"], "all three registered valid finite suffixes required")
        with Path(target["fixture_file"]).open("rb") as source:
            fixture = pickle.load(source)
        arms = {**fixture["cached_arms"], "native_finite": read_jobs(row["jobs_file"], protocol["registered_at_utc"], target["tape_id"])}
        require(all(births(arm) == births(fixture["checkpoint"]) for arm in arms.values()), "four-arm canonical births differ")
        effects = {controller: paired_outcomes(arms["native_" + controller], arms["parent_" + controller], protocol,
            fixture["model"].workloads) for controller in ("finite", "persistent")}
        require(same(effects["persistent"], fixture["previous_persistent_outcomes"]), "cached Persistent pair recomputation differs")
        controllers = {root: paired_outcomes(arms[root + "_finite"], arms[root + "_persistent"], protocol,
            fixture["model"].workloads) for root in ("parent", "native")}
        complete = all(group["arrivals"] == group["common_jobs"] == target["cohort_population"]
            for effect in (*effects.values(), *controllers.values())
            for group in effect["snapshots"]["observation_end"].values())
        interaction = {key: effects["persistent"][key] - effects["finite"][key]
            if effects["persistent"][key] is not None and effects["finite"][key] is not None
                and (key != "delta_T_common_p90" or complete) else None for key in METRICS}
        finite_delta, persistent_delta = (effects[name]["delta_T_mean"] for name in ("finite", "persistent"))
        records.append(dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "parent_root", "native_root", "cohort_population", "predictions")},
            root_effects=effects, controller_effects=controllers, interaction_persistent_minus_finite=interaction,
            complete_four_arm_common_cohort_p90=complete, strict_mean_ranking_reversal=finite_delta * persistent_delta < 0,
            parent_mean_helps_both=finite_delta < 0 and persistent_delta < 0))
    summary = dict(registered_at_utc=protocol["registered_at_utc"], decision="PERSISTENT_ACTION_CONTROLLER_CROSS_COMPLETE",
        execution_checks_passed=True, completed_new_suffixes=3, reused_complete_arms=9, records=records,
        full24_decision=protocol["full24_decision"], performance_gain_established=False, interpretation=protocol["interpretation"])
    dump(path(directory, "_summary.json"), summary)
    print("complete crossings=3 new_suffixes=3 cached_arms=9 full24=HOLD", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--shard-index", type=int, choices=range(3))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--workers", type=int, default=22)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir, args.workers)
    elif args.merge:
        merge(args.artifact_dir)
    else:
        run_shard(args.shard_index, args.artifact_dir)


if __name__ == "__main__":
    main()
