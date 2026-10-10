"""Swap one cached BG first root, retaining the complete finite continuation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import socket
from time import perf_counter, process_time

from algorithm.parallel_root_lane_rollout import ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout as Planner
from algorithm.experiments.aligned_hardware_persistent_action_controller_cross import prepare_finite_prefix
from algorithm.experiments.aligned_hardware_persistent_continuation_cross import native_first_frame
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_two_step_current_policy_bg_incumbent_development import check_rollout
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes, sign_counts
from algorithm.experiments.cached_finite_queue_propagation import decomposition, exact
from algorithm.experiments.persistent_action_controller_cross_replay import FiniteActionCrossSelector
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree, collect_work
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.cached_finite_bg_regression import FINITE, BG, CONTEXT, trajectories


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_finite_BG_first_action_cross_20261003"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
SOURCES = (SOURCE, "algorithm/experiments/aligned_hardware_persistent_action_controller_cross.py",
    "algorithm/experiments/persistent_action_controller_cross_replay.py",
    "algorithm/experiments/persistent_continuation_cross_selector.py",
    "algorithm/experiments/single_action_counterfactual_replay.py",
    "algorithm/experiments/two_step_bg_incumbent_continuation_cross_replay.py",
    "algorithm/experiments/persistent_parent_incumbent_scores.py",
    "algorithm/experiments/aligned_hardware_persistent_continuation_cross.py",
    "algorithm/experiments/cached_finite_queue_propagation.py",
    "algorithm/experiments/cached_finite_bg_regression.py")
METRICS = ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished", "delta_1200_unfinished")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read(file):
    return json.loads(Path(file).read_text())


def dump(file, value):
    with Path(file).open("x") as output:
        json.dump(value, output, separators=(",", ":"), allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def births(arm):
    return [(j["coordinate"], j["index"], j["arrival"]) for j in arm["jobs"]]


def register(directory, workers=13):
    started, cpu = perf_counter(), process_time()
    design = read(path(directory, "_design.json"))
    distribution = read(Path(directory) / (FINITE + "_protocol.json"))
    science, completed = distribution["scientific_protocol"], read(Path(directory) / (FINITE + ".json"))
    analysis = read(Path(directory) / design["source_analysis"])
    require(workers == design["workers_per_suffix"] == 13
        and science["registered_at_utc"] == completed["registered_at_utc"] == design["finite_scientific_epoch"]
        and science["planning_samples"] == design["planning_samples"] == 8
        and science["planning_event_cap"] == design["event_cap"] == 128
        and analysis["design_registered_at_utc"] == design["source_analysis_design_epoch"]
        and analysis["metric_reproduction_passed"] and completed["execution_checks_passed"]
        and [target["seed"] for target in design["targets"]] == [41, 42, 43], "unchanged frozen three-seed design required")
    bg_result = read(Path(directory) / (BG + ".json"))
    require(bg_result["registered_at_utc"] == design["BG_epoch"] and bg_result["execution_checks_passed"], "cached BG identity differs")
    bg_runs = trajectories(ROOT / science["previous_trajectories"], design["BG_epoch"])
    models = {m.node: m for m in load_models(science["models"])}
    origins = {a["tape_id"]: a for a in completed["execution_origins"]}
    targets, preparation = [], Counter(BG_gzip_reads=1)
    for index, target in enumerate(design["targets"]):
        tape_id, anchor = target["tape_id"], target["frame_index"]
        source = read(origins[tape_id]["row_path"])
        require(source["tape_id"] == tape_id and source["scientific_registered_at_utc"] == design["finite_scientific_epoch"], "finite cache identity differs")
        finite = trajectories(source["trajectory_file"], design["finite_scientific_epoch"])[tape_id]["arrival_rollout"]
        bg, frame = bg_runs[tape_id]["arrival_rollout"], finite["frames"][anchor]
        require(births(finite) == births(bg) and frame["option_indices"] == [target["factual_finite_root"]]
            and bg["frames"][anchor]["option_indices"] == [target["replacement_BG_root"]]
            and all(same(frame[key], target["context"][key]) for key in CONTEXT), "frozen natural first roots/context/births differ")
        for j in range(anchor + 1):
            require(all(same(finite["frames"][j][key], bg["frames"][j][key]) for key in CONTEXT)
                and (j == anchor or finite["frames"][j]["option_indices"] == bg["frames"][j]["option_indices"]),
                "cached BG/finite common physical prefix differs")
        cached = next(r for r in analysis["records"] if r["tape_id"] == tape_id)["first_divergence"]
        require(cached["frame_index"] == anchor and same(cached["context"], target["context"]), "registered first divergence differs from analysis")
        tape = next(t for t in science["tapes"] if t["tape_id"] == tape_id)
        checkpoint = prepare_checkpoint(finite, anchor)
        preparation.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
        preparation["finite_gzip_reads"] += 1
        fixture_file = path(directory, f"_fixture_{index:02d}.pkl")
        with fixture_file.open("xb") as output:
            pickle.dump(dict(model=models[tape["node"]], seed=f"arrival-plan-v1:{tape['node']}:{tape['load']}:online",
                checkpoint=checkpoint, finite_prefix={"jobs": finite["jobs"], "frames": finite["frames"][:anchor + 1]},
                finite_frame=frame, cached_finite={key: finite[key] for key in ("jobs", "snapshots")},
                finite_trajectory_file=source["trajectory_file"]), output)
        targets.append(dict(**target, fixture_file=str(fixture_file)))
    sources = {**science["execution_sources"], **distribution["execution_sources"],
        **{name: (ROOT / name).read_text() for name in SOURCES}}
    for name, literal in sources.items():
        require((ROOT / name).read_text() == literal, "frozen source changed: " + name)
    protocol = {key: science[key] for key in ("models", "tapes", "birth_cohort", "arrival_horizon", "observation_end",
        "environment_arrival_end", "planning_samples", "planning_event_cap", "clock")}
    protocol.update(registered_at_utc=datetime.now(timezone.utc).isoformat(), design=design, targets=targets, workers=workers,
        full24_decision=completed["decision"], scientific_source_registered_at_utc=science["registered_at_utc"],
        finite_distribution_registered_at_utc=distribution["distribution_registered_at_utc"], execution_sources=sources,
        expected_scientific_suffixes=3, expected_validation_suffixes=1, null_target_index=1,
        preparation_work=dict(preparation), registration_cpu_s=process_time() - cpu, registration_wall_s=perf_counter() - started)
    dump(path(directory, "_protocol.json"), protocol)
    print("registered science_suffixes=3 validation_suffixes=1 workers=13", flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    require(read(path(directory, "_design.json")) == protocol["design"], "registered design changed")
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_shard(index, directory, *, null=False):
    protocol = load_protocol(directory)
    require(not null or index == protocol["null_target_index"] == 1, "null validation is fixed to seed42")
    if not null:
        checked = read(path(directory, "_null_seed42.json"))
        require(checked["registered_at_utc"] == protocol["registered_at_utc"] and checked["cached_factual_suffix_reproduced"], "seed42 null validation must pass before science")
    target = protocol["targets"][index]
    suffix = "_null_seed42" if null else f"_shard_{index:02d}"
    trajectory_file = path(directory, suffix + "_trajectories.json.gz")
    require(not trajectory_file.exists(), "completed suffix already saved; inspect it before any new simulation")
    with Path(target["fixture_file"]).open("rb") as source:
        fixture = pickle.load(source)
    model, work = fixture["model"], Counter()
    started, cpu = perf_counter(), process_time()

    class RecordingPlanner(Planner):
        def __call__(self, *context):
            decision = super().__call__(*context)
            if self.decision_counter == target["frame_index"] + 1:
                self.anchor_post_tree = capture_planner_tree(self, work)
            return decision

    with RecordingPlanner(model.options, fixture["seed"], 8, 128, workers=protocol["workers"]) as planner:
        last, before = prepare_finite_prefix(planner, fixture["finite_prefix"], target["frame_index"], work)
        prefix_physical = json.loads(json.dumps(planner.reuse_work))
        selector = FiniteActionCrossSelector(planner, root=None if null else target["replacement_BG_root"])

        def select(*context):
            decision = selector(*context)
            if selector.calls == 1:
                require(same(native_first_frame(fixture["finite_frame"], selector.first_native_decision), fixture["finite_frame"])
                    and all(same(context[j], target["context"][key]) for j, key in ((1, "queue"), (2, "available_before"),
                        (3, "start"), (5, "waiting_ages"), (6, "observed_arrival_counts")))
                    and context[4] == target["context"]["lane_ids"][0]
                    and capture_planner_tree(planner, work) == planner.anchor_post_tree,
                    "native anchor/context/fees or complete post-proposal tree differ")
                print(target["tape_id"], "finite anchor verified", flush=True)
            if selector.calls % 20 == 0:
                print(target["tape_id"], "suffix dispatch", selector.calls, "service time", context[3], flush=True)
            return decision

        # Match original raw metadata; paired outcomes separately use cohort900.
        result = resume_single_action_trajectory(model, protocol["environment_arrival_end"], protocol["observation_end"],
            select, fixture["checkpoint"], policy="null_finite" if null else "forced_BG_then_finite")
        with trajectory_file.open("xb") as raw, gzip.open(raw, "wt") as output:
            json.dump(dict(registered_at_utc=protocol["registered_at_utc"], tape_id=target["tape_id"], result=result),
                output, separators=(",", ":"), allow_nan=False)
        require(selector.work["override_calls"] == int(not null)
            and result["frames"][0]["option_indices"] == (target["factual_finite_root"] if null else target["replacement_BG_root"],)
            and selector.status["finite_proposal_calls"] == len(result["frames"])
            and planner.decision_counter == target["frame_index"] + len(result["frames"])
            and births(result) == births(fixture["checkpoint"]), "single intervention/free finite suffix/births differ")
        require(all(old == new for old, new in zip(fixture["checkpoint"]["jobs"], result["jobs"], strict=True)
            if old["start"] is not None), "committed prefix jobs changed")
        checked = {**result, "frames": [native_first_frame(result["frames"][0], selector.first_native_decision)] + result["frames"][1:]}
        require(check_rollout(checked, model, protocol, planner), "native finite proposal/suffix integrity failed")
        before900 = target["context"]["start"] > protocol["arrival_horizon"]
        if before900:
            require(same(trajectory_snapshot(result["jobs"], protocol["arrival_horizon"], len(model.workloads)),
                trajectory_snapshot(fixture["cached_finite"]["jobs"], protocol["arrival_horizon"], len(model.workloads))),
                "after900 action changed actual snapshot900")
            require(all(new == old for old, new in zip(fixture["cached_finite"]["jobs"], result["jobs"], strict=True)
                if old["completion"] is not None and old["completion"] <= protocol["arrival_horizon"]), "after900 action changed completed jobs900")
        if null:
            factual = trajectories(fixture["finite_trajectory_file"], protocol["scientific_source_registered_at_utc"])[target["tape_id"]]["arrival_rollout"]
            require(same(result["frames"], factual["frames"][target["frame_index"]:])
                and result["jobs"] == factual["jobs"] and same(result["snapshots"], factual["snapshots"])
                and same(result["lane_events"], factual["lane_events"][target["frame_index"]:])
                and result["last_frame_end"] == factual["last_frame_end"], "seed42 complete null actions/native fees/jobs/snapshots differ")
        work.update(collect_work(selector, result))
        final = capture_planner_tree(planner, work)
        row = dict(index=index, tape_id=target["tape_id"], registered_at_utc=protocol["registered_at_utc"],
            validation_suffix=null, execution_checks_passed=True, cached_factual_suffix_reproduced=null,
            first_native_proposal_and_fees_verified=True, complete_post_proposal_tree_retained=True,
            snapshot900_unchanged=before900, last_touched_frames=last, calls=selector.calls,
            own_BG_incumbent_root=selector.first_native_decision.incumbent_root_index,
            own_BG_matches_registered_root=selector.first_native_decision.incumbent_root_index == target["replacement_BG_root"],
            first_proposal=selector.first_proposal, intervention=selector.intervention, work=dict(work),
            prefix_physical_work=prefix_physical, total_physical_work=planner.reuse_work,
            before_counters=[node["decision_counter"] for node in before["planners"]],
            final_counters=[node["decision_counter"] for node in final["planners"]],
            caller_cpu_s=process_time() - cpu, worker_cpu_s=planner.parallel_work["worker_cpu_s"],
            execution_host=socket.gethostname(), wall_s=perf_counter() - started)
    row["trajectory_file"] = str(trajectory_file)
    dump(path(directory, suffix + ".json"), row)
    print("complete", suffix, "calls", selector.calls, flush=True)


def flow_components(cached, forced, protocol, workloads):
    jobs = {name: {(j["coordinate"], j["index"]): j for j in arm["jobs"]} for name, arm in (("cached", cached), ("forced", forced))}
    lower, upper = protocol["birth_cohort"]
    ids = {key for key, job in jobs["cached"].items() if lower <= job["arrival"] <= upper}

    def group(selected, cutoff):
        public, _ = decomposition(selected, jobs["forced"], jobs["cached"], exact(cutoff))
        return dict(jobs=public["jobs"], forced_BG=public["parent"], cached_finite=public["native"],
            delta_forced_minus_cached=public["delta_parent_minus_native"])

    return {key: dict(all_cohort=group(ids, protocol[key]), per_coordinate=[dict(coordinate=c, workload=workload,
        **group({identity for identity in ids if identity[0] == c}, protocol[key])) for c, workload in enumerate(workloads)])
        for key in ("arrival_horizon", "observation_end")}


def merge(directory):
    protocol = load_protocol(directory)
    null = read(path(directory, "_null_seed42.json"))
    require(null["registered_at_utc"] == protocol["registered_at_utc"] and null["cached_factual_suffix_reproduced"], "registered seed42 null PASS required")
    records, work, cpu = [], Counter(), Counter()
    for index, target in enumerate(protocol["targets"]):
        row = read(path(directory, f"_shard_{index:02d}.json"))
        require(row["index"] == index and row["tape_id"] == target["tape_id"]
            and row["registered_at_utc"] == protocol["registered_at_utc"] and row["execution_checks_passed"], "all three matching science suffixes required")
        with Path(target["fixture_file"]).open("rb") as source:
            fixture = pickle.load(source)
        with gzip.open(row["trajectory_file"], "rt") as source:
            saved = json.load(source)
        require(saved["registered_at_utc"] == protocol["registered_at_utc"] and saved["tape_id"] == target["tape_id"], "science trajectory identity differs")
        forced = saved["result"]
        outcomes = paired_outcomes(fixture["cached_finite"], forced, protocol, fixture["model"].workloads)
        records.append(dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "factual_finite_root", "replacement_BG_root")},
            outcomes=outcomes, waiting_service=flow_components(fixture["cached_finite"], forced, protocol, fixture["model"].workloads),
            own_BG_incumbent_root=row["own_BG_incumbent_root"], own_BG_matches_registered_root=row["own_BG_matches_registered_root"],
            snapshot900_unchanged=row["snapshot900_unchanged"], calls=row["calls"]))
        work.update(row["work"])
        cpu.update({key: row[key] for key in ("caller_cpu_s", "worker_cpu_s")})
    summary = dict(registered_at_utc=protocol["registered_at_utc"], decision="FINITE_BG_FIRST_ACTION_CROSS_COMPLETE",
        execution_checks_passed=True, completed_new_suffixes=3, completed_validation_suffixes=1,
        reused_complete_finite_arms=3, records=records, signs={key: sign_counts([r["outcomes"][key] for r in records]) for key in METRICS},
        scientific_work=dict(work), scientific_cpu_s=dict(cpu), validation_work=null["work"],
        validation_cpu_s={key: null[key] for key in ("caller_cpu_s", "worker_cpu_s")}, registration_work=protocol["preparation_work"],
        registration_cpu_s=protocol["registration_cpu_s"], registration_wall_s=protocol["registration_wall_s"],
        full24_decision=protocol["full24_decision"], performance_gain_established=False,
        interpretation=protocol["design"]["decision_boundary"] + " " + protocol["design"]["outcomes"])
    dump(path(directory, "_summary.json"), summary)
    print("merged three scientific suffixes; separate seed42 null PASS; full24 remains HOLD", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--shard-index", type=int, choices=range(3))
    mode.add_argument("--run-null", action="store_true")
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--workers", type=int, default=13)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir, args.workers)
    elif args.merge:
        merge(args.artifact_dir)
    else:
        run_shard(1 if args.run_null else args.shard_index, args.artifact_dir, null=args.run_null)


if __name__ == "__main__":
    main()
