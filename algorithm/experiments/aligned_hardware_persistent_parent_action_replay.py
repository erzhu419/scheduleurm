"""Replace one parent action at three true GPU2 prefixes, then freely replan."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import socket
from time import perf_counter

from algorithm.tick_persistent_current_policy_lane_rollout import (
    ParallelTickPersistentCurrentPolicyBGIncumbentLaneRollout as Planner,
)
from algorithm.experiments import aligned_hardware_persistent_parent_incumbent_scores as scores
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_persistent_continuation_cross import native_first_frame
from algorithm.experiments.aligned_hardware_two_step_current_policy_root_ranking import same
from algorithm.experiments.persistent_continuation_cross_selector import (
    PersistentContinuationCrossSelector, capture_planner_tree, restore_planner_tree, collect_work,
)
from algorithm.experiments.persistent_current_policy_checks import check_rollout
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_persistent_parent_action_replay_20261002"
SOURCE = "algorithm/experiments/aligned_hardware_persistent_parent_action_replay.py"
SOURCES = (SOURCE, "algorithm/experiments/persistent_continuation_cross_selector.py",
    "algorithm/experiments/single_action_counterfactual_replay.py",
    "algorithm/experiments/aligned_hardware_persistent_continuation_cross.py")
read, dump, require = scores.read, scores.dump, scores.require


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def births(result):
    return [(job["coordinate"], job["index"], job["arrival"]) for job in result["jobs"]]


def register(directory, workers):
    prior = scores.load_protocol(directory)
    completed = read(scores.path(directory, "_summary.json"))
    parent = read(Path(directory) / (scores.PARENT + "_protocol.json"))
    require(completed["execution_checks_passed"] and completed["completed_shards"] == 24
        and completed["registered_at_utc"] == prior["registered_at_utc"]
        and completed["full24_decision"] == "PERSISTENT_CURRENT_POLICY_DEVELOPMENT_HOLD"
        and parent["registered_at_utc"] == prior["full24_registered_at_utc"]
        and parent["planning_samples"] == 8 and parent["planning_event_cap"] == 128
        and parent["birth_cohort"] == [300, 900]
        and parent["arrival_horizon"] == 900 and parent["observation_end"] == parent["environment_arrival_end"] == 1200,
        "completed true-prefix scores and original budget/cohort/clocks required")
    model = next(m for m in load_models(parent["models"]) if m.node == "gpu_b")
    targets = []
    expected = ((21, 41, 11, 2, 5), (22, 42, 39, 7, 4), (23, 43, 29, 4, 5))
    for index, (source_index, seed, frame_index, p, native) in enumerate(expected):
        assignment = prior["assignments"][source_index]
        row = read(scores.path(directory, f"_shard_{source_index:02d}.json"))
        tape_id = f"gpu_b:R2:load0.95:seed{seed}"
        require(row["execution_checks_passed"] and row["registered_at_utc"] == prior["registered_at_utc"]
            and row["tape_id"] == assignment["tape_id"] == tape_id
            and row["frame_index"] == assignment["frame_index"] == frame_index
            and row["finite_parent_root"] == p and row["native_persistent_root"] == native,
            "retain all three literal first divergences and both actions")
        with Path(row["exact_evidence_file"]).open("rb") as source:
            evidence = pickle.load(source)
        require(evidence["registered_at_utc"] == prior["registered_at_utc"]
            and evidence["target"] == assignment
            and evidence["before_tree"]["planners"][0]["decision_counter"] == frame_index
            and evidence["after_tree"]["planners"][0]["decision_counter"] == frame_index + 1,
            "previous exact before/after prefix tree required")
        with gzip.open(assignment["trajectory_file"], "rt") as source:
            saved = json.load(source)
        require(saved["registered_at_utc"] == prior["full24_registered_at_utc"], "native factual epoch differs")
        factual = next(r["policies"]["arrival_rollout"] for r in saved["runs"] if r["tape_id"] == tape_id)
        frame = factual["frames"][frame_index]
        tape = parent["tapes"][source_index]
        require(frame["option_indices"] == [native] and same(frame["rollout"], row["native_logical_work"])
            and births(factual) == [(c, j, float(t)) for t, c, j in sorted(tape["arrivals"])],
            "registered native action, logical fees and literal canonical births required")
        checkpoint = prepare_checkpoint(factual, frame_index)
        cohort_population = sum(300 <= j["arrival"] <= 900 for j in factual["jobs"])
        costs = evidence["persistent_scores"]
        prediction = dict(parent_S1_mean=float(costs.costs[p] / 8), native_S1_mean=float(costs.costs[native] / 8),
            parent_S2_mean=float(costs.squared_costs[p] / 8), native_S2_mean=float(costs.squared_costs[native] / 8),
            parent_minus_native_S1_mean=float((costs.costs[p] - costs.costs[native]) / 8),
            parent_minus_native_S2_mean=float((costs.squared_costs[p] - costs.squared_costs[native]) / 8),
            native_strictly_better_S1=costs.costs[native] < costs.costs[p],
            native_strictly_better_S2=costs.squared_costs[native] < costs.squared_costs[p])
        fixture_file = path(directory, f"_fixture_{index:02d}.pkl")
        with fixture_file.open("xb") as output:
            pickle.dump(dict(model=model, seed=f"arrival-plan-v1:{model.node}:0.95:online",
                checkpoint=checkpoint, before_tree=evidence["before_tree"], after_tree=evidence["after_tree"],
                native_frame=frame, factual_jobs=factual["jobs"], factual_snapshots=factual["snapshots"]), output)
        targets.append(dict(index=index, source_index=source_index, tape_id=tape_id, seed=seed,
            frame_index=frame_index, start=frame["start"], parent_root=p, native_root=native,
            offline_guard_root=row["offline_root"], cohort_population=cohort_population,
            fixture_file=str(fixture_file), predictions=prediction,
            preparation_work={k: checkpoint[k] for k in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")}))
    protocol = {k: parent[k] for k in ("planning_samples", "planning_event_cap", "birth_cohort",
        "arrival_horizon", "observation_end", "environment_arrival_end", "clock")}
    protocol.update(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers, targets=targets,
        scores_registered_at_utc=prior["registered_at_utc"], full24_registered_at_utc=prior["full24_registered_at_utc"],
        full24_decision=completed["full24_decision"], expected_suffix_executions=3, cached_native_references=3,
        method="Restore each true Persistent pre-action three-controller tree and canonical FIFO/committed lanes. Consume the complete native proposal, check every cached field and post-proposal tree, replace only its first action with the cached finite-parent root, then freely replan Persistent. Reuse existing complete native factual jobs as controls. Three GPU2/.95 states fixed before new service executions; no threshold/anchor/seed scan.",
        interpretation="Parent-minus-native conditional service effects at three reused development states. Sampled bounded tagged-population predictions and complete [300,900] birth-cohort outcomes are distinct populations; opposite signs are not an unbiased calibration test. Common P90 is a complete-population comparison only when both arms finish the whole same cohort by1200. No additive full24 effect, deployable cached-parent oracle, fresh confirmation or performance gain is established.",
        execution_sources={**prior["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}})
    dump(path(directory, "_protocol.json"), protocol)
    print(f"registered_at_utc={protocol['registered_at_utc']} suffixes=3 cached_controls=3 workers={workers}", flush=True)


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
        work.update(restore_planner_tree(planner, fixture["before_tree"]))
        selector = PersistentContinuationCrossSelector(planner, root=target["parent_root"])

        def select(*context):
            selected = selector(*context)
            if selector.calls == 1:
                native = native_first_frame(fixture["native_frame"], selector.first_native_decision)
                require(same(native, fixture["native_frame"])
                    and all(same(context[j], fixture["native_frame"][key]) for j, key in
                        ((1, "queue"), (2, "available_before"), (3, "start"), (5, "waiting_ages"), (6, "observed_arrival_counts")))
                    and context[4] == fixture["native_frame"]["lane_ids"][0]
                    and capture_planner_tree(planner, work) == fixture["after_tree"],
                    "first native proposal/context/post-tree did not reproduce")
                print(f"{target['tape_id']} native_prefix_verified parent={target['parent_root']} native={target['native_root']}", flush=True)
            if selector.calls % 20 == 0:
                print(f"{target['tape_id']} suffix dispatch={selector.calls} service_time={context[3]}", flush=True)
            return selected

        result = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
            protocol["observation_end"], select, fixture["checkpoint"], policy="forced_parent_then_persistent")
        require(selector.work["override_calls"] == 1
            and result["frames"][0]["option_indices"] == (target["parent_root"],)
            and selector.intervention["registered_first_root"] == target["parent_root"]
            and selector.status["persistent_proposal_calls"] == len(result["frames"])
            and planner.decision_counter == target["frame_index"] + len(result["frames"])
            and births(result) == births(fixture["checkpoint"]), "single intervention, free suffix or births differ")
        require(all(a == b for a, b in zip(fixture["checkpoint"]["jobs"], result["jobs"], strict=True)
            if a["start"] is not None), "previously committed prefix jobs changed")
        checked = {**result, "frames": [native_first_frame(result["frames"][0], selector.first_native_decision)]
            + result["frames"][1:]}
        executed = planner.reuse_work
        require(check_rollout(checked, model, protocol, planner), "native first proposal/free suffix integrity failed")
        work.update(collect_work(selector, result))
        outcomes = paired_outcomes(dict(jobs=fixture["factual_jobs"], snapshots=fixture["factual_snapshots"]),
            result, protocol, model.workloads)
        complete = all(g["arrivals"] == g["common_jobs"] == target["cohort_population"]
            for g in outcomes["snapshots"]["observation_end"].values())
        final_tree = capture_planner_tree(planner, work)
        row = dict(index=index, registered_at_utc=protocol["registered_at_utc"], tape_id=target["tape_id"],
            execution_checks_passed=True, first_native_prefix_verified=True, calls=selector.calls,
            intervention=selector.intervention, predictions=target["predictions"], outcomes=outcomes,
            complete_common_cohort_p90=complete, first_native_proposal=selector.first_proposal,
            work=dict(work), native_physical_work=executed, final_counters=[s["decision_counter"]
                for s in final_tree["planners"]],
            execution_host=socket.gethostname(), wall_s=perf_counter() - started)
    jobs_file = path(directory, f"_shard_{index:02d}_jobs.json.gz")
    with jobs_file.open("xb") as raw, gzip.open(raw, "wt") as output:
        json.dump(dict(registered_at_utc=protocol["registered_at_utc"], tape_id=target["tape_id"],
            jobs=result["jobs"], snapshots=result["snapshots"]), output, separators=(",", ":"), allow_nan=False)
    row["jobs_file"] = str(jobs_file)
    dump(path(directory, f"_shard_{index:02d}.json"), row)
    print(f"complete index={index} parent_minus_native_mean={outcomes['delta_T_mean']} calls={selector.calls}", flush=True)


def merge(directory):
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_shard_{i:02d}.json")) for i in range(3)]
    for i, row in enumerate(rows):
        require(row["index"] == i and row["registered_at_utc"] == protocol["registered_at_utc"]
            and row["execution_checks_passed"] and row["first_native_prefix_verified"]
            and row["tape_id"] == protocol["targets"][i]["tape_id"], "all three registered valid suffixes required")
    records = [dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "start", "parent_root",
        "native_root", "offline_guard_root", "cohort_population", "predictions")},
        outcomes=row["outcomes"], complete_common_cohort_p90=row["complete_common_cohort_p90"],
        calls=row["calls"]) for target, row in zip(protocol["targets"], rows, strict=True)]
    summary = dict(registered_at_utc=protocol["registered_at_utc"], decision="PERSISTENT_PARENT_ACTION_REPLAY_COMPLETE",
        execution_checks_passed=True, completed_suffixes=3, cached_native_controls=3,
        full24_decision=protocol["full24_decision"], performance_gain_established=False,
        records=records, direction="Every delta is forced-parent minus cached native; negative helps parent.",
        interpretation=protocol["interpretation"])
    dump(path(directory, ".json"), {**summary, "suffix_records": rows})
    dump(path(directory, "_summary.json"), summary)
    print("complete suffixes=3 full24=HOLD", flush=True)


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
