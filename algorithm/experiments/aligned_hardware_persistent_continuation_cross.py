"""Cross the two frozen A/third actions with persistent-current continuation."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import socket
from time import perf_counter

from algorithm.tick_persistent_current_policy_lane_rollout import (
    ParallelTickPersistentCurrentPolicyBGIncumbentLaneRollout as ParallelPlanner,
    TickPersistentCurrentPolicyBGIncumbentLaneRollout as SerialPlanner,
)
from algorithm.experiments import aligned_hardware_current_policy_third_root_labels as labels
from algorithm.experiments import aligned_hardware_tick_background_benchmark as tick
from algorithm.experiments import aligned_hardware_two_step_current_policy_root_ranking as ranking
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_bg_incumbent_counterfactual import CONTEXT
from algorithm.experiments.persistent_continuation_cross_selector import (
    PersistentContinuationCrossSelector, capture_planner_tree, restore_planner_tree, collect_work,
)
from algorithm.experiments.persistent_current_policy_checks import check_rollout
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import (
    reconstruct_prefix_states, restore_planner_state,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_persistent_continuation_cross_20261002"
PARENT = "aligned_hardware_persistent_current_policy_development_20261002"
SOURCES = ("algorithm/experiments/aligned_hardware_persistent_continuation_cross.py",
    "algorithm/experiments/persistent_continuation_cross_selector.py")
METRICS = (*labels.METRICS, "delta_middle_queue_area_s")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read(file):
    return json.loads(Path(file).read_text())


def dump(file, data):
    with Path(file).open("x") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def seed_for(model, tape):
    return f"arrival-plan-v1:{model.node}:{tape['load']}:online"


def register(directory, workers):
    prior = labels.load_protocol(directory)
    completed = read(labels.path(directory, "_summary.json"))
    optimized = tick.load_protocol(directory)
    benchmark = read(tick.artifact(directory, "_summary.json"))
    parent = read(Path(directory) / (PARENT + "_summary.json"))
    if (not completed["execution_checks_passed"]
            or completed["registered_at_utc"] != prior["registered_at_utc"]
            or completed["native_third_roots_without_labels"]
            or not benchmark["exact_match_passed"]
            or benchmark["registered_at_utc"] != optimized["registered_at_utc"]
            or not parent["execution_checks_passed"]
            or parent["registered_at_utc"] != optimized["parent_registration"]
            or parent["decision"] != "PERSISTENT_CURRENT_POLICY_DEVELOPMENT_HOLD"):
        raise ValueError("completed OLD labels, exact Tick benchmark and retained full24 HOLD required")
    if (prior["planning_samples"] != 8 or prior["planning_event_cap"] != 128
            or prior["birth_cohort"] != [300, 900]
            or prior["environment_arrival_end"] != 1200 or prior["observation_end"] != 1200):
        raise ValueError("retain the frozen budget/cohort/environment clocks")
    expected = [("gpu_a:R2:load0.95:seed43", 76, 4, 5),
        ("gpu_b:R2:load0.95:seed42", 125, 3, 5)]
    factuals = ranking.saved_runs(prior)
    models = {m.node: m for m in ranking.load_models(prior["models"])}
    old_labels = {r["tape_id"]: r for r in completed["third_root_records"]}
    targets, assignments, preparation = [], [], Counter()
    for i, (target, wanted) in enumerate(zip(prior["targets"], expected, strict=True)):
        tape_id, frame_index, original, third = wanted
        factual = factuals[tape_id]
        frame = factual["frames"][frame_index]
        if (target["tape_id"] != tape_id or target["frame_index"] != frame_index
                or target["new_native_option_index"] != third or frame["option_indices"] != [original]):
            raise ValueError("retain both fixed A/third actions and states")
        tape = next(t for t in prior["tapes"] if t["tape_id"] == tape_id)
        model = models[tape["node"]]
        planner = SerialPlanner(model.options, seed_for(model, tape), 8, 128)
        work = Counter(preparation_controller_constructors=3)
        work.update({"preparation_preprocessing_" + k: v for k, v in planner.preprocessing_work.items()})
        # OLD factual prefix determines outer randomness; the new children enter
        # from common constructors, since no persistent factual prefix exists.
        prefix = {**factual, "frames": factual["frames"][:frame_index + 1]}
        state = reconstruct_prefix_states(planner, prefix, [frame_index], work)[frame_index]
        work.update(restore_planner_state(planner, state))
        checkpoint = prepare_checkpoint(factual, frame_index)
        work.update({k: checkpoint[k] for k in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
        tree = capture_planner_tree(planner, work)
        cohort_ids = [(j["coordinate"], j["index"]) for j in checkpoint["jobs"]
            if 300 <= j["arrival"] <= 900]
        old = old_labels[tape_id]
        if (old["frame_index"] != frame_index or old["new_native_option_index"] != third
                or not all(v["native_suffix_checks_passed"] for v in old["checks"].values())):
            raise ValueError("completed OLD paired label must match this state")
        fixture_file = path(directory, f"_fixture_{i:02d}.pkl")
        fixture = dict(model=model, seed=seed_for(model, tape), checkpoint=checkpoint, tree=tree,
            context={k: frame[k] for k in CONTEXT}, cohort_ids=cohort_ids)
        with fixture_file.open("xb") as output:
            pickle.dump(fixture, output)
        targets.append(dict(tape_id=tape_id, frame_index=frame_index, node=tape["node"],
            load=tape["load"], seed=tape["seed"], start=frame["start"], A=original, third=third,
            fixture_file=str(fixture_file), cohort_population=len(cohort_ids), old_outcomes=old["outcomes"]))
        for arm, root in (("A", original), ("third", third)):
            assignments.append(dict(index=len(assignments), target_index=i, tape_id=tape_id, arm=arm, root=root))
        preparation.update(work)
    protocol = {k: prior[k] for k in ("models", "planning_samples", "planning_event_cap", "birth_cohort",
        "environment_arrival_end", "arrival_horizon", "observation_end", "previous_trajectories")}
    protocol.update(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        old_labels_registered_at_utc=prior["registered_at_utc"], full24_registered_at_utc=parent["registered_at_utc"],
        full24_decision=parent["decision"], tick_registered_at_utc=optimized["registered_at_utc"],
        targets=targets, assignments=assignments, expected_suffix_executions=4, preparation_work=dict(preparation),
        state_initialization="Introduce persistent controller at two OLD factual states. Outer RNG/counter/rates/counts follow reconstructed OLD prefix. Inner OLD-current and deep BG use common constructor entry states. Restore this complete three-layer tree, FIFO births/admissions and committed lane ends independently in both arms.",
        method="Each first call consumes the same complete persistent native proposal, then forces only the registered A or third physical action. All later calls freely replan with the same persistent controller. Tick execution preserves native decisions, logical fees and state. Four suffixes only; cached OLD labels are not rerun.",
        interpretation="Conditional third-minus-A effects at two reused development states. Strict mean sign reversal tests continuation-dependent ranking. Compare P90 signs only when both OLD and new paired populations equal the same complete birth cohort. No additive population effect, fresh confirmation or full24 performance gain is established.",
        execution_sources={**prior["execution_sources"], **optimized["execution_sources"],
            **{name: (ROOT / name).read_text() for name in SOURCES}})
    dump(path(directory, "_protocol.json"), protocol)
    print(json.dumps({"registered_at_utc": protocol["registered_at_utc"], "suffix_executions": 4,
        "workers": workers, "states": [(r["tape_id"], r["frame_index"]) for r in targets]}), flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("registered source changed: " + name)
    return protocol


def native_first_frame(frame, decision):
    selected = decision.selection
    fields = asdict(decision)
    del fields["selection"]
    return {**frame, **{k: getattr(selected, k) for k in ("option_indices", "service", "duration",
        "score", "upper_bound", "gap", "score_exact", "canonical_exact")},
        "executed_service": selected.service, "rollout": fields}


def run_shard(index, directory):
    protocol = load_protocol(directory)
    assignment = protocol["assignments"][index]
    target = protocol["targets"][assignment["target_index"]]
    with Path(target["fixture_file"]).open("rb") as source:
        fixture = pickle.load(source)
    model = fixture["model"]
    started = perf_counter()
    with ParallelPlanner(model.options, fixture["seed"], protocol["planning_samples"],
            protocol["planning_event_cap"], workers=protocol["workers"]) as planner:
        selector = PersistentContinuationCrossSelector(planner, root=assignment["root"])
        restoration = restore_planner_tree(planner, fixture["tree"])

        def select(*context):
            decision = selector(*context)
            if selector.calls % 20 == 0:
                print(target["tape_id"], assignment["arm"], "suffix dispatch", selector.calls,
                    "service time", context[3], flush=True)
            return decision

        result = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
            protocol["observation_end"], select, fixture["checkpoint"], policy=assignment["arm"])
        first = result["frames"][0]
        if (not all(ranking.same(first[k], fixture["context"][k]) for k in CONTEXT)
                or first["option_indices"] != (assignment["root"],)
                or selector.work["override_calls"] != 1
                or selector.status["persistent_proposal_calls"] != len(result["frames"])
                or planner.decision_counter != target["frame_index"] + len(result["frames"])):
            raise ValueError("registered context/forced action or free persistent continuation differs")
        checked = {**result, "frames": [native_first_frame(first, selector.first_native_decision)] + result["frames"][1:]}
        if not check_rollout(checked, model, protocol, planner):
            raise ValueError("first native proposal or later persistent decisions failed integrity")
        work = collect_work(selector, result)
        work.update(restoration)
        final_state = capture_planner_tree(planner, work)
        executed = planner.reuse_work
        row = dict(index=index, registered_at_utc=protocol["registered_at_utc"], assignment=assignment,
            execution_checks_passed=True, calls=selector.calls, first_proposal=selector.first_proposal,
            intervention=selector.intervention, final_counters=[s["decision_counter"] for s in final_state["planners"]],
            work=dict(work), status=dict(selector.status), executed_reuse_work=executed,
            execution_host=socket.gethostname(), wall_s=perf_counter() - started)
    jobs_file = path(directory, f"_shard_{index:02d}_jobs.json.gz")
    with jobs_file.open("xb") as raw, gzip.open(raw, "wt") as output:
        json.dump(dict(registered_at_utc=protocol["registered_at_utc"], assignment=assignment,
            jobs=result["jobs"], snapshots=result["snapshots"]),
            output, separators=(",", ":"), allow_nan=False)
    row["jobs_file"] = str(jobs_file)
    dump(path(directory, f"_shard_{index:02d}.json"), row)
    print(json.dumps({"index": index, "checks_passed": True, "wall_s": row["wall_s"]}), flush=True)


def compare_effects(old, new, population):
    old_support = old["snapshots"]["observation_end"]
    new_support = new["snapshots"]["observation_end"]
    p90_comparable = all(v["arrivals"] == v["common_jobs"] == population
        for groups in (old_support, new_support) for v in groups.values())
    comparison = {}
    for metric in METRICS:
        before, after = old[metric], new[metric]
        comparable = metric != "delta_T_common_p90" or p90_comparable
        crossing = ("unavailable" if before is None or after is None or not comparable else
            "old_help_new_loss" if before < 0 < after else
            "old_loss_new_help" if after < 0 < before else
            "both_help" if before < 0 and after < 0 else
            "both_loss" if before > 0 and after > 0 else "contains_zero")
        comparison[metric] = dict(old=before, persistent=after, comparable_population=comparable,
            difference_of_conditional_effects=after - before if comparable and before is not None and after is not None else None,
            crossing=crossing)
    return comparison


def merge(directory):
    protocol = load_protocol(directory)
    rows, jobs, work = [], [], Counter(protocol["preparation_work"])
    for i, assignment in enumerate(protocol["assignments"]):
        row = read(path(directory, f"_shard_{i:02d}.json"))
        if (row["index"] != i or row["registered_at_utc"] != protocol["registered_at_utc"]
                or row["assignment"] != assignment or not row["execution_checks_passed"]):
            raise ValueError("all four matching execution-valid suffixes required")
        with gzip.open(row["jobs_file"], "rt") as source:
            document = json.load(source)
        if (document["registered_at_utc"] != protocol["registered_at_utc"] or document["assignment"] != assignment):
            raise ValueError("suffix births/results belong to a different execution")
        rows.append(row)
        jobs.append(document)
        work.update(row["work"])
    records = []
    for i, target in enumerate(protocol["targets"]):
        A, third = jobs[2 * i:2 * i + 2]
        births = lambda arm: [(j["coordinate"], j["index"], j["arrival"]) for j in arm["jobs"]]
        with Path(target["fixture_file"]).open("rb") as source:
            fixture = pickle.load(source)
        if births(A) != births(third) or births(A) != births(fixture["checkpoint"]):
            raise ValueError("paired suffixes must retain identical canonical births")
        if not ranking.same(rows[2 * i]["first_proposal"], rows[2 * i + 1]["first_proposal"]):
            raise ValueError("both arms must consume the same complete native first proposal")
        outcome = paired_outcomes(A, third, protocol, fixture["model"].workloads)
        comparison = compare_effects(target["old_outcomes"], outcome, target["cohort_population"])
        records.append({k: target[k] for k in ("tape_id", "frame_index", "start", "A", "third", "cohort_population")}
            | dict(persistent_outcomes=outcome, comparison=comparison))
    summary = dict(decision="PERSISTENT_CONTINUATION_CROSS_COMPLETE", execution_checks_passed=True,
        registered_at_utc=protocol["registered_at_utc"], completed_suffixes=4,
        performance_gain_established=False, full24_decision=protocol["full24_decision"],
        state_initialization=protocol["state_initialization"], interpretation=protocol["interpretation"], records=records,
        mean_ranking_crossings=dict(Counter(r["comparison"]["delta_T_mean"]["crossing"] for r in records)),
        p90_ranking_crossings=dict(Counter(r["comparison"]["delta_T_common_p90"]["crossing"] for r in records)))
    dump(path(directory, ".json"), {**summary, "work": dict(work), "suffix_records": rows})
    dump(path(directory, "_summary.json"), summary)
    print(json.dumps({k: summary[k] for k in ("decision", "mean_ranking_crossings", "p90_ranking_crossings")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--shard-index", type=int, choices=range(4))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir, args.workers)
    elif args.merge:
        merge(args.artifact_dir)
    else:
        run_shard(args.shard_index, args.artifact_dir)


if __name__ == "__main__":
    main()
