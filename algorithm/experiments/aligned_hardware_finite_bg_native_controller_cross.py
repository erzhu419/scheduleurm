"""Complete the four native-controller arms at the same three physical states."""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from math import inf, nextafter
from pathlib import Path
import pickle
import socket
from time import perf_counter, process_time

from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout as Planner
from algorithm.experiments import aligned_hardware_finite_bg_first_action_cross as previous
from algorithm.experiments.aligned_hardware_background_competition_continuous import check_rollout
from algorithm.experiments.aligned_hardware_background_root_counterfactual import RootCounterfactualSelector, collect_work, paired_outcomes, sign_counts
from algorithm.experiments.cached_finite_bg_regression import FINITE, CONTEXT, trajectories
from algorithm.experiments.cached_finite_queue_propagation import decomposition, exact
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import _causal_frames

ROOT, ARTIFACTS = previous.ROOT, previous.ARTIFACTS
STEM = "aligned_hardware_finite_BG_native_controller_cross_20261003"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
SOURCES = (SOURCE, "algorithm/experiments/aligned_hardware_background_root_counterfactual.py",
    "algorithm/experiments/aligned_hardware_finite_bg_first_action_cross.py")
read, dump, require, same, births = previous.read, previous.dump, previous.require, previous.same, previous.births
METRICS = previous.METRICS


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def prepare_BG_prefix(planner, factual, anchor, work):
    """Replay only BG's original outer samples; backgrounds use separate seeds."""
    for index, frame, counts, ages, _, released in _causal_frames({**factual, "frames": factual["frames"][:anchor]}):
        info, now = frame["rollout"], frame["start"]
        rates, queue = tuple(count / now if now else 0. for count in counts), tuple(map(len, ages))
        require(planner.decision_counter == index and queue == tuple(frame["queue"])
            and ages == tuple(map(tuple, frame["waiting_ages"]))
            and counts == tuple(frame["observed_arrival_counts"]) == tuple(info["observed_arrival_counts"])
            and rates == tuple(info["estimated_arrival_rates"])
            and info["eligible"] == (0 < sum(queue) <= len(frame["available_before"]) * planner.batch_width), "BG causal prefix differs")
        planner.arrival_rates = rates
        paths = []
        if info["eligible"]:
            for _ in range(planner.sample_count):
                paths.append(sample_planning_arrivals(rates, planner.maximum_duration, planner.rng, planner.event_cap))
                if paths[-1].overflow:
                    break
        events, draws, reads = (sum(getattr(sample, field) if field != "events" else len(sample.events)
            for sample in paths) for field in ("events", "random_draws", "rate_reads"))
        require(len(paths) == info["sample_count"] and events == info["tagged_planning_events"]
            and draws == info["planning_random_draws"] - info["background_planning_random_draws"]
            and reads == info["planning_rate_reads"] - info["background_planning_rate_reads"]
            and (not any(sample.overflow for sample in paths) or info["planning_overflow"]), "BG original prefix samples/fees differ")
        work.update(prefix_cached_frame_reads=1, prefix_birth_reads=released, prefix_fifo_job_reads=sum(queue),
            prefix_service_coordinate_reads=len(queue), prefix_admission_reads=sum(frame["service"]),
            prefix_rate_reads=len(counts), prefix_rate_divisions=len(counts) if now else 0,
            prefix_sample_calls=len(paths), prefix_planning_events=events, prefix_random_draws=draws, prefix_planning_rate_reads=reads)
        planner.decision_counter += 1
        work["prefix_counter_updates"] += 1
    require(planner.decision_counter == anchor, "BG own anchor counter differs")
    return capture_planner_tree(planner, work)


class BGFirstRootSelector(RootCounterfactualSelector):
    """Consume ordinary BG, force one registered root, retain its own state."""
    def __init__(self, planner, root=None):
        super().__init__(planner)
        self.root, self.first_native_decision, self.post_tree = root, None, None

    def __call__(self, *context):
        proposed = super().__call__(*context)
        if self.calls != 1:
            return proposed
        self.first_native_decision = proposed
        self.post_tree = capture_planner_tree(self.planner, self.work)
        if self.root is None:
            return proposed
        root, queue = self.root, context[1]
        require(all(need <= count for need, count in zip(self.planner.requirements[root], queue, strict=True)), "registered finite root infeasible")
        control = self.planner.maxweight(*context)
        self.work["override_root_option_probes"] += control.root_option_probes
        self.work["override_calls"] += 1
        mw, = control.selection.option_indices
        support = Fraction(sum(q * d for q, d in zip(queue, self.planner.requirements[root], strict=True))) / self.planner.durations[root]
        best = Fraction(sum(q * d for q, d in zip(queue, control.selection.service, strict=True))) / self.planner.durations[mw]
        exact_score = support == best
        upper = float(support) if exact_score else control.selection.upper_bound
        if not exact_score and Fraction.from_float(upper) < best:
            upper = nextafter(upper, inf)
        selected = replace(control.selection, option_counts=tuple(int(i == root) for i in range(len(self.planner.options))),
            option_indices=(root,), service=self.planner.requirements[root], duration=float(self.planner.durations[root]),
            score=float(support), upper_bound=upper, gap=max(0., upper - float(support)), exact=False,
            score_exact=exact_score, canonical_exact=False)
        self.intervention = dict(original=proposed.selection.option_indices[0], replacement=root, forced_single_action=True, suffix="ordinary_BG")
        return replace(proposed, selection=selected, changed=root != mw,
            risk_changed_unconstrained_root=root != proposed.unconstrained_root_index,
            chosen_predicted_completion_s=None, chosen_predicted_squared_flow_s2=None)


def register(directory):
    started, cpu = perf_counter(), process_time()
    design, prior = read(path(directory, "_design.json")), previous.load_protocol(directory)
    completed = read(Path(directory) / design["source_summary"])
    require(prior["registered_at_utc"] == completed["registered_at_utc"] == design["source_execution_epoch"]
        and prior["design"]["registered_at_utc"] == design["original_target_design_epoch"]
        and same(design["targets"], prior["design"]["targets"])
        and completed["execution_checks_passed"] and completed["completed_new_suffixes"] == 3
        and prior["planning_samples"] == design["planning_samples"] == 8
        and prior["planning_event_cap"] == design["event_cap"] == 128, "completed unchanged three finite crosses required")
    science = read(Path(directory) / (FINITE + "_protocol.json"))["scientific_protocol"]
    require(science["registered_at_utc"] == design["finite_scientific_epoch"], "finite source epoch differs")
    bg_runs = trajectories(ROOT / science["previous_trajectories"], design["BG_epoch"])
    targets, preparation = [], Counter(BG_gzip_reads=1)
    for index, target in enumerate(design["targets"]):
        old_target = prior["targets"][index]
        with Path(old_target["fixture_file"]).open("rb") as source:
            old = pickle.load(source)
        row = read(previous.path(directory, f"_shard_{index:02d}.json"))
        require(row["registered_at_utc"] == prior["registered_at_utc"] and row["tape_id"] == target["tape_id"]
            and row["execution_checks_passed"], "cached B/F suffix identity differs")
        with gzip.open(row["trajectory_file"], "rt") as source:
            saved = json.load(source)
        require(saved["registered_at_utc"] == prior["registered_at_utc"] and saved["tape_id"] == target["tape_id"], "cached B/F trajectory identity differs")
        bf, bg = saved["result"], bg_runs[target["tape_id"]]["arrival_rollout"]
        anchor, frame = target["frame_index"], bg["frames"][target["frame_index"]]
        checkpoint = prepare_checkpoint(bg, anchor)
        require(births(bg) == births(bf) == births(old["cached_finite"])
            and same(checkpoint, old["checkpoint"])
            and frame["option_indices"] == [target["replacement_BG_root"]]
            and all(same(frame[key], target["context"][key]) for key in CONTEXT), "BG own physical checkpoint/root differs")
        fixture_file = path(directory, f"_fixture_{index:02d}.pkl")
        with fixture_file.open("xb") as output:
            pickle.dump(dict(model=old["model"], seed=old["seed"], checkpoint=checkpoint,
                BG_prefix={"jobs": bg["jobs"], "frames": bg["frames"][:anchor + 1]}, BG_frame=frame,
                BG_suffix_frames=bg["frames"][anchor:], BG_suffix_lane_events=bg["lane_events"][anchor:], BG_last_frame_end=bg["last_frame_end"],
                cached_arms={"FF": old["cached_finite"], "BF": {key: bf[key] for key in ("jobs", "snapshots")},
                    "BB": {key: bg[key] for key in ("jobs", "snapshots")}},
                previous_finite_outcomes=completed["records"][index]["outcomes"]), output)
        preparation.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
        preparation["cached_BF_gzip_reads"] += 1
        targets.append(dict(**target, fixture_file=str(fixture_file)))
    sources = {**prior["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}}
    for name, literal in sources.items():
        require((ROOT / name).read_text() == literal, "frozen source changed: " + name)
    protocol = {key: prior[key] for key in ("birth_cohort", "arrival_horizon", "observation_end", "environment_arrival_end", "planning_samples", "planning_event_cap", "clock", "full24_decision")}
    protocol.update(registered_at_utc=datetime.now(timezone.utc).isoformat(), design=design, targets=targets,
        previous_registered_at_utc=prior["registered_at_utc"], execution_sources=sources, null_target_index=1,
        expected_scientific_suffixes=3, expected_validation_suffixes=1, expected_reused_complete_arms=9,
        preparation_work=dict(preparation), registration_cpu_s=process_time() - cpu, registration_wall_s=perf_counter() - started)
    dump(path(directory, "_protocol.json"), protocol)
    print("registered BG science_suffixes=3 validation_suffixes=1 reused_arms=9", flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    require(read(path(directory, "_design.json")) == protocol["design"], "registered design changed")
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_shard(index, directory, *, null=False):
    protocol = load_protocol(directory)
    require(not null or index == protocol["null_target_index"] == 1, "BG null fixed to seed42")
    if not null:
        checked = read(path(directory, "_null_seed42.json"))
        require(checked["registered_at_utc"] == protocol["registered_at_utc"] and checked["cached_factual_suffix_reproduced"], "BG null PASS required before science")
    target = protocol["targets"][index]
    suffix = "_null_seed42" if null else f"_shard_{index:02d}"
    trajectory_file = path(directory, suffix + "_trajectories.json.gz")
    require(not trajectory_file.exists(), "completed BG suffix already saved; inspect before simulating again")
    with Path(target["fixture_file"]).open("rb") as source:
        fixture = pickle.load(source)
    model, work = fixture["model"], Counter()
    started, cpu = perf_counter(), process_time()
    planner = Planner(model.options, fixture["seed"], 8, 128)
    before = prepare_BG_prefix(planner, fixture["BG_prefix"], target["frame_index"], work)
    selector = BGFirstRootSelector(planner, None if null else target["factual_finite_root"])

    def select(*context):
        decision = selector(*context)
        if selector.calls == 1:
            require(same(previous.native_first_frame(fixture["BG_frame"], selector.first_native_decision), fixture["BG_frame"])
                and all(same(context[j], target["context"][key]) for j, key in ((1, "queue"), (2, "available_before"),
                    (3, "start"), (5, "waiting_ages"), (6, "observed_arrival_counts")))
                and context[4] == target["context"]["lane_ids"][0]
                and capture_planner_tree(planner, work) == selector.post_tree, "native BG proposal/context/fees/post-state differ")
            print(target["tape_id"], "own BG anchor verified", flush=True)
        if selector.calls % 20 == 0:
            print(target["tape_id"], "BG suffix dispatch", selector.calls, "service time", context[3], flush=True)
        return decision

    result = resume_single_action_trajectory(model, protocol["environment_arrival_end"], protocol["observation_end"],
        select, fixture["checkpoint"], policy="null_BG" if null else "forced_F_then_BG")
    with trajectory_file.open("xb") as raw, gzip.open(raw, "wt") as output:
        json.dump(dict(registered_at_utc=protocol["registered_at_utc"], tape_id=target["tape_id"], result=result), output, separators=(",", ":"), allow_nan=False)
    require(selector.work["override_calls"] == int(not null) and selector.status["BG_proposal_calls"] == len(result["frames"])
        and result["frames"][0]["option_indices"] == (target["replacement_BG_root"] if null else target["factual_finite_root"],)
        and planner.decision_counter == target["frame_index"] + len(result["frames"])
        and births(result) == births(fixture["checkpoint"]), "single F override/free BG suffix/births differ")
    require(all(old == new for old, new in zip(fixture["checkpoint"]["jobs"], result["jobs"], strict=True)
        if old["start"] is not None), "committed BG prefix changed")
    checked = {**result, "frames": [previous.native_first_frame(result["frames"][0], selector.first_native_decision)] + result["frames"][1:]}
    require(check_rollout(checked, model, protocol, planner), "BG native anchor/continued proposals invalid")
    after900 = target["context"]["start"] > protocol["arrival_horizon"]
    if after900:
        require(same(trajectory_snapshot(result["jobs"], protocol["arrival_horizon"], len(model.workloads)),
            trajectory_snapshot(fixture["cached_arms"]["BB"]["jobs"], protocol["arrival_horizon"], len(model.workloads))), "after900 F root changed actual snapshot900")
    if null:
        require(same(result["frames"], fixture["BG_suffix_frames"])
            and result["jobs"] == fixture["cached_arms"]["BB"]["jobs"]
            and same(result["snapshots"], fixture["cached_arms"]["BB"]["snapshots"])
            and same(result["lane_events"], fixture["BG_suffix_lane_events"])
            and result["last_frame_end"] == fixture["BG_last_frame_end"], "complete BG null frames/fees/jobs/raw snapshots differ")
    work.update(collect_work(selector, result))
    final = capture_planner_tree(planner, work)
    row = dict(index=index, tape_id=target["tape_id"], registered_at_utc=protocol["registered_at_utc"], validation_suffix=null,
        execution_checks_passed=True, cached_factual_suffix_reproduced=null, first_native_proposal_and_fees_verified=True,
        complete_post_proposal_state_retained=True, snapshot900_unchanged=after900, calls=selector.calls,
        first_proposal=selector.first_proposal, intervention=selector.intervention, work=dict(work),
        before_counter=before["planners"][0]["decision_counter"], final_counter=final["planners"][0]["decision_counter"],
        caller_cpu_s=process_time() - cpu, worker_cpu_s=0., execution_host=socket.gethostname(),
        wall_s=perf_counter() - started, trajectory_file=str(trajectory_file))
    dump(path(directory, suffix + ".json"), row)
    print("complete", suffix, "ordinary BG calls", selector.calls, flush=True)


def component_contrast(negative, positive, protocol, workloads):
    jobs = [{(j["coordinate"], j["index"]): j for j in arm["jobs"]} for arm in (negative, positive)]
    lower, upper = protocol["birth_cohort"]
    ids = {key for key, job in jobs[0].items() if lower <= job["arrival"] <= upper}

    def group(selected, cutoff):
        public, _ = decomposition(selected, jobs[1], jobs[0], exact(cutoff))
        return dict(jobs=public["jobs"], positive=public["parent"], negative=public["native"], delta=public["delta_parent_minus_native"])

    return {key: dict(all_cohort=group(ids, protocol[key]), per_coordinate=[dict(coordinate=c, workload=workload,
        **group({identity for identity in ids if identity[0] == c}, protocol[key])) for c, workload in enumerate(workloads)])
        for key in ("arrival_horizon", "observation_end")}


def merge(directory):
    protocol, records, work, cpu = load_protocol(directory), [], Counter(), Counter()
    null = read(path(directory, "_null_seed42.json"))
    require(null["registered_at_utc"] == protocol["registered_at_utc"] and null["cached_factual_suffix_reproduced"], "registered BG null PASS required")
    for index, target in enumerate(protocol["targets"]):
        row = read(path(directory, f"_shard_{index:02d}.json"))
        require(row["index"] == index and row["tape_id"] == target["tape_id"]
            and row["registered_at_utc"] == protocol["registered_at_utc"] and row["execution_checks_passed"], "all three registered BG suffixes required")
        with Path(target["fixture_file"]).open("rb") as source:
            fixture = pickle.load(source)
        with gzip.open(row["trajectory_file"], "rt") as source:
            saved = json.load(source)
        require(saved["registered_at_utc"] == protocol["registered_at_utc"] and saved["tape_id"] == target["tape_id"], "BG science trajectory epoch differs")
        arms, model = {**fixture["cached_arms"], "FB": saved["result"]}, fixture["model"]
        require(all(births(arm) == births(arms["FF"]) for arm in arms.values()), "four-arm canonical births differ")
        effects = {"finite": paired_outcomes(arms["FF"], arms["BF"], protocol, model.workloads),
            "BG": paired_outcomes(arms["FB"], arms["BB"], protocol, model.workloads)}
        require(same(effects["finite"], fixture["previous_finite_outcomes"]), "cached finite root effect recomputation differs")
        lower, upper = protocol["birth_cohort"]
        snapshots = {key: {name: trajectory_snapshot([j for j in arm["jobs"] if lower <= j["arrival"] <= upper],
            protocol[key], len(model.workloads)) for name, arm in arms.items()} for key in ("arrival_horizon", "observation_end")}
        complete = {key: all(group["completed_jobs"] == group["arrivals"] for group in values.values()) for key, values in snapshots.items()}
        interaction = {key: effects["finite"][key] - effects["BG"][key]
            if effects["finite"][key] is not None and effects["BG"][key] is not None
                and (key != "delta_T_common_p90" or complete["observation_end"]) else None for key in METRICS}
        records.append(dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "factual_finite_root", "replacement_BG_root")},
            root_effects=effects, interaction_finite_minus_BG=interaction, all_four_complete_cohort=complete,
            strict_mean_ranking_reversal=effects["finite"]["delta_T_mean"] * effects["BG"]["delta_T_mean"] < 0,
            four_arm_snapshots=snapshots, waiting_service={"finite": component_contrast(arms["FF"], arms["BF"], protocol, model.workloads),
                "BG": component_contrast(arms["FB"], arms["BB"], protocol, model.workloads)}))
        work.update(row["work"])
        cpu.update({key: row[key] for key in ("caller_cpu_s", "worker_cpu_s")})
    summary = dict(registered_at_utc=protocol["registered_at_utc"], decision="FINITE_BG_NATIVE_CONTROLLER_CROSS_COMPLETE",
        execution_checks_passed=True, completed_new_suffixes=3, completed_validation_suffixes=1, reused_complete_arms=9,
        records=records, signs={controller: {key: sign_counts([r["root_effects"][controller][key] for r in records]) for key in METRICS}
            for controller in ("finite", "BG")}, scientific_work=dict(work), scientific_cpu_s=dict(cpu), validation_work=null["work"],
        validation_cpu_s={key: null[key] for key in ("caller_cpu_s", "worker_cpu_s")}, preparation_work=protocol["preparation_work"],
        registration_cpu_s=protocol["registration_cpu_s"], registration_wall_s=protocol["registration_wall_s"],
        full24_decision=protocol["full24_decision"], performance_gain_established=False,
        root_effect_orientation=protocol["design"]["root_effect_orientation"], interpretation=protocol["design"]["interpretation"])
    dump(path(directory, "_summary.json"), summary)
    print("complete four-arm native-controller crossing; science=3 validation=1 reused=9", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-null", action="store_true")
    mode.add_argument("--shard-index", type=int, choices=range(3))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir)
    elif args.merge:
        merge(args.artifact_dir)
    else:
        run_shard(1 if args.run_null else args.shard_index, args.artifact_dir, null=args.run_null)


if __name__ == "__main__":
    main()
