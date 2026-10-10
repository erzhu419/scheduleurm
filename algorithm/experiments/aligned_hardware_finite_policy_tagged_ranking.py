"""Compare two roots on the same tagged population with free finite recourse."""
import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import pickle
import socket
from time import perf_counter

from algorithm.parallel_root_lane_rollout import ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout as Planner
from algorithm.experiments import aligned_hardware_persistent_action_controller_cross as crossing
from algorithm.experiments.aligned_hardware_persistent_continuation_cross import native_first_frame
from algorithm.experiments.persistent_parent_incumbent_scores import _RecordingScores, _RecordingOriginalRisk
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree
from algorithm.experiments.finite_policy_tagged_rollout import evaluate_root_sample

ROOT, ARTIFACTS = crossing.ROOT, crossing.ARTIFACTS
STEM = "aligned_hardware_finite_policy_tagged_ranking_20261003"
SOURCES = ("algorithm/experiments/aligned_hardware_finite_policy_tagged_ranking.py",
    "algorithm/experiments/finite_policy_tagged_rollout.py")
read, dump, require, same = crossing.read, crossing.dump, crossing.require, crossing.same


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


class RecordingFiniteScores(_RecordingScores, Planner, _RecordingOriginalRisk):
    """Retain the existing exact scores and paths without another scoring call."""
    def score_background_roots(self, *context):
        self.recorded_paths = tuple(context[-1])
        self.recording_work["recorded_path_references"] += len(self.recorded_paths)
        return super().score_background_roots(*context)


def register(directory, workers, prepare_workers):
    prior = crossing.load_protocol(directory)
    completed = read(crossing.path(directory, "_summary.json"))
    require(completed["registered_at_utc"] == prior["registered_at_utc"]
        and completed["execution_checks_passed"] and completed["completed_new_suffixes"] == 3
        and completed["reused_complete_arms"] == 9
        and completed["full24_decision"] == "PERSISTENT_CURRENT_POLICY_DEVELOPMENT_HOLD"
        and prior["planning_samples"] == 8 and prior["planning_event_cap"] == 128,
        "completed fixed-three controller crossings and original budget required")
    targets = []
    for index, (target, record) in enumerate(zip(prior["targets"], completed["records"], strict=True)):
        require(target["index"] == index and record["tape_id"] == target["tape_id"]
            and record["frame_index"] == target["frame_index"], "retain all three registered states")
        targets.append({**{key: target[key] for key in ("index", "tape_id", "seed", "frame_index", "parent_root", "native_root")},
            "source_fixture": target["fixture_file"],
            "birth_cohort_root_effects": {name: {key: outcome[key] for key in crossing.METRICS}
                for name, outcome in record["root_effects"].items()},
            "complete_four_arm_common_cohort_p90": record["complete_four_arm_common_cohort_p90"]})
    assignments = [dict(index=16 * target["index"] + 8 * slot + sample,
        target_index=target["index"], tape_id=target["tape_id"], root=target[role + "_root"],
        role=role, sample_index=sample)
        for target in targets for slot, role in enumerate(("parent", "native")) for sample in range(8)]
    protocol = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        prepare_workers=prepare_workers, planning_samples=8, planning_event_cap=128,
        previous_registered_at_utc=prior["registered_at_utc"], full24_decision=prior["full24_decision"],
        targets=targets, assignments=assignments, native_anchor_proposals=3, root_sample_units=48,
        true_service_suffix_executions=0,
        method="At three fixed GPU2/.95 actual first divergences recover each finite controller's own exact prefix. Consume one complete native anchor proposal, matching every cached action/fee. Retain its original eight prefix paths, exact original-risk/main scores and complete before/after tree. Each of48 root/sample units evaluates the native two-OLD-current-then-MW approximation and freely continuing finite policy on the identical visible-plus-prefix tagged jobs and fixed native Poisson background stream. Free finite starts from the complete consumed anchor after-tree; every later dispatch, including idle, consumes its complete native proposal. Original risk set,8/cap128 and two registered roots are unchanged.",
        overflow="Keep every fixed root/sample, including partial cost/work and overflow. Any overflowing free sample makes that root's full8 score unavailable; pair ranking requires both roots complete. No discard, resample, raised cap or alternative anchor.",
        interpretation="Offline same-population ranking diagnostic. Native approximate8-sample exact S1/S2 must reproduce stored finite main scores. Free-finite scores compare the same tagged IDs; all later background jobs affect FIFO but remain unscored. True[300,900] birth-cohort labels are contextual separate observations, not unbiased score-error targets. This is not a production chooser or full-policy gain; full24 HOLD, sealed confirmation and paper unchanged.",
        execution_sources={**prior["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}})
    dump(path(directory, "_protocol.json"), protocol)
    print(f"registered_at_utc={protocol['registered_at_utc']} prepares=3 units=48 workers={workers} prepare_workers={prepare_workers}", flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def prepare(index, directory):
    protocol = load_protocol(directory)
    target, work = protocol["targets"][index], Counter()
    with Path(target["source_fixture"]).open("rb") as source:
        source_fixture = pickle.load(source)
    model, frame = source_fixture["model"], source_fixture["finite_frame"]
    context = dict(queue=tuple(frame["queue"]), available=tuple(frame["available_before"]), now=frame["start"],
        lane=frame["lane_ids"][0], ages=tuple(map(tuple, frame["waiting_ages"])),
        counts=tuple(frame["observed_arrival_counts"]))
    started = perf_counter()
    with RecordingFiniteScores(model.options, source_fixture["seed"], 8, 128,
            workers=protocol["prepare_workers"]) as planner:
        last, before = crossing.prepare_finite_prefix(planner, source_fixture["finite_prefix"], target["frame_index"], work)
        decision = planner(model.options, context["queue"], context["available"], context["now"],
            context["lane"], context["ages"], context["counts"])
        require(same(native_first_frame(frame, decision), frame), "finite anchor action/full native fees did not reproduce")
        original, scored, paths = planner.recorded_original_scores, planner.recorded_scores, planner.recorded_paths
        require(original is not None and scored is not None
            and not original.background_rollout_overflow and not scored.background_rollout_overflow
            and len(paths) == 8 and not any(path.overflow for path in paths)
            and sum(len(path.events) for path in paths) == frame["rollout"]["tagged_planning_events"],
            "complete native scores and the same original8 tagged paths required")
        mw = decision.incumbent_maxweight_root_index
        risk = [root for root in original.costs if original.squared_costs[root] <= original.squared_costs[mw]]
        require(target["parent_root"] in risk and target["native_root"] in risk,
            "both fixed roots must remain inside the original native risk set")
        after = capture_planner_tree(planner, work)
        require(before["planners"][0]["decision_counter"] == target["frame_index"]
            and after["planners"][0]["decision_counter"] == target["frame_index"] + 1
            and after["planners"][0]["rng_state"] == planner.recorded_post_prefix_state["rng_state"],
            "complete consumed anchor state differs from native post-prefix state")
        fixture_file = path(directory, f"_prepared_{index:02d}.pkl")
        with fixture_file.open("xb") as output:
            pickle.dump(dict(registered_at_utc=protocol["registered_at_utc"], tape_id=target["tape_id"],
                options=model.options, seed=source_fixture["seed"], paths=paths, before_tree=before,
                after_tree=after, context=context, native_scores=scored, original_scores=original), output)
        scores = {role: dict(root=target[role + "_root"], S1=scored.costs[target[role + "_root"]],
            S2=scored.squared_costs[target[role + "_root"]]) for role in ("parent", "native")}
        row = dict(index=index, tape_id=target["tape_id"], registered_at_utc=protocol["registered_at_utc"],
            execution_checks_passed=True, first_finite_prefix_verified=True, last_touched_frames=last,
            root_scores=scores, original_risk_roots=risk, path_events=[len(path.events) for path in paths],
            tagged_populations=[sum(context["queue"]) + len(path.events) for path in paths],
            fixture_file=str(fixture_file), native_logical_work=frame["rollout"],
            recovery_work=dict(work), recording_work=dict(planner.recording_work),
            total_physical_work=planner.reuse_work, execution_host=socket.gethostname(), wall_s=perf_counter() - started)
    dump(path(directory, f"_prepare_{index:02d}.json"), jsonable(row))
    print(f"prepare_complete index={index} finite_anchor_verified paths=8 roots={target['parent_root']},{target['native_root']}", flush=True)


def run_shard(index, directory):
    protocol = load_protocol(directory)
    assignment = protocol["assignments"][index]
    prepared = read(path(directory, f"_prepare_{assignment['target_index']:02d}.json"))
    require(prepared["registered_at_utc"] == protocol["registered_at_utc"]
        and prepared["execution_checks_passed"] and prepared["tape_id"] == assignment["tape_id"],
        "registered native preparation required")
    with Path(prepared["fixture_file"]).open("rb") as source:
        fixture = pickle.load(source)
    require(fixture["registered_at_utc"] == protocol["registered_at_utc"]
        and fixture["tape_id"] == assignment["tape_id"], "prepared fixture identity differs")
    started = perf_counter()
    print(f"unit_start index={index} role={assignment['role']} root={assignment['root']} sample={assignment['sample_index']}", flush=True)
    result = evaluate_root_sample(fixture["options"], fixture["seed"], fixture["paths"],
        fixture["before_tree"], fixture["after_tree"], fixture["context"],
        assignment["root"], assignment["sample_index"], workers=protocol["workers"])
    require(result["root"] == assignment["root"] and result["sample_index"] == assignment["sample_index"]
        and len(result["tagged_population"]) == prepared["tagged_populations"][assignment["sample_index"]]
        and result["approx"]["complete"] and not result["approx"]["overflow"],
        "fixed root/sample or native tagged population differs")
    row = dict(index=index, registered_at_utc=protocol["registered_at_utc"], assignment=assignment,
        execution_checks_passed=True, result=result, execution_host=socket.gethostname(), wall_s=perf_counter() - started)
    dump(path(directory, f"_shard_{index:02d}.json"), jsonable(row))
    print(f"unit_complete index={index} approx_overflow={result['approx']['overflow']} free_overflow={result['free']['overflow']}", flush=True)


def population_totals(values):
    """Retain exact source/coordinate costs on all eight registered populations."""
    def total(groups):
        return {name: sum((Fraction(*group[name]) for group in groups), Fraction(0))
            if name in ("cost", "squared") else sum(group[name] for group in groups)
            for name in ("cost", "squared", "tagged_total", "evaluations")}
    populations = [value["population"] for value in values]
    return dict(**total(populations), tagged_remaining=sum(p["tagged_remaining"] for p in populations),
        sources={source: total([p["sources"][source] for p in populations]) for source in ("visible", "prefix")},
        per_coordinate=[dict(coordinate=index, **total([p["per_coordinate"][index] for p in populations]),
            sources={source: total([p["per_coordinate"][index]["sources"][source] for p in populations])
                for source in ("visible", "prefix")}) for index in range(len(populations[0]["per_coordinate"]))])


def merge(directory):
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_shard_{index:02d}.json")) for index in range(48)]
    for assignment, row in zip(protocol["assignments"], rows, strict=True):
        require(row["index"] == assignment["index"] and row["assignment"] == assignment
            and row["registered_at_utc"] == protocol["registered_at_utc"] and row["execution_checks_passed"],
            "all48 registered root/sample units required")
    records = []
    for target in protocol["targets"]:
        prepared = read(path(directory, f"_prepare_{target['index']:02d}.json"))
        require(prepared["registered_at_utc"] == protocol["registered_at_utc"]
            and prepared["execution_checks_passed"] and prepared["tape_id"] == target["tape_id"],
            "registered complete native preparation differs")
        roots, cells = {}, rows[16 * target["index"]:16 * (target["index"] + 1)]
        for sample in range(8):
            paired = [cells[8 * slot + sample]["result"] for slot in range(2)]
            require(paired[0]["environment"] == paired[1]["environment"]
                and paired[0]["tagged_population"] == paired[1]["tagged_population"]
                and len(paired[0]["tagged_population"]) == prepared["tagged_populations"][sample],
                "two registered roots must share the same fixed environment and tagged IDs")
            streams = [result[method]["background_events"] for result in paired for method in ("approx", "free")]
            longest = max(streams, key=len)
            require(all(stream == longest[:len(stream)] for stream in streams),
                "root/method background birth prefixes differ")
            for result in paired:
                require(all(result[method]["population"]["tagged_population"] == result["tagged_population"]
                    for method in ("approx", "free")) and result["paired_complete"] ==
                    (result["approx"]["complete"] and result["free"]["complete"]),
                    "paired population or completion proof differs")
        for slot, role in enumerate(("parent", "native")):
            samples = cells[8 * slot:8 * (slot + 1)]
            scores = {}
            for continuation in ("approx", "free"):
                values = [row["result"][continuation] for row in samples]
                total = {metric: sum((Fraction(*value[field]) for value in values), Fraction(0))
                    for metric, field in (("S1", "cost"), ("S2", "squared"))}
                complete = all(value["complete"] and not value["overflow"] for value in values)
                if continuation == "approx":
                    require(complete and all(total[metric] == Fraction(*prepared["root_scores"][role][metric])
                        for metric in ("S1", "S2")), "8-sample approximate costs did not reproduce exact native root scores")
                scores[continuation] = {**total, "available": complete,
                    "S1_mean": float(total["S1"] / 8) if complete else None,
                    "S2_mean": float(total["S2"] / 8) if complete else None,
                    "overflow_samples": [row["assignment"]["sample_index"] for row, value in zip(samples, values, strict=True) if value["overflow"]],
                    "population_totals": population_totals(values),
                    "work": {name: sum(value[name] for value in values) for name in
                        ("calls", "idle_calls", "evaluations", "admitted_reads", "tag_reads", "background_admissions", "worker_cpu_s", "worker_wall_s")}}
            roots[role] = dict(root=target[role + "_root"], scores=scores)
        comparisons = {}
        for continuation in ("approx", "free"):
            p, n = (roots[role]["scores"][continuation] for role in ("parent", "native"))
            available = p["available"] and n["available"]
            delta = {metric: (p[metric] - n[metric]) / 8 if available else None for metric in ("S1", "S2")}
            comparisons[continuation] = dict(available=available,
                parent_minus_native_S1_mean=float(delta["S1"]) if available else None,
                parent_minus_native_S2_mean=float(delta["S2"]) if available else None,
                exact_parent_minus_native_mean=delta,
                S1_preference=("parent" if delta["S1"] < 0 else "native" if delta["S1"] > 0 else "tie") if available else "unavailable")
        a, f = (comparisons[name] for name in ("approx", "free"))
        records.append(dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "parent_root", "native_root", "birth_cohort_root_effects")},
            root_scores=roots, comparisons=comparisons, original_risk_roots=prepared["original_risk_roots"],
            tagged_populations=prepared["tagged_populations"], exact_native_approx_reproduction_passed=True,
            shared_tagged_population_and_environment_passed=True,
            native_anchor_logical_work=prepared["native_logical_work"],
            approximate_two_root_shared_suffix_work=[
                [max(cells[8 * slot + sample]["result"]["approx"]["suffix_work"][0][field] for slot in range(2))
                    for field in range(4)] for sample in range(8)],
            strict_S1_ranking_reversal=(a["exact_parent_minus_native_mean"]["S1"] * f["exact_parent_minus_native_mean"]["S1"] < 0)
                if f["available"] else None))
    summary = dict(registered_at_utc=protocol["registered_at_utc"], decision="FINITE_POLICY_TAGGED_RANKING_COMPLETE",
        execution_checks_passed=True, native_anchor_proposals=3, completed_root_sample_units=48,
        true_service_suffix_executions=0, full24_decision=protocol["full24_decision"], performance_gain_established=False,
        records=records, interpretation=protocol["interpretation"], overflow_rule=protocol["overflow"])
    dump(path(directory, "_summary.json"), jsonable(summary))
    print("complete tagged_rankings=3 units=48 exact_native_reproduction=passed full24=HOLD", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--prepare-index", type=int, choices=range(3))
    mode.add_argument("--shard-index", type=int, choices=range(48))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prepare-workers", type=int, default=22)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir, args.workers, args.prepare_workers)
    elif args.prepare_index is not None:
        prepare(args.prepare_index, args.artifact_dir)
    elif args.merge:
        merge(args.artifact_dir)
    else:
        run_shard(args.shard_index, args.artifact_dir)


if __name__ == "__main__":
    main()
