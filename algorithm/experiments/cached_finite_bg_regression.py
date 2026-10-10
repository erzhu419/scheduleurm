"""Decompose all24 cached finite/BG development tapes without any simulation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from math import isclose
from pathlib import Path
from statistics import mean

from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics
from algorithm.experiments.cached_finite_queue_propagation import components
from algorithm.experiments.oldest_fifo_start_opportunity import start_opportunity


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
FINITE = "aligned_hardware_parallel_root_sharded_20261001"
BG = "aligned_hardware_background_competition_continuous_development_20260930"
STEM = "aligned_hardware_cached_finite_bg_regression_20261003"
CONTEXT = ("start", "queue", "waiting_ages", "available_before", "observed_arrival_counts", "lane_ids")
METRICS = ("restricted_mean_flow_s", "common_mean_flow_s", "common_p90_flow_s", "unfinished_jobs")


def read(file):
    return json.loads(Path(file).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trajectories(file, epoch):
    with gzip.open(file, "rt") as source:
        saved = json.load(source)
    require(saved["registered_at_utc"] == epoch, "cached trajectory registration differs")
    return {run["tape_id"]: run["policies"] for run in saved["runs"]}


def shape(option):
    if option is None:
        return "idle"
    counts = option["requirements"]
    return "loaded" if option["loaded"] else "dedicated_batch" if sum(counts) > 1 else "dedicated_p1"


def action(frame, index, options, workloads):
    option = options[index] if index is not None else None
    requirements = option["requirements"] if option else [0] * len(frame["queue"])
    duration = option["duration"] if option else 1.0
    opportunity = start_opportunity(requirements, duration, frame["available_before"],
        frame["start"], frame["lane_ids"][0], frame["waiting_ages"])
    return dict(option_index=index, shape=shape(option),
        action_id=option["action_id"] if option else "idle", requirements=requirements,
        duration=duration, completion_offsets=option["completion_offsets"] if option else [],
        oldest_workloads=[workloads[c] for c in opportunity["oldest_coordinates"]],
        start_opportunity=opportunity)


def first_divergence(finite, bg, options, workloads):
    old, new = bg["frames"], finite["frames"]
    for index, (a, b) in enumerate(zip(old, new)):
        require(all(a[key] == b[key] for key in CONTEXT), "physical context differs before first action divergence")
        if a["option_indices"] != b["option_indices"]:
            return dict(frame_index=index, context={key: b[key] for key in CONTEXT},
                BG=action(a, a["option_indices"][0], options, workloads),
                finite=action(b, b["option_indices"][0], options, workloads))
        require(a["service"] == b["service"] and a["duration"] == b["duration"],
            "identical cached actions have different physical service")
    require(len(old) == len(new), "equal-action trajectories have unequal lengths")
    return None


def own_context_actions(finite, options, workloads):
    counts, transitions, requirements, opportunities = Counter(), Counter(), Counter(), Counter()
    for frame in finite["frames"]:
        info = frame["rollout"]
        if not info["candidate_count"] or info["planning_overflow"]:
            continue
        chosen, incumbent = frame["option_indices"][0], info["incumbent_root_index"]
        counts["completed_new_stage_frames"] += 1
        counts["changed_from_own_context_BG"] += chosen != incumbent
        transitions[shape(options[incumbent]) + "->" + shape(options[chosen])] += 1
        requirements[str(options[incumbent]["requirements"]) + "->" + str(options[chosen]["requirements"])] += 1
        if chosen != incumbent:
            a = action(frame, incumbent, options, workloads)["start_opportunity"]["oldest_start_sum"]
            b = action(frame, chosen, options, workloads)["start_opportunity"]["oldest_start_sum"]
            opportunities["later" if b > a else "earlier" if b < a else "equal"] += 1
    return dict(counts=dict(counts), chosen_vs_own_context_BG_shapes=dict(transitions),
        chosen_vs_own_context_BG_requirements=dict(requirements),
        changed_root_oldest_start_opportunity=dict(opportunities))


def pair_components(finite, bg, ids, cutoff, total_jobs):
    totals = {}
    for name, jobs in (("finite", finite), ("BG", bg)):
        values = [components(jobs[key], Fraction(cutoff)) for key in ids]
        waiting = sum((value[0] for value in values), Fraction())
        service = sum((value[1] for value in values), Fraction())
        totals[name] = dict(waiting=waiting, service=service, flow=waiting + service)
    delta = {key: totals["finite"][key] - totals["BG"][key] for key in totals["finite"]}
    return dict(jobs=len(ids),
        means={name: {key + "_s": float(value / len(ids)) if ids else None
            for key, value in values.items()} for name, values in totals.items()},
        delta_mean={key + "_s": float(value / len(ids)) if ids else None for key, value in delta.items()},
        whole_cohort_mean_contribution={key + "_s": float(value / total_jobs) for key, value in delta.items()})


def tails(finite, bg, ids, cutoff):
    unfinished = {name: {key for key in ids if jobs[key]["completion"] is None
        or jobs[key]["completion"] > cutoff} for name, jobs in (("finite", finite), ("BG", bg))}
    values = {}
    for name, jobs in (("finite", finite), ("BG", bg)):
        queued = sum(jobs[key]["start"] is None or jobs[key]["start"] > cutoff for key in unfinished[name])
        values[name] = dict(unfinished=len(unfinished[name]), queued=queued,
            inflight=len(unfinished[name]) - queued)
    exclusive = {}
    for name, other, jobs in (("finite", "BG", finite), ("BG", "finite", bg)):
        ids_only = unfinished[name] - unfinished[other]
        queued = sum(jobs[key]["start"] is None or jobs[key]["start"] > cutoff for key in ids_only)
        exclusive[name + "_only_unfinished_states"] = dict(queued=queued, inflight=len(ids_only) - queued)
    return dict(**values, **exclusive, finite_only_unfinished=len(unfinished["finite"] - unfinished["BG"]),
        BG_only_unfinished=len(unfinished["BG"] - unfinished["finite"]))


def snapshot_matches(actual, expected):
    return all(isclose(actual[role][key], expected[role][key], rel_tol=1e-12, abs_tol=1e-10)
        for role in ("maxweight", "arrival_rollout") for key in (*METRICS, "common_jobs"))


def aggregate(records):
    result = dict(paired_tapes=len(records),
        mean_regressed_tapes=sum(r["cutoffs"]["observation_end"]["delta_metrics"]["restricted_mean_flow_s"] > 0 for r in records),
        first_divergent_tapes=sum(r["first_divergence"] is not None for r in records), cutoffs={})
    for cutoff in ("arrival_horizon", "observation_end"):
        selected = [r["cutoffs"][cutoff] for r in records]
        result["cutoffs"][cutoff] = dict(
            delta_metrics={key: (sum if key == "unfinished_jobs" else mean)(r["delta_metrics"][key] for r in selected)
                for key in METRICS},
            delta_mean_components={key: mean(r["components"]["delta_mean"][key] for r in selected)
                for key in ("waiting_s", "service_s", "flow_s")},
            tails={name: {key: sum(r["tails"][name][key] for r in selected)
                for key in ("unfinished", "queued", "inflight")} for name in ("finite", "BG")},
            exclusive_unfinished={key: sum(r["tails"][key] for r in selected)
                for key in ("finite_only_unfinished", "BG_only_unfinished")},
            exclusive_unfinished_states={name: {key: sum(r["tails"][name][key] for r in selected)
                for key in ("queued", "inflight")} for name in ("finite_only_unfinished_states", "BG_only_unfinished_states")},
            per_coordinate=[dict(coordinate=c, workload=records[0]["workloads"][c],
                delta_whole_cohort_mean_contribution={key: mean(r["per_coordinate"][c]["components"]
                    ["whole_cohort_mean_contribution"][key] for r in selected)
                    for key in ("waiting_s", "service_s", "flow_s")},
                finite_only_unfinished=sum(r["per_coordinate"][c]["tails"]["finite_only_unfinished"] for r in selected),
                BG_only_unfinished=sum(r["per_coordinate"][c]["tails"]["BG_only_unfinished"] for r in selected))
                for c in range(len(records[0]["workloads"]))])
    for field in ("counts", "chosen_vs_own_context_BG_shapes", "chosen_vs_own_context_BG_requirements", "changed_root_oldest_start_opportunity"):
        combined = Counter()
        for row in records:
            combined.update(row["own_context_actions"][field])
        result[field] = dict(combined)
    return result


def analyze(directory):
    design = read(directory / (STEM + "_design.json"))
    completed = read(directory / (FINITE + ".json"))
    bg_result = read(directory / (BG + ".json"))
    distribution = read(directory / (FINITE + "_protocol.json"))
    protocol = distribution["scientific_protocol"]
    require(completed["execution_checks_passed"] and bg_result["execution_checks_passed"]
        and completed["registered_at_utc"] == protocol["registered_at_utc"], "execution-valid cached inputs required")
    assignments = completed["execution_origins"]
    require(len(assignments) == 24 and [a["shard_index"] for a in assignments] == list(range(24))
        and len({a["tape_id"] for a in assignments}) == 24
        and {a["tape_id"] for a in assignments} == {t["tape_id"] for t in protocol["tapes"]}, "all24 literal paired tapes required")
    require([a["tape_id"] for a in assignments] == design["tapes"]
        and completed["registered_at_utc"] == design["finite_scientific_epoch"]
        and completed["distribution_registered_at_utc"] == design["finite_distribution_epoch"]
        and distribution["distribution_registered_at_utc"] == design["finite_distribution_epoch"]
        and bg_result["registered_at_utc"] == design["BG_epoch"]
        and protocol["birth_cohort"] == design["cohort"]
        and [protocol[key] for key in ("arrival_horizon", "observation_end")] == design["cutoffs"],
        "registered cached-analysis roster, epochs or cohort differ")
    bg_runs = trajectories(ROOT / protocol["previous_trajectories"], bg_result["registered_at_utc"])
    by_id = {r["tape_id"]: r for r in completed["paired_runs"]}
    models = {m["node"]: m for m in protocol["models"]}
    records = []
    for assignment in assignments:
        tape_id = assignment["tape_id"]
        row, document = by_id[tape_id], read(assignment["row_path"])
        require(document["tape_id"] == tape_id
            and document["shard_index"] == assignment["shard_index"]
            and document["scientific_registered_at_utc"] == completed["registered_at_utc"]
            and all(row[key] for key in ("baseline_reproduction_passed", "rollout_checks_passed", "cached_reference_checks_passed")),
            "finite mixed-execution cache identity/checks differ")
        finite = trajectories(document["trajectory_file"], completed["registered_at_utc"])[tape_id]["arrival_rollout"]
        bg = bg_runs[tape_id]["arrival_rollout"]
        jobs = {name: {(j["coordinate"], j["index"]): j for j in arm["jobs"]}
            for name, arm in (("finite", finite), ("BG", bg))}
        require(jobs["finite"].keys() == jobs["BG"].keys() and all(
            jobs["finite"][key]["arrival"] == jobs["BG"][key]["arrival"] for key in jobs["finite"]), "paired canonical births differ")
        lower, upper = protocol["birth_cohort"]
        ids = {key for key, job in jobs["finite"].items() if lower <= job["arrival"] <= upper}
        model = models[row["node"]]
        cutoffs = {}
        for cutoff in ("arrival_horizon", "observation_end"):
            time = protocol[cutoff]
            actual = cohort_metrics({"maxweight": bg, "arrival_rollout": finite}, time, protocol["birth_cohort"])
            require(snapshot_matches(actual, row["reference_snapshots"][cutoff]), "recomputed full-cohort metrics differ")
            values = pair_components(jobs["finite"], jobs["BG"], ids, time, len(ids))
            require(isclose(values["delta_mean"]["flow_s"], actual["arrival_rollout"]["restricted_mean_flow_s"]
                - actual["maxweight"]["restricted_mean_flow_s"], abs_tol=1e-10), "component flow does not reconstruct cohort mean")
            classes = []
            for c, workload in enumerate(model["workloads"]):
                filtered = {role: {**arm, "jobs": [j for j in arm["jobs"] if j["coordinate"] == c]}
                    for role, arm in (("maxweight", bg), ("arrival_rollout", finite))}
                metrics = cohort_metrics(filtered, time, protocol["birth_cohort"])
                expected = next(item for item in row["class_snapshots"] if item["coordinate"] == c)
                require(snapshot_matches(metrics, expected["reference_snapshots"][cutoff]), "recomputed class metrics differ")
                class_ids = {key for key in ids if key[0] == c}
                classes.append(dict(coordinate=c, workload=workload,
                    delta_metrics={key: metrics["arrival_rollout"][key] - metrics["maxweight"][key] for key in METRICS},
                    components=pair_components(jobs["finite"], jobs["BG"], class_ids, time, len(ids)),
                    tails=tails(jobs["finite"], jobs["BG"], class_ids, time)))
            cutoffs[cutoff] = dict(components=values, tails=tails(jobs["finite"], jobs["BG"], ids, time),
                delta_metrics={key: actual["arrival_rollout"][key] - actual["maxweight"][key] for key in METRICS}, per_coordinate=classes)
        records.append(dict(**{key: row[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
            workloads=model["workloads"], execution_origin=assignment["origin"], cohort_jobs=len(ids), cutoffs=cutoffs,
            first_divergence=first_divergence(finite, bg, model["options"], model["workloads"]),
            own_context_actions=own_context_actions(finite, model["options"], model["workloads"])))
        print(tape_id, "cached cohort reproduced", flush=True)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["node"], row["load"]].append(row)
    return dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(), design_registered_at_utc=design["registered_at_utc"],
        finite_registered_at_utc=completed["registered_at_utc"],
        BG_registered_at_utc=bg_result["registered_at_utc"], source_decision=completed["decision"], new_simulation_executions=0,
        new_planner_calls=0, performance_gain_established=False,
        execution_origin_counts=dict(Counter(a["origin"] for a in assignments)),
        cached_trajectories_read=48, metric_reproduction_passed=True, pooled=aggregate(records),
        node_load_groups=[dict(node=node, load=load, **aggregate(rows)) for (node, load), rows in sorted(grouped.items())], records=records,
        scope="All24 paired development tapes and three shared seeds. Means and class contributions use equal-tape weights; P90 is the mean of each pair's common-ID cohort P90, not a job-pooled quantile. Service means completion minus admission, censored at each cutoff. Own-context BG is finite's evaluated incumbent, not the independent BG trajectory after divergence. Whole-policy differences and first-divergence descriptors are descriptive, not single-action causal effects. No policy guard or promotion gate is introduced.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    result = analyze(args.artifact_dir)
    destination = args.artifact_dir / (STEM + "_summary.json")
    with destination.open("x") as target:
        json.dump(result, target, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"output": str(destination), "paired_tapes": len(result["records"]), "new_simulations": 0}), flush=True)


if __name__ == "__main__":
    main()
