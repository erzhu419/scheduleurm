"""Diagnose the existing empirical-service development panel from saved jobs only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from math import fsum, isclose
from pathlib import Path
from time import perf_counter

from algorithm.experiments.bg_empirical_cutoff_diagnostic import analyse_pair


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "aligned_hardware_BG_empirical_service_development_20261004"
STEM = "aligned_hardware_BG_empirical_cutoff_20261004"
EPOCH = "2026-10-04T01:02:43.343470+00:00"
IDENTITY = ("unit_index", "condition", "policy", "planning_seed", "service_seed_key",
            "tape_id", "node", "lanes", "load", "seed", "service_seed")
AREAS = ("middle_area_job_s", "neighborhood_area_job_s", "full_area_job_s")


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else path.open()) as source:
        return json.load(source)


def save(path, value):
    if path.exists():
        raise ValueError("preserve existing diagnostic artifact: " + str(path))
    with (gzip.open(path, "xt") if str(path).endswith(".gz") else path.open("x")) as output:
        json.dump(value, output, separators=(",", ":"), allow_nan=False)


def sums(rows):
    return {key: fsum(row[key] for row in rows) for key in rows[0]}


def grouped_sums(rows):
    classes = rows[0]["per_workload"]
    return dict(total=sums([row["total"] for row in rows]), per_workload=[
        dict(coordinate=position, workload=identity["workload"],
             **sums([{key: value for key, value in row["per_workload"][position].items()
                      if key not in ("coordinate", "workload")} for row in rows]))
        for position, identity in enumerate(classes)])


def flow_sums(rows):
    jobs = sum(row["jobs"] for row in rows)
    totals = sums([row["delta_sums"] for row in rows])
    return dict(jobs=jobs, delta_sums=totals,
        job_weighted_delta_means={key: value / jobs if jobs else None for key, value in totals.items()},
        equal_tape_delta_means={key: fsum(row["delta_means"][key] for row in rows) / len(rows)
            if all(row["delta_means"][key] is not None for row in rows) else None for key in totals},
        signs={key: sums([row["signs"][key] for row in rows]) for key in totals})


def flow_grouped(rows):
    return dict(total=flow_sums([row["total"] for row in rows]), per_workload=[
        dict(coordinate=position, workload=identity["workload"],
             **flow_sums([row["per_workload"][position] for row in rows]))
        for position, identity in enumerate(rows[0]["per_workload"])])


def action_mix(rows):
    groups = defaultdict(list)
    for row in rows:
        for action in row["actions"]:
            groups[action["action_id"]].append(action)
    fields = ("dispatches", "changed_dispatches", "admitted_jobs", "admitted_cohort_jobs",
              "actual_duration_s", "nominal_duration_s")
    return dict(window=rows[0]["window"], right_endpoint_included=False, actions=[
        dict(action_id=action, loaded=values[0]["loaded"], requirements=values[0]["requirements"],
             **{key: fsum(value[key] for value in values) for key in fields})
        for action, values in sorted(groups.items())])


def seed_direction(row):
    result = row["diagnostic"]
    return dict(seed=row["seed"], service_seed=row["service_seed"],
        cutoff_delta=result["paired"]["cutoff_delta"],
        delta_mean_flow_parts=result["paired"]["flow_decomposition"]["total"]["delta_means"],
        area_deltas={area: {key: result["arms"]["BG"][area]["total"][key]
                           - result["arms"]["maxweight"][area]["total"][key]
                           for key in ("queued", "inflight", "unfinished")} for area in AREAS})


def compact_job(row):
    return {**{key: row[key] for key in ("coordinate", "index", "workload", "arrival", "delta")},
        "arms": {label: dict(state=arm["state"], start=arm["start"], completion=arm["completion"],
            action_id=arm["dispatch"]["action_id"] if arm["dispatch"] else None)
            for label, arm in row["arms"].items()}}


def hold_detail(row):
    result = row["diagnostic"]
    frames = {}
    for label, arm in result["arms"].items():
        frames[label] = [{key: value for key, value in frame.items() if key != "jobs"}
                         for frame in arm["active_frames"]]
    discordant = {}
    for key in ("bg_only_unfinished", "mw_only_unfinished"):
        records = result["paired"][key]
        discordant[key] = dict(count=len(records), per_workload=[
            dict(workload=workload, count=sum(job["workload"] == workload for job in records))
            for workload in sorted({job["workload"] for job in records})],
            examples_largest_absolute_flow_difference=[compact_job(job) for job in
                sorted(records, key=lambda job: (-abs(job["delta"]["flow_s"]),
                                                 job["coordinate"], job["index"]))[:3]])
    return dict(tape_id=row["tape_id"], **seed_direction(row),
        cutoff={label: arm["cutoff"] for label, arm in result["arms"].items()},
        state_transitions=result["paired"]["state_transitions"], active_frames=frames,
        flow_decomposition=result["paired"]["flow_decomposition"], discordant=discordant)


def aggregate(rows, hold):
    first = rows[0]
    result = {key: first[key] for key in ("condition", "node", "lanes", "load")}
    result.update(pairs=len(rows), original_secondary_hold=hold,
        seed_directions=[seed_direction(row) for row in rows],
        flow_decomposition=flow_grouped([row["diagnostic"]["paired"]["flow_decomposition"] for row in rows]))
    result["arms"] = {}
    for label in ("maxweight", "BG"):
        arms = [row["diagnostic"]["arms"][label] for row in rows]
        result["arms"][label] = dict(
            **{key: grouped_sums([arm[key] for arm in arms]) for key in ("cutoff", *AREAS)},
            snapshots=[dict(time=snapshot["time"], **sums([
                arm["snapshots"][position]["total"] for arm in arms]))
                for position, snapshot in enumerate(arms[0]["snapshots"])],
            action_mix={window: action_mix([arm["action_mix"][window] for arm in arms])
                        for window in ("middle", "near_cutoff")})
    if hold:
        result["hold_seed_details"] = [hold_detail(row) for row in rows]
    return result


def compact_summary(summary):
    """Project the completed diagnostic for small local transfer; retain all groups."""
    counts = ("arrived", "completed", "queued", "inflight", "unfinished")

    def cutoff(row):
        return dict(total={key: row["total"][key] for key in counts}, per_workload=[
            {key: group[key] for key in ("workload", *counts)} for group in row["per_workload"]])

    def flow(row):
        return {key: value for key, value in row.items()
                if key not in ("signs", "max_decomposition_residual_s", "coordinate")}

    def grouped_flow(row):
        return dict(total=flow(row["total"]), per_workload=[flow(value) for value in row["per_workload"]])

    def frame(row):
        selected = {key: value for key, value in row.items() if key not in
                    ("option_index", "requirements", "actual_duration", "nominal_duration", "cohort_snapshot")}
        selected["cohort_cutoff"] = cutoff(row["cohort_snapshot"])
        return selected

    result = {key: value for key, value in summary.items() if key != "condition_groups"}
    result["condition_groups"] = []
    for group in summary["condition_groups"]:
        value = {key: item for key, item in group.items() if key not in
                 ("arms", "flow_decomposition", "hold_seed_details")}
        value["flow_decomposition"] = grouped_flow(group["flow_decomposition"])
        value["arms"] = {}
        for label, arm in group["arms"].items():
            value["arms"][label] = dict(cutoff=cutoff(arm["cutoff"]),
                **{key: arm[key] for key in AREAS},
                snapshots=[{key: snapshot[key] for key in ("time", *counts)} for snapshot in arm["snapshots"]])
            if group["original_secondary_hold"]:
                value["arms"][label]["action_mix"] = arm["action_mix"]
        if "hold_seed_details" in group:
            value["hold_seed_details"] = []
            for detail in group["hold_seed_details"]:
                selected = {key: item for key, item in detail.items() if key not in
                            ("cutoff", "flow_decomposition", "active_frames")}
                selected["cutoff"] = {label: cutoff(row) for label, row in detail["cutoff"].items()}
                selected["flow_decomposition"] = grouped_flow(detail["flow_decomposition"])
                selected["active_frames"] = {label: [frame(row) for row in rows]
                    for label, rows in detail["active_frames"].items()}
                value["hold_seed_details"].append(selected)
        result["condition_groups"].append(value)
    result["full_summary_server_file"] = STEM + "_summary.json"
    result["omitted_from_local_projection"] = ["non-HOLD action mixes", "job direction counts",
        "age and remaining-time sums", "redundant frame durations and requirements"]
    return result


def run(directory):
    started = perf_counter()
    full_path, summary_path = (directory / (STEM + suffix) for suffix in ("_full.json.gz", "_summary.json"))
    if full_path.exists() or summary_path.exists():
        raise ValueError("preserve existing completed diagnostic output")
    protocol = read(directory / (SOURCE + "_protocol.json"))
    source_summary = read(directory / (SOURCE + "_summary.json"))
    if (protocol["registered_at_utc"] != EPOCH or source_summary["registered_at_utc"] != EPOCH
            or protocol["experiment"] != SOURCE or source_summary["experiment"] != SOURCE):
        raise ValueError("source identity or registered epoch differs")
    references = {(condition, row["tape_id"]): row
        for condition, summary in source_summary["statistics"]["condition_summaries"].items()
        for row in summary["paired_results"]}
    units = {(unit["condition"], unit["tape_id"], unit["policy"]): unit for unit in protocol["units"]}
    models = {model["node"]: model for model in protocol["models"]}
    if len(units) != 96 or len(references) != 48:
        raise ValueError("require the complete registered 96-arm, 48-pair panel")
    rows = []
    for (condition, tape_id), reference in references.items():
        arms = {}
        for policy in ("maxweight", "BG"):
            unit = units[condition, tape_id, policy]
            saved = read(directory / (SOURCE + f"_unit_{unit['unit_index']:03d}_trajectory.json.gz"))
            if saved["registered_at_utc"] != EPOCH or any(saved["unit"][key] != unit[key] for key in IDENTITY):
                raise ValueError("saved arm metadata differs from registration")
            arms[policy] = saved["result"]
        diagnostic = analyse_pair(arms["maxweight"], arms["BG"], models[reference["node"]])
        for policy, original in (("maxweight", "factual"), ("BG", "replacement")):
            counts = diagnostic["arms"][policy]["cutoff"]["total"]
            expected = reference["absolute_snapshots"]["arrival_horizon"][original]
            for actual, old in (("arrived", "arrivals"), ("completed", "completed_jobs"),
                                ("queued", "queued_jobs"), ("inflight", "inflight_jobs"),
                                ("unfinished", "unfinished_jobs")):
                if counts[actual] != expected[old]:
                    raise ValueError(f"cutoff count differs: {condition} {tape_id} {policy} {actual}")
        if not isclose(diagnostic["paired"]["flow_decomposition"]["total"]["delta_means"]["flow_s"],
                       reference["delta_T_mean"], abs_tol=1e-9):
            raise ValueError("new flow decomposition differs from original paired mean")
        rows.append(dict(**{key: reference[key] for key in
            ("condition", "tape_id", "node", "lanes", "load", "seed", "service_seed")}, diagnostic=diagnostic))
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"], row["node"], row["lanes"], row["load"]].append(row)
    holds = {(condition, row["node"], row["lanes"], row["load"])
        for condition, screen in source_summary["performance_screens"].items()
        for row in screen["failed_secondary_groups"]}
    metadata = dict(experiment=STEM, source_experiment=SOURCE, source_registered_at_utc=EPOCH,
        analysed_at_utc=datetime.now(timezone.utc).isoformat(), saved_trajectories_read=96,
        paired_trajectories=48, conditions=2, groups=16, joint_seed_blocks=3,
        new_simulations=0, policy_calls=0, scores_recomputed=0,
        original_cutoff_counts_matched=480, original_paired_flow_means_matched=48,
        cohort=[300, 900], cutoff=900, observation_end=1200,
        area_windows=dict(middle_area_job_s=[300, 900], neighborhood_area_job_s=[850, 950],
                          full_area_job_s=[300, 1200]),
        orientation="BG minus maxweight", original_performance_screens=source_summary["performance_screens"],
        statistical_scope="Retrospective descriptive diagnostic on the existing three joint seed blocks; no confidence intervals or new confirmation.",
        aggregation="Counts and areas sum across tapes. Flow includes both job-weighted and original equal-tape means. All windows use the same inclusive [300,900] birth cohort.",
        job_example_rule="At most three jobs per discordance direction and tape, ranked by absolute final flow difference; full discordant lists remain in the server full artifact.")
    summary = dict(**metadata, condition_groups=[aggregate(values, key in holds)
        for key, values in sorted(groups.items())], full_detail_server_file=full_path.name,
        wall_s=perf_counter() - started)
    save(full_path, dict(**metadata, paired_results=rows))
    save(summary_path, summary)
    compact_path = directory / (STEM + "_compact.json")
    save(compact_path, compact_summary(summary))
    print(json.dumps(dict(saved_trajectories_read=96, paired_trajectories=48, new_simulations=0,
        full_file=full_path.name, full_bytes=full_path.stat().st_size,
        summary_file=summary_path.name, summary_bytes=summary_path.stat().st_size,
        compact_file=compact_path.name, compact_bytes=compact_path.stat().st_size)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "md/experiment_artifacts")
    run(parser.parse_args().artifact_dir)


if __name__ == "__main__":
    main()
