"""Summarize cached real finite suffixes by the anchor's tagged-population boundary.

This reads completed trajectories on the server; it never calls a planner or
replays service. Synthetic prefix jobs and actual future births remain distinct.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from math import isclose
from pathlib import Path
import pickle


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
CROSS = "aligned_hardware_persistent_action_controller_cross_20261002"


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(job):
    return job["coordinate"], job["index"]


def flow(job, cutoff):
    end = min(cutoff, job["completion"] if job["completion"] is not None else cutoff)
    return Fraction.from_float(end) - Fraction.from_float(job["arrival"])


def pair_group(ids, parent, native, cutoff, tail_cutoff):
    """Exact binary-input sums, reported as floats after paired subtraction."""
    costs = {}
    for name, arm in (("parent", parent), ("native", native)):
        jobs = [arm[key] for key in ids]
        values = [flow(job, cutoff) for job in jobs]
        completed = [flow(job, cutoff) for job in jobs
            if job["completion"] is not None and job["completion"] <= cutoff]
        costs[name] = dict(S1=sum(values, Fraction()),
            S2=sum((value * value for value in values), Fraction()),
            completed_S1=sum(completed, Fraction()),
            completed_S2=sum((value * value for value in completed), Fraction()),
            completed_jobs=len(completed),
            unfinished=sum(job["completion"] is None or job["completion"] > cutoff for job in jobs),
            tail_unfinished=sum(job["arrival"] <= tail_cutoff and
                (job["completion"] is None or job["completion"] > tail_cutoff) for job in jobs))
    delta = {key: costs["parent"][key] - costs["native"][key] for key in costs["parent"]}
    return dict(jobs=len(ids), complete_both=not (costs["parent"]["unfinished"] or costs["native"]["unfinished"]),
        parent={key: float(value) if isinstance(value, Fraction) else value for key, value in costs["parent"].items()},
        native={key: float(value) if isinstance(value, Fraction) else value for key, value in costs["native"].items()},
        delta_parent_minus_native={**{key: float(value) if isinstance(value, Fraction) else value for key, value in delta.items()},
            "mean_flow_s": float(delta["S1"] / len(ids)) if ids else None}), costs


def analyze(directory):
    protocol = read(directory / (CROSS + "_protocol.json"))
    completed = read(directory / (CROSS + "_summary.json"))
    require(completed["registered_at_utc"] == protocol["registered_at_utc"]
        and completed["execution_checks_passed"] and completed["completed_new_suffixes"] == 3,
        "completed registered three-state crossing required")
    records = []
    for index, target in enumerate(protocol["targets"]):
        row = read(directory / (CROSS + f"_shard_{index:02d}.json"))
        require(row["registered_at_utc"] == protocol["registered_at_utc"]
            and row["tape_id"] == target["tape_id"] and row["execution_checks_passed"],
            "cached finite suffix identity differs")
        with Path(target["fixture_file"]).open("rb") as source:
            fixture = pickle.load(source)
        with gzip.open(row["jobs_file"], "rt") as source:
            saved = json.load(source)
        require(saved["registered_at_utc"] == protocol["registered_at_utc"]
            and saved["tape_id"] == target["tape_id"], "cached finite jobs identity differs")
        parent = {identity(job): job for job in fixture["cached_arms"]["parent_finite"]["jobs"]}
        native = {identity(job): job for job in saved["jobs"]}
        checkpoint = fixture["checkpoint"]
        require(parent.keys() == native.keys() and all(parent[key]["arrival"] == native[key]["arrival"]
            for key in parent), "paired canonical real births differ")
        anchor = fixture["finite_frame"]["start"]
        window = max(option.duration for option in fixture["model"].options)
        window_end = anchor + window
        visible = {identity(checkpoint["jobs"][i]) for queue in checkpoint["queues"] for i in queue}
        committed = {identity(job) for job in checkpoint["jobs"] if job["start"] is not None}
        real_prefix = {key for key, job in parent.items() if anchor < job["arrival"] <= window_end}
        real_later = {key for key, job in parent.items() if job["arrival"] > window_end}
        lower, upper = protocol["birth_cohort"]
        cohort = {key for key, job in parent.items() if lower <= job["arrival"] <= upper}
        require(window_end < lower, "fixed anchor window now overlaps the reported cohort")
        primary = dict(preanchor_nonvisible=committed, visible=visible, real_prefix=real_prefix,
            later_before_reported_cohort={key for key in real_later if parent[key]["arrival"] < lower},
            reported_cohort=cohort,
            later_after_reported_cohort={key for key in real_later if parent[key]["arrival"] > upper})
        require(sum(map(len, primary.values())) == len(parent) and set.union(*primary.values()) == parent.keys(),
            "anchor waiting/committed/real-prefix/later partition differs")
        frame = fixture["finite_frame"]
        require(tuple(map(len, checkpoint["queues"])) == tuple(frame["queue"])
            and tuple(tuple(anchor - checkpoint["jobs"][i]["arrival"] for i in queue)
                for queue in checkpoint["queues"]) == tuple(map(tuple, frame["waiting_ages"]))
            and all(parent[key] == native[key] for key in committed), "visible queue or committed prefix changed")
        groups = {**primary, "visible_plus_real_prefix": visible | real_prefix,
            "real_later": real_later, "all_jobs": set(parent)}
        outcomes, exact_groups = {}, {}
        for name, ids in groups.items():
            outcomes[name], exact_groups[name] = pair_group(ids, parent, native, protocol["observation_end"], protocol["arrival_horizon"])
            outcomes[name]["per_coordinate"] = [dict(coordinate=c, workload=workload,
                **pair_group({key for key in ids if key[0] == c}, parent, native,
                    protocol["observation_end"], protocol["arrival_horizon"])[0])
                for c, workload in enumerate(fixture["model"].workloads)]
        require(all(sum(exact_groups[name][arm][metric] for name in primary)
            == exact_groups["all_jobs"][arm][metric]
            for arm in ("parent", "native") for metric in ("S1", "S2", "completed_S1", "completed_S2")),
            "disjoint population costs did not reconstruct all cached jobs")
        reference = next(record for record in completed["records"] if record["tape_id"] == target["tape_id"])
        effect = reference["root_effects"]["finite"]
        measured = outcomes["reported_cohort"]["delta_parent_minus_native"]
        require(isclose(measured["mean_flow_s"], effect["delta_T_mean"], abs_tol=1e-10)
            and measured["unfinished"] == effect["delta_1200_unfinished"]
            and measured["tail_unfinished"] == effect["delta_900_unfinished"],
            "cached reported-cohort effects did not reproduce")
        records.append(dict(tape_id=target["tape_id"], seed=target["seed"], frame_index=target["frame_index"],
            parent_root=target["parent_root"], native_root=target["native_root"],
            anchor_s=anchor, tagged_window_s=window, tagged_window_end_s=window_end,
            visible_ids=sorted(visible), real_prefix_ids=sorted(real_prefix),
            preanchor_nonvisible_counts=dict(completed=sum(parent[key]["completion"] <= anchor for key in committed),
                busy=sum(parent[key]["completion"] > anchor for key in committed)),
            reported_cohort_all_beyond_tagged_window=cohort <= real_later,
            fixture_file=target["fixture_file"], native_finite_jobs_file=row["jobs_file"],
            original_cohort_effect_reproduced=True, groups=outcomes))
    return dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_registered_at_utc=protocol["registered_at_utc"], records=records,
        primary_disjoint_partition=list(primary), observation_end=protocol["observation_end"],
        unfinished_cutoff=protocol["arrival_horizon"], cached_service_suffixes_read=6,
        new_simulation_executions=0, full24_decision=protocol["full24_decision"],
        interpretation="Paired finite-parent-first minus finite-native-first cached real effects. S1 and S2 sum flow and squared flow with completion censored at observation_end; completed_S1/completed_S2 use completed jobs only, and complete_both identifies uncensored groups. Visible IDs match the real anchor queue. Real-prefix births share only the time-window definition with synthetic prefix jobs, not their identities or realizations. visible_plus_real_prefix, real_later and all_jobs are aggregates of the six disjoint partition cells. These costs are not an unbiased synthetic forecast-error estimate; an arrival-law mismatch is not established by this diagnostic.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    result = analyze(args.artifact_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
