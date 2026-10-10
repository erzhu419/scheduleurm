"""Integrate waiting and in-flight populations in two completed finite suffixes.

Reads server-side cached jobs only. No planner calls, service replay, or action
inference is performed. Every integration uses exact binary-input Fractions.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from math import isclose
from pathlib import Path
import pickle

from algorithm.experiments.cached_finite_population_bridge import (
    ARTIFACTS, CROSS, identity, read, require,
)
from algorithm.experiments.cached_admission_batches import reconstruct


def exact(value):
    return Fraction(value)


def components(job, cutoff):
    arrival = exact(job["arrival"])
    start = exact(job["start"]) if job["start"] is not None else cutoff
    end = exact(job["completion"]) if job["completion"] is not None else cutoff
    require(0 <= arrival <= cutoff and arrival <= start,
        "cached arrival/start outside the supported observation interval")
    require(job["completion"] is None or (job["start"] is not None and start <= end),
        "cached completion precedes admission")
    start, end = min(start, cutoff), min(end, cutoff)
    waiting, inflight = start - arrival, end - start
    require(waiting >= 0 and inflight >= 0 and waiting + inflight == end - arrival,
        "per-job waiting plus in-flight time differs from restricted flow")
    return waiting, inflight


def decomposition(ids, parent, native, cutoff):
    totals = {}
    for name, jobs in (("parent", parent), ("native", native)):
        pairs = [components(jobs[key], cutoff) for key in ids]
        waiting = sum((pair[0] for pair in pairs), Fraction())
        inflight = sum((pair[1] for pair in pairs), Fraction())
        totals[name] = dict(waiting=waiting, inflight=inflight, flow=waiting + inflight)
    totals["delta_parent_minus_native"] = {
        name: totals["parent"][name] - totals["native"][name]
        for name in ("waiting", "inflight", "flow")}
    require(totals["delta_parent_minus_native"]["waiting"]
        + totals["delta_parent_minus_native"]["inflight"]
        == totals["delta_parent_minus_native"]["flow"], "paired decomposition differs")
    public = dict(jobs=len(ids), **{
        arm: {name + "_s": float(value) for name, value in costs.items()}
        for arm, costs in totals.items()})
    public["delta_parent_minus_native"]["mean_flow_s"] = (
        float(totals["delta_parent_minus_native"]["flow"] / len(ids)) if ids else None)
    return public, totals


def timeline(ids, parent, native, cutoff, boundaries, expected, admissions=None, coordinate=None):
    """Sweep merged births, admissions and completions; intervals are [a,b)."""
    events = defaultdict(lambda: [0, 0, 0, 0])
    for slot, jobs in enumerate((parent, native)):
        for key in ids:
            job = jobs[key]
            events[exact(job["arrival"])][slot] += 1
            if job["start"] is not None and exact(job["start"]) <= cutoff:
                start = exact(job["start"])
                events[start][slot] -= 1
                events[start][2 + slot] += 1
            if job["completion"] is not None and exact(job["completion"]) <= cutoff:
                events[exact(job["completion"])][2 + slot] -= 1
    boundary_labels = defaultdict(list)
    for label, value in boundaries:
        boundary_labels[exact(value)].append(label)
        events[exact(value)]
    events[cutoff]
    counts, areas, previous = [0] * 4, [Fraction()] * 4, Fraction()
    episodes, current, checkpoints = [], None, []
    for time in sorted(events):
        elapsed = time - previous
        difference = counts[0] - counts[1]
        sign = (difference > 0) - (difference < 0)
        if elapsed:
            areas = [area + elapsed * count for area, count in zip(areas, counts)]
            if current is not None and current["sign"] != sign:
                episodes.append(current)
                current = None
            if sign:
                if current is None:
                    current = dict(sign=sign, start=previous, end=time, area=Fraction(),
                        peak=difference, start_counts=counts[:2], end_counts=counts[:2])
                current["end"] = time
                current["area"] += elapsed * difference
                current["end_counts"] = counts[:2]
                if abs(difference) > abs(current["peak"]):
                    current["peak"] = difference
        counts = [count + change for count, change in zip(counts, events[time])]
        require(all(count >= 0 for count in counts), "negative reconstructed queue population")
        for label in boundary_labels[time]:
            checkpoints.append(dict(boundary=label, time_s=float(time),
                delta_waiting_area_s=float(areas[0] - areas[1]),
                delta_inflight_area_s=float(areas[2] - areas[3]),
                delta_flow_area_s=float(areas[0] + areas[2] - areas[1] - areas[3]),
                waiting_counts_parent_native=counts[:2], inflight_counts_parent_native=counts[2:]))
        previous = time
    if current is not None:
        episodes.append(current)
    require(areas[0] == expected["parent"]["waiting"]
        and areas[1] == expected["native"]["waiting"]
        and areas[2] == expected["parent"]["inflight"]
        and areas[3] == expected["native"]["inflight"],
        "queue-area integration differs from per-job waiting/in-flight decomposition")
    require(sum((episode["area"] for episode in episodes), Fraction())
        == areas[0] - areas[1], "maximal signed episodes did not reconstruct waiting difference")

    def public_episode(episode):
        result = dict(start_s=float(episode["start"]), end_s=float(episode["end"]),
            duration_s=float(episode["end"] - episode["start"]),
            delta_waiting_area_s=float(episode["area"]),
            peak_signed_waiting_difference=episode["peak"],
            start_waiting_counts_parent_native=episode["start_counts"],
            end_before_transition_waiting_counts_parent_native=episode["end_counts"])
        if admissions is not None:
            result["admission_context"] = {}
            for arm, groups in admissions.items():
                selected = [group for group in groups
                    if episode["start"] <= exact(group["start"]) <= episode["end"]]
                compact = lambda group: {key: group[key] for key in (
                    "start", "unique_option_indices", "jobs_per_coordinate", "reserved_lane_ends")}
                result["admission_context"][arm] = dict(group_count=len(selected),
                    unique_option_counts=dict(Counter(index for group in selected
                        for index in (group["unique_option_indices"] or ()))),
                    first_two=[compact(group) for group in selected[:2]],
                    last_two=[compact(group) for group in selected[2:][-2:]],
                    first_two_serving_coordinate=[compact(group) for group in selected
                        if group["jobs_per_coordinate"][coordinate]][:2])
        return result

    return dict(cumulative=checkpoints, nonzero_waiting_episode_count=len(episodes),
        top_negative_waiting_episodes=[public_episode(episode) for episode in
            sorted((e for e in episodes if e["sign"] < 0), key=lambda e: e["area"])[:3]],
        top_positive_waiting_episodes=[public_episode(episode) for episode in
            sorted((e for e in episodes if e["sign"] > 0), key=lambda e: -e["area"])[:3]],
        per_job_and_queue_area_exactly_equal=True,
        signed_episodes_exactly_reconstruct_waiting_area=True)


def analyze(directory):
    protocol = read(directory / (CROSS + "_protocol.json"))
    completed = read(directory / (CROSS + "_summary.json"))
    require(completed["registered_at_utc"] == protocol["registered_at_utc"]
        and completed["execution_checks_passed"] and completed["completed_new_suffixes"] == 3,
        "completed registered three-state crossing required")
    cutoff = exact(protocol["observation_end"])
    records = []
    for index, target in enumerate(protocol["targets"]):
        if target["seed"] not in (41, 43):
            continue
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
        require(parent.keys() == native.keys() and all(parent[key]["arrival"] == native[key]["arrival"]
            for key in parent), "paired real births differ")
        anchor = fixture["finite_frame"]["start"]
        window_end = anchor + max(option.duration for option in fixture["model"].options)
        checkpoint = fixture["checkpoint"]
        visible = {identity(checkpoint["jobs"][i]) for queue in checkpoint["queues"] for i in queue}
        committed = {identity(job) for job in checkpoint["jobs"] if job["start"] is not None}
        require(all(parent[key] == native[key] for key in committed), "committed prefix changed")
        prefix = {key for key, job in parent.items() if anchor < job["arrival"] <= window_end}
        lower, upper = protocol["birth_cohort"]
        require(window_end < lower and upper <= cutoff, "fixed population boundaries changed")
        primary = dict(preanchor_nonvisible=committed, visible=visible, real_prefix=prefix,
            later_before_reported_cohort={key for key, job in parent.items() if window_end < job["arrival"] < lower},
            reported_cohort={key for key, job in parent.items() if lower <= job["arrival"] <= upper},
            later_after_reported_cohort={key for key, job in parent.items() if job["arrival"] > upper})
        require(sum(map(len, primary.values())) == len(parent) and set.union(*primary.values()) == parent.keys(),
            "population partition differs")
        groups = {**primary, "visible_plus_real_prefix": visible | prefix, "all_jobs": set(parent)}
        boundaries = [("anchor", anchor), ("anchor_plus_tau", window_end),
            ("cohort_start", lower), ("arrival_horizon", upper), ("observation_end", cutoff)]
        reconstructed = {arm: reconstruct(list(jobs.values()), fixture["model"].options,
            fixture["model"].lanes) for arm, jobs in (("parent", parent), ("native", native))}
        batch_summary = {}
        for arm, reconstructed_arm in reconstructed.items():
            bins = []
            for (_, start), (_, end) in zip(boundaries, boundaries[1:]):
                selected = [group for group in reconstructed_arm["groups"]
                    if start <= group["start"] < end or (end == cutoff and group["start"] == cutoff)]
                counts = Counter(index for group in selected
                    for index in (group["unique_option_indices"] or ()))
                bins.append(dict(start_s=float(start), end_s=float(end),
                    group_count=len(selected),
                    ambiguous_group_count=sum(group["unique_option_indices"] is None for group in selected),
                    unique_option_counts=dict(sorted(counts.items())),
                    admitted_jobs_per_coordinate=[sum(group["jobs_per_coordinate"][c] for group in selected)
                        for c in range(len(fixture["model"].workloads))]))
            batch_summary[arm] = {**{key: value for key, value in reconstructed_arm.items() if key != "groups"},
                "fixed_time_bins": bins}
        outcomes, exact_groups, temporal = {}, {}, {}
        for name, ids in groups.items():
            outcomes[name], exact_groups[name] = decomposition(ids, parent, native, cutoff)
            coordinates = []
            for coordinate, workload in enumerate(fixture["model"].workloads):
                coordinate_ids = {key for key in ids if key[0] == coordinate}
                public, totals = decomposition(coordinate_ids, parent, native, cutoff)
                coordinates.append(dict(coordinate=coordinate, workload=workload, **public))
                if name in ("all_jobs", "reported_cohort", "visible_plus_real_prefix"):
                    temporal.setdefault(name, []).append(dict(coordinate=coordinate, workload=workload,
                        **timeline(coordinate_ids, parent, native, cutoff, boundaries, totals,
                            {arm: value["groups"] for arm, value in reconstructed.items()}
                            if name == "all_jobs" else None, coordinate=coordinate)))
            outcomes[name]["per_coordinate"] = coordinates
        require(all(sum(exact_groups[name][arm][metric] for name in primary)
            == exact_groups["all_jobs"][arm][metric] for arm in ("parent", "native")
            for metric in ("waiting", "inflight", "flow")), "disjoint decompositions differ")
        reference = next(record for record in completed["records"] if record["tape_id"] == target["tape_id"])
        require(isclose(outcomes["reported_cohort"]["delta_parent_minus_native"]["mean_flow_s"],
            reference["root_effects"]["finite"]["delta_T_mean"], abs_tol=1e-9),
            "reported-cohort mean did not reproduce the completed paired effect")
        records.append(dict(seed=target["seed"], tape_id=target["tape_id"],
            parent_root=target["parent_root"], native_root=target["native_root"],
            anchor_s=anchor, window_end_s=window_end, fixture_file=target["fixture_file"],
            native_finite_jobs_file=row["jobs_file"], groups=outcomes, temporal_groups=temporal,
            reconstructed_admission_batches=batch_summary,
            exact_invariants=dict(per_job_waiting_plus_inflight_equals_restricted_flow=True,
                disjoint_population_costs_reconstruct_all_jobs=True,
                queue_areas_equal_per_job_costs=True, committed_prefix_unchanged=True)))
    require([record["seed"] for record in records] == [41, 43], "fixed seed pair missing")
    return dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_registered_at_utc=protocol["registered_at_utc"], records=records,
        observation_end=float(cutoff), source_protocol=str(directory / (CROSS + "_protocol.json")),
        cached_service_suffixes_read=4, new_simulation_executions=0,
        full24_decision=protocol["full24_decision"],
        interpretation="Parent-first minus native-first under finite continuation. Waiting is arrival to admission; in-flight is admission to completion, both censored at observation_end. Queue areas are cumulative from time zero. Episode endpoints are maximal intervals with a constant nonzero sign of the waiting-population difference; endpoint counts use the first/last interval inside the episode. Local real-prefix IDs are actual births, not synthetic planning IDs. This cached conditional comparison does not establish a full-policy gain.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.artifact_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
