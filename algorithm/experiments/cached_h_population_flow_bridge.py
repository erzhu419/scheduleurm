"""Separate cached H-population completion costs from later-born flow effects.

This is an offline diagnostic of existing actual finite suffixes, with no planner
calls or service replay. Actual finite continuation is not a synthetic MW tail.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from math import isclose
from pathlib import Path
import pickle

from algorithm.experiments.cached_common_horizon_bridge import exact, pair_area, preference
from algorithm.experiments.cached_finite_population_bridge import (
    ARTIFACTS, CROSS, identity, read, require,
)


def full_pair(ids, parent, native, *, origin=None):
    """Final known completion sums; elapsed pre-anchor cost is common and omitted."""
    totals = {}
    for name, jobs in (("parent", parent), ("native", native)):
        missing = sum(jobs[key]["completion"] is None for key in ids)
        cost = None if missing else sum((exact(jobs[key]["completion"]) - (
            max(origin, exact(jobs[key]["arrival"])) if origin is not None
            else exact(jobs[key]["arrival"])) for key in ids), Fraction(0))
        totals[name] = dict(unknown_completion_jobs=missing, cost=cost)
    complete = not any(values["unknown_completion_jobs"] for values in totals.values())
    delta = totals["parent"]["cost"] - totals["native"]["cost"] if complete else None
    return dict(jobs=len(ids), complete_both=complete,
        parent={key: float(value) if isinstance(value, Fraction) else value for key, value in totals["parent"].items()},
        native={key: float(value) if isinstance(value, Fraction) else value for key, value in totals["native"].items()},
        delta_cost=float(delta) if delta is not None else None,
        delta_mean_flow_s=float(delta / len(ids)) if delta is not None and ids else None), totals, delta


def observed_pair(ids, parent, native, cutoff):
    """Keep late unfinished jobs and label their restricted costs as censored."""
    totals = {}
    for name, jobs in (("parent", parent), ("native", native)):
        cost = sum((min(cutoff, exact(jobs[key]["completion"])) - exact(jobs[key]["arrival"])
            if jobs[key]["completion"] is not None else cutoff - exact(jobs[key]["arrival"])
            for key in ids), Fraction(0))
        totals[name] = dict(restricted_flow_cost=cost, unfinished_at_cutoff=sum(
            jobs[key]["completion"] is None or exact(jobs[key]["completion"]) > cutoff for key in ids))
    delta = totals["parent"]["restricted_flow_cost"] - totals["native"]["restricted_flow_cost"]
    return dict(jobs=len(ids), **{name: {key: float(value) if isinstance(value, Fraction) else value
            for key, value in costs.items()} for name, costs in totals.items()},
        delta_restricted_flow_cost=float(delta),
        delta_restricted_mean_flow_s=float(delta / len(ids)) if ids else None,
        full_mean_or_p90=False)


def analyze(directory):
    protocol = read(directory / (CROSS + "_protocol.json"))
    completed = read(directory / (CROSS + "_summary.json"))
    require(completed["registered_at_utc"] == protocol["registered_at_utc"]
        and completed["execution_checks_passed"] and completed["completed_new_suffixes"] == 3,
        "completed original three-state crossing required")
    records = []
    cutoff = exact(protocol["observation_end"])
    lower, upper = map(exact, protocol["birth_cohort"])
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
        require(parent.keys() == native.keys() and all(parent[key]["arrival"] == native[key]["arrival"]
            for key in parent), "paired canonical real births differ")
        anchor = exact(fixture["finite_frame"]["start"])
        tau = exact(max(max(1., float(option.duration)) for option in fixture["model"].options))
        horizon = anchor + 2 * tau
        require(horizon < lower <= upper <= cutoff, "fixed H and reported cohort boundaries differ")
        checkpoint = fixture["checkpoint"]
        visible = {identity(checkpoint["jobs"][i]) for queue in checkpoint["queues"] for i in queue}
        committed = {identity(job) for job in checkpoint["jobs"] if job["start"] is not None}
        require(all(parent[key] == native[key] for key in committed), "committed prefix changed")
        busy = {key for key in committed if exact(parent[key]["completion"]) > anchor}
        births = {key for key, job in parent.items() if anchor < exact(job["arrival"]) <= horizon}
        h_roster = visible | busy | births
        active = {key for key, job in parent.items() if exact(job["arrival"]) <= horizon
            and (exact(job["arrival"]) > anchor or job["completion"] is None
                or exact(job["completion"]) > anchor)}
        require(h_roster == active and len(h_roster) == len(visible) + len(busy) + len(births),
            "fixed H roster differs from its actual active population")
        area, exact_area = pair_area(h_roster, parent, native, anchor, horizon)
        h_full, h_totals, h_delta = full_pair(h_roster, parent, native, origin=anchor)
        residual = None
        if h_full["complete_both"]:
            residual = {name: sum((max(Fraction(0), exact(jobs[key]["completion"]) - horizon)
                for key in h_roster), Fraction(0)) for name, jobs in (("parent", parent), ("native", native))}
            require(all(exact_area[name]["common_area"] + residual[name] == h_totals[name]["cost"]
                for name in ("parent", "native")), "H area plus actual finite residual differs from full cost")
            residual["delta_parent_minus_native"] = residual["parent"] - residual["native"]
        before = {key for key, job in parent.items() if horizon < exact(job["arrival"]) < lower}
        cohort = {key for key, job in parent.items() if lower <= exact(job["arrival"]) <= upper}
        after = {key for key, job in parent.items() if upper < exact(job["arrival"]) <= cutoff}
        before_result = full_pair(before, parent, native)[0]
        cohort_result, _, cohort_delta = full_pair(cohort, parent, native)
        reference = next(record for record in completed["records"] if record["tape_id"] == target["tape_id"])
        effect = reference["root_effects"]["finite"]
        require(cohort_result["complete_both"] and isclose(cohort_result["delta_mean_flow_s"],
            effect["delta_T_mean"], abs_tol=1e-9), "cached original cohort mean effect differs")
        class_full = [dict(coordinate=c, workload=workload,
            **full_pair({key for key in h_roster if key[0] == c}, parent, native, origin=anchor)[0])
            for c, workload in enumerate(fixture["model"].workloads)]
        ordering = dict(actual_H_window=preference(exact_area["delta_parent_minus_native"]["common_area"]),
            actual_H_complete=preference(h_delta) if h_delta is not None else "unavailable",
            original_cohort_mean=preference(cohort_delta),
            original_cohort_p90=preference(effect["delta_T_common_p90"]))
        records.append(dict(tape_id=target["tape_id"], seed=target["seed"], frame_index=target["frame_index"],
            parent_root=target["parent_root"], native_root=target["native_root"],
            anchor_s=float(anchor), horizon_s=float(horizon), horizon_relative_s=float(2 * tau),
            H_population=dict(common_window=area, actual_finite_complete=h_full,
                actual_finite_completion_residual={key: float(value) for key, value in residual.items()}
                    if residual is not None else None, per_coordinate=class_full),
            later_population=dict(after_H_before_cohort=before_result,
                original_cohort=cohort_result, original_cohort_mean_p90_effect=effect,
                after_cohort_observed=observed_pair(after, parent, native, cutoff),
                all_after_H_observed=observed_pair(before | cohort | after, parent, native, cutoff)),
            ordering=ordering))
    return dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_registered_at_utc=protocol["registered_at_utc"], cached_service_suffixes_read=6,
        new_simulation_executions=0, records=records,
        interpretation="The fixed H roster excludes pre-anchor completed jobs. Its common area plus remaining committed final completions under the cached actual finite continuation equals its complete cost whenever all completions are known. This actual tail is not MW and is not a chooser input. Opposite H-complete and later-cohort orderings identify a population/objective mismatch on these paired realized suffixes; a H-window reversal by its actual finite residual instead indicates continuation matters. Late unresolved populations retain costs restricted at 1200 and are not reported as complete mean/p90. Original cohort mean/p90 effects are copied from the completed crossing; seed41 need not prefer parent because its original tail worsens.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.artifact_dir), separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
