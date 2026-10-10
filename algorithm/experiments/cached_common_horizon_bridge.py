"""Compare the fixed-H forecast area with the two cached real finite suffixes.

Only server-side fixture/job caches and small summaries are read. This does not
call a planner, replay service, or infer a terminal MW drain from factual jobs.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path
import pickle

from algorithm.experiments.cached_finite_population_bridge import (
    ARTIFACTS, CROSS, identity, read, require,
)


FORECAST = "aligned_hardware_common_horizon_flow_20261003_r1"
METRICS = ("waiting_area", "inflight_area", "common_area")


def exact(value):
    return Fraction.from_float(float(value))


def area(job, anchor, horizon):
    birth = exact(job["arrival"])
    begin = max(anchor, birth)
    if begin > horizon:
        return (Fraction(0),) * 3
    start = exact(job["start"]) if job["start"] is not None else horizon
    completion = exact(job["completion"]) if job["completion"] is not None else horizon
    waiting = max(Fraction(0), min(horizon, start) - begin)
    inflight = max(Fraction(0), min(horizon, completion) - max(begin, start))
    flow = max(Fraction(0), min(horizon, completion) - begin)
    require(waiting + inflight == flow, "restricted waiting/inflight areas differ from flow area")
    return waiting, inflight, flow


def pair_area(ids, parent, native, anchor, horizon):
    totals = {}
    for name, jobs in (("parent", parent), ("native", native)):
        values = [area(jobs[key], anchor, horizon) for key in ids]
        totals[name] = {field: sum((value[index] for value in values), Fraction(0))
            for index, field in enumerate(METRICS)}
    totals["delta_parent_minus_native"] = {field: totals["parent"][field] - totals["native"][field]
        for field in METRICS}
    return dict(jobs=len(ids), **{arm: {field: float(value) for field, value in costs.items()}
        for arm, costs in totals.items()}), totals


def preference(value):
    return "parent" if value < 0 else "native" if value > 0 else "tie"


def analyze(directory):
    protocol = read(directory / (CROSS + "_protocol.json"))
    cross = read(directory / (CROSS + "_summary.json"))
    forecast = read(directory / (FORECAST + "_summary.json"))
    require(cross["registered_at_utc"] == protocol["registered_at_utc"]
        and cross["execution_checks_passed"] and cross["completed_new_suffixes"] == 3,
        "completed original three-state crossing required")
    require(forecast["execution_checks_passed"] and forecast["completed_root_sample_units"] == 48
        and forecast["available_state_rankings"] == 3, "completed r1 fixed-three forecast required")
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
        require(parent.keys() == native.keys() and all(parent[key]["arrival"] == native[key]["arrival"]
            for key in parent), "paired canonical real births differ")
        anchor = exact(fixture["finite_frame"]["start"])
        tau = exact(max(max(1., float(option.duration)) for option in fixture["model"].options))
        horizon, prefix_end = anchor + 2 * tau, anchor + tau
        checkpoint = fixture["checkpoint"]
        visible = {identity(checkpoint["jobs"][i]) for queue in checkpoint["queues"] for i in queue}
        committed = {identity(job) for job in checkpoint["jobs"] if job["start"] is not None}
        require(all(parent[key] == native[key] for key in committed), "committed prefix changed")
        initial_busy = {key for key in committed if exact(parent[key]["completion"]) > anchor}
        prefix = {key for key, job in parent.items() if anchor < exact(job["arrival"]) <= prefix_end}
        background = {key for key, job in parent.items() if prefix_end < exact(job["arrival"]) <= horizon}
        sources = dict(initial_visible=visible, initial_committed=initial_busy,
            real_prefix=prefix, real_background=background)
        active = {key for key, job in parent.items() if exact(job["arrival"]) <= horizon
            and (exact(job["arrival"]) > anchor or job["completion"] is None
                or exact(job["completion"]) > anchor)}
        require(sum(map(len, sources.values())) == len(active)
            and set.union(*sources.values()) == active, "fixed-H active population partition differs")
        total, exact_total = pair_area(active, parent, native, anchor, horizon)
        actual_sources, exact_sources = {}, {}
        for name, ids in sources.items():
            actual_sources[name], exact_sources[name] = pair_area(ids, parent, native, anchor, horizon)
        require(all(sum(exact_sources[name][arm][field] for name in sources) == exact_total[arm][field]
            for arm in exact_total for field in METRICS), "source areas do not reconstruct all jobs")
        classes = [dict(coordinate=c, workload=workload,
            **pair_area({key for key in active if key[0] == c}, parent, native, anchor, horizon)[0])
            for c, workload in enumerate(fixture["model"].workloads)]
        h_population = {name: dict(waiting=sum(job["start"] is None or exact(job["start"]) > horizon
            for key, job in jobs.items() if key in active),
            inflight=sum(job["start"] is not None and exact(job["start"]) <= horizon
                and (job["completion"] is None or exact(job["completion"]) > horizon)
                for key, job in jobs.items() if key in active))
            for name, jobs in (("parent", parent), ("native", native))}
        prediction = next(record for record in forecast["records"] if record["tape_id"] == target["tape_id"])
        require(all(prediction[key] == target[key] for key in ("seed", "frame_index", "parent_root", "native_root"))
            and prediction["horizon_relative_s"] == float(2 * tau), "fixed forecast anchor/root/H differs")
        pdelta = prediction["comparison"]["parent_minus_native_mean"]
        population_delta = prediction["comparison"]["exact_population_parent_minus_native_mean"]
        predicted_sources = {source: {field: float(Fraction(*values[field])) for field in METRICS}
            for source, values in population_delta["sources"].items()}
        predicted_classes = [dict(coordinate=values["coordinate"],
            **{field: float(Fraction(*values[field])) for field in METRICS})
            for values in population_delta["per_coordinate"]]
        actual_delta = exact_total["delta_parent_minus_native"]["common_area"]
        predicted_delta = Fraction(*prediction["comparison"]["exact_parent_minus_native_mean"]["common_area"])
        actual_cohort = prediction["birth_cohort_root_effects"]["finite"]
        records.append(dict(tape_id=target["tape_id"], seed=target["seed"], frame_index=target["frame_index"],
            parent_root=target["parent_root"], native_root=target["native_root"], anchor_s=float(anchor),
            tau_max_s=float(tau), horizon_absolute_s=float(horizon), horizon_relative_s=float(2 * tau),
            actual_window=dict(total=total, sources=actual_sources, per_coordinate=classes,
                population_at_H=h_population,
                births_anchor_to_H=len(prefix | background), prefix_births=len(prefix), background_births=len(background),
                exact_common_area_delta=[actual_delta.numerator, actual_delta.denominator]),
            forecast=dict(mean_parent_minus_native=pdelta, mean_source_area_deltas=predicted_sources,
                mean_class_area_deltas=predicted_classes, samples=8), actual_reported_cohort_effect=actual_cohort,
            ordering=dict(actual_common_area=preference(actual_delta),
                forecast_common_area=preference(predicted_delta), forecast_with_MW_terminal=preference(pdelta["cost"]),
                actual_reported_cohort_mean=preference(actual_cohort["delta_T_mean"]),
                opposite_realized_and_forecast_window_order=actual_delta * predicted_delta < 0,
                opposite_actual_window_and_cohort_order=actual_delta * actual_cohort["delta_T_mean"] < 0)))
    return dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_cross_registered_at_utc=protocol["registered_at_utc"],
        source_forecast_registered_at_utc=forecast["registered_at_utc"], records=records,
        cached_service_suffixes_read=6, new_simulation_executions=0,
        actual_MW_terminal_cost_computed=False,
        interpretation="Exact all-unfinished waiting/inflight area from the anchor to the same fixed H in two cached real finite arms. Past cost cancels. Forecast means use eight original sampled paths; the actual area uses one realized tape, so opposite signs identify a forecast-realization ranking disagreement rather than establish systematic arrival-law error. Window-versus-later-cohort signs distinguish short-window ranking from subsequent propagation. Actual continuation after H is not the synthetic fixed MW terminal drain, which cannot be inferred from these factual jobs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.artifact_dir), separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
