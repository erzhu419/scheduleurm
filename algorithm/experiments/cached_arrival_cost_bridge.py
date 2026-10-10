"""Compare cached sampled and realized births without any service or planner calls."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path
import pickle

from algorithm.experiments.cached_finite_population_bridge import ARTIFACTS, CROSS, identity, read, require

STEM = "aligned_hardware_cached_arrival_cost_bridge_20261003"
NATIVE = "aligned_hardware_native_terminal_flow_20261003"
DEVELOPMENT = "aligned_hardware_two_step_current_policy_bg_incumbent_development_20261001"
SOURCES = ("visible", "committed", "prefix", "background")
FIELDS = ("cost", "common_area", "waiting_area", "inflight_area", "terminal_cost",
    "full_waiting_cost", "full_inflight_cost")


def exact(value):
    if isinstance(value, Fraction):
        return value
    return Fraction(*value) if isinstance(value, list) else Fraction.from_float(float(value))


def pair_stats(parent, native, horizon):
    require(len(parent) == len(native), "paired population counts differ")
    totals = {}
    states = {}
    for role, jobs in (("parent", parent), ("native", native)):
        totals[role] = {field: Fraction(0) for field in FIELDS}
        states[role] = dict(waiting=0, inflight=0, done=0, waiting_age_s=Fraction(0))
        for job in jobs:
            birth, start, completion = (exact(job[key]) for key in ("birth", "start", "completion"))
            begin = max(Fraction(0), birth)
            require(birth <= horizon and start <= completion and completion >= begin,
                "fixed H population contains an unavailable completion")
            waiting = max(Fraction(0), min(horizon, start) - begin)
            inflight = max(Fraction(0), min(horizon, completion) - max(begin, start))
            full_waiting = max(Fraction(0), start - begin)
            full_inflight = completion - max(begin, start)
            values = (completion - begin, waiting + inflight, waiting, inflight,
                max(Fraction(0), completion - horizon), full_waiting, full_inflight)
            require(values[0] == values[1] + values[4] == full_waiting + full_inflight,
                "window/residual or waiting/processing failed to reconstruct full cost")
            for field, value in zip(FIELDS, values, strict=True):
                totals[role][field] += value
            state = "done" if completion <= horizon else "waiting" if start >= horizon else "inflight"
            states[role][state] += 1
            if state == "waiting":
                states[role]["waiting_age_s"] += horizon - birth
    return dict(jobs=len(parent), **totals,
        delta={field: totals["parent"][field] - totals["native"][field] for field in FIELDS}, at_H=states)


def birth_stats(jobs, horizon):
    births = [exact(job["birth"]) for job in jobs if job["source"] in ("prefix", "background")]
    bins = [sum(horizon * index / 4 < birth <= horizon * (index + 1) / 4 for birth in births)
        for index in range(4)]
    require(sum(bins) == len(births), "H birth bins failed to partition new arrivals")
    return dict(newborn_jobs=len(births), quarter_H_counts=bins, birth_time_sum_s=sum(births, Fraction(0)),
        birth_exposure_s=sum((horizon - birth for birth in births), Fraction(0)))


def group(parent, native, horizon, coordinate=None, source=None):
    def selected(jobs):
        return [job for job in jobs if (coordinate is None or job["coordinate"] == coordinate)
            and (source is None or job["source"] == source)]
    p, n = selected(parent), selected(native)
    return dict(**pair_stats(p, n, horizon), births=birth_stats(p, horizon))


def pooled(rows):
    result = {}
    for key, value in rows[0].items():
        if isinstance(value, dict):
            result[key] = pooled([row[key] for row in rows])
        elif isinstance(value, list):
            result[key] = [sum(row[key][i] for row in rows) / len(rows) for i in range(len(value))]
        else:
            result[key] = sum((row[key] for row in rows), Fraction(0)) / len(rows)
    return result


def compare(actual, forecasts):
    predicted = pooled(forecasts)
    n_actual, n_predicted = actual["jobs"], predicted["jobs"]
    details = {}
    if n_actual and n_predicted:
        for field in FIELDS:
            da, dp = actual["delta"][field] / n_actual, predicted["delta"][field] / n_predicted
            mix, per_job = (n_predicted - n_actual) * da, n_predicted * (dp - da)
            gap = predicted["delta"][field] - actual["delta"][field]
            require(mix + per_job == gap, "count and per-job terms failed to reconstruct cost gap")
            details[field] = dict(actual_delta_per_job=da, forecast_pooled_delta_per_job=dp,
                forecast_standardized_to_actual_count=n_actual * dp, gap=gap,
                count_term_at_actual_per_job_delta=mix, per_job_term_at_forecast_count=per_job)
    return dict(actual=actual, forecast_mean=predicted, count_per_job_decomposition=details or None,
        forecast_job_count_range=[min(row["jobs"] for row in forecasts), max(row["jobs"] for row in forecasts)])


def jsonable(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def analyze(directory):
    cross, native = (read(directory / (stem + "_protocol.json")) for stem in (CROSS, NATIVE))
    summary = read(directory / (NATIVE + "_summary.json"))
    actual_summary = read(directory / "aligned_hardware_cached_h_population_flow_bridge_20261003_summary.json")
    development = read(directory / (DEVELOPMENT + "_protocol.json"))
    require(summary["registered_at_utc"] == native["registered_at_utc"]
        and summary["execution_checks_passed"] and summary["completed_root_sample_units"] == 48
        and native["source_cross_registered_at_utc"] == cross["registered_at_utc"]
        == actual_summary["source_registered_at_utc"] and native["planning_samples"] == 8,
        "completed matching fixed-three native terminal and actual H caches required")
    records = []
    for index, target in enumerate(cross["targets"]):
        reference = summary["records"][index]
        actual_reference = actual_summary["records"][index]
        require(reference["tape_id"] == actual_reference["tape_id"] == target["tape_id"], "anchor identity differs")
        with Path(target["fixture_file"]).open("rb") as source:
            fixture = pickle.load(source)
        row = read(directory / (CROSS + f"_shard_{index:02d}.json"))
        with gzip.open(row["jobs_file"], "rt") as source:
            saved = json.load(source)
        require(row["execution_checks_passed"] and row["registered_at_utc"]
            == saved["registered_at_utc"] == cross["registered_at_utc"]
            and saved["tape_id"] == target["tape_id"], "cached actual completion identity differs")
        anchor = exact(fixture["finite_frame"]["start"])
        horizon = exact(actual_reference["horizon_relative_s"])
        checkpoint = fixture["checkpoint"]
        visible = {identity(checkpoint["jobs"][i]) for queue in checkpoint["queues"] for i in queue}
        roles = {}
        for role, jobs in (("parent", fixture["cached_arms"]["parent_finite"]["jobs"]), ("native", saved["jobs"])):
            selected = []
            for job in jobs:
                key, birth = identity(job), exact(job["arrival"]) - anchor
                if key in visible:
                    source = "visible"
                elif birth <= 0 and exact(job["completion"]) > anchor:
                    source = "committed"
                elif 0 < birth <= horizon:
                    source = "prefix" if birth <= horizon / 2 else "background"
                else:
                    continue
                selected.append(dict(id=key, coordinate=job["coordinate"], source=source, birth=birth,
                    start=exact(job["start"]) - anchor, completion=exact(job["completion"]) - anchor))
            roles[role] = sorted(selected, key=lambda job: job["id"])
        p, n = roles["parent"], roles["native"]
        require([(j["id"], j["birth"], j["source"]) for j in p]
            == [(j["id"], j["birth"], j["source"]) for j in n], "paired actual H roster differs")
        pairs = []
        for sample in range(8):
            cells = [read(directory / (NATIVE + f"_shard_{16 * index + 8 * slot + sample:02d}.json")) for slot in range(2)]
            require(all(cell["registered_at_utc"] == native["registered_at_utc"]
                and cell["execution_checks_passed"] and cell["result"]["flow"]["complete"]
                and not cell["result"]["flow"]["overflow"] for cell in cells), "all fixed forecast pairs must remain complete")
            pair = [cell["result"]["flow"]["population"]["jobs"] for cell in cells]
            rosters = [[{k:job[k] for k in ("id", "coordinate", "birth", "source")} for job in arm] for arm in pair]
            require(rosters[0] == rosters[1], "paired forecast roster differs")
            pairs.append(pair)
        all_stats = compare(group(p, n, horizon), [group(*pair, horizon) for pair in pairs])
        require(abs(float(all_stats["actual"]["delta"]["cost"]) - actual_reference["H_population"]["actual_finite_complete"]["delta_cost"]) < 1e-9
            and all_stats["forecast_mean"]["delta"]["cost"] == exact(reference["comparison"]["exact_parent_minus_native_mean"]["cost"]),
            "new grouping failed to reproduce completed actual/forecast full costs")
        classes = [dict(coordinate=c, workload=workload, **compare(group(p, n, horizon, c),
            [group(*pair, horizon, c) for pair in pairs]), sources={source: compare(group(p, n, horizon, c, source),
                [group(*pair, horizon, c, source) for pair in pairs]) for source in SOURCES})
            for c, workload in enumerate(fixture["model"].workloads)]
        require(sum((item["forecast_mean"]["delta"]["cost"] for item in classes), Fraction(0))
            == all_stats["forecast_mean"]["delta"]["cost"], "class costs failed to reconstruct forecast total")
        capacity, = [value for value in development["capacities"] if value["node"] == fixture["model"].node and value["load"] == .95]
        rates = cells[0]["result"]["environment"]["rates"]
        records.append(dict(tape_id=target["tape_id"], seed=target["seed"], frame_index=target["frame_index"],
            anchor_s=anchor, horizon_relative_s=horizon, anchor_counts=fixture["finite_frame"]["observed_arrival_counts"],
            anchor_estimated_rates=rates, generating_rates=capacity["arrival_rates"],
            all_population=all_stats, per_coordinate=classes,
            class_count_standardized_forecast_cost_delta=sum(item["count_per_job_decomposition"]["cost"]["forecast_standardized_to_actual_count"] for item in classes),
            reported_cohort_effect=reference["actual_context"]["cohort_effect"]))
    return jsonable(dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_native_registered_at_utc=native["registered_at_utc"], source_cross_registered_at_utc=cross["registered_at_utc"],
        records=records, cached_native_units_read=48, cached_actual_service_suffixes_read=6,
        new_simulation_executions=0, new_planner_calls=0, full24_decision=summary["full24_decision"],
        interpretation="Source and class partitions use the same time windows and initial-job definitions, but different realized births. Forecast values average the unchanged eight paths. Per-job values pool path costs/counts; count standardization and the exact count/per-job gap identity are descriptive, not causal or a deployable chooser. A selected realized H population is not expected-cost ground truth. Known generating rates are diagnostic context only."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    result = analyze(args.artifact_dir)
    output = args.artifact_dir / (STEM + "_small_results") / (STEM + "_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"cached_arrival_cost_complete states={len(result['records'])} new_planner_calls=0 summary_bytes={output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
