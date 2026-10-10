"""Continue cached H states with same-stream arrivals and cheap native MW."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import pickle
import socket
from time import perf_counter

from algorithm.experiments import aligned_hardware_common_horizon_flow as horizon
from algorithm.experiments.competing_terminal_flow_rollout import evaluate_cached

ROOT, ARTIFACTS = horizon.ROOT, horizon.ARTIFACTS
STEM = "aligned_hardware_competing_terminal_flow_20261003"
DESIGN = "md/experiment_artifacts/" + STEM + "_design.json"
SOURCES = ("algorithm/experiments/aligned_hardware_competing_terminal_flow.py",
    "algorithm/experiments/competing_terminal_flow_rollout.py", DESIGN)
FIELDS = ("cost", "common_area", "waiting_area", "inflight_area", "terminal_cost")
GROUPS = ("visible", "prefix", "background", "committed")
read, dump, require, jsonable = horizon.read, horizon.dump, horizon.require, horizon.jsonable


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def exact(value):
    return value if isinstance(value, Fraction) else Fraction(*value)


def preference(value):
    return "parent" if value < 0 else "native" if value > 0 else "tie"


def _options(target, source_epoch):
    prepared = read(target["prepared_row"])
    require(prepared["registered_at_utc"] == source_epoch and prepared["execution_checks_passed"]
        and prepared["tape_id"] == target["tape_id"], "original native preparation identity differs")
    with Path(prepared["fixture_file"]).open("rb") as source:
        fixture = pickle.load(source)
    require(fixture["registered_at_utc"] == source_epoch and fixture["tape_id"] == target["tape_id"],
        "prepared catalogue identity differs")
    for role in ("parent", "native"):
        root = target[role + "_root"]
        require(root in target["original_risk_roots"] and root in prepared["original_risk_roots"]
            and all(need <= count for need, count in zip(fixture["options"][root].requirements,
                fixture["context"]["queue"], strict=True)), "original feasible BG-risk roots must remain fixed")
    return fixture["options"]


def register(directory):
    design = read(path(directory, "_design.json"))
    prior = horizon.load_protocol(directory)
    completed = read(horizon.path(directory, "_summary.json"))
    require(design["source_horizon_stem"] == horizon.STEM
        and design["source_horizon_epoch"] == prior["registered_at_utc"] == completed["registered_at_utc"]
        and completed["execution_checks_passed"] and completed["completed_root_sample_units"] == 48
        and completed["available_state_rankings"] == 3
        and prior["planning_samples"] == design["planning_samples"] == 8
        and prior["planning_event_cap"] == design["planning_event_cap"] == 128
        and prior["horizon_tau_max_multiple"] == design["horizon_tau_max_multiple"] == 2
        and design["root_sample_units"] == 48,
        "completed original48 H2tau states and frozen8/cap128 design required")
    targets = []
    for target, reference in zip(prior["targets"], completed["records"], strict=True):
        require(reference["tape_id"] == target["tape_id"] and reference["frame_index"] == target["frame_index"]
            and reference["comparison"]["available"], "all three complete original state rankings required")
        _options(target, prior["source_tagged_registered_at_utc"])
        targets.append({**{key: target[key] for key in ("index", "tape_id", "seed", "frame_index", "parent_root",
            "native_root", "prepared_row", "original_risk_roots", "horizon_relative_s",
            "previous_comparisons", "birth_cohort_root_effects")}, "no_arrivals_reference": reference["comparison"]})
    source_shards = []
    for assignment in prior["assignments"]:
        source_file = horizon.path(directory, f"_shard_{assignment['index']:02d}.json")
        row = read(source_file)
        require(row["registered_at_utc"] == prior["registered_at_utc"] and row["assignment"] == assignment
            and row["execution_checks_passed"] and row["result"]["flow"]["complete"]
            and not row["result"]["flow"]["overflow"], "all48 complete fixed native H caches required")
        source_shards.append(str(source_file))
    protocol = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        source_horizon_registered_at_utc=prior["registered_at_utc"],
        source_tagged_registered_at_utc=prior["source_tagged_registered_at_utc"],
        planning_samples=8, planning_event_cap=128, horizon_tau_max_multiple=2,
        root_sample_units=48, assignments=prior["assignments"], targets=targets,
        source_shards=source_shards, design=design, new_native_controller_calls=0,
        reused_native_H_units=48, true_service_suffix_executions=0,
        full24_decision=prior["full24_decision"],
        execution_sources={**prior["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}})
    dump(path(directory, "_protocol.json"), protocol)
    print(f"registered_at_utc={protocol['registered_at_utc']} cached_H_units=48 new_native_calls=0", flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def _sum_work(values):
    """Accumulate measured counters without copying cached native traces."""
    total = {}
    for value in values:
        for key, item in value.items():
            if isinstance(item, dict):
                total[key] = _sum_work((total.get(key, {}), item))
            elif isinstance(item, (int, float)):
                total[key] = total.get(key, 0) + item
    return total


def _root_scores(rows):
    valid = [row for row in rows if row["execution_checks_passed"]]
    values = [row["result"]["flow"] for row in valid]
    available = len(values) == 8 and all(value["complete"] and not value["overflow"] for value in values)
    total = {field: sum((value[field] for value in values), Fraction(0)) for field in FIELDS}

    def groups(items):
        return {field: sum((item[field] for item in items), Fraction(0)) for field in FIELDS}

    populations = [value["population"] for value in values]
    components = dict(sources={source: groups([population["sources"][source] for population in populations])
        for source in GROUPS}, per_coordinate=[dict(coordinate=coordinate,
            **groups([population["per_coordinate"][coordinate] for population in populations]))
            for coordinate in range(len(populations[0]["per_coordinate"]))]) if populations else None
    return dict(available=available, exact_partial_totals=total,
        mean={field: float(total[field] / 8) if available else None for field in FIELDS},
        population_totals=components,
        failed_samples=[row["assignment"]["sample_index"] for row in rows if not row["execution_checks_passed"]],
        overflow_samples=[row["assignment"]["sample_index"] for row in valid if row["result"]["flow"]["overflow"]],
        work={key: _sum_work([value.get(key, {}) for value in values]) for key in
            ("regeneration_work", "incremental_tail_work", "reused_window_work", "actual_work", "preprocessing")},
        runtime={key: sum(value.get(key, 0) for value in values) for key in
            ("constructor_cpu_s", "constructor_wall_s", "worker_cpu_s", "worker_wall_s",
             "total_cpu_s", "total_wall_s", "terminal_controller_constructors")})


def _summary(protocol, rows, wall_s):
    records = []
    for target in protocol["targets"]:
        cells = rows[16 * target["index"]:16 * (target["index"] + 1)]
        roots = {role: dict(root=target[role + "_root"], **_root_scores(cells[8 * slot:8 * (slot + 1)]))
            for slot, role in enumerate(("parent", "native"))}
        available = all(root["available"] for root in roots.values())
        delta = {field: (roots["parent"]["exact_partial_totals"][field] - roots["native"]["exact_partial_totals"][field]) / 8
            if available else None for field in FIELDS}
        if available:
            require(all(delta[field] == exact(target["no_arrivals_reference"]["exact_parent_minus_native_mean"][field])
                for field in ("common_area", "waiting_area", "inflight_area")),
                "only terminal competition may change; all cached common-H components must stay exact")
        samples = []
        for index in range(8):
            pair = [cells[8 * slot + index] for slot in range(2)]
            usable = all(row["execution_checks_passed"] and row["result"]["flow"]["complete"]
                and not row["result"]["flow"]["overflow"] for row in pair)
            if all(row["execution_checks_passed"] for row in pair):
                require(pair[0]["result"]["environment"] == pair[1]["result"]["environment"]
                    and pair[0]["scored_roster"] == pair[1]["scored_roster"],
                    "paired roots must retain the same fixed H roster and forecast environment")
                streams = [row["result"]["flow"]["background_events"] for row in pair]
                longest = max(streams, key=len)
                require(all(stream == longest[:len(stream)] for stream in streams),
                    "paired terminal birth streams must retain the same global prefix")
            sample_delta = pair[0]["result"]["flow"]["cost"] - pair[1]["result"]["flow"]["cost"] if usable else None
            samples.append(dict(sample_index=index, available=usable,
                parent_minus_native_cost=float(sample_delta) if usable else None,
                exact_parent_minus_native_cost=sample_delta))
        component_delta = None
        if available:
            component_delta = dict(sources={source: {field:
                (roots["parent"]["population_totals"]["sources"][source][field]
                    - roots["native"]["population_totals"]["sources"][source][field]) / 8
                for field in FIELDS} for source in GROUPS},
                per_coordinate=[dict(coordinate=coordinate, **{field:
                    (roots["parent"]["population_totals"]["per_coordinate"][coordinate][field]
                        - roots["native"]["population_totals"]["per_coordinate"][coordinate][field]) / 8
                    for field in FIELDS}) for coordinate in range(len(roots["parent"]["population_totals"]["per_coordinate"]))])
            values = [sample["exact_parent_minus_native_cost"] for sample in samples]
            summed = sum(values, Fraction(0))
            require(summed / 8 == delta["cost"], "eight sample deltas must reconstruct root mean")
            leave_one_out = [(summed - value) / 7 for value in values]
            votes = {name: sum(preference(value) == name for value in values) for name in ("parent", "native", "tie")}
        else:
            leave_one_out, votes = [], None
        records.append(dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "parent_root", "native_root",
            "original_risk_roots", "horizon_relative_s", "previous_comparisons", "birth_cohort_root_effects", "no_arrivals_reference")},
            roots=roots, samples=samples, comparison=dict(available=available,
                paired_H_roster_and_background_prefix_checks_passed=all(row["execution_checks_passed"] for row in cells),
                S1_preference=preference(delta["cost"]) if available else "unavailable",
                parent_minus_native_mean={field: float(value) if value is not None else None for field, value in delta.items()},
                exact_parent_minus_native_mean=delta, exact_population_parent_minus_native_mean=component_delta,
                cost_delta_vs_no_arrivals=float(delta["cost"] - exact(target["no_arrivals_reference"]["exact_parent_minus_native_mean"]["cost"]))
                    if available else None,
                sample_preferences=votes, leave_one_out_mean_range=[float(min(leave_one_out)), float(max(leave_one_out))]
                    if available else None)))
    failures = [row["assignment"]["index"] for row in rows if not row["execution_checks_passed"]]
    unit_status = [dict(index=row["assignment"]["index"], complete=row["result"]["flow"]["complete"]
        if row["execution_checks_passed"] else False, overflow=row["result"]["flow"]["overflow"]
        if row["execution_checks_passed"] else None,
        exact_partial_cost=row["result"]["flow"]["cost"] if row["execution_checks_passed"] else None,
        error=row.get("error")) for row in rows]
    return dict(registered_at_utc=protocol["registered_at_utc"],
        source_horizon_registered_at_utc=protocol["source_horizon_registered_at_utc"],
        decision="COMPETING_TERMINAL_FIXED_THREE_COMPLETE" if not failures else "COMPETING_TERMINAL_FAILED_UNITS",
        execution_checks_passed=not failures, completed_root_sample_units=48 - len(failures),
        recorded_root_sample_units=48, failed_units=failures, units=unit_status,
        available_state_rankings=sum(record["comparison"]["available"] for record in records),
        new_native_controller_calls=0, reused_native_H_units=48, true_service_suffix_executions=0,
        execution_host=socket.gethostname(), wall_s=wall_s, records=records,
        full24_decision=protocol["full24_decision"], performance_gain_established=False,
        score_units="Eight-path mean of total job-seconds; not mean flow time per job.",
        interpretation="Only the terminal arrival competition changed on the same48 cached H states. Every job born by H remains scored; post-H births compete but are unscored. Exact costs and every partial cap/failure record are retained. Old approximate/free/H rankings and real cohort labels are context; this does not establish full-policy performance or an arrival-law error.")


def run(directory):
    protocol = load_protocol(directory)
    options = [_options(target, protocol["source_tagged_registered_at_utc"]) for target in protocol["targets"]]
    start, rows = perf_counter(), []

    def progress(current, phase, *, force=False):
        nonlocal last_progress
        now = perf_counter()
        if force or now - last_progress >= 30:
            print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=current, total=48,
                unit="root_sample", elapsed_seconds=now - start, phase=phase), separators=(",", ":")), flush=True)
            last_progress = now

    last_progress = start
    progress(0, "cached_terminal", force=True)
    for assignment, source_file in zip(protocol["assignments"], protocol["source_shards"], strict=True):
        cached = read(source_file)
        require(cached["registered_at_utc"] == protocol["source_horizon_registered_at_utc"]
            and cached["assignment"] == assignment and cached["execution_checks_passed"],
            "registered source H cache changed")
        unit_start = perf_counter()
        row = dict(registered_at_utc=protocol["registered_at_utc"], assignment=assignment)
        try:
            scored_roster = [{key: job[key] for key in ("id", "coordinate", "birth", "source")}
                for job in cached["result"]["flow"]["population"]["jobs"]]
            require(all(exact(job["birth"]) <= exact(cached["result"]["flow"]["horizon"])
                for job in scored_roster), "cached scored roster contains jobs born after H")
            result = evaluate_cached(options[assignment["target_index"]], cached["result"], event_cap=128)
            require(result["root"] == assignment["root"] and result["sample_index"] == assignment["sample_index"],
                "cached root/global-sample identity changed")
            require(result["flow"]["common_area"] == exact(cached["result"]["flow"]["common_area"]),
                "cached common-H cost must remain unchanged")
            require(result["flow"]["population"]["total"] == len(scored_roster),
                "H-after arrivals must compete without joining the scored population")
            require(jsonable(result["scored_population"]) == scored_roster,
                "terminal costs must retain exactly the cached H job identities and births")
            row.update(execution_checks_passed=True, result=result, scored_roster=scored_roster,
                wall_s=perf_counter() - unit_start)
        except Exception as error:
            row.update(execution_checks_passed=False, result=None, wall_s=perf_counter() - unit_start,
                error=dict(type=type(error).__name__, message=str(error)))
            print(f"unit_failed index={assignment['index']} error={type(error).__name__}:{error}", flush=True)
        dump(path(directory, f"_unit_{assignment['index']:02d}.json"), jsonable(row))
        rows.append(row)
        progress(len(rows), "cached_terminal")
    summary = _summary(protocol, rows, perf_counter() - start)
    dump(path(directory, "_summary.json"), jsonable(summary))
    progress(48, "complete", force=True)
    print(f"complete units=48 available_rankings={summary['available_state_rankings']} failed={len(summary['failed_units'])}", flush=True)
    require(summary["execution_checks_passed"], "failed unit records retained; terminal evaluation unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    register(args.artifact_dir) if args.register else run(args.artifact_dir)


if __name__ == "__main__":
    main()
