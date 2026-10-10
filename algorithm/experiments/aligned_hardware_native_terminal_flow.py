"""Compare native finite terminal control on the same48 cached H populations."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import pickle
import socket
from time import perf_counter

from algorithm.experiments import aligned_hardware_common_horizon_flow as horizon
from algorithm.experiments import aligned_hardware_competing_terminal_flow as mw
from algorithm.experiments.native_terminal_flow_rollout import evaluate_cached

ROOT, ARTIFACTS = horizon.ROOT, horizon.ARTIFACTS
STEM = "aligned_hardware_native_terminal_flow_20261003"
DESIGN = "md/experiment_artifacts/" + STEM + "_design.json"
ACTUAL = "aligned_hardware_cached_h_population_flow_bridge_20261003_summary.json"
SOURCES = ("algorithm/experiments/aligned_hardware_native_terminal_flow.py",
    "algorithm/experiments/native_terminal_flow_rollout.py", DESIGN)
FIELDS, GROUPS = mw.FIELDS, mw.GROUPS
read, dump, require, jsonable, exact, preference = mw.read, mw.dump, mw.require, mw.jsonable, mw.exact, mw.preference


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def _fixture(target, source_epoch):
    prepared = read(target["prepared_row"])
    require(prepared["registered_at_utc"] == source_epoch and prepared["execution_checks_passed"]
        and prepared["tape_id"] == target["tape_id"], "original native preparation differs")
    with Path(prepared["fixture_file"]).open("rb") as source:
        fixture = pickle.load(source)
    require(fixture["registered_at_utc"] == source_epoch and fixture["tape_id"] == target["tape_id"],
        "prepared options, anchor seed/time differ")
    for role in ("parent", "native"):
        root = target[role + "_root"]
        require(root in target["original_risk_roots"] and all(need <= count for need, count in
            zip(fixture["options"][root].requirements, fixture["context"]["queue"], strict=True)),
            "same feasible original BG-risk roots required")
    return fixture


def _reference(comparison):
    return {key: comparison[key] for key in ("available", "S1_preference",
        "parent_minus_native_mean", "exact_parent_minus_native_mean")}


def register(directory, workers):
    design = read(path(directory, "_design.json"))
    prior, previous = horizon.load_protocol(directory), mw.load_protocol(directory)
    hsummary, msummary = read(horizon.path(directory, "_summary.json")), read(mw.path(directory, "_summary.json"))
    actual = read(Path(directory) / ACTUAL)
    cross = horizon.tagged.crossing.load_protocol(directory)
    require(design["source_horizon_stem"] == horizon.STEM
        and design["source_horizon_epoch"] == prior["registered_at_utc"] == hsummary["registered_at_utc"]
        and design["source_competing_terminal_epoch"] == previous["registered_at_utc"] == msummary["registered_at_utc"]
        and previous["source_horizon_registered_at_utc"] == prior["registered_at_utc"]
        and actual["source_registered_at_utc"] == cross["registered_at_utc"]
        and all(summary["execution_checks_passed"] and summary["completed_root_sample_units"] == 48
            and summary["available_state_rankings"] == 3 for summary in (hsummary, msummary))
        and prior["planning_samples"] == design["planning_samples"] == 8
        and prior["planning_event_cap"] == design["planning_event_cap"] == 128
        and prior["horizon_tau_max_multiple"] == design["horizon_tau_max_multiple"] == 2,
        "complete original48 H and MW-terminal results, bridge epoch, and saved8/cap128 design required")
    targets = []
    for target, hrecord, mrecord, real in zip(prior["targets"], hsummary["records"], msummary["records"], actual["records"], strict=True):
        require(all(record["tape_id"] == target["tape_id"] and record["frame_index"] == target["frame_index"]
            for record in (hrecord, mrecord, real)) and real["H_population"]["actual_finite_complete"]["complete_both"],
            "retain the same three actual states and complete fixed-H population labels")
        _fixture(target, prior["source_tagged_registered_at_utc"])
        targets.append({**{key: target[key] for key in ("index", "tape_id", "seed", "frame_index", "parent_root",
            "native_root", "prepared_row", "original_risk_roots", "horizon_relative_s")},
            "no_arrivals_MW_reference": _reference(hrecord["comparison"]),
            "competing_MW_reference": _reference(mrecord["comparison"]),
            "actual_context": dict(H_common_area_delta=real["H_population"]["common_window"]["delta_parent_minus_native"]["common_area"],
                H_population_complete_cost_delta=real["H_population"]["actual_finite_complete"]["delta_cost"],
                H_population_finite_residual_delta=real["H_population"]["actual_finite_completion_residual"]["delta_parent_minus_native"],
                cohort_effect={key: real["later_population"]["original_cohort_mean_p90_effect"][key]
                    for key in ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished", "delta_1200_unfinished")},
                class_cohort_effects=[{key: row[key] for key in ("coordinate", "delta_T_mean", "delta_T_common_p90")}
                    for row in real["later_population"]["original_cohort_mean_p90_effect"]["per_coordinate_deltas"]])})
    source_shards = []
    for assignment in prior["assignments"]:
        source_file = horizon.path(directory, f"_shard_{assignment['index']:02d}.json")
        row = read(source_file)
        require(row["registered_at_utc"] == prior["registered_at_utc"] and row["assignment"] == assignment
            and row["execution_checks_passed"] and row["result"]["flow"]["complete"]
            and not row["result"]["flow"]["overflow"] and len(row["result"]["flow"]["final_tree"]["planners"]) == 3,
            "all48 complete original H caches and three-node native semantic states required")
        source_shards.append(str(source_file))
    protocol = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        source_horizon_registered_at_utc=prior["registered_at_utc"],
        source_competing_terminal_registered_at_utc=previous["registered_at_utc"],
        source_tagged_registered_at_utc=prior["source_tagged_registered_at_utc"],
        source_cross_registered_at_utc=cross["registered_at_utc"],
        planning_samples=8, planning_event_cap=128, horizon_tau_max_multiple=2, root_sample_units=48,
        targets=targets, assignments=prior["assignments"], source_shards=source_shards, design=design,
        full24_decision=prior["full24_decision"],
        execution_sources={**previous["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}})
    dump(path(directory, "_protocol.json"), protocol)
    print(f"registered_at_utc={protocol['registered_at_utc']} reused_H_units=48 workers={workers}", flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_shard(index, directory):
    protocol = load_protocol(directory)
    assignment = protocol["assignments"][index]
    target = protocol["targets"][assignment["target_index"]]
    fixture = _fixture(target, protocol["source_tagged_registered_at_utc"])
    cached = read(protocol["source_shards"][index])
    require(cached["registered_at_utc"] == protocol["source_horizon_registered_at_utc"]
        and cached["assignment"] == assignment and cached["execution_checks_passed"],
        "registered H cache identity changed")
    started = perf_counter()
    print(f"unit_start index={index} role={assignment['role']} root={assignment['root']} sample={assignment['sample_index']}", flush=True)
    row = dict(index=index, registered_at_utc=protocol["registered_at_utc"], assignment=assignment,
        execution_host=socket.gethostname())

    def progress(fields):
        phase = fields["phase"]
        payload = {**fields, "current": int(phase in ("complete", "overflow")), "total": 1,
            "unit": "root_sample", "elapsed_seconds": fields["elapsed_wall_s"], "shard_index": index}
        print("SCHEDULEURM_PROGRESS " + json.dumps(payload, separators=(",", ":")), flush=True)

    try:
        roster = [{key: job[key] for key in ("id", "coordinate", "birth", "source")}
            for job in cached["result"]["flow"]["population"]["jobs"]]
        result = evaluate_cached(fixture["options"], fixture["seed"], cached["result"], fixture["context"]["now"],
            workers=protocol["workers"], sample_count=8, event_cap=128, progress=progress)
        require(result["root"] == assignment["root"] and result["sample_index"] == assignment["sample_index"]
            and jsonable(result["scored_population"]) == roster,
            "same fixed root/sample and H-scored FIFO roster required")
        require(all(result["flow"][field] == exact(cached["result"]["flow"][field])
            for field in ("common_area", "waiting_area", "inflight_area")),
            "H-native window costs must be reused exactly")
        row.update(execution_checks_passed=True, result=result, wall_s=perf_counter() - started)
        progress(dict(phase="complete" if result["flow"]["complete"] else "overflow",
            elapsed_wall_s=perf_counter() - started))
    except Exception as error:
        row.update(execution_checks_passed=False, result=None, wall_s=perf_counter() - started,
            error=dict(type=type(error).__name__, message=str(error)))
        dump(path(directory, f"_shard_{index:02d}.json"), jsonable(row))
        raise
    dump(path(directory, f"_shard_{index:02d}.json"), jsonable(row))
    print(f"unit_complete index={index} complete={result['flow']['complete']} overflow={result['flow']['overflow']}", flush=True)


def _costs(rows):
    valid = [row for row in rows if row["execution_checks_passed"]]
    values = [row["result"]["flow"] for row in valid]
    totals = {field: sum((exact(value[field]) for value in values), Fraction(0)) for field in FIELDS}
    available = len(values) == 8 and all(value["complete"] and not value["overflow"] for value in values)

    def group(items):
        return {field: sum((exact(item[field]) for item in items), Fraction(0)) for field in FIELDS}

    populations = [value["population"] for value in values]
    components = dict(sources={source: group([population["sources"][source] for population in populations])
        for source in GROUPS}, per_coordinate=[dict(coordinate=coordinate,
            **group([population["per_coordinate"][coordinate] for population in populations]))
            for coordinate in range(len(populations[0]["per_coordinate"]))]) if populations else None
    return dict(available=available, exact_partial_totals=totals,
        mean={field: float(totals[field] / 8) if available else None for field in FIELDS}, population_totals=components,
        overflow_samples=[row["assignment"]["sample_index"] for row in valid if row["result"]["flow"]["overflow"]],
        failed_samples=[row["assignment"]["sample_index"] for row in rows if not row["execution_checks_passed"]])


def _work(values):
    result = {}
    for value in values:
        for key, item in value.items():
            if isinstance(item, dict):
                result[key] = _work((result.get(key, {}), item))
            elif isinstance(item, (int, float)):
                result[key] = max(result.get(key, 0), item) if "_peak_" in key else result.get(key, 0) + item
    return result


def merge(directory):
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_shard_{index:02d}.json")) for index in range(48)]
    for assignment, row in zip(protocol["assignments"], rows, strict=True):
        require(row["index"] == assignment["index"] and row["assignment"] == assignment
            and row["registered_at_utc"] == protocol["registered_at_utc"], "all48 registered unit records required")
    records = []
    for target in protocol["targets"]:
        cells = rows[16 * target["index"]:16 * (target["index"] + 1)]
        roots = {role: dict(root=target[role + "_root"], **_costs(cells[8 * slot:8 * (slot + 1)]))
            for slot, role in enumerate(("parent", "native"))}
        available = all(root["available"] for root in roots.values())
        delta = {field: (roots["parent"]["exact_partial_totals"][field] - roots["native"]["exact_partial_totals"][field]) / 8
            if available else None for field in FIELDS}
        samples = []
        for index in range(8):
            pair = [cells[8 * slot + index] for slot in range(2)]
            valid = all(row["execution_checks_passed"] for row in pair)
            if valid:
                p, n = (row["result"] for row in pair)
                require(p["environment"] == n["environment"] and p["scored_population"] == n["scored_population"],
                    "paired roots must share fixed H roster and forecast environment")
                streams = [result["flow"]["background_events"] for result in (p, n)]
                longest = max(streams, key=len)
                require(all(stream == longest[:len(stream)] for stream in streams), "paired background stream prefixes differ")
            usable = valid and all(row["result"]["flow"]["complete"] and not row["result"]["flow"]["overflow"] for row in pair)
            sd = exact(pair[0]["result"]["flow"]["cost"]) - exact(pair[1]["result"]["flow"]["cost"]) if usable else None
            samples.append(dict(sample_index=index, available=usable, parent_minus_native_cost=float(sd) if usable else None,
                exact_parent_minus_native_cost=sd))
        components = None
        if available:
            require(all(delta[field] == exact(target["competing_MW_reference"]["exact_parent_minus_native_mean"][field])
                for field in ("common_area", "waiting_area", "inflight_area")), "only terminal control may change")
            components = dict(sources={source: {field:
                (roots["parent"]["population_totals"]["sources"][source][field]
                    - roots["native"]["population_totals"]["sources"][source][field]) / 8
                for field in FIELDS} for source in GROUPS}, per_coordinate=[dict(coordinate=coordinate, **{field:
                    (roots["parent"]["population_totals"]["per_coordinate"][coordinate][field]
                        - roots["native"]["population_totals"]["per_coordinate"][coordinate][field]) / 8
                    for field in FIELDS}) for coordinate in range(len(roots["parent"]["population_totals"]["per_coordinate"]))])
            values = [sample["exact_parent_minus_native_cost"] for sample in samples]
            summed = sum(values, Fraction(0))
            require(summed / 8 == delta["cost"], "all eight paired costs must reconstruct mean")
            loo = [(summed - value) / 7 for value in values]
        else:
            values, loo = [], []
        records.append(dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "parent_root", "native_root",
            "original_risk_roots", "no_arrivals_MW_reference", "competing_MW_reference", "actual_context")}, roots=roots, samples=samples,
            comparison=dict(available=available, S1_preference=preference(delta["cost"]) if available else "unavailable",
                parent_minus_native_mean={field: float(value) if value is not None else None for field, value in delta.items()},
                exact_parent_minus_native_mean=delta, exact_population_parent_minus_native_mean=components,
                delta_cost_vs_competing_MW=float(delta["cost"] - exact(target["competing_MW_reference"]["exact_parent_minus_native_mean"]["cost"]))
                    if available else None,
                sample_preferences={name: sum(preference(value) == name for value in values) for name in ("parent", "native", "tie")}
                    if available else None,
                leave_one_out_mean_range=[float(min(loo)), float(max(loo))] if available else None,
                paired_roster_and_stream_checks_passed=all(row["execution_checks_passed"] for row in cells))))
    valid = [row["result"]["flow"] for row in rows if row["execution_checks_passed"]]
    failed = [row["index"] for row in rows if not row["execution_checks_passed"]]
    summary = dict(registered_at_utc=protocol["registered_at_utc"], source_horizon_registered_at_utc=protocol["source_horizon_registered_at_utc"],
        source_competing_terminal_registered_at_utc=protocol["source_competing_terminal_registered_at_utc"],
        decision="NATIVE_TERMINAL_FIXED_THREE_COMPLETE" if not failed else "NATIVE_TERMINAL_WITH_FAILED_UNITS",
        execution_checks_passed=not failed, recorded_root_sample_units=48, completed_root_sample_units=48 - len(failed),
        failed_units=failed, available_state_rankings=sum(record["comparison"]["available"] for record in records), records=records,
        reused_native_H_units=48, new_anchor_proposals=0, true_service_suffix_executions=0,
        new_terminal_native_controller_calls=sum(value.get("calls", 0) for value in valid),
        new_terminal_native_idle_calls=sum(value.get("idle_calls", 0) for value in valid),
        work={key: _work([value.get(key, {}) for value in valid]) for key in ("regeneration_work", "incremental_tail_work",
            "reused_window_work", "restore_work", "actual_work", "logical_work", "proposal_status", "preprocessing")},
        runtime={key: sum(value.get(key, 0) for value in valid) for key in ("constructor_cpu_s", "constructor_wall_s",
            "worker_cpu_s", "worker_wall_s", "total_cpu_s", "total_wall_s", "controller_constructors")},
        full24_decision=protocol["full24_decision"], performance_gain_established=False,
        score_units="Eight-path mean of total job-seconds; not mean flow time per job.",
        interpretation="Same H population and sampled background stream, with only terminal control replaced by the original finite controller. Native proposal work including idle is charged; all cap/failed partial records remain on server. Actual H and later cohort labels are conditional context. A ranking flip is not a full-policy gain; preserve seed41 tail harm and original full development gates.")
    dump(path(directory, "_summary.json"), jsonable(summary))
    print(f"complete units=48 available_rankings={summary['available_state_rankings']} failed={len(failed)} full24=HOLD", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--shard-index", type=int, choices=range(48))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir, args.workers)
    elif args.merge:
        merge(args.artifact_dir)
    else:
        run_shard(args.shard_index, args.artifact_dir)


if __name__ == "__main__":
    main()
