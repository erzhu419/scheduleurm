"""Fixed-three common-horizon flow ranking, reusing complete native preparation."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import json
import pickle
import socket
from time import perf_counter

from algorithm.experiments import aligned_hardware_finite_policy_tagged_ranking as tagged
from algorithm.experiments.common_horizon_flow_rollout import evaluate_root_sample

ROOT, ARTIFACTS = tagged.ROOT, tagged.ARTIFACTS
STEM = "aligned_hardware_common_horizon_flow_20261003_r1"
SOURCES = ("algorithm/experiments/aligned_hardware_common_horizon_flow.py",
    "algorithm/experiments/common_horizon_flow_rollout.py")
read, dump, require, same, jsonable = tagged.read, tagged.dump, tagged.require, tagged.same, tagged.jsonable
SCORE_FIELDS = ("cost", "common_area", "waiting_area", "inflight_area", "terminal_cost")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def _exact(value):
    return Fraction.from_float(float(value))


def _cached_state(target, source_epoch):
    """Recover physical commitments without another proposal or service replay."""
    prepared = read(target["prepared_row"])
    require(prepared["registered_at_utc"] == source_epoch
        and prepared["execution_checks_passed"] and prepared["tape_id"] == target["tape_id"],
        "completed original native preparation differs")
    with Path(prepared["fixture_file"]).open("rb") as source:
        fixture = pickle.load(source)
    with Path(target["source_fixture"]).open("rb") as source:
        physical = pickle.load(source)
    require(fixture["registered_at_utc"] == source_epoch and fixture["tape_id"] == target["tape_id"],
        "prepared fixture identity differs")
    frame, context, checkpoint = physical["finite_frame"], fixture["context"], physical["checkpoint"]
    require(fixture["seed"] == physical["seed"] and fixture["options"] == physical["model"].options
        and same(context["queue"], frame["queue"])
        and same(context["available"], frame["available_before"])
        and context["now"] == frame["start"] and context["lane"] == frame["lane_ids"][0]
        and same(context["ages"], frame["waiting_ages"])
        and same(context["counts"], frame["observed_arrival_counts"]),
        "cached finite anchor context or catalogue differs")
    require(len(fixture["before_tree"]["planners"]) == len(fixture["after_tree"]["planners"]) == 3
        and fixture["before_tree"]["planners"][0]["decision_counter"] == target["frame_index"]
        and fixture["after_tree"]["planners"][0]["decision_counter"] == target["frame_index"] + 1,
        "complete consumed finite anchor tree required")
    jobs, anchor = checkpoint["jobs"], context["now"]
    require(same(checkpoint["available"], context["available"])
        and same(checkpoint["observed_counts"], context["counts"])
        and tuple(map(len, checkpoint["queues"])) == tuple(context["queue"])
        and tuple(tuple(anchor - jobs[index]["arrival"] for index in group)
            for group in checkpoint["queues"]) == tuple(map(tuple, context["ages"])),
        "committed checkpoint does not reproduce the visible FIFO or lane release times")
    prefix = physical["finite_prefix"]
    admitted = [sum(old["service"][coordinate] for old in prefix["frames"][:target["frame_index"]])
        for coordinate in range(len(context["queue"]))]
    require(admitted == [sum(job["coordinate"] == coordinate and job["start"] is not None for job in jobs)
        for coordinate in range(len(context["queue"]))], "checkpoint admissions include anchor/future service")
    actual = {(job["coordinate"], job["index"]): job for job in prefix["jobs"]}
    inflight = []
    for job in jobs:
        if job["start"] is None:
            require(job["completion"] is None, "unadmitted checkpoint job has a committed completion")
            continue
        original = actual[job["coordinate"], job["index"]]
        require(job["start"] <= anchor and job["completion"] is not None
            and job["start"] == original["start"] and job["completion"] == original["completion"],
            "pre-anchor service commitment changed")
        if job["completion"] > anchor:
            inflight.append(dict(id=("committed", job["coordinate"], job["index"]),
                coordinate=job["coordinate"], birth=job["arrival"], start=job["start"],
                completion=job["completion"]))
    past = sum((_exact(age) for group in context["ages"] for age in group), Fraction(0))
    past += sum((_exact(anchor) - _exact(job["birth"]) for job in inflight), Fraction(0))
    return prepared, fixture, tuple(inflight), past


def register(directory, workers):
    prior = tagged.load_protocol(directory)
    completed = read(tagged.path(directory, "_summary.json"))
    design = read(path(directory, "_design.json"))
    require(completed["registered_at_utc"] == prior["registered_at_utc"]
        and completed["execution_checks_passed"] and completed["completed_root_sample_units"] == 48
        and prior["planning_samples"] == design["planning_samples"] == 8
        and prior["planning_event_cap"] == design["planning_event_cap"] == 128
        and design["source_tagged_registered_at_utc"] == prior["registered_at_utc"]
        and design["horizon_tau_max_multiple"] == 2
        and prior["full24_decision"] == "PERSISTENT_CURRENT_POLICY_DEVELOPMENT_HOLD",
        "completed fixed-three original8/cap128 diagnostic and saved H2tau design required")
    targets = []
    for old, record, designed in zip(prior["targets"], completed["records"], design["targets"], strict=True):
        require(all(old[key] == designed[key] for key in designed)
            and record["tape_id"] == old["tape_id"] and record["frame_index"] == old["frame_index"],
            "saved design must retain exactly the three original states and two roots")
        target = {key: old[key] for key in ("index", "tape_id", "seed", "frame_index", "parent_root", "native_root", "source_fixture")}
        target["prepared_row"] = str(tagged.path(directory, f"_prepare_{old['index']:02d}.json"))
        prepared, fixture, inflight, past = _cached_state(target, prior["registered_at_utc"])
        require(len(fixture["paths"]) == 8 and len(prepared["tagged_populations"]) == 8
            and not any(sample.overflow for sample in fixture["paths"]), "all original eight complete paths required")
        for role in ("parent", "native"):
            root = target[role + "_root"]
            require(root in prepared["original_risk_roots"] and all(need <= count for need, count in
                zip(fixture["options"][root].requirements, fixture["context"]["queue"], strict=True)),
                "original risk membership and feasible physical roots required")
        tau = max(max(1., float(option.duration)) for option in fixture["options"])
        target.update(initial_inflight=jsonable(inflight), omitted_common_past_S1=jsonable(past),
            horizon_relative_s=2 * tau, original_risk_roots=prepared["original_risk_roots"],
            previous_comparisons=record["comparisons"], birth_cohort_root_effects=record["birth_cohort_root_effects"])
        targets.append(target)
    assignments = [dict(index=16 * target["index"] + 8 * slot + sample,
        target_index=target["index"], tape_id=target["tape_id"], root=target[role + "_root"],
        role=role, sample_index=sample)
        for target in targets for slot, role in enumerate(("parent", "native")) for sample in range(8)]
    protocol = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        source_tagged_registered_at_utc=prior["registered_at_utc"], planning_samples=8,
        planning_event_cap=128, horizon_tau_max_multiple=2, targets=targets, assignments=assignments,
        root_sample_units=48, new_native_anchor_proposals=0, reused_native_anchor_proposals=3,
        true_service_suffix_executions=0, full24_decision=prior["full24_decision"],
        method=design["candidate_score"], continuation=design["continuation"],
        common_randomness=design["common_randomness"], terminal_rule=design["terminal_rule"],
        past_cost=design["past_cost"], score_replacement=design["score_replacement"],
        overflow="Retain all48 original units and exact partial costs/work. A cap overflow or failed unit makes its root unavailable; paired ranking requires all16 fixed units complete. No dropped paths, resampling, changed cap, new anchors or horizon sweep.",
        interpretation="Offline fixed-three S1 ranking hypothesis. Original BG S2 risk membership and feasibility remain fixed; old approximate/tagged score deltas and actual cohort root effects are separate context. No production promotion, full24 execution, confirmation or paper change; performance gain is not established.",
        execution_sources={**prior["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}})
    dump(path(directory, "_protocol.json"), protocol)
    print(f"registered_at_utc={protocol['registered_at_utc']} reused_prepares=3 units=48 workers={workers}", flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_shard(index, directory):
    protocol = load_protocol(directory)
    assignment = protocol["assignments"][index]
    target = protocol["targets"][assignment["target_index"]]
    prepared, fixture, inflight, past = _cached_state(target, protocol["source_tagged_registered_at_utc"])
    require(jsonable(inflight) == target["initial_inflight"]
        and jsonable(past) == target["omitted_common_past_S1"], "registered anchor commitments changed")
    started = perf_counter()
    print(f"unit_start index={index} role={assignment['role']} root={assignment['root']} sample={assignment['sample_index']}", flush=True)
    row = dict(index=index, registered_at_utc=protocol["registered_at_utc"], assignment=assignment,
        execution_host=socket.gethostname())

    def progress(fields):
        phase = fields["phase"]
        payload = {**fields, "current": int(phase in ("complete", "overflow")),
            "total": 1, "unit": "root_sample", "elapsed_seconds": fields["elapsed_wall_s"],
            "phase_index": 1 if phase == "native" else 2, "phase_count": 2, "shard_index": index}
        print("SCHEDULEURM_PROGRESS " + json.dumps(payload, separators=(",", ":")), flush=True)

    try:
        result = evaluate_root_sample(fixture["options"], fixture["seed"], fixture["paths"],
            fixture["before_tree"], fixture["after_tree"], fixture["context"],
            assignment["root"], assignment["sample_index"], initial_inflight=inflight,
            workers=protocol["workers"], progress=progress)
        require(result["root"] == assignment["root"] and result["sample_index"] == assignment["sample_index"],
            "fixed root/global-sample unit differs")
        require(result["flow"]["cost"] == result["flow"]["common_area"] + result["flow"]["terminal_cost"],
            "common-horizon and terminal components do not reconstruct the S1 score")
        require(result["flow"]["common_area"] == result["flow"]["waiting_area"] + result["flow"]["inflight_area"]
            and result["flow"]["past_flow_constant"] == past
            and result["flow"]["horizon"] == _exact(target["horizon_relative_s"]),
            "all unfinished area, omitted common past cost or fixed horizon differs")
        row.update(execution_checks_passed=True, result=result, wall_s=perf_counter() - started)
    except Exception as error:
        row.update(execution_checks_passed=False, result=None, error=dict(type=type(error).__name__,
            message=str(error)), wall_s=perf_counter() - started)
        dump(path(directory, f"_shard_{index:02d}.json"), jsonable(row))
        raise
    dump(path(directory, f"_shard_{index:02d}.json"), jsonable(row))
    print(f"unit_complete index={index} complete={result['flow']['complete']} overflow={result['flow']['overflow']}", flush=True)


def _root_scores(samples):
    failures = [row["assignment"]["sample_index"] for row in samples if not row["execution_checks_passed"]]
    values = [row["result"]["flow"] for row in samples if row["execution_checks_passed"]]
    total = {field: sum((Fraction(*value[field]) for value in values), Fraction(0)) for field in SCORE_FIELDS}
    available = not failures and len(values) == 8 and all(value["complete"] and not value["overflow"] for value in values)
    populations = [value["population"] for value in values]

    def group_total(groups):
        return {field: sum((Fraction(*group[field]) for group in groups), Fraction(0))
            for field in SCORE_FIELDS} | {
            field: sum(group[field] for group in groups) for field in ("total", "admitted")}

    population_totals = dict(sources={source: group_total([population["sources"][source] for population in populations])
        for source in ("visible", "prefix", "background", "committed")},
        per_coordinate=[dict(coordinate=coordinate,
            **group_total([population["per_coordinate"][coordinate] for population in populations]),
            sources={source: group_total([population["per_coordinate"][coordinate]["sources"][source]
                for population in populations]) for source in ("visible", "prefix", "background", "committed")})
            for coordinate in range(len(populations[0]["per_coordinate"]))]) if populations else None
    return dict(available=available, exact_partial_totals=total, population_totals=population_totals,
        mean={field: float(total[field] / 8) if available else None for field in SCORE_FIELDS},
        failed_samples=failures, overflow_samples=[row["assignment"]["sample_index"] for row in samples
            if row["execution_checks_passed"] and row["result"]["flow"]["overflow"]],
        incomplete_samples=[row["assignment"]["sample_index"] for row in samples
            if row["execution_checks_passed"] and not row["result"]["flow"]["complete"]])


def merge(directory):
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_shard_{index:02d}.json")) for index in range(48)]
    for assignment, row in zip(protocol["assignments"], rows, strict=True):
        require(row["index"] == assignment["index"] and row["assignment"] == assignment
            and row["registered_at_utc"] == protocol["registered_at_utc"], "all48 original unit records required")
    records = []
    for target in protocol["targets"]:
        cells = rows[16 * target["index"]:16 * (target["index"] + 1)]
        samples = []
        for sample in range(8):
            pair = [cells[8 * slot + sample] for slot in range(2)]
            usable = all(row["execution_checks_passed"] for row in pair)
            if usable:
                p, n = (row["result"] for row in pair)
                require(p["environment"] == n["environment"], "registered paired environment differs")
                streams = [result["flow"]["background_events"] for result in (p, n)]
                longest = max(streams, key=len)
                require(all(stream == longest[:len(stream)] for stream in streams), "paired fixed background birth prefixes differ")
                usable = all(result["flow"]["complete"] and not result["flow"]["overflow"] for result in (p, n))
                if usable:
                    roster = lambda result: [{key: job[key] for key in ("id", "coordinate", "birth", "source")}
                        for job in result["flow"]["population"]["jobs"]]
                    require(roster(p) == roster(n), "complete paired roots must score the identical all-job roster")
            delta = {field: (Fraction(*pair[0]["result"]["flow"][field]) - Fraction(*pair[1]["result"]["flow"][field]))
                if usable else None for field in SCORE_FIELDS}
            samples.append(dict(sample_index=sample, available=usable,
                parent_minus_native={field: float(value) if value is not None else None for field, value in delta.items()},
                exact_parent_minus_native=delta))
        roots = {role: dict(root=target[role + "_root"], **_root_scores(cells[8 * slot:8 * (slot + 1)]))
            for slot, role in enumerate(("parent", "native"))}
        available = all(root["available"] for root in roots.values())
        delta = {field: (roots["parent"]["exact_partial_totals"][field] - roots["native"]["exact_partial_totals"][field]) / 8
            if available else None for field in SCORE_FIELDS}
        preference = ("parent" if delta["cost"] < 0 else "native" if delta["cost"] > 0 else "tie") if available else "unavailable"
        population_delta = None
        if available:
            population_delta = dict(sources={source: {field:
                (roots["parent"]["population_totals"]["sources"][source][field]
                    - roots["native"]["population_totals"]["sources"][source][field]) / 8
                for field in SCORE_FIELDS} for source in ("visible", "prefix", "background", "committed")},
                per_coordinate=[dict(coordinate=coordinate, **{field:
                    (roots["parent"]["population_totals"]["per_coordinate"][coordinate][field]
                        - roots["native"]["population_totals"]["per_coordinate"][coordinate][field]) / 8
                    for field in SCORE_FIELDS}) for coordinate in range(len(roots["parent"]["population_totals"]["per_coordinate"]))])
        records.append(dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "parent_root", "native_root",
            "initial_inflight", "omitted_common_past_S1", "horizon_relative_s", "original_risk_roots",
            "previous_comparisons", "birth_cohort_root_effects")}, roots=roots, samples=samples,
            comparison=dict(available=available, S1_preference=preference,
                parent_minus_native_mean={field: float(value) if value is not None else None for field, value in delta.items()},
                exact_parent_minus_native_mean=delta, exact_population_parent_minus_native_mean=population_delta),
            unit_work=[dict(assignment=row["assignment"], wall_s=row["wall_s"], execution_checks_passed=row["execution_checks_passed"],
                error=row.get("error"), flow_work={key: value for key, value in row["result"]["flow"].items()
                    if "work" in key or "calls" in key or key.endswith("_s")} if row["execution_checks_passed"] else None)
                for row in cells]))
    failed = [row["index"] for row in rows if not row["execution_checks_passed"]]
    summary = dict(registered_at_utc=protocol["registered_at_utc"], source_tagged_registered_at_utc=protocol["source_tagged_registered_at_utc"],
        decision="COMMON_HORIZON_FLOW_FIXED_THREE_COMPLETE" if not failed else "COMMON_HORIZON_FLOW_FIXED_THREE_WITH_FAILED_UNITS",
        execution_checks_passed=not failed, recorded_root_sample_units=len(rows),
        completed_root_sample_units=len(rows) - len(failed), failed_units=failed,
        available_state_rankings=sum(record["comparison"]["available"] for record in records),
        new_native_anchor_proposals=0, reused_native_anchor_proposals=3, true_service_suffix_executions=0,
        full24_decision=protocol["full24_decision"], performance_gain_established=False, records=records,
        interpretation=protocol["interpretation"], overflow_rule=protocol["overflow"])
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
