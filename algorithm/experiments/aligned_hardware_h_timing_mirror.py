"""Fixed-three within-H arrival timing mirror, with original native control."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import socket
from time import perf_counter

from algorithm.experiments import aligned_hardware_native_terminal_flow as prior
from algorithm.experiments.h_timing_mirror_rollout import evaluate_cached

ROOT, ARTIFACTS = prior.ROOT, prior.ARTIFACTS
STEM = "aligned_hardware_H_timing_mirror_20261003"
DESIGN = "md/experiment_artifacts/" + STEM + "_design.json"
SOURCES = ("algorithm/experiments/aligned_hardware_h_timing_mirror.py",
    "algorithm/experiments/h_timing_mirror_rollout.py", DESIGN)
FIRST_FOUR = (0, 1, 2, 3)
SECOND_FOUR = (4, 5, 6, 7)
FIELDS, GROUPS = prior.FIELDS + ("full_waiting_cost", "full_inflight_cost"), prior.GROUPS
read, dump, require, jsonable, exact, preference = prior.read, prior.dump, prior.require, prior.jsonable, prior.exact, prior.preference


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def _fixture(target, source_epoch):
    _, fixture, inflight, past = prior.horizon._cached_state(target, source_epoch)
    require(jsonable(inflight) == target["initial_inflight"]
        and jsonable(past) == target["omitted_common_past_S1"], "registered anchor commitments changed")
    for role in ("parent", "native"):
        root = target[role + "_root"]
        require(root in target["original_risk_roots"] and all(need <= count for need, count in
            zip(fixture["options"][root].requirements, fixture["context"]["queue"], strict=True)),
            "unchanged feasible roots in original BG-risk gate required")
    return fixture, inflight


def register(directory, workers):
    design = read(path(directory, "_design.json"))
    old, horizon = prior.load_protocol(directory), prior.horizon.load_protocol(directory)
    completed = read(prior.path(directory, "_summary.json"))
    require(design["source_native_terminal_epoch"] == old["registered_at_utc"] == completed["registered_at_utc"]
        and design["source_H_epoch"] == horizon["registered_at_utc"] == old["source_horizon_registered_at_utc"]
        and completed["execution_checks_passed"] and completed["completed_root_sample_units"] == 48
        and completed["available_state_rankings"] == 3
        and old["planning_samples"] == design["planning_samples"] == 8
        and old["planning_event_cap"] == design["planning_event_cap"] == 128
        and design["new_root_sample_units"] == 48 and design["new_anchor_proposals"] == 0
        and design["paired_budget_projection"]["primary_sample_indices"] == list(FIRST_FOUR)
        and design["paired_budget_projection"]["separate_sensitivity_indices"] == list(SECOND_FOUR)
        and design["paired_budget_projection"]["paths_per_projection"] == 8
        and old["assignments"] == horizon["assignments"]
        and old["full24_decision"] == design["full24_decision"] == "PERSISTENT_CURRENT_POLICY_DEVELOPMENT_HOLD",
        "completed original48 H/native epochs, eight paths/cap128, unchanged roots and HOLD required")
    targets = []
    for oldtarget, htarget, record, state in zip(old["targets"], horizon["targets"], completed["records"], design["states"], strict=True):
        require(all(state[key] == oldtarget[key] == htarget[key] == record[key]
            for key in ("seed", "frame_index", "parent_root", "native_root"))
            and record["comparison"]["available"] and oldtarget["original_risk_roots"] == htarget["original_risk_roots"],
            "retain original three states, risk membership and fixed two roots")
        target = {**oldtarget, **{key: htarget[key] for key in
            ("source_fixture", "initial_inflight", "omitted_common_past_S1")},
            "native_terminal_reference": record["comparison"]}
        _fixture(target, old["source_tagged_registered_at_utc"])
        targets.append(target)
    references = []
    for assignment, source_file in zip(old["assignments"], old["source_shards"], strict=True):
        cached = read(source_file)
        reference_file = prior.path(directory, f"_shard_{assignment['index']:02d}.json")
        reference = read(reference_file)
        require(cached["registered_at_utc"] == horizon["registered_at_utc"]
            and reference["registered_at_utc"] == old["registered_at_utc"]
            and cached["assignment"] == reference["assignment"] == assignment
            and all(row["execution_checks_passed"] and row["result"]["flow"]["complete"]
                and not row["result"]["flow"]["overflow"] for row in (cached, reference))
            and reference["result"]["environment"] == cached["result"]["environment"]
            and len(reference["result"]["flow"]["initial_tree"]["planners"]) == 3,
            "all48 original completed native references and H caches required")
        references.append(str(reference_file))
    protocol = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        source_native_terminal_registered_at_utc=old["registered_at_utc"],
        source_horizon_registered_at_utc=horizon["registered_at_utc"],
        source_tagged_registered_at_utc=old["source_tagged_registered_at_utc"],
        planning_samples=8, planning_event_cap=128, horizon_tau_max_multiple=2,
        root_sample_units=48, targets=targets, assignments=old["assignments"],
        source_shards=old["source_shards"], native_reference_shards=references, design=design,
        within_eight_primary_sample_indices=FIRST_FOUR, separate_second_four_sample_indices=SECOND_FOUR,
        full24_decision=old["full24_decision"],
        execution_sources={**old["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}})
    dump(path(directory, "_protocol.json"), protocol)
    print(f"registered_at_utc={protocol['registered_at_utc']} new_H_units=48 workers={workers}", flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def _check_roster(result, cached):
    old = {tuple(job["id"]): job for job in cached["flow"]["population"]["jobs"]}
    new = {tuple(job["id"]): job for job in result["scored_population"]}
    require(len(old) == len(new) == len(result["scored_population"]) and old.keys() == new.keys(),
        "mirror must retain every fixed H population identity")
    horizon, tau = exact(cached["flow"]["horizon"]), Fraction.from_float(cached["environment"]["window_s"])
    for identity, job in old.items():
        observed = new[identity]
        birth = exact(job["birth"])
        newborn = job["source"] in ("prefix", "background")
        expected_birth = horizon - birth if newborn else birth
        expected_source = ("prefix" if expected_birth <= tau else "background") if newborn else job["source"]
        require(observed["coordinate"] == job["coordinate"] and exact(observed["birth"]) == expected_birth
            and observed["source"] == expected_source, "mirror birth/source or fixed visible/committed birth differs")


def _stream_prefixes(results):
    streams = [jsonable([event for event in result["flow"]["background_events"]
        if exact(event[0]) > exact(result["flow"]["horizon"])]) for result in results]
    longest = max(streams, key=len)
    require(all(stream == longest[:len(stream)] for stream in streams),
        "paired original/mirror external suffix streams differ")


def run_shard(index, directory):
    protocol = load_protocol(directory)
    assignment = protocol["assignments"][index]
    target = protocol["targets"][assignment["target_index"]]
    fixture, inflight = _fixture(target, protocol["source_tagged_registered_at_utc"])
    cached, original = read(protocol["source_shards"][index]), read(protocol["native_reference_shards"][index])
    require(cached["registered_at_utc"] == protocol["source_horizon_registered_at_utc"]
        and original["registered_at_utc"] == protocol["source_native_terminal_registered_at_utc"]
        and cached["assignment"] == original["assignment"] == assignment
        and cached["execution_checks_passed"] and original["execution_checks_passed"], "registered source unit identity changed")
    started = perf_counter()
    row = dict(index=index, registered_at_utc=protocol["registered_at_utc"], assignment=assignment,
        execution_host=socket.gethostname())
    print(f"unit_start index={index} role={assignment['role']} sample={assignment['sample_index']}", flush=True)

    def progress(fields):
        payload = {**fields, "current": int(fields["phase"] in ("complete", "overflow")), "total": 1,
            "unit": "root_sample", "elapsed_seconds": fields["elapsed_wall_s"], "shard_index": index}
        print("SCHEDULEURM_PROGRESS " + json.dumps(payload, separators=(",", ":")), flush=True)

    result = None
    try:
        result = evaluate_cached(fixture["options"], fixture["seed"], cached["result"], fixture["context"],
            fixture["after_tree"], initial_inflight=inflight, workers=protocol["workers"],
            sample_count=8, event_cap=128, progress=progress)
        flow = result["flow"]
        require(result["root"] == assignment["root"] and result["sample_index"] == assignment["sample_index"]
            and result["environment"] == cached["result"]["environment"] and result["mirror"]
            and jsonable(flow["initial_tree"]) == jsonable(fixture["after_tree"])
            and flow["horizon"] == exact(cached["result"]["flow"]["horizon"])
            and flow["past_flow_constant"] == exact(target["omitted_common_past_S1"]),
            "original root/sample, after-anchor tree, horizon, rates and committed past must remain fixed")
        _check_roster(result, cached["result"])
        require(jsonable(flow["original_h_stream"]["events"]) == cached["result"]["flow"]["background_events"]
            and [flow["regeneration_work"][key] for key in
                ("background_events", "random_draws", "rate_reads", "seed_initializations")]
                == cached["result"]["flow"]["suffix_work"], "original H suffix clocks/counters must be regenerated exactly")
        _stream_prefixes((result, original["result"]))
        require(flow["cost"] == flow["common_area"] + flow["terminal_cost"]
            and flow["common_area"] == flow["waiting_area"] + flow["inflight_area"], "mirror costs must reconstruct")
        row.update(execution_checks_passed=True, result=result, wall_s=perf_counter() - started)
        progress(dict(phase="complete" if flow["complete"] else "overflow", elapsed_wall_s=perf_counter() - started))
    except Exception as error:
        row.update(execution_checks_passed=False, result=result, wall_s=perf_counter() - started,
            error=dict(type=type(error).__name__, message=str(error)))
        dump(path(directory, f"_shard_{index:02d}.json"), jsonable(row))
        raise
    dump(path(directory, f"_shard_{index:02d}.json"), jsonable(row))
    print(f"unit_complete index={index} complete={flow['complete']} overflow={flow['overflow']}", flush=True)


def _group_costs(jobs, flow):
    """Reuse the completed post-H diagnostic's full waiting/processing identity."""
    totals = {field: Fraction(0) for field in FIELDS}
    horizon = exact(flow["horizon"])
    partial = exact(flow["partial_terminal_observation_end"]) if flow["overflow"] else None
    for job in jobs:
        begin = max(Fraction(0), exact(job["birth"]))
        start = exact(job["start"]) if job["start"] is not None else None
        completion = exact(job["completion"]) if job["completion"] is not None else None
        waiting = max(Fraction(0), min(horizon, start if start is not None else horizon) - begin)
        processing = max(Fraction(0), min(horizon, completion) - max(begin, start)) if completion is not None else Fraction(0)
        terminal = max(Fraction(0), (completion if completion is not None else partial) - horizon)
        full_waiting = max(Fraction(0), (start if start is not None else partial) - begin)
        full_processing = completion - max(begin, start) if completion is not None else Fraction(0)
        require(waiting + processing + terminal == full_waiting + full_processing, "partial/full waiting and processing costs must reconstruct")
        for field, value in zip(FIELDS, (waiting + processing + terminal, waiting + processing,
                waiting, processing, terminal, full_waiting, full_processing), strict=True):
            totals[field] += value
    return totals


def _costs(rows):
    flows = [row["result"]["flow"] for row in rows if row["execution_checks_passed"]]
    available = len(flows) == 8 and all(flow["complete"] and not flow["overflow"] for flow in flows)

    def grouped(coordinate=None, source=None):
        values = [_group_costs([job for job in flow["population"]["jobs"]
            if (coordinate is None or job["coordinate"] == coordinate) and (source is None or job["source"] == source)], flow) for flow in flows]
        return {field: sum((value[field] for value in values), Fraction(0)) for field in FIELDS}

    totals = grouped()
    require(all(totals[field] == sum((exact(flow[field]) for flow in flows), Fraction(0)) for field in prior.FIELDS),
        "job decomposition must reconstruct cached score components")
    return dict(available=available, exact_partial_totals=totals,
        mean={field: float(value / 8) if available else None for field, value in totals.items()},
        population_totals=dict(sources={source: grouped(source=source) for source in GROUPS},
            per_coordinate=[dict(coordinate=c, **grouped(coordinate=c)) for c in range(3)]),
        overflow_samples=[row["assignment"]["sample_index"] for row in rows if row["execution_checks_passed"] and row["result"]["flow"]["overflow"]],
        failed_samples=[row["assignment"]["sample_index"] for row in rows if not row["execution_checks_passed"]])


def _delta(roots):
    return {field: (roots["parent"]["exact_partial_totals"][field] - roots["native"]["exact_partial_totals"][field]) / 8
        if all(root["available"] for root in roots.values()) else None for field in FIELDS}


def _four_pairs(samples, indices):
    chosen = [samples[index] for index in indices]
    available = all(sample["available"] for sample in chosen)
    pairs = [(sample["exact_original_parent_minus_native_cost"] + sample["exact_parent_minus_native_cost"]) / 2
        for sample in chosen] if available else []
    mean = sum(pairs, Fraction(0)) / 4 if available else None
    old = sum((sample["exact_original_parent_minus_native_cost"] for sample in chosen), Fraction(0)) / 4
    loo = [(sum(pairs, Fraction(0)) - value) / 3 for value in pairs] if available else []
    return dict(sample_indices=list(indices), score_paths=8, original_paths=4, mirror_paths=4, available=available,
        exact_parent_minus_native_cost=mean, parent_minus_native_cost=float(mean) if available else None,
        preference=preference(mean) if available else "unavailable", original_same_four_mean=float(old),
        original_same_four_preference=preference(old), paired_leave_one_out_mean_range=[float(min(loo)), float(max(loo))] if available else None,
        pair_preferences={name: sum(preference(value) == name for value in pairs) for name in ("parent", "native", "tie")} if available else None,
        interpretation="Fixed original/mirror pairs; differs from the original eight-independent-path roster. Exploratory projected score, not evaluated policy performance.")


def merge(directory):
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_shard_{index:02d}.json")) for index in range(48)]
    originals = [read(source) for source in protocol["native_reference_shards"]]
    cached = [read(source) for source in protocol["source_shards"]]
    for assignment, row, original, hrow in zip(protocol["assignments"], rows, originals, cached, strict=True):
        require(row["index"] == assignment["index"] and row["assignment"] == original["assignment"] == assignment
            and row["registered_at_utc"] == protocol["registered_at_utc"]
            and original["registered_at_utc"] == protocol["source_native_terminal_registered_at_utc"]
            and hrow["assignment"] == assignment and hrow["registered_at_utc"] == protocol["source_horizon_registered_at_utc"],
            "all48 registered paired units required")
    records = []
    for target in protocol["targets"]:
        cells, oldcells, hcells = (items[16 * target["index"]:16 * (target["index"] + 1)] for items in (rows, originals, cached))
        roots, oldroots = ({role: dict(root=target[role + "_root"], **_costs(items[8 * slot:8 * (slot + 1)]))
            for slot, role in enumerate(("parent", "native"))} for items in (cells, oldcells))
        delta, olddelta = _delta(roots), _delta(oldroots)
        available = all(root["available"] for root in roots.values())
        samples, values = [], []
        for index in range(8):
            pair, oldpair = ([items[8 * slot + index] for slot in range(2)] for items in (cells, oldcells))
            valid = all(row["execution_checks_passed"] for row in pair)
            if valid:
                p, n = (row["result"] for row in pair)
                require(p["environment"] == n["environment"] and p["scored_population"] == n["scored_population"]
                    and p["flow"]["original_h_stream"] == n["flow"]["original_h_stream"], "paired mirror roots must share roster and original stream state")
                _stream_prefixes([row["result"] for row in pair + oldpair])
            usable = valid and all(row["result"]["flow"]["complete"] and not row["result"]["flow"]["overflow"] for row in pair)
            oldcost = exact(oldpair[0]["result"]["flow"]["cost"]) - exact(oldpair[1]["result"]["flow"]["cost"])
            cost = exact(pair[0]["result"]["flow"]["cost"]) - exact(pair[1]["result"]["flow"]["cost"]) if usable else None
            if usable:
                values.append(cost)
            samples.append(dict(sample_index=index, available=usable, original_parent_minus_native_cost=float(oldcost),
                exact_original_parent_minus_native_cost=oldcost, parent_minus_native_cost=float(cost) if usable else None,
                exact_parent_minus_native_cost=cost, original_preference=preference(oldcost),
                preference=preference(cost) if usable else "unavailable", changed_preference=preference(cost) != preference(oldcost) if usable else None,
                H_queues={row["assignment"]["role"]: row["result"]["flow"]["window_flow"]["terminal_state"]["queue"] for row in pair} if valid else None,
                original_H_queues={row["assignment"]["role"]: row["result"]["flow"]["terminal_state"]["queue"]
                    for row in (hcells[index], hcells[8 + index])}))
        components = None
        if available:
            require(sum(values, Fraction(0)) / 8 == delta["cost"], "eight paired mirror paths must reconstruct mean")
            components = dict(sources={source: {field: (roots["parent"]["population_totals"]["sources"][source][field]
                - roots["native"]["population_totals"]["sources"][source][field]) / 8 for field in FIELDS} for source in GROUPS},
                per_coordinate=[dict(coordinate=c, **{field: (roots["parent"]["population_totals"]["per_coordinate"][c][field]
                    - roots["native"]["population_totals"]["per_coordinate"][c][field]) / 8 for field in FIELDS}) for c in range(3)])
        loo = [(sum(values, Fraction(0)) - value) / 7 for value in values] if available else []
        records.append(dict(**{key: target[key] for key in ("tape_id", "seed", "frame_index", "parent_root", "native_root", "original_risk_roots", "actual_context")},
            roots=roots, samples=samples, comparison=dict(available=available,
                S1_preference=preference(delta["cost"]) if available else "unavailable",
                parent_minus_native_mean={field: float(value) if value is not None else None for field, value in delta.items()},
                exact_parent_minus_native_mean=delta, exact_population_parent_minus_native_mean=components,
                original_parent_minus_native_mean={field: float(value) for field, value in olddelta.items()},
                original_preference=preference(olddelta["cost"]),
                delta_vs_original={field: float(delta[field] - olddelta[field]) if available else None for field in FIELDS},
                changed_sample_preferences=sum(sample["changed_preference"] is True for sample in samples),
                sample_preferences={name: sum(preference(value) == name for value in values) for name in ("parent", "native", "tie")} if available else None,
                leave_one_out_mean_range=[float(min(loo)), float(max(loo))] if available else None),
            within_eight_primary_four_pairs=_four_pairs(samples, FIRST_FOUR),
            separate_second_four_pairs=_four_pairs(samples, SECOND_FOUR)))
    valid = [row["result"]["flow"] for row in rows if row["execution_checks_passed"]]
    measured = [row["result"]["flow"] for row in rows if row["result"] is not None]
    failed = [row["index"] for row in rows if not row["execution_checks_passed"]]
    overflow = [row["index"] for row in rows if row["execution_checks_passed"] and row["result"]["flow"]["overflow"]]
    work = {key: prior._work([flow.get(key, {}) for flow in measured]) for key in ("regeneration_work", "reflection_work",
        "incremental_tail_work", "restore_work", "actual_work", "h_actual_work", "terminal_actual_work", "diagnostic_drain_work", "preprocessing")}
    work["window_logical_work"] = prior._work([flow["window_flow"]["logical_work"] for flow in measured])
    work["window_proposal_status"] = prior._work([flow["window_flow"]["proposal_status"] for flow in measured])
    work["window_dispatch_work"] = prior._work([flow["window_flow"]["dispatch_work"] for flow in measured])
    work["terminal_logical_work"] = prior._work([flow["logical_work"] for flow in measured])
    work["terminal_proposal_status"] = prior._work([flow["proposal_status"] for flow in measured])
    runtime = {key: sum(flow.get(key, 0) for flow in measured) for key in ("constructor_cpu_s", "constructor_wall_s", "worker_cpu_s",
        "worker_wall_s", "total_cpu_s", "total_wall_s", "controller_constructors")}
    runtime["window_worker_cpu_s"] = sum(flow["window_flow"]["worker_cpu_s"] for flow in measured)
    runtime["child_worker_cpu_s"] = work["actual_work"].get("parallel_root_work", {}).get("worker_cpu_s", 0)
    runtime["charged_total_cpu_s"] = runtime["total_cpu_s"] + runtime["child_worker_cpu_s"]
    summary = dict(registered_at_utc=protocol["registered_at_utc"],
        source_native_terminal_registered_at_utc=protocol["source_native_terminal_registered_at_utc"],
        source_horizon_registered_at_utc=protocol["source_horizon_registered_at_utc"],
        decision="H_TIMING_MIRROR_FIXED_THREE_COMPLETE" if not failed and not overflow else "H_TIMING_MIRROR_FIXED_THREE_UNAVAILABLE",
        execution_checks_passed=not failed, recorded_root_sample_units=48, completed_root_sample_units=len(valid),
        complete_root_sample_units=sum(flow["complete"] and not flow["overflow"] for flow in valid), failed_units=failed, overflow_units=overflow,
        available_state_rankings=sum(record["comparison"]["available"] for record in records), records=records,
        new_anchor_proposals=0, true_service_suffix_executions=0,
        new_window_native_controller_calls=sum(flow["window_flow"]["calls"] for flow in measured),
        new_window_native_idle_calls=sum(flow["window_flow"]["idle_calls"] for flow in measured),
        new_terminal_native_controller_calls=sum(flow.get("calls", 0) for flow in measured),
        new_terminal_native_idle_calls=sum(flow.get("idle_calls", 0) for flow in measured), work=work, runtime=runtime,
        full24_decision=protocol["full24_decision"], performance_gain_established=False,
        score_units="Eight-path mean of total job-seconds; not mean flow time per job.",
        interpretation="Within-H newborn times are reflected at fixed identities/class counts. Complete original after-anchor native controller is restored, while H queues/costs are outputs. Original post-H anchor-MLE stream and residual clocks are retained. Fixed first-four original/mirror pairs project an eight-path score; the second four are separate sensitivity, never selected by results. All native H/tail work, extra diagnostic MW drain work and child CPU are charged. Selected labels are conditional; full development remains HOLD.")
    dump(path(directory, "_summary.json"), jsonable(summary))
    print(f"complete units=48 available_rankings={summary['available_state_rankings']} failed={len(failed)} overflow={len(overflow)} full24=HOLD", flush=True)


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
