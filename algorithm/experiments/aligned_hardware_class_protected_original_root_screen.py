"""Replay the class veto at all frozen original roots and join their saved effects."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm.bg_incumbent_background_redecision_lane_rollout import INCUMBENT_SCORE_WORK
from algorithm.class_protected_bg_incumbent_background_redecision_lane_rollout import (
    ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout,
)
from algorithm.one_step_background_redecision_lane_rollout import RECOURSE_WORK
from algorithm.experiments.aligned_hardware_bg_incumbent_counterfactual import (
    BASE as COUNTERFACTUAL, PREVIOUS, METRICS, CONTEXT, saved_runs, same,
)
from algorithm.experiments.aligned_hardware_background_competition_continuous import path
from algorithm.experiments.aligned_hardware_background_root_counterfactual import sign_counts
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.bg_incumbent_counterfactual_replay import registered_targets


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_class_protected_original_root_screen_20261001"
SOURCES = (
    "algorithm/class_protected_bg_incumbent_background_redecision_lane_rollout.py",
    "algorithm/experiments/aligned_hardware_class_protected_original_root_screen.py",
)
NEAR_ZERO_S = 1e-9


def build_protocol():
    prior = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    prior_result = json.loads(PREVIOUS.with_suffix(".json").read_text())
    cf_protocol = json.loads(path(COUNTERFACTUAL, "_protocol.json").read_text())
    cf = json.loads(COUNTERFACTUAL.with_suffix(".json").read_text())
    if (prior_result["decision"] != "BG_INCUMBENT_BACKGROUND_REDECISION_DEVELOPMENT_HOLD"
            or not prior_result["execution_checks_passed"]
            or prior_result["registered_at_utc"] != prior["registered_at_utc"]
            or cf["decision"] != "BG_INCUMBENT_COUNTERFACTUAL_COMPLETE"
            or not cf["execution_checks_passed"]
            or cf["registered_at_utc"] != cf_protocol["registered_at_utc"]
            or cf_protocol["previous_registered_at_utc"] != prior["registered_at_utc"]):
        raise ValueError("matching completed original development and counterfactual sources required")
    sources = {**prior["execution_sources"], **cf_protocol["execution_sources"]}
    for name, source in sources.items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen original source changed: " + name)
    frozen = saved_runs()
    targets = registered_targets(prior["tapes"], frozen)
    if len(targets) != 320 or targets != cf_protocol["targets"]:
        raise ValueError("all320 registered original improvements required")
    by_key = {(row["tape_id"], row["frame_index"]): row for row in cf["records"]}
    if len(by_key) != len(targets) or cf["targets"] != len(targets):
        raise ValueError("one saved counterfactual outcome per original target required")
    records = []
    for target in targets:
        record = by_key[(target["tape_id"], target["frame_index"])]
        frame = frozen[target["tape_id"]]["frames"][target["frame_index"]]
        if (any(record[key] != value for key, value in target.items())
                or not record["first_proposal_context_match_passed"]
                or record["decision_counter"] != target["frame_index"]
                or not all(same(record["context"][key], frame[key]) for key in CONTEXT)):
            raise ValueError("saved original target/context/counter differs: " + str(target))
        records.append(record)
    if ({row["tape_id"] for row in cf["factual_replays"]}
            != {tape["tape_id"] for tape in prior["tapes"]}
            or not all(row["factual_reproduction_passed"] for row in cf["factual_replays"])
            or not all(row["null_reproduction_passed"] for row in cf["null_controls"])
            or len(cf["null_controls"]) != len(cf_protocol["active_tape_ids"])):
        raise ValueError("completed factual and null reproduction controls required")
    return {
        "experiment": "class_protected_original_root_screen_v1",
        "previous_protocol": str(path(PREVIOUS, "_protocol.json").relative_to(ROOT)),
        "previous_registered_at_utc": prior["registered_at_utc"],
        "counterfactual_protocol": str(path(COUNTERFACTUAL, "_protocol.json").relative_to(ROOT)),
        "counterfactual_result": str(COUNTERFACTUAL.with_suffix(".json").relative_to(ROOT)),
        "counterfactual_registered_at_utc": cf_protocol["registered_at_utc"],
        **{key: prior[key] for key in (
            "models", "tapes", "arrival_seeds", "planning_samples", "planning_event_cap",
            "planning_seed_rule", "clock", "birth_cohort", "observation_end")},
        "targets": targets, "counterfactual_records": records, "workers": 4,
        "near_zero_seconds_descriptive": NEAR_ZERO_S,
        "method": "Freeze all320 original improved_incumbent roots and their already completed single-action BG-restoration outcomes. Walk every cached original frame in order, reconstruct causal rates from released counts/now and the original outer prefix RNG, skip sampling at ineligible frames and stop at the first overflowing prefix. Increment the outer decision counter after each frame, including idle and overflow frames. Score only the320 targets with the native class-protected once-BG scorer, using the same cached FIFO ages, full availability, lane, post-prefix RNG and counter. Require BG and actual-root S1/S2 predictions to equal the original cached values. Apply the exact per-class S2<=same-model BG test to the original actual root; the original aggregate and old-stage feasibility are already verified by the frozen original decision. Join all saved conditional outcomes verbatim.",
        "budget": "One outer and one inner constructor per24 tapes. Charge every processed original prefix draw/read/event, all320 native scorer and recourse work, and actual short-circuit class comparisons/vector reads. No ordinary-BG first-stage rerun, outer MW scan, action execution or new single-action trajectory. Native inner BG scans remain charged. Four workers.",
        "outcomes": "Saved BG restoration minus original candidate, unchanged. Split all metrics by class gate bypass/blocked and report exact raw signs. The abs(delta)<=1e-9 seconds convention is descriptive only for mean/P90 numerical residues, separate from exact gating. Preserve every class mean/P90/900/1200 tail delta.",
        "scope": "Conditional mechanism diagnostic at original frozen development states, not the changed class-protected policy states. Reuse seeds41..43 and existing dependent single-action effects; no new performance PASS, additive intervention benefit, independent320-sample inference or fresh confirmation.",
        "execution_sources": {**sources, **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def replay_prefixes(planner, frames, work):
    """Yield each native post-prefix/pre-increment state without rerunning actions."""
    for index, frame in enumerate(frames):
        info = frame["rollout"]
        counts, now = tuple(frame["observed_arrival_counts"]), frame["start"]
        planner._outer_observed_counts = counts
        planner.arrival_rates = tuple(count / now if now > 0 else 0. for count in counts)
        work["cached_frame_reads"] += 1
        work["rate_estimation_coordinate_reads"] += len(counts)
        work["rate_estimation_divisions"] += len(counts) if now > 0 else 0
        if (planner.decision_counter != index
                or not same(planner.arrival_rates, info["estimated_arrival_rates"])
                or info["eligible"] != (0 < sum(frame["queue"]) <= info["queue_limit"])):
            raise ValueError("cached causal rates/eligibility/counter differ at frame " + str(index))
        paths = []
        if info["eligible"]:
            for _ in range(planner.sample_count):
                paths.append(sample_planning_arrivals(planner.arrival_rates,
                    planner.maximum_duration, planner.rng, planner.event_cap))
                if paths[-1].overflow:
                    break
        events = sum(len(item.events) for item in paths)
        draws = sum(item.random_draws for item in paths)
        reads = sum(item.rate_reads for item in paths)
        if (len(paths) != info["sample_count"] or events != info["tagged_planning_events"]
                or draws != info["planning_random_draws"] - info["background_planning_random_draws"]
                or reads != info["planning_rate_reads"] - info["background_planning_rate_reads"]
                or (any(item.overflow for item in paths)
                    and (not info["planning_overflow"] or info["incumbent_candidate_count"]))):
            raise ValueError("cached outer prefix population/work differ at frame " + str(index))
        work["prefix_planning_events"] += events
        work["prefix_planning_random_draws"] += draws
        work["prefix_planning_rate_reads"] += reads
        work["prefix_sample_calls"] += len(paths)
        work["prefix_overflow_frames"] += any(item.overflow for item in paths)
        yield index, frame, tuple(paths)
        planner.decision_counter += 1
        work["replayed_decision_counter_writes"] += 1


def score_original_root(planner, frame, paths, prior_record, work):
    """Evaluate only the old actual root's class gate, retaining joined outcomes."""
    info = frame["rollout"]
    original, bg = prior_record["factual_option_index"], prior_record["replacement_option_index"]
    if (not info["improved_incumbent"] or info["planning_overflow"]
            or planner.decision_counter != prior_record["decision_counter"]
            or original != frame["option_indices"][0] or bg != info["incumbent_root_index"]):
        raise ValueError("original target must be a complete same-state incumbent improvement")
    planner._recourse_work = Counter(recourse_root_count_reads=len(planner._outer_observed_counts))
    scored = planner.score_background_roots(frame["queue"], frame["available_before"],
        frame["start"], frame["lane_ids"][0], frame["waiting_ages"], paths)
    predictions = {
        "baseline_predicted_completion_s": float(scored.costs[bg] / len(paths)),
        "chosen_predicted_completion_s": float(scored.costs[original] / len(paths)),
        "baseline_predicted_squared_flow_s2": float(scored.squared_costs[bg] / len(paths)),
        "chosen_predicted_squared_flow_s2": float(scored.squared_costs[original] / len(paths)),
    }
    if scored.background_rollout_overflow or any(value != info[key] for key, value in predictions.items()):
        raise ValueError("native same-population BG/root S1/S2 differ: " + str(prior_record["frame_index"]))
    work.update({key: getattr(scored, key) for key in INCUMBENT_SCORE_WORK})
    work.update({key: planner._recourse_work[key] for key in RECOURSE_WORK})
    work["class_squared_initializations"] += scored.class_squared_initializations
    work["class_squared_accumulations"] += scored.class_squared_accumulations
    comparisons, failing = 0, None
    for coordinate, (candidate, incumbent) in enumerate(zip(
            scored.class_squared_costs[original], scored.class_squared_costs[bg], strict=True)):
        comparisons += 1
        if candidate > incumbent:
            failing = coordinate
            break
    work["class_risk_comparisons"] += comparisons
    work["class_squared_score_reads"] += 2 * comparisons + 2 * len(frame["queue"])
    return {**prior_record,
        "same_model_aggregate_predictions_match_passed": True,
        "class_gate_passed": failing is None, "first_failing_coordinate": failing,
        "class_risk_comparisons": comparisons,
        "candidate_class_squared_flow_s2": [float(value / len(paths))
            for value in scored.class_squared_costs[original]],
        "BG_class_squared_flow_s2": [float(value / len(paths))
            for value in scored.class_squared_costs[bg]],
        "same_model_predictions": predictions}


def run_tape(model, tape, protocol, frozen, records):
    planner = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])
    work = Counter(controller_constructors=1, recourse_controller_constructors=1)
    work.update({"preprocessing_" + key: value for key, value in planner.preprocessing_work.items()})
    by_index = {row["frame_index"]: row for row in records}
    result = []
    for index, frame, paths in replay_prefixes(planner, frozen["frames"], work):
        if index in by_index:
            result.append(score_original_root(planner, frame, paths, by_index[index], work))
    if len(result) != len(records):
        raise ValueError("every original target must be scored once")
    print(tape["tape_id"], "prefix frames", len(frozen["frames"]), "scored roots", len(result), flush=True)
    return {"tape_id": tape["tape_id"], "prefix_reproduction_passed": True,
            "frames_processed": len(frozen["frames"]), "records": result, "work": dict(work)}


def descriptive_signs(values, seconds):
    tolerance = NEAR_ZERO_S if seconds else 0.
    return {"negative": sum(value < -tolerance for value in values if value is not None),
            "near_zero": sum(abs(value) <= tolerance for value in values if value is not None),
            "positive": sum(value > tolerance for value in values if value is not None),
            "unavailable": sum(value is None for value in values)}


def raw_signs(values):
    counts = sign_counts(values)
    return {"negative": counts["helped"], "zero": counts["unchanged"],
            "positive": counts["worsened"], "unavailable": counts["unavailable"]}


def summarize(records):
    result = {"targets": len(records)}
    for metric in METRICS:
        values = [row["outcomes"][metric] for row in records]
        result[metric] = {"raw_signs": raw_signs(values),
            "descriptive_signs": descriptive_signs(values, metric in ("delta_T_mean", "delta_T_common_p90"))}
    result["original_helps_mean_but_hurts_any_class_900_tail"] = sum(
        row["outcomes"]["delta_T_mean"] > NEAR_ZERO_S and any(
            item["delta_900_unfinished"] < 0 for item in row["outcomes"]["per_coordinate_deltas"])
        for row in records)
    result["BG_restore_helps_mean_and_900_tail"] = sum(
        row["outcomes"]["delta_T_mean"] < -NEAR_ZERO_S and row["outcomes"]["delta_900_unfinished"] < 0
        for row in records)
    coordinates = sorted({item["coordinate"] for row in records
                          for item in row["outcomes"]["per_coordinate_deltas"]})
    result["classes"] = []
    for coordinate in coordinates:
        items = [item for row in records for item in row["outcomes"]["per_coordinate_deltas"]
                 if item["coordinate"] == coordinate]
        result["classes"].append({"coordinate": coordinate,
            "workloads": sorted({item["workload"] for item in items}),
            **{metric: {"raw_signs": raw_signs([item[metric] for item in items]),
                "descriptive_signs": descriptive_signs([item[metric] for item in items],
                    metric in ("delta_T_mean", "delta_T_common_mean", "delta_T_common_p90"))}
               for metric in ("delta_T_mean", "delta_T_common_mean", "delta_T_common_p90",
                              "delta_900_unfinished", "delta_1200_unfinished")}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    expected = build_protocol()
    if args.register:
        with path(BASE, "_protocol.json").open("x") as out:
            json.dump({**expected, "registered_at_utc": datetime.now(timezone.utc).isoformat()},
                      out, separators=(",", ":"), allow_nan=False)
        print("registered", len(expected["targets"]), "original roots; new actual trajectories0", flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered original sources, targets, saved outcomes or tapes changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve existing original-root screen outcome")
    models = {model.node: model for model in load_models(protocol["models"])}
    frozen, work, records, checks = saved_runs(), Counter(), [], []
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = [pool.submit(run_tape, models[tape["node"]], tape, protocol,
            frozen[tape["tape_id"]], [row for row in protocol["counterfactual_records"]
                                    if row["tape_id"] == tape["tape_id"]]) for tape in protocol["tapes"]]
        for future in as_completed(futures):
            row = future.result()
            work.update(row["work"])
            records.extend(row["records"])
            checks.append({key: row[key] for key in ("tape_id", "prefix_reproduction_passed", "frames_processed")})
    order = {(row["tape_id"], row["frame_index"]): index for index, row in enumerate(protocol["targets"])}
    records.sort(key=lambda row: order[(row["tape_id"], row["frame_index"])])
    if len(records) != len(protocol["targets"]):
        raise ValueError("all registered original roots required")
    work["total_option_probes"] = sum(work[key] for key in (
        "root_option_probes", "rollout_option_probes", "recourse_root_option_probes",
        "recourse_rollout_option_probes", "preprocessing_option_probes",
        "preprocessing_age_profile_option_probes", "preprocessing_rate_dimension_option_probes",
        "preprocessing_recourse_option_probes", "preprocessing_recourse_age_profile_option_probes",
        "preprocessing_recourse_rate_dimension_option_probes"))
    work["total_planning_random_draws"] = (work["prefix_planning_random_draws"]
        + work["background_planning_random_draws"] + work["recourse_planning_random_draws"])
    strata = {
        "class_gate": [{"class_gate_passed": passed,
            **summarize([row for row in records if row["class_gate_passed"] == passed])}
            for passed in (True, False)],
        "time": [{"time_group": group, "class_gate": [{"class_gate_passed": passed,
            **summarize([row for row in records if row["time_group"] == group
                        and row["class_gate_passed"] == passed])} for passed in (True, False)]}
            for group in ("pre300", "middle300_900", "late900plus")],
        "node_load": [{"node": model["node"], "load": load, "class_gate": [
            {"class_gate_passed": passed, **summarize([row for row in records
                if row["node"] == model["node"] and row["load"] == load
                and row["class_gate_passed"] == passed])} for passed in (True, False)]}
            for model in protocol["models"] for load in sorted({tape["load"] for tape in protocol["tapes"]})],
    }
    result = {"decision": "CLASS_PROTECTED_ORIGINAL_ROOT_SCREEN_COMPLETE",
        "registered_at_utc": protocol["registered_at_utc"], "scope": protocol["scope"],
        "execution_checks_passed": True, "registered_tapes": len(protocol["tapes"]),
        "targets": len(records), "new_actual_trajectory_executions": 0,
        "prefix_checks": sorted(checks, key=lambda row: row["tape_id"]),
        "summary": summarize(records), "strata": strata, "records": records,
        "actual_work": dict(work)}
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "targets": len(records),
        "class_gate": [{"passed": row["class_gate_passed"], "targets": row["targets"],
            "mean": row["delta_T_mean"], "tail900": row["delta_900_unfinished"]}
            for row in strata["class_gate"]]}), flush=True)


if __name__ == "__main__":
    main()
