"""Locate later native action disagreements in both fixed-root cached pairs."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import socket
from time import perf_counter, process_time

from algorithm.experiments import aligned_hardware_finite_bg_native_controller_cross as native
from algorithm.experiments import aligned_hardware_finite_bg_first_action_cross as previous
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.cached_finite_bg_regression import action, CONTEXT

ARTIFACTS = native.ARTIFACTS
STEM = "aligned_hardware_cached_finite_BG_common_root_continuation_20261003"
read, dump, require, same, births = native.read, native.dump, native.require, native.same, native.births
SCORES = ("baseline_predicted_completion_s", "chosen_predicted_completion_s",
    "baseline_predicted_squared_flow_s2", "chosen_predicted_squared_flow_s2",
    "incumbent_predicted_completion_s", "incumbent_mw_predicted_completion_s",
    "incumbent_predicted_squared_flow_s2", "incumbent_mw_predicted_squared_flow_s2",
    "incumbent_actual_predicted_squared_flow_s2", "incumbent_root_index", "incumbent_maxweight_root_index",
    "unconstrained_root_index", "eligible", "queue_limit", "window_s", "candidate_count",
    "sample_count", "configured_sample_count", "tagged_planning_events", "planning_overflow",
    "background_rollout_overflow", "incumbent_background_rollout_overflow", "changed", "improved_incumbent",
    "risk_feasible_candidates", "risk_filtered_candidates", "recourse_calls", "recourse_first_calls",
    "recourse_second_calls", "recourse_eligible_calls", "recourse_changed_calls", "recourse_planning_overflows")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def frame_description(frame, global_index, options, workloads):
    info = frame["rollout"]
    return dict(decision_counter_before_call=global_index,
        physical={key: frame[key] for key in (*CONTEXT, "option_indices", "service", "duration")},
        action=action(frame, frame["option_indices"][0], options, workloads),
        own_BG_incumbent=action(frame, info["incumbent_root_index"], options, workloads)
            if info.get("incumbent_root_index") is not None else None,
        rollout={key: info[key] for key in SCORES if key in info},
        saved_frame_reuse_signals={key: value for key, value in info.items()
            if any(token in key for token in ("reuse", "memo", "cache"))},
        inner_current_signals={key: info["inner_current_work"][key] for key in
            ("recourse_calls", "recourse_first_calls", "recourse_second_calls", "incumbent_candidate_count")
            if key in info.get("inner_current_work", {})})


def compare_prefix(left, right, anchor, options, workloads, work):
    require(left and right, "both cached suffixes require their common first action")
    divergence = None
    for local_index, (finite, bg) in enumerate(zip(left, right)):
        work["paired_frame_contexts_scanned"] += 1
        require(all(same(finite[key], bg[key]) for key in CONTEXT), "physical contexts differ before a later action disagreement")
        if finite["option_indices"] != bg["option_indices"]:
            require(local_index > 0, "registered common first physical root differs")
            divergence = dict(suffix_frame_index=local_index, original_frame_index=anchor + local_index,
                common_action_frames=local_index,
                finite_native=frame_description(finite, anchor + local_index, options, workloads),
                BG_native=frame_description(bg, anchor + local_index, options, workloads))
            break
        require(finite["service"] == bg["service"] and finite["duration"] == bg["duration"], "same action has different physical service")
    if divergence is None:
        require(len(left) == len(right), "no action disagreement but cached suffix lengths differ")
    return dict(first_later_divergence=divergence, common_action_frames=len(left) if divergence is None else divergence["common_action_frames"],
        no_action_divergence=divergence is None, finite_suffix_frames=len(left), BG_suffix_frames=len(right),
        common_first_finite=frame_description(left[0], anchor, options, workloads),
        common_first_BG=frame_description(right[0], anchor, options, workloads))


def analyze(directory):
    started, cpu = perf_counter(), process_time()
    design, protocol = read(path(directory, "_design.json")), native.load_protocol(directory)
    summary = read(Path(directory) / design["source_summary"])
    prior = read(previous.path(directory, "_protocol.json"))
    require(protocol["registered_at_utc"] == summary["registered_at_utc"] == design["source_execution_epoch"]
        and summary["execution_checks_passed"] and summary["completed_new_suffixes"] == 3
        and prior["registered_at_utc"] == protocol["previous_registered_at_utc"]
        and [target["seed"] for target in protocol["targets"]] == design["seeds"]
        and protocol["full24_decision"] == design["full24_decision"], "completed frozen three-seed native crossing required")
    work, cache, records = Counter(), {}, []

    def load_pickle(file):
        work["fixture_pickle_reads"] += 1
        with Path(file).open("rb") as source:
            return pickle.load(source)

    def load_gzip(file, epoch):
        name = str(file)
        if name not in cache:
            work["gzip_reads"] += 1
            work["compressed_gzip_bytes_read"] += Path(file).stat().st_size
            with gzip.open(file, "rt") as source:
                cache[name] = json.load(source)
        require(cache[name]["registered_at_utc"] == epoch, "cached gzip registration differs")
        return cache[name]

    for index, target in enumerate(protocol["targets"]):
        tape_id, anchor = target["tape_id"], target["frame_index"]
        fixture = load_pickle(target["fixture_file"])
        old = load_pickle(prior["targets"][index]["fixture_file"])
        bf_row = read(previous.path(directory, f"_shard_{index:02d}.json"))
        fb_row = read(native.path(directory, f"_shard_{index:02d}.json"))
        require(bf_row["registered_at_utc"] == prior["registered_at_utc"]
            and fb_row["registered_at_utc"] == protocol["registered_at_utc"]
            and bf_row["tape_id"] == fb_row["tape_id"] == prior["targets"][index]["tape_id"] == tape_id
            and bf_row["execution_checks_passed"] and fb_row["execution_checks_passed"], "completed cached suffix identities differ")
        bf_saved = load_gzip(bf_row["trajectory_file"], prior["registered_at_utc"])
        fb_saved = load_gzip(fb_row["trajectory_file"], protocol["registered_at_utc"])
        require(bf_saved["tape_id"] == fb_saved["tape_id"] == tape_id, "saved suffix tape identity differs")
        original = load_gzip(old["finite_trajectory_file"], prior["scientific_source_registered_at_utc"])
        ff = next(run["policies"]["arrival_rollout"] for run in original["runs"] if run["tape_id"] == tape_id)
        arms = {"FF": ff, "BF": bf_saved["result"], "FB": fb_saved["result"], "BB": fixture["cached_arms"]["BB"]}
        require(all(births(arm) == births(ff) for arm in arms.values()), "four cached canonical birth populations differ")
        suffixes = {"FF": ff["frames"][anchor:], "BF": arms["BF"]["frames"],
            "FB": arms["FB"]["frames"], "BB": fixture["BG_suffix_frames"]}
        options, workloads = [asdict(option) for option in fixture["model"].options], fixture["model"].workloads
        pairs = {}
        for name, (finite_arm, bg_arm) in design["common_root_pairs"].items():
            prefix = compare_prefix(suffixes[finite_arm], suffixes[bg_arm], anchor, options, workloads, work)
            pairs[name] = dict(finite_arm=finite_arm, BG_arm=bg_arm, **prefix,
                complete_job_outcomes_identical=arms[finite_arm]["jobs"] == arms[bg_arm]["jobs"],
                outcomes_finite_minus_BG=paired_outcomes(arms[bg_arm], arms[finite_arm], protocol, workloads))
            work["paired_job_population"] += len(arms[finite_arm]["jobs"]) + len(arms[bg_arm]["jobs"])
        records.append(dict(tape_id=tape_id, seed=target["seed"], frame_index=anchor, pairs=pairs,
            anchor_scope={"BF": "forced B physical root after complete finite native proposal",
                "FB": "forced F physical root after complete BG native proposal", "FF": "native finite", "BB": "native BG"}))
        print(tape_id, "cached fixed_B and fixed_F inspected", flush=True)
    return dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(), design_registered_at_utc=design["registered_at_utc"],
        source_execution_epoch=protocol["registered_at_utc"], completed_cached_seed_records=len(records),
        records=records, reading_work=dict(work), caller_cpu_s=process_time() - cpu, wall_s=perf_counter() - started,
        execution_host=socket.gethostname(), new_planner_calls=0, new_simulation_executions=0,
        full24_decision=protocol["full24_decision"], performance_gain_established=False,
        interpretation=design["interpretation"], score_scope="Saved native proposal costs and overflows describe each controller's own samples and continuation. Forced first-frame chosen forecasts were cleared. Missing per-frame reuse fields remain empty; suffix-level reuse is not assigned to one later frame.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    result = analyze(args.artifact_dir)
    destination = path(args.artifact_dir, "_summary.json")
    dump(destination, result)
    print(json.dumps({"output": str(destination), "cached_seeds": len(result["records"]), "new_simulations": 0}), flush=True)


if __name__ == "__main__":
    main()
