"""Compare both BG depths at all320 old roots, executing only new third roots."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TwoStepBGIncumbentBackgroundRedecisionLaneRollout
from algorithm.experiments.aligned_hardware_background_competition_continuous import path
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_bg_incumbent_counterfactual import (
    BASE as COUNTERFACTUAL, PREVIOUS, CONTEXT, METRICS, same, saved_runs,
)
from algorithm.experiments.aligned_hardware_class_protected_original_root_screen import (
    build_protocol as original_root_protocol, replay_prefixes, descriptive_signs, raw_signs,
)
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.bg_incumbent_counterfactual_replay import (
    BGIncumbentCounterfactualSelector, collect_work, make_planner, restore_planner_state,
)
from algorithm.experiments.bg_incumbent_selected_root_replay import BGIncumbentSelectedRootSelector
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.two_step_bg_incumbent_root_ranking import compare_root_scores


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_two_step_bg_incumbent_root_ranking_20261001"
TWO_STEP = ROOT / "md/experiment_artifacts/aligned_hardware_two_step_bg_incumbent_development_20261001"
SOURCES = (
    "algorithm/experiments/two_step_bg_incumbent_root_ranking.py",
    "algorithm/experiments/bg_incumbent_selected_root_replay.py",
    "algorithm/experiments/aligned_hardware_two_step_bg_incumbent_root_ranking.py",
)
CHOICES = ("original", "BG", "third")


def build_protocol():
    original = original_root_protocol()
    prior_cf = json.loads(path(COUNTERFACTUAL, "_protocol.json").read_text())
    two = json.loads(path(TWO_STEP, "_protocol.json").read_text())
    result = json.loads(TWO_STEP.with_suffix(".json").read_text())
    if (result["decision"] != "TWO_STEP_BG_INCUMBENT_DEVELOPMENT_HOLD"
            or not result["execution_checks_passed"]
            or result["registered_at_utc"] != two["registered_at_utc"]
            or any(two[key] != prior_cf[key] for key in (
                "models", "tapes", "arrival_seeds", "planning_samples", "planning_event_cap"))):
        raise ValueError("matching completed two-step HOLD on the original development tapes required")
    sources = {**original["execution_sources"], **two["execution_sources"]}
    for name, source in sources.items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    return {
        "experiment": "two_step_bg_incumbent_original_root_ranking_v1",
        **{key: original[key] for key in (
            "previous_protocol", "previous_registered_at_utc", "counterfactual_protocol",
            "counterfactual_result", "counterfactual_registered_at_utc", "targets", "counterfactual_records")},
        "previous_result": str(PREVIOUS.with_suffix(".json").relative_to(ROOT)),
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        "two_step_protocol": str(path(TWO_STEP, "_protocol.json").relative_to(ROOT)),
        "two_step_result": str(TWO_STEP.with_suffix(".json").relative_to(ROOT)),
        "two_step_registered_at_utc": two["registered_at_utc"],
        **{key: prior_cf[key] for key in (
            "models", "tapes", "arrival_seeds", "planning_samples", "planning_event_cap",
            "planning_seed_rule", "clock", "birth_cohort", "arrival_horizon", "observation_end",
            "environment_arrival_end", "cutoff_semantics")},
        "workers": 4, "near_zero_seconds_descriptive": 1e-9,
        "method": "Freeze all320 original strict BG-incumbent improvements, including every time/load/catalogue/seed. Walk every old cached frame to reconstruct causal released rates, the native prefix draws and counter. At each target recompute ordinary BG's original stage, once-BG and two-BG on the same post-prefix RNG/counts/rates/full FIFO/lane state, requiring exact cached original and once-stage predictions and the old chosen root. Select two-BG using the original S2<=MW mask, two-BG S2<=BG, BG-first ties and strict S1 gain; two-stage overflow retains BG with partial fees. Original root reuses factual identity; BG reuses the already completed BG-restoration effects verbatim; only a third root executes a new first-action override, after fully consuming the original once-BG proposal, then freely replans with frozen once-BG. Each tape with third roots has one earliest-third null suffix and complete jobs/actions/clock/rollout/final RNG/counter identity. No full factual rerun.",
        "budget": "Four CPU tape workers, original outer/inner8 samples and cap128. Two reusable outer/inner planner pairs per24 cached tapes. Charge every reconstructed prefix and state copy, native MW certificate, all three score stages and once/two recourse fees and comparisons. Each newly executed null/third suffix uses one fresh original once planner/inner and complete native work, constructor/preprocessing/context/checkpoint/restore fees. A forced third action adds one separately charged MW certificate scan. No double-counting engine aliases. Keep partial two-overflow work. Save only paired outcomes and controls, not suffix trajectories. Number of new third suffixes is determined only by frozen same-state score selection, never outcomes.",
        "outcomes": "selected action minus old factual under the same frozen once-BG continuation. Retain old BG-restoration outcomes separately. Report exact and descriptive effect signs by original/BG/third selection, per class and time/node/load/seed. For third roots, restricted-mean, unfinished counts and area differences relative to BG may be obtained by subtracting same-factual deltas; do not subtract P90/common-mean effects from differing completed-ID sets.",
        "scope": "Conditional action-ranking diagnostic on original320 states and development seeds41..43. Keep both development HOLDs. No new policy-performance PASS, new two-BG suffix, additive intervention gain or independent320-sample inference. States where the old policy retained BG are outside this diagnostic.",
        "execution_sources": {**sources, **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def selected_outcomes(record, factual, protocol, workloads):
    """Reuse the actual old action or the already executed BG-restoration arm."""
    if record["choice_kind"] == "original":
        return paired_outcomes(factual, factual, protocol, workloads)
    if record["choice_kind"] == "BG":
        return record["outcomes"]
    raise ValueError("third-root outcomes require a new single-action execution")


def run_tape(model, tape, protocol, frozen, prior_records):
    once = make_planner(model, tape, protocol)
    twice = TwoStepBGIncumbentBackgroundRedecisionLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])
    work = Counter(controller_constructors=2, recourse_controller_constructors=2)
    for planner in (once, twice):
        work.update({"preprocessing_" + key: value for key, value in planner.preprocessing_work.items()})
    by_index = {row["frame_index"]: row for row in prior_records}
    pre_state = {"rng_state": once.rng.getstate(), "decision_counter": once.decision_counter}
    work["reconstruction_rng_words_copied"] += len(pre_state["rng_state"][1])
    work["reconstruction_counter_reads"] += 1
    records, third_states = [], {}
    for index, frame, paths in replay_prefixes(once, frozen["frames"], work):
        if index in by_index:
            record = compare_root_scores(once, twice, frame, paths, by_index[index], work)
            record["suffix_policy"] = "frozen_once_BG_incumbent"
            record["effect_source_tape_id"], record["effect_source_frame_index"] = tape["tape_id"], index
            if record["choice_kind"] == "third":
                third_states[index] = pre_state
            else:
                record["selected_outcomes"] = selected_outcomes(record, frozen, protocol, model.workloads)
                record["effect_source"] = ("cached_factual_identity" if record["choice_kind"] == "original"
                                           else "cached_BG_restoration")
            records.append(record)
        pre_state = {"rng_state": once.rng.getstate(), "decision_counter": once.decision_counter + 1}
        work["reconstruction_rng_words_copied"] += len(pre_state["rng_state"][1])
        work["reconstruction_counter_reads"] += 1
    if len(records) != len(prior_records):
        raise ValueError("every registered original target must be scored once")
    print(tape["tape_id"], "scored", len(records), "choices",
          dict(Counter(record["choice_kind"] for record in records)), flush=True)
    controls, status, new_count = [], Counter(), 0
    for record in (item for item in records if item["choice_kind"] == "third"):
        index, root = record["frame_index"], record["two_step_option_index"]
        frame = frozen["frames"][index]
        checkpoint = prepare_checkpoint(frozen, index)
        work.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
        if not controls:
            null = BGIncumbentCounterfactualSelector(make_planner(model, tape, protocol))
            work.update(restore_planner_state(null.planner, third_states[index]))
            resumed = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
                protocol["observation_end"], null, checkpoint, policy="null")
            combined = {**resumed, "frames": frozen["frames"][:index] + resumed["frames"]}
            if (not replay_matches(combined, frozen)
                    or not all(same(a["rollout"], b["rollout"])
                               for a, b in zip(combined["frames"], frozen["frames"], strict=True))
                    or null.planner.rng.getstate() != once.rng.getstate()
                    or null.planner.decision_counter != once.decision_counter):
                raise ValueError("reconstructed null jobs/actions/clock/rollout/RNG/counter differ: " + tape["tape_id"])
            work.update(collect_work(null, resumed))
            status.update(null.status)
            controls.append({"tape_id": tape["tape_id"], "frame_index": index,
                             "null_reproduction_passed": True})
            new_count += 1
        scores = next(row for row in record["once_scores"] if row["option_index"] == root)
        predictions = {"chosen_predicted_completion_s": scores["S1"],
            "chosen_predicted_squared_flow_s2": scores["S2"],
            "incumbent_actual_predicted_squared_flow_s2": scores["original_stage_S2"]}
        selector = BGIncumbentSelectedRootSelector(make_planner(model, tape, protocol), root, predictions)
        work.update(restore_planner_state(selector.planner, third_states[index]))
        replacement = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
            protocol["observation_end"], selector, checkpoint, policy="third_first_once_BG_suffix")
        if (selector.first_proposal["original"] != record["factual_option_index"]
                or selector.first_proposal["replacement"] != record["BG_root_index"]
                or not same(selector.first_proposal["rollout"], frame["rollout"])
                or not all(same(replacement["frames"][0][key], frame[key]) for key in CONTEXT)
                or replacement["frames"][0]["option_indices"][0] != root
                or selector.work["override_calls"] != 1):
            raise ValueError("third-root original proposal/context/applied action differs")
        work.update(collect_work(selector, replacement))
        status.update(selector.status)
        record.update(effect_source="new_third_root_override", first_proposal_context_match_passed=True,
            intervention=selector.intervention,
            selected_outcomes=paired_outcomes(frozen, replacement, protocol, model.workloads))
        record["third_relative_to_BG"] = {metric: record["selected_outcomes"][metric] - record["outcomes"][metric]
            for metric in ("delta_T_mean", "delta_900_unfinished", "delta_1200_unfinished", "delta_middle_queue_area_s")}
        new_count += 1
        print(tape["tape_id"], "third root frame", index, "mean delta",
              record["selected_outcomes"]["delta_T_mean"], flush=True)
    return {"tape_id": tape["tape_id"], "prefix_reproduction_passed": True,
        "frames_processed": len(frozen["frames"]), "records": records, "controls": controls,
        "new_actual_trajectory_executions": new_count, "work": dict(work), "status": dict(status)}


def effect_summary(records):
    summary = {"targets": len(records)}
    for metric in METRICS:
        values = [row["selected_outcomes"][metric] for row in records]
        summary[metric] = {"raw_signs": raw_signs(values),
            "descriptive_signs": descriptive_signs(values, metric in ("delta_T_mean", "delta_T_common_p90"))}
    summary["old_action_mean_effect"] = {
        "beneficial": sum(row["outcomes"]["delta_T_mean"] > 1e-9 for row in records),
        "harmful": sum(row["outcomes"]["delta_T_mean"] < -1e-9 for row in records),
        "near_zero": sum(abs(row["outcomes"]["delta_T_mean"]) <= 1e-9 for row in records)}
    summary["selected_helps_mean_and_900_tail"] = sum(
        row["selected_outcomes"]["delta_T_mean"] < -1e-9
        and row["selected_outcomes"]["delta_900_unfinished"] < 0 for row in records)
    summary["selected_helps_mean_but_hurts_any_class_900_tail"] = sum(
        row["selected_outcomes"]["delta_T_mean"] < -1e-9 and any(
            item["delta_900_unfinished"] > 0 for item in row["selected_outcomes"]["per_coordinate_deltas"])
        for row in records)
    summary["classes"] = [{"coordinate": coordinate,
        **{metric: {"raw_signs": raw_signs([item[metric] for row in records
                    for item in row["selected_outcomes"]["per_coordinate_deltas"] if item["coordinate"] == coordinate])}
           for metric in ("delta_T_mean", "delta_T_common_mean", "delta_T_common_p90",
                          "delta_900_unfinished", "delta_1200_unfinished")}}
        for coordinate in sorted({item["coordinate"] for row in records
                                 for item in row["selected_outcomes"]["per_coordinate_deltas"]})]
    return summary


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
        print("registered all", len(expected["targets"]), "original states", flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources, targets, outcomes or tapes changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve existing ranking diagnostic result")
    frozen = saved_runs()
    models = {model.node: model for model in load_models(protocol["models"])}
    work, status, records, controls, checks = Counter(), Counter(), [], [], []
    executions = 0
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = [pool.submit(run_tape, models[tape["node"]], tape, protocol, frozen[tape["tape_id"]],
            [row for row in protocol["counterfactual_records"] if row["tape_id"] == tape["tape_id"]])
            for tape in protocol["tapes"]]
        for future in as_completed(futures):
            row = future.result()
            records.extend(row["records"])
            controls.extend(row["controls"])
            work.update(row["work"])
            status.update(row["status"])
            executions += row["new_actual_trajectory_executions"]
            checks.append({key: row[key] for key in ("tape_id", "prefix_reproduction_passed", "frames_processed")})
            print("tape complete", row["tape_id"], "total roots", len(records), flush=True)
    order = {(row["tape_id"], row["frame_index"]): i for i, row in enumerate(protocol["targets"])}
    records.sort(key=lambda row: order[(row["tape_id"], row["frame_index"])])
    if (len(records) != len(protocol["targets"])
            or len(controls) != len({row["tape_id"] for row in records if row["choice_kind"] == "third"})):
        raise ValueError("all320 registered roots and every new-third null control required")
    work["total_option_probes"] = sum(value for key, value in work.items() if key.endswith("option_probes"))
    choices = [{"choice_kind": choice, **effect_summary([row for row in records if row["choice_kind"] == choice])}
               for choice in CHOICES]
    strata = {
        "time": [{"time_group": group, **effect_summary([row for row in records if row["time_group"] == group])}
                 for group in ("pre300", "middle300_900", "late900plus")],
        "node_load": [{"node": model["node"], "load": load,
            **effect_summary([row for row in records if row["node"] == model["node"] and row["load"] == load])}
            for model in protocol["models"] for load in sorted({tape["load"] for tape in protocol["tapes"]})],
        "seed": [{"seed": seed, **effect_summary([row for row in records if row["seed"] == seed])}
                 for seed in protocol["arrival_seeds"]],
    }
    result = {"decision": "TWO_STEP_BG_INCUMBENT_ORIGINAL_ROOT_RANKING_COMPLETE",
        "registered_at_utc": protocol["registered_at_utc"], "scope": protocol["scope"],
        "execution_checks_passed": True, "targets": len(records), "registered_tapes": len(protocol["tapes"]),
        "prefix_checks": sorted(checks, key=lambda row: row["tape_id"]), "null_controls": controls,
        "new_actual_trajectory_executions": executions,
        "reused_original_action_effects": sum(row["choice_kind"] == "original" for row in records),
        "reused_BG_restoration_effects": sum(row["choice_kind"] == "BG" for row in records),
        "new_third_root_executions": sum(row["choice_kind"] == "third" for row in records),
        "two_step_background_overflows": sum(row["two_step_background_overflow"] for row in records),
        "summary": effect_summary(records), "choices": choices, "strata": strata,
        "records": records, "actual_work": dict(work), "status_counts": dict(status)}
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "targets", "new_actual_trajectory_executions",
        "reused_original_action_effects", "reused_BG_restoration_effects", "new_third_root_executions", "choices")}), flush=True)


if __name__ == "__main__":
    main()
