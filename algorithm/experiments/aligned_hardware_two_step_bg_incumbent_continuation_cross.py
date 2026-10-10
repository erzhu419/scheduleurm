"""Frozen root-by-continuation diagnostic at cohort-visible development calls."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.experiments.aligned_hardware_background_competition_continuous import (
    check_rollout as check_BG, path,
)
from algorithm.experiments.aligned_hardware_bg_incumbent_counterfactual import CONTEXT, same
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.aligned_hardware_two_step_bg_incumbent_original_risk_development import (
    BASE as PREVIOUS, build_protocol as previous_protocol, check_rollout as check_current,
)
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.single_action_counterfactual_replay import (
    prepare_checkpoint, resume_single_action_trajectory,
)
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import (
    TwoStepContinuationCrossSelector, collect_work, make_current_planner,
    reconstruct_prefix_states, registered_anchors, restore_planner_state,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_two_step_bg_incumbent_continuation_cross_20261001"
SOURCES = (
    "algorithm/experiments/two_step_bg_incumbent_continuation_cross_replay.py",
    "algorithm/experiments/aligned_hardware_two_step_bg_incumbent_continuation_cross.py",
    "algorithm/experiments/bg_incumbent_counterfactual_replay.py",
    "algorithm/experiments/aligned_hardware_bg_incumbent_counterfactual.py",
    "algorithm/experiments/single_action_counterfactual_replay.py",
)
ARMS = ("original_current", "BG_current", "original_BG", "BG_BG")
METRICS = ("delta_T_mean", "delta_T_common_p90", "delta_900_common_p90",
           "delta_900_unfinished", "delta_1200_unfinished", "delta_middle_queue_area_s")


def saved_runs():
    with gzip.open(path(PREVIOUS, "_trajectories.json.gz"), "rt") as source:
        return {row["tape_id"]: row["policies"]["arrival_rollout"]
                for row in json.load(source)["runs"]}


def build_protocol():
    parent = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    result = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if ({key: value for key, value in parent.items() if key != "registered_at_utc"}
            != previous_protocol()
            or result["registered_at_utc"] != parent["registered_at_utc"]
            or result["decision"] != "TWO_STEP_BG_INCUMBENT_ORIGINAL_RISK_DEVELOPMENT_HOLD"
            or not result["execution_checks_passed"]):
        raise ValueError("unchanged execution-valid original-risk development HOLD required")
    roster = registered_anchors(parent["tapes"], saved_runs())
    return {
        "experiment": "two_step_bg_incumbent_continuation_cross_v1",
        "previous_protocol": str(path(PREVIOUS, "_protocol.json").relative_to(ROOT)),
        "previous_result": str(PREVIOUS.with_suffix(".json").relative_to(ROOT)),
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        "previous_registered_at_utc": parent["registered_at_utc"],
        "previous_decision": result["decision"],
        **{key: parent[key] for key in (
            "models", "tapes", "loads", "arrival_seeds", "planning_samples", "planning_event_cap",
            "planning_seed_rule", "clock", "birth_cohort", "arrival_horizon", "observation_end",
            "environment_arrival_end", "cutoff_semantics", "performance")},
        **roster, "workers": 8, "near_zero_seconds_descriptive": 1e-9,
        "expected_new_trajectory_executions": 4 * len(roster["anchors"]),
        "anchor_rule": "On each of all24 cached development tapes, retain the first complete nonoverflowing strict BG-incumbent improvement with different applied/BG roots,300<=start<900, and at leastone currently waiting job born in closed cohort[300,900]. Reconstruct actual FIFO births from cached admitted services and observed counts. Keep missing tapes without later or outcome-based substitution.",
        "method": "Reconstruct every cached frame's causal released counts, FIFO ages, rates, prefix draws and outer RNG/counter without native factual reruns. At every anchor all arms consume the same complete current original-risk two-BG proposal. Apply its selected root or verified ordinary BG incumbent exactlyonce. Then freely replan with either current original-risk policy or ordinary BG. Ordinary BG receives the current proposal's post-call outer RNG/counter and the same planningseed; every later call recomputes real released-state context. Reuse original-root/current-suffix factual outcomes. Execute one exact null recovery plus BG-root/current-suffix, original-root/BG-suffix and BG-root/BG-suffix. Verify every first proposal/context and subsequent native-policy calls; null also matches all jobs/actions/clock/rollout and final outer RNG/counter.",
        "budget": "Eight CPU workers; unchanged outer/inner samples8, cap128. One current outer/inner reconstruction pair per24 cached tapes. Four new suffix executions per22 anchors, each with one current outer/inner for its full first proposal; two BG suffix arms additionally construct one ordinary BG controller, with no recourse constructor for it. Charge all native proposal/scoring/context/preprocessing and two-step work, each actual forced BG certificate scan, prefix/checkpoint/state capture/restore and BG statehandoff. Count selector costs once and exclude engine aliases. Save outcomes and controls only; no suffix trajectory cache or new server fetch.",
        "outcomes": "Root effect=original-first minus BG-first within each suffix; negative means the proposed root helps. Interaction=current-suffix root effect minus ordinary-BG-suffix root effect. At each900/1200 cutoff use one four-arm common completed birth-cohort ID intersection for all common mean/P90 and its class-specific subsets; empty class intersections remain null. Restricted means and unfinished counts retain the full same birth cohort. All-job waiting area stays[300,900]. Report raw values plus exact/descriptive signs and mean-ranking flips, including per class, node/load and shared seed.",
        "scope": "Conditional service-performance mechanism diagnostic on22 outcome-unfiltered states and reused development seeds41..43; two missing tapes remain explicit. No additive intervention benefit,22 independent samples, forecast-error decomposition, fresh confirmation or new policy-performance PASS. Keep every parent HOLD and unchanged MW/BG full-policy performance gates. Real future births remain solely in the execution harness.",
        "execution_sources": {**parent["execution_sources"],
                              **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def cross_outcomes(arms, protocol, workloads):
    lower, upper = protocol["birth_cohort"]
    cohort = {label: [job for job in arms[label]["jobs"] if lower <= job["arrival"] <= upper]
              for label in ARMS}
    snapshots = {}
    for key in ("arrival_horizon", "observation_end"):
        cutoff = protocol[key]
        done = {label: {(job["coordinate"], job["index"]): job for job in jobs
                        if job["completion"] is not None and job["completion"] <= cutoff}
                for label, jobs in cohort.items()}
        common = sorted(set.intersection(*(set(jobs) for jobs in done.values())))
        snapshots[key] = {}
        for label in ARMS:
            all_jobs = trajectory_snapshot(cohort[label], cutoff, len(workloads))
            shared = trajectory_snapshot([done[label][identity] for identity in common], cutoff, len(workloads))
            all_jobs.update(common_jobs=len(common), common_mean_flow_s=shared["completed_mean_flow_s"],
                            common_p90_flow_s=shared["completed_p90_flow_s"])
            for coordinate in range(len(workloads)):
                shared_class = shared["per_coordinate"][coordinate]
                all_jobs["per_coordinate"][coordinate].update(
                    common_class_jobs=shared_class["arrivals"],
                    common_class_mean_flow_s=shared_class["completed_mean_flow_s"],
                    common_class_p90_flow_s=shared_class["completed_p90_flow_s"])
            snapshots[key][label] = all_jobs
    areas = {label: queue_area(arms[label]) for label in ARMS}

    def difference(left, right):
        return left - right if left is not None and right is not None else None

    def root_effect(suffix):
        original, bg = "original_" + suffix, "BG_" + suffix

        def delta(key, metric, coordinate=None):
            values = [snapshots[key][label] for label in (original, bg)]
            if coordinate is not None:
                values = [value["per_coordinate"][coordinate] for value in values]
            return difference(values[0][metric], values[1][metric])

        return {
            "delta_T_mean": delta("observation_end", "restricted_mean_flow_s"),
            "delta_T_common_p90": delta("observation_end", "common_p90_flow_s"),
            "delta_900_common_p90": delta("arrival_horizon", "common_p90_flow_s"),
            "delta_900_unfinished": delta("arrival_horizon", "unfinished_jobs"),
            "delta_1200_unfinished": delta("observation_end", "unfinished_jobs"),
            "delta_middle_queue_area_s": areas[original] - areas[bg],
            "per_coordinate_deltas": [{"coordinate": coordinate, "workload": name,
                "delta_T_mean": delta("observation_end", "restricted_mean_flow_s", coordinate),
                "delta_T_common_mean": delta("observation_end", "common_class_mean_flow_s", coordinate),
                "delta_T_common_p90": delta("observation_end", "common_class_p90_flow_s", coordinate),
                "delta_900_common_p90": delta("arrival_horizon", "common_class_p90_flow_s", coordinate),
                "delta_900_unfinished": delta("arrival_horizon", "unfinished_jobs", coordinate),
                "delta_1200_unfinished": delta("observation_end", "unfinished_jobs", coordinate)}
                for coordinate, name in enumerate(workloads)],
        }

    effects = {suffix: root_effect(suffix) for suffix in ("current", "BG")}
    interaction = {key: difference(effects["current"][key], effects["BG"][key]) for key in METRICS}
    interaction["per_coordinate_deltas"] = [{
        "coordinate": current["coordinate"], "workload": current["workload"],
        **{key: difference(current[key], bg[key]) for key in current if key.startswith("delta_")}}
        for current, bg in zip(effects["current"]["per_coordinate_deltas"],
                               effects["BG"]["per_coordinate_deltas"], strict=True)]
    return {"snapshots": snapshots, "middle_queue_area_s": areas,
            "root_effects": effects, "interaction": interaction}


def run_tape(model, tape, protocol, factual, anchor):
    planner = make_current_planner(model, tape, protocol)
    work = Counter(controller_constructors=1, recourse_controller_constructors=1)
    work.update({"preprocessing_" + key: value for key, value in planner.preprocessing_work.items()})
    states = reconstruct_prefix_states(planner, factual,
        [anchor["frame_index"]] if anchor is not None else [], work)
    check = {"tape_id": tape["tape_id"], "prefix_reproduction_passed": True,
             "frames_processed": len(factual["frames"]), "anchor_present": anchor is not None}
    if anchor is None:
        return {"check": check, "record": None, "work": dict(work), "status": {}, "executions": 0}
    final_state = {"rng_state": planner.rng.getstate(), "decision_counter": planner.decision_counter}
    work["reconstruction_rng_words_copied"] += len(final_state["rng_state"][1])
    work["reconstruction_counter_reads"] += 1
    index, frame = anchor["frame_index"], factual["frames"][anchor["frame_index"]]
    checkpoint = prepare_checkpoint(factual, index)
    work.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
    arms, arm_checks, status = {"original_current": factual}, {}, Counter()
    for arm in ARMS:
        root, suffix = arm.split("_")
        selector = TwoStepContinuationCrossSelector(make_current_planner(model, tape, protocol),
                                                    first_root=root, suffix=suffix)
        work.update(restore_planner_state(selector.planner, states[index]))

        def select(*context):
            decision = selector(*context)
            if selector.calls % 100 == 0:
                print(tape["tape_id"], arm, "suffix dispatch", selector.calls,
                      "service time", context[3], flush=True)
            return decision

        result = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
            protocol["observation_end"], select, checkpoint, policy=arm)
        proposed = selector.first_proposal
        actual = anchor["factual_option_index"] if root == "original" else anchor["replacement_option_index"]
        if (proposed["original"] != anchor["factual_option_index"]
                or proposed["replacement"] != anchor["replacement_option_index"]
                or not same(proposed["rollout"], frame["rollout"])
                or not all(same(result["frames"][0][key], frame[key]) for key in CONTEXT)
                or result["frames"][0]["option_indices"][0] != actual
                or selector.work["override_calls"] != int(root == "BG")):
            raise ValueError("first proposal/context/action differs: " + tape["tape_id"] + ":" + arm)
        native_check = check_current if suffix == "current" else check_BG
        if not native_check({**result, "frames": result["frames"][1:]}, model, protocol, selector.active_planner):
            raise ValueError("native continuation checks failed: " + tape["tape_id"] + ":" + arm)
        if arm == "original_current":
            combined = {**result, "frames": factual["frames"][:index] + result["frames"]}
            if (not replay_matches(combined, factual)
                    or not all(same(a["rollout"], b["rollout"])
                               for a, b in zip(combined["frames"], factual["frames"], strict=True))
                    or combined["clock"] != factual["clock"]
                    or combined["last_frame_end"] != factual["last_frame_end"]
                    or selector.active_planner.rng.getstate() != final_state["rng_state"]
                    or selector.active_planner.decision_counter != final_state["decision_counter"]):
                raise ValueError("null jobs/actions/clock/rollout/RNG/counter differ: " + tape["tape_id"])
        else:
            arms[arm] = result
        work.update(collect_work(selector, result))
        status.update(selector.status)
        arm_checks[arm] = {"first_proposal_context_match_passed": True,
            "native_continuation_checks_passed": True, "decision_calls": selector.calls,
            "intervention": selector.intervention,
            "null_reproduction_passed": True if arm == "original_current" else None}
        print(tape["tape_id"], arm, "complete", flush=True)
    outcomes = cross_outcomes(arms, protocol, model.workloads)
    record = {**{key: tape[key] for key in ("node", "lanes", "load", "seed")},
              **anchor, "arm_checks": arm_checks, "outcomes": outcomes}
    return {"check": check, "record": record, "work": dict(work),
            "status": dict(status), "executions": len(ARMS)}


def effect_signs(values, metric):
    tolerance = 1e-9 if metric in ("delta_T_mean", "delta_T_common_mean",
        "delta_T_common_p90", "delta_900_common_p90") else 0
    return {"raw_negative": sum(value is not None and value < 0 for value in values),
            "raw_zero": sum(value == 0 for value in values),
            "raw_positive": sum(value is not None and value > 0 for value in values),
            "beneficial": sum(value is not None and value < -tolerance for value in values),
            "near_zero": sum(value is not None and abs(value) <= tolerance for value in values),
            "harmful": sum(value is not None and value > tolerance for value in values),
            "unavailable": sum(value is None for value in values)}


def summarize_effects(records):
    summary = {"anchors": len(records)}
    for suffix in ("current", "BG"):
        summary[suffix + "_root_effects"] = {metric: effect_signs(
            [row["outcomes"]["root_effects"][suffix][metric] for row in records], metric)
            for metric in METRICS}
    summary["interaction"] = {metric: effect_signs(
        [row["outcomes"]["interaction"][metric] for row in records], metric) for metric in METRICS}
    pairs = [(row["outcomes"]["root_effects"]["current"]["delta_T_mean"],
              row["outcomes"]["root_effects"]["BG"]["delta_T_mean"]) for row in records]
    summary["mean_ranking_flips"] = sum(a is not None and b is not None
        and (a < -1e-9 and b > 1e-9 or a > 1e-9 and b < -1e-9) for a, b in pairs)
    summary["current_beneficial_BG_harmful"] = sum(a is not None and b is not None
        and a < -1e-9 and b > 1e-9 for a, b in pairs)
    summary["current_harmful_BG_beneficial"] = sum(a is not None and b is not None
        and a > 1e-9 and b < -1e-9 for a, b in pairs)
    summary["classes"] = [{"coordinate": coordinate,
        **{suffix: {metric: effect_signs([item[metric] for row in records
            for item in row["outcomes"]["root_effects"][suffix]["per_coordinate_deltas"]
            if item["coordinate"] == coordinate], metric)
            for metric in ("delta_T_mean", "delta_T_common_mean", "delta_T_common_p90",
                           "delta_900_common_p90", "delta_900_unfinished", "delta_1200_unfinished")}
            for suffix in ("current", "BG")}}
        for coordinate in sorted({item["coordinate"] for row in records
            for item in row["outcomes"]["root_effects"]["current"]["per_coordinate_deltas"]})]
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
        print("registered", len(expected["anchors"]), "anchors; missing", expected["missing_tape_ids"], flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if not same({key: value for key, value in protocol.items() if key != "registered_at_utc"}, expected):
        raise ValueError("registered sources, anchors, rules or literal tapes changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve existing continuation-cross diagnostic result")
    frozen = saved_runs()
    models = {model.node: model for model in load_models(protocol["models"])}
    anchors = {row["tape_id"]: row for row in protocol["anchors"]}
    work, status, records, checks = Counter(), Counter(), [], []
    executions = 0
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = [pool.submit(run_tape, models[tape["node"]], tape, protocol,
            frozen[tape["tape_id"]], anchors.get(tape["tape_id"])) for tape in protocol["tapes"]]
        for future in as_completed(futures):
            row = future.result()
            checks.append(row["check"])
            if row["record"] is not None:
                records.append(row["record"])
            work.update(row["work"])
            status.update(row["status"])
            executions += row["executions"]
            print("tape complete", row["check"]["tape_id"], "anchors complete", len(records), flush=True)
    order = {row["tape_id"]: index for index, row in enumerate(protocol["anchors"])}
    records.sort(key=lambda row: order[row["tape_id"]])
    if (len(records) != len(protocol["anchors"]) or len(checks) != len(protocol["tapes"])
            or executions != protocol["expected_new_trajectory_executions"]):
        raise ValueError("every registered tape, anchor, null and three cross arms required")
    work["total_option_probes"] = sum(value for key, value in work.items() if key.endswith("option_probes"))
    result = {"decision": "TWO_STEP_BG_INCUMBENT_CONTINUATION_CROSS_COMPLETE",
        "registered_at_utc": protocol["registered_at_utc"], "execution_checks_passed": True,
        "performance_gain_established": False, "previous_decision": protocol["previous_decision"],
        "scope": protocol["scope"], "registered_tapes": len(protocol["tapes"]),
        "anchors": len(records), "missing_tape_ids": protocol["missing_tape_ids"],
        "new_actual_trajectory_executions": executions, "reused_factual_arms": len(records),
        "prefix_checks": sorted(checks, key=lambda row: row["tape_id"]),
        "summary": summarize_effects(records),
        "strata": {
            "seed": [{"seed": seed, **summarize_effects([row for row in records if row["seed"] == seed])}
                     for seed in protocol["arrival_seeds"]],
            "node_load": [{"node": model["node"], "load": load,
                **summarize_effects([row for row in records if row["node"] == model["node"] and row["load"] == load])}
                for model in protocol["models"] for load in protocol["loads"]]},
        "records": records, "actual_work": dict(work), "status_counts": dict(status)}
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "anchors", "missing_tape_ids",
        "new_actual_trajectory_executions", "summary")}), flush=True)


if __name__ == "__main__":
    main()
