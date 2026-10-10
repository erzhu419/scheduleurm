"""One future candidate redecision under a frozen finite search budget."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from math import ceil
from pathlib import Path

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.redecision_continuous_lane_rollout import RedecisionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area, check_rollout as original_checks
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_continuous_arrival_age_tail_development_20260930"
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_redecision_continuous_development_20260930"
LABELS = ("maxweight", "arrival_rollout")
SOURCES = ("algorithm/redecision_continuous_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_redecision_continuous.py")
EXTRA_OPTION_FIELDS = ("pair_control_option_probes", "pair_root_option_probes", "pair_rollout_option_probes",
                       "continuation_root_option_probes", "continuation_rollout_option_probes", "restore_control_option_probes")


def path(prefix, suffix):
    return prefix.with_name(prefix.name + suffix)


def build_protocol():
    old = json.loads(path(PREVIOUS, "_protocol.json").read_text())
    result = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if result["decision"] != "CONTINUOUS_ARRIVAL_AGE_TAIL_DEVELOPMENT_HOLD":
        raise ValueError("original continuous-arrival HOLD must remain recorded")
    for name, source in old["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    return {
        "experiment": "redecision_continuous_development_v1", "phase": "development",
        "previous_protocol": str(path(PREVIOUS, "_protocol.json").relative_to(ROOT)),
        "previous_trajectories": str(path(PREVIOUS, "_trajectories.json.gz").relative_to(ROOT)),
        **{key: old[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "tapes", "arrival_seeds",
            "environment_arrival_end", "observation_end", "birth_cohort", "arrival_horizon",
            "cutoff_semantics", "primary", "information", "limitations")},
        "labels": list(LABELS),
        "method": "Run the frozen original eight-sample, cap128 aggregate S2 policy first. Only strict changed proposals trigger a comparison of that candidate against same-state MW on the same saved eight outer planning paths. Each branch executes one subsequent nonidle decision of the frozen original policy, with eight fresh full-tau_max planning samples, causal released counts/now and a shared per-outer-sample independent stream across the root pair, then MW complete drain. Accept only strict paired S1 improvement with paired S2 nonincrease; otherwise execute MW. No real future tape reaches either policy. Original proposal sampling and its continuation costs remain charged even when rejected.",
        "budget": "Finite-depth only: at most16 inner decisions; inner queues outside the original R*B envelope fall back to MW, with no recursive new policy call. For m options, s=8, A=R*B+128, C=A+R*(ceil(tau_max)+1), per-call option-probe cap=4m+2+s*m*m*C+2s*(m*C+2m+s*m*m*C), flow-evaluation cap=s*m*A+2s*(A+s*m*A). Initial proposal plus both outer branches and every inner planning generation/flow/root/continuation operation are charged. Catalogue preprocessing and actual engine context extraction are separately charged. These explicit enlarged bounds are fixed before results, not claimed to equal the old cheaper policy.",
        "statistics": "Three reused development seeds41..43, eight paired catalogue/load cells per seed. No new independent confirmation or 24-tape independence claim.",
        "execution_sources": {**old["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def check_rollout(result, model, protocol, planner):
    if not original_checks(result, model, protocol, planner):
        return False
    m, s, cap, r = len(model.options), protocol["planning_samples"], protocol["planning_event_cap"], model.lanes
    for frame in result["frames"]:
        info, n = frame["rollout"], sum(frame["queue"])
        work = info["redecision_work"]
        if (info["search_option_probe_limit"], info["search_flow_evaluation_limit"]) != planner.budget_limits(frame["queue"], r):
            return False
        probes = info["root_option_probes"] + info["rollout_option_probes"] + sum(work.get(key, 0) for key in EXTRA_OPTION_FIELDS)
        flow = info["flow_evaluations"] + work.get("pair_flow_evaluations", 0) + work.get("continuation_flow_evaluations", 0)
        if probes > info["search_option_probe_limit"] or flow > info["search_flow_evaluation_limit"]:
            return False
        if not info["redecision_attempted"]:
            if work or info["redecision_rejected"] or info["changed"]:
                return False
            continue
        if (work["pair_control_option_probes"] != m or work["pair_root_option_probes"] != 2
                or work["pair_rollout_option_probes"] != m * work["pair_rollout_calls"]
                or work["pair_flow_evaluations"] != 2 * (s * n + info["planning_events"])
                or work["pair_risk_comparisons"] != 2 or work.get("continuation_calls", 0) > 2 * s
                or work.get("continuation_rollout_option_probes", 0) != m * work.get("continuation_rollout_calls", 0)
                or work.get("restore_control_option_probes", 0) != m * info["redecision_rejected"]):
            return False
        inner_calls = work.get("continuation_calls", 0)
        if (not m * inner_calls <= work.get("continuation_root_option_probes", 0) <= 2 * m * inner_calls
                or work.get("continuation_seed_initializations", 0) != inner_calls
                or work.get("continuation_rate_estimation_coordinate_reads", 0) != len(model.workloads) * inner_calls
                or work.get("pair_event_releases", 0) != 2 * info["planning_events"]):
            return False
        polling = r * (ceil(planner.maximum_duration) + 1)
        if (work["pair_rollout_calls"] > 2 * s * (n + cap - 1 + polling)
                or work.get("continuation_rollout_calls", 0) > 2 * s * s * m * (r * planner.batch_width + cap - 1 + polling)):
            return False
        if info["chosen_redecision_squared_flow_s2"] > info["baseline_redecision_squared_flow_s2"]:
            return False
        if info["changed"] != (not info["redecision_rejected"]):
            return False
        if info["changed"] and not info["chosen_redecision_flow_s"] < info["baseline_redecision_flow_s"]:
            return False
        if info["redecision_rejected"] and info["chosen_redecision_flow_s"] != info["baseline_redecision_flow_s"]:
            return False
    return True


def extra_work(result, planner, horizon):
    total = work_totals(result, planner, horizon)
    for frame in result["frames"]:
        info = frame["rollout"]
        for key, value in info.get("redecision_work", {}).items():
            total[key] = total.get(key, 0) + value
        total["redecision_attempts"] = total.get("redecision_attempts", 0) + info.get("redecision_attempted", False)
        total["redecision_rejections"] = total.get("redecision_rejections", 0) + info.get("redecision_rejected", False)
    total["total_option_probes"] += sum(total.get(key, 0) for key in EXTRA_OPTION_FIELDS)
    total["total_option_probes"] += sum(planner.preprocessing_work.get("continuation_" + key, 0)
        for key in ("option_probes", "age_profile_option_probes", "rate_dimension_option_probes"))
    total["total_flow_evaluations"] = total["flow_evaluations"] + total.get("pair_flow_evaluations", 0) + total.get("continuation_flow_evaluations", 0)
    return total


def run_tape(model, tape, protocol, frozen_baseline):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {
        "maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed, protocol["planning_samples"], protocol["planning_event_cap"]),
        "arrival_rollout": RedecisionContinuousLaneRollout(model.options, seed, protocol["planning_samples"], protocol["planning_event_cap"]),
    }
    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"],
        planners[label].maxweight if label == "maxweight" else planners[label], policy=label)
        for label in LABELS}
    row = {**{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": replay_matches(policies["maxweight"], frozen_baseline),
        "rollout_checks_passed": check_rollout(policies["arrival_rollout"], model, protocol, planners["arrival_rollout"]),
        "snapshots": {key: cohort_metrics(policies, protocol[key], protocol["birth_cohort"])
                      for key in ("arrival_horizon", "observation_end")},
        "work": {label: extra_work(result, planners[label], protocol["arrival_horizon"])
                 for label, result in policies.items()},
        "middle_queue_area_s": {label: queue_area(result) for label, result in policies.items()}}
    return row, policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    expected = build_protocol()
    if args.register:
        with path(BASE, "_protocol.json").open("x") as out:
            json.dump({**expected, "registered_at_utc": datetime.now(timezone.utc).isoformat()}, out,
                      separators=(",", ":"), allow_nan=False)
        print(path(BASE, "_protocol.json"), flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources, budget, rules or literal tapes changed")
    if BASE.with_suffix(".json").exists() or path(BASE, "_trajectories.json.gz").exists():
        raise FileExistsError("preserve existing results")
    with gzip.open(ROOT / protocol["previous_trajectories"], "rt") as source:
        old = {run["tape_id"]: run["policies"]["maxweight"] for run in json.load(source)["runs"]}
    models = {model.node: model for model in load_models(protocol["models"])}
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_tape(models[tape["node"]], tape, protocol, old[tape["tape_id"]])
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(tape["tape_id"], "complete", row["baseline_reproduction_passed"], row["rollout_checks_passed"], flush=True)
    result = summarize(rows, protocol)
    status = "REPLAY_FAIL" if not result["execution_checks_passed"] else "PASS" if result["performance_gain_established"] else "HOLD"
    result["decision"] = "REDECISION_CONTINUOUS_DEVELOPMENT_" + status
    result["birth_cohort"], result["cutoff_semantics"] = protocol["birth_cohort"], protocol["cutoff_semantics"]
    result["total_executed_trajectories"] = 2 * len(rows)
    result["middle_queue_area_s"] = {label: sum(row["middle_queue_area_s"][label] for row in rows) for label in LABELS}
    previous = json.loads(PREVIOUS.with_suffix(".json").read_text())
    result["previous_candidate_comparison"] = {
        "scope": "Descriptive reused development contrast; no third primary trajectory or promotion gate.",
        "decision": previous["decision"], "pooled": previous["pooled"],
    }
    with gzip.open(path(BASE, "_trajectories.json.gz"), "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "runs": runs}, out, separators=(",", ":"), allow_nan=False)
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "pooled", "failed_secondary_groups")}), flush=True)


if __name__ == "__main__":
    main()
