"""Causal observed-arrival-rate replacement of the confirmed age-tail rule."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.observed_rate_age_tail_lane_rollout import ObservedRateAgeTailLaneRollout
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.aligned_hardware_age_tail_constrained_lane_rollout import rollout_checks as age_tail_checks
from algorithm.experiments.aligned_hardware_performance_necessity import common_metrics, load_models, replay_matches
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_age_tail_constrained_lane_rollout_development_20260930"
DEFAULT = ROOT / "md/experiment_artifacts/aligned_hardware_observed_rate_age_tail_development_20260930"
LABELS = ("maxweight", "arrival_rollout")
NEW_SOURCES = (
    "algorithm/observed_rate_age_tail_lane_rollout.py",
    "algorithm/experiments/observed_rate_contextual_lane_refill_replay.py",
    "algorithm/experiments/aligned_hardware_observed_rate_age_tail_development.py",
)


def build_protocol(previous):
    for path, source in previous["execution_sources"].items():
        if (ROOT / path).read_text() != source:
            raise ValueError(f"preceding frozen dependency changed: {path}")
    return {
        "experiment": "observed_rate_age_tail_development_v1",
        "candidate_method": "cumulative_observed_rate_FIFO_age_mean_flow_with_MW_second_moment_constraint",
        "previous_protocol": str(PREVIOUS.with_name(PREVIOUS.name + "_protocol.json").relative_to(ROOT)),
        "previous_trajectories": str(PREVIOUS.with_name(PREVIOUS.name + "_trajectories.json.gz").relative_to(ROOT)),
        **{key: previous[key] for key in ("models", "tapes", "capacities", "loads", "arrival_seeds",
            "arrival_horizon", "observation_end", "execution_order", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "primary", "statistics", "budget")},
        "labels": LABELS,
        "method": "Keep the confirmed FIFO age S1/S2 root rule, all feasible nonidle roots, ordinary exact single-lane MW complete-drain continuation, eight shared independent planning samples, tau_max window clamped by public H, event cap128 and whole-decision MW overflow fallback, queue envelope and tie rules. Replace the true planning rate with per-class cumulative observed births divided by exposure=min(now,H). Observed births include previously admitted jobs and are incremented only upon actual causal release. At zero exposure use zero rates. No smoothing, prior count, warm-up, moving window, rate floor or parameter sweep. AfterH the exposure remainsH and the public planning arrival window is zero.",
        "rate_estimator": {"rule": "observed_arrival_counts / min(now,H); zero exposure gives zero rates",
                           "observation_origin": 0.0, "includes_admitted_history": True},
        "information": "Constructor accepts catalogue, public H, independent planning seed and fixed budgets; it has no arrival-rate or load input. Selector sees waiting Q, true FIFO waiting ages, complete lane availability, now/lane and per-class cumulative released arrival counts. The environment retains true rates solely to generate literal evaluation tapes; these rates never enter either selector. MW receives the same observable context and ignores ages/counts. The independently seeded planner is not passed the real future tape.",
        "validation": "On every development tape, the new observed-count engine's plain MW must reproduce all original control action/start/Q/lane/service/duration and job times. Retain all old age/risk/work checks. Per-frame estimates must equal the recorded observed counts divided by min(now,H), with zero-exposure zero rates; counts include admitted history and no future jobs. Charge causal birth-counter updates, context vector reads, every actual rate-estimation coordinate read/division, and all original preprocessing/rollout/generation costs. Overflow decisions do not rank truncated populations. Preserve all seed/group/per-class regressions.",
        "confirmation_plan": "Only after full development gates pass, freeze12 fresh actual arrival seeds10001..10012, retaining the exact same rule, metadata, budgets and1%/5%gates. The old known-rate1001..1012 results are already observed and are only historical references for this new variant.",
        "limitations": "Deterministic catalogue-reported2/4 cached independent-lane replay; public H and service catalogue remain known. Cumulative rate estimates assume the fixed-rate observation interval; no unknown-rate drift or physical execution guarantee. The finite-window sampled S2 constraint is a proxy, not an actual P90 bound. Three reused development seeds; any advance needs the separately frozen new-seed confirmation. Computing delays are excluded from the service clock.",
        "execution_sources": {**previous["execution_sources"], **{path: (ROOT / path).read_text() for path in NEW_SOURCES}},
    }


def rollout_checks(result, model, protocol):
    if not age_tail_checks(result, model, protocol):
        return False
    dimension = len(model.workloads)
    for frame in result["frames"]:
        info = frame["rollout"]
        counts = tuple(frame["observed_arrival_counts"])
        exposure = min(frame["start"], protocol["arrival_horizon"])
        rates = tuple(count / exposure if exposure > 0 else 0. for count in counts)
        if (tuple(info["observed_arrival_counts"]) != counts or tuple(info["estimated_arrival_rates"]) != rates
                or info["rate_exposure_s"] != exposure
                or info["rate_estimation_coordinate_reads"] != dimension
                or info["rate_estimation_divisions"] != (dimension if exposure > 0 else 0)):
            return False
    return True


def work_totals(result, planner, horizon):
    total = Counter(result["rollout_work"])
    total.update({"preprocessing_" + key: value for key, value in planner.preprocessing_work.items()})
    fields = ("planning_events", "planning_random_draws", "planning_rate_reads", "rollout_idle_calls",
              "waiting_age_reads", "flow_evaluations", "risk_filtered_candidates", "risk_feasible_candidates",
              "rate_estimation_coordinate_reads", "rate_estimation_divisions")
    for frame in result["frames"]:
        info = frame["rollout"]
        total.update({key: info[key] for key in fields})
        total["planning_overflows"] += info["planning_overflow"]
        total["risk_changed_unconstrained_calls"] += info["risk_changed_unconstrained_root"]
        total["pre_h_changed_calls" if frame["start"] < horizon else "post_h_changed_calls"] += info["changed"]
    total["max_rollout_calls"] = total.pop("maximum_rollout_calls")
    total["total_option_probes"] = (total["root_option_probes"] + total["rollout_option_probes"]
        + total["preprocessing_option_probes"] + total["preprocessing_age_profile_option_probes"]
        + total["preprocessing_rate_dimension_option_probes"])
    return dict(total)


def run_tape(model, tape, protocol, frozen_baseline):
    # True environment rates are deliberately absent from policy construction.
    planners = {label: ObservedRateAgeTailLaneRollout(model.options, protocol["arrival_horizon"],
                    f"arrival-plan-v1:{model.node}:{tape['load']}:online",
                    sample_count=protocol["planning_samples"], event_cap=protocol["planning_event_cap"])
                for label in LABELS}
    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"], protocol["arrival_horizon"],
                    protocol["observation_end"], planners[label].maxweight if label == "maxweight" else planners[label],
                    policy=label) for label in LABELS}
    return {**{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
            "baseline_reproduction_passed": replay_matches(policies["maxweight"], frozen_baseline),
            "rollout_checks_passed": rollout_checks(policies["arrival_rollout"], model, protocol),
            "snapshots": {key: common_metrics(policies, protocol[key]) for key in ("arrival_horizon", "observation_end")},
            "work": {label: work_totals(result, planners[label], protocol["arrival_horizon"])
                     for label, result in policies.items()}}, policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    previous = json.loads(PREVIOUS.with_name(PREVIOUS.name + "_protocol.json").read_text())
    protocol_path = args.output.with_name(args.output.name + "_protocol.json")
    if args.register:
        with protocol_path.open("x") as out:
            json.dump({**build_protocol(previous), "registered_at_utc": datetime.now(timezone.utc).isoformat()},
                      out, separators=(",", ":"), allow_nan=False)
        print(protocol_path, flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != json.loads(json.dumps(build_protocol(previous))):
        raise ValueError("registered estimator, models, budgets, sources or literal tapes changed")
    output, trajectories = args.output.with_suffix(".json"), args.output.with_name(args.output.name + "_trajectories.json.gz")
    if output.exists() or trajectories.exists():
        raise FileExistsError("preserve frozen results")
    with gzip.open(ROOT / protocol["previous_trajectories"], "rt") as source:
        old = {run["tape_id"]: run["policies"]["maxweight"] for run in json.load(source)["runs"]}
    previous_result = json.loads(PREVIOUS.with_suffix(".json").read_text())
    models = {model.node: model for model in load_models(protocol["models"])}
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_tape(models[tape["node"]], tape, protocol, old[tape["tape_id"]])
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(f"{tape['tape_id']}: pair complete; baseline={row['baseline_reproduction_passed']} observed-rate/age/risk/work={row['rollout_checks_passed']}", flush=True)
    with gzip.open(trajectories, "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "candidate_method": protocol["candidate_method"],
                   "runs": runs}, out, separators=(",", ":"), allow_nan=False)
    result = summarize(rows, protocol)
    result["decision"] = result["decision"].replace("ARRIVAL_AWARE_LANE_ROLLOUT", "OBSERVED_RATE_AGE_TAIL")
    result["candidate_method"] = protocol["candidate_method"]
    result["previous_known_rate_comparison"] = {
        "scope": "Descriptive reused development contrast, not a third primary-cohort member or gate.",
        "previous_decision": previous_result["decision"],
        "T_mean_flow_known_rate": previous_result["pooled"]["observation_end"]["arrival_rollout"]["restricted_mean_flow_s"],
        "T_mean_flow_observed_rate": result["pooled"]["observation_end"]["arrival_rollout"]["restricted_mean_flow_s"],
    }
    with output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "execution_checks_passed", "performance_gain_established", "pooled")}), flush=True)


if __name__ == "__main__":
    main()
