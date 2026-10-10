"""FIFO-age flow optimization constrained by predicted second-moment cost."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.age_tail_constrained_lane_rollout import AgeTailConstrainedLaneRollout
from algorithm.experiments.age_contextual_lane_refill_replay import simulate_age_contextual_trajectory
from algorithm.experiments.aligned_hardware_arrival_aware_lane_rollout import rollout_checks as population_checks
from algorithm.experiments.aligned_hardware_performance_necessity import common_metrics, load_models, replay_matches
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_arrival_aware_lane_rollout_development_20260930"
DEFAULT = ROOT / "md/experiment_artifacts/aligned_hardware_age_tail_constrained_lane_rollout_development_20260930"
# Retain the existing aggregate schema; the experiment and candidate_method identify the new rule.
LABELS = ("maxweight", "arrival_rollout")
NEW_SOURCES = (
    "algorithm/age_tail_constrained_lane_rollout.py",
    "algorithm/experiments/age_contextual_lane_refill_replay.py",
    "algorithm/experiments/aligned_hardware_age_tail_constrained_lane_rollout.py",
)


def build_protocol(previous):
    for path, source in previous["execution_sources"].items():
        if (ROOT / path).read_text() != source:
            raise ValueError(f"preceding frozen dependency changed: {path}")
    return {
        "experiment": "age_tail_constrained_lane_rollout_development_v1",
        "candidate_method": "FIFO_age_mean_flow_with_MW_second_moment_constraint",
        "previous_protocol": str(PREVIOUS.with_name(PREVIOUS.name + "_protocol.json").relative_to(ROOT)),
        "previous_trajectories": str(PREVIOUS.with_name(PREVIOUS.name + "_trajectories.json.gz").relative_to(ROOT)),
        **{key: previous[key] for key in ("models", "tapes", "capacities", "loads", "arrival_seeds",
            "arrival_horizon", "observation_end", "execution_order", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "primary", "statistics")},
        "labels": LABELS,
        "method": "Keep the preceding queue envelope, eight independent known-rate planning populations, tau_max window clamped by public H, event cap128 with whole-decision MW overflow fallback, and ordinary exact single-lane MW complete-drain continuation. Add actual FIFO waiting ages from released unadmitted jobs. Assign each admitted class's oldest jobs to sorted real completion offsets. For every root on the same populations compute S1=sum full job flow and S2=sum squared full job flow; current flow=waiting age+relative completion, planned flow=relative completion-planned arrival. Omit only root-independent already-admitted costs while preserving their complete lane commitments. Allow roots only if their sample-summed S2 is no larger than the MW root's; among allowed roots minimize S1, retain MW on ties, and change only for strict S1 improvement. No risk slack, weighted penalty, new coefficient, window change or sample sweep.",
        "information": "Only actual released FIFO waiting ages, counts, complete lane availability, catalogue, registered known Poisson rates, public arrival stop H and independent planner RNG reach the selector. No evaluation future, future IDs or audit outcomes. Plain MW receives the same observable ages and known metadata, and ignores them.",
        "budget": "Retain at most m feasible nonidle roots and8*m*(N+128-1+R*(ceil(window)+1)) continuation calls, each scanning m options including idle polling. Add at most8*m*(N+128) per-job flow and squared-flow evaluations in eligible scoring. Catalogue/offset preprocessing and generation draws/events are charged. The engine reads actual sum(Q) jobs to expose ages even outside the scoring envelope; this state extraction is charged and has no envelope-based bound. No LP/IP or fluid terminal approximation.",
        "validation": "The FIFO-age engine with prepared plain MW must reproduce every original baseline action/start/Q/lane/service/duration and job time on all24 literal tapes. Check each recorded age list matches waiting Q in FIFO order, bounded continuation work, predicted S1 strict improvement when changed, and predicted S2 nonincrease for every scored root decision. Full performance gates unchanged; retain all regressions and three paired seed aggregates. The old unconstrained result is reused descriptively, not rerun or treated as a third primary-cohort member.",
        "limitations": "Deterministic catalogue-reported2/4 cached independent-lane replay, known arrival rates and H, three reused development seeds. A finite-window sample second-moment constraint is a tail proxy, not an actual P90 bound, physical execution result or asynchronous stability proof. An advance requires separate frozen new-seed confirmation; wall time is not the scheduling endpoint.",
        "execution_sources": {**previous["execution_sources"], **{path: (ROOT / path).read_text() for path in NEW_SOURCES}},
    }


def rollout_checks(result, model, protocol):
    if not population_checks(result, model, protocol):
        return False
    for frame in result["frames"]:
        info, n = frame["rollout"], sum(frame["queue"])
        ages = frame["waiting_ages"]
        if tuple(map(len, ages)) != tuple(frame["queue"]):
            return False
        if any(any(age < 0 for age in group) or list(group) != sorted(group, reverse=True) for group in ages):
            return False
        ranked = info["eligible"] and not info["planning_overflow"]
        if ranked:
            if info["chosen_predicted_squared_flow_s2"] > info["baseline_predicted_squared_flow_s2"]:
                return False
            if not 1 <= info["risk_feasible_candidates"] <= info["candidate_count"]:
                return False
            if info["risk_feasible_candidates"] + info["risk_filtered_candidates"] != info["candidate_count"]:
                return False
            if info["flow_evaluations"] > info["candidate_count"] * protocol["planning_samples"] * (n + protocol["planning_event_cap"]):
                return False
        elif info["flow_evaluations"] or info["risk_filtered_candidates"] or info["risk_changed_unconstrained_root"]:
            return False
    return True


def work_totals(result, planner, horizon):
    total = Counter(result["rollout_work"])
    total.update({"preprocessing_" + key: value for key, value in planner.preprocessing_work.items()})
    fields = ("planning_events", "planning_random_draws", "planning_rate_reads", "rollout_idle_calls",
              "waiting_age_reads", "flow_evaluations", "risk_filtered_candidates", "risk_feasible_candidates")
    for frame in result["frames"]:
        info = frame["rollout"]
        total.update({key: info[key] for key in fields})
        total["planning_overflows"] += info["planning_overflow"]
        total["risk_changed_unconstrained_calls"] += info["risk_changed_unconstrained_root"]
        total["pre_h_changed_calls" if frame["start"] < horizon else "post_h_changed_calls"] += info["changed"]
    total["max_rollout_calls"] = total.pop("maximum_rollout_calls")
    total["total_option_probes"] = (total["root_option_probes"] + total["rollout_option_probes"]
        + total["preprocessing_option_probes"] + total["preprocessing_age_profile_option_probes"])
    return dict(total)


def run_tape(model, tape, protocol, frozen_baseline):
    rates = next(row["arrival_rates"] for row in protocol["capacities"]
                 if row["node"] == tape["node"] and row["load"] == tape["load"])
    planners = {label: AgeTailConstrainedLaneRollout(model.options, rates, protocol["arrival_horizon"],
                    f"arrival-plan-v1:{model.node}:{tape['load']}:online",
                    sample_count=protocol["planning_samples"], event_cap=protocol["planning_event_cap"])
                for label in LABELS}
    policies = {label: simulate_age_contextual_trajectory(model, tape["arrivals"], protocol["arrival_horizon"],
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
        raise ValueError("registered age constraint, sources, budget or literal tapes changed")
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
        print(f"{tape['tape_id']}: pair complete; baseline={row['baseline_reproduction_passed']} age/risk/work={row['rollout_checks_passed']}", flush=True)
    with gzip.open(trajectories, "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "candidate_method": protocol["candidate_method"],
                   "runs": runs}, out, separators=(",", ":"), allow_nan=False)
    result = summarize(rows, protocol)
    result["decision"] = result["decision"].replace("ARRIVAL_AWARE_LANE_ROLLOUT", "AGE_TAIL_CONSTRAINED_LANE_ROLLOUT")
    result["candidate_method"] = protocol["candidate_method"]
    result["previous_unconstrained_comparison"] = {
        "scope": "Descriptive paired-tape aggregate; previous outcomes reused without new primary cohort or gate.",
        "previous_decision": previous_result["decision"],
        "T_mean_flow_previous": previous_result["pooled"]["observation_end"]["arrival_rollout"]["restricted_mean_flow_s"],
        "T_mean_flow_current": result["pooled"]["observation_end"]["arrival_rollout"]["restricted_mean_flow_s"],
    }
    with output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "execution_checks_passed", "performance_gain_established", "pooled")}), flush=True)


if __name__ == "__main__":
    main()
