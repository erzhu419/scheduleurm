"""Known-rate asynchronous rollout on the unchanged 24 development tapes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from math import ceil
from pathlib import Path

from algorithm.arrival_aware_lane_rollout import ArrivalAwareLaneRollout
from algorithm.experiments.aligned_hardware_performance_necessity import common_metrics, load_models, replay_matches
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_visible_queue_clearing_development_20260930"
DEFAULT = ROOT / "md/experiment_artifacts/aligned_hardware_arrival_aware_lane_rollout_development_20260930"
LABELS = ("maxweight", "arrival_rollout")
NEW_SOURCES = (
    "algorithm/arrival_aware_lane_rollout.py",
    "algorithm/experiments/aligned_hardware_arrival_aware_lane_rollout.py",
    "algorithm/experiments/arrival_aware_lane_rollout_summary.py",
)


def build_protocol(previous):
    # Preserve the original population and execution clock; no new rate fitting.
    sources = {path: text for path, text in previous["execution_sources"].items()
               if path not in ("algorithm/visible_queue_clearing_rollout.py",
                               "algorithm/experiments/aligned_hardware_visible_queue_clearing.py",
                               "algorithm/experiments/visible_queue_clearing_summary.py")}
    for path, text in sources.items():
        if (ROOT / path).read_text() != text:
            raise ValueError(f"registered execution dependency changed: {path}")
    return {
        "experiment": "arrival_aware_lane_rollout_development_v1",
        "previous_protocol": str(PREVIOUS.with_name(PREVIOUS.name + "_protocol.json").relative_to(ROOT)),
        "previous_trajectories": str(PREVIOUS.with_name(PREVIOUS.name + "_trajectories.json.gz").relative_to(ROOT)),
        **{key: previous[key] for key in ("models", "tapes", "capacities", "loads", "arrival_seeds",
                                         "arrival_horizon", "observation_end", "execution_order", "performance")},
        "labels": LABELS, "planning_samples": 8, "planning_event_cap": 128,
        "planning_seed_rule": "arrival-plan-v1:{node}:{load}:online",
        "method": "When 0<N=sum(waiting Q)<=R*B, B=max catalogue batch width, enumerate all feasible nonidle single-lane roots. Sample eight independent Poisson planning populations using the declared arrival rates, over min(tau_max,max(0,H-now)); share the complete same population across roots. Execute each root followed by ordinary exact single-lane MaxWeight until every current and planned job is admitted, and retain all exact completion offsets through complete drain. Minimize mean total flow of current waiting and planned jobs; current ages and already admitted completions are root-independent constants. Keep the MaxWeight root on ties, change only for strict predicted improvement. Outside the queue envelope or if any planning population exceeds 128 arrivals, use ordinary MaxWeight without truncated-population scoring.",
        "information": "The planner receives current waiting counts, full lane availability, catalogue, registered known Poisson rates, public H stop contract and an independent RNG. It receives neither the evaluation tape nor actual future jobs. The MaxWeight comparator is constructed with the same known metadata but does not use the arrival rates. This is a known-rate experiment, not an unknown-rate or equal-information-use claim.",
        "budget": "Eight paths per eligible decision, at most128 arrivals per path, inspect a129th only to detect overflow. Any overflow discards the whole ranking decision. At most m feasible roots and8*m*(N+128-1+R*(ceil(window)+1)) continuation calls, including paid idle polling. Each continuation scans m original options; root scans, preprocessing option/offset reads, arrival generation draws and events are separately charged. No fluid terminal, LP/IP, actual future input or parameter search.",
        "clock": previous["clock"], "primary": previous["primary"], "statistics": previous["statistics"],
        "validation": "New prepared exact MaxWeight must reproduce every action/start/Q/lane/service/duration and job timestamp of the frozen original MaxWeight on all24 tapes. Check feasible certificates, strict predicted nonincrease, explicit overflow/envelope fallback and cumulative work including idle continuation. Rank diagnostic uses another32 independent planning samples on all24 previous first-change states, with no feedback to online decisions and no separate promotion gate. All24 actual performance pairs run regardless of ranking outcomes.",
        "limitations": "Deterministic cached independent-lane replay with catalogue-reported2/4 lanes, not physical execution or an asynchronous stability proof. Full drain completes the finite planning population, but the tau_max arrival window omits later new jobs. Rates and H are known benchmark metadata. Reused three development seeds; any advance requires separately frozen independent confirmation. Computing wall time is not a performance endpoint.",
        "execution_sources": {**sources, **{path: (ROOT / path).read_text() for path in NEW_SOURCES}},
    }


def rollout_checks(result, model, protocol):
    m = len(model.options)
    limit = model.lanes * max(sum(option.requirements) for option in model.options)
    for frame in result["frames"]:
        info, n = frame["rollout"], sum(frame["queue"])
        if info["queue_limit"] != limit or info["eligible"] != (0 < n <= limit):
            return False
        if info["candidate_count"] > m or info["rollout_option_probes"] != m * info["rollout_calls"]:
            return False
        ranked = info["eligible"] and not info["planning_overflow"]
        if info["root_option_probes"] != (2 * m if ranked else m):
            return False
        if ranked and info["sample_count"] != protocol["planning_samples"]:
            return False
        calls_cap = (protocol["planning_samples"] * info["candidate_count"] *
                     (n + protocol["planning_event_cap"] - 1 + model.lanes * (ceil(info["window_s"]) + 1)))
        if info["rollout_calls"] > calls_cap:
            return False
        if not info["eligible"] or info["planning_overflow"]:
            if info["changed"] or info["rollout_calls"] or info["candidate_count"]:
                return False
        elif info["chosen_predicted_completion_s"] > info["baseline_predicted_completion_s"]:
            return False
        if info["changed"] and not (info["chosen_predicted_completion_s"] < info["baseline_predicted_completion_s"]):
            return False
    return True


def work_totals(result, planner, horizon):
    total = Counter(result["rollout_work"])
    total.update({"preprocessing_" + key: value for key, value in planner.preprocessing_work.items()})
    fields = ("planning_events", "planning_random_draws", "planning_rate_reads", "rollout_idle_calls")
    for frame in result["frames"]:
        info = frame["rollout"]
        total.update({key: info[key] for key in fields})
        total["planning_overflows"] += info["planning_overflow"]
        total["pre_h_changed_calls" if frame["start"] < horizon else "post_h_changed_calls"] += info["changed"]
    total["max_rollout_calls"] = total.pop("maximum_rollout_calls")
    total["total_option_probes"] = (total["root_option_probes"] + total["rollout_option_probes"]
                                    + total["preprocessing_option_probes"])
    return dict(total)


def run_tape(model, tape, protocol, frozen_baseline):
    rates = next(row["arrival_rates"] for row in protocol["capacities"]
                 if row["node"] == tape["node"] and row["load"] == tape["load"])
    planners = {label: ArrivalAwareLaneRollout(model.options, rates, protocol["arrival_horizon"],
                       f"arrival-plan-v1:{model.node}:{tape['load']}:online",
                       sample_count=protocol["planning_samples"], event_cap=protocol["planning_event_cap"])
                for label in LABELS}
    policies = {label: simulate_contextual_trajectory(model, tape["arrivals"], protocol["arrival_horizon"],
                    protocol["observation_end"], planners[label].maxweight if label == "maxweight" else planners[label],
                    policy=label) for label in LABELS}
    reproduction = replay_matches(policies["maxweight"], frozen_baseline)
    valid = rollout_checks(policies["arrival_rollout"], model, protocol)
    row = {**{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
           "baseline_reproduction_passed": reproduction, "rollout_checks_passed": valid,
           "snapshots": {key: common_metrics(policies, protocol[key]) for key in ("arrival_horizon", "observation_end")},
           "work": {label: work_totals(result, planners[label], protocol["arrival_horizon"])
                    for label, result in policies.items()}}
    return row, policies


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
        raise ValueError("registered method, budgets, models, literal tapes or sources changed")
    output = args.output.with_suffix(".json")
    trajectories = args.output.with_name(args.output.name + "_trajectories.json.gz")
    if output.exists() or trajectories.exists():
        raise FileExistsError("preserve frozen results")
    with gzip.open(ROOT / protocol["previous_trajectories"], "rt") as source:
        old = {row["tape_id"]: row["policies"]["maxweight"] for row in json.load(source)["runs"]}
    models = {model.node: model for model in load_models(protocol["models"])}
    rows, runs = [], []
    for tape in protocol["tapes"]:
        row, policies = run_tape(models[tape["node"]], tape, protocol, old[tape["tape_id"]])
        rows.append(row)
        runs.append({"tape_id": tape["tape_id"], "policies": policies})
        print(f"{tape['tape_id']}: pair complete; baseline={row['baseline_reproduction_passed']} work={row['rollout_checks_passed']}", flush=True)
    with gzip.open(trajectories, "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "runs": runs},
                  out, separators=(",", ":"), allow_nan=False)
    from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize
    result = summarize(rows, protocol)
    with output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "execution_checks_passed", "performance_gain_established", "pooled")}), flush=True)


if __name__ == "__main__":
    main()
