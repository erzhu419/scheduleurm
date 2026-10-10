"""Audit fixed roots with independent planning arrivals, without policy tuning.

The 24 states come from the preceding clearing policy's first altered action.
Eight planning paths select the new root. Thirty-two separate paths compare
that fixed root, ordinary MaxWeight and the old zero-arrival clearing root.
Only the full trajectory experiment can establish a performance increment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
from random import Random
from statistics import mean, stdev

from algorithm.arrival_aware_lane_rollout import (
    ArrivalAwareLaneRollout, sample_planning_arrivals,
)
from algorithm.experiments.aligned_hardware_performance_necessity import load_models


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_PROTOCOL = ROOT / "md/experiment_artifacts/aligned_hardware_visible_queue_clearing_development_20260930_protocol.json"
PREVIOUS_RESULT = ROOT / "md/experiment_artifacts/visible_queue_first_change_diagnosis_20260930.json"
DEFAULT = ROOT / "md/experiment_artifacts/arrival_aware_root_ranking_20260930"
SOURCE = "algorithm/experiments/arrival_aware_root_ranking.py"
CORE_SOURCE = "algorithm/arrival_aware_lane_rollout.py"
LABELS = ("arrival_aware", "maxweight", "visible_queue_clearing")


def extract_states(previous):
    """Keep causal state and the two already fixed action IDs; omit real futures."""
    states = []
    for row in previous["paired_runs"]:
        first = row["first_change"]
        state = {key: row[key] for key in ("tape_id", "node", "lanes", "load", "seed")}
        state["first_change"] = None if first is None else {
            **{key: first[key] for key in ("frame", "start", "queue", "available_before", "lane_ids")},
            "maxweight_action_id": first["old_root"]["action_id"],
            "visible_queue_action_id": first["new_root"]["action_id"],
        }
        states.append(state)
    return states


def build_protocol(previous_protocol, previous_result):
    for path, source in previous_protocol["execution_sources"].items():
        if (ROOT / path).read_text() != source:
            raise ValueError(f"frozen preceding development source changed: {path}")
    return {
        "experiment": "arrival_aware_root_ranking_v1",
        "previous_protocol": str(PREVIOUS_PROTOCOL.relative_to(ROOT)),
        "previous_result": str(PREVIOUS_RESULT.relative_to(ROOT)),
        "previous_registered_at_utc": previous_result["registered_at_utc"],
        **{key: previous_protocol[key] for key in ("models", "capacities", "arrival_horizon")},
        "states": extract_states(previous_result),
        "selection_samples": 8,
        "audit_samples": 32,
        "event_cap": 128,
        "selection_seed_template": "arrival-plan-v1:{node}:{load}:rank-selection",
        "audit_seed_template": "arrival-audit-v1:{node}:{load}:rank:{seed}",
        "rule": "Keep all 24 original first-change states, including absent or overflowing states. At each state, select once using 8 independent known-rate Poisson planning paths. Generate 32 separate audit paths and reuse the complete identical paths for every feasible root. Compare the already chosen arrival-aware root, the original MaxWeight root, and the original visible-queue clearing root. Arrivals stop after min(catalogue maximum effective duration, H-now), followed by complete ordinary MaxWeight draining on the true busy-lane state. Any audit path exceeding 128 arrivals leaves that state without a cost comparison; no truncation, resampling, replacement state, parameter tuning or audit-root selection. Report paired path cost differences and descriptive seed groups only. There is no performance promotion gate.",
        "cost": "Current waiting jobs' completion times relative to now plus each independent planned job's completion minus its planned arrival. Busy admitted jobs have root-independent fixed costs. The audit evaluates one root followed by ordinary MaxWeight, not the whole arrival-aware policy.",
        "limitations": "Retrospective fixed development states from three shared arrival seeds. Planning arrival rates and finite arrival-stop time are known experimental inputs. Independent conditional rollout ranking does not establish scheduling performance, physical execution gain, or a stability guarantee. The audit's later path costs never enter online selection.",
        "execution_sources": {**previous_protocol["execution_sources"],
                              CORE_SOURCE: (ROOT / CORE_SOURCE).read_text(),
                              SOURCE: (ROOT / SOURCE).read_text()},
    }


def paired_costs(values, reference):
    differences = [float(value - old) for value, old in zip(values, reference, strict=True)]
    return {
        "mean_cost_delta_job_s": mean(differences),
        "path_cost_delta_job_s": differences,
        "sample_sd_job_s": stdev(differences) if len(differences) > 1 else 0.,
        "lower_cost_paths": sum(value < 0 for value in differences),
        "higher_cost_paths": sum(value > 0 for value in differences),
        "equal_cost_paths": sum(value == 0 for value in differences),
    }


def run_state(model, state, protocol):
    row = {**state, "status": "NO_PRECEDING_CHANGED_ROOT"}
    first = state["first_change"]
    if first is None:
        return row
    rates = next(capacity["arrival_rates"] for capacity in protocol["capacities"]
                 if capacity["node"] == state["node"] and capacity["load"] == state["load"])
    policy = ArrivalAwareLaneRollout(model.options, rates, protocol["arrival_horizon"],
                                    protocol["selection_seed_template"].format(**state),
                                    sample_count=protocol["selection_samples"],
                                    event_cap=protocol["event_cap"])
    queue, available = tuple(first["queue"]), tuple(first["available_before"])
    now, lane = first["start"], first["lane_ids"][0]
    baseline = policy.maxweight(model.options, queue, available, now, lane)
    chosen = policy(model.options, queue, available, now, lane)
    by_action = {option.action_id: index for index, option in enumerate(model.options)}
    roots = {"arrival_aware": chosen.selection.option_indices[0],
             "maxweight": by_action[first["maxweight_action_id"]],
             "visible_queue_clearing": by_action[first["visible_queue_action_id"]]}
    if baseline.selection.option_indices != (roots["maxweight"],):
        raise ValueError("new exact MaxWeight differs at a frozen preceding state")
    selection_work = asdict(chosen)
    selection_work.pop("selection")
    row.update({"roots": {label: {"index": index, "action_id": model.options[index].action_id}
                          for label, index in roots.items()},
                "selection": selection_work,
                "preprocessing_work": policy.preprocessing_work,
                "identity_option_probes": baseline.root_option_probes,
                "selection_changed_vs_maxweight": roots["arrival_aware"] != roots["maxweight"]})
    window = min(max(max(1., option.duration) for option in model.options),
                 max(0., protocol["arrival_horizon"] - now))
    rng = Random(protocol["audit_seed_template"].format(**state))
    paths = tuple(sample_planning_arrivals(rates, window, rng, protocol["event_cap"])
                  for _ in range(protocol["audit_samples"]))
    row["audit_sampling"] = {
        "window_s": window, "samples": len(paths),
        "events": sum(len(path.events) for path in paths),
        "random_draws": sum(path.random_draws for path in paths),
        "rate_reads": sum(path.rate_reads for path in paths),
        "overflow_samples": sum(path.overflow for path in paths),
    }
    if any(path.overflow for path in paths):
        row["status"] = "AUDIT_EVENT_BUDGET_OVERFLOW"
        return row
    costs = {label: [] for label in LABELS}
    work = {name: 0 for name in ("root_option_probes", "rollout_calls", "rollout_option_probes", "rollout_idle_calls")}
    candidate_counts = []
    for path in paths:
        scored = policy.score_roots(queue, available, now, lane, (path,))
        for label, root in roots.items():
            costs[label].append(scored.costs[root])
        candidate_counts.append(scored.candidate_count)
        for name in work:
            work[name] += getattr(scored, name)
    row.update({"status": "INDEPENDENT_ROOT_AUDIT_COMPLETE",
                "audit_work": {**work, "candidate_evaluations": sum(candidate_counts)},
                "mean_cost_job_s": {label: float(sum(values, Fraction(0)) / len(values))
                                    for label, values in costs.items()},
                "path_cost_job_s": {label: [float(value) for value in values]
                                    for label, values in costs.items()},
                "paired_vs_maxweight": {label: paired_costs(costs[label], costs["maxweight"])
                                        for label in ("arrival_aware", "visible_queue_clearing")}})
    return row


def summarize(rows, protocol):
    if [row["tape_id"] for row in rows] != [state["tape_id"] for state in protocol["states"]]:
        raise ValueError("preserve every registered state in its original order")
    valid = [row for row in rows if row["status"] == "INDEPENDENT_ROOT_AUDIT_COMPLETE"]

    def comparisons(selected):
        if not selected:
            return None
        return {label: {
            "mean_state_cost_delta_job_s": mean(row["paired_vs_maxweight"][label]["mean_cost_delta_job_s"] for row in selected),
            "lower_cost_states": sum(row["paired_vs_maxweight"][label]["mean_cost_delta_job_s"] < 0 for row in selected),
            "higher_cost_states": sum(row["paired_vs_maxweight"][label]["mean_cost_delta_job_s"] > 0 for row in selected),
            "equal_cost_states": sum(row["paired_vs_maxweight"][label]["mean_cost_delta_job_s"] == 0 for row in selected),
        } for label in ("arrival_aware", "visible_queue_clearing")}

    return {"experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
            "decision": "INDEPENDENT_ROOT_RANKING_DIAGNOSIS_COMPLETE",
            "registered_states": len(rows), "audited_states": len(valid),
            "absent_states": sum(row["status"] == "NO_PRECEDING_CHANGED_ROOT" for row in rows),
            "audit_overflow_states": sum(row["status"] == "AUDIT_EVENT_BUDGET_OVERFLOW" for row in rows),
            "selection_changed_states": sum(row.get("selection_changed_vs_maxweight", False) for row in rows),
            "paired_vs_maxweight": comparisons(valid),
            "seed_groups": [{"seed": seed, "audited_states": len(selected),
                             "paired_vs_maxweight": comparisons(selected)}
                            for seed in sorted({state["seed"] for state in protocol["states"]})
                            for selected in ([row for row in valid if row["seed"] == seed],)],
            "paired_states": rows, "limitations": protocol["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    previous_protocol = json.loads(PREVIOUS_PROTOCOL.read_text())
    previous_result = json.loads(PREVIOUS_RESULT.read_text())
    path = args.output.with_name(args.output.name + "_protocol.json")
    if args.register:
        with path.open("x") as out:
            json.dump({**build_protocol(previous_protocol, previous_result),
                       "registered_at_utc": datetime.now(timezone.utc).isoformat()},
                      out, separators=(",", ":"), allow_nan=False)
        print(path, flush=True)
        return
    protocol = json.loads(path.read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(previous_protocol, previous_result):
        raise ValueError("registered root audit, fixed states, or execution sources changed")
    output = args.output.with_suffix(".json")
    if output.exists():
        raise FileExistsError("preserve existing root-ranking diagnostic")
    models = {model.node: model for model in load_models(protocol["models"])}
    rows = []
    for state in protocol["states"]:
        row = run_state(models[state["node"]], state, protocol)
        rows.append(row)
        contrast = row.get("paired_vs_maxweight", {}).get("arrival_aware", {}).get("mean_cost_delta_job_s")
        print(f"{state['tape_id']}: {row['status']} arrival-aware audit delta={contrast}", flush=True)
    result = summarize(rows, protocol)
    with output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "paired_states"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
