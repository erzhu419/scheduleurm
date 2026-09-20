"""Enumerate every inactive queue; no arrival tapes or experiment reruns."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from algorithm.backlog_gated_frame_candidates import build_backlog_gate
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.lookahead_frame_candidates import select_lookahead_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model

ARTIFACTS = Path(__file__).resolve().parents[2] / "md" / "experiment_artifacts"
SOURCE = ARTIFACTS / "aligned_hardware_backlog_gate_development_20260910.json"
OUTPUT = ARTIFACTS / "aligned_hardware_backlog_gate_inactive_states_20260910.json"


def diagnose(source_path=SOURCE, output_path=OUTPUT):
    source = json.loads(Path(source_path).read_text())
    model = load_model()
    gate = build_backlog_gate(model.options, model.lanes)
    states = [q for q in itertools.product(*(range(gate.capacity // w + 1) for w in gate.weights))
              if gate.load(q) <= gate.capacity]
    protocol = source["protocol"]
    rates_by_load = {row["load_factor"]: row["arrival_rates"] for row in source["certificates"]}
    terminal_rates = protocol["terminal_service_rates"]
    groups, attempts = [], 0
    for load in protocol["base_protocol"]["loads"]:
        for budget in protocol["base_protocol"]["budgets"]:
            differences, witnesses = 0, []
            for queues in states:
                support = select_budgeted_frame(model.options, queues, model.lanes, budget)
                predicted = select_lookahead_frame(
                    model.options, queues, model.lanes, budget,
                    arrival_rates=rates_by_load[load], terminal_service_rates=terminal_rates,
                    terminal_weight=1.0, max_relative_support_loss=0.01,
                )
                attempts += support.attempts + predicted.attempts
                if support.option_indices != predicted.option_indices:
                    differences += 1
                    if len(witnesses) < 3:
                        witnesses.append({"queue": queues, "v4_indices": support.option_indices,
                                          "lookahead_indices": predicted.option_indices,
                                          "v4_score": support.score, "lookahead_score": predicted.score})
            groups.append({"load": load, "budget": budget, "state_count": len(states),
                           "different_action_count": differences, "witnesses": witnesses})
    equivalent = all(row["different_action_count"] == 0 for row in groups)
    report = {
        "diagnostic": "exhaustive_inactive_backlog_gate_states", "status": "COMPLETE",
        "source": str(Path(source_path).resolve()), "scenario_id": model.scenario_id,
        "workloads": model.workloads, "gate_weights": gate.weights, "gate_capacity": gate.capacity,
        "inactive_state_count": len(states), "arrival_rates_by_load": rates_by_load,
        "terminal_service_rates": terminal_rates, "groups": groups,
        "all_inactive_actions_equal": equivalent, "arrival_tapes_used": False,
        "selector_calls": 2 * len(states) * len(groups), "diagnostic_candidate_attempts": attempts,
        "accounting": "Extra diagnostic double searches; excluded from the 108-run experiment candidate budget.",
        "conclusion": (
            "Inactive actions agree exhaustively; active branches use the same lookahead. The two action "
            "policies coincide on all integer queues for this catalogue, these recorded rates, and four budgets."
            if equivalent else "Some inactive states have different actions; witnesses identify them."
        ),
        "scope": "No claim for other arrival rates, terminal rates, catalogues, budgets, or gate thresholds.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = diagnose(args.source, args.output)
    print(json.dumps({"states": result["inactive_state_count"], "groups": len(result["groups"]),
                      "different_actions": sum(row["different_action_count"] for row in result["groups"])}))
