"""Recheck the previously diagnosed low-load counterexample without new arrivals."""
from dataclasses import asdict
import json

from algorithm.batch_rollout_frame_candidates import select_batch_rollout_frame
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS


def diagnose():
    source = ARTIFACTS / "aligned_hardware_lookahead_continuation_development_20260910.json"
    previous = json.loads(source.read_text())
    # Every low-load substitution is included; the frozen study has one.
    model = load_model()
    certificate = capacity_certificate(model, 0.5)
    mu = terminal_service_rates(model, certificate)
    cases = []
    for run in previous["runs"]:
        if run["load"] != 0.5:
            continue
        for case in run["cases"]:
            result = select_batch_rollout_frame(model.options, case["queue"], model.lanes, 26,
                                                arrival_rates=certificate["arrival_rates"],
                                                terminal_service_rates=mu)
            cases.append({
                "load": run["load"], "seed": run["seed"], "frame_index": case["frame_index"],
                "queue": case["queue"], "previous_effect": case["effect"],
                "previous_v4_indices": case["v4_action"]["option_indices"],
                "previous_lookahead_indices": case["lookahead_action"]["option_indices"],
                "selection": asdict(result),
                "retains_previous_lookahead_action": list(result.option_indices) == case["lookahead_action"]["option_indices"],
            })
    report = {
        "diagnostic": "batch_rollout_previous_low_load_counterexample", "source": str(source),
        "role": "post-run descriptive recheck; no parameter changes and no new arrival tape or continuation replay",
        "cases": cases,
        "diagnostic_candidate_attempts": sum(row["selection"]["attempts"] for row in cases),
        "interpretation": "Previous realized effects refer to the fixed common-v4 continuation study, not a fresh run of the new full policy.",
    }
    output = ARTIFACTS / "aligned_hardware_batch_rollout_known_state_20260910.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = diagnose()
    print(json.dumps({"cases": len(result["cases"]),
                      "retained_previous_lookahead": sum(row["retains_previous_lookahead_action"] for row in result["cases"])}))
