"""Development test of one-step Poisson lookahead with a fluid terminal value.

The terminal reference clears each class at fixed feasible time-average rates,
with no arrivals after the next boundary. It is an analytic value proxy, not
another candidate search or a physical finite-batch clearing guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.lookahead_frame_candidates import select_lookahead_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_mechanism import _assert_frozen_sources
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_support_cap_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, PolicyRecorder, summarize_runs


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_lookahead_development_20260910.json"
POLICIES = ("v4_support", "holding_cap_1pct", "frame_area_cap_1pct", "lookahead_cap_1pct")


def terminal_service_rates(model, certificate):
    """The same fixed feasible boundary time allocation serves every load."""
    weights = certificate["boundary_time_weights"]
    return tuple(model.lanes * sum(
        weight * option.requirements[i] / max(1.0, option.duration)
        for weight, option in zip(weights, model.options, strict=True)
    ) for i in range(len(model.workloads)))


class LookaheadRecorder:
    def __init__(self, arrival_rates, service_rates, terminal_weight):
        self.arrival_rates, self.service_rates = arrival_rates, service_rates
        self.terminal_weight = terminal_weight
        self.now, self.selector_wall_s = 0.0, 0.0
        self.frames = []
        self.cost_evaluations = self.offset_reads = self.next_queue_evaluations = self.terminal_evaluations = 0
        self.excluded_candidates = self.lower_cost_frames = self.lower_support_frames = self.search_exact_frames = 0
        self.minimum_seed_margin = self.minimum_floor_margin = float("inf")
        self.maximum_relative_loss = self.maximum_cost_excess = self.cost_reduction_sum = 0.0

    def __call__(self, options, queues, lanes, budget):
        start = perf_counter()
        selected = select_lookahead_frame(
            options, queues, lanes, budget, arrival_rates=self.arrival_rates,
            terminal_service_rates=self.service_rates, terminal_weight=self.terminal_weight,
            max_relative_support_loss=0.01,
        )
        self.selector_wall_s += perf_counter() - start
        self.cost_evaluations += selected.cost_candidate_evaluations
        self.offset_reads += selected.completion_offset_reads
        self.next_queue_evaluations += selected.next_queue_coordinate_evaluations
        self.terminal_evaluations += selected.terminal_value_coordinate_evaluations
        self.excluded_candidates += selected.support_cap_excluded_candidate_count
        saving = selected.support_incumbent_lookahead_cost - selected.lookahead_cost
        self.lower_cost_frames += int(saving > 1e-10)
        self.lower_support_frames += int(selected.score < selected.support_incumbent_score - 1e-10)
        self.search_exact_frames += int(selected.support_search_exact)
        self.minimum_seed_margin = min(self.minimum_seed_margin, selected.score - selected.seed_support)
        self.minimum_floor_margin = min(self.minimum_floor_margin, selected.score - selected.support_floor)
        self.maximum_relative_loss = max(self.maximum_relative_loss, selected.relative_support_loss)
        self.maximum_cost_excess = max(self.maximum_cost_excess, -saving)
        self.cost_reduction_sum += saving
        self.frames.append({"start_s": self.now, "duration_s": selected.duration,
                            "queue_before": tuple(queues), "departures": selected.service,
                            "option_indices": selected.option_indices})
        self.now += selected.duration
        return selected

    def accounting(self):
        count = len(self.frames)
        return {
            "selector_wall_s": self.selector_wall_s, "selector_mean_wall_ms": 1000 * self.selector_wall_s / count,
            "cost_candidate_evaluations": self.cost_evaluations,
            "mean_cost_candidate_evaluations": self.cost_evaluations / count,
            "completion_offset_reads": self.offset_reads,
            "next_queue_coordinate_evaluations": self.next_queue_evaluations,
            "terminal_value_coordinate_evaluations": self.terminal_evaluations,
            "support_cap_excluded_candidates": self.excluded_candidates,
            "lower_objective_frames": self.lower_cost_frames,
            "lower_support_than_search_incumbent_frames": self.lower_support_frames,
            "support_search_exact_frames": self.search_exact_frames,
            "minimum_seed_support_margin": self.minimum_seed_margin,
            "minimum_support_floor_margin": self.minimum_floor_margin,
            "maximum_relative_support_loss": self.maximum_relative_loss,
            "maximum_objective_excess_over_support_incumbent": self.maximum_cost_excess,
            "mean_objective_reduction_job_seconds": self.cost_reduction_sum / count,
        }


def _check_frozen_sources():
    _assert_frozen_sources()
    saved = json.loads((ARTIFACTS / "aligned_hardware_support_cap_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != saved:
        raise ValueError("the confirmed support-cap source must remain frozen")


def build_experiment(output=DEFAULT_OUTPUT):
    _check_frozen_sources()
    development = json.loads((ARTIFACTS / "aligned_hardware_support_cap_development_20260910.json").read_text())
    base = development["protocol"]["base_protocol"]
    model = load_model()
    if json.loads(json.dumps(model.snapshot())) != development["model"]:
        raise ValueError("lookahead experiment changed the frozen frame model")
    certificates = [capacity_certificate(model, load) for load in base["loads"]]
    if json.loads(json.dumps(certificates)) != development["certificates"]:
        raise ValueError("lookahead experiment changed the registered capacity model")
    service_rates = terminal_service_rates(model, certificates[0])
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "base_protocol": base,
        "role": "development only; previously used confirmation seeds are excluded",
        "policies": POLICIES, "main_policy": "lookahead_cap_1pct", "main_budget": 26,
        "max_relative_support_loss": 0.01,
        "eligibility": "S(a) >= max(S_seed, 0.99*S_inc), using only the original paid pool",
        "current_frame_area": "A=(sum(Q)-sum(D))*tau+sum(completion_offsets)+sum(lambda)*tau^2/2",
        "next_queue": "Q_next=Q-D+independent Poisson(lambda*tau); m=Q-D+lambda*tau",
        "terminal_value": "V(q)=sum(q_i^2/(2*mu_i)); E[V(Q_next)]=sum((m_i^2+lambda_i*tau)/(2*mu_i))",
        "objective": "J=A+terminal_weight*E[V(Q_next)] in job-seconds; main weight 1, area-only control weight 0",
        "terminal_service_rates": service_rates,
        "terminal_reference_time_weights": certificates[0]["boundary_time_weights"],
        "terminal_reference_source": "one fixed feasible boundary time allocation computed from the first registered load; identical mu at every load",
        "arrival_rate_input": "registered per-class lambda, fixed through arrival window and drain",
        "terminal_interpretation": "fluid clearing at fixed mu with no arrivals beyond the next boundary; analytic surrogate rather than a second executed frame",
        "accounting": "unchanged paid candidate construction; completion reads, mean-queue and terminal second-moment evaluations counted separately for both weights",
        "reference": "v4 at K120, never K120 of a cost selector",
        "stopping_rule": "one fixed 144-run development roster; no coefficient fitting or post-result budget/seed selection",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    original = {(r["load"], r["seed"], r["budget"], r["policy"]): r for r in development["runs"]}
    runs = []
    for load, certificate in zip(base["loads"], certificates, strict=True):
        for seed in base["seeds"]:
            tape = poisson_arrivals(certificate["arrival_rates"], base["arrival_horizon_s"], seed)
            for budget in base["budgets"]:
                for policy in POLICIES:
                    if policy in ("v4_support", "holding_cap_1pct"):
                        recorder = PolicyRecorder(policy, sum(certificate["arrival_rates"]), max_relative_support_loss=0.01)
                    else:
                        recorder = LookaheadRecorder(certificate["arrival_rates"], service_rates,
                                                     1.0 if policy == "lookahead_cap_1pct" else 0.0)
                    metrics = simulate_frames(model, tape, base["arrival_horizon_s"], certificate, budget, selector=recorder)
                    decomposition = decompose_flow(model, tape, recorder.frames, base["arrival_horizon_s"])
                    accounting = recorder.accounting()
                    if abs(decomposition["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                        raise ValueError("lookahead flow decomposition differs from simulator")
                    if policy in ("v4_support", "holding_cap_1pct"):
                        old = original[(load, seed, budget, policy)]
                        if json.loads(json.dumps(metrics)) != old["metrics"] or json.loads(json.dumps(decomposition)) != old["decomposition"]:
                            raise ValueError(f"frozen control changed for {(load, seed, budget, policy)}")
                        for key, value in accounting.items():
                            if key not in ("selector_wall_s", "selector_mean_wall_ms") and value != old["selector_accounting"][key]:
                                raise ValueError(f"frozen control accounting changed: {key}")
                    runs.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                                 "metrics": metrics, "decomposition": decomposition, "selector_accounting": accounting})
            print(json.dumps({"load": load, "seed": seed, "runs_completed": len(runs)}), flush=True)
    _check_frozen_sources()
    paired, summary = summarize_runs(runs, base["loads"], base["budgets"])
    indexed = {(r["load"], r["seed"], r["budget"], r["policy"]): r for r in runs}
    for row in paired:
        if row["policy"] == "lookahead_cap_1pct":
            for control in ("holding_cap_1pct", "frame_area_cap_1pct"):
                before = indexed[(row["load"], row["seed"], row["budget"], control)]["metrics"]["mean_flow_s"]
                row[f"flow_ratio_over_{control}"] = row["waiting_rule"]["mean_flow_s"] / before
    for group in summary:
        if group["policy"] == "lookahead_cap_1pct":
            members = [r for r in paired if (r["load"], r["budget"], r["policy"]) == (group["load"], group["budget"], group["policy"])]
            for control in ("holding_cap_1pct", "frame_area_cap_1pct"):
                group[f"mean_paired_flow_ratio_over_{control}"] = mean(r[f"flow_ratio_over_{control}"] for r in members)
    new_runs = [r for r in runs if r["policy"] in ("frame_area_cap_1pct", "lookahead_cap_1pct")]
    checks = {
        "frozen_sources_and_72_control_runs_unchanged": True,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "candidate_construction_within_budget": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= r["budget"] for r in runs),
        "seed_and_support_floors_preserved": all(r["selector_accounting"]["minimum_seed_support_margin"] >= 0 and r["selector_accounting"]["minimum_support_floor_margin"] >= 0 for r in new_runs),
        "relative_loss_within_one_percent": all(r["selector_accounting"]["maximum_relative_support_loss"] <= 0.01 + 1e-12 for r in new_runs),
        "chosen_objective_no_worse_than_incumbent": all(r["selector_accounting"]["maximum_objective_excess_over_support_incumbent"] <= 1e-8 for r in new_runs),
        "original_drift_bound_holds": all(r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "queue_recursion_holds": all(r["metrics"]["maximum_queue_recursion_violation"] == 0 for r in runs),
        "k120_support_search_exact": all(r["selector_accounting"]["support_search_exact_frames"] == r["metrics"]["frame_count"] for r in new_runs if r["budget"] == 120),
        "prediction_and_terminal_work_accounted": all(r["selector_accounting"]["next_queue_coordinate_evaluations"] == r["selector_accounting"]["terminal_value_coordinate_evaluations"] == len(model.workloads) * r["selector_accounting"]["cost_candidate_evaluations"] for r in new_runs),
    }
    report = {
        "experiment": "aligned_hardware_lookahead_development_v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "model": model.snapshot(), "certificates": certificates,
        "runs": runs, "paired_comparisons": paired, "summary_by_load_and_budget": summary,
        "limitations": [
            "The terminal value is a fluid clearing proxy, not an exact value of future synchronous batches or a long-run flow guarantee.",
            "The stationary arrival model remains fixed through drain, and terminal clearing assumes no further arrivals after the next boundary.",
            "Both area-only and lookahead controls compute terminal diagnostics, so their recorded computational work includes these evaluations.",
            "Support constraints preserve the existing drift guarantee; they do not certify the accuracy of the terminal model.",
            "Only the paid candidate pool is optimized. This reused development roster provides no new independent confirmation.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_experiment(args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "output": str(args.output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
