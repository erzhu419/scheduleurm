"""Test a seed-constrained frame holding-cost rule on the development roster.

The v4 candidate generator, model, simulator and confirmation remain frozen.
The new selector ranks its paid candidates using a stationary arrival-rate
proxy, including during the finite experiment's drain phase.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.waiting_cost_frame_candidates import select_waiting_cost_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_mechanism import _assert_frozen_sources
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model


ARTIFACTS = Path(__file__).resolve().parents[2] / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_waiting_cost_development_20260909.json"
POLICIES = ("v4_support", "seed_guarded_holding_v1")
COMPONENTS = ("mean_deferred_visible_wait_s", "mean_arrival_barrier_wait_s", "mean_dispatch_to_completion_s")


class PolicyRecorder:
    def __init__(self, policy, arrival_rate, *, max_relative_support_loss=1.0):
        self.policy, self.arrival_rate = policy, arrival_rate
        self.max_relative_support_loss = max_relative_support_loss
        self.now = 0.0
        self.frames = []
        self.selector_wall_s = 0.0
        self.cost_evaluations = self.offset_reads = self.eligible_candidates = 0
        self.lower_cost_frames = self.lower_support_frames = self.support_search_exact_frames = 0
        self.minimum_seed_margin = float("inf")
        self.maximum_cost_excess = 0.0
        self.holding_cost_reduction_sum = 0.0
        self.minimum_support_floor_margin = float("inf")
        self.maximum_relative_support_loss = 0.0
        self.support_cap_excluded_candidates = 0

    def __call__(self, options, queues, lanes, budget):
        start = perf_counter()
        if self.policy == "v4_support":
            selected = select_budgeted_frame(options, queues, lanes, budget)
        else:
            selected = select_waiting_cost_frame(
                options, queues, lanes, budget, arrival_rate=self.arrival_rate,
                max_relative_support_loss=self.max_relative_support_loss,
            )
        self.selector_wall_s += perf_counter() - start
        if self.policy != "v4_support":
            self.cost_evaluations += selected.cost_candidate_evaluations
            self.offset_reads += selected.completion_offset_reads
            self.eligible_candidates += selected.eligible_candidate_count
            self.support_search_exact_frames += int(selected.support_search_exact)
            self.lower_support_frames += int(selected.score < selected.support_incumbent_score - 1e-10)
            saving = selected.support_incumbent_holding_rate_cost - selected.holding_rate_cost
            self.lower_cost_frames += int(saving > 1e-10)
            self.holding_cost_reduction_sum += saving
            self.minimum_seed_margin = min(self.minimum_seed_margin, selected.score - selected.seed_support)
            self.maximum_cost_excess = max(self.maximum_cost_excess, -saving)
            self.minimum_support_floor_margin = min(self.minimum_support_floor_margin, selected.score - selected.support_floor)
            self.maximum_relative_support_loss = max(self.maximum_relative_support_loss, selected.relative_support_loss)
            self.support_cap_excluded_candidates += selected.support_cap_excluded_candidate_count
        self.frames.append({
            "start_s": self.now, "duration_s": selected.duration,
            "queue_before": tuple(queues), "departures": selected.service,
            "option_indices": selected.option_indices,
        })
        self.now += selected.duration
        return selected

    def accounting(self):
        count = len(self.frames)
        return {
            "selector_wall_s": self.selector_wall_s,
            "selector_mean_wall_ms": 1000.0 * self.selector_wall_s / count,
            "cost_candidate_evaluations": self.cost_evaluations,
            "completion_offset_reads": self.offset_reads,
            "mean_cost_candidate_evaluations": self.cost_evaluations / count,
            "eligible_candidates": self.eligible_candidates,
            "lower_holding_cost_frames": self.lower_cost_frames,
            "lower_support_than_search_incumbent_frames": self.lower_support_frames,
            "support_search_exact_frames": self.support_search_exact_frames if self.policy != "v4_support" else None,
            "minimum_seed_support_margin": self.minimum_seed_margin if self.policy != "v4_support" else None,
            "maximum_holding_cost_excess_over_support_incumbent": self.maximum_cost_excess,
            "mean_selected_holding_cost_reduction": self.holding_cost_reduction_sum / count,
            "minimum_support_floor_margin": self.minimum_support_floor_margin if self.policy != "v4_support" else None,
            "maximum_relative_support_loss": self.maximum_relative_support_loss,
            "support_cap_excluded_candidates": self.support_cap_excluded_candidates,
        }


def _run_metrics(row):
    return {key: row["metrics"][key] for key in ("mean_flow_s", "p90_flow_s", "makespan_s", "mean_candidate_attempts")}


def _ratio(numerator, denominator):
    return {key: numerator[key] / denominator[key] if denominator[key] else None for key in numerator}


def summarize_runs(runs, loads, budgets):
    indexed = {(row["load"], row["seed"], row["budget"], row["policy"]): row for row in runs}
    paired = []
    for row in runs:
        if row["policy"] == "v4_support":
            continue
        load, seed, budget = row["load"], row["seed"], row["budget"]
        before = indexed[(load, seed, budget, "v4_support")]
        exact = indexed[(load, seed, 120, "v4_support")]
        new_metrics, old_metrics = _run_metrics(row), _run_metrics(before)
        paired.append({
            "load": load, "seed": seed, "budget": budget, "policy": row["policy"],
            "job_count": row["metrics"]["job_count"],
            "baseline": old_metrics, "waiting_rule": new_metrics,
            "waiting_over_v4_same_budget": _ratio(new_metrics, old_metrics),
            "waiting_over_v4_exact": _ratio(new_metrics, _run_metrics(exact)),
            "component_deltas_s": {key: row["decomposition"][key] - before["decomposition"][key] for key in COMPONENTS},
            "selector_time_ratio": row["selector_accounting"]["selector_mean_wall_ms"] / before["selector_accounting"]["selector_mean_wall_ms"],
        })
    summary = []
    policies = tuple(dict.fromkeys(row["policy"] for row in paired))
    for load, budget, policy in product(loads, budgets, policies):
        group = [row for row in paired if (row["load"], row["budget"], row["policy"]) == (load, budget, policy)]
        ratios = [row["waiting_over_v4_same_budget"]["mean_flow_s"] for row in group]
        summary.append({
            "load": load, "budget": budget, "policy": policy, "seeds": [row["seed"] for row in group],
            "baseline_mean_flow_s": mean(row["baseline"]["mean_flow_s"] for row in group),
            "waiting_rule_mean_flow_s": mean(row["waiting_rule"]["mean_flow_s"] for row in group),
            "mean_paired_flow_ratio_over_v4": mean(ratios), "paired_flow_ratio_range": [min(ratios), max(ratios)],
            "improved_seed_count": sum(ratio < 1.0 for ratio in ratios),
            "mean_paired_flow_ratio_over_v4_exact": mean(row["waiting_over_v4_exact"]["mean_flow_s"] for row in group),
            "mean_paired_p90_ratio_over_v4": mean(row["waiting_over_v4_same_budget"]["p90_flow_s"] for row in group),
            "mean_component_deltas_s": {key: mean(row["component_deltas_s"][key] for row in group) for key in COMPONENTS},
            "mean_selector_time_ratio": mean(row["selector_time_ratio"] for row in group),
        })
    return paired, summary


def build_experiment(output=DEFAULT_OUTPUT):
    _assert_frozen_sources()
    development = json.loads((ARTIFACTS / "aligned_hardware_frame_experiment_v4_20260909.json").read_text())
    base_protocol = development["protocol"]
    protocol = {
        "role": "development test of an explicit holding-cost selector",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_protocol": base_protocol, "policies": POLICIES, "main_budget": 26,
        "new_objective": "H(Q,a) = sum(Q)-sum(D(a)) + sum(completion_offsets(a))/tau(a) + Lambda*tau(a)/2",
        "eligibility": "Q dot D(a)/tau(a) >= maximum paid homogeneous-seed support",
        "candidate_generation": "same budgeted construction, order and max-support pruning as frozen v4",
        "cost_scope": "minimize H only over already paid eligible candidates; no global holding-cost guarantee",
        "arrival_rate_input": "Lambda=sum(registered class arrival rates); fixed throughout arrival window and drain",
        "drain_cost_interpretation": "stationary-arrival proxy, not actual conditional arrivals after the cutoff",
        "reference": "v4 support selector at K120; K120 of the new rule is not a support-exact reference",
        "stopping_rule": "run the full development roster once; no coefficient or budget selection from the results",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model = load_model()
    if json.loads(json.dumps(model.snapshot())) != development["model"]:
        raise ValueError("waiting-cost experiment changed the frozen model")
    loads, seeds, budgets = (base_protocol[key] for key in ("loads", "seeds", "budgets"))
    horizon = base_protocol["arrival_horizon_s"]
    runs, certificates = [], []
    for load in loads:
        certificate = capacity_certificate(model, load)
        certificates.append(certificate)
        for seed in seeds:
            tape = poisson_arrivals(certificate["arrival_rates"], horizon, seed)
            for budget in budgets:
                for policy in POLICIES:
                    recorder = PolicyRecorder(policy, sum(certificate["arrival_rates"]))
                    metrics = simulate_frames(model, tape, horizon, certificate, budget, selector=recorder)
                    decomposition = decompose_flow(model, tape, recorder.frames, horizon)
                    if abs(decomposition["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                        raise ValueError("flow decomposition disagrees with the frozen simulator")
                    if policy == "v4_support":
                        original = next(row for row in development["runs"] if (row["load"], row["seed"], row["budget"]) == (load, seed, budget))
                        observed = json.loads(json.dumps(metrics))
                        changed = [key for key, value in observed.items() if original[key] != value]
                        if changed:
                            raise ValueError(f"v4 baseline changed during the new-rule test: {changed}")
                    runs.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                                 "metrics": metrics, "decomposition": decomposition, "selector_accounting": recorder.accounting()})
            print(json.dumps({"load": load, "seed": seed, "runs_completed": len(runs)}), flush=True)
    _assert_frozen_sources()
    paired, summary = summarize_runs(runs, loads, budgets)
    holding_runs = [row for row in runs if row["policy"] != "v4_support"]
    checks = {
        "frozen_sources_and_v4_runs_unchanged": True,
        "all_jobs_complete": all(row["metrics"]["completed_jobs"] == row["metrics"]["job_count"] for row in runs),
        "candidate_construction_within_budget": all(row["metrics"]["maximum_candidate_attempts_per_frame"] <= row["budget"] for row in runs),
        "all_selected_actions_preserve_seed_support": all(row["selector_accounting"]["minimum_seed_support_margin"] >= 0 for row in holding_runs),
        "holding_cost_no_worse_than_paid_support_incumbent": all(row["selector_accounting"]["maximum_holding_cost_excess_over_support_incumbent"] <= 1e-10 for row in holding_runs),
        "original_conditional_drift_bound_holds": all(row["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for row in runs),
        "queue_recursion_holds": all(row["metrics"]["maximum_queue_recursion_violation"] == 0 for row in runs),
    }
    report = {
        "experiment": "aligned_hardware_waiting_cost_development_v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "model": model.snapshot(), "certificates": certificates,
        "runs": runs, "paired_comparisons": paired, "summary_by_load_and_budget": summary,
        "limitations": [
            "The holding objective is a one-frame proxy. A lower objective does not certify lower long-run mean flow.",
            "For homogeneous frames whose admitted jobs all finish at tau, H=sum(Q)+Lambda*tau/2; the departure count cancels, so this objective favors short frames subject to the seed floor.",
            "The frozen max-support pruning can omit candidates with better holding cost; only the paid eligible pool is optimized.",
            "The stationary Lambda remains fixed in the drain phase; it does not describe the actual zero arrivals there.",
            "The selected action may have less support than an exactly solved support search. Its gap and exact flag refer to that actual selection.",
            "Candidate attempts retain the original budget unit; holding-cost evaluations, offset reads and wall time are additional reported work.",
            "These reused development seeds and the deterministic cache model provide no new independent or physical-hardware confirmation.",
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
