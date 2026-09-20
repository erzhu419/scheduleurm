"""Develop a two-frame known-backlog terminal estimate within total budget K."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.batch_rollout_frame_candidates import current_frame_budget, select_batch_rollout_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_backlog_gate import _assert_control_reproduced
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import LookaheadRecorder, _check_frozen_sources, terminal_service_rates
from algorithm.experiments.aligned_hardware_lookahead_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, PolicyRecorder, summarize_runs


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_batch_rollout_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_lookahead_development_20260910.json"
POLICIES = ("v4_support", "lookahead_cap_1pct", "allocated_lookahead_cap_1pct", "batch_rollout_cap_1pct")
COUNTERS = (
    "root_attempts", "rollout_attempts", "cost_candidate_evaluations", "rollout_cost_candidate_evaluations",
    "completion_offset_reads", "physical_action_completion_offset_reads", "rollout_completion_offset_reads",
    "next_queue_coordinate_evaluations", "terminal_value_coordinate_evaluations",
    "rollout_terminal_value_coordinate_evaluations", "rollout_class_evaluations",
    "rollout_profile_probes", "rollout_setup_requirement_reads",
)


class AllocatedLookaheadRecorder(LookaheadRecorder):
    def __call__(self, options, queues, lanes, budget):
        return super().__call__(options, queues, lanes, current_frame_budget(len(options), budget))


class BatchRolloutRecorder:
    def __init__(self, arrival_rates, service_rates):
        self.arrival_rates, self.service_rates = arrival_rates, service_rates
        self.now = self.selector_wall_s = 0.0
        self.frames = []
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.used = self.vetoed = self.search_exact = self.lower_support = 0
        self.minimum_seed_margin = self.minimum_floor_margin = float("inf")
        self.maximum_relative_loss = self.maximum_rollout_cost_excess = 0.0
        self.maximum_budget_identity_error = 0

    def __call__(self, options, queues, lanes, budget):
        start = perf_counter()
        selected = select_batch_rollout_frame(options, queues, lanes, budget,
                                               arrival_rates=self.arrival_rates,
                                               terminal_service_rates=self.service_rates)
        self.selector_wall_s += perf_counter() - start
        for key in COUNTERS:
            self.counts[key] += getattr(selected, key)
        self.used += int(selected.rollout_used)
        self.vetoed += int(selected.rollout_vetoed)
        self.search_exact += int(selected.support_search_exact)
        self.lower_support += int(selected.score < selected.support_incumbent_score - 1e-10)
        self.minimum_seed_margin = min(self.minimum_seed_margin, selected.score - selected.seed_support)
        self.minimum_floor_margin = min(self.minimum_floor_margin, selected.score - selected.support_floor)
        self.maximum_relative_loss = max(self.maximum_relative_loss, selected.relative_support_loss)
        self.maximum_budget_identity_error = max(self.maximum_budget_identity_error,
                                                abs(selected.attempts - selected.root_attempts - selected.rollout_attempts))
        if selected.rollout_used:
            self.maximum_rollout_cost_excess = max(self.maximum_rollout_cost_excess,
                                                  selected.selected_rollout_cost - selected.incumbent_rollout_cost)
        self.frames.append({"start_s": self.now, "duration_s": selected.duration,
                            "queue_before": tuple(queues), "departures": selected.service,
                            "option_indices": selected.option_indices})
        self.now += selected.duration
        return selected

    def accounting(self):
        count = len(self.frames)
        return {
            **self.counts, "selector_wall_s": self.selector_wall_s,
            "selector_mean_wall_ms": 1000 * self.selector_wall_s / count,
            "mean_cost_candidate_evaluations": self.counts["cost_candidate_evaluations"] / count,
            "rollout_frames": self.used, "rollout_vetoed_frames": self.vetoed,
            "rollout_accepted_frames": self.used - self.vetoed,
            "rollout_frame_fraction": self.used / count,
            "support_search_exact_frames": self.search_exact,
            "lower_support_than_search_incumbent_frames": self.lower_support,
            "minimum_seed_support_margin": self.minimum_seed_margin,
            "minimum_support_floor_margin": self.minimum_floor_margin,
            "maximum_relative_support_loss": self.maximum_relative_loss,
            "maximum_rollout_cost_excess_over_incumbent": self.maximum_rollout_cost_excess,
            "maximum_budget_identity_error": self.maximum_budget_identity_error,
        }


def _run_seed(task):
    model, certificate, load, seed, base, mu = task
    tape = poisson_arrivals(certificate["arrival_rates"], base["arrival_horizon_s"], seed)
    rows = []
    for budget in base["budgets"]:
        for policy in POLICIES:
            rates = certificate["arrival_rates"]
            if policy == "v4_support":
                recorder = PolicyRecorder(policy, sum(rates), max_relative_support_loss=0.01)
            elif policy == "lookahead_cap_1pct":
                recorder = LookaheadRecorder(rates, mu, 1.0)
            elif policy == "allocated_lookahead_cap_1pct":
                recorder = AllocatedLookaheadRecorder(rates, mu, 1.0)
            else:
                recorder = BatchRolloutRecorder(rates, mu)
            metrics = simulate_frames(model, tape, base["arrival_horizon_s"], certificate, budget, selector=recorder)
            parts = decompose_flow(model, tape, recorder.frames, base["arrival_horizon_s"])
            if abs(parts["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                raise ValueError("batch rollout trace does not reconstruct executed flow")
            rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                         "metrics": metrics, "decomposition": parts, "selector_accounting": recorder.accounting()})
    return rows


def build_experiment(output=DEFAULT_OUTPUT):
    _check_frozen_sources()
    snapshot = json.loads((ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != snapshot:
        raise ValueError("confirmed model and selectors must remain frozen")
    development = json.loads(DEVELOPMENT.read_text())
    base = development["protocol"]["base_protocol"]
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in base["loads"]]
    mu = terminal_service_rates(model, certificates[0])
    if (json.loads(json.dumps(model.snapshot())) != development["model"]
            or json.loads(json.dumps(certificates)) != development["certificates"]
            or list(mu) != development["protocol"]["terminal_service_rates"]):
        raise ValueError("batch rollout must retain the frozen model and terminal reference")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "base_protocol": base,
        "role": "development only, original seeds 41/42/43",
        "policies": POLICIES, "main_policy": POLICIES[-1], "main_budget": 26,
        "terminal_service_rates": mu,
        "budget_split": {str(k): {"current": current_frame_budget(len(model.options), k),
                                  "reserved_successors": k - current_frame_budget(len(model.options), k)} for k in base["budgets"]},
        "allocation": "reserve four attempts if K covers current homogeneous seeds plus four; otherwise current K and return support incumbent",
        "contenders": "current paid support incumbent and old analytic J minimizer; identical physical actions skip rollout; no unpriced candidates compete",
        "support_floor": "max(current homogeneous seed support, 0.99*current paid incumbent), not the full-K incumbent",
        "terminal": "V2(r)=two no-new-arrival executable frame holding areas plus Vfluid(r_after2); r=Q-D",
        "objective": "Jbatch=A+V2(r)+E[Vfluid(r+Poisson(lambda*tau))]-Vfluid(r)",
        "successor": "choose nonempty class maximizing q_i/mu_i; choose largest fitting measured pure-class batch; maximal identical lane copies, remaining lanes idle",
        "ties": "class index; same batch shortest duration then option index; final objective then higher support, shorter duration and lexical indices",
        "successor_budget": "one directly constructed and verified full frame per step; both contenders receive exactly two steps, including paid idle when empty",
        "work_accounting": "total attempts=current+successor; catalogue/index probes, class scans, completion reads and fluid value coordinates also reported",
        "arrival_input": "same registered Poisson rates through window and drain; new-arrival interaction with discrete batches remains in the old fluid correction",
        "controls": "v4 and analytic lookahead at full K; analytic lookahead at the identical reduced current budget isolates search allocation",
        "reference": "v4 at K120 is the exact support reference; a rollout selector at K120 need not select the exact-support action",
        "stopping_rule": "one fixed 144-run development roster; no depth, priority or budget tuning after outcomes",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    original = {(r["load"], r["seed"], r["budget"], r["policy"]): r for r in development["runs"]}
    tasks = [(model, cert, load, seed, base, mu)
             for load, cert in zip(base["loads"], certificates, strict=True) for seed in base["seeds"]]
    runs = []
    with ProcessPoolExecutor(max_workers=protocol["execution_workers"]) as pool:
        for rows in pool.map(_run_seed, tasks):
            for row in rows:
                if row["policy"] in POLICIES[:2]:
                    _assert_control_reproduced(row, original[(row["load"], row["seed"], row["budget"], row["policy"])])
            runs.extend(rows)
            print(json.dumps({"load": rows[0]["load"], "seed": rows[0]["seed"], "runs_completed": len(runs)}), flush=True)
    paired, summary = summarize_runs(runs, base["loads"], base["budgets"])
    indexed = {(r["load"], r["seed"], r["budget"], r["policy"]): r for r in runs}
    for row in paired:
        if row["policy"] == POLICIES[-1]:
            key = (row["load"], row["seed"], row["budget"])
            selected = indexed[(*key, POLICIES[-1])]
            for control in POLICIES[1:3]:
                row[f"flow_ratio_over_{control}"] = selected["metrics"]["mean_flow_s"] / indexed[(*key, control)]["metrics"]["mean_flow_s"]
    for group in summary:
        if group["policy"] == POLICIES[-1]:
            key = (group["load"], group["budget"], group["policy"])
            members = [r for r in runs if (r["load"], r["budget"], r["policy"]) == key]
            comparisons = [r for r in paired if (r["load"], r["budget"], r["policy"]) == key]
            for control in POLICIES[1:3]:
                group[f"mean_paired_flow_ratio_over_{control}"] = mean(r[f"flow_ratio_over_{control}"] for r in comparisons)
            for field in ("rollout_frames", "rollout_vetoed_frames", "rollout_accepted_frames", "rollout_frame_fraction", "rollout_attempts"):
                group[f"mean_{field}"] = mean(r["selector_accounting"][field] for r in members)
    selected_runs = [r for r in runs if r["policy"] == POLICIES[-1]]
    checks = {
        "frozen_sources_and_72_controls_unchanged": _source_texts() == snapshot,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "total_candidate_construction_within_budget": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= r["budget"] for r in runs),
        "current_seed_and_support_floors_preserved": all(r["selector_accounting"]["minimum_seed_support_margin"] >= 0 and r["selector_accounting"]["minimum_support_floor_margin"] >= 0 for r in selected_runs),
        "relative_support_loss_within_one_percent_of_current_incumbent": all(r["selector_accounting"]["maximum_relative_support_loss"] <= 0.01 + 1e-12 for r in selected_runs),
        "total_work_identity": all(r["selector_accounting"]["maximum_budget_identity_error"] == 0 and r["selector_accounting"]["root_attempts"] + r["selector_accounting"]["rollout_attempts"] == r["metrics"]["candidate_attempts"] for r in selected_runs),
        "four_attempts_and_two_costs_per_rollout": all(r["selector_accounting"]["rollout_attempts"] == 4*r["selector_accounting"]["rollout_frames"] and r["selector_accounting"]["rollout_cost_candidate_evaluations"] == 2*r["selector_accounting"]["rollout_frames"] for r in selected_runs),
        "rollout_objective_no_worse_than_incumbent": all(r["selector_accounting"]["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8 for r in selected_runs),
        "original_drift_bound_holds": all(r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "queue_recursion_holds": all(r["metrics"]["maximum_queue_recursion_violation"] == 0 for r in runs),
        "v4_k120_reference_exact": all(r["metrics"]["exact_selection_frame_count"] == r["metrics"]["frame_count"] for r in runs if r["budget"] == 120 and r["policy"] == "v4_support"),
    }
    report = {
        "experiment": "aligned_hardware_two_frame_batch_rollout_development_v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "model": model.snapshot(), "certificates": certificates,
        "runs": runs, "paired_comparisons": paired, "summary_by_load_and_budget": summary,
        "limitations": [
            "Two homogeneous successor frames plus a fluid tail approximate clearing the known residual; arrivals do not affect these successor choices.",
            "Only two current contenders are compared; the 1% cap is against the reduced current pool, not the full-K search.",
            "This reused development roster provides no independent confirmation. Wall times include shared three-worker execution.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_experiment(args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
