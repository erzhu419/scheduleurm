"""Return unused successor reservations to the same current-frame search."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.recycled_rollout_frame_candidates import select_recycled_rollout_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_backlog_gate import _assert_control_reproduced
from algorithm.experiments.aligned_hardware_batch_rollout import BatchRolloutRecorder, COUNTERS
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import LookaheadRecorder, _check_frozen_sources, terminal_service_rates
from algorithm.experiments.aligned_hardware_lookahead_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, PolicyRecorder, summarize_runs


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_recycled_rollout_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_batch_rollout_development_20260910.json"
POLICIES = ("v4_support", "lookahead_cap_1pct", "batch_rollout_cap_1pct", "recycled_rollout_cap_1pct")


class RecycledRolloutRecorder(BatchRolloutRecorder):
    def __init__(self, arrival_rates, service_rates):
        super().__init__(arrival_rates, service_rates)
        self.initial_attempts = self.recycled_attempts = self.reclaimed_frames = self.resumed_frames = 0
        self.initial_cost_evaluations = self.recycled_cost_evaluations = 0
        self.maximum_stage_identity_error = 0

    def __call__(self, options, queues, lanes, budget):
        start = perf_counter()
        selected = select_recycled_rollout_frame(options, queues, lanes, budget,
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
        self.initial_attempts += selected.initial_root_attempts
        self.recycled_attempts += selected.recycled_candidate_attempts
        self.reclaimed_frames += int(selected.reclaimed_reserve)
        self.resumed_frames += int(selected.current_search_resumed)
        self.initial_cost_evaluations += selected.initial_cost_candidate_evaluations
        self.recycled_cost_evaluations += selected.recycled_cost_candidate_evaluations
        self.maximum_stage_identity_error = max(self.maximum_stage_identity_error,
                                               abs(selected.root_attempts - selected.initial_root_attempts - selected.recycled_candidate_attempts),
                                               abs(selected.cost_candidate_evaluations - selected.initial_cost_candidate_evaluations - selected.recycled_cost_candidate_evaluations))
        self.frames.append({"start_s": self.now, "duration_s": selected.duration,
                            "queue_before": tuple(queues), "departures": selected.service,
                            "option_indices": selected.option_indices})
        self.now += selected.duration
        return selected

    def accounting(self):
        return {**super().accounting(),
                "initial_root_attempts": self.initial_attempts,
                "recycled_candidate_attempts": self.recycled_attempts,
                "initial_cost_candidate_evaluations": self.initial_cost_evaluations,
                "recycled_cost_candidate_evaluations": self.recycled_cost_evaluations,
                "reclaimed_reserve_frames": self.reclaimed_frames,
                "current_search_resumed_frames": self.resumed_frames,
                "maximum_stage_identity_error": self.maximum_stage_identity_error}


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
            elif policy == "batch_rollout_cap_1pct":
                recorder = BatchRolloutRecorder(rates, mu)
            else:
                recorder = RecycledRolloutRecorder(rates, mu)
            metrics = simulate_frames(model, tape, base["arrival_horizon_s"], certificate, budget, selector=recorder)
            parts = decompose_flow(model, tape, recorder.frames, base["arrival_horizon_s"])
            if abs(parts["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                raise ValueError("recycled rollout trace differs from executed flow")
            rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                         "metrics": metrics, "decomposition": parts, "selector_accounting": recorder.accounting()})
    return rows


def _known_counterexample(model, certificate, mu):
    old = json.loads((ARTIFACTS / "aligned_hardware_batch_rollout_known_state_20260910.json").read_text())
    comparisons = []
    for case in old["cases"]:
        selected = select_recycled_rollout_frame(model.options, case["queue"], model.lanes, 26,
                                                 arrival_rates=certificate["arrival_rates"], terminal_service_rates=mu)
        normalized = json.loads(json.dumps(asdict(selected)))
        if any(normalized[key] != value for key, value in case["selection"].items()):
            raise ValueError("budget recovery changed the already active rollout counterexample")
        comparisons.append({"queue": case["queue"], "old_rollout_fields_unchanged": True,
                            "selection": normalized})
    return {"cases": comparisons, "role": "static diagnostic only; no new arrival tape or paired continuation",
            "diagnostic_candidate_attempts": sum(c["selection"]["attempts"] for c in comparisons)}


def build_experiment(output=DEFAULT_OUTPUT):
    _check_frozen_sources()
    snapshot = json.loads((ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != snapshot:
        raise ValueError("confirmed source must remain frozen")
    root = Path(__file__).resolve().parents[2]
    fixed_sources = {name: (root / name).read_text() for name in (
        "algorithm/batch_rollout_frame_candidates.py", "algorithm/experiments/aligned_hardware_batch_rollout.py")}
    development = json.loads(DEVELOPMENT.read_text())
    base = development["protocol"]["base_protocol"]
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in base["loads"]]
    mu = terminal_service_rates(model, certificates[0])
    if (json.loads(json.dumps(model.snapshot())) != development["model"]
            or json.loads(json.dumps(certificates)) != development["certificates"]
            or list(mu) != development["protocol"]["terminal_service_rates"]):
        raise ValueError("budget recovery changed the registered model or rates")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "base_protocol": base,
        "role": "development only; no confirmation seeds", "policies": POLICIES,
        "main_policy": POLICIES[-1], "main_budget": 26,
        "changed_rule": "when initial challenger and incumbent have the same physical action, release the four reserved attempts and continue the same search frontier to total K",
        "fallback": "after recovery choose the final support incumbent; newly found unrolled challengers cannot compete",
        "unchanged_rollout": development["protocol"],
        "search_state": "keep paid seeds, bounds, frontier, incumbent and candidate pool; never restart or repay earlier attempts",
        "pricing": "cache candidate costs by object identity; count actual evaluations, including initially eligible candidates later excluded by the final floor",
        "certificates": "upper bound, incumbent and 1% eligibility use the final current pool; total attempts=current+successors<=K",
        "natural_completion": "no extra attempts when the current search is already complete",
        "known_counterexample": "the previous active-rollout state (1,4,8) must preserve every old selection field; this step does not revise the terminal model",
        "reference": "v4 K120 remains the exact support reference",
        "stopping_rule": "all 144 cases once; no change to the terminal rule or budget policy after outcomes",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    output.with_name(output.stem + "_fixed_sources.json").write_text(json.dumps(fixed_sources, indent=2) + "\n")
    known = _known_counterexample(model, certificates[0], mu)
    original = {(r["load"], r["seed"], r["budget"], r["policy"]): r for r in development["runs"]}
    tasks = [(model, cert, load, seed, base, mu)
             for load, cert in zip(base["loads"], certificates, strict=True) for seed in base["seeds"]]
    runs = []
    with ProcessPoolExecutor(max_workers=protocol["execution_workers"]) as pool:
        for rows in pool.map(_run_seed, tasks):
            for row in rows:
                if row["policy"] != POLICIES[-1]:
                    _assert_control_reproduced(row, original[(row["load"], row["seed"], row["budget"], row["policy"])])
            runs.extend(rows)
            print(json.dumps({"load": rows[0]["load"], "seed": rows[0]["seed"], "runs_completed": len(runs)}), flush=True)
    paired, summary = summarize_runs(runs, base["loads"], base["budgets"])
    indexed = {(r["load"], r["seed"], r["budget"], r["policy"]): r for r in runs}
    for row in paired:
        if row["policy"] == POLICIES[-1]:
            key = (row["load"], row["seed"], row["budget"])
            for control in POLICIES[1:3]:
                row[f"flow_ratio_over_{control}"] = indexed[(*key, POLICIES[-1])]["metrics"]["mean_flow_s"] / indexed[(*key, control)]["metrics"]["mean_flow_s"]
    for group in summary:
        if group["policy"] == POLICIES[-1]:
            key = (group["load"], group["budget"], group["policy"])
            members = [r for r in runs if (r["load"], r["budget"], r["policy"]) == key]
            pairs = [r for r in paired if (r["load"], r["budget"], r["policy"]) == key]
            for control in POLICIES[1:3]:
                group[f"mean_paired_flow_ratio_over_{control}"] = mean(r[f"flow_ratio_over_{control}"] for r in pairs)
            for field in ("recycled_candidate_attempts", "current_search_resumed_frames", "rollout_frames", "rollout_vetoed_frames"):
                group[f"mean_{field}"] = mean(r["selector_accounting"][field] for r in members)
    new = [r for r in runs if r["policy"] == POLICIES[-1]]
    checks = {
        "108_frozen_control_runs_reproduced": True,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "total_attempts_within_budget": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= r["budget"] for r in runs),
        "seed_and_current_support_floors_hold": all(r["selector_accounting"]["minimum_seed_support_margin"] >= 0 and r["selector_accounting"]["minimum_support_floor_margin"] >= 0 for r in new),
        "one_percent_current_incumbent_loss_cap_holds": all(r["selector_accounting"]["maximum_relative_support_loss"] <= .01+1e-12 for r in new),
        "stage_and_total_accounting_hold": all(r["selector_accounting"]["maximum_stage_identity_error"] == r["selector_accounting"]["maximum_budget_identity_error"] == 0 and r["selector_accounting"]["root_attempts"] + r["selector_accounting"]["rollout_attempts"] == r["metrics"]["candidate_attempts"] for r in new),
        "four_successor_attempts_per_rollout": all(r["selector_accounting"]["rollout_attempts"] == 4*r["selector_accounting"]["rollout_frames"] for r in new),
        "rollout_objective_no_worse_than_incumbent": all(r["selector_accounting"]["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8 for r in new),
        "original_drift_bound_holds": all(r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "queue_recursion_holds": all(r["metrics"]["maximum_queue_recursion_violation"] == 0 for r in runs),
        "v4_k120_reference_exact": all(r["metrics"]["exact_selection_frame_count"] == r["metrics"]["frame_count"] for r in runs if r["budget"] == 120 and r["policy"] == "v4_support"),
        "frozen_sources_unchanged": _source_texts() == snapshot and all((root/name).read_text() == text for name, text in fixed_sources.items()),
    }
    report = {
        "experiment": "aligned_hardware_recycled_rollout_development_v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "model": model.snapshot(), "certificates": certificates,
        "runs": runs, "paired_comparisons": paired, "summary_by_load_and_budget": summary,
        "known_counterexample": known,
        "limitations": [
            "Budget recovery improves available current search, not the validity of the unchanged terminal proxy.",
            "The inherited two-frame known-backlog approximation and Poisson correction remain fixed; recovery selects support when a new challenger has not been rolled out.",
            "These are reused development seeds; comparisons and shared-worker wall times are descriptive rather than independent confirmation.",
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
