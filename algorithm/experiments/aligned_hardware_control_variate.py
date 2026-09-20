"""Test fixed analytic control variates on the same paid Poisson rollout."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import random
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.control_variate_rollout_frame_candidates import select_control_variate_rollout_frame
from algorithm.experiments.aligned_hardware_poisson_rollout import PoissonRolloutRecorder
from algorithm.experiments.aligned_hardware_control_variate_diagnostic import diagnose, DIAGNOSTIC_QUEUES, DIAGNOSTIC_SAMPLES
from algorithm.experiments.aligned_hardware_arrival_rollout import ARRIVAL_COUNTERS
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_batch_rollout import BatchRolloutRecorder, COUNTERS
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import LookaheadRecorder, _check_frozen_sources, terminal_service_rates
from algorithm.experiments.aligned_hardware_lookahead_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, PolicyRecorder, summarize_runs


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_control_variate_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_poisson_rollout_development_20260910.json"
CONTROLS = ("v4_support", "lookahead_cap_1pct", "poisson_rollout_cap_1pct")
NEW_POLICY = "control_variate_rollout_cap_1pct"
POLICIES = (*CONTROLS, NEW_POLICY)
PLANNING_SEEDS = (0, 1, 2)


class ControlVariateRolloutRecorder(BatchRolloutRecorder):
    def __init__(self, arrival_rates, service_rates, planning_seed):
        super().__init__(arrival_rates, service_rates)
        started = perf_counter()
        self.arrival_rngs = tuple(random.Random(f"aligned-planning:{planning_seed}:{coordinate}")
                                 for coordinate in range(len(arrival_rates)))
        self.rng_setup_wall_s = perf_counter() - started
        self.initial_attempts = self.recycled_attempts = self.reclaimed_frames = self.resumed_frames = 0
        self.initial_cost_evaluations = self.recycled_cost_evaluations = 0
        self.maximum_stage_identity_error = 0
        self.maximum_path_population_error = 0
        self.maximum_random_draw_identity_error = 0
        self.maximum_cv_cost_identity_error = 0
        self.counts.update(dict.fromkeys((*ARRIVAL_COUNTERS, "rollout_random_draws", "cv_terminal_value_coordinate_evaluations"), 0))

    def __call__(self, options, queues, lanes, budget):
        start = perf_counter()
        selected = select_control_variate_rollout_frame(options, queues, lanes, budget,
                                                 arrival_rates=self.arrival_rates,
                                                 terminal_service_rates=self.service_rates, arrival_rngs=self.arrival_rngs)
        self.selector_wall_s += perf_counter() - start
        for key in (*COUNTERS, *ARRIVAL_COUNTERS, "rollout_random_draws", "cv_terminal_value_coordinate_evaluations"):
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
            self.maximum_path_population_error = max(self.maximum_path_population_error,
                                                    abs(selected.rollout_population_jobs - sum(queues) - selected.rollout_arrival_event_generations))
        expected_draws = selected.rollout_arrival_event_generations + int(selected.rollout_used) * sum(rate > 0 for rate in self.arrival_rates)
        self.maximum_random_draw_identity_error = max(self.maximum_random_draw_identity_error,
                                                      abs(selected.rollout_random_draws - expected_draws))
        if selected.rollout_used:
            expected_margin = (selected.raw_rollout_margin
                - (selected.analytic_challenger_sampled_control_value - selected.incumbent_sampled_control_value)
                + selected.root_lookahead_cost - selected.support_incumbent_lookahead_cost)
            self.maximum_cv_cost_identity_error = max(self.maximum_cv_cost_identity_error,
                                                      abs(selected.corrected_rollout_margin - expected_margin))
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
                "maximum_stage_identity_error": self.maximum_stage_identity_error,
                "maximum_path_population_error": self.maximum_path_population_error,
                "maximum_random_draw_identity_error": self.maximum_random_draw_identity_error,
                "maximum_cv_cost_identity_error": self.maximum_cv_cost_identity_error,
                "rollout_rng_initializations": len(self.arrival_rngs),
                "rng_setup_wall_s": self.rng_setup_wall_s}


def _run_seed(task):
    model, certificate, load, seed, base, mu = task
    tape = poisson_arrivals(certificate["arrival_rates"], base["arrival_horizon_s"], seed)
    rows = []
    roster = ([(policy, None) for policy in CONTROLS[:2]]
              + [(policy, planning_seed) for policy in (CONTROLS[-1], NEW_POLICY) for planning_seed in PLANNING_SEEDS])
    for budget in base["budgets"]:
        for policy, planning_seed in roster:
            rates = certificate["arrival_rates"]
            if policy == "v4_support":
                recorder = PolicyRecorder(policy, sum(rates), max_relative_support_loss=0.01)
            elif policy == "lookahead_cap_1pct":
                recorder = LookaheadRecorder(rates, mu, 1.0)
            elif policy == "poisson_rollout_cap_1pct":
                recorder = PoissonRolloutRecorder(rates, mu, planning_seed)
            else:
                recorder = ControlVariateRolloutRecorder(rates, mu, planning_seed)
            metrics = simulate_frames(model, tape, base["arrival_horizon_s"], certificate, budget, selector=recorder)
            parts = decompose_flow(model, tape, recorder.frames, base["arrival_horizon_s"])
            if abs(parts["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                raise ValueError("control variate trace differs from executed flow")
            rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                         "planning_seed": planning_seed, "metrics": metrics, "decomposition": parts,
                         "selector_accounting": recorder.accounting()})
    return rows


def _known_counterexample(model, certificate, mu, cases):
    comparisons = []
    for case in cases:
        planning_seed = case["planning_seed"]
        recorder = ControlVariateRolloutRecorder(certificate["arrival_rates"], mu, planning_seed)
        selected = recorder(model.options, case["queue"], model.lanes, 26)
        old = case["selection"]
        if (selected.incumbent_raw_rollout_cost != old["incumbent_rollout_cost"]
                or selected.analytic_challenger_raw_rollout_cost != old["analytic_challenger_rollout_cost"]
                or selected.rollout_random_draws != old["rollout_random_draws"]):
            raise ValueError("known-state control variate changed the frozen raw sample")
        comparisons.append({"queue": case["queue"], "planning_seed": planning_seed,
                            "selection": asdict(selected), "previous_poisson_selection": old,
                            "retains_previous_lookahead_action": not selected.rollout_vetoed,
                            "previous_realized_delta_job_s": case["previous_realized_delta_job_s"]})
    return {"cases": comparisons, "role": "same fresh planning samples as the prior static diagnostic; not full-run stream positions or fresh realized interventions",
            "diagnostic_candidate_attempts": sum(c["selection"]["attempts"] for c in comparisons),
            "diagnostic_random_draws": sum(c["selection"]["rollout_random_draws"] for c in comparisons),
            "diagnostic_cv_coordinate_evaluations": sum(c["selection"]["cv_terminal_value_coordinate_evaluations"] for c in comparisons),
            "diagnostic_rng_initializations": len(comparisons) * len(mu)}


def _assert_control_reproduced(row, old):
    for key in ("metrics", "decomposition"):
        if json.loads(json.dumps(row[key])) != old[key]:
            raise ValueError(f"frozen control {key} changed at {(row['load'], row['seed'], row['budget'], row['policy'], row['planning_seed'])}")
    # RNG setup joins selector timing as measured runtime, not a deterministic counter.
    timing = ("selector_wall_s", "selector_mean_wall_ms", "rng_setup_wall_s")
    new_counts = {k: v for k, v in row["selector_accounting"].items() if k not in timing}
    old_counts = {k: v for k, v in old["selector_accounting"].items() if k not in timing}
    if new_counts != old_counts:
        raise ValueError("frozen control work accounting changed")


def summarize_control_variate_runs(runs, loads, budgets):
    # The frozen pairing helper indexes only v4 denominators when pairing.
    # Restore planning identifiers before doing our own repeated-run summary.
    paired, old_summary = summarize_runs(runs, loads, budgets)
    controls = {(r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"]): r
                for r in runs if r["policy"] in CONTROLS}
    for row, pair in zip((r for r in runs if r["policy"] != "v4_support"), paired, strict=True):
        pair["planning_seed"] = row["planning_seed"]
        if row["policy"] == NEW_POLICY:
            key = (row["load"], row["seed"], row["budget"])
            for control in CONTROLS[1:]:
                pair[f"flow_ratio_over_{control}"] = row["metrics"]["mean_flow_s"] / controls[(*key, control, row["planning_seed"] if control == CONTROLS[-1] else None)]["metrics"]["mean_flow_s"]
    summary = [r for r in old_summary if r["policy"] in CONTROLS[:2]]
    for load in loads:
        for budget in budgets:
            group = [r for r in paired if (r["load"], r["budget"], r["policy"]) == (load, budget, NEW_POLICY)]
            seeds = sorted({r["seed"] for r in group})
            by_seed = []
            for seed in seeds:
                members = [r for r in group if r["seed"] == seed]
                if sorted(r["planning_seed"] for r in members) != list(PLANNING_SEEDS):
                    raise ValueError("each real arrival seed must retain every planning repetition exactly once")
                ratios = [r["waiting_over_v4_same_budget"]["mean_flow_s"] for r in members]
                by_seed.append({
                    "seed": seed, "planning_seeds": PLANNING_SEEDS,
                    "mean_flow_s": mean(r["waiting_rule"]["mean_flow_s"] for r in members),
                    "baseline_mean_flow_s": members[0]["baseline"]["mean_flow_s"],
                    "mean_flow_ratio_over_v4": mean(ratios),
                    "planning_flow_ratio_range_over_v4": [min(ratios), max(ratios)],
                    "mean_flow_ratio_over_v4_exact": mean(r["waiting_over_v4_exact"]["mean_flow_s"] for r in members),
                    "mean_p90_ratio_over_v4": mean(r["waiting_over_v4_same_budget"]["p90_flow_s"] for r in members),
                    **{f"mean_flow_ratio_over_{control}": mean(r[f"flow_ratio_over_{control}"] for r in members)
                       for control in CONTROLS[1:]},
                })
            seed_ratios = [r["mean_flow_ratio_over_v4"] for r in by_seed]
            raw_ratios = [r["waiting_over_v4_same_budget"]["mean_flow_s"] for r in group]
            by_planning_seed = []
            for planning_seed in PLANNING_SEEDS:
                members = [r for r in group if r["planning_seed"] == planning_seed]
                by_planning_seed.append({
                    "planning_seed": planning_seed,
                    "mean_paired_flow_ratio_over_v4": mean(r["waiting_over_v4_same_budget"]["mean_flow_s"] for r in members),
                    **{f"mean_paired_flow_ratio_over_{control}": mean(r[f"flow_ratio_over_{control}"] for r in members)
                       for control in CONTROLS[1:]},
                })
            raw_members = [r for r in runs if (r["load"], r["budget"], r["policy"]) == (load, budget, NEW_POLICY)]
            summary.append({
                "load": load, "budget": budget, "policy": NEW_POLICY, "seeds": seeds,
                "planning_seeds": PLANNING_SEEDS, "run_count": len(group),
                "by_arrival_seed": by_seed, "by_planning_seed": by_planning_seed,
                "baseline_mean_flow_s": mean(r["baseline_mean_flow_s"] for r in by_seed),
                "waiting_rule_mean_flow_s": mean(r["mean_flow_s"] for r in by_seed),
                "mean_paired_flow_ratio_over_v4": mean(seed_ratios),
                "paired_seed_mean_flow_ratio_range": [min(seed_ratios), max(seed_ratios)],
                "raw_repetition_flow_ratio_range": [min(raw_ratios), max(raw_ratios)],
                "improved_arrival_seed_count": sum(ratio < 1.0 for ratio in seed_ratios),
                "mean_paired_flow_ratio_over_v4_exact": mean(r["mean_flow_ratio_over_v4_exact"] for r in by_seed),
                "mean_paired_p90_ratio_over_v4": mean(r["mean_p90_ratio_over_v4"] for r in by_seed),
                "mean_component_deltas_s": {key: mean(r["component_deltas_s"][key] for r in group)
                                            for key in group[0]["component_deltas_s"]},
                "mean_selector_time_ratio": mean(r["selector_time_ratio"] for r in group),
                **{f"mean_paired_flow_ratio_over_{control}": mean(r[f"mean_flow_ratio_over_{control}"] for r in by_seed)
                   for control in CONTROLS[1:]},
                **{f"mean_{field}": mean(r["selector_accounting"][field] for r in raw_members)
                   for field in ("recycled_candidate_attempts", "rollout_frames", "rollout_vetoed_frames",
                                 "rollout_random_draws", "rng_setup_wall_s", "cv_terminal_value_coordinate_evaluations", *ARRIVAL_COUNTERS)},
            })
    return paired, summary


def build_experiment(output=DEFAULT_OUTPUT):
    _check_frozen_sources()
    snapshot = json.loads((ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != snapshot:
        raise ValueError("confirmed source must remain frozen")
    root = Path(__file__).resolve().parents[2]
    fixed_sources = {name: (root / name).read_text() for name in (
        "algorithm/batch_rollout_frame_candidates.py", "algorithm/experiments/aligned_hardware_batch_rollout.py",
        "algorithm/recycled_rollout_frame_candidates.py", "algorithm/experiments/aligned_hardware_recycled_rollout.py",
        "algorithm/arrival_rollout_frame_candidates.py", "algorithm/arrival_rollout_terminal.py",
        "algorithm/experiments/aligned_hardware_arrival_rollout.py", "algorithm/poisson_rollout_frame_candidates.py",
        "algorithm/poisson_rollout_terminal.py", "algorithm/experiments/aligned_hardware_poisson_rollout.py")}
    development = json.loads(DEVELOPMENT.read_text())
    base = development["protocol"]["base_protocol"]
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in base["loads"]]
    mu = terminal_service_rates(model, certificates[0])
    if (json.loads(json.dumps(model.snapshot())) != development["model"]
            or json.loads(json.dumps(certificates)) != development["certificates"]
            or list(mu) != development["protocol"]["terminal_service_rates"]):
        raise ValueError("control variate changed the registered model or rates")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "base_protocol": base,
        "role": "development only; no confirmation seeds", "policies": POLICIES,
        "main_policy": NEW_POLICY, "main_budget": 26, "terminal_service_rates": mu,
        "planning_seeds": PLANNING_SEEDS, "control_variate_coefficient": 1.0,
        "objective": "Xcv=X-Y+Janalytic; Y=sampled first-frame area+Vfluid(sampled next queue); implement Janalytic+successor_area+tail_area-Vfluid(sampled next queue)",
        "expectation": "two candidates and first-frame durations fixed before sampling; E[Y|pre-sample history]=Janalytic, preserving each candidate's hybrid-model expectation; selected argmin need not be unbiased",
        "costs": "same root plus four successor attempts and same random draws; add two*d fluid-coordinate evaluations, separately reported and included in terminal-value totals",
        "unchanged_sampling": development["protocol"],
        "clipping": "no nonnegative truncation or post-sample coefficient fitting",
        "controls": "v4 and analytic lookahead each 36; frozen Poisson policy 108 with all three planning repetitions; reproduce all 180 controls",
        "pairing": "match Poisson control on load, real seed, K AND planning_seed; ratio per matched run then planning mean within real seed then mean across original real seeds",
        "known_counterexample": "all three prior fresh planning samples; raw costs and draws must reproduce; corrected ordering is descriptive, not a required success",
        "variance_diagnostic": {
            "queues": DIAGNOSTIC_QUEUES, "loads": base["loads"], "budget": 26, "samples_per_active_state": DIAGNOSTIC_SAMPLES,
            "rng": "aligned-cv-diagnostic:{sample_index}:{coordinate}, independent of full-run planning streams; same sample ids crossed across Q/load",
            "inactive": "probe once and retain inactive states without repeating identical non-rollout calls",
            "measure": "fixed challenger-minus-incumbent raw and corrected margins, within-state sample variance ratio and paired mean-shift MC error; no cross-state variance pooling or performance gate",
            "cost": "all diagnostic searches, draws and coordinate work reported separately; not free work inside online K",
        },
        "stopping_rule": "288 full runs once plus registered diagnostics; no coefficients, sample ids, state roster, horizons or budget changes after outcomes",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    output.with_name(output.stem + "_fixed_sources.json").write_text(json.dumps(fixed_sources, indent=2) + "\n")
    known = _known_counterexample(model, certificates[0], mu, development["known_counterexample"]["cases"])
    original = {(r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"]): r for r in development["runs"]}
    tasks = [(model, cert, load, seed, base, mu)
             for load, cert in zip(base["loads"], certificates, strict=True) for seed in base["seeds"]]
    runs = []
    with ProcessPoolExecutor(max_workers=protocol["execution_workers"]) as pool:
        for rows in pool.map(_run_seed, tasks):
            for row in rows:
                if row["policy"] != POLICIES[-1]:
                    _assert_control_reproduced(row, original[(row["load"], row["seed"], row["budget"], row["policy"], row["planning_seed"])])
            runs.extend(rows)
            print(json.dumps({"load": rows[0]["load"], "seed": rows[0]["seed"], "runs_completed": len(runs)}), flush=True)
    variance_diagnostic = diagnose(model, certificates, base["loads"], mu)
    paired, summary = summarize_control_variate_runs(runs, base["loads"], base["budgets"])
    new = [r for r in runs if r["policy"] == POLICIES[-1]]
    checks = {
        "180_frozen_control_runs_reproduced": True,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "total_attempts_within_budget": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= r["budget"] for r in runs),
        "seed_and_current_support_floors_hold": all(r["selector_accounting"]["minimum_seed_support_margin"] >= 0 and r["selector_accounting"]["minimum_support_floor_margin"] >= 0 for r in new),
        "one_percent_current_incumbent_loss_cap_holds": all(r["selector_accounting"]["maximum_relative_support_loss"] <= .01+1e-12 for r in new),
        "stage_and_total_accounting_hold": all(r["selector_accounting"]["maximum_stage_identity_error"] == r["selector_accounting"]["maximum_budget_identity_error"] == 0 and r["selector_accounting"]["root_attempts"] + r["selector_accounting"]["rollout_attempts"] == r["metrics"]["candidate_attempts"] for r in new),
        "four_successor_attempts_per_rollout": all(r["selector_accounting"]["rollout_attempts"] == 4*r["selector_accounting"]["rollout_frames"] for r in new),
        "path_population_accounted": all(r["selector_accounting"]["maximum_path_population_error"] == 0 for r in new),
        "all_random_draws_accounted": all(r["selector_accounting"]["maximum_random_draw_identity_error"] == 0 for r in new),
        "complete_crossed_roster": len(runs) == 288 and len({(r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"]) for r in runs}) == 288,
        "cv_cost_identity_holds": all(r["selector_accounting"]["maximum_cv_cost_identity_error"] <= 1e-8 for r in new),
        "cv_coordinates_fully_charged": all(r["selector_accounting"]["cv_terminal_value_coordinate_evaluations"] == 2*len(mu)*r["selector_accounting"]["rollout_frames"] for r in new),
        "diagnostic_identity_holds": variance_diagnostic["status"] == "PASS",
        "rollout_objective_no_worse_than_incumbent": all(r["selector_accounting"]["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8 for r in new),
        "original_drift_bound_holds": all(r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "queue_recursion_holds": all(r["metrics"]["maximum_queue_recursion_violation"] == 0 for r in runs),
        "v4_k120_reference_exact": all(r["metrics"]["exact_selection_frame_count"] == r["metrics"]["frame_count"] for r in runs if r["budget"] == 120 and r["policy"] == "v4_support"),
        "frozen_sources_unchanged": _source_texts() == snapshot and all((root/name).read_text() == text for name, text in fixed_sources.items()),
    }
    report = {
        "experiment": "aligned_hardware_control_variate_development_v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "model": model.snapshot(), "certificates": certificates,
        "runs": runs, "paired_comparisons": paired, "summary_by_load_and_budget": summary,
        "known_counterexample": known, "variance_diagnostic": variance_diagnostic,
        "limitations": [
            "The fixed coefficient preserves each pre-sampling candidate expectation, but does not guarantee variance reduction or an unbiased selected minimum.",
            "Only two successor frames per contender are discrete; the remaining continuation is a fixed-rate fluid approximation and only the original analytic challenger can compete.",
            "Planning repetitions are crossed with the same three reused arrival seeds, not additional independent environments; shared-worker timing is descriptive.",
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
