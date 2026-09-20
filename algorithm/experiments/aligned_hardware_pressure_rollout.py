"""Change only discrete successor priority in the frozen Poisson rollout."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.pressure_rollout_frame_candidates import select_pressure_rollout_frame
from algorithm.pressure_rollout_terminal import static_pressure_rates
from algorithm.batch_rollout_frame_candidates import _build_rollout_profiles
from algorithm.experiments.aligned_hardware_poisson_rollout import PoissonRolloutRecorder
from algorithm.experiments.aligned_hardware_control_variate import _assert_control_reproduced
from algorithm.experiments.aligned_hardware_arrival_rollout import ARRIVAL_COUNTERS
from algorithm.experiments.aligned_hardware_batch_rollout import COUNTERS
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import LookaheadRecorder, _check_frozen_sources, terminal_service_rates
from algorithm.experiments.aligned_hardware_lookahead_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, COMPONENTS, PolicyRecorder


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_pressure_rollout_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_control_variate_development_20260910.json"
CONTINUATION = ARTIFACTS / "aligned_hardware_control_variate_continuation_development_20260910.json"
CONTROLS = ("v4_support", "lookahead_cap_1pct", "poisson_rollout_cap_1pct")
NEW_POLICY = "pressure_rollout_cap_1pct"
PLANNING_SEEDS = (0, 1, 2)
PRESSURE_COUNTERS = ("rollout_pressure_profile_reads", "rollout_pressure_rate_evaluations",
                     "rollout_priority_coordinate_evaluations")


class PressureRolloutRecorder(PoissonRolloutRecorder):
    def __init__(self, arrival_rates, service_rates, planning_seed):
        super().__init__(arrival_rates, service_rates, planning_seed)
        self.counts.update(dict.fromkeys(PRESSURE_COUNTERS, 0))

    def __call__(self, options, queues, lanes, budget):
        start = perf_counter()
        selected = select_pressure_rollout_frame(options, queues, lanes, budget,
                                                 arrival_rates=self.arrival_rates,
                                                 terminal_service_rates=self.service_rates, arrival_rngs=self.arrival_rngs)
        self.selector_wall_s += perf_counter() - start
        for key in (*COUNTERS, *ARRIVAL_COUNTERS, "rollout_random_draws", *PRESSURE_COUNTERS):
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


def _run_seed(task):
    model, certificate, load, seed, base, mu = task
    rates = certificate["arrival_rates"]
    tape = poisson_arrivals(rates, base["arrival_horizon_s"], seed)
    rows = []
    roster = ([(policy, None) for policy in CONTROLS[:2]]
              + [(policy, planning_seed) for policy in (CONTROLS[-1], NEW_POLICY) for planning_seed in PLANNING_SEEDS])
    for budget in base["budgets"]:
        for policy, planning_seed in roster:
            if policy == "v4_support":
                recorder = PolicyRecorder(policy, sum(rates), max_relative_support_loss=.01)
            elif policy == "lookahead_cap_1pct":
                recorder = LookaheadRecorder(rates, mu, 1.)
            else:
                cls = PoissonRolloutRecorder if policy == CONTROLS[-1] else PressureRolloutRecorder
                recorder = cls(rates, mu, planning_seed)
            metrics = simulate_frames(model, tape, base["arrival_horizon_s"], certificate, budget, selector=recorder)
            parts = decompose_flow(model, tape, recorder.frames, base["arrival_horizon_s"])
            if abs(parts["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                raise ValueError("pressure trace does not reconstruct executed flow")
            rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                         "planning_seed": planning_seed, "metrics": metrics, "decomposition": parts,
                         "selector_accounting": recorder.accounting()})
    return rows


def summarize_pressure_runs(runs, loads, budgets):
    controls = {(r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"]): r
                for r in runs if r["policy"] in CONTROLS}
    paired = []
    for row in runs:
        if row["policy"] != NEW_POLICY:
            continue
        pair = {key: row[key] for key in ("load", "seed", "budget", "planning_seed")}
        pair["policy"] = NEW_POLICY
        pair["mean_flow_s"] = row["metrics"]["mean_flow_s"]
        for control in CONTROLS:
            key = (row["load"], row["seed"], row["budget"], control,
                   row["planning_seed"] if control == CONTROLS[-1] else None)
            old = controls[key]
            if row["metrics"]["job_count"] != old["metrics"]["job_count"]:
                raise ValueError("paired policies must complete the same population")
            pair[f"flow_ratio_over_{control}"] = pair["mean_flow_s"] / old["metrics"]["mean_flow_s"]
            pair[f"p90_ratio_over_{control}"] = row["metrics"]["p90_flow_s"] / old["metrics"]["p90_flow_s"]
        old = controls[(row["load"], row["seed"], row["budget"], CONTROLS[-1], row["planning_seed"])]
        pair["component_deltas_vs_poisson_s"] = {k: row["decomposition"][k] - old["decomposition"][k] for k in COMPONENTS}
        paired.append(pair)
    summary = []
    for load in loads:
        for budget in budgets:
            group = [r for r in paired if (r["load"], r["budget"]) == (load, budget)]
            ratio_fields = [f"{metric}_ratio_over_{control}" for metric in ("flow", "p90") for control in CONTROLS]
            by_seed = []
            for seed in sorted({r["seed"] for r in group}):
                members = [r for r in group if r["seed"] == seed]
                if sorted(r["planning_seed"] for r in members) != list(PLANNING_SEEDS):
                    raise ValueError("each real seed must retain every planning repetition exactly once")
                by_seed.append({"seed": seed, "mean_flow_s": mean(r["mean_flow_s"] for r in members),
                                **{f"mean_{k}": mean(r[k] for r in members) for k in ratio_fields}})
            raw = [r for r in runs if (r["load"], r["budget"], r["policy"]) == (load, budget, NEW_POLICY)]
            summary.append({
                "load": load, "budget": budget, "policy": NEW_POLICY, "run_count": len(group),
                "by_arrival_seed": by_seed,
                "by_planning_seed": [{"planning_seed": p, **{f"mean_{k}": mean(r[k] for r in group if r["planning_seed"] == p) for k in ratio_fields}}
                                     for p in PLANNING_SEEDS],
                "mean_flow_s": mean(r["mean_flow_s"] for r in by_seed),
                **{f"mean_paired_{k}": mean(r[f"mean_{k}"] for r in by_seed) for k in ratio_fields},
                **{f"improved_arrival_seed_count_vs_{control}": sum(r[f"mean_flow_ratio_over_{control}"] < 1 for r in by_seed) for control in CONTROLS},
                "mean_component_deltas_vs_poisson_s": {k: mean(r["component_deltas_vs_poisson_s"][k] for r in group) for k in COMPONENTS},
                "mean_selector_accounting": {k: mean(r["selector_accounting"][k] for r in raw)
                                             for k in ("selector_mean_wall_ms", "rollout_frames", "rollout_vetoed_frames", "rollout_random_draws", *PRESSURE_COUNTERS)},
            })
    return paired, summary


def build_experiment(output=DEFAULT_OUTPUT):
    from algorithm.experiments.aligned_hardware_pressure_rollout_diagnostic import diagnose

    _check_frozen_sources()
    snapshot = json.loads((ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != snapshot:
        raise ValueError("confirmed sources changed")
    root = Path(__file__).resolve().parents[2]
    fixed = json.loads((ARTIFACTS / "aligned_hardware_control_variate_development_20260910_fixed_sources.json").read_text())
    if any((root / name).read_text() != text for name, text in fixed.items()):
        raise ValueError("frozen Poisson lineage changed")
    previous = json.loads(DEVELOPMENT.read_text())
    continuation = json.loads(CONTINUATION.read_text())
    base = previous["protocol"]["base_protocol"]
    model = load_model()
    certificates = {load: capacity_certificate(model, load) for load in base["loads"]}
    mu = terminal_service_rates(model, certificates[base["loads"][0]])
    if (json.loads(json.dumps(model.snapshot())) != previous["model"]
            or json.loads(json.dumps(list(certificates.values()))) != previous["certificates"]
            or list(mu) != previous["protocol"]["terminal_service_rates"]):
        raise ValueError("new priority must use the frozen model and rates")
    durations = [max(1., option.duration) for option in model.options]
    profiles, setup_reads = _build_rollout_profiles([option.requirements for option in model.options], durations)
    nu, pressure_reads, pressure_evals = static_pressure_rates(profiles, durations, model.lanes)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "base_protocol": base,
        "role": "development only; consumed real arrival seeds", "policies": (*CONTROLS, NEW_POLICY),
        "main_budget": 26, "main_policy": NEW_POLICY, "planning_seeds": PLANNING_SEEDS,
        "terminal_service_rates": mu, "successor_pressure_rates": nu,
        "single_change": "successor class priority q_i/mu_i becomes q_i*nu_i; nu_i=lanes*max(pure batch/effective duration), fixed independently of queue and samples",
        "successors": "choose nonempty class by priority, tie by coordinate, then unchanged largest fitting pure batch and maximal lane copies; two successors per contender",
        "unchanged": "raw Poisson current search, support cap, paid pool, recovery, single shared path and horizon, first frame, original-mu fluid tail, final candidate tie break",
        "sampling": "aligned-planning:{planning_seed}:{coordinate}; caller-owned streams advance only on active calls; trajectories may diverge after changed decisions",
        "cost": "same root plus four paid successor attempts; static pure-profile rates computed once per active comparison; profile/rate evaluations reported separately; priority-coordinate count is a subtotal of class evaluations",
        "controls": "v4 36 and analytic 36 plus Poisson 108; reproduce all 180 frozen runs and deterministic counters",
        "pairing": "same load, real seed, budget, planning seed; mean of per-run ratios across planning seeds within real seed, then across real seeds",
        "frozen_state_diagnostic": {
            "source": CONTINUATION.name, "states": 63, "occurrences": 174, "samples_per_state": 64,
            "rng": "aligned-cv-continuation:{sample_index}:{coordinate}, same samples as frozen raw margins",
            "target": "same fixed contender pairs and old single-action effects under common v4 continuation; all states and occurrences, no new replays",
            "measure": "within-state pressure/raw fixed-margin means, paired changes and MCSE; compare oriented means to frozen actual effects",
            "cost": "charge every diagnostic search, draw, rate setup and class evaluation separately from the online budget",
        },
        "protocol_setup_work": {"requirement_reads": setup_reads, "pressure_profile_reads": pressure_reads, "pressure_rate_evaluations": pressure_evals},
        "stopping_rule": "288 full runs once and all 63 frozen states with 64 samples; no coefficients, seed additions, outcome filters or post-result changes",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    originals = {(r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"]): r for r in previous["runs"]}
    tasks = [(model, certificate, load, seed, base, mu)
             for load, certificate in certificates.items() for seed in base["seeds"]]
    runs = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for rows in pool.map(_run_seed, tasks):
            for row in rows:
                if row["policy"] in CONTROLS:
                    _assert_control_reproduced(row, originals[(row["load"], row["seed"], row["budget"], row["policy"], row["planning_seed"])])
            runs.extend(rows)
            print(json.dumps({"load": rows[0]["load"], "seed": rows[0]["seed"], "runs_completed": len(runs)}), flush=True)
    diagnostic = diagnose(model, certificates, mu, continuation)
    paired, summary = summarize_pressure_runs(runs, base["loads"], base["budgets"])
    new = [r for r in runs if r["policy"] == NEW_POLICY]
    counts = [r["selector_accounting"] for r in new]
    pure_count = sum(map(len, profiles))
    checks = {
        "180_frozen_control_runs_reproduced": True,
        "complete_288_run_roster": len(runs) == 288 and len({(r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"]) for r in runs}) == 288,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "candidate_budget_holds": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= r["budget"] for r in runs),
        "seed_and_support_floors_hold": all(c["minimum_seed_support_margin"] >= 0 and c["minimum_support_floor_margin"] >= 0 and c["maximum_relative_support_loss"] <= .01 + 1e-12 for c in counts),
        "stage_and_total_accounting_hold": all(c["maximum_stage_identity_error"] == c["maximum_budget_identity_error"] == 0 for c in counts) and all(r["selector_accounting"]["root_attempts"] + r["selector_accounting"]["rollout_attempts"] == r["metrics"]["candidate_attempts"] for r in new),
        "four_successor_attempts_per_active_comparison": all(c["rollout_attempts"] == 4 * c["rollout_frames"] for c in counts),
        "population_and_draw_accounting_hold": all(c["maximum_path_population_error"] == c["maximum_random_draw_identity_error"] == 0 for c in counts),
        "static_pressure_work_fully_charged": all(c["rollout_pressure_profile_reads"] == c["rollout_pressure_rate_evaluations"] == pure_count * c["rollout_frames"] for c in counts),
        "priority_work_is_class_evaluation_subtotal": all(c["rollout_priority_coordinate_evaluations"] == c["rollout_class_evaluations"] for c in counts),
        "chosen_rollout_cost_no_worse_than_incumbent": all(c["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8 for c in counts),
        "queue_recursion_and_original_drift_hold": all(r["metrics"]["maximum_queue_recursion_violation"] == 0 and r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "v4_k120_exact": all(r["metrics"]["exact_selection_frame_count"] == r["metrics"]["frame_count"] for r in runs if r["budget"] == 120 and r["policy"] == "v4_support"),
        "frozen_state_diagnostic_integrity": diagnostic["status"] == "PASS",
    }
    report = {
        "experiment": "aligned_hardware_pressure_rollout_development_v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "model": model.snapshot(), "certificates": list(certificates.values()),
        "runs": runs, "paired_comparisons": paired, "summary_by_load_and_budget": summary,
        "frozen_state_diagnostic": diagnostic,
        "limitations": [
            "Static homogeneous throughput pressure is a parameter-free continuation heuristic; it does not equal v4 joint-action selection on finite queues.",
            "Only two successor frames are discrete; the fluid tail and one-sample final comparison remain approximations.",
            "Full runs reuse development arrival seeds with crossed planning repetitions. Frozen diagnostic states were selected from earlier raw/CV disagreements, and their overlapping intervention effects are not independent policy outcomes.",
            "Pressure changes the proxy expectation. Diagnostic target residuals still mix arrival noise, horizon cutoff and continuation differences; execution PASS does not require better performance.",
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
