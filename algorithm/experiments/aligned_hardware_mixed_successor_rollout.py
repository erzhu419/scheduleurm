"""Test lane-by-lane mixed successors on the frozen raw Poisson rollout."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.mixed_successor_rollout_frame_candidates import select_mixed_successor_rollout_frame
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


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_mixed_successor_rollout_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_control_variate_development_20260910.json"
CONTINUATION = ARTIFACTS / "aligned_hardware_control_variate_continuation_development_20260910.json"
CONTROLS = ("v4_support", "lookahead_cap_1pct", "poisson_rollout_cap_1pct")
NEW_POLICY = "mixed_successor_rollout_cap_1pct"
PLANNING_SEEDS = (0, 1, 2)
MIXED_COUNTERS = (
    "rollout_lane_fill_evaluations", "rollout_mixed_priority_coordinate_evaluations",
    "rollout_mixed_profile_probes", "rollout_temporary_queue_updates",
)


class MixedSuccessorRolloutRecorder(PoissonRolloutRecorder):
    def __init__(self, arrival_rates, service_rates, planning_seed):
        super().__init__(arrival_rates, service_rates, planning_seed)
        self.counts.update(dict.fromkeys(MIXED_COUNTERS, 0))

    def __call__(self, options, queues, lanes, budget):
        started = perf_counter()
        selected = select_mixed_successor_rollout_frame(
            options, queues, lanes, budget, arrival_rates=self.arrival_rates,
            terminal_service_rates=self.service_rates, arrival_rngs=self.arrival_rngs,
        )
        self.selector_wall_s += perf_counter() - started
        for key in (*COUNTERS, *ARRIVAL_COUNTERS, "rollout_random_draws", *MIXED_COUNTERS):
            self.counts[key] += getattr(selected, key)
        self.used += int(selected.rollout_used)
        self.vetoed += int(selected.rollout_vetoed)
        self.search_exact += int(selected.support_search_exact)
        self.lower_support += int(selected.score < selected.support_incumbent_score - 1e-10)
        self.minimum_seed_margin = min(self.minimum_seed_margin, selected.score - selected.seed_support)
        self.minimum_floor_margin = min(self.minimum_floor_margin, selected.score - selected.support_floor)
        self.maximum_relative_loss = max(self.maximum_relative_loss, selected.relative_support_loss)
        self.maximum_budget_identity_error = max(
            self.maximum_budget_identity_error,
            abs(selected.attempts - selected.root_attempts - selected.rollout_attempts),
        )
        if selected.rollout_used:
            self.maximum_rollout_cost_excess = max(
                self.maximum_rollout_cost_excess,
                selected.selected_rollout_cost - selected.incumbent_rollout_cost,
            )
            self.maximum_path_population_error = max(
                self.maximum_path_population_error,
                abs(selected.rollout_population_jobs - sum(queues)
                    - selected.rollout_arrival_event_generations),
            )
        expected_draws = (selected.rollout_arrival_event_generations
                          + int(selected.rollout_used) * sum(rate > 0 for rate in self.arrival_rates))
        self.maximum_random_draw_identity_error = max(
            self.maximum_random_draw_identity_error,
            abs(selected.rollout_random_draws - expected_draws),
        )
        self.initial_attempts += selected.initial_root_attempts
        self.recycled_attempts += selected.recycled_candidate_attempts
        self.reclaimed_frames += int(selected.reclaimed_reserve)
        self.resumed_frames += int(selected.current_search_resumed)
        self.initial_cost_evaluations += selected.initial_cost_candidate_evaluations
        self.recycled_cost_evaluations += selected.recycled_cost_candidate_evaluations
        self.maximum_stage_identity_error = max(
            self.maximum_stage_identity_error,
            abs(selected.root_attempts - selected.initial_root_attempts
                - selected.recycled_candidate_attempts),
            abs(selected.cost_candidate_evaluations - selected.initial_cost_candidate_evaluations
                - selected.recycled_cost_candidate_evaluations),
        )
        self.frames.append({"start_s": self.now, "duration_s": selected.duration,
                            "queue_before": tuple(queues), "departures": selected.service,
                            "option_indices": selected.option_indices})
        self.now += selected.duration
        return selected


def _run_seed(task):
    model, certificate, load, seed, base, mu = task
    rates = certificate["arrival_rates"]
    tape = poisson_arrivals(rates, base["arrival_horizon_s"], seed)
    roster = ([(policy, None) for policy in CONTROLS[:2]]
              + [(policy, planning_seed) for policy in (CONTROLS[-1], NEW_POLICY)
                 for planning_seed in PLANNING_SEEDS])
    rows = []
    for budget in base["budgets"]:
        for policy, planning_seed in roster:
            if policy == CONTROLS[0]:
                recorder = PolicyRecorder(policy, sum(rates), max_relative_support_loss=.01)
            elif policy == CONTROLS[1]:
                recorder = LookaheadRecorder(rates, mu, 1.)
            else:
                recorder_type = PoissonRolloutRecorder if policy == CONTROLS[2] else MixedSuccessorRolloutRecorder
                recorder = recorder_type(rates, mu, planning_seed)
            metrics = simulate_frames(model, tape, base["arrival_horizon_s"], certificate, budget, selector=recorder)
            parts = decompose_flow(model, tape, recorder.frames, base["arrival_horizon_s"])
            if abs(parts["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                raise ValueError("mixed-successor trace does not reconstruct executed flow")
            rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                         "planning_seed": planning_seed, "metrics": metrics,
                         "decomposition": parts, "selector_accounting": recorder.accounting()})
    return rows


def summarize_mixed_runs(runs, loads, budgets):
    controls = {(r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"]): r
                for r in runs if r["policy"] in CONTROLS}
    paired = []
    for row in runs:
        if row["policy"] != NEW_POLICY:
            continue
        pair = {key: row[key] for key in ("load", "seed", "budget", "planning_seed")}
        pair["policy"], pair["mean_flow_s"] = NEW_POLICY, row["metrics"]["mean_flow_s"]
        for control in CONTROLS:
            control_seed = row["planning_seed"] if control == CONTROLS[-1] else None
            old = controls[(row["load"], row["seed"], row["budget"], control, control_seed)]
            if row["metrics"]["job_count"] != old["metrics"]["job_count"]:
                raise ValueError("paired policies must complete the same population")
            pair[f"flow_ratio_over_{control}"] = pair["mean_flow_s"] / old["metrics"]["mean_flow_s"]
            pair[f"p90_ratio_over_{control}"] = row["metrics"]["p90_flow_s"] / old["metrics"]["p90_flow_s"]
        old = controls[(row["load"], row["seed"], row["budget"], CONTROLS[-1], row["planning_seed"])]
        pair["component_deltas_vs_poisson_s"] = {
            key: row["decomposition"][key] - old["decomposition"][key] for key in COMPONENTS}
        paired.append(pair)
    summaries = []
    ratio_fields = [f"{metric}_ratio_over_{control}"
                    for metric in ("flow", "p90") for control in CONTROLS]
    for load in loads:
        for budget in budgets:
            group = [r for r in paired if (r["load"], r["budget"]) == (load, budget)]
            by_seed = []
            for seed in sorted({r["seed"] for r in group}):
                members = [r for r in group if r["seed"] == seed]
                if sorted(r["planning_seed"] for r in members) != list(PLANNING_SEEDS):
                    raise ValueError("each real seed must retain every planning repetition exactly once")
                by_seed.append({"seed": seed, "mean_flow_s": mean(r["mean_flow_s"] for r in members),
                                **{f"mean_{key}": mean(r[key] for r in members) for key in ratio_fields}})
            raw = [r for r in runs
                   if (r["load"], r["budget"], r["policy"]) == (load, budget, NEW_POLICY)]
            summaries.append({
                "load": load, "budget": budget, "policy": NEW_POLICY, "run_count": len(group),
                "by_arrival_seed": by_seed,
                "by_planning_seed": [
                    {"planning_seed": planning_seed,
                     **{f"mean_{key}": mean(r[key] for r in group if r["planning_seed"] == planning_seed)
                        for key in ratio_fields}}
                    for planning_seed in PLANNING_SEEDS],
                "mean_flow_s": mean(r["mean_flow_s"] for r in by_seed),
                **{f"mean_paired_{key}": mean(r[f"mean_{key}"] for r in by_seed)
                   for key in ratio_fields},
                **{f"improved_arrival_seed_count_vs_{control}":
                   sum(r[f"mean_flow_ratio_over_{control}"] < 1 for r in by_seed)
                   for control in CONTROLS},
                "mean_component_deltas_vs_poisson_s": {
                    key: mean(r["component_deltas_vs_poisson_s"][key] for r in group)
                    for key in COMPONENTS},
                "mean_selector_accounting": {
                    key: mean(r["selector_accounting"][key] for r in raw)
                    for key in ("selector_mean_wall_ms", "rollout_frames",
                                "rollout_vetoed_frames", "rollout_random_draws", *MIXED_COUNTERS)},
            })
    return paired, summaries


def build_experiment(output=DEFAULT_OUTPUT):
    from algorithm.experiments.aligned_hardware_mixed_successor_rollout_diagnostic import diagnose

    _check_frozen_sources()
    source_snapshot = json.loads(
        (ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != source_snapshot:
        raise ValueError("confirmed sources changed")
    root = Path(__file__).resolve().parents[2]
    frozen_sources = json.loads(
        (ARTIFACTS / "aligned_hardware_control_variate_development_20260910_fixed_sources.json").read_text())
    if any((root / name).read_text() != text for name, text in frozen_sources.items()):
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
        raise ValueError("mixed successors must use the frozen model and rates")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_protocol": base, "role": "development only; consumed real arrival seeds",
        "policies": (*CONTROLS, NEW_POLICY), "main_budget": 26, "main_policy": NEW_POLICY,
        "planning_seeds": PLANNING_SEEDS, "terminal_service_rates": mu,
        "single_change": "each successor fills lanes sequentially; at each lane choose nonempty temporary queue maximizing q_i/mu_i, take its largest fitting pure profile, then decrement temporary queue",
        "successor_frame": "ordered selected profile indices; class service is accumulated; duration is max of idle and selected effective durations; unused lanes remain idle",
        "unchanged": "raw Poisson current search, paid pool, support cap, recovery, shared path and horizon, first frame, q_i/mu_i priority, original-mu fluid tail and final candidate tie break",
        "budget": "one fully constructed feasible successor frame is one attempt; two successors per contender remain four paid attempts, so root+successors <= K",
        "extra_work": "lane positions visited (including the first empty position, then stop), each nonempty priority coordinate, profile probes and assigned-profile temporary queue updates are reported; priority/profile/update fields are subtotals of existing native counters",
        "controls": "v4 36, analytic 36 and raw Poisson 108; reproduce all 180 runs and deterministic counters",
        "pairing": "same load, real seed, K and planning seed; average run ratios within real seed, then across real seeds",
        "frozen_state_diagnostic": {
            "source": CONTINUATION.name, "states": 63, "occurrences": 174,
            "samples_per_state": 64,
            "rng": "aligned-cv-continuation:{sample_index}:{coordinate}, matching frozen raw margins",
            "target": "same fixed contenders and retained CV-first-minus-raw-first common-v4 effects; no new real trajectory replay",
            "measure": "paired mixed-minus-raw fixed-contender margins and oriented target residuals; structural checks only",
        },
        "stopping_rule": "288 full runs once and all 63 frozen states x64 samples; no coefficients, seed additions, state filtering or post-result changes",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    originals = {(r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"]): r
                 for r in previous["runs"]}
    tasks = [(model, certificate, load, seed, base, mu)
             for load, certificate in certificates.items() for seed in base["seeds"]]
    runs = []
    with ProcessPoolExecutor(max_workers=protocol["execution_workers"]) as pool:
        for rows in pool.map(_run_seed, tasks):
            for row in rows:
                if row["policy"] in CONTROLS:
                    _assert_control_reproduced(
                        row, originals[(row["load"], row["seed"], row["budget"],
                                        row["policy"], row["planning_seed"])])
            runs.extend(rows)
            print(json.dumps({"load": rows[0]["load"], "seed": rows[0]["seed"],
                              "runs_completed": len(runs)}), flush=True)
    diagnostic = diagnose(model, certificates, mu, continuation)
    paired, summaries = summarize_mixed_runs(runs, base["loads"], base["budgets"])
    new = [r for r in runs if r["policy"] == NEW_POLICY]
    counts = [r["selector_accounting"] for r in new]
    checks = {
        "180_frozen_control_runs_reproduced": True,
        "complete_288_run_roster": len(runs) == 288 and len({
            (r["load"], r["seed"], r["budget"], r["policy"], r["planning_seed"])
            for r in runs}) == 288,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "candidate_budget_holds": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= r["budget"] for r in runs),
        "seed_and_support_floors_hold": all(
            c["minimum_seed_support_margin"] >= 0 and c["minimum_support_floor_margin"] >= 0
            and c["maximum_relative_support_loss"] <= .01 + 1e-12 for c in counts),
        "stage_and_total_accounting_hold": all(
            c["maximum_stage_identity_error"] == c["maximum_budget_identity_error"] == 0
            for c in counts) and all(
                r["selector_accounting"]["root_attempts"] + r["selector_accounting"]["rollout_attempts"]
                == r["metrics"]["candidate_attempts"] for r in new),
        "four_successor_attempts_per_active_comparison": all(
            c["rollout_attempts"] == 4 * c["rollout_frames"] for c in counts),
        "population_and_draw_accounting_hold": all(
            c["maximum_path_population_error"] == c["maximum_random_draw_identity_error"] == 0
            for c in counts),
        "mixed_work_subtotals_hold": all(
            c["rollout_mixed_priority_coordinate_evaluations"] == c["rollout_class_evaluations"]
            and c["rollout_mixed_profile_probes"] == c["rollout_profile_probes"]
            and c["rollout_temporary_queue_updates"] <= c["rollout_queue_coordinate_evaluations"]
            for c in counts),
        "lane_fill_work_bounded": all(
            c["rollout_temporary_queue_updates"] <= c["rollout_lane_fill_evaluations"]
            <= 8 * c["rollout_frames"] for c in counts),
        "chosen_rollout_cost_no_worse_than_incumbent": all(
            c["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8 for c in counts),
        "queue_recursion_and_original_drift_hold": all(
            r["metrics"]["maximum_queue_recursion_violation"] == 0
            and r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "v4_k120_exact": all(r["metrics"]["exact_selection_frame_count"] == r["metrics"]["frame_count"]
                             for r in runs if r["budget"] == 120 and r["policy"] == CONTROLS[0]),
        "frozen_state_diagnostic_integrity": diagnostic["status"] == "PASS",
    }
    report = {
        "experiment": "aligned_hardware_mixed_successor_rollout_development_v1",
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "protocol": protocol, "model": model.snapshot(), "certificates": list(certificates.values()),
        "runs": runs, "paired_comparisons": paired,
        "summary_by_load_and_budget": summaries, "frozen_state_diagnostic": diagnostic,
        "limitations": [
            "Sequential q/mu lane filling is a deterministic continuation heuristic; it does not optimize all feasible joint successor frames.",
            "The candidate-attempt budget is unchanged, but mixed construction performs more per-lane work; operation counts and measured time remain material.",
            "Only two successor frames are discrete; the fluid tail and single-sample final comparison remain approximations.",
            "Full runs reuse development seeds. Frozen states were selected from earlier disagreements, and overlapping retained intervention effects are not independent policy outcomes or pure model-bias estimates.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_experiment(args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]),
                      "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
