"""Develop the strict two-frame backlog gate on previously used seeds only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.mixed_successor_rollout_frame_candidates import MixedSuccessorRolloutFrameSelection
from algorithm.rollout_depth_gated_frame_candidates import build_backlog_gate, select_rollout_depth_gated_frame
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_arrival_rollout import ARRIVAL_COUNTERS
from algorithm.experiments.aligned_hardware_batch_rollout import COUNTERS
from algorithm.experiments.aligned_hardware_control_variate import _assert_control_reproduced
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import LookaheadRecorder, terminal_service_rates
from algorithm.experiments.aligned_hardware_mixed_successor_rollout import (
    CONTROLS, MIXED_COUNTERS, NEW_POLICY as MIXED_POLICY, MixedSuccessorRolloutRecorder,
)
from algorithm.experiments.aligned_hardware_poisson_rollout import PoissonRolloutRecorder
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, PolicyRecorder


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_rollout_depth_gate_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_mixed_successor_rollout_development_20260910.json"
CONTINUATION = ARTIFACTS / "aligned_hardware_mixed_successor_continuation_development_20260910.json"
CONTROL_POLICIES = (*CONTROLS, MIXED_POLICY)
NEW_POLICY = "rollout_depth_gated_cap_1pct"
REAL_SEEDS, PLANNING_SEEDS = (41, 42, 43), (0, 1, 2)
GATE_WEIGHTS, GATE_CAPACITY, GATE_THRESHOLD = (8, 8, 3), 48, 96
ADVANCE_CRITERIA = {
    "execution_checks_all_pass": True,
    "each_load_mean_flow_ratio_vs_v4_upper": 1.0,
    "each_load_mean_p90_ratio_vs_v4_upper": 1.0,
    "strict_improvement_loads": [.8, .95],
    "each_real_seed_planning_mean_flow_ratio_vs_v4_strict_upper": 1.0,
    "role": "development advancement only",
}
SOURCE_FILES = (
    "algorithm/backlog_gated_frame_candidates.py",
    "algorithm/rollout_depth_gated_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_control_variate.py",
    "algorithm/experiments/aligned_hardware_frame_confirmation.py",
    "algorithm/experiments/aligned_hardware_rollout_depth_gate.py",
)


class RolloutDepthGateRecorder(MixedSuccessorRolloutRecorder):
    def __init__(self, options, lanes, arrival_rates, service_rates, planning_seed):
        super().__init__(arrival_rates, service_rates, planning_seed)
        started = perf_counter()
        self.gate = build_backlog_gate(options, lanes)
        self.gate_setup_wall_s = perf_counter() - started
        self.mixed_branch_frames = 0
        self.mixed_branch_duration_s = 0.0

    def __call__(self, options, queues, lanes, budget):
        started = perf_counter()
        selected = select_rollout_depth_gated_frame(
            options, queues, lanes, budget, gate=self.gate,
            arrival_rates=self.arrival_rates, terminal_service_rates=self.service_rates,
            arrival_rngs=self.arrival_rngs,
        )
        self.selector_wall_s += perf_counter() - started
        active = isinstance(selected, MixedSuccessorRolloutFrameSelection)
        self.mixed_branch_frames += int(active)
        self.mixed_branch_duration_s += selected.duration if active else 0.0
        for key in (*COUNTERS, *ARRIVAL_COUNTERS, "rollout_random_draws", *MIXED_COUNTERS):
            self.counts[key] += getattr(selected, key, selected.attempts if key == "root_attempts" else 0)
        self.initial_attempts += selected.initial_root_attempts if active else selected.attempts
        self.search_exact += int(selected.support_search_exact if active else selected.exact)
        if active:
            self.used += int(selected.rollout_used)
            self.vetoed += int(selected.rollout_vetoed)
            self.lower_support += int(selected.score < selected.support_incumbent_score - 1e-10)
            self.minimum_seed_margin = min(self.minimum_seed_margin, selected.score - selected.seed_support)
            self.minimum_floor_margin = min(self.minimum_floor_margin, selected.score - selected.support_floor)
            self.maximum_relative_loss = max(self.maximum_relative_loss, selected.relative_support_loss)
            self.maximum_budget_identity_error = max(self.maximum_budget_identity_error,
                abs(selected.attempts - selected.root_attempts - selected.rollout_attempts))
            self.recycled_attempts += selected.recycled_candidate_attempts
            self.reclaimed_frames += int(selected.reclaimed_reserve)
            self.resumed_frames += int(selected.current_search_resumed)
            self.initial_cost_evaluations += selected.initial_cost_candidate_evaluations
            self.recycled_cost_evaluations += selected.recycled_cost_candidate_evaluations
            self.maximum_stage_identity_error = max(self.maximum_stage_identity_error,
                abs(selected.root_attempts - selected.initial_root_attempts - selected.recycled_candidate_attempts),
                abs(selected.cost_candidate_evaluations - selected.initial_cost_candidate_evaluations - selected.recycled_cost_candidate_evaluations))
            expected_draws = selected.rollout_arrival_event_generations + int(selected.rollout_used) * sum(rate > 0 for rate in self.arrival_rates)
            self.maximum_random_draw_identity_error = max(self.maximum_random_draw_identity_error,
                abs(selected.rollout_random_draws - expected_draws))
            if selected.rollout_used:
                self.maximum_rollout_cost_excess = max(self.maximum_rollout_cost_excess,
                    selected.selected_rollout_cost - selected.incumbent_rollout_cost)
                self.maximum_path_population_error = max(self.maximum_path_population_error,
                    abs(selected.rollout_population_jobs - sum(queues) - selected.rollout_arrival_event_generations))
        self.frames.append({"start_s": self.now, "duration_s": selected.duration,
                            "queue_before": tuple(queues), "departures": selected.service,
                            "option_indices": selected.option_indices})
        self.now += selected.duration
        return selected

    def accounting(self):
        result = super().accounting()
        if not self.mixed_branch_frames:
            result["minimum_seed_support_margin"] = result["minimum_support_floor_margin"] = None
        return {**result, "gate_setup_wall_s": self.gate_setup_wall_s,
                "gate_setup_requirement_reads": self.gate.setup_requirement_reads,
                "gate_queue_coordinate_evaluations": len(self.frames) * len(self.gate.weights),
                "gate_threshold_comparisons": len(self.frames),
                "mixed_branch_frames": self.mixed_branch_frames,
                "v4_branch_frames": len(self.frames) - self.mixed_branch_frames,
                "mixed_branch_duration_s": self.mixed_branch_duration_s}


def _run_seed(task):
    model, certificate, load, seed, protocol = task
    rates, mu = certificate["arrival_rates"], protocol["terminal_service_rates"]
    horizon, budget = protocol["arrival_horizon_s"], protocol["budget"]
    tape = poisson_arrivals(rates, horizon, seed)
    roster = ([(policy, None) for policy in CONTROLS[:2]]
              + [(policy, planning_seed) for policy in (CONTROLS[-1], MIXED_POLICY, NEW_POLICY)
                 for planning_seed in protocol["planning_seeds"]])
    rows = []
    for policy, planning_seed in roster:
        if policy == CONTROLS[0]:
            recorder = PolicyRecorder(policy, sum(rates), max_relative_support_loss=.01)
        elif policy == CONTROLS[1]:
            recorder = LookaheadRecorder(rates, mu, 1.)
        elif policy == NEW_POLICY:
            recorder = RolloutDepthGateRecorder(model.options, model.lanes, rates, mu, planning_seed)
        else:
            cls = PoissonRolloutRecorder if policy == CONTROLS[-1] else MixedSuccessorRolloutRecorder
            recorder = cls(rates, mu, planning_seed)
        metrics = simulate_frames(model, tape, horizon, certificate, budget, selector=recorder)
        parts = decompose_flow(model, tape, recorder.frames, horizon)
        if abs(parts["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
            raise ValueError("depth-gate trace does not reconstruct executed flow")
        rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                     "planning_seed": planning_seed, "metrics": metrics,
                     "decomposition": parts, "selector_accounting": recorder.accounting()})
    return rows


def summarize_runs(runs, loads):
    controls = {(r["load"], r["seed"], r["policy"], r["planning_seed"]): r
                for r in runs if r["policy"] in CONTROL_POLICIES}
    paired = []
    for row in runs:
        if row["policy"] != NEW_POLICY:
            continue
        pair = {key: row[key] for key in ("load", "seed", "planning_seed")}
        for policy in CONTROL_POLICIES:
            p = row["planning_seed"] if policy in (CONTROLS[-1], MIXED_POLICY) else None
            old = controls[(row["load"], row["seed"], policy, p)]
            if row["metrics"]["job_count"] != old["metrics"]["job_count"]:
                raise ValueError("paired depth-gate runs have different populations")
            for field, metric in (("flow", "mean_flow_s"), ("p90", "p90_flow_s")):
                pair[f"{field}_ratio_over_{policy}"] = row["metrics"][metric] / old["metrics"][metric]
        paired.append(pair)
    summaries = []
    fields = [f"{metric}_ratio_over_{policy}" for metric in ("flow", "p90") for policy in CONTROL_POLICIES]
    for load in loads:
        group = [row for row in paired if row["load"] == load]
        by_seed = []
        for seed in REAL_SEEDS:
            members = [row for row in group if row["seed"] == seed]
            if sorted(row["planning_seed"] for row in members) != list(PLANNING_SEEDS):
                raise ValueError("every old real seed must retain planning repetitions 0, 1 and 2")
            by_seed.append({"seed": seed, **{key: mean(row[key] for row in members) for key in fields}})
        summaries.append({"load": load, "by_arrival_seed": by_seed,
                          **{f"mean_paired_{key}": mean(row[key] for row in by_seed) for key in fields}})
    return paired, summaries


def advance_decision(summaries, checks):
    return {
        "execution_passed": all(checks.values()),
        "mean_flow_and_p90_no_worse_than_v4_at_each_load": all(
            row["mean_paired_flow_ratio_over_v4_support"] <= 1
            and row["mean_paired_p90_ratio_over_v4_support"] <= 1 for row in summaries),
        "every_medium_high_real_seed_improves_flow_over_v4": all(
            seed["flow_ratio_over_v4_support"] < 1
            for row in summaries if row["load"] in (.8, .95) for seed in row["by_arrival_seed"]),
    }


def continuation_summary(continuation, gate):
    """Filter retained single-action effects; do not interpret them as gated-policy effects."""
    rows = [(run["load"], run["source_policy"], case,
             gate.load(case["queue"]) > 2 * gate.capacity)
            for run in continuation["runs"] for case in run["cases"]]

    def summarize(group):
        retained = [case for _, _, case, active in group if active]
        return {"all_disagreement_occurrences": len(group), "retained_occurrences": len(retained),
                "retained_unique_states": len({case["state_id"] for case in retained}),
                **{outcome: sum(case["effect"]["outcome"] == outcome for case in retained)
                   for outcome in ("improved", "worsened", "unchanged")},
                "mean_mixed_minus_raw_delta_job_s": mean(case["effect"]["realized_total_delta_job_s"] for case in retained) if retained else None}

    return {"source": CONTINUATION.name, "summary": summarize(rows),
            "summary_by_load_and_source": [
                {"load": load, "source_policy": policy,
                 **summarize([row for row in rows if row[:2] == (load, policy)])}
                for load in continuation["protocol"]["loads"]
                for policy in (CONTROLS[-1], MIXED_POLICY)],
            "gate_queue_coordinate_evaluations": len(rows) * len(gate.weights),
            "gate_threshold_comparisons": len(rows),
            "new_selector_calls": 0, "new_replays": 0, "new_random_draws": 0,
            "role": "retained raw-versus-mixed interventions under common v4; gate filtering only, with no new replay"}


def _source_snapshot():
    sources = {}
    for name in ("aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json",
                 "aligned_hardware_control_variate_development_20260910_fixed_sources.json",
                 "aligned_hardware_mixed_successor_continuation_development_20260910_fixed_sources.json"):
        sources.update(json.loads((ARTIFACTS / name).read_text()))
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("frozen development sources changed")
    return {**sources, **{name: (ROOT / name).read_text() for name in SOURCE_FILES}}


def register(output=DEFAULT_OUTPUT):
    output = Path(output)
    paths = [output, output.with_name(output.stem + "_protocol.json"),
             output.with_name(output.stem + "_fixed_sources.json"), output.with_name(output.stem + "_runs.jsonl")]
    if any(path.exists() for path in paths):
        raise ValueError("depth-gate artifacts already exist")
    old = json.loads(DEVELOPMENT.read_text())
    base = old["protocol"]["base_protocol"]
    if (old["status"] != "PASS" or tuple(base["seeds"]) != REAL_SEEDS
            or tuple(old["protocol"]["planning_seeds"]) != PLANNING_SEEDS
            or tuple(base["loads"]) != (.5, .8, .95) or 26 not in base["budgets"]):
        raise ValueError("depth-gate development requires the completed original-seed source")
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in base["loads"]]
    mu = terminal_service_rates(model, certificates[0])
    if (json.loads(json.dumps(model.snapshot())) != old["model"]
            or json.loads(json.dumps(certificates)) != old["certificates"]
            or list(mu) != old["protocol"]["terminal_service_rates"]):
        raise ValueError("registered depth gate must retain the cached development model and rates")
    gate = build_backlog_gate(model.options, model.lanes)
    if (gate.weights != GATE_WEIGHTS or gate.capacity != GATE_CAPACITY
            or GATE_THRESHOLD != 2 * gate.capacity):
        raise ValueError("catalogue gate differs from the fixed (8,8,3), C48 derivation")
    continuation = json.loads(CONTINUATION.read_text())
    if continuation["status"] != "PASS" or not all(continuation["checks"].values()):
        raise ValueError("retained development continuation did not pass its execution checks")
    retained_summary = continuation_summary(continuation, gate)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development only; reuse original real and planning seeds",
        "loads": base["loads"], "seeds": REAL_SEEDS, "planning_seeds": PLANNING_SEEDS,
        "budget": 26, "arrival_horizon_s": base["arrival_horizon_s"],
        "terminal_service_rates": mu, "model": model.snapshot(), "certificates": certificates,
        "policies": (*CONTROL_POLICIES, NEW_POLICY),
        "gate_weights": gate.weights, "gate_capacity": gate.capacity, "strict_threshold": GATE_THRESHOLD,
        "activation": "w dot Q > 2C = 96 before candidate construction, including drain",
        "branches": "active frozen mixed successors; inactive frozen v4; exactly one selector per frame",
        "roster": "3 loads x 3 old real seeds x 11 runs = 99; v4 and analytic once, raw/mixed/gated each at planning seeds 0,1,2",
        "controls": "all 72 v4, analytic, raw and always-mixed rows reproduce the frozen development metrics, decomposition and deterministic accounting",
        "advance_criteria": ADVANCE_CRITERIA,
        "pairing": "matched load/real seed/planning seed; mean planning ratios within real seed, then mean real seeds",
        "accounting": "gate setup once per run; one weighted queue scan per frame; inactive v4 attempts enter root and initial totals; mixed counters retain native totals and named subtotals",
        "retained_continuation": "filter all old raw/mixed single-action cases by the same strict gate; no new replay and no gated-policy causal claim",
        "retained_continuation_summary": retained_summary,
        "protocol_setup_requirement_reads": gate.setup_requirement_reads,
        "stopping_rule": "all 99 registered development runs once; no threshold fitting, added seeds or confirmation data",
        "execution_workers": 3,
    }
    sources = _source_snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(paths[1], protocol)
    _write_new(paths[2], sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output = Path(output)
    protocol_path = output.with_name(output.stem + "_protocol.json")
    snapshot_path = output.with_name(output.stem + "_fixed_sources.json")
    runs_path = output.with_name(output.stem + "_runs.jsonl")
    if output.exists() or runs_path.exists():
        raise ValueError("registered depth-gate results already exist")
    protocol, sources = json.loads(protocol_path.read_text()), json.loads(snapshot_path.read_text())
    expected_protocol = {
        "seeds": list(REAL_SEEDS), "planning_seeds": list(PLANNING_SEEDS),
        "loads": [.5, .8, .95], "budget": 26,
        "arrival_horizon_s": 20000.0,
        "gate_weights": list(GATE_WEIGHTS), "gate_capacity": GATE_CAPACITY,
        "strict_threshold": GATE_THRESHOLD,
        "policies": list((*CONTROL_POLICIES, NEW_POLICY)), "advance_criteria": ADVANCE_CRITERIA,
    }
    if any(protocol[key] != value for key, value in expected_protocol.items()):
        raise ValueError("registered development roster, gate or advancement criteria changed before arrivals")
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("registered depth-gate sources changed")
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in protocol["loads"]]
    mu = terminal_service_rates(model, certificates[0])
    if (json.loads(json.dumps(model.snapshot())) != protocol["model"]
            or json.loads(json.dumps(certificates)) != protocol["certificates"]
            or list(mu) != protocol["terminal_service_rates"]):
        raise ValueError("registered depth-gate model or rates changed before arrivals")
    old = json.loads(DEVELOPMENT.read_text())
    originals = {(row["load"], row["seed"], row["policy"], row["planning_seed"]): row
                 for row in old["runs"] if row["budget"] == 26 and row["policy"] in CONTROL_POLICIES}
    tasks = [(model, certificate, load, seed, protocol)
             for load, certificate in zip(protocol["loads"], certificates, strict=True)
             for seed in protocol["seeds"]]
    started = perf_counter()
    runs, reproduced = [], 0
    with runs_path.open("x") as stream:
        with ProcessPoolExecutor(max_workers=protocol["execution_workers"]) as pool:
            for rows in pool.map(_run_seed, tasks):
                for row in rows:
                    if row["policy"] in CONTROL_POLICIES:
                        _assert_control_reproduced(row, originals[(row["load"], row["seed"], row["policy"], row["planning_seed"])])
                        reproduced += 1
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                runs.extend(rows)
                print(json.dumps({"load": rows[0]["load"], "seed": rows[0]["seed"], "runs_completed": len(runs)}), flush=True)
    gated = [row for row in runs if row["policy"] == NEW_POLICY]
    counts = [row["selector_accounting"] for row in gated]
    populations = {}
    for row in runs:
        populations.setdefault((row["load"], row["seed"]), set()).add(row["metrics"]["job_count"])
    checks = {
        "all_72_controls_reproduced": reproduced == 72,
        "complete_99_unique_runs": len(runs) == 99 and len({(r["load"], r["seed"], r["policy"], r["planning_seed"]) for r in runs}) == 99,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "same_tape_population": all(len(group) == 1 for group in populations.values()),
        "flow_decomposition_holds": all(abs(r["metrics"]["mean_flow_s"] - r["decomposition"]["mean_flow_s"]) <= 1e-8 for r in runs),
        "candidate_budget_holds": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= 26 for r in runs),
        "seed_and_support_floors_hold": all((c["minimum_seed_support_margin"] is None or c["minimum_seed_support_margin"] >= 0) and (c["minimum_support_floor_margin"] is None or c["minimum_support_floor_margin"] >= 0) and c["maximum_relative_support_loss"] <= .01 + 1e-12 for c in counts),
        "native_work_and_draws_hold": all(c["maximum_stage_identity_error"] == c["maximum_budget_identity_error"] == c["maximum_path_population_error"] == c["maximum_random_draw_identity_error"] == 0 and c["rollout_attempts"] == 4 * c["rollout_frames"] and c["rollout_rng_initializations"] == len(model.workloads) for c in counts),
        "root_attempts_include_inactive_v4": all(r["selector_accounting"]["root_attempts"] + r["selector_accounting"]["rollout_attempts"] == r["metrics"]["candidate_attempts"] for r in gated),
        "aggregate_initial_recycled_identities_hold": all(c["root_attempts"] == c["initial_root_attempts"] + c["recycled_candidate_attempts"] and c["cost_candidate_evaluations"] == c["initial_cost_candidate_evaluations"] + c["recycled_cost_candidate_evaluations"] for c in counts),
        "gate_and_lane_work_hold": all(c["mixed_branch_frames"] + c["v4_branch_frames"] == r["metrics"]["frame_count"] and c["gate_queue_coordinate_evaluations"] == len(model.workloads) * r["metrics"]["frame_count"] and c["rollout_temporary_queue_updates"] <= c["rollout_lane_fill_evaluations"] <= 4 * model.lanes * c["rollout_frames"] and c["rollout_mixed_priority_coordinate_evaluations"] == c["rollout_class_evaluations"] and c["rollout_mixed_profile_probes"] == c["rollout_profile_probes"] for r, c in zip(gated, counts, strict=True)),
        "gate_setup_and_threshold_work_hold": all(c["gate_setup_requirement_reads"] == 54 and c["gate_threshold_comparisons"] == r["metrics"]["frame_count"] for r, c in zip(gated, counts, strict=True)),
        "temporary_queue_subtotal_and_rollout_cost_hold": all(c["rollout_temporary_queue_updates"] <= c["rollout_queue_coordinate_evaluations"] and c["maximum_rollout_cost_excess_over_incumbent"] <= 1e-8 for c in counts),
        "queue_recursion_and_drift_hold": all(r["metrics"]["maximum_queue_recursion_violation"] == 0 and r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "registered_sources_unchanged": all((ROOT / name).read_text() == text for name, text in sources.items()),
    }
    paired, summaries = summarize_runs(runs, protocol["loads"])
    advance = advance_decision(summaries, checks)
    report = {"experiment": "aligned_hardware_rollout_depth_gate_development_v1",
              "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "protocol": protocol, "runs": runs, "paired_comparisons": paired,
              "summary_by_load": summaries, "development_advance_checks": advance,
              "development_advance": all(advance.values()),
              "retained_continuation_summary": protocol["retained_continuation_summary"],
              "wall_s": perf_counter() - started,
              "limitations": ["The threshold certifies remaining visible backlog, not flow-time improvement.",
                              "Development seeds and retained single-action targets are reused; this result is not an independent confirmation."]}
    _write_new(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = register(args.output) if args.register else run_registered(args.output)
    print(json.dumps({"status": "REGISTERED" if args.register else result["status"], "output": str(args.output)}))
    raise SystemExit(0 if args.register or result["status"] == "PASS" else 1)
