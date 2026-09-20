"""Develop one catalogue-derived backlog gate on the original arrival seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.backlog_gated_frame_candidates import build_backlog_gate, select_backlog_gated_frame
from algorithm.lookahead_frame_candidates import LookaheadFrameSelection
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import (
    LookaheadRecorder, _check_frozen_sources, terminal_service_rates,
)
from algorithm.experiments.aligned_hardware_lookahead_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, PolicyRecorder, summarize_runs


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_backlog_gate_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_lookahead_development_20260910.json"
POLICIES = ("v4_support", "lookahead_cap_1pct", "backlog_gated_lookahead_cap_1pct")


class BacklogGateRecorder:
    def __init__(self, options, lanes, arrival_rates, service_rates):
        start = perf_counter()
        self.gate = build_backlog_gate(options, lanes)
        self.gate_setup_wall_s = perf_counter() - start
        self.arrival_rates, self.service_rates = arrival_rates, service_rates
        self.frames = []
        self.now = self.selector_wall_s = self.active_duration_s = 0.0
        self.active_frames = self.search_exact_frames = self.lower_support_frames = 0
        self.cost_evaluations = self.offset_reads = self.queue_evaluations = self.terminal_evaluations = 0
        self.minimum_active_seed_margin = self.minimum_active_floor_margin = float("inf")
        self.maximum_relative_loss = 0.0

    def __call__(self, options, queues, lanes, budget):
        start = perf_counter()
        selected = select_backlog_gated_frame(
            options, queues, lanes, budget, gate=self.gate,
            arrival_rates=self.arrival_rates, terminal_service_rates=self.service_rates,
        )
        self.selector_wall_s += perf_counter() - start
        active = isinstance(selected, LookaheadFrameSelection)
        self.active_frames += int(active)
        if active:
            self.active_duration_s += selected.duration
            self.cost_evaluations += selected.cost_candidate_evaluations
            self.offset_reads += selected.completion_offset_reads
            self.queue_evaluations += selected.next_queue_coordinate_evaluations
            self.terminal_evaluations += selected.terminal_value_coordinate_evaluations
            self.minimum_active_seed_margin = min(self.minimum_active_seed_margin, selected.score - selected.seed_support)
            self.minimum_active_floor_margin = min(self.minimum_active_floor_margin, selected.score - selected.support_floor)
            self.maximum_relative_loss = max(self.maximum_relative_loss, selected.relative_support_loss)
            self.lower_support_frames += int(selected.score < selected.support_incumbent_score - 1e-10)
            self.search_exact_frames += int(selected.support_search_exact)
        else:
            self.search_exact_frames += int(selected.exact)
        self.frames.append({"start_s": self.now, "duration_s": selected.duration,
                            "queue_before": tuple(queues), "departures": selected.service,
                            "option_indices": selected.option_indices, "lookahead_active": active})
        self.now += selected.duration
        return selected

    def accounting(self):
        count = len(self.frames)
        return {
            "selector_wall_s": self.selector_wall_s,
            "selector_mean_wall_ms": 1000 * self.selector_wall_s / count,
            "gate_setup_wall_s": self.gate_setup_wall_s,
            "gate_setup_requirement_reads": self.gate.setup_requirement_reads,
            "gate_queue_coordinate_evaluations": count * len(self.gate.weights),
            "lookahead_active_frames": self.active_frames,
            "v4_inactive_frames": count - self.active_frames,
            "lookahead_active_frame_fraction": self.active_frames / count,
            "lookahead_active_duration_s": self.active_duration_s,
            "lookahead_active_time_fraction": self.active_duration_s / self.now,
            "cost_candidate_evaluations": self.cost_evaluations,
            "mean_cost_candidate_evaluations": self.cost_evaluations / count,
            "completion_offset_reads": self.offset_reads,
            "next_queue_coordinate_evaluations": self.queue_evaluations,
            "terminal_value_coordinate_evaluations": self.terminal_evaluations,
            "minimum_active_seed_support_margin": self.minimum_active_seed_margin if self.active_frames else None,
            "minimum_active_support_floor_margin": self.minimum_active_floor_margin if self.active_frames else None,
            "maximum_relative_support_loss": self.maximum_relative_loss,
            "lower_support_than_search_incumbent_frames": self.lower_support_frames,
            "support_search_exact_frames": self.search_exact_frames,
        }


def _run_seed(task):
    model, certificate, load, seed, base, service_rates = task
    tape = poisson_arrivals(certificate["arrival_rates"], base["arrival_horizon_s"], seed)
    rows = []
    for budget in base["budgets"]:
        for policy in POLICIES:
            if policy == "v4_support":
                recorder = PolicyRecorder(policy, sum(certificate["arrival_rates"]), max_relative_support_loss=0.01)
            elif policy == "lookahead_cap_1pct":
                recorder = LookaheadRecorder(certificate["arrival_rates"], service_rates, 1.0)
            else:
                recorder = BacklogGateRecorder(model.options, model.lanes, certificate["arrival_rates"], service_rates)
            metrics = simulate_frames(model, tape, base["arrival_horizon_s"], certificate, budget, selector=recorder)
            decomposition = decompose_flow(model, tape, recorder.frames, base["arrival_horizon_s"])
            if abs(decomposition["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8:
                raise ValueError("backlog-gate trace differs from executed flow time")
            rows.append({"load": load, "seed": seed, "budget": budget, "policy": policy,
                         "metrics": metrics, "decomposition": decomposition,
                         "selector_accounting": recorder.accounting()})
    return rows


def _assert_control_reproduced(row, old):
    for key in ("metrics", "decomposition"):
        if json.loads(json.dumps(row[key])) != old[key]:
            raise ValueError(f"frozen {row['policy']} {key} changed at {(row['load'], row['seed'], row['budget'])}")
    for key, value in row["selector_accounting"].items():
        if key not in ("selector_wall_s", "selector_mean_wall_ms") and value != old["selector_accounting"][key]:
            raise ValueError(f"frozen control accounting changed: {key}")


def build_experiment(output=DEFAULT_OUTPUT):
    _check_frozen_sources()
    snapshot = json.loads((ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != snapshot:
        raise ValueError("confirmed lookahead sources must remain frozen")
    development = json.loads(DEVELOPMENT.read_text())
    base = development["protocol"]["base_protocol"]
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in base["loads"]]
    service_rates = terminal_service_rates(model, certificates[0])
    if (json.loads(json.dumps(model.snapshot())) != development["model"]
            or json.loads(json.dumps(certificates)) != development["certificates"]
            or list(service_rates) != development["protocol"]["terminal_service_rates"]):
        raise ValueError("backlog gating must retain the registered model and service reference")
    gate = build_backlog_gate(model.options, model.lanes)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "base_protocol": base,
        "role": "development only, original seeds 41/42/43; no confirmation seeds",
        "policies": POLICIES, "main_policy": POLICIES[-1], "main_budget": 26,
        "gate_derivation": "b_i=max_a D_ai; w_i=lcm(b)/b_i; C=lanes*max_a(w dot D_a)",
        "gate_weights": gate.weights, "gate_capacity": gate.capacity,
        "activation": "strict w dot Q > C, from visible queues before candidate generation, including drain",
        "gate_interpretation": "every feasible frame has w dot D<=C, so activation implies some visible work must remain; the converse need not hold",
        "branches": "active: frozen lookahead with terminal weight 1 and 1% cap; inactive: frozen v4",
        "support": "both branches retain the seed lower bound; active loss<=1% of paid support incumbent, inactive loss=0",
        "accounting": "exactly one search per frame; gate setup reads counted once per run, one weighted queue scan per frame; prediction work only on active frames",
        "terminal_service_rates": service_rates,
        "arrival_rate_input": "same registered per-class rates throughout arrival window and drain",
        "reference": "v4 at K120; K120 of a cost selector is not the exact support reference",
        "estimator": "arithmetic mean of paired seed mean-flow ratios; development descriptions, no confirmation intervals",
        "stopping_rule": "all 108 cases once, one catalogue-derived gate; no threshold fitting or roster changes after results",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    original = {(r["load"], r["seed"], r["budget"], r["policy"]): r for r in development["runs"]}
    tasks = [(model, cert, load, seed, base, service_rates)
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
            always = indexed[(*key, "lookahead_cap_1pct")]
            gated = indexed[(*key, POLICIES[-1])]
            row["flow_ratio_over_always_lookahead"] = gated["metrics"]["mean_flow_s"] / always["metrics"]["mean_flow_s"]
            row["prediction_work_ratio_over_always_lookahead"] = (
                gated["selector_accounting"]["next_queue_coordinate_evaluations"] /
                always["selector_accounting"]["next_queue_coordinate_evaluations"])
    for group in summary:
        if group["policy"] == POLICIES[-1]:
            members = [r for r in runs if (r["load"], r["budget"], r["policy"]) == (group["load"], group["budget"], group["policy"])]
            pairs = [r for r in paired if (r["load"], r["budget"], r["policy"]) == (group["load"], group["budget"], group["policy"])]
            for key in ("lookahead_active_frame_fraction", "lookahead_active_time_fraction", "mean_cost_candidate_evaluations"):
                group[f"mean_{key}"] = mean(r["selector_accounting"][key] for r in members)
            for key in ("flow_ratio_over_always_lookahead", "prediction_work_ratio_over_always_lookahead"):
                group[f"mean_paired_{key}"] = mean(r[key] for r in pairs)
    gated = [r for r in runs if r["policy"] == POLICIES[-1]]
    checks = {
        "frozen_sources_and_72_controls_unchanged": _source_texts() == snapshot,
        "all_jobs_complete": all(r["metrics"]["job_count"] == r["metrics"]["completed_jobs"] for r in runs),
        "candidate_construction_within_budget": all(r["metrics"]["maximum_candidate_attempts_per_frame"] <= r["budget"] for r in runs),
        "active_seed_and_support_floors_preserved": all(
            r["selector_accounting"][key] is None or r["selector_accounting"][key] >= 0
            for r in gated for key in ("minimum_active_seed_support_margin", "minimum_active_support_floor_margin")),
        "relative_support_loss_within_one_percent": all(r["selector_accounting"]["maximum_relative_support_loss"] <= 0.01 + 1e-12 for r in gated),
        "original_drift_bound_holds": all(r["metrics"]["maximum_expected_drift_bound_violation"] <= 1e-8 for r in runs),
        "queue_recursion_holds": all(r["metrics"]["maximum_queue_recursion_violation"] == 0 for r in runs),
        "k120_support_search_exact": all(r["selector_accounting"]["support_search_exact_frames"] == r["metrics"]["frame_count"] for r in gated if r["budget"] == 120),
        "gate_work_and_branches_accounted": all(
            r["selector_accounting"]["gate_queue_coordinate_evaluations"] == len(model.workloads) * r["metrics"]["frame_count"]
            and r["selector_accounting"]["lookahead_active_frames"] + r["selector_accounting"]["v4_inactive_frames"] == r["metrics"]["frame_count"]
            for r in gated),
        "prediction_work_accounted": all(
            r["selector_accounting"]["next_queue_coordinate_evaluations"] == r["selector_accounting"]["terminal_value_coordinate_evaluations"]
            == len(model.workloads) * r["selector_accounting"]["cost_candidate_evaluations"] for r in gated),
    }
    report = {
        "experiment": "aligned_hardware_backlog_gate_development_v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "model": model.snapshot(), "certificates": certificates,
        "runs": runs, "paired_comparisons": paired, "summary_by_load_and_budget": summary,
        "limitations": [
            "The gate certifies an unavoidable residual queue, not a flow improvement; inactive queues may also require multiple frames.",
            "The active terminal value retains the frozen fluid approximation and stationary arrival rates during drain.",
            "These are reused development seeds; the previously confirmed always-lookahead high-load result does not confirm the gated policy.",
            "Wall times are descriptive measurements under three-worker execution, not dedicated latency benchmarks.",
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
