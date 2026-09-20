"""Run the aligned hardware model and budgeted selector without paper exports.

The observation is a finite prefix of a continuous-time Poisson arrival stream,
followed by draining those jobs. Arrivals during a frame become eligible at its
end. The same complete-job frame supplies selection, service, job completion
offsets, elapsed arrival time, capacity, and conditional Lyapunov drift.

Run: python3 -m algorithm.experiments.aligned_hardware_frame_experiment
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Sequence

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import (
    FrameModel, capacity_certificate, load_model,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "md/experiment_artifacts/aligned_hardware_frame_experiment_v4_20260909.json"


def poisson_arrivals(rates: Sequence[float], horizon: float, seed: int) -> tuple[tuple[float, int, int], ...]:
    """One exogenous tape shared by every budget; never passed to the selector."""
    arrivals = []
    for coordinate, rate in enumerate(rates):
        rng = random.Random(f"aligned-frame:{seed}:{coordinate}")
        time = 0.0
        index = 0
        while rate > 0:
            time += rng.expovariate(rate)
            if time >= horizon:
                break
            arrivals.append((time, coordinate, index))
            index += 1
    return tuple(sorted(arrivals))


def conditional_lyapunov_drift(queues, service, arrival_means) -> float:
    """Exact conditional drift for feasible integer departures and Poisson arrivals."""
    return sum(
        q * (a - d) + 0.5 * ((a - d) ** 2 + a)
        for q, d, a in zip(queues, service, arrival_means, strict=True)
    )


def exact_search_budget(option_count: int, lanes: int) -> int:
    # A bound on all lazy local-option attempts, including the seed catalogue.
    width = option_count + 1
    return width + sum(width ** depth for depth in range(1, lanes + 1))


def simulate_frames(
    model: FrameModel,
    arrivals: Sequence[tuple[float, int, int]],
    arrival_horizon: float,
    certificate: dict,
    budget: int,
    *,
    audit_limit: int = 8,
    selector=select_budgeted_frame,
) -> dict:
    queues = [deque() for _ in model.workloads]
    pending = iter(sorted(arrivals))
    next_arrival = next(pending, None)
    now = 0.0
    frame_count = 0
    total_attempts = total_probes = loaded_frames = exact_frames = 0
    total_threshold_evaluations = 0
    total_local_scores = total_feasible_candidates = peak_frontier = max_attempts = 0
    peak_jobs = 0
    backlog_sum = 0
    flows: list[float] = []
    last_completion = 0.0
    max_gap = max_epsilon = max_support_deficit = 0.0
    max_drift_violation = max_count_violation = 0.0
    selected_counts: dict[str, int] = {}
    audits = []
    rates = certificate["arrival_rates"]
    C = certificate["availability_loss_C"]
    B = certificate["second_moment_B"]
    eta = certificate["effective_slack_eta"]

    while now < arrival_horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            time, coordinate, index = next_arrival
            queues[coordinate].append((time, index))
            next_arrival = next(pending, None)
        before = tuple(len(queue) for queue in queues)
        # Only the released queues and static measured catalogue reach selection.
        selected = selector(model.options, before, model.lanes, budget)
        duration = selected.duration
        end = now + duration
        frame_count += 1
        total_attempts += selected.attempts
        total_probes += selected.bound_option_probes
        total_threshold_evaluations += selected.bound_threshold_evaluations
        total_local_scores += selected.local_score_evaluations
        total_feasible_candidates += selected.feasible_candidate_count
        peak_frontier = max(peak_frontier, selected.frontier_peak)
        max_attempts = max(max_attempts, selected.attempts)
        exact_frames += int(selected.exact)
        backlog_sum += sum(before)
        peak_jobs = max(peak_jobs, sum(before))
        max_gap = max(max_gap, selected.gap)
        max_epsilon = max(max_epsilon, selected.epsilon)
        frame_loaded = False
        departures = [0] * len(queues)
        for option_index in selected.option_indices:
            if option_index is None:
                continue
            option = model.options[option_index]
            frame_loaded |= option.loaded
            selected_counts[option.action_id] = selected_counts.get(option.action_id, 0) + 1
            for coordinate, offsets in enumerate(option.completion_offsets):
                for offset in offsets:
                    arrival, _ = queues[coordinate].popleft()
                    completion = now + offset
                    flows.append(completion - arrival)
                    last_completion = max(last_completion, completion)
                    departures[coordinate] += 1
        loaded_frames += int(frame_loaded)
        if tuple(departures) != selected.service:
            raise ValueError("selected service differs from executed job departures")
        full_support = model.lanes * max(
            (sum(q * n for q, n in zip(before, option.requirements, strict=True))
             / max(1.0, option.duration) for option in model.options),
            default=0.0,
        )
        max_support_deficit = max(max_support_deficit, full_support - selected.score)
        # The finite observation stops arrivals at H. Before H these are exactly
        # lambda*tau; afterwards they are zero, so the same upper drift bound holds.
        exposure = max(0.0, min(end, arrival_horizon) - now)
        means = tuple(rate * exposure for rate in rates)
        drift = conditional_lyapunov_drift(before, departures, means)
        drift_bound = B + duration * C - duration * eta * sum(before)
        max_drift_violation = max(max_drift_violation, drift - drift_bound)
        during = [0] * len(queues)
        while next_arrival is not None and next_arrival[0] <= end:
            time, coordinate, index = next_arrival
            queues[coordinate].append((time, index))
            during[coordinate] += 1
            next_arrival = next(pending, None)
        after = tuple(len(queue) for queue in queues)
        max_count_violation = max(max_count_violation, max(
            abs(q1 - (q0 - d + a))
            for q0, d, a, q1 in zip(before, departures, during, after, strict=True)
        ))
        if len(audits) < audit_limit:
            audits.append({
                "start_s": now, "duration_s": duration,
                "queue_before": before, "departures": departures,
                "arrivals_during_frame": during, "queue_after": after,
                "option_indices": selected.option_indices,
                "score": selected.score, "upper_bound": selected.upper_bound,
                "candidate_gap": selected.gap, "attempts": selected.attempts,
                "expected_drift": drift, "expected_drift_bound": drift_bound,
            })
        now = end
    ordered = sorted(flows)
    return {
        "budget": budget, "arrival_horizon_s": arrival_horizon,
        "elapsed_including_drain_s": now,
        "job_count": len(arrivals), "completed_jobs": len(flows),
        "makespan_s": last_completion,
        "mean_flow_s": mean(flows) if flows else None,
        "p90_flow_s": ordered[math.ceil(0.9 * len(ordered)) - 1] if ordered else None,
        "time_average_unfinished_jobs": sum(flows) / now,
        "frame_boundary_mean_backlog": backlog_sum / frame_count,
        "frame_boundary_peak_backlog": peak_jobs,
        "frame_count": frame_count, "loaded_frame_count": loaded_frames,
        "exact_selection_frame_count": exact_frames,
        "candidate_attempts": total_attempts,
        "mean_candidate_attempts": total_attempts / frame_count,
        "bound_option_probes": total_probes,
        "bound_threshold_evaluations": total_threshold_evaluations,
        "local_score_evaluations": total_local_scores,
        "feasible_candidate_evaluations": total_feasible_candidates,
        "peak_search_frontier": peak_frontier,
        "maximum_candidate_attempts_per_frame": max_attempts,
        "maximum_certified_candidate_gap": max_gap,
        "maximum_certified_gap_per_job": max_epsilon,
        "maximum_full_support_deficit": max_support_deficit,
        "maximum_expected_drift_bound_violation": max_drift_violation,
        "maximum_queue_recursion_violation": max_count_violation,
        "selected_option_counts": selected_counts,
        "frame_audit_prefix": audits,
    }


def scaling_probes(model: FrameModel, budgets: Sequence[int], *, selector=select_budgeted_frame) -> list[dict]:
    """Computational replication of the catalogue, not additional hardware evidence."""
    rows = []
    for lanes in (2, 4, 8):
        for budget in budgets:
            result = selector(model.options, (3, 2, 8), lanes, budget)
            rows.append({
                "lanes": lanes, "represented_configurations": (len(model.options) + 1) ** lanes,
                "queues": (3, 2, 8), "budget": budget, **asdict(result),
            })
    return rows


def build_experiment(
    *, horizon: float = 20_000.0,
    loads: Sequence[float] = (0.50, 0.80, 0.95),
    seeds: Sequence[int] = (41, 42, 43),
    budgets: Sequence[int] | None = None,
    selector=select_budgeted_frame,
) -> dict:
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("arrival observation horizon must be positive and finite")
    model = load_model()
    seed_cost = len(model.options) + 1
    exact_budget = exact_search_budget(len(model.options), model.lanes)
    selected_budgets = tuple(sorted(set(budgets or (seed_cost, seed_cost + 4, seed_cost + 16, exact_budget))))
    # Always include the same-family exact reference, accounted as its own run.
    selected_budgets = tuple(sorted(set((*selected_budgets, exact_budget))))
    certificates = []
    runs = []
    for load in loads:
        certificate = capacity_certificate(model, float(load))
        certificates.append(certificate)
        for seed in seeds:
            tape = poisson_arrivals(certificate["arrival_rates"], horizon, int(seed))
            for budget in selected_budgets:
                run = simulate_frames(model, tape, horizon, certificate, budget, selector=selector)
                runs.append({"load": float(load), "seed": int(seed), **run})
    comparisons = []
    for run in runs:
        reference = next(row for row in runs if row["load"] == run["load"] and row["seed"] == run["seed"] and row["budget"] == exact_budget)
        comparisons.append({
            "load": run["load"], "seed": run["seed"], "budget": run["budget"],
            "makespan_over_exact": run["makespan_s"] / reference["makespan_s"] if reference["makespan_s"] else None,
            "mean_flow_over_exact": run["mean_flow_s"] / reference["mean_flow_s"] if reference["mean_flow_s"] else None,
            "attempts_over_exact": run["candidate_attempts"] / reference["candidate_attempts"],
        })
    checks = {
        "all_mask_projection_certificates_hold": all(row["capacity_projection_passed"] for row in certificates),
        "all_loads_have_positive_same_model_slack": all(row["effective_slack_eta"] > 0 for row in certificates),
        "all_registered_jobs_completed": all(row["completed_jobs"] == row["job_count"] for row in runs),
        "same_frame_queue_recursion": all(row["maximum_queue_recursion_violation"] == 0 for row in runs),
        "conditional_drift_within_proved_bound": all(row["maximum_expected_drift_bound_violation"] <= 1e-8 for row in runs),
        "reference_search_is_exact_in_every_frame": all(row["exact_selection_frame_count"] == row["frame_count"] for row in runs if row["budget"] == exact_budget),
        "candidate_attempts_respect_budget": all(row["maximum_candidate_attempts_per_frame"] <= row["budget"] for row in runs),
    }
    return {
        "experiment": "aligned_hardware_frame_and_budget_v4", "schema_version": 1,
        "selector": "paid_seed_feasibility_weight_order_with_queue_capped_suffix_bounds",
        "evaluation_role": "development comparison on reused seeds; independent confirmation pending",
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "model": model.snapshot(),
        "protocol": {
            "arrival_horizon_s": horizon, "loads": list(loads), "seeds": list(seeds),
            "budgets": selected_budgets, "exact_reference_budget": exact_budget,
            "arrival_model": "independent Poisson arrivals per second; common tape across budgets",
            "job_sizes": "fixed task-native canonical units",
            "dispatch": "global synchronous frames; released jobs only; FIFO within class",
            "score": "queue weighted actual frame departures divided by actual frame duration",
            "budget_unit": "candidate construction or feasibility attempt; seeds included",
            "stopping_rule": "stop arrivals at horizon, then drain the same finite prefix",
        },
        "certificates": certificates, "runs": runs, "paired_budget_comparisons": comparisons,
        "catalogue_scaling_probes": scaling_probes(model, selected_budgets[:-1], selector=selector),
        "limitations": [
            "The certificate is for the deterministic cached phase model with stochastic arrivals; no new physical service confidence guarantee is inferred.",
            "Canonical job sizes, resident phase accounting and synchronized barriers define a new experiment; this does not certify the previous asynchronous lognormal-size replay.",
            "All-state drift uses the proved homogeneous-seed support bound. The trace maximum of the adaptive candidate gap is not substituted for a uniform theorem parameter.",
            "Four- and eight-lane catalogue replication measures candidate construction only, not measured performance on additional hardware.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=20_000.0)
    parser.add_argument("--loads", default="0.50,0.80,0.95")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_experiment(
        horizon=args.horizon,
        loads=tuple(float(value) for value in args.loads.split(",")),
        seeds=tuple(int(value) for value in args.seeds.split(",")),
        budgets=tuple(int(value) for value in args.budgets.split(",")) if args.budgets else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]), "checks": report["checks"], "output": str(args.output)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
