"""SOTA-style replay baselines built from measured Scheduleurm service curves.

These are policy-semantics baselines, not direct executions of external
scheduler binaries. Each policy uses the same exact service cache as the
candidate so the comparison isolates scheduling logic from profiling mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .defaults import calibrated_candidate_policy, calibrated_policy
from .fast_forward import ReplayComparison, ReplayPolicy, WorkloadSpec, compare_policies
from .service_cache import ServiceRateCache


@dataclass(frozen=True)
class SotaBaselineSpec:
    name: str
    policy: ReplayPolicy
    representative_systems: tuple[str, ...]
    coverage: tuple[str, ...]
    objective: str
    note: str

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": self.policy.snapshot(),
            "representative_systems": list(self.representative_systems),
            "coverage": list(self.coverage),
            "objective": self.objective,
            "note": self.note,
        }


def sota_throughput_table_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="sota_gavel_pollux_sia_table_goodput",
        calibrated=True,
        calibrated_objective="makespan",
    )


def sota_delay_oracle_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="sota_srpt_gittins_mean_flow_oracle",
        calibrated=True,
        calibrated_objective="mean_flow",
    )


def sota_interference_guard_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="sota_iadeep_salus_interference_guard",
        calibrated=True,
        calibrated_objective="makespan",
        max_makespan_regret=0.02,
        statewise_regret_slack=0.6,
        statewise=True,
        guarded_resource_kinds=("gpu_heavy", "hybrid_rl"),
        statewise_resource_kinds=("gpu_heavy", "hybrid_rl"),
    )


def sota_quadrant_composite_policy() -> ReplayPolicy:
    policy = calibrated_policy()
    return ReplayPolicy(
        name="sota_quadrant_composite",
        calibrated=policy.calibrated,
        calibrated_objective=policy.calibrated_objective,
        max_makespan_regret=policy.max_makespan_regret,
        statewise_regret_slack=policy.statewise_regret_slack,
        statewise=policy.statewise,
        guarded_resource_kinds=policy.guarded_resource_kinds,
        statewise_resource_kinds=policy.statewise_resource_kinds,
    )


def sota_baseline_specs() -> tuple[SotaBaselineSpec, ...]:
    return (
        SotaBaselineSpec(
            name="throughput_table_goodput",
            policy=sota_throughput_table_policy(),
            representative_systems=("Gavel", "Pollux", "Sia"),
            coverage=("q00_light_control", "q10_cpu_host_bound", "q11_cpu_gpu_coupled"),
            objective="measured throughput/goodput table; minimize all-task makespan proxy",
            note=(
                "Closest to table-driven goodput schedulers when the service table is "
                "already measured on the Scheduleurm fabric."
            ),
        ),
        SotaBaselineSpec(
            name="delay_oracle",
            policy=sota_delay_oracle_policy(),
            representative_systems=("SRPT", "Gittins", "SERPT"),
            coverage=("q00_light_control", "q01_gpu_bound_compute"),
            objective="minimize deterministic mean completion time proxy",
            note=(
                "A strong finite-batch delay baseline. It is not a throughput-optimal "
                "cluster baseline and may lose makespan."
            ),
        ),
        SotaBaselineSpec(
            name="interference_guard",
            policy=sota_interference_guard_policy(),
            representative_systems=("IADeep", "Salus"),
            coverage=("q01_gpu_bound_compute", "q11_cpu_gpu_coupled"),
            objective="interference-aware guarded co-location with statewise drain",
            note=(
                "Captures the GPU multiplexing/interference-control idea using exact "
                "Scheduleurm co-location profiles."
            ),
        ),
        SotaBaselineSpec(
            name="quadrant_composite",
            policy=sota_quadrant_composite_policy(),
            representative_systems=("Gavel/Pollux/Sia", "IADeep/Salus", "SRPT/Gittins"),
            coverage=(
                "q00_light_control",
                "q01_gpu_bound_compute",
                "q10_cpu_host_bound",
                "q11_cpu_gpu_coupled",
                "hybrid_research_portfolio",
            ),
            objective="per-quadrant composite: goodput where service dominates, guarded delay where interference dominates",
            note=(
                "This is the reviewer-facing SOTA-style composite when no single "
                "external scheduler covers all four Scheduleurm quadrants."
            ),
        ),
    )


def compare_against_sota_suite(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    *,
    candidate: ReplayPolicy | None = None,
    trials: int = 101,
    seed: int = 7,
) -> dict[str, Any]:
    candidate_policy = candidate or calibrated_candidate_policy(cache, specs)
    rows = []
    for baseline in sota_baseline_specs():
        comparison = compare_policies(
            cache,
            specs,
            baseline=baseline.policy,
            candidate=candidate_policy,
            trials=trials,
            seed=seed,
        )
        rows.append(_comparison_row(baseline, comparison))
    return {
        "candidate_policy": candidate_policy.snapshot(),
        "baselines": rows,
        "best_baseline_by_makespan": _best_baseline(rows, "baseline_total_makespan_s"),
        "best_baseline_by_mean_flow": _best_baseline(rows, "baseline_weighted_mean_flow_s"),
        "candidate_pareto_dominated_by": _pareto_dominators(rows),
        "candidate_not_pareto_dominated": not _pareto_dominators(rows),
        "candidate_beats_all_sota_style_makespan": all(
            row["candidate_vs_baseline_makespan"] >= 1.0 for row in rows
        ),
        "candidate_beats_all_sota_style_mean_flow": all(
            row["candidate_vs_baseline_mean_flow"] >= 1.0 for row in rows
        ),
    }


def _comparison_row(spec: SotaBaselineSpec, comparison: ReplayComparison) -> dict[str, Any]:
    baseline = comparison.baseline
    candidate = comparison.candidate
    return {
        "baseline": spec.snapshot(),
        "baseline_total_makespan_s": baseline.total_makespan_s,
        "baseline_weighted_mean_flow_s": baseline.weighted_mean_flow_s,
        "candidate_total_makespan_s": candidate.total_makespan_s,
        "candidate_weighted_mean_flow_s": candidate.weighted_mean_flow_s,
        "candidate_vs_baseline_makespan": comparison.makespan_improvement,
        "candidate_vs_baseline_mean_flow": comparison.mean_flow_improvement,
        "baseline_profiles": {
            row.workload_key: row.selected_profile for row in baseline.workloads
        },
        "candidate_profiles": {
            row.workload_key: row.selected_profile for row in candidate.workloads
        },
        "per_workload_improvements": comparison.per_workload_improvements,
    }


def _best_baseline(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    if not rows:
        return None
    row = min(rows, key=lambda item: float(item[metric]))
    return {
        "name": row["baseline"]["name"],
        "representative_systems": row["baseline"]["representative_systems"],
        metric: row[metric],
        "baseline_profiles": row["baseline_profiles"],
    }


def _pareto_dominators(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dominators = []
    for row in rows:
        ms = float(row["candidate_vs_baseline_makespan"])
        flow = float(row["candidate_vs_baseline_mean_flow"])
        if ms <= 1.0 and flow <= 1.0 and (ms < 1.0 or flow < 1.0):
            dominators.append(
                {
                    "name": row["baseline"]["name"],
                    "representative_systems": row["baseline"]["representative_systems"],
                    "candidate_vs_baseline_makespan": ms,
                    "candidate_vs_baseline_mean_flow": flow,
                    "baseline_profiles": row["baseline_profiles"],
                    "candidate_profiles": row["candidate_profiles"],
                }
            )
    return dominators
