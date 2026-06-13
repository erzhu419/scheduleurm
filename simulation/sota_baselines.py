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
from .service_cache import (
    ProfileRecord,
    ServiceRateCache,
    deterministic_makespan_s,
    deterministic_mean_flow_s,
)


PARETO_TOLERANCE = 0.005


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


@dataclass(frozen=True)
class SotaCandidateUnionPolicy(ReplayPolicy):
    """Policy over the union of Scheduleurm and SOTA-style candidate actions.

    This is intentionally still a replay policy, not an execution of the
    external systems.  Its role is to certify the stronger theorem-facing claim:
    if the robust candidate set explicitly contains the SOTA policy families'
    actions, the selector can evaluate those actions in the same measured
    service units instead of merely comparing against them after the fact.
    """

    policy_families: tuple[ReplayPolicy, ...] = ()
    selection_objective: str = "guarded_mean_flow"
    max_union_makespan_regret: float = 0.02
    scalarization_delay_weight: float = 0.35
    scalarization_uncertainty_weight: float = 0.05

    def select_profile(self, cache: ServiceRateCache, spec: WorkloadSpec) -> ProfileRecord:
        rows = self.candidate_rows(cache, spec)
        if not rows:
            raise KeyError(f"no union candidate rows for workload {spec.workload_key!r}")
        if self.selection_objective == "makespan":
            return min(
                rows,
                key=lambda row: (
                    row["makespan_s"],
                    row["mean_flow_s"],
                    row["profile"],
                ),
            )["record"]
        if self.selection_objective == "mean_flow":
            return min(
                rows,
                key=lambda row: (
                    row["mean_flow_s"],
                    row["makespan_s"],
                    row["profile"],
                ),
            )["record"]
        if self.selection_objective == "adaptive_scalarized":
            return min(
                rows,
                key=lambda row: (
                    self._adaptive_scalarized_loss(row, rows, spec),
                    row["mean_flow_s"],
                    row["makespan_s"],
                    row["profile"],
                ),
            )["record"]
        best_makespan = min(float(row["makespan_s"]) for row in rows)
        guard = best_makespan * (1.0 + max(0.0, float(self.max_union_makespan_regret)))
        guarded = [row for row in rows if float(row["makespan_s"]) <= guard] or rows
        return min(
            guarded,
            key=lambda row: (
                row["mean_flow_s"],
                row["makespan_s"],
                row["profile"],
            ),
        )["record"]

    def candidate_rows(self, cache: ServiceRateCache, spec: WorkloadSpec) -> list[dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        for family in self.policy_families:
            try:
                record = family.select_profile(cache, spec)
            except KeyError:
                continue
            if record.capacity_boundary or record.aggregate_rate <= 0:
                continue
            profile = int(record.profile)
            makespan = deterministic_makespan_s(
                task_count=spec.task_count,
                total_units=spec.total_units,
                resource_count=spec.resource_count,
                profile=profile,
                aggregate_rate=float(record.aggregate_rate),
            )
            mean_flow = deterministic_mean_flow_s(
                task_count=spec.task_count,
                total_units=spec.total_units,
                resource_count=spec.resource_count,
                profile=profile,
                aggregate_rate=float(record.aggregate_rate),
            )
            existing = rows.get(profile)
            provenance = [family.name]
            if existing is not None:
                provenance = sorted(set(existing["provenance"]) | {family.name})
            rows[profile] = {
                "record": record,
                "workload_key": spec.workload_key,
                "profile": profile,
                "makespan_s": float(makespan),
                "mean_flow_s": float(mean_flow),
                "provenance": provenance,
            }
        return sorted(rows.values(), key=lambda row: row["profile"])

    def _adaptive_scalarized_loss(
        self,
        row: dict[str, Any],
        rows: list[dict[str, Any]],
        spec: WorkloadSpec,
    ) -> float:
        """Backlog/load-aware scalar loss for a single Pareto-facing policy.

        The loss is dimensionless.  It uses the best available makespan and
        mean-flow rows as scales, then increases the makespan weight under high
        backlog and increases the delay weight under small queues.  The
        uncertainty term is a bounded penalty for rows supported by fewer policy
        families, which is the replay analogue of the theorem's bounded
        implementation penalty.
        """

        best_makespan = min(float(item["makespan_s"]) for item in rows)
        best_flow = min(float(item["mean_flow_s"]) for item in rows)
        backlog = max(1.0, float(spec.task_count))
        pressure = min(1.0, backlog / 64.0)
        delay_weight = max(0.05, float(self.scalarization_delay_weight)) * (1.0 - 0.65 * pressure)
        makespan_weight = 1.0 - delay_weight
        provenance_count = max(1, len(row.get("provenance") or ()))
        max_provenance = max(1, max(len(item.get("provenance") or ()) for item in rows))
        uncertainty = 1.0 - float(provenance_count) / float(max_provenance)
        return (
            makespan_weight * float(row["makespan_s"]) / max(1e-12, best_makespan)
            + delay_weight * float(row["mean_flow_s"]) / max(1e-12, best_flow)
            + float(self.scalarization_uncertainty_weight) * uncertainty
        )

    def snapshot(self) -> dict[str, Any]:
        out = super().snapshot()
        out.update(
            {
                "policy_families": [policy.snapshot() for policy in self.policy_families],
                "selection_objective": self.selection_objective,
                "max_union_makespan_regret": self.max_union_makespan_regret,
                "scalarization_delay_weight": self.scalarization_delay_weight,
                "scalarization_uncertainty_weight": self.scalarization_uncertainty_weight,
                "candidate_set_semantics": "scheduleurm_plus_sota_policy_family_union",
            }
        )
        return out


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
        guarded_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl"),
        statewise_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl"),
    )


def sota_finish_time_fairness_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="sota_gavel_finish_time_fairness",
        calibrated=True,
        calibrated_objective="guarded_mean_flow",
        max_makespan_regret=0.01,
        min_makespan_regret=0.0,
        statewise_regret_slack=0.25,
        statewise=True,
        guarded_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl"),
        statewise_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl"),
        backlog_aware_guard=True,
        backlog_reference_tasks=32,
        statewise_service_dominance_guard=True,
    )


def sota_sia_pollux_resource_adaptive_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="sota_sia_pollux_resource_adaptive",
        calibrated=True,
        calibrated_objective="guarded_mean_flow",
        max_makespan_regret=0.06,
        min_makespan_regret=0.0,
        statewise_regret_slack=0.8,
        statewise=True,
        guarded_resource_kinds=(
            "gpu_heavy",
            "gpu_cnn",
            "gpu_llm",
            "hybrid_rl",
            "cpu_heavy",
            "cpu_sumo_transit",
        ),
        statewise_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl"),
        backlog_aware_guard=True,
        backlog_reference_tasks=96,
        statewise_service_dominance_guard=True,
    )


def sota_packing_guard_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="sota_salus_iadeep_packing_guard",
        calibrated=True,
        calibrated_objective="makespan",
        max_makespan_regret=0.0,
        statewise_regret_slack=0.0,
        statewise=True,
        guarded_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl"),
        statewise_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl"),
        statewise_service_dominance_guard=True,
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
            coverage=(
                "q00_light_control",
                "q01_gpu_bound_compute",
                "q01_gpu_bound_cnn_resnet50",
                "q01_gpu_bound_llm_inference",
                "q10_cpu_host_bound",
                "q11_cpu_gpu_coupled",
            ),
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
            coverage=(
                "q00_light_control",
                "q01_gpu_bound_compute",
                "q01_gpu_bound_cnn_resnet50",
                "q01_gpu_bound_llm_inference",
                "q01_gpu_model_portfolio",
            ),
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
            coverage=(
                "q01_gpu_bound_compute",
                "q01_gpu_bound_cnn_resnet50",
                "q01_gpu_bound_llm_inference",
                "q01_gpu_model_portfolio",
                "q11_cpu_gpu_coupled",
            ),
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
                "q01_gpu_bound_cnn_resnet50",
                "q01_gpu_bound_llm_inference",
                "q01_gpu_model_portfolio",
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
        SotaBaselineSpec(
            name="finish_time_fairness",
            policy=sota_finish_time_fairness_policy(),
            representative_systems=("Gavel", "Themis"),
            coverage=(
                "q01_gpu_bound_compute",
                "q01_gpu_bound_cnn_resnet50",
                "q01_gpu_bound_llm_inference",
                "q01_gpu_model_portfolio",
                "q11_cpu_gpu_coupled",
                "hybrid_research_portfolio",
            ),
            objective="finish-time fairness with a tight measured makespan guard",
            note=(
                "Adds a fairness-oriented SOTA family to the candidate union. "
                "It is still evaluated as policy semantics on Scheduleurm's cache."
            ),
        ),
        SotaBaselineSpec(
            name="resource_adaptive_goodput",
            policy=sota_sia_pollux_resource_adaptive_policy(),
            representative_systems=("Sia", "Pollux"),
            coverage=(
                "q00_light_control",
                "q01_gpu_bound_compute",
                "q01_gpu_bound_cnn_resnet50",
                "q01_gpu_bound_llm_inference",
                "q01_gpu_model_portfolio",
                "q10_cpu_host_bound",
                "q11_cpu_gpu_coupled",
                "hybrid_research_portfolio",
            ),
            objective="resource-adaptive goodput with backlog-aware guarded profile selection",
            note=(
                "Represents Sia/Pollux-style adaptive goodput as an additional "
                "finite measured-cache action family."
            ),
        ),
        SotaBaselineSpec(
            name="packing_guard",
            policy=sota_packing_guard_policy(),
            representative_systems=("Salus", "IADeep", "Gandiva"),
            coverage=(
                "q01_gpu_bound_compute",
                "q01_gpu_bound_cnn_resnet50",
                "q01_gpu_bound_llm_inference",
                "q01_gpu_model_portfolio",
                "q11_cpu_gpu_coupled",
            ),
            objective="conservative packing/interference guard",
            note=(
                "Represents strict sharing-control policies that only admit "
                "measured co-location when service does not degrade."
            ),
        ),
    )


def sota_candidate_union_policy(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    *,
    selection_objective: str = "guarded_mean_flow",
    max_union_makespan_regret: float = 0.02,
) -> SotaCandidateUnionPolicy:
    """Build the theorem-facing action-union policy for replay experiments."""

    candidate = calibrated_candidate_policy(cache, specs)
    families = (candidate, *(spec.policy for spec in sota_baseline_specs()))
    return SotaCandidateUnionPolicy(
        name=f"scheduleurm_sota_union_{selection_objective}",
        calibrated=False,
        statewise=True,
        guarded_resource_kinds=(
            "gpu_heavy",
            "gpu_cnn",
            "gpu_llm",
            "hybrid_rl",
            "cpu_heavy",
            "cpu_sumo_transit",
            "light_control",
        ),
        statewise_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl"),
        backlog_aware_guard=True,
        backlog_reference_tasks=64,
        statewise_service_dominance_guard=True,
        policy_families=families,
        selection_objective=selection_objective,
        max_union_makespan_regret=max_union_makespan_regret,
    )


def sota_candidate_union_policies(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
) -> tuple[SotaCandidateUnionPolicy, ...]:
    return (
        sota_candidate_union_policy(cache, specs, selection_objective="guarded_mean_flow"),
        sota_candidate_union_policy(cache, specs, selection_objective="adaptive_scalarized"),
        sota_candidate_union_policy(cache, specs, selection_objective="makespan"),
        sota_candidate_union_policy(cache, specs, selection_objective="mean_flow"),
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
        "candidate_action_union": _union_summary(
            cache,
            specs,
            candidate_policy=candidate_policy,
        ),
    }


def _union_summary(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    *,
    candidate_policy: ReplayPolicy,
) -> dict[str, Any]:
    policies = [
        sota_candidate_union_policy(
            cache,
            specs,
            selection_objective=objective,
        )
        for objective in ("guarded_mean_flow", "adaptive_scalarized", "makespan", "mean_flow")
    ]
    rows = []
    for policy in policies:
        comparison = compare_policies(
            cache,
            specs,
            baseline=candidate_policy,
            candidate=policy,
            trials=31,
            seed=29,
        )
        rows.append(
            {
                "policy": policy.snapshot(),
                "candidate_vs_union_makespan": comparison.makespan_improvement,
                "candidate_vs_union_mean_flow": comparison.mean_flow_improvement,
                "union_vs_candidate_makespan": (
                    1.0 / comparison.makespan_improvement
                    if comparison.makespan_improvement > 0 else 0.0
                ),
                "union_vs_candidate_mean_flow": (
                    1.0 / comparison.mean_flow_improvement
                    if comparison.mean_flow_improvement > 0 else 0.0
                ),
                "per_workload": comparison.per_workload_improvements,
            }
        )
    return {
        "semantics": (
            "Replay of policies whose candidate set explicitly contains the "
            "Scheduleurm candidate action and all SOTA-style policy-family "
            "actions under the same measured service cache."
        ),
        "rows": rows,
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
    floor = 1.0 - PARETO_TOLERANCE
    for row in rows:
        ms = float(row["candidate_vs_baseline_makespan"])
        flow = float(row["candidate_vs_baseline_mean_flow"])
        if ms < floor and flow < floor:
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
