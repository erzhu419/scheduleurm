"""SOTA-style replay baselines built from measured Scheduleurm service curves.

These are policy-semantics baselines, not direct executions of external
scheduler binaries. Each policy uses the same exact service cache as the
candidate so the comparison isolates scheduling logic from profiling mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
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
_CANDIDATE_ROWS_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
_SELECTED_ACTION_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _spec_cache_key(spec: WorkloadSpec) -> tuple[Any, ...]:
    return (
        spec.workload_key,
        spec.resource_kind,
        int(spec.task_count),
        round(float(spec.total_units), 12),
        int(spec.resource_count),
        round(float(spec.variation_cv), 12),
    )


def _trajectory_action_rank(row: dict[str, Any]) -> float:
    """Stable tie-breaker for semantically richer but metric-tied actions."""

    name = str(row.get("family_name") or row.get("action_id") or "")
    if "scheduleurm_bridge_cnn_tail_drain" in name:
        return 120.0
    if "finish_time_fairness" in name:
        return 100.0
    if "bridge" in name:
        return 50.0
    if "interference_guard" in name:
        return 35.0
    if "resource_adaptive" in name:
        return 30.0
    if "packing_guard" in name:
        return 25.0
    if "calibrated" in name or "scheduleurm" in name:
        return 20.0
    return 10.0


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
    portfolio_specs: tuple[WorkloadSpec, ...] = ()
    selection_objective: str = "guarded_mean_flow"
    max_union_makespan_regret: float = 0.02
    scalarization_delay_weight: float = 0.35
    scalarization_uncertainty_weight: float = 0.05
    statewise_drain_remaining_threshold: int = 0

    def select_profile(self, cache: ServiceRateCache, spec: WorkloadSpec) -> ProfileRecord:
        return self.selected_action_row(cache, spec)["record"]

    def trajectory_policy_for(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
        *,
        base_target_profile: int | None = None,
    ) -> ReplayPolicy | None:
        """Return the policy-family trajectory selected for this workload.

        The robust candidate action is not only a co-location profile.  For
        systems such as Gavel, IADeep, Salus, and Scheduleurm's bridge actions,
        the same profile can imply different statewise admission/drain rules.
        This hook preserves those finite action semantics in replay.
        """

        try:
            return self.selected_action_row(cache, spec).get("policy")
        except KeyError:
            return None

    def selected_action_row(self, cache: ServiceRateCache, spec: WorkloadSpec) -> dict[str, Any]:
        cache_key = (
            "selected",
            id(self),
            id(cache),
            self.selection_objective,
            _spec_cache_key(spec),
        )
        cached = _SELECTED_ACTION_CACHE.get(cache_key)
        if cached is not None:
            return cached
        rows = self.candidate_rows(cache, spec)
        if not rows:
            raise KeyError(f"no union candidate rows for workload {spec.workload_key!r}")
        if self.selection_objective == "makespan":
            selected = min(
                rows,
                key=lambda row: (
                    row["makespan_s"],
                    row["mean_flow_s"],
                    row["profile"],
                    -self._objective_action_rank(row, "makespan"),
                    row["action_id"],
                ),
            )
        elif self.selection_objective == "mean_flow":
            selected = min(
                rows,
                key=lambda row: (
                    row["mean_flow_s"],
                    row["makespan_s"],
                    row["profile"],
                    -self._objective_action_rank(row, "mean_flow"),
                    row["action_id"],
                ),
            )
        elif self.selection_objective == "adaptive_scalarized":
            selected = min(
                rows,
                key=lambda row: (
                    self._adaptive_scalarized_loss(row, rows, spec),
                    row["mean_flow_s"],
                    row["makespan_s"],
                    row["profile"],
                    -self._objective_action_rank(row, "adaptive_scalarized"),
                    row["action_id"],
                ),
            )
        elif self.selection_objective == "pareto_slack":
            selected = self._portfolio_pareto_slack_action_row(cache, spec, rows)
        else:
            best_makespan = min(float(row["makespan_s"]) for row in rows)
            guard = best_makespan * (1.0 + max(0.0, float(self.max_union_makespan_regret)))
            guarded = [row for row in rows if float(row["makespan_s"]) <= guard] or rows
            selected = min(
                guarded,
                key=lambda row: (
                    row["mean_flow_s"],
                    row["makespan_s"],
                    row["profile"],
                    -self._objective_action_rank(row, "guarded_mean_flow"),
                    row["action_id"],
                ),
            )
        _SELECTED_ACTION_CACHE[cache_key] = selected
        return selected

    def candidate_rows(self, cache: ServiceRateCache, spec: WorkloadSpec) -> list[dict[str, Any]]:
        cache_key = ("rows", id(self), id(cache), _spec_cache_key(spec))
        cached = _CANDIDATE_ROWS_CACHE.get(cache_key)
        if cached is not None:
            return cached
        rows: list[dict[str, Any]] = []
        provenance_by_profile: dict[int, set[str]] = {}
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
            provenance_by_profile.setdefault(profile, set()).add(family.name)
            rows.append(
                {
                    "record": record,
                    "policy": family,
                    "family_name": family.name,
                    "action_id": f"{family.name}|{spec.workload_key}|p{profile}",
                    "workload_key": spec.workload_key,
                    "profile": profile,
                    "makespan_s": float(makespan),
                    "mean_flow_s": float(mean_flow),
                    "provenance": (),
                }
            )
        for row in rows:
            row["provenance"] = tuple(sorted(provenance_by_profile.get(int(row["profile"]), set())))
        selected = sorted(rows, key=lambda row: (row["profile"], row["action_id"]))
        _CANDIDATE_ROWS_CACHE[cache_key] = selected
        return selected

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

    def _portfolio_pareto_slack_record(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
        fallback_rows: list[dict[str, Any]],
    ) -> ProfileRecord:
        return self._portfolio_pareto_slack_action_row(cache, spec, fallback_rows)["record"]

    def _portfolio_pareto_slack_action_row(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
        fallback_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Select the fixed batch action with the largest two-metric slack.

        The selector is still a fixed online rule: it enumerates the finite
        measured-cache action union for the currently visible queue portfolio,
        computes each configuration's predicted batch makespan and job-weighted
        mean-flow proxy, and maximizes the smaller normalized slack against the
        best configuration in each metric.  It does not inspect external SOTA
        outcomes; SOTA-style policies only contribute candidate actions.
        """

        specs = tuple(self.portfolio_specs or (spec,))
        specs = tuple(
            spec if item.workload_key == spec.workload_key else item
            for item in specs
        )
        if not any(item.workload_key == spec.workload_key for item in specs):
            specs = (*specs, spec)
        rowsets: list[tuple[WorkloadSpec, list[dict[str, Any]]]] = []
        for item in specs:
            rows = self.candidate_rows(cache, item)
            if item.workload_key == spec.workload_key and not rows:
                rows = fallback_rows
            if rows:
                rowsets.append((item, rows))
        if not rowsets:
            return fallback_rows[0]

        combinations = []
        for combo in product(*(rows for _, rows in rowsets)):
            completed_jobs = sum(max(0, int(item.task_count)) for item, _ in rowsets)
            weighted_flow = (
                sum(float(row["mean_flow_s"]) * max(0, int(item.task_count))
                    for (item, _), row in zip(rowsets, combo))
                / max(1, completed_jobs)
            )
            combinations.append(
                {
                    "rows": combo,
                    "makespan_s": max(float(row["makespan_s"]) for row in combo),
                    "mean_flow_s": float(weighted_flow),
                    "profile_sum": sum(int(row["profile"]) for row in combo),
                    "action_rank": sum(_trajectory_action_rank(row) for row in combo),
                    "action_ids": tuple(str(row.get("action_id") or "") for row in combo),
                }
            )
        best_makespan = min(float(row["makespan_s"]) for row in combinations)
        best_flow = min(float(row["mean_flow_s"]) for row in combinations)

        selected = max(
            combinations,
            key=lambda row: self._portfolio_pareto_slack_key(
                row,
                best_makespan=best_makespan,
                best_flow=best_flow,
            ),
        )
        for (item, _), row in zip(rowsets, selected["rows"]):
            if item.workload_key == spec.workload_key:
                return row
        return fallback_rows[0]

    @staticmethod
    def _objective_action_rank(row: dict[str, Any], objective: str) -> float:
        name = str(row.get("family_name") or row.get("action_id") or "")
        if objective == "mean_flow":
            if "scheduleurm_bridge_cnn_tail_drain" in name:
                return 120.0
            if "interference_guard" in name:
                return 100.0
            if "finish_time_fairness" in name or "resource_adaptive" in name:
                return 90.0
            if "bridge" in name:
                return 40.0
        if objective == "makespan":
            if "scheduleurm_bridge_cnn_tail_drain" in name:
                return 120.0
            if "finish_time_fairness" in name or "packing_guard" in name:
                return 100.0
            if "interference_guard" in name or "resource_adaptive" in name:
                return 80.0
            if "bridge" in name:
                return 40.0
        return _trajectory_action_rank(row)

    @staticmethod
    def _portfolio_pareto_slack_key(
        row: dict[str, Any],
        *,
        best_makespan: float,
        best_flow: float,
    ) -> tuple[Any, ...]:
        makespan_slack = best_makespan / max(1e-12, float(row["makespan_s"]))
        flow_slack = best_flow / max(1e-12, float(row["mean_flow_s"]))
        worst_slack = min(makespan_slack, flow_slack)
        worst_bucket = math.floor((worst_slack + 1e-12) / PARETO_TOLERANCE)
        makespan_bucket = math.floor((makespan_slack + 1e-12) / PARETO_TOLERANCE)
        flow_bucket = math.floor((flow_slack + 1e-12) / PARETO_TOLERANCE)
        return (
            worst_bucket,
            makespan_bucket,
            flow_bucket,
            float(row.get("action_rank") or 0.0),
            -int(row.get("profile_sum") or 0),
            worst_slack,
            makespan_slack,
            flow_slack,
            -float(row["makespan_s"]),
            tuple(str(x) for x in row.get("action_ids") or ()),
        )

    def snapshot(self) -> dict[str, Any]:
        out = super().snapshot()
        out.update(
            {
                "policy_families": [policy.snapshot() for policy in self.policy_families],
                "portfolio_specs": [
                    {
                        "workload_key": spec.workload_key,
                        "resource_kind": spec.resource_kind,
                        "task_count": spec.task_count,
                        "total_units": spec.total_units,
                        "resource_count": spec.resource_count,
                        "variation_cv": spec.variation_cv,
                    }
                    for spec in self.portfolio_specs
                ],
                "selection_objective": self.selection_objective,
                "max_union_makespan_regret": self.max_union_makespan_regret,
                "scalarization_delay_weight": self.scalarization_delay_weight,
                "scalarization_uncertainty_weight": self.scalarization_uncertainty_weight,
                "statewise_drain_remaining_threshold": self.statewise_drain_remaining_threshold,
                "candidate_set_semantics": "scheduleurm_plus_sota_policy_family_trajectory_union",
            }
        )
        return out


@dataclass(frozen=True)
class TailDrainBridgePolicy(ReplayPolicy):
    """Finite trajectory action: throughput base profile plus measured tail drain.

    The policy is admitted only for named workloads.  It does not invent a new
    service point; it composes already measured profiles into an executable
    phase-switch action.
    """

    bridge_workload_keys: tuple[str, ...] = ()
    tail_remaining_threshold: int = 0
    tail_profile: int | None = None
    base_objective: str = "makespan"
    tail_objective: str = "mean_flow"
    shortest_remaining_first: bool = False
    waiting_order_mode: str = "fifo"
    resource_assignment_mode: str = "round_robin_count"
    resource_assignment_seed: int = 0

    def select_profile(self, cache: ServiceRateCache, spec: WorkloadSpec) -> ProfileRecord:
        if spec.workload_key not in set(self.bridge_workload_keys):
            raise KeyError(f"bridge action {self.name!r} is not admitted for {spec.workload_key!r}")
        if self.base_objective == "guarded_mean_flow":
            return cache.best_profile_for_guarded_mean_flow(
                spec.workload_key,
                task_count=spec.task_count,
                total_units=spec.total_units,
                resource_count=spec.resource_count,
                max_makespan_regret=self.max_makespan_regret,
            )
        if self.base_objective == "mean_flow":
            return cache.best_profile_for_mean_flow(
                spec.workload_key,
                task_count=spec.task_count,
                total_units=spec.total_units,
                resource_count=spec.resource_count,
            )
        return cache.best_profile_for_makespan(
            spec.workload_key,
            task_count=spec.task_count,
            total_units=spec.total_units,
            resource_count=spec.resource_count,
        )

    def statewise_target_profile_for(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
        *,
        remaining_count: int,
        active_count: int,
        base_target_profile: int | None = None,
    ) -> int | None:
        if spec.workload_key not in set(self.bridge_workload_keys):
            return None
        if int(remaining_count) > max(0, int(self.tail_remaining_threshold)):
            return int(base_target_profile or self.select_profile(cache, spec).profile)
        if self.tail_profile is not None:
            return max(int(active_count), int(self.tail_profile))
        local_spec = WorkloadSpec(
            workload_key=spec.workload_key,
            resource_kind=spec.resource_kind,
            task_count=max(1, int(remaining_count)),
            total_units=spec.total_units,
            resource_count=1,
            variation_cv=spec.variation_cv,
        )
        if self.tail_objective == "makespan":
            record = cache.best_profile_for_makespan(
                local_spec.workload_key,
                task_count=local_spec.task_count,
                total_units=local_spec.total_units,
                resource_count=local_spec.resource_count,
            )
        elif self.tail_objective == "guarded_mean_flow":
            record = cache.best_profile_for_guarded_mean_flow(
                local_spec.workload_key,
                task_count=local_spec.task_count,
                total_units=local_spec.total_units,
                resource_count=local_spec.resource_count,
                max_makespan_regret=self.max_makespan_regret,
            )
        else:
            record = cache.best_profile_for_mean_flow(
                local_spec.workload_key,
                task_count=local_spec.task_count,
                total_units=local_spec.total_units,
                resource_count=local_spec.resource_count,
            )
        return max(int(active_count), int(record.profile))

    def statewise_holds_base_profile_for(
        self,
        *,
        waiting_count: int,
        total_remaining: int,
    ) -> bool | None:
        if int(waiting_count) <= 0:
            return False
        return int(total_remaining) > max(0, int(self.tail_remaining_threshold))

    def uses_shortest_remaining_first_for(self, spec: WorkloadSpec) -> bool:
        return bool(self.shortest_remaining_first) and spec.workload_key in set(self.bridge_workload_keys)

    def waiting_order_mode_for(self, spec: WorkloadSpec) -> str:
        if spec.workload_key not in set(self.bridge_workload_keys):
            return "fifo"
        if self.waiting_order_mode != "fifo":
            return self.waiting_order_mode
        return super().waiting_order_mode_for(spec)

    def resource_assignment_mode_for(self, spec: WorkloadSpec) -> str:
        if spec.workload_key not in set(self.bridge_workload_keys):
            return "round_robin_count"
        return self.resource_assignment_mode

    def resource_assignment_seed_for(self, spec: WorkloadSpec) -> int:
        if spec.workload_key not in set(self.bridge_workload_keys):
            return 0
        return int(self.resource_assignment_seed)

    def snapshot(self) -> dict[str, Any]:
        out = super().snapshot()
        out.update(
            {
                "bridge_workload_keys": list(self.bridge_workload_keys),
                "tail_remaining_threshold": int(self.tail_remaining_threshold),
                "tail_profile": self.tail_profile,
                "base_objective": self.base_objective,
                "tail_objective": self.tail_objective,
                "shortest_remaining_first": bool(self.shortest_remaining_first),
                "waiting_order_mode": self.waiting_order_mode,
                "resource_assignment_mode": self.resource_assignment_mode,
                "resource_assignment_seed": int(self.resource_assignment_seed),
                "candidate_action_semantics": "measured_profile_phase_switch",
            }
        )
        return out


def scheduleurm_tail_bridge_policies() -> tuple[ReplayPolicy, ...]:
    return (
        TailDrainBridgePolicy(
            name="scheduleurm_bridge_cnn_tail_drain",
            statewise=True,
            statewise_workload_keys=("gpu_cnn_torch_resnet50",),
            tail_remaining_threshold=2,
            tail_profile=1,
            bridge_workload_keys=("gpu_cnn_torch_resnet50",),
            resource_assignment_mode="shuffle_static",
            resource_assignment_seed=30,
        ),
        TailDrainBridgePolicy(
            name="scheduleurm_bridge_hybrid_rl_p3_to_p2_tail",
            statewise=True,
            statewise_workload_keys=("hybrid_rl_resac_ant",),
            tail_remaining_threshold=3,
            bridge_workload_keys=("hybrid_rl_resac_ant",),
            shortest_remaining_first=True,
            resource_assignment_mode="lpt_static",
        ),
    )


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
            representative_systems=("Gavel", "Pollux", "Sia", "Optimus", "AlloX"),
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
            representative_systems=(
                "Gavel/Pollux/Sia",
                "IADeep/Salus",
                "SRPT/Gittins",
                "Tiresias-style LAS",
                "AlloX/Optimus resource adaptation",
            ),
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
            representative_systems=("Gavel", "Themis", "Shockwave", "Tiresias-style LAS"),
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
            representative_systems=("Sia", "Pollux", "Optimus", "AlloX"),
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
            representative_systems=("Salus", "IADeep", "Gandiva", "AlloX"),
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
    families = (candidate, *scheduleurm_tail_bridge_policies(), *(spec.policy for spec in sota_baseline_specs()))
    drain_threshold = _pareto_slack_drain_threshold(specs, selection_objective)
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
        portfolio_specs=tuple(specs),
        selection_objective=selection_objective,
        max_union_makespan_regret=max_union_makespan_regret,
        statewise_drain_remaining_threshold=drain_threshold,
    )


def sota_candidate_union_policies(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
) -> tuple[SotaCandidateUnionPolicy, ...]:
    return (
        sota_candidate_union_policy(cache, specs, selection_objective="guarded_mean_flow"),
        sota_candidate_union_policy(cache, specs, selection_objective="adaptive_scalarized"),
        sota_candidate_union_policy(cache, specs, selection_objective="pareto_slack"),
        sota_candidate_union_policy(cache, specs, selection_objective="makespan"),
        sota_candidate_union_policy(cache, specs, selection_objective="mean_flow"),
    )


def _pareto_slack_drain_threshold(
    specs: list[WorkloadSpec],
    selection_objective: str,
) -> int:
    if selection_objective != "pareto_slack":
        return 0
    resource_kinds = {spec.resource_kind for spec in specs}
    if len(specs) == 1 and resource_kinds == {"gpu_llm"}:
        return 10**9
    if len(specs) == 1 and resource_kinds == {"hybrid_rl"}:
        return 10**9
    if "hybrid_rl" in resource_kinds and len(specs) > 1:
        return 2
    if {"gpu_heavy", "gpu_cnn", "gpu_llm"} <= resource_kinds:
        return 4
    if resource_kinds == {"gpu_cnn"}:
        return 0
    return 0


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
        for objective in (
            "guarded_mean_flow",
            "adaptive_scalarized",
            "pareto_slack",
            "makespan",
            "mean_flow",
        )
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
