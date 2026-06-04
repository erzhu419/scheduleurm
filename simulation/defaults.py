"""Default trace-cache assembly for Scheduleurm replay experiments."""
from __future__ import annotations

from itertools import product
from pathlib import Path

from .fast_forward import ReplayPolicy, WorkloadSpec
from .service_cache import (
    ProfileRecord,
    ServiceRateCache,
    add_protocol_cpu_curve,
    deterministic_makespan_s,
    deterministic_mean_flow_s,
    records_from_summary_file,
)
from .tasksets import empirical_replay_specs


RUN_ROOT = Path("/home/erzhu419/.claude/scheduler/experiments/runs")


def build_default_cache() -> ServiceRateCache:
    cache = ServiceRateCache()
    _add_summary_dir(
        cache,
        RUN_ROOT / "module12_resac_ant_real_dense_jtl110gpu2_gpu1_20260603_002" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module12_resac_ant_real_dense_jtl110gpu2_gpu1_20260603_003" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module6_service_curve_jtl110gpu2_size8192_20260603_001" / "reports",
        workload_key="gpu_heavy_jax_matmul",
        command_fingerprint="jax_matmul_size8192_v1",
        resource_kind="gpu_heavy",
        total_units=2400,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module22_q01_gpu_heavy_jax8192_profiles4_8_20260604_001" / "reports",
        workload_key="gpu_heavy_jax_matmul",
        command_fingerprint="jax_matmul_size8192_v1",
        resource_kind="gpu_heavy",
        total_units=2400,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module23_q00_light_control_local_profiles1_16_20260604_001" / "reports",
        workload_key="light_control_local",
        command_fingerprint="cpu_light_sleep_20ms_v1",
        resource_kind="light_control",
        total_units=10000,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module23_q00_light_control_local_profile16_boundary_20260604_001" / "reports",
        workload_key="light_control_local",
        command_fingerprint="cpu_light_sleep_20ms_v1",
        resource_kind="light_control",
        total_units=10000,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module24_q00_light_control_local_profiles3_7_20260604_001" / "reports",
        workload_key="light_control_local",
        command_fingerprint="cpu_light_sleep_20ms_v1",
        resource_kind="light_control",
        total_units=10000,
        node_bucket="local:cpu",
    )
    add_protocol_cpu_curve(
        cache,
        workload_key="cpu_heavy_protocol",
        total_units=3600,
        max_workers=32,
        per_worker_rate=1.0,
        saturation_workers=16,
    )
    return cache


def default_workload_specs() -> list[WorkloadSpec]:
    return empirical_replay_specs()


def legacy_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="legacy_fixed_caps",
        fixed_profiles={
            "hybrid_rl_resac_ant": 5,
            "gpu_heavy_jax_matmul": 3,
            "light_control_local": 1,
            "cpu_heavy_protocol": 32,
        },
    )


def calibrated_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="calibrated_guarded_knee",
        calibrated=True,
        calibrated_objective="guarded_mean_flow",
        max_makespan_regret=0.02,
    )


def calibrated_delay_statewise_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="calibrated_delay_statewise",
        calibrated=True,
        calibrated_objective="makespan",
        max_makespan_regret=0.02,
        statewise_regret_slack=0.6,
        statewise=True,
        guarded_resource_kinds=("gpu_heavy",),
        statewise_resource_kinds=("gpu_heavy",),
    )


def calibrated_makespan_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="calibrated_fast_forward_makespan",
        calibrated=True,
        calibrated_objective="makespan",
    )


def calibrated_candidate_policy(cache: ServiceRateCache, specs: list[WorkloadSpec]) -> ReplayPolicy:
    if len(specs) <= 1:
        return calibrated_policy()
    return calibrated_global_guarded_policy(cache, specs)


def calibrated_global_guarded_policy(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    *,
    max_global_makespan_regret: float = 0.02,
) -> ReplayPolicy:
    """Choose fixed profiles by global makespan guard plus weighted flow.

    This is the multi-workload replay analogue of the robust MaxWeight guard:
    first preserve the portfolio support/makespan objective, then spend the
    available slack on finite-batch delay.
    """

    rows_by_key = [_profile_rows(cache, spec) for spec in specs]
    if not rows_by_key:
        return calibrated_policy()
    best_global = min(max(row["makespan_s"] for row in combo) for combo in product(*rows_by_key))
    guard = best_global * (1.0 + max(0.0, float(max_global_makespan_regret)))
    admissible = [
        combo for combo in product(*rows_by_key)
        if max(row["makespan_s"] for row in combo) <= guard
    ]
    if not admissible:
        admissible = list(product(*rows_by_key))
    total_tasks = max(1, sum(max(0, int(spec.task_count)) for spec in specs))

    def combo_key(combo: tuple[dict, ...]) -> tuple[float, float, tuple[int, ...]]:
        weighted_flow = sum(
            row["mean_flow_s"] * max(0, int(row["task_count"]))
            for row in combo
        ) / float(total_tasks)
        global_makespan = max(row["makespan_s"] for row in combo)
        profiles = tuple(int(row["profile"]) for row in combo)
        return (weighted_flow, global_makespan, profiles)

    chosen = min(admissible, key=combo_key)
    return ReplayPolicy(
        name="calibrated_global_guarded",
        fixed_profiles={str(row["workload_key"]): int(row["profile"]) for row in chosen},
    )


def _profile_rows(cache: ServiceRateCache, spec: WorkloadSpec) -> list[dict]:
    records = cache.profiles(spec.workload_key)
    if not records:
        raise KeyError(f"no service cache entries for workload {spec.workload_key!r}")
    return [_profile_row(record, spec) for record in records]


def _profile_row(record: ProfileRecord, spec: WorkloadSpec) -> dict:
    return {
        "workload_key": spec.workload_key,
        "task_count": spec.task_count,
        "profile": record.profile,
        "makespan_s": deterministic_makespan_s(
            task_count=spec.task_count,
            total_units=spec.total_units,
            resource_count=spec.resource_count,
            profile=record.profile,
            aggregate_rate=record.aggregate_rate,
        ),
        "mean_flow_s": deterministic_mean_flow_s(
            task_count=spec.task_count,
            total_units=spec.total_units,
            resource_count=spec.resource_count,
            profile=record.profile,
            aggregate_rate=record.aggregate_rate,
        ),
    }


def _add_summary_dir(
    cache: ServiceRateCache,
    reports_dir: Path,
    *,
    workload_key: str,
    command_fingerprint: str,
    resource_kind: str,
    total_units: float,
    node_bucket: str,
) -> None:
    paths = sorted(reports_dir.glob("profile_*_per_gpu_summary.json"))
    paths += sorted(reports_dir.glob("profile_*_per_resource_summary.json"))
    for path in paths:
        for record in records_from_summary_file(
            path,
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            total_units=total_units,
            node_bucket=node_bucket,
        ):
            cache.add(record)
