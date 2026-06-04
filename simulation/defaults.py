"""Default trace-cache assembly for Scheduleurm replay experiments."""
from __future__ import annotations

from pathlib import Path

from .fast_forward import ReplayPolicy, WorkloadSpec
from .service_cache import ServiceRateCache, add_protocol_cpu_curve, records_from_summary_file


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
    return [
        WorkloadSpec(
            workload_key="hybrid_rl_resac_ant",
            resource_kind="hybrid_rl",
            task_count=120,
            total_units=80,
            resource_count=1,
            variation_cv=0.08,
        ),
        WorkloadSpec(
            workload_key="gpu_heavy_jax_matmul",
            resource_kind="gpu_heavy",
            task_count=24,
            total_units=2400,
            resource_count=2,
            variation_cv=0.04,
        ),
        WorkloadSpec(
            workload_key="cpu_heavy_protocol",
            resource_kind="cpu_heavy",
            task_count=256,
            total_units=3600,
            resource_count=1,
            variation_cv=0.10,
        ),
    ]


def legacy_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="legacy_fixed_caps",
        fixed_profiles={
            "hybrid_rl_resac_ant": 5,
            "gpu_heavy_jax_matmul": 3,
            "cpu_heavy_protocol": 32,
        },
    )


def calibrated_policy() -> ReplayPolicy:
    return ReplayPolicy(name="calibrated_fast_forward", calibrated=True)


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
    for path in sorted(reports_dir.glob("profile_*_per_gpu_summary.json")):
        for record in records_from_summary_file(
            path,
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            total_units=total_units,
            node_bucket=node_bucket,
        ):
            cache.add(record)
