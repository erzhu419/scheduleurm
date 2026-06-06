"""Benchmark task-set registry for Scheduleurm replay experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .fast_forward import WorkloadSpec
from .service_cache import ServiceRateCache, missing_exact_profiles


@dataclass(frozen=True)
class TaskSetMember:
    workload_key: str
    resource_kind: str
    task_count: int
    total_units: float
    resource_count: int
    variation_cv: float
    quadrant: str
    role: str
    benchmark_source: str
    required_profiles: tuple[int, ...]
    empirical_status: str = "real"
    note: str = ""

    def to_workload_spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            workload_key=self.workload_key,
            resource_kind=self.resource_kind,
            task_count=self.task_count,
            total_units=self.total_units,
            resource_count=self.resource_count,
            variation_cv=self.variation_cv,
        )

    def missing_profiles(self, cache: ServiceRateCache) -> list[int]:
        if self.empirical_status == "probe_required":
            return list(self.required_profiles)
        required = tuple(sorted({int(x) for x in self.required_profiles}))
        boundary = _first_capacity_boundary(cache, self.workload_key, required)
        profiles = [profile for profile in required if boundary is None or profile < boundary]
        return missing_exact_profiles(cache, self.workload_key, profiles)

    def snapshot(self, cache: ServiceRateCache | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "workload_key": self.workload_key,
            "resource_kind": self.resource_kind,
            "task_count": self.task_count,
            "total_units": self.total_units,
            "resource_count": self.resource_count,
            "variation_cv": self.variation_cv,
            "quadrant": self.quadrant,
            "role": self.role,
            "benchmark_source": self.benchmark_source,
            "required_profiles": list(self.required_profiles),
            "empirical_status": self.empirical_status,
            "note": self.note,
        }
        if cache is not None:
            out["missing_profiles"] = self.missing_profiles(cache)
            out["closed_by_capacity_boundary_profile"] = _first_capacity_boundary(
                cache,
                self.workload_key,
                self.required_profiles,
            )
        return out


@dataclass(frozen=True)
class TaskSet:
    name: str
    purpose: str
    arrival_model: str
    members: tuple[TaskSetMember, ...]

    def workload_specs(self, *, replayable_only: bool = False, cache: ServiceRateCache | None = None) -> list[WorkloadSpec]:
        specs: list[WorkloadSpec] = []
        for member in self.members:
            if replayable_only:
                if cache is None:
                    raise ValueError("cache is required when replayable_only=True")
                if member.empirical_status == "probe_required" or member.missing_profiles(cache):
                    continue
            specs.append(member.to_workload_spec())
        return specs

    def missing_measurements(self, cache: ServiceRateCache) -> dict[str, list[int]]:
        missing: dict[str, list[int]] = {}
        for member in self.members:
            profiles = member.missing_profiles(cache)
            if profiles:
                missing[member.workload_key] = profiles
        return missing

    def snapshot(self, cache: ServiceRateCache | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "arrival_model": self.arrival_model,
            "members": [member.snapshot(cache) for member in self.members],
            "missing_measurements": self.missing_measurements(cache) if cache is not None else {},
        }


def benchmark_tasksets() -> dict[str, TaskSet]:
    tasksets = [
        TaskSet(
            name="q00_light_control",
            purpose=(
                "Low-CPU, low-GPU control surface. This catches scheduler overhead, queue churn, "
                "and short-task fragmentation before the resource-pressure modules are enabled."
            ),
            arrival_model="static batch first; later Poisson arrivals at subcritical load",
            members=(
                TaskSetMember(
                    workload_key="light_control_local",
                    resource_kind="light_control",
                    task_count=512,
                    total_units=10000,
                    resource_count=1,
                    variation_cv=0.20,
                    quadrant="low_cpu_low_gpu",
                    role="control",
                    benchmark_source="Scheduleurm modules23+24+38 local CPU-only light-control curve",
                    required_profiles=tuple(range(1, 15)),
                    empirical_status="real",
                    note=(
                        "Profiles 1-13 are clean local measurements; profile 14 is a measured "
                        "local scheduling-capacity boundary under the exact light-control workload."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="q01_gpu_bound_compute",
            purpose=(
                "Low-host, high-GPU compute pressure. This is the pure GPU curve that should "
                "prefer fewer co-located jobs once aggregate service saturates."
            ),
            arrival_model="static batch plus Gavel-style Poisson seeds after module validation",
            members=(
                TaskSetMember(
                    workload_key="gpu_heavy_jax_matmul",
                    resource_kind="gpu_heavy",
                    task_count=48,
                    total_units=2400,
                    resource_count=2,
                    variation_cv=0.04,
                    quadrant="low_cpu_high_gpu",
                    role="single-bottleneck validation",
                    benchmark_source="Scheduleurm module6; analogous to Gavel/Pollux DNN throughput-table replay",
                    required_profiles=(1, 2, 3, 4, 5, 6, 7, 8),
                    empirical_status="real",
                    note="Profiles 1-8 are clean real measurements on jtl110gpu2.",
                ),
            ),
        ),
        TaskSet(
            name="q10_cpu_host_bound",
            purpose=(
                "High-CPU/host pressure with low GPU pressure. This separates scheduler CPU/RAM "
                "packing from GPU-placement decisions on a real local CPU-heavy bucket."
            ),
            arrival_model="static local CPU-heavy batch; then Poisson rate sweep for host saturation",
            members=(
                TaskSetMember(
                    workload_key="cpu_heavy_local_bench",
                    resource_kind="cpu_heavy",
                    task_count=256,
                    total_units=1000,
                    resource_count=1,
                    variation_cv=0.10,
                    quadrant="high_cpu_low_gpu",
                    role="single-bottleneck validation",
                    benchmark_source="Scheduleurm modules25+33 local CPU-heavy progress-bearing curve",
                    required_profiles=tuple(range(1, 11)),
                    empirical_status="real",
                    note=(
                        "Profiles 1-9 are real local measurements; profile 10 is a clean "
                        "local capacity boundary. This closes q10 for the declared local "
                        "CPU bucket, not for a remote CPU-node bucket."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="q10_cpu_host_bound_local_real_probe",
            purpose=(
                "Explicit alias for the real local CPU-heavy q10 calibration curve. "
                "The active q10 benchmark uses the same declared local CPU bucket."
            ),
            arrival_model="static local CPU-heavy calibration batch",
            members=(
                TaskSetMember(
                    workload_key="cpu_heavy_local_bench",
                    resource_kind="cpu_heavy",
                    task_count=256,
                    total_units=1000,
                    resource_count=1,
                    variation_cv=0.10,
                    quadrant="high_cpu_low_gpu",
                    role="real local q10 calibration probe",
                    benchmark_source="Scheduleurm modules25+33 local CPU-heavy progress-bearing curve",
                    required_profiles=tuple(range(1, 11)),
                    empirical_status="real",
                    note=(
                        "Profiles 1-9 are real local measurements and profile 10 is the "
                        "capacity boundary. This taskset is retained as an explicit alias "
                        "for the local q10 probe."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="q11_cpu_gpu_coupled",
            purpose=(
                "High-CPU, high-GPU coupled pressure. This is the class matching RE-SAC/BAPR-like "
                "RL jobs where several jobs per GPU can retain near-solo ETA."
            ),
            arrival_model="dense static batch; later long trace with bursty arrivals",
            members=(
                TaskSetMember(
                    workload_key="hybrid_rl_resac_ant",
                    resource_kind="hybrid_rl",
                    task_count=160,
                    total_units=80,
                    resource_count=1,
                    variation_cv=0.08,
                    quadrant="high_cpu_high_gpu",
                    role="coupled interference validation",
                    benchmark_source="Scheduleurm module12 real RE-SAC Ant dense co-location profile",
                    required_profiles=tuple(range(1, 17)),
                    empirical_status="partial_real",
                    note="Profiles 1-12 are clean real measurements; profile 13 hit runtime OOM/placement invalid and closes higher-profile measurement obligations for this node bucket.",
                ),
            ),
        ),
        TaskSet(
            name="hybrid_research_portfolio",
            purpose=(
                "Mixed workload portfolio used after individual modules pass. This is the closest "
                "Scheduleurm analogue of Gavel/Pollux/Sia trace replay."
            ),
            arrival_model="static portfolio now; Poisson and production-like bursts after service cache expansion",
            members=(
                TaskSetMember(
                    workload_key="hybrid_rl_resac_ant",
                    resource_kind="hybrid_rl",
                    task_count=120,
                    total_units=80,
                    resource_count=1,
                    variation_cv=0.08,
                    quadrant="high_cpu_high_gpu",
                    role="dominant Scheduleurm research workload",
                    benchmark_source="Scheduleurm module12 real RE-SAC Ant service curve",
                    required_profiles=tuple(range(1, 11)),
                    empirical_status="real",
                ),
                TaskSetMember(
                    workload_key="gpu_heavy_jax_matmul",
                    resource_kind="gpu_heavy",
                    task_count=24,
                    total_units=2400,
                    resource_count=2,
                    variation_cv=0.04,
                    quadrant="low_cpu_high_gpu",
                    role="pure GPU counterexample class",
                    benchmark_source="Scheduleurm module6 real JAX matmul service curve",
                    required_profiles=(1, 2, 3),
                    empirical_status="real",
                ),
                TaskSetMember(
                    workload_key="cpu_heavy_local_bench",
                    resource_kind="cpu_heavy",
                    task_count=256,
                    total_units=1000,
                    resource_count=1,
                    variation_cv=0.10,
                    quadrant="high_cpu_low_gpu",
                    role="host saturation validation",
                    benchmark_source="Scheduleurm modules25+33 local CPU-heavy service curve",
                    required_profiles=tuple(range(1, 11)),
                    empirical_status="real",
                ),
            ),
        ),
    ]
    return {taskset.name: taskset for taskset in tasksets}


def taskset_by_name(name: str) -> TaskSet:
    tasksets = benchmark_tasksets()
    try:
        return tasksets[name]
    except KeyError as exc:
        known = ", ".join(sorted(tasksets))
        raise KeyError(f"unknown taskset {name!r}; known tasksets: {known}") from exc


def empirical_replay_taskset() -> TaskSet:
    return taskset_by_name("hybrid_research_portfolio")


def empirical_replay_specs() -> list[WorkloadSpec]:
    return empirical_replay_taskset().workload_specs()


def all_missing_measurements(cache: ServiceRateCache, names: Iterable[str] | None = None) -> dict[str, dict[str, list[int]]]:
    selected = benchmark_tasksets()
    if names is not None:
        selected = {name: taskset_by_name(name) for name in names}
    return {
        name: taskset.missing_measurements(cache)
        for name, taskset in selected.items()
        if taskset.missing_measurements(cache)
    }


def _first_capacity_boundary(cache: ServiceRateCache, workload_key: str, profiles: Iterable[int]) -> int | None:
    for profile in sorted({int(x) for x in profiles}):
        record = cache.get(workload_key, profile)
        if record is not None and record.capacity_boundary:
            return profile
    return None
