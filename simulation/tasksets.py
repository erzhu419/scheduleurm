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
            name="production_freqduet_cpu_c17_32",
            purpose=(
                "The first production CPU/SUMO/transit closure sub-bucket from Module53. "
                "It covers FreqDuet ablation/control records requesting 17-32 CPU cores "
                "and no GPU on the Windows jtl110cpu2 CPU node."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped only "
                "when the record text matches the FreqDuet ablation family and the "
                "CPU request is in c_17_32."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_cpu_ablation_c17_32",
                    resource_kind="cpu_sumo_transit",
                    task_count=125,
                    total_units=72,
                    resource_count=1,
                    variation_cv=0.20,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker sub-bucket",
                    benchmark_source=(
                        "Scheduleurm module56 real FreqDuet CPU ablation curve on "
                        "jtl110cpu2; each task ran 24 seeds/workers with CSV progress."
                    ),
                    required_profiles=(1, 2, 4, 8),
                    empirical_status="real",
                    note=(
                        "Profiles 1, 2, and 4 are real feasible measurements. Profile 8 "
                        "is a real capacity-boundary probe with 5 running/progressing "
                        "tasks and 3 CPU-fit blocked tasks, so it closes the feasible "
                        "domain above profile 4 for this node bucket."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_runner_v3_allfreq_alllayers_c9_16",
            purpose=(
                "A narrow production CPU/SUMO/transit closure slice from Module57. "
                "It covers only direct FreqDuet runner_v3 records using "
                "configs_freqduet/F_allfreq_alllayers_hiro.yaml with 9-16 requested "
                "CPU cores and no GPU."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped only "
                "when the command exactly matches the measured direct runner_v3 config."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_runner_v3_allfreq_alllayers_c9_16",
                    resource_kind="cpu_sumo_transit",
                    task_count=1,
                    total_units=20,
                    resource_count=1,
                    variation_cv=0.20,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker exact-config sub-slice",
                    benchmark_source=(
                        "Scheduleurm module57 real FreqDuet runner_v3 curve on "
                        "jtl110cpu2; each task ran runner_v3 with diagnostics CSV "
                        "progress for configs_freqduet/F_allfreq_alllayers_hiro.yaml."
                    ),
                    required_profiles=(1, 2, 4, 8),
                    empirical_status="real",
                    note=(
                        "Profiles 1, 2, 4, and 8 are feasible real measurements. "
                        "This is intentionally an exact-config certificate; other "
                        "runner_v3 c9_16 configs remain unmeasured until Module58+ "
                        "adds lower-service or per-config curves."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_cpu_ablation_c3_8_completed_history",
            purpose=(
                "A strict completed-history closure slice for the largest remaining "
                "FreqDuet c_3_8 sub-bucket after Module59. It covers only "
                "run_freqduet_ablation.py records whose CPU request is 3-8 cores, "
                "GPU request is zero, and job-shard episode units are parseable."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped with "
                "parsed jobs times episodes. Only profile 1 is loaded from realized "
                "completed-task wall-clock history; no live co-location profile is "
                "claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_cpu_ablation_c3_8_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=40,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.30,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history sub-slice",
                    benchmark_source=(
                        "Scheduleurm module60 completed-history wall-clock audit for "
                        "parseable c_3_8 run_freqduet_ablation.py production records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "episode rate as a conservative profile-1 lower-service point. "
                        "Runner_v3, native validation, and unknown-size commands in "
                        "the same c_3_8 bucket remain unmeasured."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_cpu_ablation_c33_64_completed_history",
            purpose=(
                "A strict completed-history closure slice for the largest remaining "
                "FreqDuet c_33_64 sub-bucket after Module60. It covers only "
                "run_freqduet_ablation.py records whose CPU request is 33-64 cores, "
                "GPU request is zero, and job-shard episode units are parseable."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped with "
                "parsed jobs times episodes. Only profile 1 is loaded from realized "
                "completed-task wall-clock history; no live co-location profile is "
                "claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_cpu_ablation_c33_64_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=63,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.30,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history sub-slice",
                    benchmark_source=(
                        "Scheduleurm module61 completed-history wall-clock audit for "
                        "parseable c_33_64 run_freqduet_ablation.py production records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "episode rate as a conservative profile-1 lower-service point. "
                        "Runner_v3 and native validation commands in the same c_33_64 "
                        "bucket remain unmeasured."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_cpu_ablation_c65p_completed_history",
            purpose=(
                "A strict completed-history closure slice for high-CPU FreqDuet "
                "ablation records after Module68. It covers only no-GPU "
                "run_freqduet_ablation.py commands requesting more than 64 CPU "
                "cores, with worker-threads fixed at one when specified, and "
                "with parseable job-shard episode units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed jobs times episodes. Only profile 1 is loaded from "
                "realized completed-task wall-clock history; no live co-location "
                "profile is claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_cpu_ablation_c65p_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=7,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.30,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history c65p ablation sub-slice",
                    benchmark_source=(
                        "Scheduleurm module69 completed-history wall-clock audit for "
                        "parseable c65p run_freqduet_ablation.py production records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "episode rate as a conservative profile-1 lower-service point. "
                        "Promoted ep100 shell batches and native-promotion validation "
                        "records are separate Module69 classes."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_promoted_ep100_c65p_completed_history",
            purpose=(
                "A strict completed-history closure slice for high-CPU FreqDuet "
                "promoted ep100 shell-batch records. It covers only "
                "run_freqduet_promoted_ep100_hpc_batch.sh commands with parseable "
                "job-start/job-end bounds and the script's verified EPISODES=100 "
                "default."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed job-count times 100 episode units. Only profile 1 is "
                "loaded from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_promoted_ep100_c65p_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=6,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.30,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history c65p promoted-batch sub-slice",
                    benchmark_source=(
                        "Scheduleurm module69 completed-history wall-clock audit for "
                        "FreqDuet promoted ep100 HPC batch records. The work-unit "
                        "definition is checked against the local shell script, which "
                        "passes EPISODES=100 to its underlying ablation/baseline "
                        "runners."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The parser rejects commands with a non-100 EPISODES override "
                        "or missing job bounds. Other promoted or external-baseline "
                        "shell commands remain outside this class."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_runner_v3_c3_8_completed_history",
            purpose=(
                "A strict completed-history closure slice for the dominant direct "
                "runner_v3 portion of the remaining FreqDuet c_3_8 residual bucket. "
                "It covers no-GPU runner_v3.py records requesting 3-8 CPU cores "
                "when an explicit --episodes work unit is present."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped with "
                "parsed episode units. Only profile 1 is loaded from realized "
                "completed-task wall-clock history; native validation, shell-loop, "
                "and higher co-location profiles are not claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_runner_v3_c3_8_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=86,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history runner sub-slice",
                    benchmark_source=(
                        "Scheduleurm module66 completed-history wall-clock audit for "
                        "c_3_8 direct runner_v3.py production records with explicit "
                        "episode counts."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "episode rate as a conservative profile-1 lower-service point. "
                        "This does not close native validation, shell-expanded, or "
                        "other non-runner command shapes in the same c_3_8 residual."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_cpu_ablation_c9_16",
            purpose=(
                "The second high-impact production CPU/SUMO/transit closure slice "
                "from Module53. It covers FreqDuet ablation/control records invoking "
                "scripts/run_freqduet_ablation.py with 9-16 requested CPU cores, no GPU, "
                "and parseable job-shard episode units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped only "
                "when the command shape matches run_freqduet_ablation.py in c_9_16. "
                "The production load certificate uses parsed per-record units "
                "jobs times episodes, not a single fixed task size."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_cpu_ablation_c9_16",
                    resource_kind="cpu_sumo_transit",
                    task_count=110,
                    total_units=2200,
                    resource_count=1,
                    variation_cv=0.25,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker sub-bucket",
                    benchmark_source=(
                        "Scheduleurm module58 real FreqDuet CPU ablation curve on "
                        "jtl110cpu2; each task ran a production-like 13-job shard "
                        "with diagnostics CSV progress."
                    ),
                    required_profiles=(1, 2, 4, 8),
                    empirical_status="real",
                    note=(
                        "Profiles 1, 2, 4, and 8 are feasible real measurements. "
                        "The classifier additionally requires parseable job-shard "
                        "units, so records with unknown work size stay unmeasured."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_native_promotion_c9_16_bounded_wait_completed_history",
            purpose=(
                "A strict completed-history closure slice for the slow c_9_16 "
                "Transit/FreqHRL native-promotion bounded_wait_nofinal_v14 stress "
                "profile. It covers no-GPU native_promotion_replan_validation "
                "commands requesting 9-16 CPU cores with that exact stress profile "
                "and statically parsed seed-episode work units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed seed-count times episodes. Only profile 1 is loaded "
                "from realized completed-task wall-clock history; no live "
                "co-location profile is claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_native_promotion_c9_16_bounded_wait_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=7,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.40,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history native-validation c9_16 bounded-wait sub-slice",
                    benchmark_source=(
                        "Scheduleurm module71 completed-history wall-clock audit for "
                        "c_9_16 bounded_wait_nofinal_v14 native_promotion_replan_validation "
                        "records with statically parsed seed work units."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This slow stress profile is isolated because applying its "
                        "minimum realized lower-service rate to all c_9_16 native "
                        "promotion records makes the mapped capacity LP infeasible. "
                        "Other c_9_16 native records use a separate residual class."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_native_promotion_c9_16_residual_completed_history",
            purpose=(
                "A strict completed-history closure slice for residual c_9_16 "
                "Transit/FreqHRL native-promotion validation records outside "
                "bounded_wait_nofinal_v14. It covers no-GPU "
                "native_promotion_replan_validation commands requesting 9-16 CPU "
                "cores when seed work units can be statically parsed."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed seed-count times episodes. Only profile 1 is loaded "
                "from realized completed-task wall-clock history; no live "
                "co-location profile is claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_native_promotion_c9_16_residual_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=40,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.40,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history native-validation c9_16 residual sub-slice",
                    benchmark_source=(
                        "Scheduleurm module71 completed-history wall-clock audit for "
                        "residual c_9_16 native_promotion_replan_validation records "
                        "with statically parsed seed work units."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The parser accepts explicit seed-index ranges, CLI seed "
                        "lists, Python-AST run_validation calls, and shell "
                        "command-substitution seed lists without executing the "
                        "production command. The bounded_wait_nofinal_v14 slow "
                        "profile and runner_v3 residuals stay separate."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_runner_v3_c9_16_residual_completed_history",
            purpose=(
                "A strict completed-history closure slice for residual c_9_16 "
                "FreqDuet direct runner_v3 records. It covers no-GPU runner_v3.py "
                "commands requesting 9-16 CPU cores with explicit --episodes, "
                "excluding the exact allfreq/alllayers Module57 config."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed episode units. Only profile 1 is loaded from realized "
                "completed-task wall-clock history; no live co-location profile is "
                "claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_runner_v3_c9_16_residual_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=12,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history runner c9_16 residual sub-slice",
                    benchmark_source=(
                        "Scheduleurm module71 completed-history wall-clock audit for "
                        "residual c_9_16 runner_v3.py production records with "
                        "explicit episode counts."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This class does not include run_freqduet_ablation.py "
                        "records or the Module57 allfreq/alllayers exact-config "
                        "curve. It is a conservative portfolio lower-service point "
                        "over the remaining runner_v3 c_9_16 configs."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_cfcmt_feed_conversion_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU CFCMT "
                "GTFS/LTA feed conversion commands. It covers CFCMT no-GPU "
                "gtfs_to_h2o_xlsx.py and lta_to_h2o_xlsx.py records requesting "
                "at most 2 CPU cores."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "as feed-conversion jobs. Only profile 1 is loaded from realized "
                "completed-task wall-clock history; no live co-location profile "
                "is claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="cfcmt_feed_conversion_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=5,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.45,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history CFCMT feed conversion c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module72 completed-history wall-clock audit "
                        "for CFCMT feed conversion scripts."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "Feed conversion is separated from simulation, snapshot, "
                        "policy-rollout, and traffic-signal scripts because the "
                        "progress unit is a completed conversion job."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_cfcmt_env_validation_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU CFCMT "
                "H2O city-environment validation commands with explicit "
                "--max-steps."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed validation-step units. Only profile 1 is loaded "
                "from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="cfcmt_env_validation_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=5,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history CFCMT env validation c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module72 completed-history wall-clock audit "
                        "for validate_h2o_city_env.py records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note="The service unit is the command's parsed --max-steps value.",
                ),
            ),
        ),
        TaskSet(
            name="production_cfcmt_sumo_generation_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU CFCMT "
                "SUMO/APC/AVL generation commands with explicit --duration-sec."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed simulated-second units. Only profile 1 is loaded "
                "from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="cfcmt_sumo_generation_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=4,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.40,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history CFCMT SUMO generation c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module72 completed-history wall-clock audit "
                        "for cf_h2o.eval.sumo_apc_avl_sumo_generation records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "Policy rollout and snapshot-generation records are parsed "
                        "before this class because their stage2-report paths contain "
                        "sumo_generation filenames."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_cfcmt_snapshot_generation_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU CFCMT "
                "SUMO/APC/AVL snapshot-generation commands."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with snapshot-window units inferred from --snapshot-period and "
                "the stage2-report duration family. Only profile 1 is loaded from "
                "realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="cfcmt_snapshot_generation_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=5,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.45,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history CFCMT snapshot generation c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module72 completed-history wall-clock audit "
                        "for cf_h2o.eval.sumo_apc_avl_snapshot_generation records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "Duration is inferred conservatively from the stage2-report "
                        "family: full_day -> 86400 s, 4h -> 14400 s, otherwise "
                        "1800 s."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_cfcmt_policy_rollout_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU CFCMT "
                "SUMO policy-rollout validation commands."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with policy-event-budget units equal to parsed policy count times "
                "--max-events-per-city-policy, falling back to --max-steps. Only "
                "profile 1 is loaded from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="cfcmt_policy_rollout_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=10,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.45,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history CFCMT policy rollout c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module72 completed-history wall-clock audit "
                        "for cf_h2o/eval/sumo_policy_rollout_validation.py records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This class is intentionally separated from SUMO generation "
                        "and snapshot generation because rollout commands contain "
                        "their inputs only as stage2-report paths."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_cfcmt_traffic_signal_phase2_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for the low-CPU CFCMT "
                "traffic_signal_sumo_phase2.py singleton."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "as one phase2 run. Only profile 1 is loaded from the realized "
                "completed-task wall-clock record."
            ),
            members=(
                TaskSetMember(
                    workload_key="cfcmt_traffic_signal_phase2_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=1,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.50,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history CFCMT traffic signal phase2 singleton",
                    benchmark_source=(
                        "Scheduleurm module72 completed-history wall-clock audit "
                        "for traffic_signal_sumo_phase2.py."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This singleton is kept separate so a generic phase2 run "
                        "does not depress the progress units of simulation or "
                        "rollout classes."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_freqduet_runner_v3_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for the dominant direct "
                "runner_v3 portion of the FreqDuet c_le2 bucket. It covers no-GPU "
                "runner_v3.py records requesting at most 2 CPU cores and having an "
                "explicit --episodes value."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped with "
                "parsed episode units. Only profile 1 is loaded from realized "
                "completed-task wall-clock history; no live co-location profile is "
                "claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="freqduet_runner_v3_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=84,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history runner sub-slice",
                    benchmark_source=(
                        "Scheduleurm module62 completed-history wall-clock audit for "
                        "c_le2 direct runner_v3.py production records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This is a conservative portfolio lower-service point over "
                        "multiple runner_v3 configs. The service cache uses the "
                        "minimum realized completed-task episode rate; ablation, "
                        "baseline-rule, scheduler wait, and native validation records "
                        "in c_le2 remain unmeasured unless covered by other modules."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_native_promotion_c17_32_seedrange_completed_history",
            purpose=(
                "A strict completed-history closure slice for Transit/FreqHRL native "
                "promotion validation records in the c_17_32 residual bucket. It "
                "covers no-GPU native_promotion_replan_validation commands with "
                "explicit --seed-index-start/--seed-index-end and parsed episode "
                "work units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped with "
                "parsed seed-count times episodes. Only profile 1 is loaded from "
                "realized completed-task wall-clock history; no live co-location "
                "profile is claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_native_promotion_c17_32_seedrange_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=71,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history native-validation sub-slice",
                    benchmark_source=(
                        "Scheduleurm module63 completed-history wall-clock audit for "
                        "c_17_32 native_promotion_replan_validation production records "
                        "with explicit seed-index ranges."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "seed-episode rate as a conservative profile-1 lower-service "
                        "point. Native commands without seed-index ranges remain "
                        "unmeasured."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_native_promotion_c33_64_batch_completed_history",
            purpose=(
                "A strict completed-history closure slice for Transit/FreqHRL native "
                "promotion validation records in the c_33_64 residual bucket. It "
                "covers no-GPU native_promotion_replan_validation commands with "
                "parseable seed-list, seed-range, or Python-AST seed-comprehension "
                "work units, excluding single-seed smoke/fix records."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped with "
                "parsed seed-count times episodes. Only profile 1 is loaded from "
                "realized completed-task wall-clock history; no live co-location "
                "profile is claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_native_promotion_c33_64_batch_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=45,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history native-validation c33 batch sub-slice",
                    benchmark_source=(
                        "Scheduleurm module68 completed-history wall-clock audit for "
                        "c_33_64 native_promotion_replan_validation batch records "
                        "with parseable seed work units."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "seed-episode rate for the batch class. Two single-seed "
                        "smoke/fix records are isolated into their own class so they "
                        "do not dominate the lower-service bound for large batches."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_native_promotion_c33_64_single_seed_completed_history",
            purpose=(
                "A strict completed-history closure slice for two single-seed "
                "FreqHRLNative c_33_64 smoke/fix native-promotion records."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "only when the parsed native-promotion work unit is exactly one "
                "seed-episode. Only profile 1 is loaded from realized wall-clock "
                "history."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_native_promotion_c33_64_single_seed_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=2,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history native-validation c33 single-seed sub-slice",
                    benchmark_source=(
                        "Scheduleurm module68 completed-history wall-clock audit for "
                        "single-seed c_33_64 native_promotion_replan_validation "
                        "production records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This intentionally separate class keeps smoke/fix overhead "
                        "from being applied to batch native-promotion workloads."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_native_promotion_c65p_completed_history",
            purpose=(
                "A strict completed-history closure slice for high-CPU "
                "Transit/FreqHRL native-promotion validation records. It covers "
                "no-GPU native_promotion_replan_validation commands requesting "
                "more than 64 CPU cores when seed work units can be parsed from "
                "explicit CLI seeds, seed-index ranges, Python-AST seed "
                "comprehensions, or shell command-substitution Python snippets."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed seed-count times episodes. Only profile 1 is loaded "
                "from realized completed-task wall-clock history; no live "
                "co-location profile is claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_native_promotion_c65p_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=57,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history native-validation c65p sub-slice",
                    benchmark_source=(
                        "Scheduleurm module69 completed-history wall-clock audit for "
                        "c65p native_promotion_replan_validation records with "
                        "statically parsed seed work units."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The AST parser proves shell-generated seed lists without "
                        "executing the command. Native commands with unparseable "
                        "dynamic seed logic remain measurement-required."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_trading_sweep_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU TransitDuet "
                "trading validation records. It covers c_le2 no-GPU "
                "freq_hrl.experiments.trading promotion, performance, pressure, "
                "and encoder-validation entry points with statically parsed "
                "market-step grid work units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with seed-count times steps times assets times the script's "
                "explicit or default validation grid. Only profile 1 is loaded "
                "from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_trading_sweep_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=14,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history TransitDuet trading sweep c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module70 completed-history wall-clock audit "
                        "for TransitDuet trading sweep and validation commands."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The parser uses CLI defaults from the local scripts: "
                        "performance_validation runs 11 baselines, pressure_test_matrix "
                        "defaults to six scenarios and twelve baselines, "
                        "encoder_ablation defaults to six encoders, and "
                        "promotion_sweep includes its default promotion grid."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_trading_policy_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU TransitDuet "
                "trading policy-training records. It covers c_le2 no-GPU "
                "policy_entry and ppo_actor_critic commands with statically "
                "parsed train/eval market-step work units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with train seeds times steps times assets times optimizer "
                "iterations plus one eval pass. Only profile 1 is loaded from "
                "realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_trading_policy_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=18,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history TransitDuet trading policy c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module70 completed-history wall-clock audit "
                        "for TransitDuet policy_entry and ppo_actor_critic commands."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This class is separated from sweep validation because policy "
                        "training has optimizer-iteration work units and a different "
                        "realized lower-service rate."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_surrogate_validation_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU Transit "
                "surrogate-validation records. It covers c_le2 no-GPU "
                "gap_closure_validation and ppo_surrogate commands with parsed "
                "surrogate corridor-step work units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with train/eval seeds, steps, corridors, optimizer iterations, "
                "and the gap-closure variant count when applicable. Only profile "
                "1 is loaded from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_surrogate_validation_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=3,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history Transit surrogate c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module70 completed-history wall-clock audit "
                        "for Transit Freq-HRL surrogate validation commands."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This class is not merged with native simulator validation; "
                        "it uses synthetic surrogate corridor-step units."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_native_promotion_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU native "
                "Transit promotion-replan validation. It covers c_le2 no-GPU "
                "native_promotion_replan_validation commands with parsed "
                "variant-count times seed-count times episode work units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with native validation variant-episode units. Only profile 1 "
                "is loaded from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_native_promotion_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=19,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.40,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history native promotion c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module70 completed-history wall-clock audit "
                        "for low-CPU native_promotion_replan_validation commands."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The min-pairs CLI argument is a statistical claim gate, "
                        "not an execution-loop multiplier, and is therefore not "
                        "included in the work-unit definition."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_native_control_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for low-CPU native "
                "Transit control validation records. It covers c_le2 no-GPU "
                "native_wait_credit_validation and "
                "native_real_demand_control_validation commands with parsed "
                "native control episode units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with variant-count times source-count when applicable times "
                "seed-count times episodes. Only profile 1 is loaded from "
                "realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_native_control_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=7,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.40,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history native control c_le2 sub-slice",
                    benchmark_source=(
                        "Scheduleurm module70 completed-history wall-clock audit "
                        "for native wait-credit and real-demand control validation."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This class stays separate from native promotion-replan "
                        "because the real-demand source loop and wait-credit variants "
                        "have different command semantics."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_transit_freqhrl_import_smoke_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for a single low-CPU "
                "Transit/FreqHRL Windows import-smoke record. It covers only the "
                "python -c IMPORT_OK command that imports "
                "native_promotion_replan_validation."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "as one import-check unit. Only profile 1 is loaded from the "
                "realized completed-task wall-clock record."
            ),
            members=(
                TaskSetMember(
                    workload_key="transit_freqhrl_import_smoke_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=1,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.10,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history import-smoke c_le2 singleton",
                    benchmark_source=(
                        "Scheduleurm module70 completed-history wall-clock audit "
                        "for the Windows Transit/FreqHRL import sanity command."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This singleton does not authorize charging real native "
                        "promotion validation records to an import-smoke rate."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_bamor_cpu_training_c3_8_completed_history",
            purpose=(
                "A strict completed-history closure slice for BAMOR c_3_8 CPU "
                "training records. It covers no-GPU BAMOR commands requesting "
                "3-8 CPU cores when train_compare_baselines.py, train_bamor_mujoco.py, "
                "or run_bamor_diagnostic_shard.py exposes parseable training-step "
                "work units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped with "
                "parsed training-step units. Only profile 1 is loaded from realized "
                "completed-task wall-clock history; no live co-location profile is "
                "claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="bamor_cpu_training_c3_8_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=142,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history BAMOR training sub-slice",
                    benchmark_source=(
                        "Scheduleurm module64 completed-history wall-clock audit for "
                        "BAMOR c_3_8 CPU training production records with parseable "
                        "training-step units."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "training-step rate as a conservative profile-1 lower-service "
                        "point. BAMOR CPU buckets outside c_3_8 and commands without "
                        "parseable training steps remain unmeasured."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_bamor_train_compare_c3_8_completed_history",
            purpose=(
                "A script-level refinement of the Module64 BAMOR c_3_8 slice. "
                "It covers no-GPU train_compare_baselines.py records requesting "
                "3-8 CPU cores with parseable training-step units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed method-count times seed-count times total_steps. Only "
                "profile 1 is loaded from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="bamor_train_compare_c3_8_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=58,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history BAMOR train-compare sub-slice",
                    benchmark_source=(
                        "Scheduleurm module67 completed-history wall-clock audit for "
                        "BAMOR c_3_8 train_compare_baselines.py records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "This split prevents the slowest train_compare lower-service "
                        "point from being incorrectly applied to faster BAMOR Mujoco "
                        "or diagnostic-shard work units."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_bamor_mujoco_c3_8_completed_history",
            purpose=(
                "A script-level refinement of the Module64 BAMOR c_3_8 slice. "
                "It covers no-GPU train_bamor_mujoco.py records requesting 3-8 CPU "
                "cores with parseable training-step units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed method-count times num_seeds times total_steps. Only "
                "profile 1 is loaded from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="bamor_mujoco_c3_8_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=89,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history BAMOR Mujoco sub-slice",
                    benchmark_source=(
                        "Scheduleurm module67 completed-history wall-clock audit for "
                        "BAMOR c_3_8 train_bamor_mujoco.py records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "training-step rate for this script family only."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_bamor_diagnostic_shard_c3_8_completed_history",
            purpose=(
                "A script-level refinement of the Module64 BAMOR c_3_8 slice. "
                "It covers no-GPU run_bamor_diagnostic_shard.py records requesting "
                "3-8 CPU cores with parseable shard training-step units."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped "
                "with parsed shard-item count times total_steps. Only profile 1 is "
                "loaded from realized completed-task wall-clock history."
            ),
            members=(
                TaskSetMember(
                    workload_key="bamor_diagnostic_shard_c3_8_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=25,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history BAMOR diagnostic-shard sub-slice",
                    benchmark_source=(
                        "Scheduleurm module67 completed-history wall-clock audit for "
                        "BAMOR c_3_8 run_bamor_diagnostic_shard.py records."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "training-step rate for parseable diagnostic shards. BAMOR "
                        "CPU buckets outside c_3_8 remain separate obligations."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_zsw_tsp_sumo_eval_c_le2_completed_history",
            purpose=(
                "A strict completed-history closure slice for ZSW TSP/SUMO CPU eval "
                "records in the c_le2 bucket. It covers no-GPU baseline/oracle/TSP "
                "runner commands requesting at most 2 CPU cores when simulated SUMO "
                "duration is explicit in the command."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped with "
                "parsed simulated SUMO seconds from --duration. Only profile 1 is "
                "loaded from realized completed-task wall-clock history; no live "
                "co-location profile is claimed by this module."
            ),
            members=(
                TaskSetMember(
                    workload_key="zsw_tsp_sumo_eval_c_le2_completed_history",
                    resource_kind="cpu_sumo_transit",
                    task_count=50,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.35,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history ZSW SUMO eval sub-slice",
                    benchmark_source=(
                        "Scheduleurm module65 completed-history wall-clock audit for "
                        "ZSW TSP/SUMO c_le2 runner records with explicit --duration."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "simulated-second rate as a conservative profile-1 lower-service "
                        "point. CFCMT, offline-sumo, H2Oplus, and direct SUMO binary "
                        "records remain unmeasured unless covered by other modules."
                    ),
                ),
            ),
        ),
        TaskSet(
            name="production_sumo_eval_simple_sac_c_le2",
            purpose=(
                "A strict completed-history closure slice for the largest clean "
                "SUMO c_le2 family after Module58. It covers SimpleSAC "
                "run_multiseed_eval.sh tasks whose method, SUMO seed, and OD scale "
                "are explicit in the command."
            ),
            arrival_model=(
                "30-day Scheduleurm completed/active production arrivals, mapped only "
                "for clean one-eval JSON SimpleSAC commands. Only profile 1 is loaded "
                "from completed wall-clock history; higher co-location profiles remain "
                "unclaimed until a controlled progress-bearing probe is run."
            ),
            members=(
                TaskSetMember(
                    workload_key="sumo_eval_simple_sac_c_le2",
                    resource_kind="cpu_sumo_transit",
                    task_count=54,
                    total_units=1,
                    resource_count=1,
                    variation_cv=0.20,
                    quadrant="high_cpu_low_gpu",
                    role="production coverage blocker completed-history sub-slice",
                    benchmark_source=(
                        "Scheduleurm module59 completed-history wall-clock audit for "
                        "clean SimpleSAC run_multiseed_eval.sh tasks on local CPU."
                    ),
                    required_profiles=(1,),
                    empirical_status="real",
                    note=(
                        "The service cache uses the minimum realized completed-task "
                        "rate as a conservative profile-1 lower service point. "
                        "No profile 2+ service is claimed in this module."
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
                    note=(
                        "Profiles below 10 have exact real measurements used for robust replay; "
                        "a fresh profile-10 live sanity run hit runtime OOM, so 10/GPU and above "
                        "are excluded for the current robust node bucket."
                    ),
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
                    required_profiles=tuple(range(1, 10)),
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
