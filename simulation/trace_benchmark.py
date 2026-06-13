"""Explicit task-list benchmark replay.

This layer mirrors the common systems-paper workflow: build a workload trace,
then replay several scheduling policies on the same task list using measured
service curves. It complements the aggregate fast-forward replay.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .defaults import calibrated_candidate_policy, legacy_policy
from .fast_forward import (
    ReplayPolicy,
    WorkloadSpec,
    _covering_service_profile,
    _statewise_target_profile,
)
from .service_cache import ServiceRateCache
from .sota_baselines import sota_baseline_specs
from .tasksets import TaskSet, taskset_by_name


PARETO_TOLERANCE = 0.005


@dataclass(frozen=True)
class TraceJob:
    job_id: str
    workload_key: str
    resource_kind: str
    arrival_s: float
    total_units: float
    resource_count: int

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "workload_key": self.workload_key,
            "resource_kind": self.resource_kind,
            "arrival_s": self.arrival_s,
            "total_units": self.total_units,
            "resource_count": self.resource_count,
        }

    @staticmethod
    def from_snapshot(data: dict[str, Any]) -> "TraceJob":
        return TraceJob(
            job_id=str(data["job_id"]),
            workload_key=str(data["workload_key"]),
            resource_kind=str(data["resource_kind"]),
            arrival_s=float(data["arrival_s"]),
            total_units=float(data["total_units"]),
            resource_count=max(1, int(data.get("resource_count") or 1)),
        )


@dataclass(frozen=True)
class TaskTrace:
    name: str
    taskset_name: str
    arrival_mode: str
    seed: int
    jobs: tuple[TraceJob, ...]

    def workload_specs(self) -> list[WorkloadSpec]:
        grouped: dict[str, list[TraceJob]] = {}
        for job in self.jobs:
            grouped.setdefault(job.workload_key, []).append(job)
        specs = []
        for key, jobs in sorted(grouped.items()):
            first = jobs[0]
            specs.append(
                WorkloadSpec(
                    workload_key=key,
                    resource_kind=first.resource_kind,
                    task_count=len(jobs),
                    total_units=mean(job.total_units for job in jobs),
                    resource_count=first.resource_count,
                    variation_cv=0.0,
                )
            )
        return specs

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "taskset_name": self.taskset_name,
            "arrival_mode": self.arrival_mode,
            "seed": self.seed,
            "job_count": len(self.jobs),
            "jobs": [job.snapshot() for job in self.jobs],
        }

    @staticmethod
    def from_snapshot(data: dict[str, Any]) -> "TaskTrace":
        return TaskTrace(
            name=str(data["name"]),
            taskset_name=str(data["taskset_name"]),
            arrival_mode=str(data["arrival_mode"]),
            seed=int(data.get("seed") or 0),
            jobs=tuple(TraceJob.from_snapshot(row) for row in data.get("jobs") or []),
        )


@dataclass(frozen=True)
class TracePolicyResult:
    policy: str
    policy_config: dict[str, Any]
    profiles: dict[str, int]
    makespan_s: float
    mean_flow_s: float
    p50_flow_s: float
    p90_flow_s: float
    completed_jobs: int

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "policy_config": self.policy_config,
            "profiles": self.profiles,
            "makespan_s": self.makespan_s,
            "mean_flow_s": self.mean_flow_s,
            "p50_flow_s": self.p50_flow_s,
            "p90_flow_s": self.p90_flow_s,
            "completed_jobs": self.completed_jobs,
        }


def build_task_trace(
    taskset: TaskSet | str,
    *,
    arrival_mode: str = "static",
    seed: int = 42,
) -> TaskTrace:
    selected = taskset_by_name(taskset) if isinstance(taskset, str) else taskset
    rng = random.Random(seed)
    jobs: list[TraceJob] = []
    for member in selected.members:
        arrivals = _arrival_times(
            rng,
            count=member.task_count,
            mode=arrival_mode,
            mean_interarrival_s=_mean_interarrival_s(member.task_count),
        )
        for idx, arrival_s in enumerate(arrivals):
            units = _positive_lognormal(rng, member.total_units, member.variation_cv)
            jobs.append(
                TraceJob(
                    job_id=f"{selected.name}:{member.workload_key}:{idx:05d}",
                    workload_key=member.workload_key,
                    resource_kind=member.resource_kind,
                    arrival_s=arrival_s,
                    total_units=units,
                    resource_count=member.resource_count,
                )
            )
    jobs.sort(key=lambda job: (job.arrival_s, job.job_id))
    return TaskTrace(
        name=f"{selected.name}_{arrival_mode}_seed{seed}",
        taskset_name=selected.name,
        arrival_mode=arrival_mode,
        seed=seed,
        jobs=tuple(jobs),
    )


def benchmark_policies(cache: ServiceRateCache, trace: TaskTrace) -> tuple[ReplayPolicy, ...]:
    specs = trace.workload_specs()
    return (
        legacy_policy(),
        calibrated_candidate_policy(cache, specs),
        *(spec.policy for spec in sota_baseline_specs()),
    )


def replay_trace_suite(
    cache: ServiceRateCache,
    trace: TaskTrace,
    *,
    policies: Iterable[ReplayPolicy] | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    selected = tuple(policies or benchmark_policies(cache, trace))
    results = [replay_trace(cache, trace, policy, seed=seed) for policy in selected]
    sota_comparison = _sota_tasklist_comparison(results)
    return {
        "trace": trace.snapshot(),
        "results": [result.snapshot() for result in results],
        "relative_to_legacy": _relative(results, baseline_name="legacy_fixed_caps"),
        "relative_to_candidate": _relative(results, baseline_name=_candidate_name(results)),
        "sota_tasklist_comparison": sota_comparison,
    }


def replay_trace_matrix(
    cache: ServiceRateCache,
    taskset_names: Iterable[str],
    *,
    arrival_mode: str = "static",
    trace_seed: int = 42,
    replay_seed: int = 7,
) -> dict[str, Any]:
    """Replay every policy on every taskset and aggregate by fixed policy.

    This is the non-stitched SOTA comparison: a SOTA policy that is best on one
    quadrant must also complete all other task lists under the same semantics.
    """
    taskset_reports = []
    for name in taskset_names:
        trace = build_task_trace(name, arrival_mode=arrival_mode, seed=trace_seed)
        report = replay_trace_suite(cache, trace, seed=replay_seed)
        taskset_reports.append(
            {
                "taskset": name,
                "trace_name": trace.name,
                "job_count": len(trace.jobs),
                "results": report["results"],
            }
        )
    policy_rows = _policy_matrix_rows(taskset_reports)
    aggregate_rows = _aggregate_policy_rows(policy_rows)
    aggregate_dominators = _aggregate_sota_pareto_dominators(aggregate_rows)
    return {
        "arrival_mode": arrival_mode,
        "trace_seed": trace_seed,
        "replay_seed": replay_seed,
        "tasksets": [
            {
                "name": row["taskset"],
                "trace_name": row["trace_name"],
                "job_count": row["job_count"],
            }
            for row in taskset_reports
        ],
        "policy_matrix": policy_rows,
        "aggregate_by_policy": aggregate_rows,
        "aggregate_candidate_pareto_dominated_by": aggregate_dominators,
        "aggregate_candidate_not_pareto_dominated": not aggregate_dominators,
        "winner_transfer": _winner_transfer_rows(policy_rows, aggregate_rows),
    }


def replay_trace(
    cache: ServiceRateCache,
    trace: TaskTrace,
    policy: ReplayPolicy,
    *,
    seed: int = 7,
) -> TracePolicyResult:
    rng = random.Random(seed)
    specs = {spec.workload_key: spec for spec in trace.workload_specs()}
    profiles = {key: policy.select_profile(cache, spec).profile for key, spec in specs.items()}
    profile_counts: dict[str, dict[int, int]] = {}
    completions: dict[str, float] = {}
    grouped: dict[str, list[TraceJob]] = {}
    for job in trace.jobs:
        grouped.setdefault(job.workload_key, []).append(job)
    for key, jobs in grouped.items():
        resource_count = specs[key].resource_count
        per_resource = _assign_to_resources(jobs, resource_count)
        for resource_jobs in per_resource:
            completions.update(
                _replay_resource_jobs(
                    cache,
                    workload_key=key,
                    jobs=resource_jobs,
                    target_profile=profiles[key],
                    rng=rng,
                    policy=policy,
                    spec=specs[key],
                    profile_counter=profile_counts.setdefault(key, {}),
                )
            )
    for key, counts in profile_counts.items():
        if counts:
            profiles[key] = _mode_profile(counts)
    flows = [
        completions[job.job_id] - job.arrival_s
        for job in trace.jobs
        if job.job_id in completions
    ]
    completion_times = list(completions.values())
    return TracePolicyResult(
        policy=policy.name,
        policy_config=policy.snapshot(),
        profiles=profiles,
        makespan_s=max(completion_times) if completion_times else 0.0,
        mean_flow_s=mean(flows) if flows else 0.0,
        p50_flow_s=_quantile(flows, 0.50),
        p90_flow_s=_quantile(flows, 0.90),
        completed_jobs=len(completions),
    )


def save_trace(trace: TaskTrace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.snapshot(), indent=2, sort_keys=True), encoding="utf-8")


def load_trace(path: Path) -> TaskTrace:
    return TaskTrace.from_snapshot(json.loads(path.read_text(encoding="utf-8")))


def _arrival_times(
    rng: random.Random,
    *,
    count: int,
    mode: str,
    mean_interarrival_s: float,
) -> list[float]:
    if count <= 0:
        return []
    if mode == "static":
        return [0.0] * count
    if mode == "poisson":
        now = 0.0
        out = []
        for _ in range(count):
            out.append(now)
            now += rng.expovariate(1.0 / max(1e-9, mean_interarrival_s))
        return out
    raise ValueError(f"unknown arrival_mode {mode!r}")


def _assign_to_resources(jobs: list[TraceJob], resource_count: int) -> list[list[TraceJob]]:
    resources: list[list[TraceJob]] = [[] for _ in range(max(1, int(resource_count)))]
    load_heap = [(0, idx) for idx in range(len(resources))]
    heapq.heapify(load_heap)
    for job in sorted(jobs, key=lambda item: (item.arrival_s, item.job_id)):
        count, idx = heapq.heappop(load_heap)
        resources[idx].append(job)
        heapq.heappush(load_heap, (count + 1, idx))
    return resources


def _replay_resource_jobs(
    cache: ServiceRateCache,
    *,
    workload_key: str,
    jobs: list[TraceJob],
    target_profile: int,
    rng: random.Random,
    policy: ReplayPolicy | None = None,
    spec: WorkloadSpec | None = None,
    profile_counter: dict[int, int] | None = None,
) -> dict[str, float]:
    pending = sorted(jobs, key=lambda job: (job.arrival_s, job.job_id))
    next_idx = 0
    now = 0.0
    waiting: list[TraceJob] = []
    active: list[dict[str, Any]] = []
    completions: dict[str, float] = {}
    current_service_profile = max(1, int(target_profile))

    def admit_arrivals(until_s: float) -> None:
        nonlocal next_idx
        while next_idx < len(pending) and pending[next_idx].arrival_s <= until_s + 1e-12:
            waiting.append(pending[next_idx])
            next_idx += 1

    def fill() -> None:
        nonlocal current_service_profile
        total_remaining = len(waiting) + len(active)
        if total_remaining <= 0:
            return
        desired_target = max(1, int(target_profile))
        if policy is not None and spec is not None and policy.uses_statewise_for(spec):
            desired_target = _statewise_target_profile(
                cache=cache,
                spec=spec,
                policy=policy,
                remaining_count=total_remaining,
                active_count=len(active),
                base_target_profile=target_profile,
            )
            if policy.statewise_service_dominance_guard and waiting:
                desired_target = max(int(desired_target), int(target_profile))
        admission_target = max(len(active), min(desired_target, total_remaining))
        current_service_profile = _covering_service_profile(
            cache,
            workload_key=workload_key,
            active_count=admission_target,
            preferred_profile=max(desired_target, admission_target),
        )
        if profile_counter is not None:
            profile_counter[current_service_profile] = profile_counter.get(current_service_profile, 0) + 1
        while waiting and len(active) < admission_target:
            job = waiting.pop(0)
            active.append({"job": job, "remaining": float(job.total_units)})

    while next_idx < len(pending) or waiting or active:
        if not active and not waiting and next_idx < len(pending):
            now = max(now, pending[next_idx].arrival_s)
            admit_arrivals(now)
            fill()
            continue
        admit_arrivals(now)
        fill()
        if not active:
            continue
        rates = _sample_rates(
            cache,
            workload_key,
            len(active),
            rng,
            service_profile_count=current_service_profile,
        )
        completion_dt = min(
            row["remaining"] / max(1e-12, rate)
            for row, rate in zip(active, rates)
        )
        next_arrival_s = pending[next_idx].arrival_s if next_idx < len(pending) else float("inf")
        if now + completion_dt > next_arrival_s:
            dt = max(0.0, next_arrival_s - now)
            now = next_arrival_s
            for row, rate in zip(active, rates):
                row["remaining"] = max(0.0, row["remaining"] - rate * dt)
            admit_arrivals(now)
            fill()
            continue
        now += completion_dt
        still_active: list[dict[str, Any]] = []
        for row, rate in zip(active, rates):
            row["remaining"] = max(0.0, row["remaining"] - rate * completion_dt)
            if row["remaining"] <= 1e-9:
                completions[row["job"].job_id] = now
            else:
                still_active.append(row)
        active = still_active
        fill()
    return completions


def _mode_profile(counts: dict[int, int]) -> int:
    if not counts:
        return 1
    return max(sorted(counts), key=lambda profile: (counts[profile], -profile))


def _sample_rates(
    cache: ServiceRateCache,
    workload_key: str,
    active_count: int,
    rng: random.Random,
    *,
    service_profile_count: int | None = None,
) -> list[float]:
    service_count = int(service_profile_count or active_count)
    if service_count < int(active_count):
        raise KeyError(
            f"service profile {service_count}/resource cannot cover "
            f"{active_count} active jobs for {workload_key!r}"
        )
    profile = cache.get(workload_key, service_count)
    if profile is None or profile.capacity_boundary or profile.aggregate_rate <= 0:
        raise KeyError(
            f"missing exact service profile for {workload_key!r} at {service_count}/resource; "
            "task-list benchmark does not interpolate co-location service"
        )
    rates = [float(x) for x in profile.per_task_rates if float(x) > 0]
    if not rates:
        rates = [profile.mean_rate] * max(1, profile.profile)
    sampled = [rates[rng.randrange(len(rates))] for _ in range(active_count)]
    active_fraction = float(active_count) / float(max(1, profile.profile))
    target_aggregate = float(profile.aggregate_rate) * active_fraction
    scale = target_aggregate / max(1e-12, sum(sampled))
    return [max(1e-12, rate * scale) for rate in sampled]


def _relative(results: list[TracePolicyResult], *, baseline_name: str) -> dict[str, dict[str, float]]:
    baseline = next((row for row in results if row.policy == baseline_name), None)
    if baseline is None:
        return {}
    out = {}
    for row in results:
        out[row.policy] = {
            "makespan_improvement": baseline.makespan_s / row.makespan_s if row.makespan_s > 0 else 0.0,
            "mean_flow_improvement": baseline.mean_flow_s / row.mean_flow_s if row.mean_flow_s > 0 else 0.0,
            "p90_flow_improvement": baseline.p90_flow_s / row.p90_flow_s if row.p90_flow_s > 0 else 0.0,
        }
    return out


def _candidate_name(results: list[TracePolicyResult]) -> str:
    for row in results:
        if row.policy.startswith("calibrated_"):
            return row.policy
    return results[0].policy if results else ""


def _sota_tasklist_comparison(results: list[TracePolicyResult]) -> dict[str, Any]:
    candidate = next((row for row in results if row.policy.startswith("calibrated_")), None)
    if candidate is None:
        return {
            "candidate_policy": "",
            "rows": [],
            "candidate_pareto_dominated_by": [],
            "candidate_not_pareto_dominated": False,
        }
    metadata = {spec.policy.name: spec for spec in sota_baseline_specs()}
    rows = []
    for result in results:
        spec = metadata.get(result.policy)
        if spec is None:
            continue
        makespan_ratio = _improvement(result.makespan_s, candidate.makespan_s)
        mean_flow_ratio = _improvement(result.mean_flow_s, candidate.mean_flow_s)
        p90_ratio = _improvement(result.p90_flow_s, candidate.p90_flow_s)
        rows.append(
            {
                "baseline": spec.snapshot(),
                "baseline_policy": result.policy,
                "candidate_policy": candidate.policy,
                "baseline_profiles": result.profiles,
                "candidate_profiles": candidate.profiles,
                "baseline_makespan_s": result.makespan_s,
                "candidate_makespan_s": candidate.makespan_s,
                "candidate_vs_baseline_makespan": makespan_ratio,
                "baseline_mean_flow_s": result.mean_flow_s,
                "candidate_mean_flow_s": candidate.mean_flow_s,
                "candidate_vs_baseline_mean_flow": mean_flow_ratio,
                "baseline_p90_flow_s": result.p90_flow_s,
                "candidate_p90_flow_s": candidate.p90_flow_s,
                "candidate_vs_baseline_p90_flow": p90_ratio,
            }
        )
    dominators = _sota_pareto_dominators(rows)
    return {
        "candidate_policy": candidate.policy,
        "candidate_profiles": candidate.profiles,
        "rows": rows,
        "candidate_pareto_dominated_by": dominators,
        "candidate_not_pareto_dominated": not dominators,
    }


def _sota_pareto_dominators(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dominators = []
    floor = 1.0 - PARETO_TOLERANCE
    for row in rows:
        makespan = float(row["candidate_vs_baseline_makespan"])
        mean_flow = float(row["candidate_vs_baseline_mean_flow"])
        baseline = row["baseline"]
        if makespan < floor and mean_flow < floor:
            dominators.append(
                {
                    "name": baseline["name"],
                    "policy": row["baseline_policy"],
                    "representative_systems": baseline["representative_systems"],
                    "candidate_vs_baseline_makespan": makespan,
                    "candidate_vs_baseline_mean_flow": mean_flow,
                    "baseline_profiles": row["baseline_profiles"],
                    "candidate_profiles": row["candidate_profiles"],
                }
            )
    return dominators


def _improvement(baseline_value: float, candidate_value: float) -> float:
    return baseline_value / candidate_value if candidate_value > 0 else 0.0


def _policy_matrix_rows(taskset_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = _policy_metadata()
    rows: list[dict[str, Any]] = []
    for report in taskset_reports:
        results = report["results"]
        candidate = next(row for row in results if row["policy"].startswith("calibrated_"))
        for result in results:
            policy = result["policy"]
            policy_key = _policy_key(policy)
            policy_meta = _policy_row_metadata(policy, metadata)
            row = {
                "taskset": report["taskset"],
                "job_count": report["job_count"],
                "policy": policy,
                "policy_key": policy_key,
                "baseline_name": policy_meta["baseline_name"],
                "representative_systems": policy_meta["representative_systems"],
                "policy_family": policy_meta["policy_family"],
                "profiles": result["profiles"],
                "makespan_s": result["makespan_s"],
                "mean_flow_s": result["mean_flow_s"],
                "p90_flow_s": result["p90_flow_s"],
                "completed_jobs": result["completed_jobs"],
                "candidate_policy": candidate["policy"],
                "candidate_makespan_s": candidate["makespan_s"],
                "candidate_mean_flow_s": candidate["mean_flow_s"],
                "candidate_p90_flow_s": candidate["p90_flow_s"],
                "candidate_vs_policy_makespan": _improvement(
                    result["makespan_s"],
                    candidate["makespan_s"],
                ),
                "candidate_vs_policy_mean_flow": _improvement(
                    result["mean_flow_s"],
                    candidate["mean_flow_s"],
                ),
                "candidate_vs_policy_p90_flow": _improvement(
                    result["p90_flow_s"],
                    candidate["p90_flow_s"],
                ),
            }
            rows.append(row)
    return rows


def _aggregate_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["policy_key"], []).append(row)
    out = []
    for policy_key, policy_rows in sorted(grouped.items()):
        job_count = sum(int(row["completed_jobs"]) for row in policy_rows)
        total_makespan = sum(float(row["makespan_s"]) for row in policy_rows)
        weighted_mean_flow = (
            sum(float(row["mean_flow_s"]) * int(row["completed_jobs"]) for row in policy_rows)
            / max(1, job_count)
        )
        candidate_total_makespan = sum(float(row["candidate_makespan_s"]) for row in policy_rows)
        candidate_weighted_mean_flow = (
            sum(float(row["candidate_mean_flow_s"]) * int(row["completed_jobs"]) for row in policy_rows)
            / max(1, job_count)
        )
        ratios_makespan = [float(row["candidate_vs_policy_makespan"]) for row in policy_rows]
        ratios_mean_flow = [float(row["candidate_vs_policy_mean_flow"]) for row in policy_rows]
        first = policy_rows[0]
        out.append(
            {
                "policy": policy_key,
                "policies": sorted({row["policy"] for row in policy_rows}),
                "baseline_name": first["baseline_name"],
                "representative_systems": first["representative_systems"],
                "policy_family": first["policy_family"],
                "taskset_count": len(policy_rows),
                "completed_jobs": job_count,
                "sum_makespan_s": total_makespan,
                "job_weighted_mean_flow_s": weighted_mean_flow,
                "candidate_sum_makespan_s": candidate_total_makespan,
                "candidate_job_weighted_mean_flow_s": candidate_weighted_mean_flow,
                "candidate_vs_policy_sum_makespan": _improvement(
                    total_makespan,
                    candidate_total_makespan,
                ),
                "candidate_vs_policy_job_weighted_mean_flow": _improvement(
                    weighted_mean_flow,
                    candidate_weighted_mean_flow,
                ),
                "candidate_vs_policy_geom_makespan": _geomean(ratios_makespan),
                "candidate_vs_policy_geom_mean_flow": _geomean(ratios_mean_flow),
            }
        )
    return out


def _winner_transfer_rows(
    policy_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregate_by_policy = {row["policy"]: row for row in aggregate_rows}
    tasksets = sorted({row["taskset"] for row in policy_rows})
    out = []
    for taskset in tasksets:
        sota_rows = [
            row for row in policy_rows
            if row["taskset"] == taskset and row["policy_family"] == "sota_style"
        ]
        for objective, metric in (
            ("best_all_job_makespan", "makespan_s"),
            ("best_mean_flow", "mean_flow_s"),
        ):
            winner = min(sota_rows, key=lambda row: float(row[metric]))
            best_value = float(winner[metric])
            tied = [
                row["policy"] for row in sota_rows
                if abs(float(row[metric]) - best_value) <= 1e-9
            ]
            aggregate = aggregate_by_policy[winner["policy"]]
            out.append(
                {
                    "source_taskset": taskset,
                    "objective": objective,
                    "winner_policy": winner["policy"],
                    "winner_baseline_name": winner["baseline_name"],
                    "tied_winner_policies": tied,
                    "source_metric_s": best_value,
                    "winner_profiles_on_source": winner["profiles"],
                    "aggregate_completed_jobs": aggregate["completed_jobs"],
                    "candidate_vs_winner_sum_makespan": aggregate[
                        "candidate_vs_policy_sum_makespan"
                    ],
                    "candidate_vs_winner_job_weighted_mean_flow": aggregate[
                        "candidate_vs_policy_job_weighted_mean_flow"
                    ],
                    "candidate_vs_winner_geom_makespan": aggregate[
                        "candidate_vs_policy_geom_makespan"
                    ],
                    "candidate_vs_winner_geom_mean_flow": aggregate[
                        "candidate_vs_policy_geom_mean_flow"
                    ],
                }
            )
    return out


def _aggregate_sota_pareto_dominators(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dominators = []
    floor = 1.0 - PARETO_TOLERANCE
    for row in rows:
        if row["policy_family"] != "sota_style":
            continue
        makespan = float(row["candidate_vs_policy_sum_makespan"])
        mean_flow = float(row["candidate_vs_policy_job_weighted_mean_flow"])
        if makespan < floor and mean_flow < floor:
            dominators.append(
                {
                    "policy": row["policy"],
                    "baseline_name": row["baseline_name"],
                    "representative_systems": row["representative_systems"],
                    "candidate_vs_policy_sum_makespan": makespan,
                    "candidate_vs_policy_job_weighted_mean_flow": mean_flow,
                }
            )
    return dominators


def _policy_metadata() -> dict[str, dict[str, Any]]:
    metadata = {
        "legacy_fixed_caps": {
            "baseline_name": "Scheduleurm legacy",
            "representative_systems": [],
            "policy_family": "legacy",
        }
    }
    for spec in sota_baseline_specs():
        metadata[spec.policy.name] = {
            "baseline_name": spec.name,
            "representative_systems": list(spec.representative_systems),
            "policy_family": "sota_style",
        }
    return metadata


def _policy_key(policy: str) -> str:
    if policy.startswith("calibrated_"):
        return "scheduleurm_candidate"
    return policy


def _policy_row_metadata(policy: str, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if policy.startswith("calibrated_"):
        return {
            "baseline_name": "Scheduleurm candidate",
            "representative_systems": [],
            "policy_family": "candidate",
        }
    return metadata.get(
        policy,
        {
            "baseline_name": policy,
            "representative_systems": [],
            "policy_family": "unknown",
        },
    )


def _geomean(values: list[float]) -> float:
    positives = [max(1e-12, float(value)) for value in values]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def _positive_lognormal(rng: random.Random, mean_value: float, cv: float) -> float:
    cv = max(0.0, float(cv))
    if cv <= 0:
        return float(mean_value)
    sigma = min(1.0, cv)
    mu_adjust = -0.5 * sigma * sigma
    return max(1e-9, float(mean_value) * rng.lognormvariate(mu_adjust, sigma))


def _mean_interarrival_s(task_count: int) -> float:
    return 60.0 if task_count <= 64 else 15.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(q)))))
    return ordered[idx]
