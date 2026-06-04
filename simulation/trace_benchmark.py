"""Explicit task-list benchmark replay.

This layer mirrors the common systems-paper workflow: build a workload trace,
then replay several scheduling policies on the same task list using measured
service curves. It complements the aggregate fast-forward replay.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .defaults import calibrated_candidate_policy, legacy_policy
from .fast_forward import ReplayPolicy, WorkloadSpec
from .service_cache import ServiceRateCache
from .sota_baselines import sota_baseline_specs
from .tasksets import TaskSet, taskset_by_name


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
    return {
        "trace": trace.snapshot(),
        "results": [result.snapshot() for result in results],
        "relative_to_legacy": _relative(results, baseline_name="legacy_fixed_caps"),
        "relative_to_candidate": _relative(results, baseline_name=_candidate_name(results)),
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
    profiles = {
        key: policy.select_profile(cache, spec).profile
        for key, spec in specs.items()
    }
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
                )
            )
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
) -> dict[str, float]:
    pending = sorted(jobs, key=lambda job: (job.arrival_s, job.job_id))
    next_idx = 0
    now = 0.0
    waiting: list[TraceJob] = []
    active: list[dict[str, Any]] = []
    completions: dict[str, float] = {}

    def admit_arrivals(until_s: float) -> None:
        nonlocal next_idx
        while next_idx < len(pending) and pending[next_idx].arrival_s <= until_s + 1e-12:
            waiting.append(pending[next_idx])
            next_idx += 1

    def fill() -> None:
        while waiting and len(active) < target_profile:
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
        rates = _sample_rates(cache, workload_key, len(active), rng)
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


def _sample_rates(
    cache: ServiceRateCache,
    workload_key: str,
    active_count: int,
    rng: random.Random,
) -> list[float]:
    profile = cache.get(workload_key, active_count)
    if profile is None or profile.capacity_boundary or profile.aggregate_rate <= 0:
        raise KeyError(
            f"missing exact service profile for {workload_key!r} at {active_count}/resource; "
            "task-list benchmark does not interpolate co-location service"
        )
    rates = [float(x) for x in profile.per_task_rates if float(x) > 0]
    if not rates:
        rates = [profile.mean_rate] * max(1, profile.profile)
    sampled = [rates[rng.randrange(len(rates))] for _ in range(active_count)]
    scale = profile.aggregate_rate / max(1e-12, sum(sampled))
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
