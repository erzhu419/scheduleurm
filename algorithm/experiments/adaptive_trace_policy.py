"""Queue-adaptive replay policy for the Scheduleurm candidate algorithm.

The legacy scheduler is intentionally not changed here.  This module provides
the experiment-side policy semantics used to test the robust MaxWeight route:
large queues prioritize measured service, while bounded profile penalties allow
low-concurrency actions in finite sublevel sets.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import mean
from typing import Any, Mapping

from simulation.fast_forward import ReplayPolicy
from simulation.service_cache import ProfileRecord, ServiceRateCache
from simulation.trace_benchmark import (
    TaskTrace,
    TraceJob,
    TracePolicyResult,
    _assign_to_resources,
    _quantile,
    _sample_rates,
)


@dataclass(frozen=True)
class AdaptiveMaxWeightReplayPolicy:
    """Measured-service MaxWeight with a bounded profile penalty.

    ``support_regret_tolerance`` defines the finite candidate support envelope:
    actions below ``(1-tolerance)`` times the best measured aggregate service
    for the workload are not used.  ``profile_penalty_fraction`` is dimensionless
    and is multiplied by the best measured service, giving a bounded penalty
    term in the same service units as the MaxWeight score.
    """

    name: str = "calibrated_adaptive_maxweight_penalty"
    support_regret_tolerance: float = 0.04
    profile_penalty_fraction: float = 0.10
    minimum_backlog_weight: float = 1.0
    tie_break: str = "low_profile"

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "support_regret_tolerance": self.support_regret_tolerance,
            "profile_penalty_fraction": self.profile_penalty_fraction,
            "minimum_backlog_weight": self.minimum_backlog_weight,
            "tie_break": self.tie_break,
            "policy_semantics": (
                "argmax_k Q_i * lower_service_i(k) - bounded_profile_penalty(k) "
                "over measured support-envelope profiles"
            ),
        }


def calibrated_adaptive_maxweight_policy() -> AdaptiveMaxWeightReplayPolicy:
    return AdaptiveMaxWeightReplayPolicy()


def replay_adaptive_trace(
    cache: ServiceRateCache,
    trace: TaskTrace,
    policy: AdaptiveMaxWeightReplayPolicy,
    *,
    seed: int = 7,
) -> tuple[TracePolicyResult, dict[str, float], dict[str, dict[int, int]]]:
    rng = random.Random(seed)
    specs = {spec.workload_key: spec for spec in trace.workload_specs()}
    grouped: dict[str, list[TraceJob]] = {}
    for job in trace.jobs:
        grouped.setdefault(job.workload_key, []).append(job)
    completions: dict[str, float] = {}
    profile_counts: dict[str, dict[int, int]] = {}
    for key, jobs in grouped.items():
        resource_count = specs[key].resource_count
        per_resource = _assign_to_resources(jobs, resource_count)
        for resource_jobs in per_resource:
            local_completions, local_counts = _replay_resource_jobs_adaptive(
                cache,
                workload_key=key,
                jobs=resource_jobs,
                policy=policy,
                rng=rng,
            )
            completions.update(local_completions)
            counts = profile_counts.setdefault(key, {})
            for profile, count in local_counts.items():
                counts[profile] = counts.get(profile, 0) + count
    flows = [
        completions[job.job_id] - job.arrival_s
        for job in trace.jobs
        if job.job_id in completions
    ]
    completion_times = list(completions.values())
    representative_profiles = {
        key: _mode_profile(counts)
        for key, counts in profile_counts.items()
    }
    result = TracePolicyResult(
        policy=policy.name,
        policy_config={
            **policy.snapshot(),
            "adaptive_profile_counts": {
                key: {str(profile): count for profile, count in sorted(counts.items())}
                for key, counts in profile_counts.items()
            },
        },
        profiles=representative_profiles,
        makespan_s=max(completion_times) if completion_times else 0.0,
        mean_flow_s=mean(flows) if flows else 0.0,
        p50_flow_s=_quantile(flows, 0.50),
        p90_flow_s=_quantile(flows, 0.90),
        completed_jobs=len(completions),
    )
    return result, completions, profile_counts


def robust_maxweight_profile(
    cache: ServiceRateCache,
    workload_key: str,
    backlog_jobs: int,
    policy: AdaptiveMaxWeightReplayPolicy,
) -> int:
    candidates = _support_envelope(cache, workload_key, policy)
    if not candidates:
        raise KeyError(f"no measured feasible profiles for workload {workload_key!r}")
    best_service = max(float(record.aggregate_rate) for record in candidates)
    q_weight = max(float(policy.minimum_backlog_weight), float(backlog_jobs))

    def score(record: ProfileRecord) -> tuple[float, float, float, int]:
        service = float(record.aggregate_rate)
        penalty = (
            max(0.0, float(policy.profile_penalty_fraction))
            * best_service
            * float(record.profile)
        )
        if policy.tie_break == "high_profile":
            tie = float(record.profile)
        elif policy.tie_break == "high_service":
            tie = service
        else:
            tie = -float(record.profile)
        return (q_weight * service - penalty, tie, -float(record.profile), -int(record.profile))

    return max(candidates, key=score).profile


def replay_policy_like_snapshot(policy: ReplayPolicy | AdaptiveMaxWeightReplayPolicy) -> dict[str, Any]:
    return policy.snapshot()


def _replay_resource_jobs_adaptive(
    cache: ServiceRateCache,
    *,
    workload_key: str,
    jobs: list[TraceJob],
    policy: AdaptiveMaxWeightReplayPolicy,
    rng: random.Random,
) -> tuple[dict[str, float], dict[int, int]]:
    pending = sorted(jobs, key=lambda job: (job.arrival_s, job.job_id))
    next_idx = 0
    now = 0.0
    waiting: list[TraceJob] = []
    active: list[dict[str, Any]] = []
    completions: dict[str, float] = {}
    profile_counts: dict[int, int] = {}

    def admit_arrivals(until_s: float) -> None:
        nonlocal next_idx
        while next_idx < len(pending) and pending[next_idx].arrival_s <= until_s + 1e-12:
            waiting.append(pending[next_idx])
            next_idx += 1

    def selected_limit(*, count: bool = True) -> int:
        backlog = len(waiting) + len(active)
        if backlog <= 0:
            return 1
        profile = robust_maxweight_profile(cache, workload_key, backlog, policy)
        if count:
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
        return max(1, int(profile))

    def fill() -> None:
        limit = selected_limit()
        while waiting and len(active) < limit:
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
        service_profile = _covering_service_profile(
            cache,
            workload_key=workload_key,
            active_count=len(active),
            preferred_profile=selected_limit(count=False),
        )
        profile_counts[service_profile] = profile_counts.get(service_profile, 0) + 1
        rates = _sample_rates(
            cache,
            workload_key,
            len(active),
            rng,
            service_profile_count=service_profile,
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
    return completions, profile_counts


def _support_envelope(
    cache: ServiceRateCache,
    workload_key: str,
    policy: AdaptiveMaxWeightReplayPolicy,
) -> list[ProfileRecord]:
    records = [
        record for record in cache.profiles(workload_key)
        if not record.capacity_boundary and float(record.aggregate_rate) > 0.0
    ]
    if not records:
        return []
    best = max(float(record.aggregate_rate) for record in records)
    threshold = (1.0 - max(0.0, min(0.95, float(policy.support_regret_tolerance)))) * best
    envelope = [record for record in records if float(record.aggregate_rate) >= threshold]
    return envelope or records


def _covering_service_profile(
    cache: ServiceRateCache,
    *,
    workload_key: str,
    active_count: int,
    preferred_profile: int,
) -> int:
    """Return a measured profile that can certify service for current active jobs."""

    active = max(1, int(active_count))
    preferred = max(1, int(preferred_profile))
    preferred_record = cache.get(workload_key, preferred)
    if (
        preferred >= active
        and preferred_record is not None
        and not preferred_record.capacity_boundary
        and float(preferred_record.aggregate_rate) > 0.0
    ):
        return preferred
    feasible = [
        record.profile
        for record in cache.profiles(workload_key)
        if int(record.profile) >= active
        and not record.capacity_boundary
        and float(record.aggregate_rate) > 0.0
    ]
    if not feasible:
        raise KeyError(
            f"no measured feasible service profile for {workload_key!r} "
            f"covering {active}/resource active jobs"
        )
    return min(feasible)


def _mode_profile(counts: Mapping[int, int]) -> int:
    if not counts:
        return 1
    return max(counts.items(), key=lambda row: (row[1], -row[0]))[0]
