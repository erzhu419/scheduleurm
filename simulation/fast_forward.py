"""Event-driven fast-forward replay using cached service curves."""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from statistics import mean, median
from typing import Any

from .service_cache import ProfileRecord, ServiceRateCache


@dataclass(frozen=True)
class WorkloadSpec:
    workload_key: str
    resource_kind: str
    task_count: int
    total_units: float
    resource_count: int = 1
    variation_cv: float = 0.05


@dataclass(frozen=True)
class ReplayPolicy:
    name: str
    fixed_profiles: dict[str, int] | None = None
    calibrated: bool = False
    calibrated_objective: str = "makespan"
    max_makespan_regret: float = 0.0
    statewise_regret_slack: float = 0.0
    statewise: bool = False
    guarded_resource_kinds: tuple[str, ...] = ()
    statewise_resource_kinds: tuple[str, ...] = ()

    def select_profile(self, cache: ServiceRateCache, spec: WorkloadSpec) -> ProfileRecord:
        if self.calibrated:
            if self.uses_guarded_objective(spec):
                return cache.best_profile_for_guarded_mean_flow(
                    spec.workload_key,
                    task_count=spec.task_count,
                    total_units=spec.total_units,
                    resource_count=spec.resource_count,
                    max_makespan_regret=self.max_makespan_regret,
                )
            return cache.best_profile_for_makespan(
                spec.workload_key,
                task_count=spec.task_count,
                total_units=spec.total_units,
                resource_count=spec.resource_count,
            )
        requested = int((self.fixed_profiles or {}).get(spec.workload_key, 1))
        record = cache.get(spec.workload_key, requested)
        if record is not None and not record.capacity_boundary and record.aggregate_rate > 0:
            return record
        if spec.resource_kind in ("hybrid_rl", "gpu_heavy"):
            raise KeyError(
                f"missing exact service profile for {spec.workload_key!r} at {requested}/resource; "
                "legacy/adaptive GPU co-location caps must be measured in the real environment"
            )
        candidates = cache.profiles(spec.workload_key)
        if not candidates:
            raise KeyError(f"no service cache entries for workload {spec.workload_key!r}")
        return min(candidates, key=lambda r: (abs(r.profile - requested), r.profile))

    def uses_guarded_objective(self, spec: WorkloadSpec) -> bool:
        return (
            self.calibrated_objective == "guarded_mean_flow"
            or spec.resource_kind in set(self.guarded_resource_kinds)
        )

    def uses_statewise_for(self, spec: WorkloadSpec) -> bool:
        if not self.statewise:
            return False
        if not self.statewise_resource_kinds:
            return True
        return spec.resource_kind in set(self.statewise_resource_kinds)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fixed_profiles": dict(self.fixed_profiles or {}),
            "calibrated": self.calibrated,
            "calibrated_objective": self.calibrated_objective,
            "max_makespan_regret": self.max_makespan_regret,
            "statewise_regret_slack": self.statewise_regret_slack,
            "statewise": self.statewise,
            "guarded_resource_kinds": list(self.guarded_resource_kinds),
            "statewise_resource_kinds": list(self.statewise_resource_kinds),
        }


@dataclass
class WorkloadReplayResult:
    workload_key: str
    resource_kind: str
    policy: str
    selected_profile: int
    task_count: int
    makespan_s: float
    mean_flow_s: float
    p50_completion_s: float
    p90_completion_s: float
    profile_trace: tuple[int, ...] = ()

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "resource_kind": self.resource_kind,
            "policy": self.policy,
            "selected_profile": self.selected_profile,
            "task_count": self.task_count,
            "makespan_s": self.makespan_s,
            "mean_flow_s": self.mean_flow_s,
            "p50_completion_s": self.p50_completion_s,
            "p90_completion_s": self.p90_completion_s,
            "profile_trace": list(self.profile_trace[:128]),
            "profile_trace_len": len(self.profile_trace),
            "profile_trace_counts": _profile_trace_counts(self.profile_trace),
        }


@dataclass
class PolicyReplaySummary:
    policy: str
    workloads: list[WorkloadReplayResult]
    policy_config: dict[str, Any] = field(default_factory=dict)

    @property
    def total_makespan_s(self) -> float:
        return max((x.makespan_s for x in self.workloads), default=0.0)

    @property
    def weighted_mean_flow_s(self) -> float:
        total = sum(x.task_count for x in self.workloads)
        if total <= 0:
            return 0.0
        return sum(x.mean_flow_s * x.task_count for x in self.workloads) / float(total)

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "policy_config": self.policy_config,
            "total_makespan_s": self.total_makespan_s,
            "weighted_mean_flow_s": self.weighted_mean_flow_s,
            "workloads": [x.snapshot() for x in self.workloads],
        }


@dataclass
class ReplayComparison:
    baseline: PolicyReplaySummary
    candidate: PolicyReplaySummary

    @property
    def makespan_improvement(self) -> float:
        cand = self.candidate.total_makespan_s
        return self.baseline.total_makespan_s / cand if cand > 0 else 0.0

    @property
    def mean_flow_improvement(self) -> float:
        cand = self.candidate.weighted_mean_flow_s
        return self.baseline.weighted_mean_flow_s / cand if cand > 0 else 0.0

    @property
    def per_workload_improvements(self) -> dict[str, dict[str, float]]:
        baseline = {row.workload_key: row for row in self.baseline.workloads}
        candidate = {row.workload_key: row for row in self.candidate.workloads}
        out: dict[str, dict[str, float]] = {}
        for key in sorted(set(baseline) & set(candidate)):
            base = baseline[key]
            cand = candidate[key]
            out[key] = {
                "makespan_improvement": base.makespan_s / cand.makespan_s if cand.makespan_s > 0 else 0.0,
                "mean_flow_improvement": base.mean_flow_s / cand.mean_flow_s if cand.mean_flow_s > 0 else 0.0,
                "baseline_profile": float(base.selected_profile),
                "candidate_profile": float(cand.selected_profile),
            }
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.snapshot(),
            "candidate": self.candidate.snapshot(),
            "makespan_improvement": self.makespan_improvement,
            "mean_flow_improvement": self.mean_flow_improvement,
            "per_workload_improvements": self.per_workload_improvements,
        }


def compare_policies(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    *,
    baseline: ReplayPolicy,
    candidate: ReplayPolicy,
    trials: int = 101,
    seed: int = 7,
) -> ReplayComparison:
    return ReplayComparison(
        baseline=_summarize_policy(cache, specs, baseline, trials=trials, seed=seed),
        candidate=_summarize_policy(cache, specs, candidate, trials=trials, seed=seed),
    )


def _summarize_policy(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    policy: ReplayPolicy,
    *,
    trials: int,
    seed: int,
) -> PolicyReplaySummary:
    results = []
    for idx, spec in enumerate(specs):
        profile = policy.select_profile(cache, spec)
        trial_results = [
            replay_workload(cache, spec, policy, seed=seed + idx * 100000 + t)
            for t in range(max(1, int(trials)))
        ]
        results.append(_median_result(trial_results, profile.profile))
    return PolicyReplaySummary(policy=policy.name, workloads=results, policy_config=policy.snapshot())


def replay_workload(
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    *,
    seed: int = 0,
) -> WorkloadReplayResult:
    rng = random.Random(seed)
    profile = policy.select_profile(cache, spec)
    tasks = [
        {
            "remaining": _positive_lognormal(rng, spec.total_units, spec.variation_cv),
            "completion": 0.0,
        }
        for _ in range(max(0, int(spec.task_count)))
    ]
    waiting = list(range(len(tasks)))
    completion_times: list[float] = []
    profile_trace: list[int] = []
    resource_times = [0.0 for _ in range(max(1, int(spec.resource_count)))]
    for resource_idx in range(len(resource_times)):
        local_waiting = waiting[resource_idx::len(resource_times)]
        local_completions, local_trace = _replay_one_resource(
            cache=cache,
            spec=spec,
            policy=policy,
            target_profile=profile.profile,
            task_ids=local_waiting,
            tasks=tasks,
            rng=rng,
        )
        completion_times.extend(local_completions)
        profile_trace.extend(local_trace)
    completion_times.sort()
    selected_profile = max(profile_trace) if profile_trace else profile.profile
    return WorkloadReplayResult(
        workload_key=spec.workload_key,
        resource_kind=spec.resource_kind,
        policy=policy.name,
        selected_profile=selected_profile,
        task_count=spec.task_count,
        makespan_s=max(completion_times) if completion_times else 0.0,
        mean_flow_s=mean(completion_times) if completion_times else 0.0,
        p50_completion_s=median(completion_times) if completion_times else 0.0,
        p90_completion_s=_quantile(completion_times, 0.90),
        profile_trace=tuple(profile_trace),
    )


def _replay_one_resource(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    target_profile: int,
    task_ids: list[int],
    tasks: list[dict[str, float]],
    rng: random.Random,
) -> tuple[list[float], list[int]]:
    now = 0.0
    waiting = list(task_ids)
    active: list[int] = []
    completions: list[float] = []
    profile_trace: list[int] = []

    def fill() -> None:
        total_remaining = len(waiting) + len(active)
        if total_remaining <= 0:
            return
        target = target_profile
        if policy.uses_statewise_for(spec):
            target = _statewise_target_profile(
                cache=cache,
                spec=spec,
                policy=policy,
                remaining_count=total_remaining,
                active_count=len(active),
            )
        target = max(len(active), min(max(1, int(target)), total_remaining))
        profile_trace.append(target)
        while waiting and len(active) < target:
            active.append(waiting.pop(0))

    fill()
    while active:
        m = len(active)
        rates = _sample_rates(cache, spec.workload_key, m, rng)
        event_times = []
        for tid, rate in zip(active, rates):
            event_times.append(tasks[tid]["remaining"] / max(1e-12, rate))
        dt = min(event_times)
        now += dt
        still_active = []
        for tid, rate in zip(active, rates):
            tasks[tid]["remaining"] = max(0.0, tasks[tid]["remaining"] - rate * dt)
            if tasks[tid]["remaining"] <= 1e-9:
                tasks[tid]["completion"] = now
                completions.append(now)
            else:
                still_active.append(tid)
        active = still_active
        fill()
    return completions, profile_trace


def _statewise_target_profile(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    remaining_count: int,
    active_count: int,
) -> int:
    local_spec = WorkloadSpec(
        workload_key=spec.workload_key,
        resource_kind=spec.resource_kind,
        task_count=max(1, int(remaining_count)),
        total_units=spec.total_units,
        resource_count=1,
        variation_cv=spec.variation_cv,
    )
    if policy.calibrated and policy.uses_guarded_objective(local_spec):
        record = cache.best_profile_for_guarded_mean_flow(
            local_spec.workload_key,
            task_count=local_spec.task_count,
            total_units=local_spec.total_units,
            resource_count=local_spec.resource_count,
            max_makespan_regret=(
                policy.max_makespan_regret
                + max(0.0, float(policy.statewise_regret_slack)) / float(max(1, remaining_count))
            ),
        )
        return max(int(active_count), int(record.profile))
    record = policy.select_profile(cache, local_spec)
    return max(int(active_count), int(record.profile))


def _profile_trace_counts(trace: tuple[int, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in trace:
        key = str(int(profile))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sample_rates(cache: ServiceRateCache, workload_key: str, active_count: int, rng: random.Random) -> list[float]:
    profile = cache.get(workload_key, active_count)
    if profile is None or profile.capacity_boundary or profile.aggregate_rate <= 0:
        raise KeyError(
            f"missing exact service profile for {workload_key!r} at {active_count}/resource; "
            "multi-task co-location rates must be measured in the real environment"
        )
    rates = [float(x) for x in profile.per_task_rates if float(x) > 0]
    if not rates:
        rates = [profile.mean_rate] * max(1, profile.profile)
    sampled = [rates[rng.randrange(len(rates))] for _ in range(active_count)]
    scale = profile.aggregate_rate / max(1e-12, sum(sampled))
    return [max(1e-12, rate * scale) for rate in sampled]


def _median_result(results: list[WorkloadReplayResult], selected_profile: int) -> WorkloadReplayResult:
    ordered = sorted(results, key=lambda r: r.makespan_s)
    result = ordered[len(ordered) // 2]
    if not result.profile_trace:
        result.selected_profile = selected_profile
    return result


def _positive_lognormal(rng: random.Random, mean_value: float, cv: float) -> float:
    cv = max(0.0, float(cv))
    if cv <= 0:
        return float(mean_value)
    sigma = min(1.0, cv)
    mu_adjust = -0.5 * sigma * sigma
    return max(1e-9, float(mean_value) * rng.lognormvariate(mu_adjust, sigma))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(q)))))
    return ordered[idx]
