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
    min_makespan_regret: float = 0.0
    statewise_regret_slack: float = 0.0
    statewise: bool = False
    guarded_resource_kinds: tuple[str, ...] = ()
    statewise_resource_kinds: tuple[str, ...] = ()
    statewise_workload_keys: tuple[str, ...] = ()
    statewise_excluded_workload_keys: tuple[str, ...] = ()
    backlog_aware_guard: bool = False
    backlog_reference_tasks: int = 64
    statewise_service_dominance_guard: bool = False

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
            if self.calibrated_objective == "mean_flow":
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
        if spec.workload_key in set(self.statewise_excluded_workload_keys):
            return False
        if spec.workload_key in set(self.statewise_workload_keys):
            return True
        if not self.statewise_resource_kinds:
            return True
        return spec.resource_kind in set(self.statewise_resource_kinds)

    def trajectory_policy_for(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
        *,
        base_target_profile: int | None = None,
    ) -> "ReplayPolicy | None":
        """Return the policy that owns this action's statewise trajectory.

        Ordinary policies are self-contained and return ``None``.  Candidate-set
        policies can use this hook to select over finite actions that share the
        same measured co-location profile but use different admission/drain
        semantics.
        """

        return None

    def statewise_target_profile_for(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
        *,
        remaining_count: int,
        active_count: int,
        base_target_profile: int | None = None,
    ) -> int | None:
        """Optional custom statewise target for finite trajectory actions."""

        return None

    def statewise_holds_base_profile_for(
        self,
        *,
        waiting_count: int,
        total_remaining: int,
    ) -> bool | None:
        """Optional custom base-profile hold rule for trajectory actions."""

        return None

    def uses_shortest_remaining_first_for(self, spec: WorkloadSpec) -> bool:
        """Whether this finite trajectory action orders admissions by ETA."""

        return False

    def waiting_order_mode_for(self, spec: WorkloadSpec) -> str:
        if self.uses_shortest_remaining_first_for(spec):
            return "shortest_remaining_first"
        return "fifo"

    def resource_assignment_mode_for(self, spec: WorkloadSpec) -> str:
        """How queued jobs are partitioned across identical resources."""

        return "round_robin_count"

    def resource_assignment_seed_for(self, spec: WorkloadSpec) -> int:
        return 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fixed_profiles": dict(self.fixed_profiles or {}),
            "calibrated": self.calibrated,
            "calibrated_objective": self.calibrated_objective,
            "max_makespan_regret": self.max_makespan_regret,
            "min_makespan_regret": self.min_makespan_regret,
            "statewise_regret_slack": self.statewise_regret_slack,
            "statewise": self.statewise,
            "guarded_resource_kinds": list(self.guarded_resource_kinds),
            "statewise_resource_kinds": list(self.statewise_resource_kinds),
            "statewise_workload_keys": list(self.statewise_workload_keys),
            "statewise_excluded_workload_keys": list(self.statewise_excluded_workload_keys),
            "backlog_aware_guard": self.backlog_aware_guard,
            "backlog_reference_tasks": self.backlog_reference_tasks,
            "statewise_service_dominance_guard": self.statewise_service_dominance_guard,
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
    per_resource_waiting = _assign_task_ids_to_resources(
        waiting,
        tasks,
        resource_count=len(resource_times),
        mode=_resource_assignment_mode_for(
            cache=cache,
            spec=spec,
            policy=policy,
            base_target_profile=profile.profile,
        ),
        seed=_resource_assignment_seed_for(
            cache=cache,
            spec=spec,
            policy=policy,
            base_target_profile=profile.profile,
        ),
    )
    for local_waiting in per_resource_waiting:
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
    current_service_profile = max(1, int(target_profile))

    def fill() -> None:
        nonlocal current_service_profile
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
                base_target_profile=target_profile,
            )
            if _statewise_holds_base_profile_for(
                cache=cache,
                spec=spec,
                policy=policy,
                base_target_profile=target_profile,
                waiting_count=len(waiting),
                total_remaining=total_remaining,
            ):
                target = max(int(target), int(target_profile))
        desired_target = max(1, int(target))
        admission_target = max(len(active), min(desired_target, total_remaining))
        current_service_profile = _covering_service_profile(
            cache,
            workload_key=spec.workload_key,
            active_count=admission_target,
            preferred_profile=max(desired_target, admission_target),
        )
        profile_trace.append(current_service_profile)
        order_mode = _waiting_order_mode_for(
            cache=cache,
            spec=spec,
            policy=policy,
            base_target_profile=target_profile,
        )
        if order_mode == "shortest_remaining_first":
            waiting.sort(key=lambda tid: (float(tasks[tid]["remaining"]), int(tid)))
        elif order_mode == "critical_shortest_remaining_first":
            waiting.sort(key=lambda tid: (float(tasks[tid]["remaining"]), int(tid)))
            if not active and len(waiting) > 1:
                critical = max(waiting, key=lambda tid: (float(tasks[tid]["remaining"]), -int(tid)))
                waiting.remove(critical)
                waiting.insert(0, critical)
        elif order_mode == "critical_batch_shortest_remaining_first":
            waiting.sort(key=lambda tid: (float(tasks[tid]["remaining"]), int(tid)))
            if not active and len(waiting) > admission_target:
                critical = sorted(
                    waiting,
                    key=lambda tid: (float(tasks[tid]["remaining"]), -int(tid)),
                    reverse=True,
                )[:admission_target]
                critical_set = set(critical)
                rest = [tid for tid in waiting if tid not in critical_set]
                waiting[:] = critical + rest
        while waiting and len(active) < admission_target:
            active.append(waiting.pop(0))

    fill()
    while active:
        m = len(active)
        rates = _sample_rates(
            cache,
            spec.workload_key,
            m,
            rng,
            service_profile_count=current_service_profile,
        )
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
    base_target_profile: int | None = None,
) -> int:
    local_spec = WorkloadSpec(
        workload_key=spec.workload_key,
        resource_kind=spec.resource_kind,
        task_count=max(1, int(remaining_count)),
        total_units=spec.total_units,
        resource_count=1,
        variation_cv=spec.variation_cv,
    )
    effective_policy = _trajectory_policy_for(
        cache=cache,
        spec=spec,
        policy=policy,
        base_target_profile=base_target_profile,
    )
    custom = effective_policy.statewise_target_profile_for(
        cache,
        spec,
        remaining_count=remaining_count,
        active_count=active_count,
        base_target_profile=base_target_profile,
    )
    if custom is not None:
        return int(custom)
    if not effective_policy.uses_statewise_for(spec):
        return int(base_target_profile or effective_policy.select_profile(cache, spec).profile)
    if effective_policy.calibrated and effective_policy.uses_guarded_objective(local_spec):
        record = cache.best_profile_for_guarded_mean_flow(
            local_spec.workload_key,
            task_count=local_spec.task_count,
            total_units=local_spec.total_units,
            resource_count=local_spec.resource_count,
            max_makespan_regret=_effective_guard_regret(effective_policy, remaining_count),
        )
        return _statewise_service_guarded_profile(
            cache=cache,
            spec=spec,
            policy=effective_policy,
            remaining_count=remaining_count,
            active_count=active_count,
            candidate_profile=record.profile,
            base_target_profile=base_target_profile,
        )
    record = effective_policy.select_profile(cache, local_spec)
    return _statewise_service_guarded_profile(
        cache=cache,
        spec=spec,
        policy=effective_policy,
        remaining_count=remaining_count,
        active_count=active_count,
        candidate_profile=record.profile,
        base_target_profile=base_target_profile,
    )


def _statewise_holds_base_profile(
    policy: ReplayPolicy,
    *,
    waiting_count: int,
    total_remaining: int,
) -> bool:
    if not policy.statewise_service_dominance_guard or int(waiting_count) <= 0:
        return False
    threshold = int(getattr(policy, "statewise_drain_remaining_threshold", 0) or 0)
    if threshold > 0 and int(total_remaining) <= threshold:
        return False
    return True


def _statewise_holds_base_profile_for(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    base_target_profile: int | None,
    waiting_count: int,
    total_remaining: int,
) -> bool:
    effective_policy = _trajectory_policy_for(
        cache=cache,
        spec=spec,
        policy=policy,
        base_target_profile=base_target_profile,
    )
    custom = effective_policy.statewise_holds_base_profile_for(
        waiting_count=waiting_count,
        total_remaining=total_remaining,
    )
    if custom is not None:
        return bool(custom)
    return _statewise_holds_base_profile(
        effective_policy,
        waiting_count=waiting_count,
        total_remaining=total_remaining,
    )


def _trajectory_policy_for(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    base_target_profile: int | None,
) -> ReplayPolicy:
    delegated = policy.trajectory_policy_for(
        cache,
        spec,
        base_target_profile=base_target_profile,
    )
    if delegated is None or delegated is policy:
        return policy
    return delegated


def _uses_shortest_remaining_first_for(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    base_target_profile: int | None,
) -> bool:
    effective_policy = _trajectory_policy_for(
        cache=cache,
        spec=spec,
        policy=policy,
        base_target_profile=base_target_profile,
    )
    return bool(effective_policy.uses_shortest_remaining_first_for(spec))


def _waiting_order_mode_for(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    base_target_profile: int | None,
) -> str:
    effective_policy = _trajectory_policy_for(
        cache=cache,
        spec=spec,
        policy=policy,
        base_target_profile=base_target_profile,
    )
    mode = str(effective_policy.waiting_order_mode_for(spec) or "fifo")
    if mode not in {
        "fifo",
        "shortest_remaining_first",
        "critical_shortest_remaining_first",
        "critical_batch_shortest_remaining_first",
    }:
        return "fifo"
    return mode


def _resource_assignment_mode_for(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    base_target_profile: int | None,
) -> str:
    effective_policy = _trajectory_policy_for(
        cache=cache,
        spec=spec,
        policy=policy,
        base_target_profile=base_target_profile,
    )
    mode = str(effective_policy.resource_assignment_mode_for(spec) or "round_robin_count")
    if mode not in {"round_robin_count", "lpt_static", "shuffle_static"}:
        return "round_robin_count"
    return mode


def _resource_assignment_seed_for(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    base_target_profile: int | None,
) -> int:
    effective_policy = _trajectory_policy_for(
        cache=cache,
        spec=spec,
        policy=policy,
        base_target_profile=base_target_profile,
    )
    return int(effective_policy.resource_assignment_seed_for(spec))


def _assign_task_ids_to_resources(
    task_ids: list[int],
    tasks: list[dict[str, float]],
    *,
    resource_count: int,
    mode: str,
    seed: int = 0,
) -> list[list[int]]:
    resources: list[list[int]] = [[] for _ in range(max(1, int(resource_count)))]
    if mode == "shuffle_static" and len(resources) > 1:
        shuffled = list(task_ids)
        random.Random(int(seed)).shuffle(shuffled)
        for idx, task_id in enumerate(shuffled):
            resources[idx % len(resources)].append(task_id)
        return resources
    if mode != "lpt_static" or len(resources) <= 1:
        for idx, task_id in enumerate(task_ids):
            resources[idx % len(resources)].append(task_id)
        return resources
    loads = [(0.0, idx) for idx in range(len(resources))]
    for task_id in sorted(task_ids, key=lambda tid: (float(tasks[tid]["remaining"]), -int(tid)), reverse=True):
        load, idx = min(loads, key=lambda item: (item[0], item[1]))
        resources[idx].append(task_id)
        loads[idx] = (load + float(tasks[task_id]["remaining"]), idx)
    return resources


def _statewise_service_guarded_profile(
    *,
    cache: ServiceRateCache,
    spec: WorkloadSpec,
    policy: ReplayPolicy,
    remaining_count: int,
    active_count: int,
    candidate_profile: int,
    base_target_profile: int | None,
) -> int:
    active = max(1, int(active_count))
    candidate = max(active, int(candidate_profile))
    if not policy.statewise_service_dominance_guard or base_target_profile is None:
        return candidate
    threshold = int(getattr(policy, "statewise_drain_remaining_threshold", 0) or 0)
    if threshold > 0 and int(remaining_count) <= threshold:
        return candidate
    baseline = max(active, int(base_target_profile))
    candidate = min(candidate, baseline)
    candidate_rate = _effective_profile_rate_for_active(
        cache,
        workload_key=spec.workload_key,
        active_count=active,
        preferred_profile=candidate,
    )
    baseline_rate = _effective_profile_rate_for_active(
        cache,
        workload_key=spec.workload_key,
        active_count=active,
        preferred_profile=baseline,
    )
    if candidate_rate + 1e-12 >= baseline_rate:
        return candidate
    return baseline


def _effective_profile_rate_for_active(
    cache: ServiceRateCache,
    *,
    workload_key: str,
    active_count: int,
    preferred_profile: int,
) -> float:
    active = max(1, int(active_count))
    service_profile = _covering_service_profile(
        cache,
        workload_key=workload_key,
        active_count=active,
        preferred_profile=max(active, int(preferred_profile)),
    )
    record = cache.get(workload_key, service_profile)
    if record is None or record.capacity_boundary:
        return 0.0
    return float(record.aggregate_rate) * float(active) / float(max(1, int(record.profile)))


def _effective_guard_regret(policy: ReplayPolicy, remaining_count: int) -> float:
    base = max(0.0, float(policy.max_makespan_regret))
    slack = max(0.0, float(policy.statewise_regret_slack)) / float(max(1, remaining_count))
    if not policy.backlog_aware_guard:
        return base + slack
    floor = max(0.0, float(policy.min_makespan_regret))
    ceiling = max(floor, base)
    reference = max(1.0, float(policy.backlog_reference_tasks or 1))
    pressure = min(1.0, max(0.0, float(remaining_count)) / reference)
    dynamic = floor + (ceiling - floor) * (1.0 - pressure)
    return dynamic + slack


def _profile_trace_counts(trace: tuple[int, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in trace:
        key = str(int(profile))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _covering_service_profile(
    cache: ServiceRateCache,
    *,
    workload_key: str,
    active_count: int,
    preferred_profile: int,
) -> int:
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
            "multi-task co-location rates must be measured in the real environment"
        )
    rates = [float(x) for x in profile.per_task_rates if float(x) > 0]
    if not rates:
        rates = [profile.mean_rate] * max(1, profile.profile)
    sampled = [rates[rng.randrange(len(rates))] for _ in range(active_count)]
    active_fraction = float(active_count) / float(max(1, profile.profile))
    target_aggregate = float(profile.aggregate_rate) * active_fraction
    scale = target_aggregate / max(1e-12, sum(sampled))
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
