"""Deterministic PSPLIB MMRCPSP benchmark sidecar.

This module runs a non-delay parallel schedule-generation scheme over the finite feasible
activity/mode action family parsed by :mod:`mmrcpsp_instances`.  The Scheduleurm
policy uses a bounded lower-service-minus-penalty score; the comparison policies
are deterministic priority-rule heuristics.  This is an OR generalization
experiment, not an exact MMRCPSP solver or a claim of state-of-the-art dominance.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.mmrcpsp_instances import (
    MMRCPSPInstance,
    ModeAction,
    feasible_mode_actions,
    parse_psplib_mm,
)


SCHEMA_VERSION = "scheduleurm.mmrcpsp.benchmark.v1"
OURS_POLICY = "scheduleurm_robust_maxweight"
POLICY_ORDER = (
    OURS_POLICY,
    "shortest_processing_time",
    "minimum_slack",
    "greatest_rank_positional_weight",
    "most_total_successors",
)
POLICY_DEFINITIONS = {
    OURS_POLICY: (
        "Finite eligible activity-mode actions scored by criticality-weighted "
        "lower service minus renewable scarcity, nonrenewable opportunity, and "
        "non-nominal-mode penalties."
    ),
    "shortest_processing_time": (
        "Non-delay parallel schedule-generation scheme (SGS) using shortest processing time "
        "(SPT), then lower normalized resource demand."
    ),
    "minimum_slack": (
        "Parallel SGS using minimum slack (MINSLK), equivalently greatest remaining "
        "minimum-duration critical path at a common decision epoch."
    ),
    "greatest_rank_positional_weight": (
        "Parallel SGS using greatest rank positional weight (GRPW): own minimum "
        "duration plus minimum durations of all transitive successors."
    ),
    "most_total_successors": (
        "Parallel SGS using most total successors (MTS), then critical-path and "
        "shorter-duration tie breaks."
    ),
}


@dataclass(frozen=True)
class ScheduleEntry:
    job_id: int
    mode_id: int
    start: int
    finish: int
    duration: int
    renewable_demands: tuple[int, ...]
    nonrenewable_demands: tuple[int, ...]
    action_score: float
    queue_weight_proxy: float
    lower_service: float
    penalty_units: float
    action_type: str

    @property
    def action_id(self) -> str:
        return f"job_{self.job_id}:mode_{self.mode_id}@{self.start}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "job_id": self.job_id,
            "mode_id": self.mode_id,
            "start": self.start,
            "finish": self.finish,
            "duration": self.duration,
            "renewable_demands": list(self.renewable_demands),
            "nonrenewable_demands": list(self.nonrenewable_demands),
            "queue_weight_proxy": self.queue_weight_proxy,
            "lower_service": self.lower_service,
            "penalty_units": self.penalty_units,
            "action_score": self.action_score,
        }


@dataclass(frozen=True)
class ScheduleResult:
    policy: str
    feasible: bool
    failure_reason: str | None
    entries: tuple[ScheduleEntry, ...]
    metrics: Mapping[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "policy_definition": POLICY_DEFINITIONS[self.policy],
            "selection_semantics": (
                "finite_project_lower_service_minus_bounded_penalty"
                if self.policy == OURS_POLICY
                else "deterministic_priority_rule"
            ),
            "feasible": self.feasible,
            "failure_reason": self.failure_reason,
            "metrics": dict(self.metrics),
            "schedule": [entry.snapshot() for entry in self.entries],
        }


@dataclass(frozen=True)
class _InstanceAnalysis:
    minimum_duration: Mapping[int, int]
    critical_path: Mapping[int, int]
    descendants: Mapping[int, frozenset[int]]
    positional_weight: Mapping[int, int]


def schedule_instance(
    instance: MMRCPSPInstance,
    policy: str,
) -> ScheduleResult:
    """Build one deterministic nonpreemptive schedule using ``policy``."""

    if policy not in POLICY_DEFINITIONS:
        raise ValueError(f"unknown MMRCPSP policy: {policy}")
    analysis = _analyze_instance(instance)
    activity_map = instance.activity_map
    unscheduled = set(instance.job_ids)
    completed: set[int] = set()
    running: list[ScheduleEntry] = []
    entries: list[ScheduleEntry] = []
    nonrenewable_consumed = [0] * len(instance.nonrenewable_capacities)
    now = 0
    decision_count = 0
    decision_limit = max(100, 20 * len(instance.job_ids) ** 2)
    failure_reason: str | None = None

    while unscheduled or running:
        finished = [entry for entry in running if entry.finish <= now]
        if finished:
            completed.update(entry.job_id for entry in finished)
            running = [entry for entry in running if entry.finish > now]

        eligible = tuple(
            sorted(
                job_id
                for job_id in unscheduled
                if set(activity_map[job_id].predecessors).issubset(completed)
            )
        )
        renewable_used = tuple(
            sum(entry.renewable_demands[index] for entry in running)
            for index in range(len(instance.renewable_capacities))
        )
        renewable_available = tuple(
            capacity - used
            for capacity, used in zip(
                instance.renewable_capacities,
                renewable_used,
                strict=True,
            )
        )
        actions = feasible_mode_actions(
            instance,
            eligible_job_ids=eligible,
            unscheduled_job_ids=unscheduled,
            renewable_available=renewable_available,
            nonrenewable_consumed=nonrenewable_consumed,
        )
        if actions:
            selected, score, queue_weight, lower_service, penalty = _select_action(
                instance,
                actions,
                policy=policy,
                now=now,
                renewable_available=renewable_available,
                nonrenewable_consumed=tuple(nonrenewable_consumed),
                analysis=analysis,
            )
            nominal_mode = min(
                mode.mode_id for mode in activity_map[selected.job_id].modes
            )
            entry = ScheduleEntry(
                job_id=selected.job_id,
                mode_id=selected.mode_id,
                start=now,
                finish=now + selected.duration,
                duration=selected.duration,
                renewable_demands=selected.renewable_demands,
                nonrenewable_demands=selected.nonrenewable_demands,
                action_score=_round(score),
                queue_weight_proxy=_round(queue_weight),
                lower_service=_round(lower_service),
                penalty_units=_round(penalty),
                action_type=(
                    "launch_nominal_mode"
                    if selected.mode_id == nominal_mode
                    else "launch_reconfigured_mode"
                ),
            )
            entries.append(entry)
            unscheduled.remove(selected.job_id)
            nonrenewable_consumed = [
                consumed + demand
                for consumed, demand in zip(
                    nonrenewable_consumed,
                    selected.nonrenewable_demands,
                    strict=True,
                )
            ]
            if selected.duration == 0:
                completed.add(selected.job_id)
            else:
                running.append(entry)
            decision_count += 1
            if decision_count > decision_limit:
                failure_reason = "decision_limit_exceeded"
                break
            continue

        if running:
            next_finish = min(entry.finish for entry in running)
            if next_finish <= now:
                failure_reason = "nonadvancing_event_time"
                break
            now = next_finish
            continue

        if unscheduled:
            failure_reason = _deadlock_reason(
                instance,
                unscheduled=unscheduled,
                completed=completed,
                nonrenewable_consumed=nonrenewable_consumed,
            )
            break

    metrics = _evaluate_schedule(instance, entries)
    feasible = failure_reason is None and bool(metrics["schedule_feasible"])
    if failure_reason is None and not feasible:
        failure_reason = "post_schedule_feasibility_audit_failed"
    return ScheduleResult(
        policy=policy,
        feasible=feasible,
        failure_reason=failure_reason,
        entries=tuple(entries),
        metrics=metrics,
    )


def build_mmrcpsp_benchmark(
    instance_paths: Sequence[str | Path],
    *,
    policies: Sequence[str] = POLICY_ORDER,
) -> dict[str, Any]:
    """Parse instances, run all policies, and return a deterministic report."""

    if not instance_paths:
        raise ValueError("at least one PSPLIB .mm instance is required")
    selected_policies = tuple(str(policy) for policy in policies)
    if not selected_policies or len(set(selected_policies)) != len(selected_policies):
        raise ValueError("policies must be a nonempty unique sequence")
    unknown = sorted(set(selected_policies) - set(POLICY_DEFINITIONS))
    if unknown:
        raise ValueError(f"unknown MMRCPSP policies: {unknown}")
    if OURS_POLICY not in selected_policies:
        raise ValueError(f"benchmark must include {OURS_POLICY}")

    instances = [
        parse_psplib_mm(path)
        for path in sorted(
            (Path(path) for path in instance_paths),
            key=lambda path: (path.name, str(path)),
        )
    ]
    names = [instance.name for instance in instances]
    if len(set(names)) != len(names):
        raise ValueError(f"instance stems must be unique: {names}")

    instance_reports: dict[str, Any] = {}
    results_by_policy: dict[str, list[ScheduleResult]] = {
        policy: [] for policy in selected_policies
    }
    for instance in instances:
        policy_results: dict[str, Any] = {}
        for policy in selected_policies:
            result = schedule_instance(instance, policy)
            results_by_policy[policy].append(result)
            policy_results[policy] = result.snapshot()
        instance_reports[instance.name] = {
            "instance": instance.snapshot(),
            "policy_results": policy_results,
            "ours_vs_baselines": _instance_comparison(policy_results),
        }

    aggregates = {
        policy: _aggregate_results(results)
        for policy, results in results_by_policy.items()
    }
    every_result = [
        result
        for results in results_by_policy.values()
        for result in results
    ]
    all_feasible = all(result.feasible for result in every_result)
    zero_resource_violation = all(
        float(result.metrics["resource_violation_units"]) == 0.0
        for result in every_result
    )
    zero_precedence_violation = all(
        int(result.metrics["precedence_violation_count"]) == 0
        for result in every_result
    )
    gate_pass = all_feasible and zero_resource_violation and zero_precedence_violation
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "psplib_core_mmrcpsp_priority_rule_comparison",
        "deterministic": True,
        "random_seed": None,
        "instance_count": len(instances),
        "policy_order": list(selected_policies),
        "policy_definitions": {
            policy: POLICY_DEFINITIONS[policy] for policy in selected_policies
        },
        "instances": instance_reports,
        "aggregate": aggregates,
        "gate": {
            "status": (
                "MMRCPSP_CORE_FEASIBILITY_PASS"
                if gate_pass
                else "MMRCPSP_CORE_FEASIBILITY_FAIL"
            ),
            "pass": gate_pass,
            "all_policy_schedules_feasible": all_feasible,
            "zero_resource_violation": zero_resource_violation,
            "zero_precedence_violation": zero_precedence_violation,
            "ours_dominance_not_required_for_pass": True,
        },
        "metric_contract": {
            "makespan": "Maximum finish time over all activities.",
            "mean_flow_time": (
                "Arithmetic mean completion time of non-dummy activities; "
                "standard PSPLIB release times are zero."
            ),
            "resource_violation_units": (
                "Renewable excess-capacity area plus terminal nonrenewable "
                "budget excess; zero is required."
            ),
            "mode_reconfiguration_count": (
                "Number of non-dummy activities assigned a mode other than the "
                "lowest PSPLIB mode id; this is a declared proxy, not measured "
                "physical setup time."
            ),
            "renewable_configuration_l1": (
                "Capacity-normalized L1 variation of renewable load between "
                "successive event intervals, including entry from and return to zero."
            ),
        },
        "score_contract": {
            "ours_formula": "queue_weight_proxy * lower_service - penalty_units",
            "queue_weight_proxy": (
                "One plus remaining minimum-duration critical path plus half the "
                "transitive-successor count, with a bounded event-time age term. "
                "It is a finite-project priority proxy, not a stochastic queue length."
            ),
            "lower_service": "Reciprocal selected-mode duration for non-dummy activities.",
            "penalty_units": (
                "Bounded normalized renewable scarcity, nonrenewable opportunity, "
                "and non-nominal-mode terms on the finite parsed instance."
            ),
        },
        "references": {
            "psplib_multi_mode_data": "https://www.om-db.wi.tum.de/psplib/getdata.php?mode=mm",
            "psplib_library_paper": "https://doi.org/10.1016/S0377-2217(96)00170-1",
            "priority_rule_context": "https://doi.org/10.1016/0272-6963(95)00032-1",
        },
        "claim_boundary": {
            "parser_scope": (
                "Core single-project PSPLIB .mm precedence, duration, renewable, "
                "and nonrenewable fields; no doubly constrained resources, time "
                "lags, calendars, stochastic durations, or multi-project extension."
            ),
            "algorithm_scope": (
                "Deterministic non-delay parallel-SGS heuristics over finite feasible mode "
                "actions; neither an exact solver nor a reproduction of a named "
                "external MMRCPSP software stack."
            ),
            "evidence_scope": (
                "Only the explicitly supplied, hash-recorded .mm files. Fixture "
                "success is parser/simulator validation, not a PSPLIB-wide campaign."
            ),
            "theorem_scope": (
                "The Scheduleurm score instantiates the finite-action lower-service "
                "minus bounded-penalty shape. It does not by itself validate the "
                "stochastic queueing stability theorem or server service-rate cache."
            ),
            "reconfiguration_scope": (
                "Reported mode/load reconfiguration metrics are deterministic "
                "schedule proxies, not measured industrial changeover costs."
            ),
            "strong_claim_ready": False,
            "strong_claim_blockers": [
                "No full official PSPLIB instance-set campaign or optimality-gap table.",
                "No exact/MMRCPSP state-of-the-art solver baseline.",
                "No stochastic arrivals, online rescheduling, or measured mode-switch cost.",
            ],
        },
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    """Render a deterministic human-auditable Markdown artifact."""

    gate = report["gate"]
    lines = [
        "# PSPLIB MMRCPSP Benchmark",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Status: `{gate['status']}`",
        f"- Pass: `{str(bool(gate['pass'])).lower()}`",
        f"- Instances: `{report['instance_count']}`",
        "- Deterministic: `true`",
        "",
        "## Results",
        "",
        (
            "| Instance | Policy | Feasible | Makespan | Mean flow | Resource "
            "violation | Mode reconfigurations | Renewable config L1 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for instance_name, instance_row in report["instances"].items():
        for policy in report["policy_order"]:
            result = instance_row["policy_results"][policy]
            metrics = result["metrics"]
            lines.append(
                f"| `{instance_name}` | `{policy}` | "
                f"{str(bool(result['feasible'])).lower()} | "
                f"{_format_number(metrics['makespan'])} | "
                f"{_format_number(metrics['mean_flow_time'])} | "
                f"{_format_number(metrics['resource_violation_units'])} | "
                f"{int(metrics['mode_reconfiguration_count'])} | "
                f"{_format_number(metrics['renewable_configuration_l1'])} |"
            )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Policy | Feasible instances | Mean makespan | Mean flow | Mean resource violation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy in report["policy_order"]:
        row = report["aggregate"][policy]
        lines.append(
            f"| `{policy}` | {row['feasible_instance_count']}/{row['instance_count']} | "
            f"{_format_number(row['mean_makespan'])} | "
            f"{_format_number(row['mean_flow_time'])} | "
            f"{_format_number(row['mean_resource_violation_units'])} |"
        )

    lines.extend(["", "## Policy Contract", ""])
    for policy in report["policy_order"]:
        lines.append(f"- `{policy}`: {report['policy_definitions'][policy]}")
    lines.extend(["", "## Claim Boundary", ""])
    boundary = report["claim_boundary"]
    for key in (
        "parser_scope",
        "algorithm_scope",
        "evidence_scope",
        "theorem_scope",
        "reconfiguration_scope",
    ):
        lines.append(f"- **{key.replace('_', ' ').title()}**: {boundary[key]}")
    lines.append(
        f"- **Strong claim ready**: `{str(bool(boundary['strong_claim_ready'])).lower()}`"
    )
    for blocker in boundary["strong_claim_blockers"]:
        lines.append(f"- **Blocker**: {blocker}")
    lines.append("")
    return "\n".join(lines)


def write_artifacts(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write byte-deterministic JSON and Markdown artifacts atomically."""

    _atomic_write(
        Path(json_path),
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(Path(markdown_path), markdown_report(report))


def _select_action(
    instance: MMRCPSPInstance,
    actions: Sequence[ModeAction],
    *,
    policy: str,
    now: int,
    renewable_available: Sequence[int],
    nonrenewable_consumed: Sequence[int],
    analysis: _InstanceAnalysis,
) -> tuple[ModeAction, float, float, float, float]:
    scored = [
        (
            action,
            *_action_score(
                instance,
                action,
                now=now,
                renewable_available=renewable_available,
                nonrenewable_consumed=nonrenewable_consumed,
                analysis=analysis,
            ),
        )
        for action in actions
    ]
    if policy == OURS_POLICY:
        action, score, queue_weight, lower_service, penalty = max(
            scored,
            key=lambda row: (
                row[1],
                -row[0].duration,
                -row[0].job_id,
                -row[0].mode_id,
            ),
        )
        return action, score, queue_weight, lower_service, penalty

    def baseline_key(
        row: tuple[ModeAction, float, float, float, float]
    ) -> tuple[float, ...]:
        action = row[0]
        pressure = _normalized_resource_pressure(instance, action)
        if action.duration == 0:
            return (1e12, -action.job_id, -action.mode_id)
        if policy == "shortest_processing_time":
            return (-action.duration, -pressure, -action.job_id, -action.mode_id)
        if policy == "minimum_slack":
            slack = instance.horizon - now - analysis.critical_path[action.job_id]
            return (-slack, -action.duration, -pressure, -action.job_id, -action.mode_id)
        if policy == "greatest_rank_positional_weight":
            return (
                analysis.positional_weight[action.job_id],
                -action.duration,
                -pressure,
                -action.job_id,
                -action.mode_id,
            )
        if policy == "most_total_successors":
            return (
                len(analysis.descendants[action.job_id]),
                analysis.critical_path[action.job_id],
                -action.duration,
                -pressure,
                -action.job_id,
                -action.mode_id,
            )
        raise AssertionError(policy)

    action, _, queue_weight, lower_service, penalty = max(scored, key=baseline_key)
    priority_score = baseline_key(
        (action, 0.0, queue_weight, lower_service, penalty)
    )[0]
    return action, float(priority_score), queue_weight, lower_service, penalty


def _action_score(
    instance: MMRCPSPInstance,
    action: ModeAction,
    *,
    now: int,
    renewable_available: Sequence[int],
    nonrenewable_consumed: Sequence[int],
    analysis: _InstanceAnalysis,
) -> tuple[float, float, float, float]:
    if action.duration == 0:
        queue_weight = 1e9 - action.job_id
        return queue_weight, queue_weight, 1.0, 0.0
    criticality = 1.0 + float(analysis.critical_path[action.job_id])
    criticality += 0.5 * len(analysis.descendants[action.job_id])
    lower_service = 1.0 / float(action.duration)

    renewable_penalty = 0.0
    for demand, capacity, available in zip(
        action.renewable_demands,
        instance.renewable_capacities,
        renewable_available,
        strict=True,
    ):
        if capacity > 0:
            renewable_penalty += demand / capacity
        if available > 0:
            renewable_penalty += 0.5 * demand / available

    nonrenewable_penalty = 0.0
    for demand, consumed, capacity in zip(
        action.nonrenewable_demands,
        nonrenewable_consumed,
        instance.nonrenewable_capacities,
        strict=True,
    ):
        remaining = max(1, capacity - consumed)
        nonrenewable_penalty += demand / remaining

    nominal_mode = min(
        mode.mode_id for mode in instance.activity_map[action.job_id].modes
    )
    reconfiguration_penalty = 1.0 if action.mode_id != nominal_mode else 0.0
    age_bonus = min(max(0, now), max(1, instance.horizon)) / max(
        1, instance.horizon
    )
    penalty = (
        0.12 * renewable_penalty
        + 0.25 * nonrenewable_penalty
        + 0.05 * reconfiguration_penalty
    )
    queue_weight = criticality + age_bonus
    score = queue_weight * lower_service - penalty
    return score, queue_weight, lower_service, penalty


def _analyze_instance(instance: MMRCPSPInstance) -> _InstanceAnalysis:
    activity_map = instance.activity_map
    order = _topological_order(instance)
    minimum_duration = {
        job_id: min(mode.duration for mode in activity_map[job_id].modes)
        for job_id in instance.job_ids
    }
    descendants: dict[int, frozenset[int]] = {}
    critical_path: dict[int, int] = {}
    for job_id in reversed(order):
        successor_descendants: set[int] = set()
        successor_paths: list[int] = []
        for successor in activity_map[job_id].successors:
            successor_descendants.add(successor)
            successor_descendants.update(descendants[successor])
            successor_paths.append(critical_path[successor])
        descendants[job_id] = frozenset(successor_descendants)
        critical_path[job_id] = minimum_duration[job_id] + (
            max(successor_paths) if successor_paths else 0
        )
    positional_weight = {
        job_id: minimum_duration[job_id]
        + sum(minimum_duration[descendant] for descendant in descendants[job_id])
        for job_id in instance.job_ids
    }
    return _InstanceAnalysis(
        minimum_duration=minimum_duration,
        critical_path=critical_path,
        descendants=descendants,
        positional_weight=positional_weight,
    )


def _topological_order(instance: MMRCPSPInstance) -> tuple[int, ...]:
    activity_map = instance.activity_map
    indegree = {
        job_id: len(activity_map[job_id].predecessors)
        for job_id in instance.job_ids
    }
    ready = sorted(job_id for job_id, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        job_id = ready.pop(0)
        order.append(job_id)
        for successor in activity_map[job_id].successors:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    return tuple(order)


def _evaluate_schedule(
    instance: MMRCPSPInstance,
    entries: Sequence[ScheduleEntry],
) -> dict[str, Any]:
    activity_map = instance.activity_map
    entry_map = {entry.job_id: entry for entry in entries}
    duplicate_count = len(entries) - len(entry_map)
    missing_jobs = sorted(set(instance.job_ids) - set(entry_map))
    invalid_mode_count = sum(
        entry.mode_id
        not in {mode.mode_id for mode in activity_map[entry.job_id].modes}
        for entry in entries
        if entry.job_id in activity_map
    )
    precedence_violation_count = 0
    for entry in entries:
        for predecessor in activity_map[entry.job_id].predecessors:
            predecessor_entry = entry_map.get(predecessor)
            if predecessor_entry is None or entry.start < predecessor_entry.finish:
                precedence_violation_count += 1

    event_times = sorted(
        {0, *(entry.start for entry in entries), *(entry.finish for entry in entries)}
    )
    renewable_violation_area = 0.0
    renewable_peak_excess = [0] * len(instance.renewable_capacities)
    interval_states: list[tuple[int, ...]] = []
    renewable_busy_area = [0] * len(instance.renewable_capacities)
    for start, finish in zip(event_times, event_times[1:]):
        if finish <= start:
            continue
        usage = tuple(
            sum(
                entry.renewable_demands[index]
                for entry in entries
                if entry.start <= start < entry.finish
            )
            for index in range(len(instance.renewable_capacities))
        )
        interval_states.append(usage)
        width = finish - start
        for index, (used, capacity) in enumerate(
            zip(usage, instance.renewable_capacities, strict=True)
        ):
            excess = max(0, used - capacity)
            renewable_peak_excess[index] = max(
                renewable_peak_excess[index], excess
            )
            renewable_violation_area += excess * width
            renewable_busy_area[index] += used * width

    nonrenewable_used = tuple(
        sum(entry.nonrenewable_demands[index] for entry in entries)
        for index in range(len(instance.nonrenewable_capacities))
    )
    nonrenewable_excess = tuple(
        max(0, used - capacity)
        for used, capacity in zip(
            nonrenewable_used,
            instance.nonrenewable_capacities,
            strict=True,
        )
    )
    nonrenewable_violation = sum(nonrenewable_excess)
    resource_violation = renewable_violation_area + nonrenewable_violation
    makespan = max((entry.finish for entry in entries), default=0)
    real_finishes = [
        entry_map[job_id].finish
        for job_id in instance.real_job_ids
        if job_id in entry_map
    ]
    mean_flow = fmean(real_finishes) if real_finishes else 0.0
    mode_reconfiguration_count = sum(
        entry.mode_id
        != min(mode.mode_id for mode in activity_map[entry.job_id].modes)
        for entry in entries
        if entry.job_id in instance.real_job_ids
    )
    configuration_l1 = _renewable_configuration_l1(
        interval_states,
        instance.renewable_capacities,
    )
    utilization = [
        (
            busy / (capacity * makespan)
            if capacity > 0 and makespan > 0
            else 0.0
        )
        for busy, capacity in zip(
            renewable_busy_area,
            instance.renewable_capacities,
            strict=True,
        )
    ]
    schedule_feasible = (
        not missing_jobs
        and duplicate_count == 0
        and invalid_mode_count == 0
        and precedence_violation_count == 0
        and resource_violation == 0
    )
    return {
        "schedule_feasible": schedule_feasible,
        "makespan": makespan,
        "mean_flow_time": _round(mean_flow),
        "resource_violation_units": _round(resource_violation),
        "renewable_violation_area": _round(renewable_violation_area),
        "renewable_peak_excess": renewable_peak_excess,
        "nonrenewable_used": list(nonrenewable_used),
        "nonrenewable_excess": list(nonrenewable_excess),
        "precedence_violation_count": precedence_violation_count,
        "missing_job_ids": missing_jobs,
        "duplicate_job_count": duplicate_count,
        "invalid_mode_count": invalid_mode_count,
        "mode_reconfiguration_count": mode_reconfiguration_count,
        "renewable_configuration_l1": _round(configuration_l1),
        "renewable_utilization": [_round(value) for value in utilization],
        "horizon": instance.horizon,
        "horizon_exceeded": makespan > instance.horizon,
        "real_activity_count": len(instance.real_job_ids),
    }


def _renewable_configuration_l1(
    interval_states: Sequence[Sequence[int]],
    capacities: Sequence[int],
) -> float:
    zero = tuple(0 for _ in capacities)
    states = [zero, *(tuple(state) for state in interval_states), zero]
    total = 0.0
    for previous, current in zip(states, states[1:]):
        for old, new, capacity in zip(
            previous, current, capacities, strict=True
        ):
            delta = abs(new - old)
            total += delta / capacity if capacity > 0 else float(delta)
    return total


def _normalized_resource_pressure(
    instance: MMRCPSPInstance, action: ModeAction
) -> float:
    renewable = sum(
        demand / capacity if capacity > 0 else float(demand)
        for demand, capacity in zip(
            action.renewable_demands,
            instance.renewable_capacities,
            strict=True,
        )
    )
    nonrenewable = sum(
        demand / capacity if capacity > 0 else float(demand)
        for demand, capacity in zip(
            action.nonrenewable_demands,
            instance.nonrenewable_capacities,
            strict=True,
        )
    )
    return renewable + nonrenewable


def _deadlock_reason(
    instance: MMRCPSPInstance,
    *,
    unscheduled: set[int],
    completed: set[int],
    nonrenewable_consumed: Sequence[int],
) -> str:
    activity_map = instance.activity_map
    eligible = [
        job_id
        for job_id in sorted(unscheduled)
        if set(activity_map[job_id].predecessors).issubset(completed)
    ]
    if not eligible:
        return f"precedence_deadlock:unscheduled={sorted(unscheduled)}"
    return (
        "resource_deadlock:"
        f"eligible={eligible}:nonrenewable_consumed={list(nonrenewable_consumed)}"
    )


def _instance_comparison(
    policy_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ours = policy_results[OURS_POLICY]
    ours_metrics = ours["metrics"]
    baselines: dict[str, Any] = {}
    for policy, result in policy_results.items():
        if policy == OURS_POLICY:
            continue
        metrics = result["metrics"]
        makespan_ratio = _cost_ratio(metrics["makespan"], ours_metrics["makespan"])
        flow_ratio = _cost_ratio(
            metrics["mean_flow_time"], ours_metrics["mean_flow_time"]
        )
        weakly_better = (
            ours_metrics["makespan"] <= metrics["makespan"]
            and ours_metrics["mean_flow_time"] <= metrics["mean_flow_time"]
        )
        strictly_better = (
            ours_metrics["makespan"] < metrics["makespan"]
            or ours_metrics["mean_flow_time"] < metrics["mean_flow_time"]
        )
        baselines[policy] = {
            "baseline_over_ours_makespan": makespan_ratio,
            "baseline_over_ours_mean_flow": flow_ratio,
            "ours_pareto_dominates": bool(weakly_better and strictly_better),
            "ours_weakly_pareto_dominates": bool(weakly_better),
        }
    return {
        "baselines": baselines,
        "dominance_is_descriptive_not_gate": True,
    }


def _aggregate_results(results: Sequence[ScheduleResult]) -> dict[str, Any]:
    metrics = [result.metrics for result in results]
    return {
        "instance_count": len(results),
        "feasible_instance_count": sum(result.feasible for result in results),
        "mean_makespan": _round(fmean(row["makespan"] for row in metrics)),
        "mean_flow_time": _round(
            fmean(row["mean_flow_time"] for row in metrics)
        ),
        "mean_resource_violation_units": _round(
            fmean(row["resource_violation_units"] for row in metrics)
        ),
        "mean_mode_reconfiguration_count": _round(
            fmean(row["mode_reconfiguration_count"] for row in metrics)
        ),
        "mean_renewable_configuration_l1": _round(
            fmean(row["renewable_configuration_l1"] for row in metrics)
        ),
    }


def _cost_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    return _round(float(numerator) / float(denominator))


def _round(value: float, digits: int = 9) -> float:
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite MMRCPSP metric: {value}")
    return round(float(value), digits)


def _format_number(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6g}"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic priority-rule comparisons on PSPLIB .mm files."
    )
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    report = build_mmrcpsp_benchmark(args.instances)
    write_artifacts(
        report,
        json_path=args.output,
        markdown_path=args.markdown_output,
    )
    print(args.output)
    print(args.markdown_output)


if __name__ == "__main__":
    main()
