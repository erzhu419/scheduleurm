"""Deterministic Flexible Job-Shop Scheduling Problem benchmark sidecar.

This module executes standard non-preemptive FJSP instances independently of
the legacy Scheduleurm scheduler.  At each serial schedule-generation step, it
enumerates the next precedence-feasible operation of every job on every
eligible machine.  ``ours_robust_maxweight`` selects the exact best candidate
row under the repository's theorem-dispatch score

    Q^T lower_service - bounded_penalty.

The classic dispatch rules see the identical action family.  The optional
reassignment experiment is explicitly an extension for *unstarted*
operations: the fastest eligible machine is treated as a precommitted route,
and selecting another eligible machine incurs a declared setup time.  Standard
FJSPLIB files are non-preemptive and do not justify running-operation migration.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.fjsp_instances import FJSPInstance, Operation, load_fjsp_instance
from algorithm.theorem_dispatch.global_dispatch import select_global_action


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_ORDER = (
    "ours_robust_maxweight",
    "earliest_completion_time",
    "shortest_processing_time",
    "most_work_remaining",
    "most_operations_remaining",
    "fifo_job_order",
)
POLICY_LABELS = {
    "ours_robust_maxweight": "Ours: robust MaxWeight",
    "earliest_completion_time": "Earliest Completion Time (ECT)",
    "shortest_processing_time": "Shortest Processing Time (SPT)",
    "most_work_remaining": "Most Work Remaining (MWKR)",
    "most_operations_remaining": "Most Operations Remaining (MOR)",
    "fifo_job_order": "First-In, First-Out (FIFO)",
}


@dataclass(frozen=True)
class ReassignmentConfig:
    enabled: bool = False
    setup_time: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.setup_time)) or float(self.setup_time) < 0.0:
            raise ValueError("reassignment setup time must be finite and non-negative")
        if not self.enabled and float(self.setup_time) != 0.0:
            raise ValueError("nonzero reassignment setup time requires the explicit extension")

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "scope": "unstarted_operation_only" if self.enabled else "disabled",
            "reference_assignment": "fastest_eligible_machine",
            "setup_time_per_reassignment": _number(self.setup_time),
            "cost_source": "explicit_experiment_parameter" if self.enabled else "not_applicable",
            "running_operation_migration_supported": False,
            "preemption_supported": False,
            "source_file_encodes_reassignment_cost": False,
        }


@dataclass(frozen=True)
class ScheduledOperation:
    job_id: int
    operation_id: int
    machine_id: int
    start_time: float
    end_time: float
    processing_time: float
    reconfiguration_time: float
    preferred_machine_id: int
    reassignment_semantics_enabled: bool

    @property
    def was_reassigned(self) -> bool:
        return bool(
            self.reassignment_semantics_enabled
            and self.machine_id != self.preferred_machine_id
        )

    @property
    def uses_nonpreferred_machine(self) -> bool:
        return self.machine_id != self.preferred_machine_id

    @property
    def occupied_time(self) -> float:
        return self.processing_time + self.reconfiguration_time

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": int(self.job_id),
            "operation_id": int(self.operation_id),
            "machine_id": int(self.machine_id),
            "preferred_machine_id": int(self.preferred_machine_id),
            "start_time": _number(self.start_time),
            "end_time": _number(self.end_time),
            "processing_time": _number(self.processing_time),
            "reconfiguration_time": _number(self.reconfiguration_time),
            "occupied_time": _number(self.occupied_time),
            "uses_nonpreferred_machine": bool(self.uses_nonpreferred_machine),
            "unstarted_operation_reassigned": bool(self.was_reassigned),
        }


@dataclass(frozen=True)
class CandidateAction:
    operation: Operation
    machine_id: int
    processing_time: float
    reconfiguration_time: float
    start_time: float
    end_time: float
    remaining_minimum_work: float
    remaining_operation_count: int
    reassignment_semantics_enabled: bool

    @property
    def action_id(self) -> str:
        return (
            f"fjsp|job={self.operation.job_id}|op={self.operation.operation_id}|"
            f"machine={self.machine_id}"
        )

    @property
    def task_id(self) -> str:
        return f"fjsp:job:{self.operation.job_id}:op:{self.operation.operation_id}"

    @property
    def workload_key(self) -> str:
        return f"fjsp_job_{self.operation.job_id}"

    @property
    def occupied_time(self) -> float:
        return self.processing_time + self.reconfiguration_time

    @property
    def was_reassigned(self) -> bool:
        return bool(
            self.reassignment_semantics_enabled
            and self.machine_id != self.operation.preferred_machine_id
        )

    @property
    def uses_nonpreferred_machine(self) -> bool:
        return self.machine_id != self.operation.preferred_machine_id

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "task_id": self.task_id,
            "job_id": int(self.operation.job_id),
            "operation_id": int(self.operation.operation_id),
            "machine_id": int(self.machine_id),
            "preferred_machine_id": int(self.operation.preferred_machine_id),
            "processing_time": _number(self.processing_time),
            "reconfiguration_time": _number(self.reconfiguration_time),
            "start_time": _number(self.start_time),
            "end_time": _number(self.end_time),
            "remaining_minimum_work": _number(self.remaining_minimum_work),
            "remaining_operation_count": int(self.remaining_operation_count),
            "uses_nonpreferred_machine": bool(self.uses_nonpreferred_machine),
            "unstarted_operation_reassigned": bool(self.was_reassigned),
        }


def enumerate_feasible_actions(
    instance: FJSPInstance,
    *,
    next_operation_by_job: Sequence[int],
    job_ready_times: Sequence[float],
    machine_timelines: Sequence[Sequence[ScheduledOperation]],
    reassignment: ReassignmentConfig | None = None,
) -> list[CandidateAction]:
    """Enumerate all precedence-feasible next-operation/machine actions."""

    config = reassignment or ReassignmentConfig()
    if len(next_operation_by_job) != instance.job_count:
        raise ValueError("next-operation vector does not match job count")
    if len(job_ready_times) != instance.job_count:
        raise ValueError("job-ready vector does not match job count")
    if len(machine_timelines) != instance.machine_count:
        raise ValueError("machine timeline vector does not match machine count")

    actions: list[CandidateAction] = []
    for job in instance.jobs:
        operation_id = int(next_operation_by_job[job.job_id])
        if operation_id >= len(job.operations):
            continue
        if operation_id < 0:
            raise ValueError("next-operation indices must be non-negative")
        operation = job.operations[operation_id]
        remaining_work = sum(
            item.minimum_processing_time for item in job.operations[operation_id:]
        )
        remaining_count = len(job.operations) - operation_id
        for option in operation.alternatives:
            reassigned = option.machine_id != operation.preferred_machine_id
            reconfiguration_time = (
                float(config.setup_time) if config.enabled and reassigned else 0.0
            )
            occupied_time = float(option.processing_time) + reconfiguration_time
            start_time = _earliest_gap(
                machine_timelines[option.machine_id],
                earliest=float(job_ready_times[job.job_id]),
                duration=occupied_time,
            )
            actions.append(
                CandidateAction(
                    operation=operation,
                    machine_id=option.machine_id,
                    processing_time=float(option.processing_time),
                    reconfiguration_time=reconfiguration_time,
                    start_time=start_time,
                    end_time=start_time + occupied_time,
                    remaining_minimum_work=float(remaining_work),
                    remaining_operation_count=remaining_count,
                    reassignment_semantics_enabled=bool(config.enabled),
                )
            )
    return sorted(actions, key=lambda action: action.action_id)


def build_schedule(
    instance: FJSPInstance,
    policy: str,
    *,
    reassignment: ReassignmentConfig | None = None,
) -> dict[str, Any]:
    if policy not in POLICY_ORDER:
        raise ValueError(f"unknown FJSP policy: {policy}")
    config = reassignment or ReassignmentConfig()
    next_operation = [0 for _ in instance.jobs]
    job_ready = [0.0 for _ in instance.jobs]
    machine_timelines: list[list[ScheduledOperation]] = [
        [] for _ in range(instance.machine_count)
    ]
    scheduled: list[ScheduledOperation] = []
    trace: list[dict[str, Any]] = []
    penalty_time_scale = _penalty_time_scale(instance)

    while len(scheduled) < instance.operation_count:
        candidates = enumerate_feasible_actions(
            instance,
            next_operation_by_job=next_operation,
            job_ready_times=job_ready,
            machine_timelines=machine_timelines,
            reassignment=config,
        )
        if not candidates:
            raise RuntimeError("no precedence-feasible FJSP action remains")
        selected, selection_audit = _select_candidate(
            policy,
            candidates,
            penalty_time_scale=penalty_time_scale,
        )
        scheduled_operation = ScheduledOperation(
            job_id=selected.operation.job_id,
            operation_id=selected.operation.operation_id,
            machine_id=selected.machine_id,
            start_time=selected.start_time,
            end_time=selected.end_time,
            processing_time=selected.processing_time,
            reconfiguration_time=selected.reconfiguration_time,
            preferred_machine_id=selected.operation.preferred_machine_id,
            reassignment_semantics_enabled=bool(config.enabled),
        )
        machine_timeline = machine_timelines[selected.machine_id]
        machine_timeline.append(scheduled_operation)
        machine_timeline.sort(
            key=lambda item: (item.start_time, item.end_time, item.job_id, item.operation_id)
        )
        scheduled.append(scheduled_operation)
        next_operation[selected.operation.job_id] += 1
        job_ready[selected.operation.job_id] = selected.end_time
        trace.append(
            {
                "decision_index": len(trace),
                "candidate_count": len(candidates),
                "selected_action": selected.snapshot(),
                "selection_audit": selection_audit,
            }
        )

    feasibility = verify_schedule(instance, scheduled, reassignment=config)
    completion_times = [0.0 for _ in instance.jobs]
    for operation in scheduled:
        completion_times[operation.job_id] = max(
            completion_times[operation.job_id], operation.end_time
        )
    makespan = max(completion_times, default=0.0)
    mean_flow = sum(completion_times) / max(1, len(completion_times))
    reconfigured = [operation for operation in scheduled if operation.was_reassigned]
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "schedule": [
            operation.snapshot()
            for operation in sorted(
                scheduled,
                key=lambda item: (item.start_time, item.machine_id, item.job_id, item.operation_id),
            )
        ],
        "decision_trace": trace,
        "metrics": {
            "makespan": _number(makespan),
            "mean_flow_time": _number(mean_flow),
            "reconfiguration_count": len(reconfigured),
            "reconfiguration_time": _number(
                sum(operation.reconfiguration_time for operation in reconfigured)
            ),
            "makespan_over_lower_bound": _number(
                makespan / max(1.0, float(instance.lower_bound_makespan))
            ),
        },
        "job_completion_times": [_number(value) for value in completion_times],
        "feasibility": feasibility,
    }


def verify_schedule(
    instance: FJSPInstance,
    schedule: Sequence[ScheduledOperation],
    *,
    reassignment: ReassignmentConfig | None = None,
) -> dict[str, Any]:
    config = reassignment or ReassignmentConfig()
    errors: list[str] = []
    by_operation: dict[tuple[int, int], ScheduledOperation] = {}
    for item in schedule:
        key = (item.job_id, item.operation_id)
        if key in by_operation:
            errors.append(f"duplicate operation {key}")
            continue
        by_operation[key] = item
        try:
            operation = instance.operation(*key)
        except KeyError:
            errors.append(f"unknown operation {key}")
            continue
        option = operation.option_for(item.machine_id)
        if option is None:
            errors.append(f"ineligible machine {item.machine_id} for operation {key}")
            continue
        if not _close(item.processing_time, float(option.processing_time)):
            errors.append(f"processing time mismatch for operation {key}")
        expected_reconfiguration = (
            float(config.setup_time)
            if config.enabled and item.machine_id != operation.preferred_machine_id
            else 0.0
        )
        if not _close(item.reconfiguration_time, expected_reconfiguration):
            errors.append(f"reconfiguration time mismatch for operation {key}")
        if bool(item.reassignment_semantics_enabled) != bool(config.enabled):
            errors.append(f"reassignment semantic flag mismatch for operation {key}")
        if item.start_time < -1e-9 or item.end_time + 1e-9 < item.start_time:
            errors.append(f"invalid time interval for operation {key}")
        if not _close(item.end_time - item.start_time, item.occupied_time):
            errors.append(f"occupied duration mismatch for operation {key}")

    expected_keys = {
        (operation.job_id, operation.operation_id)
        for job in instance.jobs
        for operation in job.operations
    }
    missing = sorted(expected_keys - set(by_operation))
    if missing:
        errors.append(f"missing operations {missing}")

    precedence_ok = True
    for job in instance.jobs:
        for operation_id in range(1, len(job.operations)):
            previous = by_operation.get((job.job_id, operation_id - 1))
            current = by_operation.get((job.job_id, operation_id))
            if previous is not None and current is not None:
                if current.start_time + 1e-9 < previous.end_time:
                    precedence_ok = False
                    errors.append(
                        f"precedence violation job {job.job_id}: {operation_id - 1}->{operation_id}"
                    )

    machine_nonoverlap_ok = True
    for machine_id in range(instance.machine_count):
        timeline = sorted(
            (item for item in schedule if item.machine_id == machine_id),
            key=lambda item: (item.start_time, item.end_time, item.job_id, item.operation_id),
        )
        for previous, current in zip(timeline, timeline[1:]):
            if current.start_time + 1e-9 < previous.end_time:
                machine_nonoverlap_ok = False
                errors.append(
                    f"machine {machine_id} overlap between "
                    f"({previous.job_id},{previous.operation_id}) and "
                    f"({current.job_id},{current.operation_id})"
                )

    all_operations_once = len(by_operation) == instance.operation_count and not missing
    return {
        "pass": not errors,
        "all_operations_scheduled_once": bool(all_operations_once),
        "operation_precedence": bool(precedence_ok),
        "machine_nonoverlap": bool(machine_nonoverlap_ok),
        "eligible_machine_assignments": not any("ineligible machine" in item for item in errors),
        "declared_reconfiguration_semantics": not any(
            "reconfiguration time mismatch" in item
            or "reassignment semantic flag mismatch" in item
            for item in errors
        ),
        "error_count": len(errors),
        "errors": errors,
    }


def run_benchmark(
    instance: FJSPInstance,
    *,
    reassignment: ReassignmentConfig | None = None,
    known_best_makespan: float | None = None,
) -> dict[str, Any]:
    config = reassignment or ReassignmentConfig()
    if known_best_makespan is not None and (
        not math.isfinite(float(known_best_makespan)) or float(known_best_makespan) <= 0.0
    ):
        raise ValueError("known best makespan must be finite and positive")
    policies = {
        policy: build_schedule(instance, policy, reassignment=config)
        for policy in POLICY_ORDER
    }
    if known_best_makespan is not None:
        for result in policies.values():
            result["metrics"]["makespan_over_known_best"] = _number(
                float(result["metrics"]["makespan"]) / float(known_best_makespan)
            )

    baselines = [policy for policy in POLICY_ORDER if policy != "ours_robust_maxweight"]
    objective_keys = ("makespan", "mean_flow_time", "reconfiguration_time")
    best_baseline = {
        metric: min(
            baselines,
            key=lambda policy: (float(policies[policy]["metrics"][metric]), policy),
        )
        for metric in objective_keys
    }
    ours_metrics = policies["ours_robust_maxweight"]["metrics"]
    ours_ratios = {
        metric: _cost_ratio(
            float(ours_metrics[metric]),
            float(policies[best_baseline[metric]]["metrics"][metric]),
        )
        for metric in objective_keys
    }
    dominated_by = [
        policy
        for policy in baselines
        if _dominates(policies[policy]["metrics"], ours_metrics, objective_keys)
    ]
    dominates = [
        policy
        for policy in baselines
        if _dominates(ours_metrics, policies[policy]["metrics"], objective_keys)
    ]
    return {
        "schema_version": "scheduleurm.fjsp_benchmark.v1",
        "deterministic": True,
        "implementation": _implementation_manifest(),
        "instance": instance.snapshot(),
        "known_best_makespan": (
            _number(known_best_makespan) if known_best_makespan is not None else None
        ),
        "semantics": {
            "problem": "nonpreemptive_flexible_job_shop_scheduling",
            "release_times": "all_jobs_available_at_time_zero",
            "schedule_generation": "serial_insertion_with_earliest_feasible_machine_gap",
            "candidate_action": "next_precedence_feasible_operation_x_eligible_machine",
            "ours_score": "Q_transpose_lower_service_minus_bounded_penalty",
            "ours_oracle": "exact_over_each_finite_dispatch_candidate_set",
            "penalty_normalization": "minimum_processing_time_unit",
            "penalty_bounded_on_each_finite_instance": True,
            "reassignment": config.snapshot(),
        },
        "policy_order": list(POLICY_ORDER),
        "policies": policies,
        "comparison": {
            "objective_direction": "all_reported_metrics_are_costs_to_minimize",
            "best_baseline_by_metric": best_baseline,
            "ours_over_best_baseline_ratio": ours_ratios,
            "ours_pareto_dominated_by_baselines": dominated_by,
            "baselines_pareto_dominated_by_ours": dominates,
        },
        "gate": {
            "all_schedules_feasible": all(
                bool(result["feasibility"]["pass"]) for result in policies.values()
            ),
            "all_policies_share_instance_and_action_semantics": True,
            "running_operation_migration_claimed": False,
            "pass": all(bool(result["feasibility"]["pass"]) for result in policies.values()),
        },
        "scope": (
            "Deterministic benchmark evidence on the declared FJSP instance.  This is not "
            "a claim of optimality, arbitrary-instance dominance, shop-floor deployment, "
            "or running-operation migration."
        ),
    }


def artifact_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def markdown_report(report: Mapping[str, Any]) -> str:
    instance = report["instance"]
    semantics = report["semantics"]
    reassign = semantics["reassignment"]
    lines = [
        "# Flexible Job-Shop Scheduling Problem Benchmark",
        "",
        f"- Instance: `{instance['name']}`",
        f"- Source SHA-256: `{instance['source_sha256']}`",
        (
            f"- Size: {instance['job_count']} jobs, {instance['machine_count']} machines, "
            f"{instance['operation_count']} operations"
        ),
        f"- Constructive lower bound: `{_format_number(instance['lower_bound_makespan'])}`",
        (
            f"- Declared best-known makespan: `{_format_number(report['known_best_makespan'])}`"
            if report.get("known_best_makespan") is not None
            else "- Declared best-known makespan: `not supplied`"
        ),
        f"- Feasibility gate: `{str(bool(report['gate']['pass'])).lower()}`",
        "",
        (
            "| Policy | Makespan | Makespan/BKS | Mean flow | Reassignments | "
            "Reconfiguration time | Feasible |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in report["policy_order"]:
        result = report["policies"][policy]
        metrics = result["metrics"]
        lines.append(
            f"| {result['policy_label']} | {_format_number(metrics['makespan'])} | "
            f"{_known_best_ratio_text(metrics)} | "
            f"{_format_number(metrics['mean_flow_time'])} | "
            f"{metrics['reconfiguration_count']} | "
            f"{_format_number(metrics['reconfiguration_time'])} | "
            f"{str(bool(result['feasibility']['pass'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Action And Reassignment Semantics",
            "",
            (
                "Every decision considers the next precedence-feasible operation of each "
                "unfinished job on every machine listed for that operation."
            ),
            "",
            (
                f"The reassignment extension is `{str(bool(reassign['enabled'])).lower()}`. "
                f"Its scope is `{reassign['scope']}` and its declared setup time is "
                f"`{_format_number(reassign['setup_time_per_reassignment'])}`."
            ),
            (
                "Standard FJSPLIB semantics are non-preemptive; running-operation "
                "migration is not claimed."
            ),
            "",
            "## Comparison Boundary",
            "",
            str(report["scope"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(artifact_json(report), encoding="utf-8")
    output_markdown.write_text(markdown_report(report), encoding="utf-8")
    return output_json, output_markdown


def _select_candidate(
    policy: str,
    candidates: Sequence[CandidateAction],
    *,
    penalty_time_scale: float,
) -> tuple[CandidateAction, dict[str, Any]]:
    if policy == "ours_robust_maxweight":
        frontier = min(candidate.start_time for candidate in candidates)
        rows = []
        queue: dict[str, float] = {}
        for candidate in candidates:
            queue[candidate.workload_key] = max(
                queue.get(candidate.workload_key, 0.0),
                candidate.remaining_minimum_work,
            )
            delay_penalty = (
                max(0.0, candidate.start_time - frontier) / penalty_time_scale
            )
            reassignment_penalty = candidate.reconfiguration_time / penalty_time_scale
            rows.append(
                {
                    "domain": "fjsp",
                    "action_type": (
                        "reassign_unstarted_operation"
                        if candidate.was_reassigned
                        else "dispatch"
                    ),
                    "task_id": candidate.task_id,
                    "action_id": candidate.action_id,
                    "resource_id": f"fjsp:machine:{candidate.machine_id}",
                    "resource_ids": (
                        f"fjsp:job:{candidate.operation.job_id}",
                        f"fjsp:machine:{candidate.machine_id}",
                    ),
                    "workload_key": candidate.workload_key,
                    "service_workload_key": candidate.workload_key,
                    "lower_service": {
                        candidate.workload_key: 1.0 / max(1e-12, candidate.occupied_time)
                    },
                    "selected_class_lower_service": 1.0
                    / max(1e-12, candidate.occupied_time),
                    "penalty_units": delay_penalty + reassignment_penalty,
                    "delay_penalty_units": delay_penalty,
                    "reassignment_penalty_units": reassignment_penalty,
                    "score_semantics": "robust_maxweight_lower_service",
                    "theorem_ready": True,
                    "scheduler_hint_only": True,
                }
            )
        selected = select_global_action(
            rows,
            queue,
            max_batch_size=1,
            max_configurations=max(1, len(rows)),
        )
        selected_id = selected.selected_action_ids[0]
        candidate = next(item for item in candidates if item.action_id == selected_id)
        return candidate, {
            "rule": "robust_maxweight_lower_service",
            "queue_vector": {key: _number(value) for key, value in sorted(queue.items())},
            "selected_global_action": selected.snapshot(),
            "penalty_time_scale": _number(penalty_time_scale),
        }

    key_functions = {
        "earliest_completion_time": lambda item: (
            item.end_time,
            item.occupied_time,
            item.operation.job_id,
            item.operation.operation_id,
            item.machine_id,
        ),
        "shortest_processing_time": lambda item: (
            item.occupied_time,
            item.end_time,
            item.operation.job_id,
            item.operation.operation_id,
            item.machine_id,
        ),
        "most_work_remaining": lambda item: (
            -item.remaining_minimum_work,
            item.end_time,
            item.occupied_time,
            item.operation.job_id,
            item.operation.operation_id,
            item.machine_id,
        ),
        "most_operations_remaining": lambda item: (
            -item.remaining_operation_count,
            item.end_time,
            item.occupied_time,
            item.operation.job_id,
            item.operation.operation_id,
            item.machine_id,
        ),
        "fifo_job_order": lambda item: (
            item.operation.job_id,
            item.operation.operation_id,
            item.end_time,
            item.occupied_time,
            item.machine_id,
        ),
    }
    selected = min(candidates, key=key_functions[policy])
    return selected, {
        "rule": policy,
        "tie_break": "lexicographic_job_operation_machine",
    }


def _earliest_gap(
    timeline: Sequence[ScheduledOperation],
    *,
    earliest: float,
    duration: float,
) -> float:
    if duration <= 0.0:
        raise ValueError("operation duration must be positive")
    start = max(0.0, float(earliest))
    for item in sorted(
        timeline,
        key=lambda row: (row.start_time, row.end_time, row.job_id, row.operation_id),
    ):
        if start + duration <= item.start_time + 1e-9:
            return start
        if start < item.end_time - 1e-9:
            start = item.end_time
    return start


def _penalty_time_scale(instance: FJSPInstance) -> float:
    minimum_duration = min(
        float(option.processing_time)
        for job in instance.jobs
        for operation in job.operations
        for option in operation.alternatives
    )
    return max(1.0, minimum_duration)


def _dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    metrics: Iterable[str],
) -> bool:
    pairs = [(float(left[key]), float(right[key])) for key in metrics]
    return all(lhs <= rhs + 1e-9 for lhs, rhs in pairs) and any(
        lhs < rhs - 1e-9 for lhs, rhs in pairs
    )


def _cost_ratio(value: float, reference: float) -> int | float | None:
    if abs(float(reference)) <= 1e-12:
        return 1 if abs(float(value)) <= 1e-12 else None
    return _number(float(value) / float(reference))


def _known_best_ratio_text(metrics: Mapping[str, Any]) -> str:
    if "makespan_over_known_best" not in metrics:
        return "n/a"
    return _format_number(metrics["makespan_over_known_best"])


def _implementation_manifest() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("fjsp_instances.py"),
    )
    files = []
    digest = sha256()
    for path in paths:
        data = path.read_bytes()
        file_hash = sha256(data).hexdigest()
        relative = path.relative_to(REPO_ROOT).as_posix()
        files.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return {
        "files": files,
        "combined_sha256": digest.hexdigest(),
    }


def _number(value: Any) -> int | float:
    number = float(value)
    rounded = round(number, 9)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _format_number(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-8 * max(1.0, abs(float(right)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic FJSPLIB/Brandimarte dispatch benchmarks."
    )
    parser.add_argument("instance", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--enable-unstarted-reassignment", action="store_true")
    parser.add_argument("--reassignment-time", type=float, default=0.0)
    parser.add_argument("--known-best-makespan", type=float)
    args = parser.parse_args()

    if args.reassignment_time and not args.enable_unstarted_reassignment:
        parser.error("--reassignment-time requires --enable-unstarted-reassignment")
    instance = load_fjsp_instance(args.instance)
    config = ReassignmentConfig(
        enabled=bool(args.enable_unstarted_reassignment),
        setup_time=float(args.reassignment_time),
    )
    report = run_benchmark(
        instance,
        reassignment=config,
        known_best_makespan=args.known_best_makespan,
    )
    output_json = args.output_json or args.instance.with_name(
        f"{args.instance.stem}_benchmark.json"
    )
    output_markdown = args.output_md or args.instance.with_name(
        f"{args.instance.stem}_benchmark.md"
    )
    json_path, markdown_path = write_artifacts(
        report,
        json_path=output_json,
        markdown_path=output_markdown,
    )
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
