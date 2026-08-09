"""Optional CP-SAT references for the deterministic OR portability domains.

OR-Tools is intentionally imported only inside solver entry points.  The live
scheduler and its algorithm hooks therefore have no dependency on this module.
The references optimize makespan under a declared wall-clock budget; they do
not replace the multiobjective renewal-frame policy.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from algorithm.experiments.fjsp_benchmark import (
    ReassignmentConfig,
    ScheduledOperation,
    verify_schedule,
)
from algorithm.experiments.fjsp_instances import FJSPInstance
from algorithm.experiments.mmrcpsp_benchmark import ScheduleEntry, _evaluate_schedule
from algorithm.experiments.mmrcpsp_instances import MMRCPSPInstance


class ExactReferenceUnavailable(RuntimeError):
    """Raised when the isolated OR-Tools dependency is unavailable."""


def solve_fjsp_cp_sat(
    instance: FJSPInstance,
    *,
    time_limit_s: float,
    random_seed: int = 20260809,
    workers: int = 1,
    incumbent_schedule: Sequence[ScheduledOperation | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    limit, worker_count = _validate_budget(time_limit_s, workers)
    cp_model = _cp_model()
    model = cp_model.CpModel()
    horizon = sum(
        max(option.processing_time for option in operation.alternatives)
        for job in instance.jobs
        for operation in job.operations
    )
    starts: dict[tuple[int, int], Any] = {}
    ends: dict[tuple[int, int], Any] = {}
    presences: dict[tuple[int, int, int], Any] = {}
    machine_intervals: list[list[Any]] = [
        [] for _ in range(instance.machine_count)
    ]
    for job in instance.jobs:
        for operation in job.operations:
            key = (operation.job_id, operation.operation_id)
            start = model.new_int_var(0, horizon, f"s_{key[0]}_{key[1]}")
            end = model.new_int_var(0, horizon, f"e_{key[0]}_{key[1]}")
            starts[key] = start
            ends[key] = end
            alternatives = []
            for option in operation.alternatives:
                presence = model.new_bool_var(
                    f"x_{key[0]}_{key[1]}_{option.machine_id}"
                )
                interval = model.new_optional_interval_var(
                    start,
                    option.processing_time,
                    end,
                    presence,
                    f"i_{key[0]}_{key[1]}_{option.machine_id}",
                )
                presences[(key[0], key[1], option.machine_id)] = presence
                machine_intervals[option.machine_id].append(interval)
                alternatives.append(presence)
            model.add_exactly_one(alternatives)
        for previous, current in zip(job.operations, job.operations[1:]):
            model.add(
                ends[(previous.job_id, previous.operation_id)]
                <= starts[(current.job_id, current.operation_id)]
            )
    for intervals in machine_intervals:
        if intervals:
            model.add_no_overlap(intervals)
    final_ends = [ends[(job.job_id, job.operations[-1].operation_id)] for job in instance.jobs]
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, final_ends)
    model.minimize(makespan)
    incumbent = _coerce_fjsp_incumbent(instance, incumbent_schedule)
    if incumbent is not None:
        incumbent_map = {(row.job_id, row.operation_id): row for row in incumbent}
        for key, row in incumbent_map.items():
            model.add_hint(starts[key], int(round(row.start_time)))
            model.add_hint(ends[key], int(round(row.end_time)))
            operation = instance.operation(*key)
            for option in operation.alternatives:
                model.add_hint(
                    presences[(key[0], key[1], option.machine_id)],
                    int(option.machine_id == row.machine_id),
                )
        model.add_hint(makespan, int(round(max(row.end_time for row in incumbent))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_search_workers = worker_count
    solver.parameters.random_seed = int(random_seed)
    status = solver.solve(model)
    status_name = solver.status_name(status)
    feasible = status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    schedule: list[ScheduledOperation] = []
    if feasible:
        for job in instance.jobs:
            for operation in job.operations:
                selected = [
                    option
                    for option in operation.alternatives
                    if solver.boolean_value(
                        presences[(operation.job_id, operation.operation_id, option.machine_id)]
                    )
                ]
                if len(selected) != 1:
                    raise RuntimeError("CP-SAT FJSP solution lacks a unique machine choice")
                option = selected[0]
                start = solver.value(starts[(operation.job_id, operation.operation_id)])
                end = solver.value(ends[(operation.job_id, operation.operation_id)])
                schedule.append(
                    ScheduledOperation(
                        job_id=operation.job_id,
                        operation_id=operation.operation_id,
                        machine_id=option.machine_id,
                        start_time=float(start),
                        end_time=float(end),
                        processing_time=float(option.processing_time),
                        reconfiguration_time=0.0,
                        preferred_machine_id=operation.preferred_machine_id,
                        reassignment_semantics_enabled=False,
                    )
                )
    audit = verify_schedule(instance, schedule, reassignment=ReassignmentConfig()) if feasible else None
    completion = [
        max(
            (row.end_time for row in schedule if row.job_id == job.job_id),
            default=0.0,
        )
        for job in instance.jobs
    ]
    return {
        "schema_version": "scheduleurm.fjsp_cp_sat_reference.v1",
        "solver": "OR-Tools CP-SAT",
        "objective": "makespan",
        "time_limit_s": limit,
        "worker_count": worker_count,
        "random_seed": int(random_seed),
        "incumbent_hint_provided": incumbent is not None,
        "incumbent_hint_independently_verified": incumbent is not None,
        "status": status_name,
        "feasible_solution": feasible,
        "optimality_proved": status == cp_model.OPTIMAL,
        "objective_makespan": (
            float(solver.objective_value) if feasible else None
        ),
        "best_bound_makespan": (
            float(solver.best_objective_bound) if feasible else None
        ),
        "relative_gap": _relative_gap(solver) if feasible else None,
        "mean_flow_time": (
            sum(completion) / len(completion) if completion else None
        ),
        "wall_time_s": float(solver.wall_time),
        "verifier": audit,
        "schedule": [row.snapshot() for row in schedule],
        "claim_boundary": (
            "A time-bounded makespan reference. FEASIBLE is not an optimality proof; "
            "mean flow is descriptive because CP-SAT optimizes makespan only."
        ),
    }


def solve_mmrcpsp_cp_sat(
    instance: MMRCPSPInstance,
    *,
    time_limit_s: float,
    random_seed: int = 20260809,
    workers: int = 1,
    incumbent_schedule: Sequence[ScheduleEntry | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    limit, worker_count = _validate_budget(time_limit_s, workers)
    cp_model = _cp_model()
    model = cp_model.CpModel()
    horizon = max(
        int(instance.horizon),
        sum(max(mode.duration for mode in activity.modes) for activity in instance.activities),
        1,
    )
    starts: dict[int, Any] = {}
    ends: dict[int, Any] = {}
    presences: dict[tuple[int, int], Any] = {}
    renewable_intervals: list[list[Any]] = [
        [] for _ in instance.renewable_capacities
    ]
    renewable_demands: list[list[int]] = [
        [] for _ in instance.renewable_capacities
    ]
    for activity in instance.activities:
        start = model.new_int_var(0, horizon, f"s_{activity.job_id}")
        end = model.new_int_var(0, horizon, f"e_{activity.job_id}")
        starts[activity.job_id] = start
        ends[activity.job_id] = end
        choices = []
        for mode in activity.modes:
            presence = model.new_bool_var(f"x_{activity.job_id}_{mode.mode_id}")
            interval = model.new_optional_interval_var(
                start,
                mode.duration,
                end,
                presence,
                f"i_{activity.job_id}_{mode.mode_id}",
            )
            presences[(activity.job_id, mode.mode_id)] = presence
            choices.append(presence)
            for resource, demand in enumerate(mode.renewable_demands):
                renewable_intervals[resource].append(interval)
                renewable_demands[resource].append(int(demand))
        model.add_exactly_one(choices)
    # Nonrenewable capacity is shared across all activities, not per activity.
    for resource, capacity in enumerate(instance.nonrenewable_capacities):
        model.add(
            sum(
                int(mode.nonrenewable_demands[resource])
                * presences[(activity.job_id, mode.mode_id)]
                for activity in instance.activities
                for mode in activity.modes
            )
            <= int(capacity)
        )
    for activity in instance.activities:
        for successor in activity.successors:
            model.add(ends[activity.job_id] <= starts[successor])
    for resource, capacity in enumerate(instance.renewable_capacities):
        if renewable_intervals[resource]:
            model.add_cumulative(
                renewable_intervals[resource],
                renewable_demands[resource],
                int(capacity),
            )
    sink_ends = [ends[job_id] for job_id in instance.sink_job_ids]
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, sink_ends)
    model.minimize(makespan)
    incumbent = _coerce_mmrcpsp_incumbent(instance, incumbent_schedule)
    if incumbent is not None:
        incumbent_map = {row.job_id: row for row in incumbent}
        for activity in instance.activities:
            row = incumbent_map[activity.job_id]
            model.add_hint(starts[activity.job_id], row.start)
            model.add_hint(ends[activity.job_id], row.finish)
            for mode in activity.modes:
                model.add_hint(
                    presences[(activity.job_id, mode.mode_id)],
                    int(mode.mode_id == row.mode_id),
                )
        model.add_hint(makespan, max(row.finish for row in incumbent))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_search_workers = worker_count
    solver.parameters.random_seed = int(random_seed)
    status = solver.solve(model)
    status_name = solver.status_name(status)
    feasible = status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    entries: list[ScheduleEntry] = []
    if feasible:
        for activity in instance.activities:
            selected = [
                mode
                for mode in activity.modes
                if solver.boolean_value(presences[(activity.job_id, mode.mode_id)])
            ]
            if len(selected) != 1:
                raise RuntimeError("CP-SAT MMRCPSP solution lacks a unique mode choice")
            mode = selected[0]
            start = int(solver.value(starts[activity.job_id]))
            end = int(solver.value(ends[activity.job_id]))
            entries.append(
                ScheduleEntry(
                    job_id=activity.job_id,
                    mode_id=mode.mode_id,
                    start=start,
                    finish=end,
                    duration=mode.duration,
                    renewable_demands=mode.renewable_demands,
                    nonrenewable_demands=mode.nonrenewable_demands,
                    action_score=0.0,
                    queue_weight_proxy=1.0,
                    lower_service=0.0,
                    penalty_units=0.0,
                    action_type="cp_sat_reference",
                )
            )
    audit = _evaluate_schedule(instance, tuple(entries)) if feasible else None
    return {
        "schema_version": "scheduleurm.mmrcpsp_cp_sat_reference.v1",
        "solver": "OR-Tools CP-SAT",
        "objective": "makespan",
        "time_limit_s": limit,
        "worker_count": worker_count,
        "random_seed": int(random_seed),
        "incumbent_hint_provided": incumbent is not None,
        "incumbent_hint_independently_verified": incumbent is not None,
        "status": status_name,
        "feasible_solution": feasible,
        "optimality_proved": status == cp_model.OPTIMAL,
        "objective_makespan": (
            float(solver.objective_value) if feasible else None
        ),
        "best_bound_makespan": (
            float(solver.best_objective_bound) if feasible else None
        ),
        "relative_gap": _relative_gap(solver) if feasible else None,
        "mean_flow_time": (
            float(audit["mean_flow_time"]) if audit is not None else None
        ),
        "wall_time_s": float(solver.wall_time),
        "verifier": audit,
        "schedule": [row.snapshot() for row in entries],
        "claim_boundary": (
            "A time-bounded makespan reference. FEASIBLE is not an optimality proof; "
            "mean flow is descriptive because CP-SAT optimizes makespan only."
        ),
    }


def _cp_model():
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ExactReferenceUnavailable(
            "OR-Tools is optional; install the pinned experiment dependency"
        ) from exc
    return cp_model


def _coerce_fjsp_incumbent(
    instance: FJSPInstance,
    schedule: Sequence[ScheduledOperation | Mapping[str, Any]] | None,
) -> tuple[ScheduledOperation, ...] | None:
    if schedule is None:
        return None
    rows = tuple(
        row
        if isinstance(row, ScheduledOperation)
        else ScheduledOperation(
            job_id=int(row["job_id"]),
            operation_id=int(row["operation_id"]),
            machine_id=int(row["machine_id"]),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            processing_time=float(row["processing_time"]),
            reconfiguration_time=float(row.get("reconfiguration_time", 0.0)),
            preferred_machine_id=int(row["preferred_machine_id"]),
            reassignment_semantics_enabled=False,
        )
        for row in schedule
    )
    audit = verify_schedule(instance, rows, reassignment=ReassignmentConfig())
    if audit["pass"] is not True:
        raise ValueError(f"invalid FJSP incumbent hint: {audit['errors']!r}")
    return rows


def _coerce_mmrcpsp_incumbent(
    instance: MMRCPSPInstance,
    schedule: Sequence[ScheduleEntry | Mapping[str, Any]] | None,
) -> tuple[ScheduleEntry, ...] | None:
    if schedule is None:
        return None
    rows = tuple(
        row
        if isinstance(row, ScheduleEntry)
        else ScheduleEntry(
            job_id=int(row["job_id"]),
            mode_id=int(row["mode_id"]),
            start=int(row["start"]),
            finish=int(row["finish"]),
            duration=int(row["duration"]),
            renewable_demands=tuple(int(value) for value in row["renewable_demands"]),
            nonrenewable_demands=tuple(
                int(value) for value in row["nonrenewable_demands"]
            ),
            action_score=0.0,
            queue_weight_proxy=1.0,
            lower_service=0.0,
            penalty_units=0.0,
            action_type="cp_sat_incumbent_hint",
        )
        for row in schedule
    )
    audit = _evaluate_schedule(instance, rows)
    if audit["schedule_feasible"] is not True:
        raise ValueError("invalid MMRCPSP incumbent hint")
    return rows


def _validate_budget(time_limit_s: float, workers: int) -> tuple[float, int]:
    limit = float(time_limit_s)
    worker_count = int(workers)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("time_limit_s must be finite and positive")
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    return limit, worker_count


def _relative_gap(solver: Any) -> float:
    objective = float(solver.objective_value)
    bound = float(solver.best_objective_bound)
    return max(0.0, (objective - bound) / max(1.0, abs(objective)))


__all__ = [
    "ExactReferenceUnavailable",
    "solve_fjsp_cp_sat",
    "solve_mmrcpsp_cp_sat",
]
