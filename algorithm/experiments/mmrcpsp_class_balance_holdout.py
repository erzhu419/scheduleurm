"""Prospective arrival-tape holdout for MMRCPSP class-balance costs.

Version 1 of the renewal-stream experiment preregistered aggregate final
backlog and aggregate time-average queue.  Its shortest-trajectory baseline
improved both totals while concentrating more backlog in slow project classes.
This follow-up does not rewrite that result.  It reuses the exact committed v1
trajectory library, generates previously unseen arrival tapes, and freezes a
three-coordinate cost vector before those tapes are run:

* total time-average project queue;
* maximum class-specific time-average project queue; and
* maximum class-specific final project backlog.

All pathwise recurrence, drift-identity, and conservation checks continue to
come from the committed v1 simulator.  This is an arrival-tape holdout over a
known instance/library family, not a new structural-instance holdout and not a
positive-recurrence proof.
"""
from __future__ import annotations

from dataclasses import fields
import math
from statistics import fmean
from typing import Any, Mapping, Sequence

from algorithm.experiments.mmrcpsp_renewal_stream_benchmark import (
    NUMERIC_TOLERANCE,
    OURS_POLICY,
    POLICY_ORDER,
    ArrivalScenario,
    RegisteredTrajectory,
    RenewalStreamError,
    _arrival_rates,
    _arrival_tape,
    _arrival_tape_sha256,
    _simulate,
)


SCHEMA_VERSION = "scheduleurm.mmrcpsp_class_balance_holdout.v1"
PARETO_COST_COORDINATES = (
    "total_time_average_queue",
    "max_class_time_average_queue",
    "max_class_final_backlog",
)


def run_class_balance_holdout(
    source_artifact: Mapping[str, Any],
    *,
    scenarios: Sequence[ArrivalScenario],
    seeds: Sequence[int],
    policies: Sequence[str] = POLICY_ORDER,
) -> dict[str, Any]:
    """Run new tapes against the exact registered library stored in v1."""

    source_report = source_artifact.get("report") or {}
    if source_report.get("schema_version") != (
        "scheduleurm.mmrcpsp_renewal_stream_benchmark.v1"
    ):
        raise RenewalStreamError("unexpected v1 source artifact schema")
    source_gate = source_report.get("gate") or {}
    if source_gate.get("pass") is not True:
        raise RenewalStreamError("v1 source artifact protocol is not closed")
    library = _reconstruct_library(source_report["registered_trajectory_library"])
    classes = tuple(sorted({action.workload_class for action in library}))
    if classes != tuple(source_report["workload_classes"]):
        raise RenewalStreamError("v1 source class order or membership drifted")
    selected_scenarios = tuple(scenarios)
    selected_seeds = tuple(int(seed) for seed in seeds)
    selected_policies = tuple(str(policy) for policy in policies)
    if not selected_scenarios or not selected_seeds:
        raise RenewalStreamError("holdout scenarios and seeds must be nonempty")
    if len(selected_seeds) != len(set(selected_seeds)):
        raise RenewalStreamError("holdout seeds must be unique")
    if selected_policies != POLICY_ORDER:
        raise RenewalStreamError("class-balance holdout policy order changed")
    if len({row.scenario_id for row in selected_scenarios}) != len(
        selected_scenarios
    ):
        raise RenewalStreamError("holdout scenario ids must be unique")

    common_horizon = max(action.common_instance_horizon for action in library)
    minimum_duration = {
        workload_class: min(
            action.duration
            for action in library
            if action.workload_class == workload_class
        )
        for workload_class in classes
    }
    activity_count = {
        workload_class: next(
            action.real_activity_count
            for action in library
            if action.workload_class == workload_class
        )
        for workload_class in classes
    }
    runs = []
    tape_hashes = {}
    for scenario in selected_scenarios:
        rates = _arrival_rates(
            scenario, classes=classes, minimum_duration=minimum_duration
        )
        for seed in selected_seeds:
            tape = _arrival_tape(
                scenario, classes=classes, rates=rates, seed=seed
            )
            tape_hash = _arrival_tape_sha256(tape)
            tape_hashes[f"{scenario.scenario_id}:seed_{seed}"] = tape_hash
            for policy in selected_policies:
                base = _simulate(
                    library,
                    classes=classes,
                    activity_count=activity_count,
                    scenario=scenario,
                    policy=policy,
                    seed=seed,
                    arrival_rates=rates,
                    arrival_tape=tape,
                    arrival_tape_sha256=tape_hash,
                    common_horizon=common_horizon,
                )
                runs.append(
                    {
                        **base,
                        "class_balance": _class_balance_metrics(
                            base, classes=classes
                        ),
                    }
                )

    dispositions = _performance_dispositions(runs)
    protocol_pass = all(bool(run["gate"]["pass"]) for run in runs)
    ours_runs = [run for run in runs if run["policy"] == OURS_POLICY]
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "mmrcpsp_prospective_class_balance_arrival_tape_holdout",
        "source_library_contract": {
            "reused_exact_v1_library": True,
            "known_instance_family": True,
            "new_structural_instance_holdout": False,
            "library_action_count": len(library),
            "workload_class_count": len(classes),
            "all_source_actions_feasible": all(
                action.schedule_feasible for action in library
            ),
        },
        "arrival_contract": {
            "previously_unseen_tapes": True,
            "same_scenario_seed_tape_for_every_policy": True,
            "tape_hashes": tape_hashes,
        },
        "policy_order": list(selected_policies),
        "scenario_order": [row.scenario_id for row in selected_scenarios],
        "seed_order": list(selected_seeds),
        "pareto_cost_coordinates": list(PARETO_COST_COORDINATES),
        "runs": runs,
        "performance_dispositions": dispositions,
        "aggregate": {
            "run_count": len(runs),
            "protocol_pass_count": sum(bool(run["gate"]["pass"]) for run in runs),
            "ours_exact_oracle_run_count": sum(
                bool(run["gate"]["exact_duration_normalized_oracle"])
                for run in ours_runs
            ),
            "scenario_seed_count": dispositions["scenario_seed_count"],
            "ours_nondominated_count": dispositions["ours_nondominated_count"],
            "ours_strict_every_baseline_count": dispositions[
                "ours_strict_every_baseline_count"
            ],
            "baseline_dominates_count": dispositions[
                "baseline_dominates_count"
            ],
        },
        "gate": {
            "pass": protocol_pass,
            "status": (
                "MMRCPSP_CLASS_BALANCE_HOLDOUT_PROTOCOL_PASS"
                if protocol_pass
                else "MMRCPSP_CLASS_BALANCE_HOLDOUT_PROTOCOL_FAIL"
            ),
            "performance_superiority_not_required": True,
            "all_pathwise_v1_gates_pass": protocol_pass,
        },
        "claim_boundary": {
            "v1_total_cost_result_retained": True,
            "metric_family_registered_after_v1_total_cost_result": True,
            "arrival_tapes_observed_before_metric_freeze": False,
            "known_instance_and_action_library": True,
            "new_structural_instance_holdout": False,
            "positive_recurrence_claim_ready": False,
            "global_mmrcpsp_optimality_claim_ready": False,
            "performance_superiority_claim_ready": False,
        },
    }
    return _canonical(report)


def compact_class_balance_report(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for run in report["runs"]:
        rows.append(
            {
                "run_id": run["run_id"],
                "scenario_id": run["scenario"]["scenario_id"],
                "scenario_family": run["scenario"]["family"],
                "load_factor": run["scenario"]["load_factor"],
                "seed": run["seed"],
                "policy": run["policy"],
                "arrival_tape_sha256": run["arrival_tape_sha256"],
                "total_time_average_queue": run["class_balance"][
                    "total_time_average_queue"
                ],
                "max_class_time_average_queue": run["class_balance"][
                    "max_class_time_average_queue"
                ],
                "max_class_final_backlog": run["class_balance"][
                    "max_class_final_backlog"
                ],
                "max_class_peak_backlog": run["class_balance"][
                    "max_class_peak_backlog"
                ],
                "class_time_average_queue_std": run["class_balance"][
                    "class_time_average_queue_std"
                ],
                "run_gate_pass": run["gate"]["pass"],
                "exact_duration_normalized_oracle": run["gate"][
                    "exact_duration_normalized_oracle"
                ],
            }
        )
    return {
        "schema_version": "scheduleurm.mmrcpsp_class_balance_holdout.compact.v1",
        "gate": report["gate"],
        "source_library_contract": report["source_library_contract"],
        "arrival_contract": report["arrival_contract"],
        "policy_order": report["policy_order"],
        "scenario_order": report["scenario_order"],
        "seed_order": report["seed_order"],
        "pareto_cost_coordinates": report["pareto_cost_coordinates"],
        "aggregate": report["aggregate"],
        "performance_dispositions": report["performance_dispositions"],
        "rows": rows,
        "claim_boundary": report["claim_boundary"],
    }


def _reconstruct_library(
    snapshots: Sequence[Mapping[str, Any]],
) -> tuple[RegisteredTrajectory, ...]:
    expected_fields = {field.name for field in fields(RegisteredTrajectory)}
    actions = []
    for snapshot in snapshots:
        metrics = snapshot["metrics"]
        values = {
            "action_id": str(snapshot["action_id"]),
            "workload_class": str(snapshot["workload_class"]),
            "source_sha256": str(snapshot["source_sha256"]),
            "sources": tuple(str(row) for row in snapshot["sources"]),
            "trajectory_sha256": str(snapshot["trajectory_sha256"]),
            "duration": int(snapshot["duration_tau"]),
            "common_instance_horizon": int(snapshot["common_instance_horizon"]),
            "real_activity_count": int(snapshot["real_activity_count"]),
            "completion_times": tuple(float(row) for row in snapshot["completion_times"]),
            "lower_project_departure": float(snapshot["lower_project_departure"]),
            "bounded_penalty": float(snapshot["bounded_penalty"]),
            "penalty_uniform_bound": float(snapshot["penalty_uniform_bound"]),
            "makespan": float(metrics["makespan"]),
            "mean_flow_time": float(metrics["mean_flow_time"]),
            "schedule_feasible": bool(metrics["schedule_feasible"]),
            "resource_violation_units": float(metrics["resource_violation_units"]),
            "precedence_violation_count": int(metrics["precedence_violation_count"]),
        }
        if set(values) != expected_fields:
            raise RenewalStreamError("registered trajectory schema drifted")
        action = RegisteredTrajectory(**values)
        if (
            not action.schedule_feasible
            or action.duration <= 0
            or action.lower_project_departure != 1.0
            or action.resource_violation_units > NUMERIC_TOLERANCE
            or action.precedence_violation_count != 0
            or action.bounded_penalty
            > action.penalty_uniform_bound + NUMERIC_TOLERANCE
        ):
            raise RenewalStreamError(f"invalid v1 action {action.action_id}")
        actions.append(action)
    result = tuple(sorted(actions, key=lambda row: row.action_id))
    if not result or len({row.action_id for row in result}) != len(result):
        raise RenewalStreamError("v1 registered library is empty or duplicated")
    return result


def _class_balance_metrics(
    run: Mapping[str, Any], *, classes: Sequence[str]
) -> dict[str, Any]:
    horizon = int(run["scenario"]["horizon"])
    queue_area = {workload_class: 0.0 for workload_class in classes}
    peak = {workload_class: 0 for workload_class in classes}
    for frame in run["frames"]:
        duration = int(frame["duration_tau"])
        for workload_class, value in frame["queue_before"].items():
            queue = int(value)
            queue_area[workload_class] += queue * duration
            peak[workload_class] = max(peak[workload_class], queue)
    averages = {
        workload_class: queue_area[workload_class] / horizon
        for workload_class in classes
    }
    finals = {
        workload_class: int(
            run["workload_conservation"][workload_class][
                "final_queued_projects"
            ]
        )
        for workload_class in classes
    }
    mean = fmean(averages.values())
    variance = fmean((value - mean) ** 2 for value in averages.values())
    total = sum(averages.values())
    if abs(total - float(run["performance"]["time_average_queue"])) > 1e-7:
        raise RenewalStreamError("class queue areas do not recover aggregate queue")
    return {
        "total_time_average_queue": total,
        "max_class_time_average_queue": max(averages.values()),
        "max_class_final_backlog": max(finals.values()),
        "max_class_peak_backlog": max(peak.values()),
        "class_time_average_queue_std": math.sqrt(variance),
        "per_class_time_average_queue": averages,
        "per_class_final_backlog": finals,
        "per_class_peak_backlog": peak,
    }


def _performance_dispositions(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for run in runs:
        key = (str(run["scenario"]["scenario_id"]), int(run["seed"]))
        grouped.setdefault(key, []).append(run)
    rows = []
    for (scenario_id, seed), group in sorted(grouped.items()):
        ours = next(run for run in group if run["policy"] == OURS_POLICY)
        baselines = [run for run in group if run["policy"] != OURS_POLICY]
        dominators = [
            str(run["policy"])
            for run in baselines
            if _dominates(run["class_balance"], ours["class_balance"])
        ]
        dominated = [
            str(run["policy"])
            for run in baselines
            if _dominates(ours["class_balance"], run["class_balance"])
        ]
        if dominators:
            disposition = "baseline_dominates"
        elif len(dominated) == len(baselines):
            disposition = "ours_strict_every_baseline"
        else:
            disposition = "ours_nondominated_tradeoff_or_tie"
        rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "ours_disposition": disposition,
                "baseline_dominators": sorted(dominators),
                "baselines_strictly_dominated_by_ours": sorted(dominated),
            }
        )
    return {
        "scenario_seed_count": len(rows),
        "ours_nondominated_count": sum(
            row["ours_disposition"] != "baseline_dominates" for row in rows
        ),
        "ours_strict_every_baseline_count": sum(
            row["ours_disposition"] == "ours_strict_every_baseline" for row in rows
        ),
        "baseline_dominates_count": sum(
            row["ours_disposition"] == "baseline_dominates" for row in rows
        ),
        "performance_is_not_protocol_gate": True,
        "rows": rows,
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_cost = tuple(float(left[key]) for key in PARETO_COST_COORDINATES)
    right_cost = tuple(float(right[key]) for key in PARETO_COST_COORDINATES)
    return all(
        a <= b + NUMERIC_TOLERANCE for a, b in zip(left_cost, right_cost)
    ) and any(
        a < b - NUMERIC_TOLERANCE for a, b in zip(left_cost, right_cost)
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(row) for row in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RenewalStreamError("non-finite value in holdout report")
        return round(value, 12)
    return value


__all__ = [
    "PARETO_COST_COORDINATES",
    "compact_class_balance_report",
    "run_class_balance_holdout",
]
