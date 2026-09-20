"""A complete-job renewal model for the registered jtl110gpu2 GPU workload.

Every lane option finishes all admitted canonical jobs.  Loaded options include
fresh-resident initialization, the measured directional overlap, and any
dedicated resident drain and finalization.  Selection, departures, and capacity
therefore use the same ``requirements / max(1, duration)`` service vectors.  Durations
are deterministic within this cached phase model; this does not create a new
statistical guarantee for physical execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

from algorithm.experiments import unified_hardware_or_replay as replay


DEFAULT_CACHE = replay.DEFAULT_CACHE
DEFAULT_LOADED_LEDGER = replay.DEFAULT_LOADED_LEDGER
WORKLOADS = (
    "gpu_cnn_torch_resnet50",
    "gpu_heavy_jax_matmul",
    "gpu_llm_distilgpt2",
)


@dataclass(frozen=True)
class FrameOption:
    action_id: str
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]
    loaded: bool = False
    source_action_ids: tuple[str, ...] = ()
    phases: tuple[tuple[str, float], ...] = ()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameModel:
    node: str
    scenario_id: str
    lanes: int
    workloads: tuple[str, ...]
    scales: tuple[float, ...]
    options: tuple[FrameOption, ...]
    excluded_actions: tuple[dict[str, str], ...] = ()
    model_scope: str = "deterministic_cached_phase_complete_job_renewal"

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def load_model(
    cache_path: Path = DEFAULT_CACHE,
    node: str = "jtl110gpu2",
    *,
    loaded_ledger_path: Path = DEFAULT_LOADED_LEDGER,
) -> FrameModel:
    """Read the small frozen cache and build the nine supported lane options."""
    if node != "jtl110gpu2":
        raise ValueError("the aligned experiment registers only jtl110gpu2")
    _, payload = replay._read_json(Path(cache_path), "aligned hardware cache")
    _, rows, exact = replay._validate_cache(payload)
    _, ledger = replay._read_json(Path(loaded_ledger_path), "loaded action ledger")
    loaded = replay._validate_loaded_ledger(ledger, exact=exact)
    node_bucket = "gpu_3080ti_12gb_dual:jtl110gpu2"
    dedicated = tuple(
        row for row in rows
        if row.node_bucket == node_bucket
        and row.workload_key in WORKLOADS
        and row.completion_model_ready
        and not row.capacity_boundary
        and row.resource_state == "empty"
        and not row.resident_mix
    )
    evaluations = {
        replay._record_action_id(row): replay._evaluation_from_record(
            replay._record_action_id(row), row, kind="aligned_dedicated"
        )
        for row in dedicated
    }
    scales = tuple(
        min(
            evaluation.replay_job_units
            for row in dedicated if row.workload_key == workload
            for evaluation in (evaluations[replay._record_action_id(row)],)
        )
        for workload in WORKLOADS
    )
    options: list[FrameOption] = []
    for row in dedicated:
        action_id = replay._record_action_id(row)
        evaluation = evaluations[action_id]
        coordinate = WORKLOADS.index(row.workload_key)
        width = int(row.colocation_count)
        duration = (
            evaluation.fixed_overhead_s
            + width * scales[coordinate] / evaluation.phase_service_rate
        )
        options.append(
            FrameOption(
                action_id=action_id,
                requirements=tuple(width if i == coordinate else 0 for i in range(3)),
                duration=duration,
                completion_offsets=tuple(
                    (duration,) * width if i == coordinate else () for i in range(3)
                ),
                source_action_ids=(action_id,),
                phases=(
                    ("initialization_s", float(row.startup_overhead_s)),
                    ("loop_s", width * scales[coordinate] / evaluation.phase_service_rate),
                    ("finalization_s", float(row.finalization_overhead_s)),
                ),
            )
        )
    excluded: list[dict[str, str]] = []
    for action in loaded:
        if action.lower.node_bucket != node_bucket:
            continue
        if action.lower.workload_key not in WORKLOADS:
            continue
        required = action.lower.job_requirements()
        if not set(required).issubset(WORKLOADS):
            excluded.append({
                "action_id": action.lower.action_id,
                "reason": "resident class has no arrivals in the three-class q01 protocol",
            })
            continue
        target = WORKLOADS.index(action.lower.workload_key)
        resident_key = next(key for key in required if key != action.lower.workload_key)
        resident = WORKLOADS.index(resident_key)
        resident_record, = (
            row for row in dedicated
            if row.workload_key == resident_key and row.colocation_count == 1
        )
        resident_id = replay._record_action_id(resident_record)
        resident_evaluation = evaluations[resident_id]
        initialization = float(resident_record.startup_overhead_s)
        target_frame = (
            action.evaluation.fixed_overhead_s
            + scales[target] / action.evaluation.phase_service_rate
        )
        resident_rate = action.lower.service_vector()[resident_key]
        residual = max(0.0, scales[resident] - resident_rate * target_frame)
        drain = residual / resident_evaluation.phase_service_rate
        finalization = float(resident_record.finalization_overhead_s)
        target_finish = initialization + target_frame
        duration = target_finish + drain + finalization
        options.append(
            FrameOption(
                action_id=action.lower.action_id,
                requirements=tuple(required.get(key, 0) for key in WORKLOADS),
                duration=duration,
                completion_offsets=tuple(
                    (target_finish,) if i == target else
                    (duration,) if i == resident else () for i in range(3)
                ),
                loaded=True,
                source_action_ids=(action.lower.action_id, resident_id),
                phases=(
                    ("resident_initialization_s", initialization),
                    ("target_overlap_s", target_frame),
                    ("resident_overlap_rate_native_per_s", resident_rate),
                    ("resident_remaining_native_units", residual),
                    ("resident_dedicated_drain_s", drain),
                    ("resident_finalization_s", finalization),
                ),
            )
        )
    return FrameModel(
        node=node,
        scenario_id="q01:gpu_3080ti_12gb_dual_jtl110gpu2",
        lanes=2,
        workloads=WORKLOADS,
        scales=scales,
        options=tuple(sorted(options, key=lambda option: option.action_id)),
        excluded_actions=tuple(excluded),
    )


def _feasible_time_weights(weights: np.ndarray) -> np.ndarray:
    """Turn a numerical LP witness into nonnegative time shares totaling <= 1."""
    feasible = np.maximum(weights, 0.0)
    total = float(feasible.sum())
    return feasible / total if total > 1.0 else feasible


def capacity_certificate(model: FrameModel, load: float) -> dict[str, Any]:
    """Capacity and uniform drift constants for the same complete-job catalog.

    The global convex region is R times the one-lane convex region: a mixed
    synchronous frame contributes sum_l tau_l/max(tau_l) times its lane rates,
    using the selector's one-second duration floor;
    replicating one option across all lanes attains every R-scaled vertex.

    Missing jobs are handled uniformly over all integer queues.  Let M_i be
    the largest global requirement and J={i:Q_i>=M_i}.  The mask LPs verify
    that projecting any full rate onto J lies in the convex service region of
    actions supported only on J.  These actions are feasible, and the omitted
    coordinates contribute at most C=sum_i (M_i-1) R max_a r_i(a).  Positive
    numerical projection shortfalls are retained as an additional linear loss.
    """
    if not np.isfinite(load) or load <= 0.0:
        raise ValueError("load must be positive and finite")
    requirements = np.asarray([option.requirements for option in model.options], dtype=float)
    durations = np.maximum(
        1.0, np.asarray([option.duration for option in model.options], dtype=float)
    )
    rates = requirements / durations[:, None]
    count, dimension = rates.shape
    lanes = model.lanes
    max_departures = lanes * requirements.max(axis=0)
    maximum_rates = lanes * rates.max(axis=0)
    projection_failures: list[dict[str, Any]] = []
    projection_checks = 0
    projection_shortfall = 0.0
    for size in range(1, dimension + 1):
        for subset in combinations(range(dimension), size):
            outside = tuple(i for i in range(dimension) if i not in subset)
            supported = np.asarray([
                index for index, need in enumerate(requirements)
                if all(need[i] == 0 for i in outside)
            ], dtype=int)
            restricted = rates[supported][:, subset]
            for option_index, rate in enumerate(rates):
                projection_checks += 1
                if len(supported) == 0:
                    successful = bool(np.all(rate[list(subset)] == 0))
                else:
                    fit = linprog(
                        np.zeros(len(supported)),
                        A_ub=np.vstack((-restricted.T, np.ones(len(supported)))),
                        b_ub=np.concatenate((-rate[list(subset)], [1.0])),
                        bounds=(0.0, None), method="highs",
                    )
                    successful = bool(fit.success)
                    if successful:
                        actual_projection = _feasible_time_weights(fit.x) @ restricted
                        shortfall = float(np.maximum(
                            rate[list(subset)] - actual_projection, 0.0
                        ).max())
                        projection_shortfall = max(projection_shortfall, shortfall)
                if not successful:
                    projection_failures.append({
                        "workloads": [model.workloads[i] for i in subset],
                        "action_id": model.options[option_index].action_id,
                    })

    best_dedicated = np.zeros(dimension)
    for index, option in enumerate(model.options):
        if not option.loaded and np.count_nonzero(requirements[index]) == 1:
            best_dedicated = np.maximum(best_dedicated, rates[index])
    if np.any(best_dedicated <= 0):
        raise ValueError("every arrival class needs a dedicated capacity direction")
    direction = lanes * best_dedicated / dimension
    theta = linprog(
        np.concatenate((np.zeros(count), [-1.0])),
        A_ub=np.vstack((
            np.column_stack((-lanes * rates.T, direction)),
            np.concatenate((np.ones(count), [0.0])),
        )),
        b_ub=np.concatenate((np.zeros(dimension), [1.0])),
        bounds=(0.0, None), method="highs",
    )
    if not theta.success:
        raise ValueError(f"capacity boundary LP failed: {theta.message}")
    boundary_weights = _feasible_time_weights(theta.x[:-1])
    boundary_scale = float(np.min(lanes * (boundary_weights @ rates) / direction))
    arrivals = float(load) * boundary_scale * direction
    slack = linprog(
        np.concatenate((np.zeros(count), [-1.0])),
        A_ub=np.vstack((
            np.column_stack((-lanes * rates.T, np.ones(dimension))),
            np.concatenate((np.ones(count), [0.0])),
        )),
        b_ub=np.concatenate((-arrivals, [1.0])),
        bounds=[(0.0, None)] * count + [(None, None)], method="highs",
    )
    if not slack.success:
        raise ValueError(f"capacity slack LP failed: {slack.message}")
    slack_weights = _feasible_time_weights(slack.x[:-1])
    delta = float(np.min(lanes * (slack_weights @ rates) - arrivals))
    projection_loss = lanes * projection_shortfall
    effective_slack = delta - projection_loss
    tau_min = 1.0
    tau_max = float(durations.max())
    availability_loss = float(np.sum((max_departures - 1.0) * maximum_rates))
    poisson_means = arrivals * tau_max
    second_moment = float(0.5 * np.sum(
        poisson_means ** 2 + poisson_means + max_departures ** 2
    ))
    drift_constant = second_moment + tau_max * availability_loss
    projection_passed = not projection_failures
    certified = projection_passed and effective_slack > 0.0
    return {
        "model_scope": model.model_scope,
        "arrival_model": "independent Poisson job arrivals in physical time",
        "arrival_rates": tuple(float(value) for value in arrivals),
        "arrival_direction": tuple(float(value) for value in direction),
        "load_factor": float(load),
        "full_boundary_scale": boundary_scale,
        "boundary_time_weights": tuple(float(value) for value in boundary_weights),
        "slack_time_weights": tuple(float(value) for value in slack_weights),
        "support_slack_delta": delta,
        "projection_support_loss_rate": projection_loss,
        "effective_slack_eta": effective_slack,
        "capacity_projection_passed": projection_passed,
        "projection_check_count": projection_checks,
        "projection_failures": projection_failures,
        "availability_loss_C": availability_loss,
        "second_moment_B": second_moment,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "max_departures": tuple(int(value) for value in max_departures),
        "drift_constant": drift_constant,
        "negative_drift_backlog_threshold": (
            drift_constant / (tau_min * effective_slack) if certified else None
        ),
        "conditional_model_drift_certified": certified,
        "physical_service_confidence_increased": False,
        "queue_update": "Q_next = Q - physical_job_departures + Poisson(lambda * tau)",
        "drift_bound": "E[Delta V | Q] <= B + tau_max*C - tau_min*effective_slack*sum(Q)",
        "capacity_basis": "lanes times convex hull of all admitted lane requirements/max(1,duration)",
        "capacity_global_combinations_enumerated": 0,
    }
