"""Renewal-frame score shared by deterministic OR portability experiments.

Complete FJSP and MMRCPSP schedules drain every registered job.  Their physical
departure vectors are therefore terminal completion indicators, whereas
``(H - C_j) / H`` is a cumulative holding-cost reward.  This module keeps those
objects separate and evaluates a complete trajectory with the renewal ratio

    (normalized Q^T D(a) - bounded holding penalty(a)) / (tau(a) / H).

The score is a deterministic finite-action selection rule.  A stochastic
recurrence claim still requires a domain arrival model and the variable-frame
drift hypotheses proved separately in the Lean artifact.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class RenewalFrameScoreError(ValueError):
    """Raised when a trajectory cannot satisfy the renewal-frame contract."""


def complete_trajectory_renewal_score(
    *,
    completion_times: Sequence[float],
    frame_duration: float,
    common_horizon_bound: float,
    job_queue_weights: Sequence[float] | None = None,
    project_queue_weight: float = 1.0,
    holding_penalty_weight: float = 0.05,
    extra_penalty: float = 0.0,
    extra_penalty_bound: float = 0.0,
) -> dict[str, Any]:
    """Score one complete trajectory and return an auditable decomposition."""

    completion = tuple(float(value) for value in completion_times)
    tau = float(frame_duration)
    horizon = float(common_horizon_bound)
    weights = (
        tuple(float(value) for value in job_queue_weights)
        if job_queue_weights is not None
        else tuple(1.0 for _ in completion)
    )
    project_weight = float(project_queue_weight)
    gamma = float(holding_penalty_weight)
    extra = float(extra_penalty)
    extra_bound = float(extra_penalty_bound)
    finite = (
        completion
        + weights
        + (tau, horizon, project_weight, gamma, extra, extra_bound)
    )
    if any(not math.isfinite(value) for value in finite):
        raise RenewalFrameScoreError("renewal-frame inputs must be finite")
    if len(weights) != len(completion):
        raise RenewalFrameScoreError("one queue weight is required per completion")
    if not completion:
        raise RenewalFrameScoreError("at least one physical job departure is required")
    if tau <= 0.0 or horizon <= 0.0 or tau > horizon + 1e-9:
        raise RenewalFrameScoreError("frame duration must lie in (0, H]")
    if any(value <= 0.0 or value > tau + 1e-9 for value in completion):
        raise RenewalFrameScoreError("job completions must lie in (0, tau]")
    if any(value <= 0.0 for value in weights) or project_weight <= 0.0:
        raise RenewalFrameScoreError("queue weights must be positive")
    if gamma < 0.0 or gamma >= 1.0:
        raise RenewalFrameScoreError("holding-penalty weight must lie in [0, 1)")
    if extra < -1e-12 or extra_bound < 0.0 or extra > extra_bound + 1e-12:
        raise RenewalFrameScoreError("extra penalty violates its declared bound")
    total_penalty_bound = gamma + extra_bound
    if total_penalty_bound >= 1.0:
        raise RenewalFrameScoreError("total penalty bound must be below one")

    total_queue_weight = sum(weights) + project_weight
    weighted_physical_departures = sum(weights) + project_weight
    normalized_departure_reward = weighted_physical_departures / total_queue_weight
    weighted_completion = sum(
        weight * value for weight, value in zip(weights, completion, strict=True)
    ) + project_weight * tau
    normalized_holding_cost = weighted_completion / (
        total_queue_weight * horizon
    )
    holding_penalty = gamma * normalized_holding_cost
    bounded_penalty = holding_penalty + extra
    normalized_frame_duration = tau / horizon
    renewal_numerator = normalized_departure_reward - bounded_penalty
    renewal_score = renewal_numerator / normalized_frame_duration
    if renewal_numerator <= 0.0:
        raise RenewalFrameScoreError("renewal numerator must remain positive")

    holding_reward = tuple((horizon - value) / horizon for value in completion)
    return _canonical(
        {
            "schema_version": "scheduleurm.trajectory_renewal_frame_score.v1",
            "score_semantics": (
                "normalized_Q_dot_physical_departures_minus_bounded_holding_"
                "penalty_over_normalized_frame_duration"
            ),
            "renewal_score": renewal_score,
            "renewal_numerator": renewal_numerator,
            "frame": {
                "duration_tau": tau,
                "common_horizon_bound_H": horizon,
                "normalized_duration_tau_over_H": normalized_frame_duration,
                "variable_duration_action": True,
            },
            "physical_departures": {
                "job_departure_indicators": [1 for _ in completion],
                "project_departure_indicator": 1,
                "job_queue_weights": list(weights),
                "project_queue_weight": project_weight,
                "weighted_total": weighted_physical_departures,
                "normalized_Q_dot_D": normalized_departure_reward,
                "all_registered_jobs_complete": True,
            },
            "holding_cost": {
                "job_completion_times": list(completion),
                "project_completion_time": tau,
                "normalized_weighted_cost": normalized_holding_cost,
                "holding_reward_coordinates": list(holding_reward),
                "holding_reward_is_not_physical_service": True,
            },
            "penalty": {
                "holding_penalty_weight": gamma,
                "holding_penalty": holding_penalty,
                "extra_penalty": extra,
                "extra_penalty_bound": extra_bound,
                "bounded_penalty": bounded_penalty,
                "uniform_bound": total_penalty_bound,
                "queue_scaled_beta": 0.0,
            },
            "theorem_boundary": {
                "finite_family_renewal_ratio": True,
                "pareto_monotone_when_extra_penalty_is_constant": True,
                "stochastic_recurrence_claimed": False,
                "requires_domain_arrival_model_for_recurrence": True,
            },
        }
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(row) for row in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


__all__ = [
    "RenewalFrameScoreError",
    "complete_trajectory_renewal_score",
]
