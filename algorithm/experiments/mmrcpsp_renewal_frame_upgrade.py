"""Renewal-frame trajectory selector for deterministic MMRCPSP instances.

The hash-frozen v1 experiment used ``H - C_j`` as a cumulative service
coordinate.  That quantity is a holding reward, not a physical departure.
This sidecar reuses the same fixed rollout, beam, and local candidate generator
while separating terminal departures, holding cost, bounded penalty, and the
variable trajectory duration.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from algorithm.experiments.mmrcpsp_benchmark import (
    ScheduleEntry,
    _evaluate_schedule,
    schedule_instance,
)
from algorithm.experiments.mmrcpsp_instances import MMRCPSPInstance
from algorithm.experiments.mmrcpsp_trajectory_upgrade import (
    BASELINE_POLICIES,
    BEAM_SEEDS,
    LOCAL_SEED_LIMIT,
    ROLLOUT_POLICIES,
    _Candidate,
    _analyze,
    _beam_trajectories,
    _deduplicate,
    _local_neighbors,
)
from algorithm.experiments.trajectory_renewal_frame_score import (
    complete_trajectory_renewal_score,
)


POLICY = "scheduleurm_mmrcpsp_renewal_frame"
SCHEMA_VERSION = "scheduleurm.mmrcpsp_renewal_frame_upgrade.v1"
HOLDING_PENALTY_WEIGHT = 0.05


def build_mmrcpsp_renewal_frame_schedule(
    instance: MMRCPSPInstance,
) -> dict[str, Any]:
    """Select an exact renewal-score maximizer over the frozen generated family."""

    analysis = _analyze(instance)
    raw: list[_Candidate] = []
    baseline_runs: dict[str, Mapping[str, Any]] = {}
    construction_counts = {
        "rollout_seed_count": 0,
        "beam_seed_count": 0,
        "beam_state_expansion_count": 0,
        "local_neighbor_attempt_count": 0,
        "local_neighbor_feasible_count": 0,
    }
    for policy in ROLLOUT_POLICIES:
        result = schedule_instance(instance, policy)
        if not result.feasible:
            raise ValueError(
                f"{instance.name}: frozen rollout {policy} is infeasible: "
                f"{result.failure_reason}"
            )
        candidate = _Candidate(
            candidate_id=f"rollout:{policy}",
            origin="rollout",
            entries=result.entries,
        )
        raw.append(candidate)
        construction_counts["rollout_seed_count"] += 1
        if policy in BASELINE_POLICIES:
            baseline_runs[policy] = result.snapshot()

    for seed_name, weights in BEAM_SEEDS:
        beam_candidates, expansion_count = _beam_trajectories(
            instance,
            analysis=analysis,
            seed_name=seed_name,
            weights=weights,
        )
        raw.extend(beam_candidates)
        construction_counts["beam_seed_count"] += 1
        construction_counts["beam_state_expansion_count"] += expansion_count

    local_seeds = _deduplicate(raw)[:LOCAL_SEED_LIMIT]
    local_candidates, attempted = _local_neighbors(instance, local_seeds)
    raw.extend(local_candidates)
    construction_counts["local_neighbor_attempt_count"] = attempted
    construction_counts["local_neighbor_feasible_count"] = len(local_candidates)

    candidates = _deduplicate(raw)
    if not candidates:
        raise ValueError(f"{instance.name}: empty renewal-frame candidate family")
    evaluated = [_evaluate_candidate(instance, candidate) for candidate in candidates]
    infeasible = [row for row in evaluated if row["feasible"] is not True]
    if infeasible:
        raise ValueError(
            f"{instance.name}: infeasible renewal-frame candidate "
            f"{infeasible[0]['candidate_id']}"
        )
    ranked = sorted(
        evaluated,
        key=lambda row: (
            -float(row["score_audit"]["renewal_score"]),
            float(row["metrics"]["makespan"]),
            float(row["metrics"]["mean_flow_time"]),
            str(row["candidate_id"]),
        ),
    )
    selected = ranked[0]
    maximum = max(float(row["score_audit"]["renewal_score"]) for row in ranked)
    oracle_gap = maximum - float(selected["score_audit"]["renewal_score"])
    if abs(oracle_gap) > 1e-12:
        raise ValueError(f"{instance.name}: nonzero generated-family oracle gap")

    dominated_by = [
        policy
        for policy, run in baseline_runs.items()
        if _dominates(run["metrics"], selected["metrics"])
    ]
    if dominated_by:
        raise ValueError(
            f"{instance.name}: renewal score selected a baseline-dominated schedule: "
            + ", ".join(dominated_by)
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "instance": instance.snapshot(),
        "schedule": selected["schedule"],
        "metrics": selected["metrics"],
        "selection": {
            "selected_candidate_id": selected["candidate_id"],
            "selected_origin": selected["origin"],
            "selected_trajectory_sha256": selected["trajectory_sha256"],
            "raw_candidate_count": len(raw),
            "unique_candidate_count": len(ranked),
            "exact_generated_family_argmax": True,
            "generated_family_oracle_gap": round(oracle_gap, 12),
            "score_audit": selected["score_audit"],
            "configuration_diagnostics": selected["configuration_diagnostics"],
        },
        "candidate_family": [
            {
                "candidate_id": row["candidate_id"],
                "origin": row["origin"],
                "trajectory_sha256": row["trajectory_sha256"],
                "metrics": row["metrics"],
                "renewal_score": row["score_audit"]["renewal_score"],
                "feasible": True,
            }
            for row in ranked
        ],
        "construction_counts": construction_counts,
        "baseline_policy_count": len(BASELINE_POLICIES),
        "baseline_union_in_candidate_family": True,
        "performance_superiority_claim_ready": False,
        "theorem_mapping": {
            "candidate_action": "complete feasible MMRCPSP activity-mode trajectory",
            "physical_service": "terminal real-activity and project departure indicators",
            "frame_duration": "trajectory makespan tau(a)",
            "holding_cost": "weighted activity completion times plus project drain time",
            "score": (
                "(normalized Q^T D(a) - bounded holding penalty(a)) / "
                "(tau(a)/H)"
            ),
            "mode_and_resource_configuration_are_diagnostics_not_service": True,
            "holding_reward_is_not_physical_service": True,
            "exact_only_over_generated_family": True,
            "global_mmrcpsp_optimality_claimed": False,
            "stochastic_recurrence_claimed": False,
        },
    }


def _evaluate_candidate(
    instance: MMRCPSPInstance,
    candidate: _Candidate,
) -> dict[str, Any]:
    metrics = _evaluate_schedule(instance, candidate.entries)
    if metrics["schedule_feasible"] is not True:
        return {
            "candidate_id": candidate.candidate_id,
            "origin": candidate.origin,
            "feasible": False,
            "metrics": metrics,
        }
    entry_map = {entry.job_id: entry for entry in candidate.entries}
    completion = tuple(float(entry_map[job_id].finish) for job_id in instance.real_job_ids)
    makespan = float(metrics["makespan"])
    horizon = float(
        max(
            1,
            sum(
                max(mode.duration for mode in instance.activity_map[job_id].modes)
                for job_id in instance.job_ids
            ),
        )
    )
    score = complete_trajectory_renewal_score(
        completion_times=completion,
        frame_duration=makespan,
        common_horizon_bound=horizon,
        project_queue_weight=float(len(completion)),
        holding_penalty_weight=HOLDING_PENALTY_WEIGHT,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "origin": candidate.origin,
        "trajectory_sha256": _trajectory_sha256(candidate.entries),
        "feasible": True,
        "metrics": metrics,
        "score_audit": score,
        "configuration_diagnostics": {
            "mode_reconfiguration_count": metrics["mode_reconfiguration_count"],
            "renewable_configuration_l1": metrics["renewable_configuration_l1"],
            "excluded_from_physical_service": True,
        },
        "schedule": [entry.snapshot() for entry in candidate.entries],
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    objectives = ("makespan", "mean_flow_time")
    return all(float(left[key]) <= float(right[key]) + 1e-12 for key in objectives) and any(
        float(left[key]) < float(right[key]) - 1e-12 for key in objectives
    )


def _trajectory_sha256(entries: Sequence[ScheduleEntry]) -> str:
    payload = [
        (entry.job_id, entry.mode_id, entry.start, entry.finish) for entry in entries
    ]
    return sha256(
        json.dumps(payload, separators=(",", ":")).encode("ascii")
    ).hexdigest()


__all__ = [
    "HOLDING_PENALTY_WEIGHT",
    "POLICY",
    "build_mmrcpsp_renewal_frame_schedule",
]
