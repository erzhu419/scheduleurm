"""Renewal-frame trajectory selector for deterministic FJSP instances.

This v2 module leaves the hash-frozen FJSP artifact untouched.  It reuses the
same fixed seed and prefix-beam candidate generator, then scores every complete
schedule with explicit physical departures and variable frame duration.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from algorithm.experiments.fjsp_benchmark import (
    POLICY_ORDER,
    ReassignmentConfig,
    ScheduledOperation,
    build_schedule,
    verify_schedule,
)
from algorithm.experiments.fjsp_instances import FJSPInstance
from algorithm.experiments.fjsp_trajectory_upgrade import (
    FROZEN_CONFIG,
    _complete_state,
    _completion_times,
    _dominates,
    _frame_upper_bound,
    _prefix_beam,
    _schedule_fingerprint,
    _schedule_from_run,
    _schedule_metrics,
    _schedule_output_order,
    _state_fingerprint,
)
from algorithm.experiments.trajectory_renewal_frame_score import (
    complete_trajectory_renewal_score,
)


POLICY = "scheduleurm_fjsp_renewal_frame"
SCHEMA_VERSION = "scheduleurm.fjsp_renewal_frame_upgrade.v1"


def build_fjsp_renewal_frame_schedule(instance: FJSPInstance) -> dict[str, Any]:
    raw: list[tuple[str, str, tuple[ScheduledOperation, ...]]] = []
    baseline_runs = {}
    for policy in POLICY_ORDER:
        run = build_schedule(instance, policy)
        baseline_runs[policy] = run
        raw.append(
            (f"seed:{policy}", f"fixed_policy:{policy}", _schedule_from_run(run))
        )

    beam = _prefix_beam(instance, FROZEN_CONFIG)
    for beam_index, state in enumerate(beam):
        prefix_hash = _state_fingerprint(state)[:12]
        for policy in FROZEN_CONFIG.rollout_policies:
            completed = _complete_state(instance, state, policy)
            raw.append(
                (
                    f"beam:{beam_index:02d}:{prefix_hash}:rollout:{policy}",
                    f"prefix_beam_depth_{FROZEN_CONFIG.prefix_depth}+{policy}",
                    completed.scheduled,
                )
            )

    candidates = _deduplicate_and_score(instance, raw)
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["score_audit"]["renewal_score"]),
            float(row["metrics"]["makespan"]),
            float(row["metrics"]["mean_flow_time"]),
            str(row["fingerprint"]),
        ),
    )
    if not ranked:
        raise ValueError("FJSP renewal-frame candidate family is empty")
    selected = ranked[0]
    best = max(float(row["score_audit"]["renewal_score"]) for row in ranked)
    gap = best - float(selected["score_audit"]["renewal_score"])
    if abs(gap) > 1e-12:
        raise ValueError("FJSP renewal-frame family argmax is inconsistent")
    dominated_by = [
        policy
        for policy, run in baseline_runs.items()
        if _dominates(
            {
                "makespan": float(run["metrics"]["makespan"]),
                "mean_flow_time": float(run["metrics"]["mean_flow_time"]),
            },
            selected["metrics"],
        )
    ]
    if dominated_by:
        raise ValueError(
            "renewal-frame score selected a baseline-dominated schedule: "
            + ", ".join(dominated_by)
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "schedule": [
            row.snapshot()
            for row in _schedule_output_order(selected["schedule"])
        ],
        "metrics": selected["metrics"],
        "selection": {
            "selected_candidate_id": selected["candidate_id"],
            "selected_origins": selected["origins"],
            "selected_fingerprint": selected["fingerprint"],
            "raw_candidate_count": len(raw),
            "unique_candidate_count": len(ranked),
            "exact_generated_family_argmax": True,
            "generated_family_oracle_gap": round(gap, 12),
            "score_audit": selected["score_audit"],
        },
        "candidate_family": [
            {
                "candidate_id": row["candidate_id"],
                "origins": row["origins"],
                "fingerprint": row["fingerprint"],
                "metrics": row["metrics"],
                "renewal_score": row["score_audit"]["renewal_score"],
                "feasible": True,
            }
            for row in ranked
        ],
        "baseline_policy_count": len(POLICY_ORDER),
        "baseline_union_in_candidate_family": True,
        "performance_superiority_claim_ready": False,
        "theorem_mapping": {
            "candidate_action": "complete nonpreemptive FJSP trajectory",
            "physical_service": "terminal job and system departure indicators",
            "frame_duration": "trajectory makespan tau(a)",
            "holding_cost": "weighted job completion times plus system drain time",
            "score": "(normalized Q^T D(a) - bounded holding penalty(a)) / (tau(a)/H)",
            "holding_reward_is_not_physical_service": True,
            "exact_only_over_generated_family": True,
            "global_fjsp_optimality_claimed": False,
            "stochastic_recurrence_claimed": False,
        },
    }


def _deduplicate_and_score(
    instance: FJSPInstance,
    raw: Sequence[tuple[str, str, tuple[ScheduledOperation, ...]]],
) -> list[dict[str, Any]]:
    by_fingerprint: dict[str, dict[str, Any]] = {}
    horizon = _frame_upper_bound(instance)
    for candidate_id, origin, schedule in raw:
        feasibility = verify_schedule(
            instance, schedule, reassignment=ReassignmentConfig()
        )
        if feasibility.get("pass") is not True:
            raise ValueError(
                f"infeasible FJSP renewal candidate {candidate_id}: "
                f"{feasibility.get('errors')!r}"
            )
        fingerprint = _schedule_fingerprint(schedule)
        existing = by_fingerprint.get(fingerprint)
        if existing is not None:
            existing["origins"].append(origin)
            continue
        metrics = _schedule_metrics(instance, schedule)
        completion = _completion_times(instance, schedule)
        score = complete_trajectory_renewal_score(
            completion_times=completion,
            frame_duration=float(metrics["makespan"]),
            common_horizon_bound=horizon,
            project_queue_weight=(
                FROZEN_CONFIG.system_queue_weight_per_job * instance.job_count
            ),
            holding_penalty_weight=FROZEN_CONFIG.bounded_latency_penalty,
        )
        by_fingerprint[fingerprint] = {
            "candidate_id": candidate_id,
            "origins": [origin],
            "schedule": tuple(schedule),
            "metrics": {
                "makespan": round(float(metrics["makespan"]), 12),
                "mean_flow_time": round(float(metrics["mean_flow_time"]), 12),
            },
            "score_audit": score,
            "fingerprint": fingerprint,
        }
    return list(by_fingerprint.values())


def report_sha256(report: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = ["POLICY", "build_fjsp_renewal_frame_schedule", "report_sha256"]
