"""Finite FJSP action union with a fixed-budget solver trajectory.

The frozen renewal-frame policy already returns the exact maximizer of its
generated heuristic family.  Therefore the exact maximizer of the enlarged
family can be found by comparing that representative with each newly generated
solver trajectory.  This module performs that comparison with the unchanged
renewal-frame score; it does not treat a time-bounded CP-SAT incumbent as a
global optimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from algorithm.experiments.fjsp_instances import FJSPInstance
from algorithm.experiments.fjsp_renewal_frame_upgrade import (
    _deduplicate_and_score,
    build_fjsp_renewal_frame_schedule,
)
from algorithm.experiments.fjsp_trajectory_upgrade import (
    _schedule_from_run,
    _schedule_output_order,
)
from algorithm.experiments.or_exact_reference import solve_fjsp_cp_sat


POLICY = "scheduleurm_fjsp_solver_action_union"
SCHEMA_VERSION = "scheduleurm.fjsp_solver_action_union.v1"


class FJSPSolverActionUnionError(ValueError):
    """Raised when the finite solver-action union cannot be audited."""


@dataclass(frozen=True)
class FrozenSolverActionConfig:
    time_limit_s: float = 5.0
    workers: int = 1
    random_seed: int = 20260830

    def snapshot(self) -> dict[str, Any]:
        return {
            "time_limit_s": self.time_limit_s,
            "workers": self.workers,
            "random_seed": self.random_seed,
            "per_instance_tuning": False,
            "solver_objective": "makespan",
            "solver_runtime_in_physical_schedule": False,
        }


FROZEN_CONFIG = FrozenSolverActionConfig()
SolverRunner = Callable[..., Mapping[str, Any]]


def build_fjsp_solver_action_union(
    instance: FJSPInstance,
    *,
    solver_runner: SolverRunner | None = None,
    config: FrozenSolverActionConfig = FROZEN_CONFIG,
) -> dict[str, Any]:
    """Return the exact renewal-score maximizer of heuristic plus solver actions."""

    if config.time_limit_s <= 0 or config.workers <= 0:
        raise FJSPSolverActionUnionError("solver budget and worker count must be positive")

    base = build_fjsp_renewal_frame_schedule(instance)
    if base["selection"]["exact_generated_family_argmax"] is not True:
        raise FJSPSolverActionUnionError("base family does not provide an exact maximizer")

    runner = solver_runner or solve_fjsp_cp_sat
    solver = dict(
        runner(
            instance,
            time_limit_s=float(config.time_limit_s),
            workers=int(config.workers),
            random_seed=int(config.random_seed),
            incumbent_schedule=base["schedule"],
        )
    )

    raw = [
        (
            "base_family_exact_max",
            "frozen_heuristic_family_exact_max",
            _schedule_from_run(base),
        )
    ]
    solver_feasible = bool(solver.get("feasible_solution"))
    if solver_feasible:
        verifier = solver.get("verifier")
        if not isinstance(verifier, Mapping) or verifier.get("pass") is not True:
            raise FJSPSolverActionUnionError("solver trajectory failed feasibility audit")
        raw.append(
            (
                "fixed_budget_cp_sat",
                "fixed_budget_cp_sat_makespan_trajectory",
                _schedule_from_run(solver),
            )
        )

    representatives = _deduplicate_and_score(instance, raw)
    ranked = sorted(
        representatives,
        key=lambda row: (
            -float(row["score_audit"]["renewal_score"]),
            float(row["metrics"]["makespan"]),
            float(row["metrics"]["mean_flow_time"]),
            str(row["fingerprint"]),
        ),
    )
    if not ranked:
        raise FJSPSolverActionUnionError("solver-action union is empty")
    selected = ranked[0]

    base_score = float(base["selection"]["score_audit"]["renewal_score"])
    materialized_max = max(
        float(row["score_audit"]["renewal_score"]) for row in representatives
    )
    if abs(float(selected["score_audit"]["renewal_score"]) - materialized_max) > 1e-12:
        raise FJSPSolverActionUnionError("selected action is not the union argmax")
    base_representative = next(
        row for row in representatives if "frozen_heuristic_family_exact_max" in row["origins"]
    )
    if abs(float(base_representative["score_audit"]["renewal_score"]) - base_score) > 1e-12:
        raise FJSPSolverActionUnionError("base representative changed score semantics")

    old_fingerprints = {
        str(row["fingerprint"]) for row in base["candidate_family"]
    }
    solver_fingerprint = None
    if solver_feasible:
        solver_row = next(
            row
            for row in representatives
            if "fixed_budget_cp_sat_makespan_trajectory" in row["origins"]
        )
        solver_fingerprint = str(solver_row["fingerprint"])
    solver_adds_distinct_action = bool(
        solver_fingerprint is not None and solver_fingerprint not in old_fingerprints
    )
    effective_union_size = int(base["selection"]["unique_candidate_count"]) + int(
        solver_adds_distinct_action
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "schedule": [
            row.snapshot() for row in _schedule_output_order(selected["schedule"])
        ],
        "metrics": dict(selected["metrics"]),
        "selection": {
            "selected_candidate_id": selected["candidate_id"],
            "selected_origins": list(selected["origins"]),
            "selected_fingerprint": selected["fingerprint"],
            "renewal_score": selected["score_audit"]["renewal_score"],
            "score_audit": selected["score_audit"],
            "base_family_unique_action_count": int(
                base["selection"]["unique_candidate_count"]
            ),
            "solver_adds_distinct_action": solver_adds_distinct_action,
            "effective_union_action_count": effective_union_size,
            "exact_union_argmax": True,
            "union_oracle_gap": 0.0,
            "exactness_reduction": (
                "max(base-family exact maximizer, fixed solver trajectories)"
            ),
        },
        "materialized_union_representatives": [
            {
                "candidate_id": row["candidate_id"],
                "origins": list(row["origins"]),
                "fingerprint": row["fingerprint"],
                "metrics": dict(row["metrics"]),
                "renewal_score": row["score_audit"]["renewal_score"],
            }
            for row in ranked
        ],
        "solver_candidate": {
            "feasible": solver_feasible,
            "status": str(solver.get("status", "UNKNOWN")),
            "optimality_proved_for_makespan": bool(solver.get("optimality_proved")),
            "fingerprint": solver_fingerprint,
            "adds_distinct_action": solver_adds_distinct_action,
            "objective": str(solver.get("objective", "makespan")),
            "wall_time_s": solver.get("wall_time_s"),
            "schedule": list(solver.get("schedule") or ()) if solver_feasible else [],
            "verifier": solver.get("verifier") if solver_feasible else None,
            "runtime_reported_separately": True,
            "global_multiobjective_optimality_claimed": False,
        },
        "base_family": {
            "policy": base["policy"],
            "exact_generated_family_argmax": True,
            "selected_fingerprint": base["selection"]["selected_fingerprint"],
            "selected_renewal_score": base["selection"]["score_audit"][
                "renewal_score"
            ],
            "selected_metrics": dict(base["metrics"]),
            "selected_schedule": list(base["schedule"]),
        },
        "config": config.snapshot(),
        "gate": {
            "pass": solver_feasible,
            "status": (
                "FJSP_SOLVER_ACTION_UNION_PASS"
                if solver_feasible
                else "FJSP_SOLVER_ACTION_UNION_NO_SOLVER_ACTION"
            ),
            "base_policy_preserved_when_solver_unavailable": True,
            "solver_action_admitted_only_after_feasibility_audit": True,
        },
        "claim_boundary": {
            "scope": "frozen heuristic family union fixed-budget solver trajectory",
            "exact_only_over_finite_registered_union": True,
            "solver_feasible_is_not_global_optimality": True,
            "solver_runtime_excluded_from_processing_times_but_reported": True,
            "arbitrary_fjsp_superiority_claimed": False,
            "stochastic_recurrence_claimed": False,
        },
    }


__all__ = [
    "FROZEN_CONFIG",
    "FJSPSolverActionUnionError",
    "FrozenSolverActionConfig",
    "POLICY",
    "SCHEMA_VERSION",
    "build_fjsp_solver_action_union",
]
