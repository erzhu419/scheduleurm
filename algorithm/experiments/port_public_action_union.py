"""Exact finite trajectory-action union for adapted BACASP-S instances.

The family is fixed independently of the evaluated instance outcomes.  It
contains every registered single policy, the globally frozen rolling seeds,
and the previously selected development plan.  Each complete trajectory is
scored on the same candidate-independent terminal-cost frame.  The outer
penalty is fixed to zero; action-level robust MaxWeight penalties and oracle
audits remain part of every simulated trajectory.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from algorithm.experiments.port_public_benchmark_adapter import (
    AdaptedPublicPortInstance,
)
from algorithm.experiments.port_scheduling_benchmark import (
    ALL_POLICIES,
    BASELINE_POLICIES,
    RECONFIGURATION_ACTIONS,
    THEOREM_POLICY,
)
from algorithm.experiments.port_trajectory_upgrade import (
    GLOBAL_TRAJECTORY_CONFIG,
    SOURCE_COST_METRICS,
    TRAJECTORY_POLICY,
    TrajectoryPlan,
    _pareto,
    _run_public_plan,
    _trajectory_frame_spec,
    initial_trajectory_family,
    trajectory_objective,
)


POLICY = "trajectory_robust_action_union"
SCHEMA_VERSION = "scheduleurm.port_public_action_union.v1"
DEVELOPMENT_PLAN = TrajectoryPlan(("spt_static", THEOREM_POLICY, THEOREM_POLICY))
OUTER_SCORE_CONFIG = replace(GLOBAL_TRAJECTORY_CONFIG, terminal_penalty_cap=0.0)


class PortPublicActionUnionError(ValueError):
    """Raised when the finite public-port action union cannot be audited."""


def frozen_action_family(
    extra_plans: Sequence[TrajectoryPlan] = (DEVELOPMENT_PLAN,),
) -> tuple[TrajectoryPlan, ...]:
    plans = {plan.plan_id: plan for plan in initial_trajectory_family()}
    for plan in extra_plans:
        plans[plan.plan_id] = plan
    return tuple(plans[key] for key in sorted(plans))


def build_port_public_action_union(
    adapted: AdaptedPublicPortInstance,
    *,
    action_family: Sequence[TrajectoryPlan] | None = None,
) -> dict[str, Any]:
    """Evaluate and exactly select the best registered complete trajectory."""

    family = tuple(action_family or frozen_action_family())
    if not family or len({plan.plan_id for plan in family}) != len(family):
        raise PortPublicActionUnionError("action family must be finite and duplicate-free")
    required_singletons = {TrajectoryPlan((policy,)).plan_id for policy in ALL_POLICIES}
    present = {plan.plan_id for plan in family}
    if not required_singletons.issubset(present):
        raise PortPublicActionUnionError("action family omits a registered single policy")

    frame = _trajectory_frame_spec(adapted.port_instance, adapted)
    rows: list[dict[str, Any]] = []
    for plan in family:
        result = _run_public_plan(adapted, plan)
        source_audit = result.get("public_source_feasibility_audit") or {}
        no_mid_service_reconfiguration = all(
            int(result["action_counts"].get(action_type, 0)) == 0
            for action_type in RECONFIGURATION_ACTIONS
        )
        feasible = bool(
            result.get("pass") is True
            and source_audit.get("resource_feasibility_ready") is True
            and no_mid_service_reconfiguration
        )
        if not feasible:
            raise PortPublicActionUnionError(
                f"registered trajectory is infeasible: {plan.plan_id}"
            )
        objective = trajectory_objective(result, frame, OUTER_SCORE_CONFIG)
        if float(objective["total_bounded_penalty"]) != 0.0:
            raise PortPublicActionUnionError("outer zero-penalty contract was violated")
        rows.append(
            {
                "plan": plan,
                "result": result,
                "objective": objective,
                "metrics": dict(result["public_source_core_metrics"]),
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["objective"]["robust_terminal_objective"]),
            *(float(row["metrics"][name]) for name in SOURCE_COST_METRICS),
            row["plan"].plan_id,
        ),
    )
    selected = ranked[0]
    best_score = max(
        float(row["objective"]["robust_terminal_objective"]) for row in rows
    )
    gap = best_score - float(selected["objective"]["robust_terminal_objective"])
    if abs(gap) > 1e-12:
        raise PortPublicActionUnionError("selected trajectory is not the exact family argmax")

    metrics = {row["plan"].plan_id: row["metrics"] for row in rows}
    pareto = _pareto(metrics, SOURCE_COST_METRICS)
    if selected["plan"].plan_id not in pareto["nondominated_policies"]:
        raise PortPublicActionUnionError(
            "strictly monotone terminal-service score selected a dominated trajectory"
        )

    selected_result = dict(selected["result"])
    selected_result["policy"] = POLICY
    selected_result["action_union_selection"] = {
        "selected_plan": selected["plan"].snapshot(),
        "selected_score_audit": selected["objective"],
        "candidate_count": len(rows),
        "exact_family_argmax": True,
        "oracle_gap": 0.0,
    }
    baseline_plan_ids = {
        policy: TrajectoryPlan((policy,)).plan_id for policy in BASELINE_POLICIES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "selected_result": selected_result,
        "selected_plan": selected["plan"].snapshot(),
        "selected_source_core_metrics": selected["metrics"],
        "selected_objective": selected["objective"],
        "selection": {
            "candidate_count": len(rows),
            "exact_family_argmax": True,
            "oracle_gap": 0.0,
            "selected_plan_id": selected["plan"].plan_id,
            "selected_pareto_nondominated": True,
            "baseline_singletons_in_family": all(
                plan_id in present for plan_id in baseline_plan_ids.values()
            ),
            "theorem_singleton_in_family": TrajectoryPlan(
                (THEOREM_POLICY,)
            ).plan_id
            in present,
            "development_plan_in_family": DEVELOPMENT_PLAN.plan_id in present,
        },
        "candidate_family": [
            {
                "plan": row["plan"].snapshot(),
                "source_core_metrics": row["metrics"],
                "objective": row["objective"],
                "selected": row is selected,
            }
            for row in sorted(rows, key=lambda item: item["plan"].plan_id)
        ],
        "pareto": pareto,
        "gate": {
            "pass": True,
            "status": "PORT_PUBLIC_ACTION_UNION_PASS",
            "all_candidates_feasible": True,
            "selected_pareto_nondominated": True,
        },
        "theorem_mapping": {
            "candidate_action": "complete registered port trajectory",
            "outer_score": (
                "uniform terminal virtual service with candidate-independent bounds"
            ),
            "outer_penalty_bound": 0.0,
            "low_level_score": "statewise Q^T lower_service minus bounded penalty",
            "exact_argmax_scope": "fixed finite registered trajectory family",
        },
        "claim_boundary": {
            "same_deterministic_model_used_for_selection_and_evaluation": True,
            "unrestricted_bacasp_optimality_claimed": False,
            "author_algorithm_superiority_claimed": False,
            "physical_port_deployment_claimed": False,
            "stochastic_recurrence_claimed": False,
        },
    }


__all__ = [
    "DEVELOPMENT_PLAN",
    "OUTER_SCORE_CONFIG",
    "POLICY",
    "PortPublicActionUnionError",
    "build_port_public_action_union",
    "frozen_action_family",
]
