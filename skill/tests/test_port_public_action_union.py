from __future__ import annotations

from algorithm.experiments.port_public_action_union import (
    DEVELOPMENT_PLAN,
    build_port_public_action_union,
    frozen_action_family,
)
from algorithm.experiments.port_public_benchmark_adapter import (
    DEFAULT_FIXTURE,
    adapt_bacasp_s_to_port_instance,
    parse_bacasp_s_instance,
)
from algorithm.experiments.port_scheduling_benchmark import (
    ALL_POLICIES,
    THEOREM_POLICY,
)
from algorithm.experiments.port_trajectory_upgrade import TrajectoryPlan


def test_frozen_action_family_contains_singletons_and_development_plan():
    plan_ids = {plan.plan_id for plan in frozen_action_family()}

    assert DEVELOPMENT_PLAN.plan_id in plan_ids
    assert TrajectoryPlan((THEOREM_POLICY,)).plan_id in plan_ids
    assert all(TrajectoryPlan((policy,)).plan_id in plan_ids for policy in ALL_POLICIES)


def test_public_action_union_is_feasible_exact_and_nondominated():
    adapted = adapt_bacasp_s_to_port_instance(parse_bacasp_s_instance(DEFAULT_FIXTURE))
    report = build_port_public_action_union(adapted)

    assert report["gate"]["pass"] is True
    assert report["selection"]["exact_family_argmax"] is True
    assert report["selection"]["oracle_gap"] == 0
    assert report["selection"]["selected_pareto_nondominated"] is True
    assert report["selection"]["baseline_singletons_in_family"] is True
    assert report["selected_objective"]["total_bounded_penalty"] == 0
    assert report["claim_boundary"]["unrestricted_bacasp_optimality_claimed"] is False
