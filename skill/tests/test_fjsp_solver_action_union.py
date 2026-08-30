from __future__ import annotations

from pathlib import Path

from algorithm.experiments.fjsp_instances import load_fjsp_instance
from algorithm.experiments.fjsp_solver_action_union import (
    build_fjsp_solver_action_union,
)


ROOT = Path(__file__).resolve().parents[2]
INSTANCE = ROOT / "tests" / "data" / "fjsp_brandimarte_tiny.fjs"


def _incumbent_solver(_instance, *, incumbent_schedule, **_kwargs):
    return {
        "feasible_solution": True,
        "status": "FEASIBLE",
        "optimality_proved": False,
        "objective": "makespan",
        "wall_time_s": 0.01,
        "verifier": {"pass": True},
        "schedule": incumbent_schedule,
    }


def _infeasible_solver(_instance, **_kwargs):
    return {
        "feasible_solution": False,
        "status": "UNKNOWN",
        "optimality_proved": False,
        "objective": "makespan",
        "wall_time_s": 0.01,
    }


def test_solver_action_union_is_exact_and_deduplicates_same_action():
    report = build_fjsp_solver_action_union(
        load_fjsp_instance(INSTANCE), solver_runner=_incumbent_solver
    )

    assert report["gate"]["pass"] is True
    assert report["selection"]["exact_union_argmax"] is True
    assert report["selection"]["union_oracle_gap"] == 0
    assert report["selection"]["solver_adds_distinct_action"] is False
    assert report["selection"]["effective_union_action_count"] == report[
        "selection"
    ]["base_family_unique_action_count"]
    assert report["solver_candidate"]["global_multiobjective_optimality_claimed"] is False
    assert report["solver_candidate"]["schedule"]
    assert report["solver_candidate"]["verifier"]["pass"] is True
    assert report["base_family"]["selected_schedule"]


def test_infeasible_solver_preserves_base_policy_without_admitting_action():
    report = build_fjsp_solver_action_union(
        load_fjsp_instance(INSTANCE), solver_runner=_infeasible_solver
    )

    assert report["gate"]["pass"] is False
    assert report["gate"]["base_policy_preserved_when_solver_unavailable"] is True
    assert report["solver_candidate"]["feasible"] is False
    assert report["selection"]["exact_union_argmax"] is True
    assert report["selection"]["selected_fingerprint"] == report["base_family"][
        "selected_fingerprint"
    ]
