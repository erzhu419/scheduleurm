from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments.fjsp_instances import load_fjsp_instance
from algorithm.experiments.fjsp_renewal_frame_upgrade import (
    build_fjsp_renewal_frame_schedule,
)
from algorithm.experiments.mmrcpsp_instances import parse_psplib_mm
from algorithm.experiments.mmrcpsp_renewal_frame_upgrade import (
    build_mmrcpsp_renewal_frame_schedule,
)
from algorithm.experiments.or_exact_reference import (
    ExactReferenceUnavailable,
    solve_fjsp_cp_sat,
    solve_mmrcpsp_cp_sat,
)


ROOT = Path(__file__).resolve().parents[2]


def test_fjsp_cp_sat_reference_is_independently_verified():
    instance = load_fjsp_instance(ROOT / "tests" / "data" / "fjsp_brandimarte_tiny.fjs")
    incumbent = build_fjsp_renewal_frame_schedule(instance)["schedule"]
    try:
        report = solve_fjsp_cp_sat(
            instance,
            time_limit_s=5.0,
            workers=1,
            incumbent_schedule=incumbent,
        )
    except ExactReferenceUnavailable:
        pytest.skip("isolated OR-Tools experiment dependency is not on PYTHONPATH")

    assert report["feasible_solution"] is True
    assert report["verifier"]["pass"] is True
    assert report["objective_makespan"] >= report["best_bound_makespan"]
    assert report["objective"] == "makespan"
    assert report["incumbent_hint_independently_verified"] is True


def test_mmrcpsp_cp_sat_reference_is_independently_verified():
    instance = parse_psplib_mm(ROOT / "tests" / "data" / "mmrcpsp_core.mm")
    incumbent = build_mmrcpsp_renewal_frame_schedule(instance)["schedule"]
    try:
        report = solve_mmrcpsp_cp_sat(
            instance,
            time_limit_s=5.0,
            workers=1,
            incumbent_schedule=incumbent,
        )
    except ExactReferenceUnavailable:
        pytest.skip("isolated OR-Tools experiment dependency is not on PYTHONPATH")

    assert report["feasible_solution"] is True
    assert report["verifier"]["schedule_feasible"] is True
    assert report["verifier"]["resource_violation_units"] == 0
    assert report["objective_makespan"] >= report["best_bound_makespan"]
    assert report["incumbent_hint_independently_verified"] is True


@pytest.mark.parametrize("budget", (0.0, -1.0, float("inf")))
def test_reference_budget_fails_closed(budget):
    instance = load_fjsp_instance(ROOT / "tests" / "data" / "fjsp_brandimarte_tiny.fjs")
    with pytest.raises(ValueError, match="time_limit_s"):
        solve_fjsp_cp_sat(instance, time_limit_s=budget)
