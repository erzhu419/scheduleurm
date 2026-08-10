from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments.mmrcpsp_renewal_stream_benchmark import (
    ArrivalScenario,
    OURS_POLICY,
    POLICY_ORDER,
    build_registered_trajectory_library,
    run_mmrcpsp_renewal_stream_benchmark,
    select_registered_action,
)


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_PATHS = (
    ROOT / "tests" / "data" / "mmrcpsp_core.mm",
    ROOT / "tests" / "data" / "mmrcpsp_nonrenewable_reserve.mm",
)
TEST_SCENARIOS = (
    ArrivalScenario("static_test", "static", "none", 0.0, 72, 2),
    ArrivalScenario("poisson_test", "poisson", "poisson", 0.65, 72, 1),
    ArrivalScenario(
        "bursty_test", "bursty", "markov_bursty", 0.65, 72, 1
    ),
    ArrivalScenario("load_interior", "load_sweep", "poisson", 0.75, 72, 1),
    ArrivalScenario("load_exterior", "load_sweep", "poisson", 1.05, 72, 1),
)


@pytest.fixture(scope="module")
def library():
    return build_registered_trajectory_library(INSTANCE_PATHS)


@pytest.fixture(scope="module")
def report():
    return run_mmrcpsp_renewal_stream_benchmark(
        INSTANCE_PATHS,
        scenarios=TEST_SCENARIOS,
        seeds=(17,),
    )


def test_registered_library_is_finite_deduplicated_and_feasible(library):
    assert library
    assert {action.workload_class for action in library} == {
        "mmrcpsp_core",
        "mmrcpsp_nonrenewable_reserve",
    }
    identities = {
        (action.workload_class, action.trajectory_sha256) for action in library
    }
    assert len(identities) == len(library)
    assert all(action.schedule_feasible for action in library)
    assert all(action.resource_violation_units == 0 for action in library)
    assert all(action.precedence_violation_count == 0 for action in library)
    assert all(
        0 <= action.bounded_penalty <= action.penalty_uniform_bound < 1
        for action in library
    )
    assert len({action.duration for action in library}) > 1


def test_duration_normalized_selector_is_exact_over_registered_family(library):
    classes = sorted({action.workload_class for action in library})
    queue = {workload_class: 3 for workload_class in classes}
    common_horizon = max(action.common_instance_horizon for action in library)
    selected, audit = select_registered_action(
        library,
        queue=queue,
        policy=OURS_POLICY,
        common_horizon=common_horizon,
        remaining_time=100,
    )

    assert selected is not None
    assert audit["exact_argmax"] is True
    assert audit["oracle_gap"] == 0
    assert audit["selected_score"] == max(audit["scores"].values())


def test_static_poisson_bursty_and_load_sweep_execute(report):
    assert report["gate"]["status"] == "MMRCPSP_RENEWAL_STREAM_PROTOCOL_PASS"
    assert report["gate"]["pass"] is True
    assert set(report["scenario_order"]) == {
        "static_test",
        "poisson_test",
        "bursty_test",
        "load_interior",
        "load_exterior",
    }
    assert {row["scenario"]["family"] for row in report["runs"]} == {
        "static",
        "poisson",
        "bursty",
        "load_sweep",
    }
    assert len(report["runs"]) == len(TEST_SCENARIOS) * len(POLICY_ORDER)


def test_queue_recurrence_drift_identity_and_conservation_are_exact(report):
    for run in report["runs"]:
        gate = run["gate"]
        assert gate["pass"] is True
        assert gate["queue_recurrence_pathwise_exact"] is True
        assert gate["quadratic_drift_identity_pathwise_exact"] is True
        assert gate["workload_conservation_exact"] is True
        assert gate["bounded_penalty_respected"] is True
        assert gate["no_project_overservice"] is True
        assert gate["fixed_wall_clock_horizon_respected"] is True
        assert all(
            row["project_identity_residual"] == 0
            and row["admitted_activity_obligations"]
            == row["completed_internal_activities"]
            + row["remaining_activity_obligations"]
            for row in run["workload_conservation"].values()
        )
        assert all(frame["recurrence_exact"] for frame in run["frames"])


def test_same_seed_uses_same_exogenous_tape_for_every_policy(report):
    grouped = {}
    for run in report["runs"]:
        key = (run["scenario"]["scenario_id"], run["seed"])
        grouped.setdefault(key, set()).add(run["arrival_tape_sha256"])
    assert grouped
    assert all(len(digests) == 1 for digests in grouped.values())
    assert report["arrival_contract"][
        "policy_cannot_change_random_draws_by_changing_frame_duration"
    ] is True


def test_load_sweep_exposes_interior_and_exterior_points(report):
    ours = [row for row in report["runs"] if row["policy"] == OURS_POLICY]
    interior = next(
        row for row in ours if row["scenario"]["scenario_id"] == "load_interior"
    )
    exterior = next(
        row for row in ours if row["scenario"]["scenario_id"] == "load_exterior"
    )
    assert interior["stability_diagnostics"]["nominal_capacity_interior"] is True
    assert exterior["stability_diagnostics"]["capacity_exterior_or_boundary"] is True
    assert interior["stability_diagnostics"][
        "quadratic_drift_identity_pathwise_exact"
    ] is True


def test_claim_boundary_does_not_turn_diagnostics_into_a_theorem(report):
    boundary = report["claim_boundary"]
    assert boundary["registered_finite_library_only"] is True
    assert boundary[
        "finite_sample_drift_is_diagnostic_not_positive_recurrence_proof"
    ] is True
    assert boundary["global_mmrcpsp_optimality_claim"] is False
    assert boundary["psplib_wide_generalization_claim"] is False
    assert boundary["production_arrival_model_claim"] is False
    assert boundary["stochastic_stability_claim_ready"] is False


def test_benchmark_is_deterministic_for_fixed_seed(report):
    repeated = run_mmrcpsp_renewal_stream_benchmark(
        tuple(reversed(INSTANCE_PATHS)),
        scenarios=TEST_SCENARIOS,
        seeds=(17,),
    )
    assert repeated == report
