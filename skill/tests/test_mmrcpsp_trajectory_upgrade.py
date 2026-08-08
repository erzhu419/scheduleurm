from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments.mmrcpsp_instances import MMRCPSPFormatError
from algorithm.experiments.mmrcpsp_trajectory_upgrade import (
    BASELINE_POLICIES,
    BEAM_SEEDS,
    BEAM_WIDTH,
    PENALTY_BOUND,
    SCHEMA_VERSION,
    TRAJECTORY_POLICY,
    TrajectoryUpgradeError,
    build_trajectory_upgrade,
    write_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_ROOT = REPO_ROOT / "tests" / "data"
OFFICIAL_INSTANCE = (
    Path(__file__).resolve().parent / "data" / "psplib_mm_suite" / "j1010_1.mm"
)
CORE_INSTANCE = SYNTHETIC_ROOT / "mmrcpsp_core.mm"
RESERVE_INSTANCE = SYNTHETIC_ROOT / "mmrcpsp_nonrenewable_reserve.mm"
INFEASIBLE_INSTANCE = (
    SYNTHETIC_ROOT / "mmrcpsp_nonrenewable_coupled_infeasible.mm"
)


def _without_wall_clock(report):
    payload = json.loads(json.dumps(report))
    for row in payload["instances"].values():
        row["runtime"]["wall_clock_seconds"] = None
    return payload


def test_official_and_synthetic_instances_pass_the_trajectory_gate():
    report = build_trajectory_upgrade(
        (OFFICIAL_INSTANCE, CORE_INSTANCE, RESERVE_INSTANCE)
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["gate"] == {
        "status": "MMRCPSP_TRAJECTORY_ACTION_PASS",
        "pass": True,
        "all_evaluated_schedules_feasible": True,
        "zero_resource_violation": True,
        "zero_precedence_violation": True,
        "exact_oracle_over_generated_family": True,
        "performance_dominance_not_required": True,
    }
    assert report["per_instance_tuning"] is False
    assert report["construction_contract"]["beam_width"] == BEAM_WIDTH
    assert len(report["construction_contract"]["beam_seeds"]) == len(BEAM_SEEDS)


@pytest.mark.parametrize(
    "path",
    (OFFICIAL_INSTANCE, CORE_INSTANCE, RESERVE_INSTANCE),
)
def test_selected_trajectory_is_feasible_and_resource_certified(path):
    report = build_trajectory_upgrade((path,))
    row = report["instances"][path.stem]
    selected = row["selected_candidate"]
    metrics = selected["metrics"]

    assert selected["feasible"] is True
    assert metrics["schedule_feasible"] is True
    assert metrics["precedence_violation_count"] == 0
    assert metrics["resource_violation_units"] == 0.0
    assert all(value == 0 for value in metrics["renewable_peak_excess"])
    assert all(value == 0 for value in metrics["nonrenewable_excess"])
    assert row["resource_audit"]["renewable_peak_excess"] == metrics[
        "renewable_peak_excess"
    ]
    assert row["resource_audit"]["nonrenewable_used"] == metrics[
        "nonrenewable_used"
    ]
    assert len(selected["schedule"]) == row["instance"]["declared_job_count"]


def test_exact_generated_family_oracle_and_bounded_score_identity():
    report = build_trajectory_upgrade((OFFICIAL_INSTANCE,))
    row = report["instances"][OFFICIAL_INSTANCE.stem]
    selected = row["selected_candidate"]
    objective = selected["objective"]

    assert row["oracle_audit"]["oracle_gap"] == 0.0
    assert row["oracle_audit"]["exact_over_candidate_family"] is True
    assert row["oracle_audit"]["exact_over_full_feasible_set"] is False
    assert row["oracle_audit"]["selected_score"] == row["oracle_audit"][
        "maximum_generated_family_score"
    ]
    assert objective["bounded_trajectory_penalty"] <= PENALTY_BOUND
    assert objective["robust_score"] == pytest.approx(
        objective["cumulative_weighted_lower_service"]
        - objective["bounded_trajectory_penalty"],
        abs=2e-9,
    )
    coordinate_sum = sum(
        item["weighted_contribution"]
        for item in objective["activity_coordinates"]
    ) + objective["project_coordinate"]["weighted_contribution"]
    assert objective["cumulative_weighted_lower_service"] == pytest.approx(
        coordinate_sum, abs=5e-9
    )


def test_candidate_family_contains_rollout_beam_and_local_actions():
    report = build_trajectory_upgrade((OFFICIAL_INSTANCE,))
    family = report["instances"][OFFICIAL_INSTANCE.stem]["action_family"]

    assert family["finite"] is True
    assert family["unique_candidate_count"] > len(BASELINE_POLICIES)
    assert family["origin_counts"]["rollout"] > 0
    assert family["origin_counts"]["beam"] > 0
    assert family["origin_counts"]["local_improvement"] > 0
    counts = family["construction_counts"]
    assert counts["rollout_seed_count"] == len(BASELINE_POLICIES) + 1
    assert counts["beam_seed_count"] == len(BEAM_SEEDS)
    assert counts["beam_state_expansion_count"] > 0
    assert counts["local_neighbor_attempt_count"] >= counts[
        "local_neighbor_feasible_count"
    ]


def test_pareto_is_reported_honestly_and_not_used_as_pass_gate():
    report = build_trajectory_upgrade((OFFICIAL_INSTANCE, CORE_INSTANCE))

    assert report["gate"]["pass"] is True
    assert report["gate"]["performance_dominance_not_required"] is True
    assert report["claim_boundary"]["strong_claim_ready"] is False
    for row in report["instances"].values():
        pareto = row["pareto"]
        assert pareto["dominance_is_descriptive_not_gate"] is True
        assert set(pareto["comparisons"]) == set(BASELINE_POLICIES)
        expected = TRAJECTORY_POLICY in pareto["frontier"]
        assert pareto["selected_is_nondominated"] is expected
        # Every baseline trajectory belongs to the generated family.  The
        # cumulative-service objective is Pareto-monotone and its smallest
        # strict integer improvement exceeds the full penalty range.
        assert pareto["selected_is_nondominated"] is True
        for baseline, comparison in pareto["comparisons"].items():
            assert not (
                comparison["selected_pareto_dominates"]
                and comparison["baseline_pareto_dominates_selected"]
            ), baseline


def test_fixed_protocol_is_deterministic_except_observed_wall_clock():
    first = build_trajectory_upgrade((OFFICIAL_INSTANCE, CORE_INSTANCE))
    second = build_trajectory_upgrade((CORE_INSTANCE, OFFICIAL_INSTANCE))

    assert _without_wall_clock(first) == _without_wall_clock(second)
    for row in first["instances"].values():
        assert row["runtime"]["wall_clock_seconds"] >= 0
        assert row["runtime"]["deterministic_work_units"] > 0
        assert row["runtime"]["wall_clock_is_not_a_determinism_contract"] is True


def test_artifact_removes_nondeterministic_timing_and_is_byte_stable(tmp_path):
    report = build_trajectory_upgrade((OFFICIAL_INSTANCE, CORE_INSTANCE))
    output = tmp_path / "trajectory.json"

    write_artifact(report, output)
    first = output.read_bytes()
    write_artifact(report, output)

    assert output.read_bytes() == first
    payload = json.loads(first)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert all(
        row["runtime"]["wall_clock_seconds"] is None
        for row in payload["instances"].values()
    )


def test_infeasible_nonrenewable_fixture_fails_closed():
    with pytest.raises(TrajectoryUpgradeError, match="infeasible|deadlock"):
        build_trajectory_upgrade((INFEASIBLE_INSTANCE,))


def test_parser_failure_still_fails_before_candidate_construction():
    unsupported = SYNTHETIC_ROOT / "mmrcpsp_unsupported_doubly.mm"

    with pytest.raises(MMRCPSPFormatError, match="doubly constrained"):
        build_trajectory_upgrade((unsupported,))


def test_unknown_duplicate_instance_names_fail_closed(tmp_path):
    duplicate = tmp_path / OFFICIAL_INSTANCE.name
    duplicate.write_bytes(OFFICIAL_INSTANCE.read_bytes())

    with pytest.raises(ValueError, match="instance stems must be unique"):
        build_trajectory_upgrade((OFFICIAL_INSTANCE, duplicate))
