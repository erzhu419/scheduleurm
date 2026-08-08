from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import algorithm.experiments.fjsp_trajectory_upgrade as module
from algorithm.experiments.fjsp_instances import load_fjsp_instance
from algorithm.experiments.fjsp_trajectory_upgrade import (
    CLASSIC_POLICIES,
    DEFAULT_MANIFEST,
    DEFAULT_SUITE_DIR,
    FROZEN_CONFIG,
    FROZEN_CONFIG_SHA256,
    SPLITS,
    TRAJECTORY_POLICY,
    TrajectoryGateError,
    build_fjsp_trajectory_suite,
    build_trajectory_schedule,
    frozen_config_sha256,
    write_artifacts,
)


@pytest.fixture(scope="module")
def mk01():
    return load_fjsp_instance(DEFAULT_SUITE_DIR / "mk01.txt")


@pytest.fixture(scope="module")
def mk01_run(mk01):
    return build_trajectory_schedule(mk01)


@pytest.fixture(scope="module")
def suite_report():
    return build_fjsp_trajectory_suite()


def test_frozen_configuration_and_split_are_literal_and_disjoint():
    assert frozen_config_sha256() == FROZEN_CONFIG_SHA256
    assert FROZEN_CONFIG.snapshot()["per_instance_tuning"] is False
    assert FROZEN_CONFIG.snapshot()["cp_sat_used_by_ours"] is False
    assert SPLITS == {
        "development": ("mk01", "mk02", "mk03", "mk04", "mk05"),
        "calibration": ("mk06", "mk07", "mk08", "mk09", "mk10"),
        "holdout": ("mk11", "mk12", "mk13", "mk14", "mk15"),
    }
    flattened = [item for split in SPLITS.values() for item in split]
    assert flattened == [f"mk{index:02d}" for index in range(1, 16)]
    assert len(flattened) == len(set(flattened))


def test_generated_family_is_finite_feasible_and_exactly_maximized(mk01_run):
    selection = mk01_run["selection"]
    family = mk01_run["candidate_family"]
    assert selection["generated_candidate_count"] >= selection["unique_action_family_size"]
    assert selection["unique_action_family_size"] == len(family)
    assert selection["exact_family_argmax"] is True
    assert selection["oracle_gap_within_generated_family"] == 0
    assert selection["global_fjsp_oracle_gap"] is None
    assert selection["global_fjsp_oracle_claimed"] is False
    assert mk01_run["feasibility"]["pass"] is True
    assert all(row["feasible"] for row in family)
    assert float(family[0]["score"]) == max(float(row["score"]) for row in family)


def test_every_registered_seed_is_present_before_deduplication(mk01_run):
    assert set(mk01_run["baseline_runs"]) == set(module.POLICY_ORDER)
    origins = {
        origin
        for row in mk01_run["candidate_family"]
        for origin in row["origins"]
    }
    for policy in module.POLICY_ORDER:
        assert f"fixed_policy:{policy}" in origins


def test_score_identity_and_bounded_penalty_are_auditable(mk01_run):
    audit = mk01_run["selection"]["score_audit"]
    lower = float(audit["lower_service"]["normalized_weighted_value"])
    penalty = float(audit["penalty"]["penalty_units"])
    bound = float(audit["penalty"]["uniform_upper_bound"])
    assert float(audit["score"]) == pytest.approx(lower - penalty, abs=1e-12)
    assert 0 <= penalty <= bound
    assert bound == FROZEN_CONFIG.bounded_latency_penalty
    assert audit["penalty"]["queue_linear_coefficient_beta"] == 0
    assert audit["lower_service"]["source"] == "deterministic_processing_times_exact_lower_bound"


def test_algorithm_is_deterministic_excluding_observed_runtime(mk01, mk01_run):
    repeated = build_trajectory_schedule(mk01)
    assert repeated["schedule"] == mk01_run["schedule"]
    assert repeated["metrics"] == mk01_run["metrics"]
    assert repeated["selection"] == mk01_run["selection"]
    assert repeated["candidate_family"] == mk01_run["candidate_family"]
    assert repeated["runtime"]["wall_clock_seconds"] >= 0


def test_configuration_hash_mismatch_fails_before_search(monkeypatch, mk01):
    monkeypatch.setattr(module, "FROZEN_CONFIG_SHA256", "0" * 64)
    with pytest.raises(TrajectoryGateError, match="frozen configuration hash mismatch"):
        build_trajectory_schedule(mk01)


def test_source_byte_tampering_fails_closed(tmp_path: Path):
    copied = tmp_path / "suite"
    shutil.copytree(DEFAULT_SUITE_DIR, copied)
    manifest = copied / DEFAULT_MANIFEST.name
    (copied / "mk11.txt").write_bytes((copied / "mk11.txt").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_fjsp_trajectory_suite(copied, manifest)


def test_full_suite_preserves_holdout_and_theorem_claim_boundaries(suite_report):
    gate = suite_report["gate"]
    protocol = suite_report["frozen_protocol"]
    theorem = suite_report["theorem_mapping"]
    assert gate["pass"] is True
    assert gate["instance_count"] == 15
    assert gate["all_schedules_feasible"] is True
    assert gate["exact_generated_family_argmax_on_every_instance"] is True
    assert gate["zero_generated_family_oracle_gap_on_every_instance"] is True
    assert gate["ours_instance_pareto_nondominated_count"] == 15
    assert gate["global_fjsp_optimality_claim"] is False
    assert gate["stochastic_throughput_optimality_claim"] is False
    assert protocol["calibration_updates_applied"] is False
    assert protocol["holdout_feedback_used_for_configuration"] == (
        "not_auditable_before_repository_freeze"
    )
    assert protocol["holdout_execution_guard"] == "runtime_literal_config_hash_only"
    assert protocol["internal_partition_is_prospective_holdout"] is False
    assert gate["prospective_external_holdout_ready"] is False
    assert [row["split"] for row in protocol["phase_audit"]] == [
        "development",
        "calibration",
        "holdout",
    ]
    assert all(row["configuration_updates_applied"] is False for row in protocol["phase_audit"])
    assert theorem["oracle"]["exact_over_generated_family"] is True
    assert theorem["oracle"]["global_fjsp_action_space_exact"] is False
    assert theorem["oracle"]["alpha0"] == 0
    assert theorem["oracle"]["alpha1"] == 0


def test_comparison_with_every_classic_policy_is_complete_and_honest(suite_report):
    aggregate = suite_report["aggregate"]
    pairwise = aggregate["ours_vs_each_classic"]
    assert set(pairwise) == set(CLASSIC_POLICIES)
    assert TRAJECTORY_POLICY in aggregate["policies"]
    assert all(row["baseline_pareto_dominates_count"] == 0 for row in pairwise.values())
    assert aggregate["ours_strictly_pareto_dominates_every_classic_on_every_instance"] is False
    assert suite_report["gate"]["performance_superiority_claim_ready"] is False
    assert sum(row["ours_pareto_dominates_count"] for row in pairwise.values()) > 0


def test_runtime_family_size_and_oracle_gap_are_reported_per_instance(suite_report):
    for row in suite_report["instances"]:
        search = row["trajectory_search"]
        assert search["runtime_seconds"] >= 0
        assert search["unique_action_family_size"] > 0
        assert search["generated_candidate_count"] >= search["unique_action_family_size"]
        assert search["oracle_gap_within_generated_family"] == 0
        assert TRAJECTORY_POLICY in row["pareto_frontier_policies"]


def test_artifact_writer_emits_parseable_json_and_markdown(tmp_path: Path, suite_report):
    json_path, markdown_path = write_artifacts(
        suite_report,
        json_path=tmp_path / "gate.json",
        markdown_path=tmp_path / "gate.md",
    )
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["frozen_protocol"]["config_sha256"] == FROZEN_CONFIG_SHA256
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Mk11-Mk15 evaluation" in markdown
    assert "Prospective external holdout: not yet run" in markdown
    assert "not all FJSP schedules" in markdown
