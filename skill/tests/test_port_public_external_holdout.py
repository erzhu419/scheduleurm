from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments.port_public_external_holdout import (
    DEFAULT_SUITE_MANIFEST,
    TRAJECTORY_POLICY,
    PortHoldoutContractError,
    _evaluate_registered_instance,
    _factor_cluster_bootstrap,
    _relation,
    verify_suite_manifest,
)
from algorithm.experiments.port_public_external_holdout import _digest
from algorithm.experiments.port_scheduling_benchmark import BASELINE_POLICIES
from algorithm.experiments.port_trajectory_upgrade import TrajectoryPlan


DEVELOPMENT_COMPACT = (
    Path(__file__).resolve().parents[2]
    / "md"
    / "experiment_artifacts"
    / "port_trajectory_upgrade_v2_compact_20260809.json"
)


def test_suite_manifest_is_factor_complete_and_disjoint():
    audit = verify_suite_manifest()

    assert audit["ready"] is True
    assert audit["instance_count"] == 54
    assert audit["factor_cell_count"] == 27
    assert audit["replications_per_cell"] == 2
    assert audit["registered_replications"] == [89, 90]
    assert audit["development_holdout_path_disjoint"] is True
    assert audit["development_holdout_hash_disjoint"] is True


def test_suite_manifest_tamper_fails_closed(tmp_path):
    manifest = json.loads(DEFAULT_SUITE_MANIFEST.read_text(encoding="utf-8"))
    manifest["instances"][0]["byte_count"] += 1
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PortHoldoutContractError, match="digest mismatch"):
        verify_suite_manifest(changed)


def test_suite_manifest_accepts_only_explicit_registered_replication_pair(tmp_path):
    manifest = json.loads(DEFAULT_SUITE_MANIFEST.read_text(encoding="utf-8"))
    for row in manifest["instances"]:
        row["replication"] = 87 if int(row["replication"]) == 89 else 88
    manifest.pop("manifest_content_sha256_excluding_self", None)
    manifest["manifest_content_sha256_excluding_self"] = _digest(manifest)
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")

    audit = verify_suite_manifest(changed, expected_replications=(87, 88))

    assert audit["registered_replications"] == [87, 88]
    with pytest.raises(PortHoldoutContractError, match="registered replication pair"):
        verify_suite_manifest(changed, expected_replications=(89, 90))


def test_frozen_plan_runs_one_public_holdout_without_candidate_search():
    manifest = json.loads(DEFAULT_SUITE_MANIFEST.read_text(encoding="utf-8"))
    development = json.loads(DEVELOPMENT_COMPACT.read_text(encoding="utf-8"))
    selected = TrajectoryPlan(tuple(development["selected_global_plan"]["policies"]))
    plans = {
        TRAJECTORY_POLICY: selected,
        **{policy: TrajectoryPlan((policy,)) for policy in BASELINE_POLICIES},
    }

    first = _evaluate_registered_instance(manifest["instances"][0], plans)
    second = _evaluate_registered_instance(manifest["instances"][0], plans)

    assert first == second
    assert first["selected_feasible"] is True
    assert first["selected_mid_service_reconfiguration_count"] == 0
    assert set(first["baseline_relations"]) == set(BASELINE_POLICIES)
    assert set(first["policy_source_core_metrics"]) == {
        TRAJECTORY_POLICY,
        *BASELINE_POLICIES,
    }


def test_q5_holdout_row_is_source_feasible_after_capacity_intersection():
    manifest = json.loads(DEFAULT_SUITE_MANIFEST.read_text(encoding="utf-8"))
    development = json.loads(DEVELOPMENT_COMPACT.read_text(encoding="utf-8"))
    selected = TrajectoryPlan(tuple(development["selected_global_plan"]["policies"]))
    plans = {
        TRAJECTORY_POLICY: selected,
        **{policy: TrajectoryPlan((policy,)) for policy in BASELINE_POLICIES},
    }
    registered = next(
        row for row in manifest["instances"] if int(row["q_max_factor"]) == 5
    )

    result = _evaluate_registered_instance(registered, plans)

    assert result["crane_count"] == 5
    assert result["selected_feasible"] is True
    assert result["selected_mid_service_reconfiguration_count"] == 0


def test_pareto_relation_distinguishes_dominance_tradeoff_and_tie():
    metrics = ("a", "b")

    assert _relation({"a": 1, "b": 1}, {"a": 2, "b": 2}, metrics) == (
        "ours_dominates"
    )
    assert _relation({"a": 2, "b": 2}, {"a": 1, "b": 1}, metrics) == (
        "baseline_dominates"
    )
    assert _relation({"a": 1, "b": 2}, {"a": 2, "b": 1}, metrics) == "tradeoff"
    assert _relation({"a": 1, "b": 1}, {"a": 1, "b": 1}, metrics) == "tie"


def test_factor_cluster_bootstrap_is_deterministic():
    rows = []
    for q in (5, 10):
        for replication in (89, 90):
            rows.append(
                {
                    "q_max_factor": q,
                    "speed_setup_factor": "160-0.2",
                    "deadline_factor": "1",
                    "selected_source_core_metrics": {"m": 1.0 + q / 100.0},
                    "policy_source_core_metrics": {
                        "fcfs_static": {"m": 2.0 + replication / 1000.0}
                    },
                }
            )

    first = _factor_cluster_bootstrap(rows, "fcfs_static", "m")
    second = _factor_cluster_bootstrap(rows, "fcfs_static", "m")

    assert first == second
    assert 0.0 < first[0] <= first[1]
