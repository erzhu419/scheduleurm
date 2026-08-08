from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from algorithm.experiments.fjsp_benchmark import POLICY_ORDER
from algorithm.experiments.fjsp_suite_benchmark import (
    DEFAULT_MANIFEST,
    DEFAULT_MARKDOWN,
    DEFAULT_SUITE_DIR,
    DEFAULT_JSON,
    EXPECTED_SOURCE_COMMIT,
    SuiteSourceError,
    artifact_json,
    build_fjsp_suite_gate,
    markdown_report,
    validate_source_manifest,
    write_artifacts,
)


EXPECTED_RAW_SHA256 = {
    "mk01": "d41e39a41d218ec19f5e9fb83a5a0bdc76dcfb93fa38a0d565318b3795c7d9fa",
    "mk02": "44360812291e71cb7f891a21495eecfa281b3f196940cd13d5f6c2a528679ad2",
    "mk03": "ede843eb1a261ebbeef09265d4de6adcf578b369d1af04cc779f91afbf73cd92",
    "mk04": "b751cfef26d2b4a2af392107e69a9272eb3eebbd960592ed6557a1166a8670ee",
    "mk05": "5b02c162a32a2089736310842d63ca84a14c6b2c471581d425e54f0edde0aa74",
    "mk06": "a9978b79a4d3b22e7eb6721f64a9b08a3e9ee0f24a7dfbbf874fa021cf6049d1",
    "mk07": "77f883325b0cde6b1349a015c6dccdbb2d80c41d6452e56d1bc148587a3c7e7a",
    "mk08": "d06776e82c0fe7b00a705bda06759ad4668025fefd1125f6639fd62d9a8bbf18",
    "mk09": "93a7f6dc3ff60d87b88511f7f0a73ce7116f9e1399822176bb8bd0adf51b80eb",
    "mk10": "f926d3522a57571182799de36be8d883f2b9996740a7a05292687edcab6d6d06",
    "mk11": "f1f02c236aa2db35fb9c4841f3367dc8b5c3be560f4abbcd50ac00c08e3aea89",
    "mk12": "4cd974c0d63caf0c07c55996ce6833e4c424a7eb5c52163d0497b497dbff43b2",
    "mk13": "43a9feab27578eae73846cd79682cc4960664b1fdd070f4275d538db5f153702",
    "mk14": "82117f847c16ca93edaad40075f0ebacbbad2d3daa8c8a084e6800dd4a8b0d9e",
    "mk15": "f0e124f5ffb8b702268e0010e596021a37d3d976f29bfde20271ed9acad83ded",
}


@pytest.fixture(scope="module")
def suite_report():
    return build_fjsp_suite_gate()


def test_source_manifest_pins_all_fifteen_raw_instances_and_attribution():
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["instances"]

    assert manifest["source"]["commit"] == EXPECTED_SOURCE_COMMIT
    assert manifest["source"]["bounds_sha256"] == (
        "9917b8d914a095fed5f822941b06c0e9e9cd12813aceff7fdbb3c46aae8cd5d9"
    )
    assert manifest["license"]["spdx"] == "CC-BY-SA-4.0"
    assert "ScheduleOpt/benchmarks" in manifest["license"]["attribution"]
    assert manifest["license"]["local_copy_modification"].startswith("none")
    assert [row["id"] for row in rows] == [f"mk{index:02d}" for index in range(1, 16)]
    assert {row["id"]: row["sha256"] for row in rows} == EXPECTED_RAW_SHA256


def test_source_validation_recomputes_every_sha_and_parses_every_instance():
    gate = validate_source_manifest()

    assert gate["pass"] is True
    assert len(gate["verified_instances"]) == 15
    assert {
        row["id"]: row["raw_sha256"] for row in gate["verified_instances"]
    } == EXPECTED_RAW_SHA256
    assert all(row["instance"].operation_count > 0 for row in gate["verified_instances"])


def test_source_validation_fails_closed_after_byte_tampering(tmp_path: Path):
    copied = tmp_path / "suite"
    shutil.copytree(DEFAULT_SUITE_DIR, copied)
    target = copied / "mk01.txt"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(SuiteSourceError, match="SHA-256 mismatch for mk01"):
        validate_source_manifest(copied, copied / "source_manifest.json")


def test_open_bound_intervals_are_preserved_without_optimality_claim(suite_report):
    by_id = {row["instance_id"]: row for row in suite_report["instances"]}

    assert (by_id["mk10"]["lower_bound"], by_id["mk10"]["bks"]) == (189, 193)
    assert (by_id["mk13"]["lower_bound"], by_id["mk13"]["bks"]) == (381, 390)
    assert by_id["mk10"]["bounds_equal"] is False
    assert by_id["mk13"]["bounds_equal"] is False
    assert all(
        result["reported_as_optimal"] is False
        for instance in suite_report["instances"]
        for result in instance["policies"].values()
    )
    assert suite_report["semantics"]["heuristic_optimality_claim"] is False


def test_all_ninety_policy_runs_are_feasible_and_gap_accounting_is_exact(suite_report):
    assert suite_report["gate"]["pass"] is True
    assert suite_report["gate"]["instance_count"] == 15
    assert suite_report["gate"]["policy_run_count"] == 90
    assert suite_report["gate"]["all_schedules_feasible"] is True
    assert suite_report["gate"]["source_lower_bound_violations"] == []

    for instance in suite_report["instances"]:
        lower_bound = float(instance["lower_bound"])
        bks = float(instance["bks"])
        assert set(instance["policies"]) == set(POLICY_ORDER)
        for result in instance["policies"].values():
            makespan = float(result["makespan"])
            assert result["feasible"] is True
            assert result["reconfiguration_count"] == 0
            assert result["reconfiguration_time"] == 0
            assert result["matched_bks"] is (makespan == bks)
            assert float(result["bks_gap_absolute"]) == makespan - bks
            assert float(result["lower_bound_gap_absolute"]) == makespan - lower_bound


def test_every_reported_pareto_frontier_is_recomputed_from_two_costs(suite_report):
    for instance in suite_report["instances"]:
        policies = instance["policies"]
        expected = []
        for policy in POLICY_ORDER:
            current = policies[policy]
            dominated = False
            for other_policy, other in policies.items():
                if other_policy == policy:
                    continue
                no_worse = (
                    float(other["makespan"]) <= float(current["makespan"])
                    and float(other["mean_flow_time"]) <= float(current["mean_flow_time"])
                )
                strict = (
                    float(other["makespan"]) < float(current["makespan"])
                    or float(other["mean_flow_time"]) < float(current["mean_flow_time"])
                )
                dominated = dominated or (no_worse and strict)
            if not dominated:
                expected.append(policy)
        assert instance["pareto_frontier_policies"] == expected


def test_performance_gate_is_honest_about_current_ours_result(suite_report):
    gate = suite_report["gate"]
    aggregate = suite_report["aggregate"]

    assert gate["gate_scope"] == "source_integrity_and_schedule_feasibility"
    assert gate["performance_superiority_claim_ready"] is False
    assert gate["ours_aggregate_pareto_nondominated"] is False
    assert gate["ours_instance_pareto_nondominated_count"] == 4
    assert gate["ours_instance_pareto_dominated_count"] == 11
    assert gate["ours_matched_bks_count"] == 0
    assert "ours_robust_maxweight" not in aggregate["aggregate_pareto_frontier_policies"]
    assert aggregate["policies"]["most_work_remaining"]["matched_bks_count"] == 1


def test_aggregate_counts_are_complete_and_pairwise_partitions_sum_to_fifteen(
    suite_report,
):
    aggregate = suite_report["aggregate"]
    assert aggregate["instance_count"] == 15
    assert aggregate["policy_run_count"] == 90
    for row in aggregate["policies"].values():
        assert row["instance_count"] == 15
        assert row["feasible_count"] == 15
        assert row["reported_as_optimal_count"] == 0
    for row in aggregate["ours_vs_baselines"].values():
        assert (
            row["makespan_win_count"]
            + row["makespan_tie_count"]
            + row["makespan_loss_count"]
            == 15
        )
        assert (
            row["ours_pareto_dominates_count"]
            + row["baseline_pareto_dominates_count"]
            + row["tradeoff_or_equal_count"]
            == 15
        )


def test_suite_artifacts_are_deterministic_hash_bound_and_checked_in(
    suite_report,
    tmp_path: Path,
):
    json_text = artifact_json(suite_report)
    markdown_text = markdown_report(suite_report)

    assert "timestamp" not in json_text.lower()
    assert json_text == DEFAULT_JSON.read_text(encoding="utf-8")
    assert markdown_text == DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    assert "Performance-superiority claim ready: `false`" in markdown_text
    assert "no heuristic optimality claim" in markdown_text

    json_path, markdown_path = write_artifacts(
        suite_report,
        json_path=tmp_path / "suite.json",
        markdown_path=tmp_path / "suite.md",
    )
    assert sha256(json_path.read_bytes()).hexdigest() == sha256(
        DEFAULT_JSON.read_bytes()
    ).hexdigest()
    assert markdown_path.read_text(encoding="utf-8") == markdown_text
