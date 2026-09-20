from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments.unified_hardware_or_replay import (
    _common_complete_runs,
    _legacy_comparison_rows,
    _pareto_rows,
    build_unified_hardware_or_replay,
)
from algorithm.experiments.unified_hardware_paper_results import (
    build_unified_hardware_paper_results,
    tex_macros,
    write_outputs,
)
from skill.tests.test_unified_hardware_or_replay import _write_fixture, _write_json


def test_passing_schema_v3_replay_exports_one_consistent_paper_summary(tmp_path):
    inputs = _write_fixture(tmp_path / "inputs")
    replay = build_unified_hardware_or_replay(
        cache_path=inputs["cache"],
        loaded_ledger_path=inputs["ledger"],
        migration_certificate_path=inputs["migration"],
        loads=(0.50,),
        seeds=(17,),
        jobs_per_workload=3,
    )
    replay_path = tmp_path / "replay.json"
    slack_path = tmp_path / "slack.json"
    merge_path = tmp_path / "loaded_merge.json"
    _write_json(replay_path, replay)
    _write_json(slack_path, _slack_fixture())
    _write_json(merge_path, _merge_fixture(replay))

    report = build_unified_hardware_paper_results(
        replay_path=replay_path,
        slack_path=slack_path,
        loaded_merge_path=merge_path,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["population"]["hardware_scenario_count"] == 7
    assert report["population"]["migration_scenario_count"] == 9
    assert report["population"]["hardware_trace_count"] == (
        report["population"]["hardware_static_trace_count"]
        + report["population"]["hardware_online_trace_count"]
    )
    assert report["legacy"]["comparison_count"] > 0
    assert report["static_legacy"]["comparison_count"] == report["population"][
        "hardware_static_trace_count"
    ]
    assert 0 < report["online_legacy"]["comparison_count"] < report["legacy"]["comparison_count"]
    assert report["online_legacy"]["comparison_count"] == report["population"][
        "hardware_online_trace_count"
    ]
    assert len(report["static_scenarios"]) == 7
    assert {row["scenario_id"] for row in report["static_scenarios"]} == {
        row["scenario_id"]
        for row in replay["scenarios"]
        if row["scenario_kind"] == "hardware_local"
    }
    assert all(row["trace_count"] == 1 for row in report["static_scenarios"])
    assert all(
        row["display_label"].startswith(f"{row['quadrant']} ")
        for row in report["static_scenarios"]
    )
    assert len(report["sota_style"]["policies"]) == 7
    assert report["sota_style"]["policy_trace_comparison_count"] == (
        report["sota_style"]["comparison_trace_count"] * 7
    )
    assert set(report["sota_style"]["by_quadrant"]) == {"q00", "q01", "q10", "q11"}
    assert all(
        row["policy_trace_comparison_count"] == row["trace_count"] * 7
        for row in report["sota_style"]["by_quadrant"].values()
    )
    assert len(report["ablations"]) == 6
    assert report["migration"]["comparison_count"] == 9
    assert (
        report["queues"]["distributions"]["time_weighted_queue_backlog_jobs"]["count"]
        == report["population"]["hardware_online_scheduleurm_run_count"]
    )
    assert report["queues"] == report["online_queues"]
    assert report["slack"]["minimum_eta"] == 0.05
    tex_source = tex_macros(report)
    assert "\\UnifiedLegacyMakespanGeoRatio" in tex_source
    assert "\\UnifiedOnlineLegacyMakespanGeoRatio" in tex_source
    assert "\\UnifiedStaticLegacyMakespanGeoRatio" in tex_source
    assert "\\UnifiedQueueMeanBacklogMax" in tex_source
    assert "\\UnifiedQueueMaxBacklogMax" in tex_source
    assert "\\UnifiedSotaConclusion" in tex_source
    assert "\\UnifiedSotaPolicyTraceComparisonCount" in tex_source
    assert "\\UnifiedSotaDominatesOursCount" in tex_source
    assert "\\UnifiedSotaQZeroOneWorstCostRatio" in tex_source
    assert "\\UnifiedAblationDominatesFullCount" in tex_source
    assert "\\UnifiedStaticLegacyRows" in tex_source
    assert "\\UnifiedSotaPolicyRows" in tex_source
    assert "\\UnifiedAblationRows" in tex_source
    assert "\\UnifiedHardwareSlackRows" in tex_source
    assert "\\UnifiedOnlineArrivalFamilyRows" in tex_source
    assert "\\UnifiedOnlineQuadrantRows" in tex_source

    output = tmp_path / "summary.json"
    markdown = tmp_path / "summary.md"
    tex = tmp_path / "summary.tex"
    write_outputs(report, output=output, markdown=markdown, tex=tex)
    assert json.loads(output.read_text(encoding="utf-8"))["pass"] is True
    assert "SOTA-style policies" in markdown.read_text(encoding="utf-8")
    assert "do not edit manually" in tex.read_text(encoding="utf-8")


@pytest.mark.parametrize("failed_category", ["ours", "sota_style", "ablation"])
def test_incomplete_tail_excludes_whole_trace_but_preserves_registered_population(
    tmp_path, failed_category
):
    inputs = _write_fixture(tmp_path / "inputs")
    replay = build_unified_hardware_or_replay(
        cache_path=inputs["cache"],
        loaded_ledger_path=inputs["ledger"],
        migration_certificate_path=inputs["migration"],
        loads=(0.50,),
        seeds=(17,),
        jobs_per_workload=3,
    )
    assert all(row["completion_status"] == "complete" for row in replay["runs"])
    registered_ours = [
        row for row in replay["runs"]
        if row["scenario_kind"] == "hardware_local"
        and row["migration_mode"] == "without_migration"
        and row["policy_category"] == "ours"
    ]
    registered_online = sum(row["arrival_family"] != "static" for row in registered_ours)
    failed_run = next(
        row for row in replay["runs"]
        if row["scenario_kind"] == "hardware_local"
        and row["migration_mode"] == "without_migration"
        and row["arrival_family"] != "static"
        and row["policy_category"] == failed_category
    )
    excluded_key = (failed_run["scenario_id"], failed_run["trace_id"], failed_run["migration_mode"])
    metrics = ("makespan_s", "mean_flow_s", "p90_flow_s")
    for row in replay["runs"]:
        # On retained traces every comparator beats ours by exactly twofold.
        duration = 2.0 if row["policy_category"] == "ours" else 1.0
        if (row["scenario_id"], row["trace_id"], row["migration_mode"]) == excluded_key:
            # A per-policy deletion would leak an extreme apparent advantage
            # for ours into the other comparators' aggregate ratios.
            duration = 1.0 if row["policy_category"] == "ours" else 1e30
        row.update({metric: duration for metric in metrics})
    failed_run.update({metric: None for metric in metrics})
    failed_run["completion_status"] = "incomplete_tail"
    failed_run["completed_jobs"] -= 1
    failed_run["unfinished_jobs"] = 1

    common_runs = _common_complete_runs(replay["runs"])
    assert not any(
        (row["scenario_id"], row["trace_id"], row["migration_mode"]) == excluded_key
        for row in common_runs
    )
    assert sum(
        row["trace_id"] == failed_run["trace_id"] and row["migration_mode"] == "with_migration"
        for row in common_runs
    ) == replay["policy_count"]
    replay["pareto_rows"] = _pareto_rows(common_runs)
    replay["legacy_comparison_rows"] = _legacy_comparison_rows(common_runs)
    replay_path = tmp_path / "replay.json"
    slack_path = tmp_path / "slack.json"
    merge_path = tmp_path / "loaded_merge.json"
    _write_json(replay_path, replay)
    _write_json(slack_path, _slack_fixture())
    _write_json(merge_path, _merge_fixture(replay))

    report = build_unified_hardware_paper_results(
        replay_path=replay_path,
        slack_path=slack_path,
        loaded_merge_path=merge_path,
    )

    assert report["status"] == "PASS"
    population = report["population"]
    assert population["complete_matrix_run_count"] == len(replay["runs"])
    assert population["registered_hardware_trace_count"] == len(registered_ours)
    assert population["registered_online_trace_count"] == registered_online
    assert population["hardware_trace_count"] == len(registered_ours) - 1
    assert population["hardware_online_trace_count"] == registered_online - 1
    assert population["excluded_incomplete_trace_count"] == 1
    assert population["excluded_incomplete_online_trace_count"] == 1
    assert population["incomplete_scheduleurm_run_count"] == int(failed_category == "ours")
    assert population["incomplete_runs_by_policy"] == {
        row["policy"]: int(row["policy"] == failed_run["policy"])
        for row in replay["runs"]
    }
    assert population["incomplete_tail_rows"] == [
        {key: failed_run[key] for key in (
            "scenario_id", "trace_id", "policy", "completed_jobs",
            "unfinished_jobs", "observation_horizon_s",
        )}
    ]
    comparison_count = len(registered_ours) - 1
    assert report["legacy"]["comparison_count"] == comparison_count
    assert report["legacy"]["legacy_to_ours_makespan_geomean_ratio"] == pytest.approx(0.5)
    assert report["legacy"]["legacy_to_ours_mean_flow_geomean_ratio"] == pytest.approx(0.5)
    assert report["legacy"]["ours_strictly_dominates_every_row"] is False
    assert report["online_legacy"]["comparison_count"] == registered_online - 1
    assert report["sota_style"]["comparison_trace_count"] == comparison_count
    assert report["sota_style"]["policy_trace_comparison_count"] == 7 * comparison_count
    assert report["sota_style"]["ours_better_on_both_geomean_policy_count"] == 0
    assert report["sota_style"]["ours_strictly_dominates_every_policy_in_every_trace"] is False
    for row in report["sota_style"]["policies"].values():
        assert row["comparison_count"] == comparison_count
        assert row["baseline_to_ours_makespan_geomean_ratio"] == pytest.approx(0.5)
        assert row["baseline_to_ours_mean_flow_geomean_ratio"] == pytest.approx(0.5)
    for row in report["ablations"].values():
        assert row["comparison_count"] == comparison_count
        assert row["ablation_to_full_makespan_geomean_ratio"] == pytest.approx(0.5)
        assert row["ablation_to_full_mean_flow_geomean_ratio"] == pytest.approx(0.5)
    assert report["online_queues"]["run_count"] == registered_online - 1
    assert "conditional ratios do not measure overall superiority" in report["claim_boundary"]


@pytest.mark.parametrize("schema_version", [1, 2])
def test_earlier_replay_schemas_are_rejected(tmp_path, schema_version):
    replay_path = tmp_path / "replay.json"
    slack_path = tmp_path / "slack.json"
    merge_path = tmp_path / "loaded_merge.json"
    _write_json(
        replay_path,
        {
            "gate": "unified_hardware_or_replay",
            "schema_version": schema_version,
            "status": "PASS",
            "pass": True,
        },
    )
    _write_json(slack_path, _slack_fixture())
    _write_json(merge_path, _merge_fixture({"input_manifest": {}}))

    report = build_unified_hardware_paper_results(
        replay_path=replay_path,
        slack_path=slack_path,
        loaded_merge_path=merge_path,
    )

    assert report["status"] == "FAIL_INPUT_CONTRACT"
    assert report["pass"] is False
    assert "schema_version" in report["errors"][0]["detail"]


def test_loaded_merge_must_hash_bind_slack_cache_replay_cache_and_ledger(tmp_path):
    inputs = _write_fixture(tmp_path / "inputs")
    replay = build_unified_hardware_or_replay(
        cache_path=inputs["cache"],
        loaded_ledger_path=inputs["ledger"],
        migration_certificate_path=inputs["migration"],
        loads=(0.50,),
        seeds=(3,),
        jobs_per_workload=2,
    )
    replay_path = tmp_path / "replay.json"
    slack_path = tmp_path / "slack.json"
    merge_path = tmp_path / "loaded_merge.json"
    _write_json(replay_path, replay)
    _write_json(slack_path, _slack_fixture())
    merge = _merge_fixture(replay)
    merge["cache_output_sha256"] = "0" * 64
    _write_json(merge_path, merge)

    report = build_unified_hardware_paper_results(
        replay_path=replay_path,
        slack_path=slack_path,
        loaded_merge_path=merge_path,
    )

    assert report["status"] == "FAIL_INPUT_CONTRACT"
    assert "cache_output_sha256" in report["errors"][0]["detail"]


def _slack_fixture() -> dict[str, object]:
    nodes = []
    for index, node in enumerate(("gpu_a", "gpu_b", "node007", "cpu_a")):
        nodes.append(
            {
                "node": node,
                "operational_execution_class": f"test:{node}",
                "delta": 0.05 + 0.01 * index,
                "eta": 0.05 + 0.01 * index,
                "nominal_lcb_coverage": 0.9,
                "ready": True,
            }
        )
    return {
        "gate": "critical_gpu_all_hardware_statewise_slack_certificate",
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "load_fraction": 0.8,
        "minimum_eta": 0.05,
        "simultaneous_four_node_union_bound_coverage": 0.6,
        "cache_sha256": "a" * 64,
        "nodes": nodes,
    }


def _merge_fixture(replay: dict[str, object]) -> dict[str, object]:
    manifest = replay.get("input_manifest") or {}
    cache_hash = (manifest.get("unified_cache") or {}).get("sha256") or "b" * 64
    ledger_hash = (manifest.get("loaded_action_ledger") or {}).get("sha256") or "c" * 64
    return {
        "gate": "critical_gpu_loaded_action_ledger_merge",
        "status": "PASS",
        "pass": True,
        "registered_loaded_action_ledger_ready": True,
        "unified_cache_ready": True,
        "base_cache_sha256": "a" * 64,
        "cache_output_sha256": cache_hash,
        "ledger_output_sha256": ledger_hash,
    }
