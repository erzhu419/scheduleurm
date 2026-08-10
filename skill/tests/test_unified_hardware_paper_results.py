from __future__ import annotations

import json
from pathlib import Path

from algorithm.experiments.unified_hardware_or_replay import (
    build_unified_hardware_or_replay,
)
from algorithm.experiments.unified_hardware_paper_results import (
    build_unified_hardware_paper_results,
    tex_macros,
    write_outputs,
)
from skill.tests.test_unified_hardware_or_replay import _write_fixture, _write_json


def test_passing_schema_v2_replay_exports_one_consistent_paper_summary(tmp_path):
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
    assert len(report["sota_style"]["policies"]) == 7
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


def test_schema_v1_or_nonpassing_replay_is_rejected(tmp_path):
    replay_path = tmp_path / "replay.json"
    slack_path = tmp_path / "slack.json"
    merge_path = tmp_path / "loaded_merge.json"
    _write_json(
        replay_path,
        {
            "gate": "unified_hardware_or_replay",
            "schema_version": 1,
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
    for index, node in enumerate(("jtl110gpu", "jtl110gpu2", "node007", "jtl311linux")):
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
