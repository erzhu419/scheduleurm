from __future__ import annotations

import hashlib
import json

from algorithm.experiments.migration_rate_recalibration_gate import (
    DEFAULT_FINAL_CACHE as RECALIBRATION_DEFAULT_CACHE,
)
from algorithm.experiments.unified_hardware_or_replay import (
    DEFAULT_CACHE as REPLAY_DEFAULT_CACHE,
)
from algorithm.experiments.unified_migration_recalibration_certificate import (
    DEFAULT_FINAL_CACHE as EXPORT_DEFAULT_CACHE,
    build_unified_migration_recalibration_certificate,
)
from algorithm.theorem_dispatch.migration import MigrationCost


def test_pipeline_defaults_bind_the_same_final_cache():
    assert RECALIBRATION_DEFAULT_CACHE == EXPORT_DEFAULT_CACHE == REPLAY_DEFAULT_CACHE


def test_exports_complete_cache_bound_replay_contract(tmp_path):
    cache, gate = _inputs(tmp_path)

    report = build_unified_migration_recalibration_certificate(
        final_cache_path=cache,
        recalibration_path=gate,
    )

    assert report["status"] == "MIGRATION_RECALIBRATION_PASS"
    assert report["pass"] is True
    assert report["service_cache_sha256"] == _sha256(cache)
    assert report["rates_recomputed_from_final_cache"] is True
    assert report["no_touch_safety_ready"] is True
    assert len(report["rows"]) == 9
    assert {row["family"] for row in report["rows"]} == {
        "pure_cpu",
        "pure_gpu",
        "hybrid_rl",
    }
    assert all(row["ordinary_running_tasks_touched"] is False for row in report["rows"])
    row = report["rows"][0]
    expected_effective = row["remaining_work"] / (
        row["migration_cost"]["total_penalty_units"]
        + row["remaining_work"] / row["target_lower_service"]
    )
    assert row["effective_migration_lower_service"] == expected_effective


def test_rejects_dimensionally_inconsistent_beneficial_flag(tmp_path):
    cache, gate = _inputs(tmp_path)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    payload["rows"][0]["beneficial_by_recalculated_threshold"] = False
    gate.write_text(json.dumps(payload), encoding="utf-8")

    report = build_unified_migration_recalibration_certificate(
        final_cache_path=cache,
        recalibration_path=gate,
    )

    assert report["status"] == "FAIL_RECALIBRATION_CONTRACT"
    assert "beneficial decision is inconsistent" in report["blockers"][0]["detail"]


def test_missing_input_waits_without_rows(tmp_path):
    report = build_unified_migration_recalibration_certificate(
        final_cache_path=tmp_path / "missing-cache.json",
        recalibration_path=tmp_path / "missing-gate.json",
    )

    assert report["status"] == "WAIT_INPUTS"
    assert report["pass"] is False
    assert report["rows"] == []


def _inputs(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"records": [{"sentinel": True}]}), encoding="utf-8")
    cache_hash = _sha256(cache)
    cost = MigrationCost(
        checkpoint_flush_s=0.1,
        sync_s=0.2,
        environment_staging_s=0.1,
        resume_warmup_s=0.1,
        lost_work_s=0.0,
        risk_penalty_units=0.1,
    ).snapshot()
    rows = []
    families = (
        ("pure_cpu", "cpu_heavy_local_bench", "iteration"),
        ("pure_gpu", "gpu_cnn_torch_resnet50", "step"),
        ("hybrid_rl", "hybrid_rl_resac_ant", "iter"),
    )
    for family, workload, unit in families:
        for point in (0.25, 0.50, 0.75):
            remaining = 100.0 * (1.0 - point)
            current = 1.0
            target = 2.0
            keep = remaining / current
            migrate = cost["total_penalty_units"] + remaining / target
            rows.append(
                {
                    "migration_id": f"{family}_source_to_target",
                    "family": family,
                    "workload_key": workload,
                    "progress_fraction": point,
                    "recalculation_ready": True,
                    "source_state_request": _state(workload, "source", unit),
                    "target_state_request": _state(workload, "target", unit),
                    "source_rate_record": {"unit": unit},
                    "target_rate_record": {"unit": unit},
                    "current_lower_service_units_per_s": current,
                    "target_lower_service_units_per_s": target,
                    "remaining_work_units": remaining,
                    "keep_remaining_completion_s": keep,
                    "migrate_remaining_completion_s": migrate,
                    "migration_net_saving_s": keep - migrate,
                    "migration_cost": cost,
                    "cost_source": {"path": "fixture"},
                    "beneficial_by_recalculated_threshold": keep > migrate,
                    "migration_action_theorem_ready": keep > migrate,
                }
            )
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "gate": "migration_rate_recalibration_gate",
                "schema_version": 1,
                "status": "PASS",
                "pass": True,
                "certificate_ready": True,
                "final_unified_cache_sha256": cache_hash,
                "ordinary_user_tasks_excluded": True,
                "legacy_or_state_fallback_allowed": False,
                "old_cost_artifact_rates_used": False,
                "old_cost_artifact_beneficial_flags_inherited": False,
                "required_row_count": len(rows),
                "ready_row_count": len(rows),
                "blocked_row_count": 0,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return cache, gate


def _state(workload, node, unit):
    del unit
    return {
        "workload_key": workload,
        "workload_env": workload,
        "node": node,
        "node_bucket": f"{node}:fixture",
        "resource_state": "empty",
        "profile": 1,
        "allocation_workers": 1,
        "colocation_count": 1,
        "resident_mix": "",
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
