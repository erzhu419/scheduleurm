from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from algorithm.experiments.critical_gpu_completion_campaign import (
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    NODE_SPECS,
    PROTOCOL,
    TRAINING_WAVES,
    campaign_cells,
)
from algorithm.experiments.critical_gpu_lcb_cache_merge import (
    COMMAND_FINGERPRINT,
    ETA_SOURCE,
    REPRESENTATIVE_NODES,
    build_critical_gpu_lcb_cache_merge,
    main,
)
from algorithm.experiments.critical_gpu_all_hardware_lcb_cache_merge import (
    DEFAULT_JTL311_GATE,
    EFFECTIVE_NODE_BUCKET,
    build_critical_gpu_all_hardware_lcb_cache_merge,
)
from simulation.service_cache import ProfileRecord, ServiceRateCache


def test_all_hardware_default_consumes_the_stochastic_gate_producer_filename():
    assert DEFAULT_JTL311_GATE.name == (
        "critical_gpu_stochastic_lcb_gate_v5_jtl311linux_20260803.json"
    )


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_cache(path: Path) -> ProfileRecord:
    legacy = ProfileRecord(
        workload_key="legacy_control",
        command_fingerprint="legacy_v1",
        resource_kind="gpu",
        node_bucket="",
        profile=2,
        unit="step",
        total_units=10.0,
        aggregate_rate=4.0,
        per_task_rates=(2.0, 2.0),
        source="unit-test",
        resource_state="unspecified",
    )
    ServiceRateCache([legacy]).save(path)
    return legacy


def _certificate_row(node: str, cell: dict) -> dict:
    profile = int(cell["profile"])
    gpu_count = len(NODE_SPECS[node].gpus)
    task_count = profile * gpu_count
    total_units = float(cell["max_iters"])
    startup = 1.0 + 0.01 * profile
    unit_s = 0.2 + 0.001 * profile
    terminal = 0.5
    point = startup + total_units * unit_s + terminal
    upper_drain = point * 1.25
    aggregate_units = total_units * task_count
    lower = aggregate_units / upper_drain
    return {
        **cell,
        "resource_state": "empty",
        "gpu_count": gpu_count,
        "total_task_count": task_count,
        "total_units_per_task": total_units,
        "service_unit": "iter" if cell["workload_key"].startswith("hybrid_") else "step",
        "training_sample_count": len(TRAINING_WAVES),
        "calibration_wave_count": len(CALIBRATION_WAVES),
        "holdout_wave": HOLDOUT_WAVE,
        "operational_pre_holdout_phase_model": {
            "fit": "componentwise_median",
            "sample_count": len(TRAINING_WAVES) + len(CALIBRATION_WAVES),
            "mean_jct_phase": {
                "startup_overhead_s": startup,
                "completion_unit_s": unit_s,
                "terminal_overhead_s": terminal,
                "total_units_per_task": total_units,
            },
        },
        "operational_point_mean_task_jct_s": point,
        "mean_jct_point_relative_error": 0.02,
        "simultaneous_upper_drain_completion_s": upper_drain,
        "simultaneous_upper_mean_task_jct_s": point * 1.25,
        "holdout_actual_drain_completion_s": point,
        "holdout_actual_mean_task_jct_s": point,
        "drain_completion_holdout_covered": True,
        "mean_task_jct_holdout_covered": True,
        "aggregate_total_units": aggregate_units,
        "lower_service_units_per_s": lower,
        "realized_service_units_per_s": aggregate_units / point,
        "lower_service_valid_on_holdout": True,
        "lower_service_bound_kind": (
            "phase_aware_wave_max_simultaneous_split_conformal"
        ),
    }


def _passing_gate(node: str) -> dict:
    spec = NODE_SPECS[node]
    rows = [
        _certificate_row(node, cell)
        for cell in campaign_cells(node=node, wave=TRAINING_WAVES[0])
    ]
    return {
        "gate": "critical_gpu_stochastic_lcb_gate",
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "certificate_ready": True,
        "node": node,
        "node_bucket": spec.node_bucket,
        "hardware_class": spec.hardware_class,
        "resource_state": "empty",
        "measurement_protocol": PROTOCOL,
        "same_measurement_protocol_all_waves": True,
        "all_measurements_ready": True,
        "same_cell_set_all_waves": True,
        "hardware_local_only": True,
        "smoke_wave_excluded": True,
        "expected_cell_count": 9,
        "expected_wave_count": 13,
        "validation_errors": [],
        "wait_reasons": [],
        "certificate": {
            "constructed": True,
            "finite_sample_rank_ready": True,
            "all_holdout_bounds_valid": True,
            "simultaneous_ratio_margin": 0.25,
            "rows": rows,
        },
    }


def _passing_gate_with_training_capacity_exclusion(node: str) -> dict:
    gate = _passing_gate(node)
    excluded = gate["certificate"]["rows"].pop(0)
    gate.update(
        {
            "admitted_cell_count": 8,
            "training_capacity_excluded_cell_count": 1,
            "candidate_support_frozen_from_training_waves": True,
            "training_capacity_excluded_cells": [
                {
                    **excluded,
                    "support_status": "TRAINING_CAPACITY_EXCLUDED",
                    "first_capacity_wave": 2,
                    "capacity_waves": [2],
                    "later_success_cannot_readmit": True,
                    "lower_service": 0.0,
                }
            ],
        }
    )
    return gate


def _gate_files(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for node in REPRESENTATIVE_NODES:
        path = tmp_path / f"{node}.json"
        _write(path, _passing_gate(node))
        paths[node] = path
    return paths


def test_merge_waits_without_all_three_gates_and_exposes_no_snapshot(tmp_path):
    base = tmp_path / "base.json"
    _base_cache(base)
    one = tmp_path / "jtl110gpu.json"
    _write(one, _passing_gate("jtl110gpu"))

    report = build_critical_gpu_lcb_cache_merge(
        base_cache_path=base,
        gate_paths={"jtl110gpu": one},
    )

    assert report["status"] == "WAIT_GPU_GATES"
    assert report["pass"] is False
    assert report["complete_cache_ready"] is False
    assert report["final_cache_snapshot_included"] is False
    assert "service_cache_snapshot" not in report


def test_merge_waits_for_source_wait_gate(tmp_path):
    base = tmp_path / "base.json"
    _base_cache(base)
    gates = _gate_files(tmp_path)
    waiting = _passing_gate("node007")
    waiting.update({"status": "WAIT_MISSING_WAVES", "pass": False})
    _write(gates["node007"], waiting)

    report = build_critical_gpu_lcb_cache_merge(
        base_cache_path=base,
        gate_paths=gates,
    )

    assert report["status"] == "WAIT_GPU_GATES"
    assert "service_cache_snapshot" not in report
    assert any(
        issue["node"] == "node007" for issue in report["wait_reasons"]
    )


def test_pass_merge_is_exact_per_gpu_and_preserves_legacy_index(tmp_path):
    base = tmp_path / "base.json"
    legacy = _base_cache(base)
    gates = _gate_files(tmp_path)

    report = build_critical_gpu_lcb_cache_merge(
        base_cache_path=base,
        gate_paths=gates,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["inserted_count"] == 27
    assert report["legacy_index_unchanged"] is True
    assert report["legacy_index_sha256_before"] == report["legacy_index_sha256_after"]
    cache = ServiceRateCache.from_snapshot(report["service_cache_snapshot"])
    assert cache.get("legacy_control", 2) == legacy
    assert cache.get("gpu_llm_distilgpt2", 10) is None

    exact = cache.lookup_statewise_exact(
        "gpu_llm_distilgpt2",
        workload_env="distilgpt2_forward_fp16_b2_s128",
        node_bucket=NODE_SPECS["node007"].node_bucket,
        resource_state="empty",
        allocation_workers=1,
        colocation_count=10,
        resident_mix="",
    )
    assert exact.is_exact
    record = exact.record
    assert record is not None
    source_row = next(
        row
        for row in _passing_gate("node007")["certificate"]["rows"]
        if row["workload_key"] == "gpu_llm_distilgpt2" and row["profile"] == 10
    )
    assert record.aggregate_rate == pytest.approx(
        source_row["lower_service_units_per_s"]
        / len(NODE_SPECS["node007"].gpus)
    )
    assert len(record.per_task_rates) == 10
    assert sum(record.per_task_rates) == pytest.approx(record.aggregate_rate)
    assert record.total_units == source_row["total_units_per_task"]
    assert record.completion_total_wall_s == source_row[
        "operational_point_mean_task_jct_s"
    ]
    assert record.completion_eta_s(include_startup=True) == pytest.approx(
        source_row["operational_point_mean_task_jct_s"]
    )
    assert record.completion_model_sample_count == 12
    assert record.hardware_class == NODE_SPECS["node007"].hardware_class
    assert record.eta_source == ETA_SOURCE
    assert record.command_fingerprint == COMMAND_FINGERPRINT


def test_merge_inserts_only_training_admitted_positive_service_actions(tmp_path):
    base = tmp_path / "base.json"
    _base_cache(base)
    gates = _gate_files(tmp_path)
    reduced = _passing_gate_with_training_capacity_exclusion("jtl311linux")
    _write(gates["jtl311linux"], reduced)

    report = build_critical_gpu_lcb_cache_merge(
        base_cache_path=base,
        gate_paths=gates,
    )

    assert report["status"] == "PASS"
    assert report["expected_total_registered_rows"] == 27
    assert report["expected_total_inserted_rows"] == 26
    assert report["inserted_count"] == 26


def test_all_hardware_merge_preserves_explicit_jtl311_capacity_exclusion(tmp_path):
    available_cache = tmp_path / "available-cache.json"
    _base_cache(available_cache)
    available_bytes = available_cache.read_bytes()
    available_report = tmp_path / "available-report.json"
    _write(
        available_report,
        {
            "gate": "critical_gpu_available_lcb_cache_merge",
            "status": "PASS",
            "pass": True,
            "available_scope_cache_ready": True,
            "inserted_count": 27,
            "cache_output_sha256": hashlib.sha256(available_bytes).hexdigest(),
        },
    )
    jtl_gate = tmp_path / "jtl311-gate.json"
    _write(jtl_gate, _passing_gate_with_training_capacity_exclusion("jtl311linux"))

    report = build_critical_gpu_all_hardware_lcb_cache_merge(
        available_cache_path=available_cache,
        available_report_path=available_report,
        jtl311_gate_path=jtl_gate,
    )

    assert report["status"] == "PASS"
    assert report["expected_registered_count"] == 9
    assert report["expected_inserted_count"] == 8
    assert report["inserted_count"] == 8
    assert report["training_capacity_excluded_count"] == 1
    assert all(
        row["node_bucket"] == EFFECTIVE_NODE_BUCKET
        for row in report["inserted_rows"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda gate: gate.update({"measurement_protocol": "legacy"}), "measurement_protocol"),
        (lambda gate: gate.update({"hardware_class": "wrong"}), "hardware_class"),
        (
            lambda gate: gate["certificate"]["rows"][0].update(
                {"lower_service_units_per_s": 0.0}
            ),
            "lower_service_units_per_s",
        ),
        (
            lambda gate: gate["certificate"]["rows"][0].update(
                {"lower_service_valid_on_holdout": False}
            ),
            "lower_service_valid_on_holdout",
        ),
    ],
)
def test_merge_fails_closed_on_invalid_pass_gate(tmp_path, mutation, message):
    base = tmp_path / "base.json"
    _base_cache(base)
    gates = _gate_files(tmp_path)
    broken = _passing_gate("node007")
    mutation(broken)
    _write(gates["node007"], broken)

    report = build_critical_gpu_lcb_cache_merge(
        base_cache_path=base,
        gate_paths=gates,
    )

    assert report["pass"] is False
    assert "service_cache_snapshot" not in report
    assert message in json.dumps(report["validation_errors"])


def test_cli_wait_writes_report_but_never_final_cache(tmp_path):
    base = tmp_path / "base.json"
    _base_cache(base)
    report = tmp_path / "merge.json"
    markdown = tmp_path / "merge.md"
    cache = tmp_path / "final-cache.json"

    returncode = main(
        [
            "--base-cache",
            str(base),
            "--jtl110gpu-gate",
            str(tmp_path / "missing-jtl110gpu.json"),
            "--node007-gate",
            str(tmp_path / "missing-node007.json"),
            "--jtl311linux-gate",
            str(tmp_path / "missing-jtl311linux.json"),
            "--report-output",
            str(report),
            "--markdown-output",
            str(markdown),
            "--cache-output",
            str(cache),
        ]
    )

    assert returncode == 3
    assert report.is_file()
    assert markdown.is_file()
    assert not cache.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "WAIT_GPU_GATES"
    assert payload["cache_output_written"] is False
    assert "service_cache_snapshot" not in payload


def test_cli_pass_writes_complete_cache_snapshot(tmp_path):
    base = tmp_path / "base.json"
    _base_cache(base)
    gates = _gate_files(tmp_path)
    report = tmp_path / "merge.json"
    markdown = tmp_path / "merge.md"
    cache = tmp_path / "final-cache.json"

    returncode = main(
        [
            "--base-cache",
            str(base),
            "--jtl110gpu-gate",
            str(gates["jtl110gpu"]),
            "--node007-gate",
            str(gates["node007"]),
            "--jtl311linux-gate",
            str(gates["jtl311linux"]),
            "--report-output",
            str(report),
            "--markdown-output",
            str(markdown),
            "--cache-output",
            str(cache),
        ]
    )

    assert returncode == 0
    assert cache.is_file()
    disk_report = json.loads(report.read_text(encoding="utf-8"))
    assert disk_report["cache_output_written"] is True
    assert "service_cache_snapshot" not in disk_report
    loaded = ServiceRateCache.load(cache)
    assert loaded.lookup_statewise_exact(
        "hybrid_rl_resac_ant",
        workload_env="Ant-v2",
        node_bucket=NODE_SPECS["jtl311linux"].node_bucket,
        resource_state="empty",
        allocation_workers=1,
        colocation_count=5,
        resident_mix="",
    ).is_exact
