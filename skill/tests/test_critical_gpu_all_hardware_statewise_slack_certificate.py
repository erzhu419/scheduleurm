from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from algorithm.experiments.critical_gpu_all_hardware_statewise_slack_certificate import (
    EXPECTED_NODES,
    build_critical_gpu_all_hardware_statewise_slack_certificate,
)
from algorithm.experiments.critical_gpu_completion_campaign import (
    NODE_SPECS,
    campaign_cells,
)
from simulation.service_cache import ProfileRecord, ServiceRateCache


EXECUTION_CLASSES = {
    "jtl110gpu": "gpu_3080ti_12gb_dual:jtl110gpu",
    "jtl110gpu2": "gpu_3080ti_12gb_dual:jtl110gpu2",
    "node007": "gpu_2080ti_11gb_quad:node007",
    "jtl311linux": "gpu_2080_8gb_dual:jtl311linux",
}


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(tmp_path: Path, *, exclude_jtl_profile: bool = True) -> tuple[Path, Path, dict[str, Path]]:
    cache = ServiceRateCache()
    gate_paths: dict[str, Path] = {}
    gate_hashes: dict[str, str] = {}
    for node in EXPECTED_NODES:
        cells = list(campaign_cells(node=node, wave=1))
        excluded = []
        if node == "jtl311linux" and exclude_jtl_profile:
            excluded = [cells.pop(1)]
        rows = []
        for index, cell in enumerate(cells, start=1):
            row = {
                "workload_key": cell["workload_key"],
                "workload_env": cell["workload_env"],
                "profile": int(cell["profile"]),
            }
            rows.append(row)
            profile = int(cell["profile"])
            cache.add(
                ProfileRecord(
                    workload_key=str(cell["workload_key"]),
                    command_fingerprint="test",
                    resource_kind="gpu",
                    node_bucket=EXECUTION_CLASSES[node],
                    profile=profile,
                    unit="iter" if "resac" in str(cell["workload_key"]) else "step",
                    total_units=100.0,
                    aggregate_rate=float(index + profile),
                    per_task_rates=tuple(float(index + profile) / profile for _ in range(profile)),
                    source="synthetic-test-gate",
                    gpu_count_observed=len(NODE_SPECS[node].gpus),
                    workload_env=str(cell["workload_env"]),
                    resource_state="empty",
                    eta_source="task_native_tqdm_natural_completion_split_conformal_lcb",
                    stable_rate_ready=True,
                    hardware_class=NODE_SPECS[node].hardware_class,
                    completion_model_ready=True,
                    completion_model_sample_count=12,
                    completion_unit_s=0.1,
                    completion_total_wall_s=10.0,
                    completion_group_total_units=100.0 * profile,
                    allocation_workers=1,
                    colocation_count=profile,
                ),
                force_replace=True,
            )
        gate = {
            "gate": (
                "critical_gpu_p10_transport_correction_gate"
                if node == "node007"
                else "critical_gpu_stochastic_lcb_gate"
            ),
            "schema_version": 1,
            "status": "PASS",
            "pass": True,
            "certificate_ready": True,
            "node": node,
            "miscoverage_alpha": 0.10,
            "certificate": {"rows": rows},
            "training_capacity_excluded_cells": excluded,
        }
        if node == "node007":
            gate.update(
                {
                    "correction_measurement_protocol": (
                        "critical_gpu_p10_transport_correction_v2"
                    ),
                    "correction_scope": (
                        "pre_registered_measurement_transport_integrity"
                    ),
                    "performance_conditioned_selection": False,
                    "all_measurements_ready": True,
                    "same_correction_measurement_code_all_waves": True,
                    "same_source_measurement_code_all_waves": True,
                }
            )
        path = tmp_path / f"{node}_gate.json"
        gate_paths[node] = path
        gate_hashes[node] = _write_json(path, gate)

    cache_path = tmp_path / "cache.json"
    cache.save(cache_path)
    available_report = {
        "gate": "critical_gpu_available_lcb_cache_merge",
        "status": "PASS",
        "pass": True,
        "node_to_operational_execution_class": {
            node: EXECUTION_CLASSES[node]
            for node in ("jtl110gpu", "jtl110gpu2", "node007")
        },
        "source_gate_audits": [
            {
                "node": node,
                "path": str(gate_paths[node]),
                "sha256": gate_hashes[node],
            }
            for node in ("jtl110gpu", "jtl110gpu2", "node007")
        ],
    }
    available_path = tmp_path / "available_report.json"
    available_hash = _write_json(available_path, available_report)
    merge = {
        "gate": "critical_gpu_all_hardware_lcb_cache_merge",
        "status": "PASS",
        "pass": True,
        "all_four_gpu_nodes_ready": True,
        "cache_output_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "available_report_path": str(available_path),
        "available_report_sha256": available_hash,
        "new_physical_node": "jtl311linux",
        "new_operational_execution_class": EXECUTION_CLASSES["jtl311linux"],
        "jtl311_gate_path": str(gate_paths["jtl311linux"]),
        "jtl311_gate_sha256": gate_hashes["jtl311linux"],
    }
    merge_path = tmp_path / "all_hardware_report.json"
    _write_json(merge_path, merge)
    return cache_path, merge_path, gate_paths


def test_all_hardware_slack_passes_with_accounted_capacity_exclusion(tmp_path: Path) -> None:
    cache_path, merge_path, _ = _sources(tmp_path)

    report = build_critical_gpu_all_hardware_statewise_slack_certificate(
        cache_path=cache_path,
        merge_report_path=merge_path,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert len(report["nodes"]) == 4
    assert report["minimum_eta"] == pytest.approx(0.05)
    assert report["simultaneous_four_node_union_bound_coverage"] == pytest.approx(0.6)
    assert all(row["class_count"] == 4 for row in report["nodes"])
    assert all(row["nominal_lcb_coverage"] == 0.9 for row in report["nodes"])


def test_all_hardware_slack_rejects_source_gate_hash_drift(tmp_path: Path) -> None:
    cache_path, merge_path, gate_paths = _sources(tmp_path)
    payload = json.loads(gate_paths["node007"].read_text(encoding="utf-8"))
    payload["miscoverage_alpha"] = 0.2
    _write_json(gate_paths["node007"], payload)

    report = build_critical_gpu_all_hardware_statewise_slack_certificate(
        cache_path=cache_path,
        merge_report_path=merge_path,
    )

    assert report["status"] == "FAIL_SOURCE"
    assert report["pass"] is False
    assert "source gate hash mismatch" in report["errors"][0]["detail"]


def test_all_hardware_slack_waits_without_final_cache(tmp_path: Path) -> None:
    report = build_critical_gpu_all_hardware_statewise_slack_certificate(
        cache_path=tmp_path / "missing-cache.json",
        merge_report_path=tmp_path / "missing-report.json",
    )

    assert report["status"] == "WAIT_SOURCES"
    assert report["pass"] is False
    assert report["wait_reasons"][0]["code"] == "MISSING_SOURCE"
