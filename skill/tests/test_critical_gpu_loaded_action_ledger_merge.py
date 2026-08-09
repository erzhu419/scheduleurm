from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from algorithm.experiments.critical_gpu_loaded_action_ledger_merge import (
    LOADED_SCOPE_NODES,
    _effective_node_bucket,
    build_critical_gpu_loaded_action_ledger_merge,
    main,
)
from algorithm.experiments.critical_gpu_loaded_completion_campaign import (
    PROTOCOL,
    SCENARIOS,
)
from algorithm.experiments.critical_gpu_completion_campaign import (
    NODE_SPECS,
    WORKLOAD_SPECS,
)
from simulation.service_cache import ProfileRecord, ServiceRateCache


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base(tmp_path: Path) -> tuple[Path, Path, ProfileRecord]:
    legacy = ProfileRecord(
        workload_key="legacy_control",
        command_fingerprint="legacy",
        resource_kind="gpu",
        node_bucket="",
        profile=3,
        unit="step",
        total_units=10.0,
        aggregate_rate=6.0,
        per_task_rates=(2.0, 2.0, 2.0),
        source="test",
    )
    cache_path = tmp_path / "base-cache.json"
    ServiceRateCache([legacy]).save(cache_path)
    report = {
        "gate": "critical_gpu_all_hardware_lcb_cache_merge",
        "status": "PASS",
        "pass": True,
        "all_four_gpu_nodes_ready": True,
        "legacy_index_unchanged": True,
        "cache_output_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
    }
    report_path = tmp_path / "base-report.json"
    _write(report_path, report)
    return cache_path, report_path, legacy


def _passing_gate(node: str) -> dict:
    workloads = {row.workload_key: row for row in WORKLOAD_SPECS}
    rows = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        resident = workloads[scenario.resident_workload]
        target = workloads[scenario.target_workload]
        target_units = float(target.max_iters)
        startup = 1.0 + 0.1 * index
        unit_s = 0.2 + 0.01 * index
        terminal = 0.5
        point = startup + target_units * unit_s + terminal
        target_lower = target_units / (point * 1.25)
        resident_lower = 2.0 + 0.1 * index
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "node": node,
                "node_bucket": NODE_SPECS[node].node_bucket,
                "hardware_class": NODE_SPECS[node].hardware_class,
                "resource_state": scenario.resource_state,
                "resident_mix": scenario.resident_mix,
                "resident_workload_key": resident.workload_key,
                "resident_workload_env": resident.workload_env,
                "target_workload_key": target.workload_key,
                "target_workload_env": target.workload_env,
                "resident_canonical_total_units": resident.max_iters,
                "target_canonical_total_units": target.max_iters,
                "resident_service_unit": resident.unit,
                "target_service_unit": target.unit,
                "profile": 2,
                "profile_axis": "total_tasks_per_gpu",
                "training_sample_count": 3,
                "calibration_wave_count": 9,
                "holdout_wave": 13,
                "target_completion_holdout_covered": True,
                "resident_service_holdout_covered": True,
                "target_lower_service_units_per_s": target_lower,
                "target_holdout_realized_service_units_per_s": target_lower * 1.1,
                "simultaneous_lower_resident_overlap_service_units_per_s": resident_lower,
                "holdout_actual_resident_overlap_service_units_per_s": resident_lower * 1.1,
                "lower_service_vector": {
                    resident.workload_key: resident_lower,
                    target.workload_key: target_lower,
                },
                "operational_pre_holdout_model": {
                    "sample_count": 12,
                    "target_phase": {
                        "startup_overhead_s": startup,
                        "completion_unit_s": unit_s,
                        "terminal_overhead_s": terminal,
                    },
                },
                "operational_target_completion_s": point,
                "target_point_relative_error": 0.03,
                "simultaneous_upper_target_completion_s": point * 1.25,
                "lower_service_bound_kind": (
                    "wave_max_joint_target_time_resident_service_split_conformal"
                ),
            }
        )
    return {
        "gate": "critical_gpu_loaded_stochastic_lcb_gate",
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "certificate_ready": True,
        "node": node,
        "node_bucket": NODE_SPECS[node].node_bucket,
        "hardware_class": NODE_SPECS[node].hardware_class,
        "resource_state": "mixed_colocation",
        "measurement_protocol": PROTOCOL,
        "expected_scenario_count": 4,
        "expected_wave_count": 13,
        "same_measurement_code_all_waves": True,
        "all_measurements_ready": True,
        "wait_reasons": [],
        "validation_errors": [],
        "certificate": {
            "constructed": True,
            "finite_sample_rank_ready": True,
            "all_holdout_bounds_valid": True,
            "simultaneous_ratio_margin": 0.25,
            "rows": rows,
        },
    }


def _gates(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for node in LOADED_SCOPE_NODES:
        path = tmp_path / f"{node}.json"
        _write(path, _passing_gate(node))
        result[node] = path
    return result


def test_merge_preserves_vector_actions_and_legacy_index(tmp_path):
    cache_path, report_path, legacy = _base(tmp_path)
    report = build_critical_gpu_loaded_action_ledger_merge(
        base_cache_path=cache_path,
        base_report_path=report_path,
        gate_paths=_gates(tmp_path),
    )

    assert report["status"] == "PASS"
    assert report["action_count"] == len(LOADED_SCOPE_NODES) * len(SCENARIOS)
    assert report["coordinate_record_count"] == 2 * len(LOADED_SCOPE_NODES) * len(SCENARIOS)
    assert report["legacy_index_unchanged"] is True
    cache = ServiceRateCache.from_snapshot(report["service_cache_snapshot"])
    assert cache.get("legacy_control", 3) == legacy
    assert cache.get("gpu_cnn_torch_resnet50", 2) is None

    action = next(
        row for row in report["actions"]
        if row["action_id"] == "gpu_loaded:node007:cnn_after_llm"
    )
    assert set(action["lower_service_vector"]) == {
        "gpu_llm_distilgpt2",
        "gpu_cnn_torch_resnet50",
    }
    for role, workload, env in (
        ("resident", "gpu_llm_distilgpt2", "distilgpt2_forward_fp16_b2_s128"),
        ("target", "gpu_cnn_torch_resnet50", "resnet50_synthetic_train_amp_b32_224"),
    ):
        exact = cache.lookup_statewise_exact(
            workload,
            workload_env=env,
            node_bucket=_effective_node_bucket("node007"),
            resource_state="mixed_colocation",
            allocation_workers=1,
            colocation_count=2,
            resident_mix="resident_llm_target_cnn",
        )
        assert exact.is_exact, role
    target = exact.record
    assert target is not None
    assert target.completion_model_ready is True
    assert target.completion_model_sample_count == 12


def test_merge_fails_closed_if_vector_does_not_match_scalar_lcb(tmp_path):
    cache_path, report_path, _ = _base(tmp_path)
    gates = _gates(tmp_path)
    broken = _passing_gate("node007")
    broken["certificate"]["rows"][0]["lower_service_vector"][
        broken["certificate"]["rows"][0]["target_workload_key"]
    ] *= 2.0
    _write(gates["node007"], broken)

    report = build_critical_gpu_loaded_action_ledger_merge(
        base_cache_path=cache_path,
        base_report_path=report_path,
        gate_paths=gates,
    )

    assert report["status"] == "FAIL_LOADED_GATES"
    assert report["pass"] is False
    assert "vector mismatch" in json.dumps(report["validation_errors"])
    assert "service_cache_snapshot" not in report


def test_cli_wait_never_writes_cache_or_ledger(tmp_path):
    cache_path, report_path, _ = _base(tmp_path)
    output_report = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    output_ledger = tmp_path / "ledger.json"
    output_cache = tmp_path / "cache.json"

    returncode = main(
        [
            "--base-cache", str(cache_path),
            "--base-report", str(report_path),
            "--jtl110gpu-gate", str(tmp_path / "missing-jtl110gpu.json"),
            "--jtl110gpu2-gate", str(tmp_path / "missing-jtl110gpu2.json"),
            "--node007-gate", str(tmp_path / "missing-node007.json"),
            "--jtl311linux-gate", str(tmp_path / "missing-jtl311.json"),
            "--report-output", str(output_report),
            "--markdown-output", str(output_md),
            "--ledger-output", str(output_ledger),
            "--cache-output", str(output_cache),
        ]
    )

    assert returncode == 3
    assert output_report.is_file()
    assert output_md.is_file()
    assert not output_ledger.exists()
    assert not output_cache.exists()


def test_cli_pass_writes_both_atomic_products(tmp_path):
    cache_path, report_path, _ = _base(tmp_path)
    gates = _gates(tmp_path)
    output_report = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    output_ledger = tmp_path / "ledger.json"
    output_cache = tmp_path / "cache.json"

    returncode = main(
        [
            "--base-cache", str(cache_path),
            "--base-report", str(report_path),
            "--jtl110gpu-gate", str(gates["jtl110gpu"]),
            "--jtl110gpu2-gate", str(gates["jtl110gpu2"]),
            "--node007-gate", str(gates["node007"]),
            "--jtl311linux-gate", str(gates["jtl311linux"]),
            "--report-output", str(output_report),
            "--markdown-output", str(output_md),
            "--ledger-output", str(output_ledger),
            "--cache-output", str(output_cache),
        ]
    )

    assert returncode == 0
    assert output_ledger.is_file()
    assert output_cache.is_file()
    disk = json.loads(output_report.read_text(encoding="utf-8"))
    assert disk["cache_output_written"] is True
    assert disk["ledger_output_written"] is True
    assert "service_cache_snapshot" not in disk
    assert "action_ledger" not in disk


@pytest.mark.parametrize("node", LOADED_SCOPE_NODES)
def test_effective_node_bucket_is_physical(node):
    assert _effective_node_bucket(node).endswith(f":{node}")
