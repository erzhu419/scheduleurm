from __future__ import annotations

import hashlib
import json
from pathlib import Path

from algorithm.experiments.critical_gpu_loaded_completion_campaign import (
    ALL_WAVES,
    CALIBRATION_WAVES,
    CAMPAIGN_PREFIX,
    HOLDOUT_WAVE,
    PROTOCOL,
    SCENARIOS,
    TRAINING_WAVES,
    _split_role,
    measurement_code_manifest,
)
from algorithm.experiments.critical_gpu_loaded_stochastic_lcb_gate import (
    MAX_UNAPPROVED_USER_CPU_EXPOSURE_FRACTION,
    _conservative_unapproved_cpu_exposure,
    _expected_scenarios,
    build_critical_gpu_loaded_stochastic_lcb_gate,
    discover_campaign_paths,
)


NODE = "node007"
CODE_HASH = "a" * 64


def test_isolated_cpu_snapshot_uses_conservative_exposure_not_allowlist():
    target_start_ns = 0
    target_end_ns = 665_972_965_002
    violation_ns = 574_091_370_691
    samples = [
        {"sample_ns": violation_ns - 5_663_808_024},
        {"sample_ns": violation_ns},
        {"sample_ns": violation_ns + 5_667_101_176},
    ]
    result = _conservative_unapproved_cpu_exposure(
        host_process_samples=samples,
        high_snapshot_totals=[
            {
                "sample_ns": violation_ns,
                "total_unapproved_cpu_fraction": 0.05670833333333333,
            }
        ],
        target_start_ns=target_start_ns,
        target_end_ns=target_end_ns,
    )

    assert len(result["bursts"]) == 1
    assert result["bursts"][0]["high_snapshot_count"] == 1
    assert (
        result["conservative_exposure_fraction"]
        < MAX_UNAPPROVED_USER_CPU_EXPOSURE_FRACTION
    )


def test_loaded_gate_constructs_joint_holdout_certificate(tmp_path):
    paths = _write_campaigns(tmp_path)

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["ready_observation_count"] == 13 * 4
    certificate = report["certificate"]
    assert certificate["conformal_rank"] == 9
    assert certificate["simultaneous_scope"]["joint_functional_count_per_wave"] == 8
    assert certificate["all_holdout_bounds_valid"] is True
    assert len(certificate["rows"]) == 4
    assert all(
        row["target_completion_holdout_covered"]
        and row["resident_service_holdout_covered"]
        and len(row["lower_service_vector"]) == 2
        for row in certificate["rows"]
    )


def test_expected_loaded_units_follow_node_specific_preregistration():
    expected = _expected_scenarios("jtl110gpu2")

    assert expected["rl_after_cnn"]["resident_measurement_total_units"] == 30_000
    assert expected["cnn_after_llm"]["resident_measurement_total_units"] == 12_000


def test_loaded_gate_fails_closed_on_partial_overlap(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[HOLDOUT_WAVE].read_text(encoding="utf-8"))
    payload["rows"][0]["resident_alive_at_target_end"] = False
    paths[HOLDOUT_WAVE].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "OVERLAP_INVALID"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_rejects_undeclared_prelaunch_gpu_load(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    diagnostics = Path(row["diagnostics_log_path"])
    lines = diagnostics.read_text(encoding="utf-8").splitlines()
    lines[0] = "0, NVIDIA GeForce RTX 2080 Ti, 11264, 2048, 9000, 100"
    diagnostics.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["diagnostics_sha256"] = hashlib.sha256(diagnostics.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "ASSIGNED_GPU_NOT_IDLE_AT_RESIDENT_START"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_rejects_undeclared_load_on_other_registered_gpu(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    assert row["gpu"] == 0
    diagnostics = Path(row["diagnostics_log_path"])
    lines = diagnostics.read_text(encoding="utf-8").splitlines()
    lines[1] = "1, NVIDIA GeForce RTX 2080 Ti, 11264, 2048, 9000, 100"
    diagnostics.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["diagnostics_sha256"] = hashlib.sha256(diagnostics.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "UNDECLARED_NODE_GPU_LOAD_AT_RESIDENT_START"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_rejects_transient_cross_gpu_load_during_target(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    monitor = Path(row["gpu_monitor_log_path"])
    lines = monitor.read_text(encoding="utf-8").splitlines()
    second_marker = [
        index for index, line in enumerate(lines) if line.startswith("__GPU_SAMPLE_NS__")
    ][1]
    gpu_one = next(
        index
        for index in range(second_marker + 1, len(lines))
        if lines[index].startswith("1,")
    )
    lines[gpu_one] = "1, NVIDIA GeForce RTX 2080 Ti, 11264, 2048, 9000, 100"
    monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "UNDECLARED_CROSS_GPU_LOAD_DURING_TARGET"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_rejects_third_compute_process_on_assigned_gpu(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    monitor = Path(row["gpu_monitor_log_path"])
    lines = monitor.read_text(encoding="utf-8").splitlines()
    lines.append("__GPU_SAMPLE_NS__ 105000000000")
    lines.extend(
        f"{gpu}, NVIDIA GeForce RTX 2080 Ti, 11264, 1, 11010, 0"
        for gpu in range(4)
    )
    lines.extend(
        (
            "__GPU_PROC__ 2001 1001 GPU-a 1024",
            "__GPU_PROC__ 2002 1002 GPU-a 1024",
            "__GPU_PROC__ 2999 1999 GPU-a 256",
        )
    )
    monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "UNDECLARED_GPU_COMPUTE_PROCESS_DURING_TARGET"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_resolves_transient_missing_pgid_for_known_controlled_pid(tmp_path):
    paths = _write_campaigns(tmp_path)
    for path in paths.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][1]
        monitor = Path(row["gpu_monitor_log_path"])
        lines = monitor.read_text(encoding="utf-8").splitlines()
        process_lines = [
            index
            for index, line in enumerate(lines)
            if line == "__GPU_PROC__ 2002 1002 GPU-a 1024"
        ]
        lines[process_lines[1]] = "__GPU_PROC__ 2002 -1 GPU-a 1024"
        monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
        row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE, campaign_paths=paths
    )

    assert report["status"] == "PASS"


def test_loaded_gate_rejects_missing_pgid_for_unseen_process(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    monitor = Path(row["gpu_monitor_log_path"])
    lines = monitor.read_text(encoding="utf-8").splitlines()
    second_marker = [
        index for index, line in enumerate(lines) if line.startswith("__GPU_SAMPLE_NS__")
    ][1]
    lines.insert(second_marker + 1, "__GPU_PROC__ 2999 -1 GPU-a 256")
    monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE, campaign_paths=paths
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "UNDECLARED_GPU_COMPUTE_PROCESS_DURING_TARGET"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_rejects_high_aggregate_host_load(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    monitor = Path(row["gpu_monitor_log_path"])
    lines = monitor.read_text(encoding="utf-8").splitlines()
    host_lines = [index for index, line in enumerate(lines) if line.startswith("__HOST_SAMPLE__")]
    lines[host_lines[1]] = "__HOST_SAMPLE__ 64 40.0 2.0 1.0 1048576 2097152"
    monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE, campaign_paths=paths
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "UNDECLARED_HOST_LOAD_DURING_TARGET"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_rejects_low_available_host_memory(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    monitor = Path(row["gpu_monitor_log_path"])
    lines = monitor.read_text(encoding="utf-8").splitlines()
    host_lines = [index for index, line in enumerate(lines) if line.startswith("__HOST_SAMPLE__")]
    lines[host_lines[1]] = "__HOST_SAMPLE__ 64 2.0 2.0 1.0 1024 2097152"
    monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE, campaign_paths=paths
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "INSUFFICIENT_HOST_MEMORY_DURING_TARGET"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_rejects_third_high_cpu_same_user_process(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    monitor = Path(row["gpu_monitor_log_path"])
    lines = monitor.read_text(encoding="utf-8").splitlines()
    marker = [
        index for index, line in enumerate(lines)
        if line.startswith("__USER_PROC_SNAPSHOT__")
    ][1]
    lines.insert(marker + 1, "__USER_PROC__ 2999 1999 400.0 4096 python")
    monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE, campaign_paths=paths
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "UNDECLARED_HOST_PROCESS_DURING_TARGET"
        for issue in report["validation_errors"]
    )


def test_loaded_gate_normalizes_small_background_process_by_host_capacity(tmp_path):
    paths = _write_campaigns(tmp_path)
    for path in paths.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][1]
        monitor = Path(row["gpu_monitor_log_path"])
        lines = monitor.read_text(encoding="utf-8").splitlines()
        marker = [
            index for index, line in enumerate(lines)
            if line.startswith("__USER_PROC_SNAPSHOT__")
        ][1]
        lines.insert(marker + 1, "__USER_PROC__ 2999 1999 81.2 4096 snap")
        monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
        row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE, campaign_paths=paths
    )

    assert report["status"] == "PASS"


def test_loaded_gate_rejects_missing_host_sample(tmp_path):
    paths = _write_campaigns(tmp_path)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    row = payload["rows"][1]
    monitor = Path(row["gpu_monitor_log_path"])
    lines = monitor.read_text(encoding="utf-8").splitlines()
    host_lines = [index for index, line in enumerate(lines) if line.startswith("__HOST_SAMPLE__")]
    del lines[host_lines[1]]
    monitor.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row["gpu_monitor_sha256"] = hashlib.sha256(monitor.read_bytes()).hexdigest()
    paths[5].write_text(json.dumps(payload), encoding="utf-8")

    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE, campaign_paths=paths
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "HOST_MONITOR_TARGET_INTERVAL_NOT_COVERED"
        for issue in report["validation_errors"]
    )


def test_discovery_excludes_filtered_smoke_artifacts(tmp_path):
    code_prefix = measurement_code_manifest()["sha256"][:12]
    full = tmp_path / (
        f"{CAMPAIGN_PREFIX}_node007_r01_20260809_code{code_prefix}.json"
    )
    smoke = tmp_path / (
        f"{CAMPAIGN_PREFIX}_node007_r01_20260809_selabc_code{code_prefix}.json"
    )
    full.write_text("{}", encoding="utf-8")
    smoke.write_text("{}", encoding="utf-8")

    assert discover_campaign_paths(node=NODE, artifact_root=tmp_path) == {1: full}


def _write_campaigns(root: Path) -> dict[int, Path]:
    paths = {}
    for wave in ALL_WAVES:
        path = root / f"wave-{wave}.json"
        path.write_text(
            json.dumps(_campaign(wave, root=root), sort_keys=True),
            encoding="utf-8",
        )
        paths[wave] = path
    return paths


def _campaign(wave: int, *, root: Path) -> dict[str, object]:
    return {
        "gate": "critical_gpu_loaded_completion_campaign",
        "schema_version": 1,
        "protocol": PROTOCOL,
        "campaign_id": f"test-{wave}",
        "node": NODE,
        "node_bucket": "gpu_2080ti_11gb_quad",
        "hardware_class": "NVIDIA_RTX_2080Ti_11GB",
        "wave": wave,
        "split_role": _split_role(wave),
        "allow_launch": True,
        "legacy_scheduler_limits_bypassed": True,
        "natural_completion_required": True,
        "task_native_progress_required": True,
        "full_resident_target_overlap_required": True,
        "terminate_on_stable": False,
        "algorithm_path": "controlled_theorem_measurement_direct",
        "ordinary_running_tasks_touched": False,
        "pre_registered_split": {
            "training": list(TRAINING_WAVES),
            "calibration": list(CALIBRATION_WAVES),
            "holdout": HOLDOUT_WAVE,
        },
        "measurement_code_manifest": {"sha256": CODE_HASH, "files": []},
        "final_measurement_code_manifest": {"sha256": CODE_HASH, "files": []},
        "measurement_code_unchanged": True,
        "selection": {"scenario_ids": [], "filtered": False},
        "rows": [
            _row(spec, wave=wave, index=index, root=root)
            for index, spec in enumerate(SCENARIOS)
        ],
        "ready_row_count": 4,
        "capacity_boundary_count": 0,
        "status": "PASS",
        "pass": True,
    }


def _row(spec, *, wave: int, index: int, root: Path) -> dict[str, object]:
    expected = _expected_scenarios(NODE)[spec.scenario_id]
    factor = _factor(wave, index)
    resident_unit_s = (0.015 + index * 0.002) * factor
    target_unit_s = (0.020 + index * 0.003) * factor
    resident_model = _model(
        units=spec.resident_total_units,
        unit=expected["resident_service_unit"],
        unit_s=resident_unit_s,
    )
    target_model = _model(
        units=spec.target_total_units,
        unit=expected["target_service_unit"],
        unit_s=target_unit_s,
    )
    resident_rate = (50.0 + index * 5.0) / factor
    diagnostics = root / f"wave-{wave}-{spec.scenario_id}.diagnostics.log"
    snapshot = "\n".join(
        f"{gpu}, NVIDIA GeForce RTX 2080 Ti, 11264, 1, 11010, 0"
        for gpu in range(4)
    )
    diagnostics.write_text(f"{snapshot}\n{snapshot}\n", encoding="utf-8")
    target_start_ns = 100_000_000_000
    target_end_ns = 110_000_000_000
    monitor = root / f"wave-{wave}-{spec.scenario_id}.gpu-monitor.log"
    process_snapshot = "\n".join(
        (
            "__GPU_PROC__ 2001 1001 GPU-a 1024",
            "__GPU_PROC__ 2002 1002 GPU-a 1024",
            "__USER_PROC_SNAPSHOT__ 0",
            "__USER_PROC__ 2001 1001 80.0 1024 python",
            "__USER_PROC__ 2002 1002 80.0 1024 python",
        )
    )
    host_snapshot = "__HOST_SAMPLE__ 64 2.0 1.5 1.0 1048576 2097152"
    monitor.write_text(
        "\n".join(
            (
                "__GPU_SAMPLE_NS__ 99000000000",
                host_snapshot,
                snapshot,
                process_snapshot,
                "__GPU_SAMPLE_NS__ 105000000000",
                host_snapshot,
                snapshot,
                process_snapshot,
                "__GPU_SAMPLE_NS__ 111000000000",
                host_snapshot,
                snapshot,
                process_snapshot,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "scenario_id": spec.scenario_id,
        "node": NODE,
        "node_bucket": expected["node_bucket"],
        "hardware_class": expected["hardware_class"],
        "gpu": 0,
        "diagnostics_log_path": str(diagnostics),
        "diagnostics_sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
        "gpu_monitor_log_path": str(monitor),
        "gpu_monitor_sha256": hashlib.sha256(monitor.read_bytes()).hexdigest(),
        "resident_process_group_id": 1001,
        "target_process_group_id": 1002,
        "target_start_ns": target_start_ns,
        "target_end_ns": target_end_ns,
        "wave": wave,
        "split_role": _split_role(wave),
        "resource_state": expected["resource_state"],
        "resident_mix": expected["resident_mix"],
        "resident_workload_key": expected["resident_workload_key"],
        "resident_workload_env": expected["resident_workload_env"],
        "target_workload_key": expected["target_workload_key"],
        "target_workload_env": expected["target_workload_env"],
        "post_colocation_count": 2,
        "resident_total_units": spec.resident_total_units,
        "target_total_units": spec.target_total_units,
        "resident_completion_model": resident_model,
        "target_completion_model": target_model,
        "resident_stable_rate": resident_rate,
        "target_stable_rate": 1.0 / target_unit_s,
        "resident_overlap_service": {
            "ready": True,
            "start_epoch_s": 100.0,
            "end_epoch_s": 110.0,
            "duration_s": 10.0,
            "progress_observation_count": 4,
            "completed_units_lower": int(resident_rate * 10.0),
            "conservative_rate_units_per_s": resident_rate,
            "reported_rate_median_units_per_s": resident_rate * 1.01,
            "unit": expected["resident_service_unit"],
        },
        "resident_alive_at_target_start": True,
        "resident_alive_at_target_end": True,
        "resident_ready_observations": 4,
        "resident_returncode": 0,
        "target_returncode": 0,
        "remote_returncode": 0,
        "ready": True,
        "capacity_boundary": False,
        "reason": "",
        "eta_source": "task_native_tqdm_natural_completion_loaded_trajectory",
        "legacy_scheduler_limits_bypassed": True,
        "ordinary_running_tasks_touched": False,
    }


def _model(*, units: int, unit: str, unit_s: float) -> dict[str, object]:
    startup = 2.0
    terminal = 1.0
    total = startup + units * unit_s + terminal
    return {
        "completion_model_ready": True,
        "natural_exit": True,
        "stopped_on_stable": False,
        "child_returncode": 0,
        "total_units": units,
        "unit": unit,
        "startup_overhead_s": startup,
        "completion_unit_s": unit_s,
        "terminal_overhead_s": terminal,
        "total_wall_s": total,
    }


def _factor(wave: int, index: int) -> float:
    if wave in TRAINING_WAVES:
        return (1.00, 1.02, 0.98)[wave - 1]
    if wave in CALIBRATION_WAVES:
        return 1.0 + 0.005 * (wave - 3) + 0.001 * index
    return 1.035 + 0.001 * index
