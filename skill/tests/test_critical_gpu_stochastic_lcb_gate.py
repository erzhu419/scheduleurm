from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from algorithm.experiments.critical_gpu_completion_campaign import (
    CALIBRATION_WAVES,
    CAMPAIGN_PREFIX,
    HOLDOUT_WAVE,
    NODE_SPECS,
    PROTOCOL,
    STAGED_TRAJECTORY_EVIDENCE,
    TRAINING_WAVES,
    WORKLOAD_SPECS,
    campaign_cells,
    wave_role,
)
from algorithm.experiments.critical_gpu_stochastic_lcb_gate import (
    EXPECTED_WAVES,
    build_critical_gpu_stochastic_lcb_gate,
    default_campaign_paths,
    main,
)


NODE = "jtl110gpu"
CODE_SHA256 = "a" * 64


def _write_campaigns(tmp_path: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for wave in EXPECTED_WAVES:
        path = (
            tmp_path
            / f"{CAMPAIGN_PREFIX}_{NODE}_r{wave:02d}_20260803.json"
        )
        path.write_text(
            json.dumps(_campaign(wave), sort_keys=True),
            encoding="utf-8",
        )
        paths[wave] = path
    return paths


def _campaign(wave: int) -> dict[str, object]:
    planned = campaign_cells(node=NODE, wave=wave)
    rows = [
        _measurement_row(cell, wave=wave, index=index)
        for index, cell in enumerate(planned)
    ]
    return {
        "gate": "critical_gpu_completion_campaign",
        "measurement_protocol": PROTOCOL,
        "campaign_id": f"{CAMPAIGN_PREFIX}_{NODE}_r{wave:02d}_20260803",
        "node": NODE,
        "node_spec": asdict(NODE_SPECS[NODE]),
        "wave": wave,
        "split_role": wave_role(wave),
        "pre_registered_split": {
            "training": list(TRAINING_WAVES),
            "calibration": list(CALIBRATION_WAVES),
            "holdout": HOLDOUT_WAVE,
            "smoke_excluded": 0,
        },
        "legacy_scheduler_limits_bypassed": True,
        "algorithm_path": "controlled_theorem_measurement_direct",
        "terminate_on_stable": False,
        "natural_completion_required": True,
        "task_native_progress_required": True,
        "coordinated_post_warmup_start_required_for_builtin_gpu_workloads": True,
        "admission_delay_included_in_completion_jct": True,
        "staged_trajectory_completion_evidence_required": True,
        "real_checkpoint_allocation_required": True,
        "measurement_code_manifest": {"sha256": CODE_SHA256, "files": []},
        "final_measurement_code_manifest": {"sha256": CODE_SHA256, "files": []},
        "measurement_code_unchanged": True,
        "selection": {
            "workload_keys": [],
            "profiles": [],
            "filtered": False,
        },
        "planned_cells": planned,
        "rows": rows,
        "ready_row_count": 9,
        "status": "PASS",
        "pass": True,
    }


def _measurement_row(
    cell: dict[str, object],
    *,
    wave: int,
    index: int,
) -> dict[str, object]:
    spec = next(
        item for item in WORKLOAD_SPECS
        if item.workload_key == cell["workload_key"]
    )
    factor = _wave_factor(wave, index)
    task_count = int(cell["total_task_count"])
    children = [
        _child(
            task=index_in_cell,
            factor=factor,
            total_units=spec.max_iters,
            unit=spec.unit,
            min_progress=spec.min_rate_samples,
            is_rl=spec.workload_key == "hybrid_rl_resac_ant",
        )
        for index_in_cell in range(task_count)
    ]
    checkpoint_files = [
        {
            "path": f"/tmp/run/checkpoints/task_{task}/checkpoint.bin",
            "size_bytes": 4096,
            "allocated_blocks_512": 8,
        }
        for task in range(task_count)
    ]
    profile = int(cell["profile"])
    gpus = list(cell["gpus"])
    per_gpu = {str(gpu): profile for gpu in gpus}
    evidence_mode = str(cell["service_evidence_mode"])
    trajectory_mode = evidence_mode == STAGED_TRAJECTORY_EVIDENCE
    progress_observations = (
        int(spec.max_iters) if trajectory_mode else int(spec.min_rate_samples) + 2
    )
    interval_samples = (
        max(1, int(spec.max_iters) - 1)
        if trajectory_mode
        else int(spec.min_rate_samples) + 1
    )
    for child in children:
        child["completion_model"]["progress_observation_count"] = progress_observations
        child["completion_model"]["interval_sample_count"] = interval_samples
    summary = {
        "probe": "remote_workload_selected_profile_probe",
        "node": NODE,
        "profile": profile,
        "gpus": gpus,
        "measurement_valid": True,
        "all_stable_rate_ready": True,
        "all_completion_models_ready": True,
        "require_stable_rate": not trajectory_mode,
        "coordinated_profile_launch": True,
        "terminate_on_stable": False,
        "capacity_boundary": False,
        "placement_valid": True,
        "running_count": task_count,
        "running_with_rate_count": task_count,
        "stable_rate_ready_count": task_count,
        "completion_model_ready_count": task_count,
        "returncode_accepted_count": task_count,
        "per_gpu_running": per_gpu,
        "expected_per_gpu_running": per_gpu,
        "rows": children,
    }
    return {
        **cell,
        "measurement_protocol": PROTOCOL,
        "preflight": {
            "ready": True,
            "returncode": 0,
            "compute_process_rows": [],
            "selected_gpus": [
                {
                    "index": gpu,
                    "name": "NVIDIA RTX 3080 Ti",
                    "total_mb": 12288,
                    "used_mb": 16,
                    "util_pct": 0,
                }
                for gpu in gpus
            ],
        },
        "probe_result": {
            "pass": True,
            "stable_rate_ready_count": task_count,
            "completion_model_ready_count": task_count,
        },
        "summary": summary,
        "artifact_audit": {
            "ready": True,
            "returncode": 0,
            "allocated_file_count": task_count,
            "total_allocated_bytes": task_count * 4096,
            "files": checkpoint_files,
        },
        "backend_audit": {
            "ready": True,
            "expected_task_count": task_count,
            "accepted_task_count": task_count,
        },
        "completion_audit": {
            "ready": True,
            "task_count": task_count,
            "all_natural_exit": True,
            "all_completion_models_ready": True,
            "checkpoint_or_finalization_observed": True,
        },
        "coordination_audit": {
            "ready": True,
            "required": spec.coordinated_start_barrier,
            "expected_task_count": task_count,
            "observed_task_count": task_count,
            "start_marker_count": task_count if spec.coordinated_start_barrier else 0,
            "end_marker_count": task_count if spec.coordinated_start_barrier else 0,
        },
        "admission_audit": {
            "ready": True,
            "delay_counted_in_startup_and_jct": True,
            "children": [{"ready": True} for _ in range(task_count)],
        },
        "trajectory_progress_audit": _trajectory_audit(
            evidence_mode=evidence_mode,
            task_count=task_count,
            total_units=int(spec.max_iters),
            cycle_units=int(spec.stable_cycle_units),
        ),
        "service_evidence_mode": evidence_mode,
        "service_evidence_ready": True,
        "measurement_code_sha256": CODE_SHA256,
        "measurement_code_identity_ready": True,
        "checkpoint_allocation_ready": True,
        "capacity_boundary": False,
        "status": "READY",
        "ready": True,
    }


def _trajectory_audit(
    *,
    evidence_mode: str,
    task_count: int,
    total_units: int,
    cycle_units: int,
) -> dict[str, object]:
    required = evidence_mode == STAGED_TRAJECTORY_EVIDENCE
    if not required:
        return {
            "ready": True,
            "required": False,
            "service_evidence_mode": evidence_mode,
        }
    intervals = max(1, total_units - 1)
    cycles = total_units // max(1, cycle_units)
    return {
        "ready": True,
        "required": True,
        "service_evidence_mode": evidence_mode,
        "expected_task_count": task_count,
        "observed_task_count": task_count,
        "expected_progress_observations_per_task": total_units,
        "expected_interval_samples_per_task": intervals,
        "cycle_units": cycle_units,
        "expected_complete_cycles_per_task": cycles,
        "completion_trajectory_includes_admission_and_drain": True,
        "children": [
            {
                "global_index": task,
                "progress_observation_count": total_units,
                "interval_sample_count": intervals,
                "complete_cycle_count": cycles,
                "stable_rate_ready_diagnostic": True,
                "ready": True,
            }
            for task in range(task_count)
        ],
    }


def _child(
    *,
    task: int,
    factor: float,
    total_units: int,
    unit: str,
    min_progress: int,
    is_rl: bool,
) -> dict[str, object]:
    startup = (2.0 + 0.01 * task) * factor
    completion_unit = 1.0 * factor
    terminal = 1.0 * factor
    total_wall = startup + total_units * completion_unit + terminal
    model = {
        "completion_model_ready": True,
        "child_returncode": 0,
        "natural_exit": True,
        "stopped_on_stable": False,
        "readiness_reason": "ready",
        "unit": unit,
        "progress_observation_count": min_progress + 2,
        "interval_sample_count": min_progress + 1,
        "total_units": total_units,
        "startup_overhead_s": startup,
        "completion_unit_s": completion_unit,
        "terminal_overhead_s": terminal,
        "total_wall_s": total_wall,
        "checkpoint_observed_s": 0.0 if is_rl else 0.2 * factor,
        "save_observed_s": 0.0 if is_rl else 0.1 * factor,
        "finalization_after_last_progress_s": terminal,
    }
    return {
        "global_index": task,
        "run_name": f"task_{task}",
        "returncode": 0,
        "stable_rate_ready": True,
        "stable_rate": 1.0 / completion_unit,
        "completion_model_ready": True,
        "unit": unit,
        "completion_model": model,
    }


def _wave_factor(wave: int, cell_index: int) -> float:
    if wave in TRAINING_WAVES:
        return 1.0
    if wave in CALIBRATION_WAVES:
        if wave == 12 and cell_index == 8:
            return 1.12
        return 1.0 + 0.01 * (wave - 3)
    return 1.10


def _mutate(path: Path, callback) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_complete_campaign_builds_wave_max_simultaneous_certificate(tmp_path):
    paths = _write_campaigns(tmp_path)

    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["same_cell_set_all_waves"] is True
    assert report["ready_observation_count"] == 13 * 9
    certificate = report["certificate"]
    assert certificate["conformal_rank"] == 9
    assert certificate["simultaneous_ratio_margin"] == pytest.approx(0.12)
    assert certificate["simultaneous_scope"] == {
        "cell_count": 9,
        "functionals_per_cell": 2,
        "functionals": ["drain_completion", "mean_task_jct"],
        "joint_functional_count_per_wave": 18,
    }
    assert len(certificate["rows"]) == 9
    assert all(row["resource_state"] == "empty" for row in certificate["rows"])
    assert all(row["lower_service_valid_on_holdout"] for row in certificate["rows"])
    assert all(
        row["node_bucket"] == NODE_SPECS[NODE].node_bucket
        for row in certificate["rows"]
    )


def test_staged_trajectory_does_not_require_a_false_static_rate_claim(tmp_path):
    paths = _write_campaigns(tmp_path)

    def remove_one_static_rate(payload):
        row = next(
            item
            for item in payload["rows"]
            if item["service_evidence_mode"] == STAGED_TRAJECTORY_EVIDENCE
        )
        row["summary"]["all_stable_rate_ready"] = False
        row["summary"]["stable_rate_ready_count"] -= 1
        row["probe_result"]["stable_rate_ready_count"] -= 1
        row["summary"]["rows"][2]["stable_rate_ready"] = False
        row["trajectory_progress_audit"]["children"][2][
            "stable_rate_ready_diagnostic"
        ] = False

    _mutate(paths[7], remove_one_static_rate)
    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "PASS"
    trajectory = next(
        row
        for row in report["certificate"]["rows"]
        if row["service_evidence_mode"] == STAGED_TRAJECTORY_EVIDENCE
    )
    assert trajectory["lower_service_valid_on_holdout"] is True


def test_staged_trajectory_missing_one_native_progress_observation_fails(tmp_path):
    paths = _write_campaigns(tmp_path)

    def remove_progress(payload):
        row = next(
            item
            for item in payload["rows"]
            if item["service_evidence_mode"] == STAGED_TRAJECTORY_EVIDENCE
        )
        row["summary"]["rows"][0]["completion_model"][
            "progress_observation_count"
        ] -= 1

    _mutate(paths[7], remove_progress)
    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "PROGRESS_SOURCE_INVALID"
        and "too few progress observations" in issue["detail"]
        for issue in report["validation_errors"]
    )


def test_stationary_action_still_requires_every_child_stable_rate(tmp_path):
    paths = _write_campaigns(tmp_path)

    def remove_static_rate(payload):
        row = next(
            item
            for item in payload["rows"]
            if item["service_evidence_mode"] != STAGED_TRAJECTORY_EVIDENCE
        )
        row["summary"]["all_stable_rate_ready"] = False
        row["summary"]["stable_rate_ready_count"] -= 1
        row["probe_result"]["stable_rate_ready_count"] -= 1
        row["summary"]["rows"][0]["stable_rate_ready"] = False

    _mutate(paths[7], remove_static_rate)
    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "PROGRESS_SOURCE_INVALID"
        for issue in report["validation_errors"]
    )


def test_default_paths_follow_campaign_prefix(tmp_path):
    paths = default_campaign_paths(node=NODE, artifact_root=tmp_path)

    assert set(paths) == set(EXPECTED_WAVES)
    assert paths[1].name == f"{CAMPAIGN_PREFIX}_{NODE}_r01_20260803.json"
    assert paths[13].name == f"{CAMPAIGN_PREFIX}_{NODE}_r13_20260803.json"


def test_legacy_artifact_prefix_is_rejected_even_for_v5_payload(tmp_path):
    paths = _write_campaigns(tmp_path)
    legacy = tmp_path / f"critical_gpu_completion_{NODE}_r06_20260803.json"
    paths[6].rename(legacy)
    paths[6] = legacy

    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "ARTIFACT_PATH_RULE_MISMATCH"
        for issue in report["validation_errors"]
    )


def test_missing_wave_is_wait_and_never_pass(tmp_path):
    paths = _write_campaigns(tmp_path)
    paths[7].unlink()

    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "WAIT_MISSING_WAVES"
    assert report["pass"] is False
    assert report["certificate_ready"] is False
    assert report["missing_waves"] == [7]
    assert report["validation_errors"] == []


def test_manifest_only_wave_is_wait_not_fail_or_pass(tmp_path):
    paths = _write_campaigns(tmp_path)

    def make_manifest(payload):
        payload["rows"] = []
        payload["status"] = "MANIFEST_ONLY"
        payload["pass"] = False

    _mutate(paths[5], make_manifest)
    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "WAIT_INCOMPLETE_WAVES"
    assert report["pass"] is False
    assert any(issue["code"] == "MISSING_CELLS" for issue in report["wait_reasons"])
    assert any(
        issue["code"] == "CAMPAIGN_NOT_COMPLETE"
        for issue in report["wait_reasons"]
    )


def test_history_progress_and_sparse_checkpoint_fail_strict_contract(tmp_path):
    paths = _write_campaigns(tmp_path)

    def poison(payload):
        row = payload["rows"][0]
        row["summary"]["rows"][0]["eta_source"] = "history"
        row["artifact_audit"]["files"] = [
            {
                "path": "/tmp/run/checkpoints/task_0/checkpoint.bin",
                "size_bytes": 4096,
                "allocated_blocks_512": 0,
            }
        ]

    _mutate(paths[13], poison)
    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    codes = {issue["code"] for issue in report["validation_errors"]}
    assert "HISTORY_ETA_FORBIDDEN" in codes
    assert "CHECKPOINT_INVALID" in codes
    assert report["pass"] is False


def test_cross_hardware_artifact_is_rejected_instead_of_pooled(tmp_path):
    paths = _write_campaigns(tmp_path)

    def change_hardware(payload):
        payload["node_spec"]["node_bucket"] = NODE_SPECS["node007"].node_bucket
        payload["node_spec"]["hardware_class"] = NODE_SPECS["node007"].hardware_class

    _mutate(paths[4], change_hardware)
    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "CROSS_HARDWARE_OR_NODE_SOURCE"
        for issue in report["validation_errors"]
    )
    assert report["hardware_local_only"] is True


def test_v1_protocol_is_strictly_rejected(tmp_path):
    paths = _write_campaigns(tmp_path)

    def use_v1(payload):
        payload["measurement_protocol"] = "critical_gpu_phase_completion_v1"
        for row in payload["rows"]:
            row["measurement_protocol"] = "critical_gpu_phase_completion_v1"

    _mutate(paths[4], use_v1)
    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "CAMPAIGN_CONTRACT_MISMATCH"
        and "measurement_protocol" in issue["detail"]
        for issue in report["validation_errors"]
    )
    assert report["same_measurement_protocol_all_waves"] is False


def test_smoke_wave_is_explicitly_forbidden(tmp_path):
    paths = _write_campaigns(tmp_path)
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps(_campaign(0)), encoding="utf-8")
    paths[0] = smoke

    report = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(
        issue["code"] == "SMOKE_WAVE_FORBIDDEN"
        for issue in report["validation_errors"]
    )


def test_cli_writes_json_and_markdown_for_wait_state(tmp_path):
    output = tmp_path / "gate.json"
    markdown = tmp_path / "gate.md"

    returncode = main(
        [
            "--node",
            NODE,
            "--artifact-root",
            str(tmp_path),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
        ]
    )

    assert returncode == 3
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "WAIT_MISSING_WAVES"
    )
    assert "Status: `WAIT_MISSING_WAVES`" in markdown.read_text(encoding="utf-8")
