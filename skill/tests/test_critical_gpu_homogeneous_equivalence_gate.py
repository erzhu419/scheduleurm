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
from algorithm.experiments.critical_gpu_homogeneous_equivalence_gate import (
    CELL_RATIO_LOWER,
    CELL_RATIO_UPPER,
    DEFAULT_MATCHED_WAVE,
    EQUIVALENCE_NODE,
    REPRESENTATIVE_NODE,
    ROBUST_CENTER_LOWER,
    ROBUST_CENTER_UPPER,
    build_critical_gpu_homogeneous_equivalence_gate,
    default_artifact_paths,
    main,
)


CODE_SHA256 = "b" * 64


def _artifact_path(root: Path, node: str, wave: int = 1) -> Path:
    return root / f"{CAMPAIGN_PREFIX}_{node}_r{wave:02d}_20260803.json"


def _write_pair(
    root: Path,
    *,
    representative_factor: float = 1.0,
    equivalence_factor: float = 1.02,
    wave: int = 1,
) -> tuple[Path, Path]:
    representative = _artifact_path(root, REPRESENTATIVE_NODE, wave)
    equivalent = _artifact_path(root, EQUIVALENCE_NODE, wave)
    representative.write_text(
        json.dumps(
            _campaign(REPRESENTATIVE_NODE, wave=wave, factor=representative_factor),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    equivalent.write_text(
        json.dumps(
            _campaign(EQUIVALENCE_NODE, wave=wave, factor=equivalence_factor),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return representative, equivalent


def _campaign(node: str, *, wave: int, factor: float) -> dict[str, object]:
    planned = campaign_cells(node=node, wave=wave)
    rows = [
        _measurement_row(node=node, cell=cell, factor=factor, index=index)
        for index, cell in enumerate(planned)
    ]
    return {
        "gate": "critical_gpu_completion_campaign",
        "measurement_protocol": PROTOCOL,
        "campaign_id": f"{CAMPAIGN_PREFIX}_{node}_r{wave:02d}_20260803",
        "node": node,
        "node_spec": asdict(NODE_SPECS[node]),
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
    *,
    node: str,
    cell: dict[str, object],
    factor: float,
    index: int,
) -> dict[str, object]:
    spec = next(
        item for item in WORKLOAD_SPECS
        if item.workload_key == cell["workload_key"]
    )
    task_count = int(cell["total_task_count"])
    children = [
        _child(
            task=task,
            factor=factor * (1.0 + 0.0005 * index),
            total_units=spec.max_iters,
            unit=spec.unit,
            min_progress=spec.min_rate_samples,
        )
        for task in range(task_count)
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
        "node": node,
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
) -> dict[str, object]:
    startup = (2.0 + 0.01 * task) * factor
    completion_unit = 1.0 * factor
    terminal = 1.0 * factor
    total_wall = startup + total_units * completion_unit + terminal
    return {
        "global_index": task,
        "run_name": f"task_{task}",
        "returncode": 0,
        "stable_rate_ready": True,
        "stable_rate": 1.0 / completion_unit,
        "completion_model_ready": True,
        "unit": unit,
        "completion_model": {
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
            "checkpoint_observed_s": 0.2 * factor,
            "save_observed_s": 0.1 * factor,
            "finalization_after_last_progress_s": terminal,
        },
    }


def _mutate(path: Path, callback) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_default_paths_use_exact_current_protocol_matched_wave_names(tmp_path):
    representative, equivalent = default_artifact_paths(artifact_root=tmp_path)

    assert representative.name == "critical_gpu_completion_v8_jtl110gpu_r01_20260803.json"
    assert equivalent.name == "critical_gpu_completion_v8_jtl110gpu2_r01_20260803.json"


def test_missing_artifact_is_wait(tmp_path):
    report = build_critical_gpu_homogeneous_equivalence_gate(artifact_root=tmp_path)

    assert report["status"] == "WAIT_MISSING_ARTIFACT"
    assert report["pass"] is False
    assert report["missing_nodes"] == [REPRESENTATIVE_NODE, EQUIVALENCE_NODE]
    assert report["validation_errors"] == []


def test_incomplete_existing_campaign_is_wait(tmp_path):
    representative, equivalent = _write_pair(tmp_path)

    def make_incomplete(payload):
        payload["rows"] = []
        payload["ready_row_count"] = 0
        payload["status"] = "WAIT_ENVIRONMENT"
        payload["pass"] = False

    _mutate(equivalent, make_incomplete)
    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=representative,
        equivalence_path=equivalent,
    )

    assert report["status"] == "WAIT_INCOMPLETE_ARTIFACT"
    assert report["pass"] is False
    assert report["validation_errors"] == []
    assert any(issue["code"] == "CAMPAIGN_NOT_COMPLETE" for issue in report["wait_reasons"])


def test_complete_matched_pair_passes_only_bucket_pooling_claim(tmp_path):
    representative, equivalent = _write_pair(tmp_path)

    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=representative,
        equivalence_path=equivalent,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["hardware_bucket_pooling_ready"] is True
    assert report["equivalence_node_participates_in_main_lcb"] is False
    assert report["same_measurement_protocol"] is True
    assert report["same_nine_workload_env_profile_cells"] is True
    assert report["comparison"]["matched_cell_count"] == 9
    assert report["comparison"]["all_cell_ratio_guardrails_pass"] is True
    assert report["comparison"]["robust_center_pass"] is True
    assert "arbitrary future states" in report["claim_boundary"]


@pytest.mark.parametrize(
    "poison",
    [
        lambda payload: payload.__setitem__(
            "measurement_protocol",
            "critical_gpu_phase_completion_v1",
        ),
        lambda payload: payload["node_spec"].__setitem__("role", "representative"),
        lambda payload: payload["rows"][0]["preflight"]["selected_gpus"][0].__setitem__(
            "used_mb",
            900,
        ),
    ],
)
def test_contract_violation_is_fail(tmp_path, poison):
    representative, equivalent = _write_pair(tmp_path)
    _mutate(equivalent, poison)

    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=representative,
        equivalence_path=equivalent,
    )

    assert report["status"] == "FAIL_CONTRACT"
    assert report["pass"] is False
    assert report["validation_errors"]


def test_history_checkpoint_and_natural_exit_are_strictly_rejected(tmp_path):
    representative, equivalent = _write_pair(tmp_path)

    def poison(payload):
        row = payload["rows"][0]
        row["summary"]["rows"][0]["eta_source"] = "history"
        row["summary"]["rows"][0]["completion_model"]["natural_exit"] = False
        row["artifact_audit"]["files"][0]["allocated_blocks_512"] = 0

    _mutate(equivalent, poison)
    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=representative,
        equivalence_path=equivalent,
    )

    assert report["status"] == "FAIL_CONTRACT"
    codes = {issue["code"] for issue in report["validation_errors"]}
    assert "HISTORY_ETA_FORBIDDEN" in codes
    assert "NATURAL_COMPLETION_INVALID" in codes
    assert "CHECKPOINT_INVALID" in codes


def test_different_workload_profile_cell_set_is_contract_failure(tmp_path):
    representative, equivalent = _write_pair(tmp_path)

    def change_cell(payload):
        payload["rows"][0]["workload_env"] = "unregistered_env"

    _mutate(equivalent, change_cell)
    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=representative,
        equivalence_path=equivalent,
    )

    assert report["status"] == "FAIL_CONTRACT"
    assert report["same_nine_workload_env_profile_cells"] is False


def test_malformed_nested_contract_returns_fail_instead_of_raising(tmp_path):
    representative, equivalent = _write_pair(tmp_path)

    def poison(payload):
        payload["node_spec"] = ["not", "an", "object"]
        payload["wave"] = {"not": "an integer"}

    _mutate(equivalent, poison)
    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=representative,
        equivalence_path=equivalent,
    )

    assert report["status"] == "FAIL_CONTRACT"
    codes = {issue["code"] for issue in report["validation_errors"]}
    assert "INVALID_NODE_SPEC" in codes
    assert "ARTIFACT_AUDIT_ERROR" in codes


def test_cell_ratio_outside_preregistered_interval_fails_equivalence(tmp_path):
    representative, equivalent = _write_pair(
        tmp_path,
        equivalence_factor=1.50,
    )

    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=representative,
        equivalence_path=equivalent,
    )

    assert report["status"] == "FAIL_EQUIVALENCE"
    assert report["comparison"]["all_cell_ratio_guardrails_pass"] is False
    assert all(
        not row["cell_equivalence_pass"]
        for row in report["comparison"]["rows"]
    )


def test_robust_center_is_tighter_than_cell_guardrail(tmp_path):
    representative, equivalent = _write_pair(
        tmp_path,
        equivalence_factor=1.20,
    )

    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=representative,
        equivalence_path=equivalent,
    )

    assert report["status"] == "FAIL_EQUIVALENCE"
    assert report["comparison"]["all_cell_ratio_guardrails_pass"] is True
    assert report["comparison"]["robust_center_pass"] is False


def test_acceptance_intervals_are_bidirectional_and_preregistered():
    assert CELL_RATIO_LOWER == pytest.approx(0.75)
    assert CELL_RATIO_UPPER == pytest.approx(4.0 / 3.0)
    assert ROBUST_CENTER_LOWER == pytest.approx(1.0 / 1.10)
    assert ROBUST_CENTER_UPPER == pytest.approx(1.10)
    assert DEFAULT_MATCHED_WAVE == 1


def test_cli_wait_returns_three_and_writes_outputs(tmp_path):
    output = tmp_path / "gate.json"
    markdown = tmp_path / "gate.md"

    rc = main(
        [
            "--artifact-root",
            str(tmp_path),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
        ]
    )

    assert rc == 3
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "WAIT_MISSING_ARTIFACT"
    assert "homogeneous-equivalence" in markdown.read_text(encoding="utf-8")
