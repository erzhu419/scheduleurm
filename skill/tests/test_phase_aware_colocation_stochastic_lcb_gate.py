from __future__ import annotations

import json

import pytest

from algorithm.experiments.phase_aware_colocation_stochastic_lcb_gate import (
    build_colocation_stochastic_lcb_gate,
)
from algorithm.experiments.file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
)


def _task(task_id: str, *, wall_s: float) -> dict:
    return {
        "task_id": task_id,
        "returncode": 0,
        "measurement_valid": True,
        "launch_offset_s": 0.05,
        "completion_model": {
            "completion_model_ready": True,
            "natural_exit": True,
            "interval_sample_count": 5,
            "total_units": 6,
            "total_wall_s": wall_s,
            "startup_overhead_s": 1.0,
            "terminal_overhead_s": 0.5,
        },
    }


def _payload(
    makespan_s: float,
    *,
    state: str = "controlled_colocation",
    bucket: str = "external_c0_2",
    regime: str = "cpu_external_light",
    run_state: str = "controlled_colocation",
    run_bucket: str = "external_c0_2",
    eta_source: str = "task_native_durable_csv",
) -> dict:
    return {
        "rows": [
            {
                "calibration_cell_id": "freqduet_node001_aw1_cc2_u6",
                "node": "node001",
                "workload_key": "freqduet_cpu_native",
                "workload_env": "freqduet",
                "allocation_workers": 1,
                "colocation_count": 2,
                "profile_axis": "colocation_count",
                "resource_state": "controlled_colocation",
                "observed_effective_resource_state": state,
                "dispatch_state_ready": True,
                "measurement_protocol": MEASUREMENT_PROTOCOL,
                "dispatch_sample_count": DISPATCH_CPU_SAMPLE_COUNT,
                "dispatch_external_cpu_regime": regime,
                "dispatch_external_cpu_fraction": 0.01,
                "dispatch_external_cpu_fraction_p95": 0.02,
                "dispatch_external_cpu_core_equiv": 1.0,
                "dispatch_external_cpu_core_equiv_p95": 2.0,
                "dispatch_external_cpu_bucket": bucket,
                "run_window_effective_resource_state": run_state,
                "run_window_external_cpu_core_equiv": 1.0,
                "run_window_external_cpu_bucket": run_bucket,
                "external_cpu_bucket": "external_c0_2",
                "init_cache_state": "warm",
                "aggregate_service_units": 12.0,
                "makespan_s": makespan_s,
                "measurement_valid": True,
                "all_tasks_ready": True,
                "near_synchronous_start": True,
                "eta_source": eta_source,
                "task_results": [
                    _task("task-1", wall_s=makespan_s - 0.1),
                    _task("task-2", wall_s=makespan_s),
                ],
            }
        ]
    }


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_colocation_split_conformal_gate_passes_exact_phase_aware_cell(tmp_path):
    training = []
    for index, makespan in enumerate((100.0, 101.0, 99.0), start=1):
        path = tmp_path / f"training-{index}.json"
        _write(path, _payload(makespan))
        training.append(path)
    calibration = []
    for index, makespan in enumerate(
        (100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0),
        start=1,
    ):
        path = tmp_path / f"calibration-{index}.json"
        _write(path, _payload(makespan))
        calibration.append(path)
    holdout = tmp_path / "holdout.json"
    _write(holdout, _payload(107.0))

    report = build_colocation_stochastic_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert report["pass"] is True
    assert report["status"] == "PASS"
    assert report["all_rows_ready"] is True
    assert report["all_lower_service_valid_on_holdout"] is True
    assert report["artifact_cell_sets_ready"] is True
    row = report["rows"][0]
    assert row["training_sample_count"] == 3
    assert row["raw_training_sample_count"] == 3
    assert row["calibration_sample_count"] == 9
    assert row["raw_calibration_sample_count"] == 9
    assert row["conformal_rank"] == 9
    assert row["statewise_match"] is True
    assert row["observed_min_interval_sample_counts"] == [5] * 13
    assert row["lower_service_units_per_s"] <= row["realized_service_units_per_s"]
    assert row["operational_point_sample_count"] == 12
    assert row["frozen_group_phase_model"] == (
        row["frozen_conformal_group_phase_model"]
    )
    assert row["point_predicted_makespan_s"] == (
        row["operational_point_predicted_makespan_s"]
    )


def test_colocation_operational_point_fit_preserves_conformal_base(tmp_path):
    training = []
    for index in range(3):
        path = tmp_path / f"training-{index}.json"
        _write(path, _payload(100.0))
        training.append(path)
    calibration = []
    for index in range(9):
        path = tmp_path / f"calibration-{index}.json"
        _write(path, _payload(120.0))
        calibration.append(path)
    holdout = tmp_path / "holdout.json"
    _write(holdout, _payload(120.0))

    report = build_colocation_stochastic_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    row = report["rows"][0]
    assert row["operational_point_predicted_makespan_s"] == pytest.approx(
        120.0
    )
    assert row["conformal_base_predicted_makespan_s"] == pytest.approx(100.0)
    assert row["conservative_upper_makespan_s"] == pytest.approx(120.0)
    assert row["point_relative_error"] == 0.0
    assert row["lower_service_valid_on_holdout"] is True


def test_colocation_gate_fails_closed_on_state_shift_or_history_eta(tmp_path):
    training = []
    for index in range(3):
        path = tmp_path / f"training-{index}.json"
        _write(path, _payload(100.0))
        training.append(path)
    calibration = []
    for index in range(9):
        path = tmp_path / f"calibration-{index}.json"
        _write(path, _payload(101.0))
        calibration.append(path)
    holdout = tmp_path / "holdout.json"
    _write(
        holdout,
        _payload(
            101.0,
            state="controlled_colocation_external",
            eta_source="history",
        ),
    )

    report = build_colocation_stochastic_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert report["pass"] is False
    assert report["status"] == "FAIL"
    row = report["rows"][0]
    assert row["statewise_match"] is False
    assert row["all_task_phase_models_ready"] is False
    assert row["ready"] is False


def test_colocation_gate_excludes_run_window_state_shift(tmp_path):
    training = []
    for index in range(3):
        path = tmp_path / f"training-{index}.json"
        _write(path, _payload(100.0 + index))
        training.append(path)
    calibration = []
    for index in range(9):
        path = tmp_path / f"calibration-{index}.json"
        _write(path, _payload(101.0 + index / 10.0))
        calibration.append(path)
    holdout = tmp_path / "holdout.json"
    _write(
        holdout,
        _payload(
            102.0,
            run_state="controlled_colocation_external",
            run_bucket="external_c3_16",
        ),
    )

    report = build_colocation_stochastic_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert report["pass"] is False
    row = report["rows"][0]
    assert row["training_sample_count"] == 0
    assert row["calibration_sample_count"] == 0
    assert row["statewise_match"] is False
    assert row["ready"] is False


def test_colocation_gate_stratifies_other_observed_load_states(tmp_path):
    training = []
    for index in range(3):
        path = tmp_path / f"training-{index}.json"
        _write(path, _payload(100.0 + index))
        training.append(path)
    other_training = tmp_path / "training-other-state.json"
    _write(
        other_training,
        _payload(
            150.0,
            state="controlled_colocation_external",
            bucket="external_c3_16",
            regime="cpu_external_moderate",
        ),
    )
    training.append(other_training)

    calibration = []
    for index in range(9):
        path = tmp_path / f"calibration-{index}.json"
        _write(path, _payload(101.0 + index % 3))
        calibration.append(path)
    other_calibration = tmp_path / "calibration-other-state.json"
    _write(
        other_calibration,
        _payload(
            160.0,
            state="controlled_colocation_external",
            bucket="external_c3_16",
            regime="cpu_external_moderate",
        ),
    )
    calibration.append(other_calibration)

    holdout = tmp_path / "holdout.json"
    _write(holdout, _payload(102.0))

    report = build_colocation_stochastic_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert report["status"] == "PASS"
    row = report["rows"][0]
    assert row["training_sample_count"] == 3
    assert row["raw_training_sample_count"] == 4
    assert row["calibration_sample_count"] == 9
    assert row["raw_calibration_sample_count"] == 10
    assert row["excluded_training_state_count"] == 1
    assert row["excluded_calibration_state_count"] == 1
    assert row["statewise_match"] is True


def test_colocation_gate_rejects_consistent_but_incomplete_artifact_subset(
    tmp_path,
):
    training = []
    for index in range(3):
        path = tmp_path / f"training-{index}.json"
        _write(path, _payload(100.0))
        training.append(path)
    calibration = []
    for index in range(9):
        path = tmp_path / f"calibration-{index}.json"
        _write(path, _payload(101.0))
        calibration.append(path)
    holdout = tmp_path / "holdout.json"
    _write(holdout, _payload(101.0))

    report = build_colocation_stochastic_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
        expected_cell_ids=(
            "freqduet_node001_aw1_cc2_u6",
            "sumo_node006_aw1_cc2_u6",
        ),
    )

    assert report["pass"] is False
    assert report["artifact_cell_sets_ready"] is False
    assert report["expected_cell_count"] == 2
    assert report["row_count"] == 2
    missing = next(
        row
        for row in report["rows"]
        if row["calibration_cell_id"] == "sumo_node006_aw1_cc2_u6"
    )
    assert missing["missing_holdout_cell"] is True
    assert missing["ready"] is False
