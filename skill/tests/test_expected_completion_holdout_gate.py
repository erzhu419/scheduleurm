from __future__ import annotations

import json

from algorithm.experiments.expected_completion_holdout_gate import (
    build_expected_completion_holdout_gate,
)
from algorithm.experiments.file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
)


def _payload(makespan: float, *, run_bucket: str = "external_c0_2") -> dict:
    return {
        "rows": [
            {
                "calibration_cell_id": "cpu_node001_aw1_cc1_u120",
                "node": "node001",
                "workload_key": "cpu_heavy_local_bench",
                "workload_env": "cpu",
                "allocation_workers": 1,
                "colocation_count": 1,
                "profile_axis": "colocation_count",
                "resource_state": "empty",
                "observed_effective_resource_state": "empty",
                "run_window_effective_resource_state": "empty",
                "dispatch_state_ready": True,
                "measurement_protocol": MEASUREMENT_PROTOCOL,
                "dispatch_sample_count": DISPATCH_CPU_SAMPLE_COUNT,
                "dispatch_external_cpu_regime": "cpu_external_idle",
                "dispatch_external_cpu_bucket": "external_c0_2",
                "run_window_external_cpu_bucket": run_bucket,
                "init_cache_state": "warm",
                "aggregate_service_units": 120.0,
                "makespan_s": makespan,
                "measurement_valid": True,
                "all_tasks_ready": True,
                "near_synchronous_start": True,
                "eta_source": "task_native_durable_csv",
                "task_results": [
                    {
                        "returncode": 0,
                        "measurement_valid": True,
                        "completion_model": {
                            "completion_model_ready": True,
                            "natural_exit": True,
                            "interval_sample_count": 8,
                            "startup_overhead_s": 1.0,
                            "terminal_overhead_s": 0.5,
                        },
                    }
                ],
            }
        ]
    }


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_expected_completion_gate_uses_future_wave_mean(tmp_path):
    fit = []
    for index, value in enumerate((10.0, 12.0, 11.0), 1):
        path = tmp_path / f"fit-{index}.json"
        _write(path, _payload(value))
        fit.append(path)
    validation = []
    for index, value in enumerate((10.8, 11.2), 1):
        path = tmp_path / f"validation-{index}.json"
        _write(path, _payload(value))
        validation.append(path)

    report = build_expected_completion_holdout_gate(
        fit_paths=fit,
        validation_paths=validation,
    )

    assert report["pass"] is True
    assert report["fit_wave_count"] == 3
    assert report["prospective_validation_wave_count"] == 2
    assert report["rows"][0]["predicted_expected_makespan_s"] == 11.0
    assert report["rows"][0]["validation_mean_makespan_s"] == 11.0


def test_expected_completion_gate_fails_closed_on_run_window_shift(tmp_path):
    fit = tmp_path / "fit.json"
    validation = tmp_path / "validation.json"
    _write(fit, _payload(11.0))
    _write(validation, _payload(11.0, run_bucket="external_c3_16"))

    report = build_expected_completion_holdout_gate(
        fit_paths=[fit],
        validation_paths=[validation],
    )

    assert report["pass"] is False
    assert report["rows"][0]["statewise_match"] is False
