from __future__ import annotations

import json

from algorithm.experiments.native_cpu_workload_completion_matrix import smoke_cells
from algorithm.experiments.native_cpu_stochastic_calibration_campaign import (
    campaign_waves,
)
from algorithm.experiments.phase_aware_stochastic_lcb_gate import (
    build_stochastic_completion_lcb_gate,
)
from algorithm.experiments.file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
)


def _row(
    *,
    total: float,
    run: int,
    bucket: str = "external_c0_2",
    regime: str = "cpu_external_light",
) -> dict[str, object]:
    return {
        "cell_id": f"freqduet_node001_seed{700000 + run}",
        "calibration_cell_id": "freqduet_cpu_native_freqduet_node001_empty_aw1_cc1_u6",
        "node": "node001",
        "workload_key": "freqduet_cpu_native",
        "workload_env": "freqduet",
        "resource_state": "empty",
        "observed_effective_resource_state": "empty",
        "dispatch_state_ready": True,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "dispatch_sample_count": DISPATCH_CPU_SAMPLE_COUNT,
        "dispatch_external_cpu_regime": regime,
        "dispatch_external_cpu_fraction": 0.01,
        "dispatch_external_cpu_fraction_p95": 0.02,
        "dispatch_external_cpu_core_equiv": 1.0,
        "dispatch_external_cpu_bucket": bucket,
        "run_window_effective_resource_state": "empty",
        "run_window_external_cpu_core_equiv": 1.0,
        "run_window_external_cpu_bucket": "external_c0_2",
        "external_cpu_bucket": "external_c0_2",
        "allocation_workers": 1,
        "colocation_count": 1,
        "init_cache_state": "warm",
        "eta_source": "task_native_durable_csv",
        "measurement_valid": True,
        "returncode": 0,
        "completion_model": {
            "completion_model_ready": True,
            "child_returncode": 0,
            "natural_exit": True,
            "startup_overhead_s": 2.0,
            "completion_unit_s": (total - 3.0) / 6.0,
            "terminal_overhead_s": 1.0,
            "interval_sample_count": 5,
            "total_units": 6,
            "total_wall_s": total,
        },
    }


def _write(path, rows):
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def test_stochastic_lcb_uses_finite_sample_conformal_rank(tmp_path):
    training = [
        _write(tmp_path / f"train{index}.json", [_row(total=100.0 + index, run=index)])
        for index in range(3)
    ]
    calibration = [
        _write(
            tmp_path / f"cal{index}.json",
            [_row(total=101.0 + (index % 3), run=10 + index)],
        )
        for index in range(9)
    ]
    holdout = _write(tmp_path / "holdout.json", [_row(total=102.0, run=99)])

    result = build_stochastic_completion_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
        miscoverage_alpha=0.10,
    )

    assert result["status"] == "PASS"
    row = result["rows"][0]
    assert row["conformal_rank"] == 9
    assert row["finite_sample_conformal_ready"] is True
    assert row["conservative_holdout_covered"] is True
    assert row["lower_service_valid_on_holdout"] is True
    assert row["operational_point_sample_count"] == 12
    assert row["frozen_phase_model"] == row["frozen_conformal_phase_model"]
    assert row["point_predicted_total_s"] == (
        row["operational_point_predicted_total_s"]
    )


def test_operational_point_fit_does_not_replace_conformal_base(tmp_path):
    training = [
        _write(
            tmp_path / f"train{index}.json",
            [_row(total=100.0, run=index)],
        )
        for index in range(3)
    ]
    calibration = [
        _write(
            tmp_path / f"cal{index}.json",
            [_row(total=120.0, run=10 + index)],
        )
        for index in range(9)
    ]
    holdout = _write(
        tmp_path / "holdout.json",
        [_row(total=120.0, run=99)],
    )

    result = build_stochastic_completion_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
        miscoverage_alpha=0.10,
    )

    row = result["rows"][0]
    assert row["operational_point_predicted_total_s"] == 120.0
    assert row["conformal_base_predicted_total_s"] == 100.0
    assert row["conservative_upper_total_s"] == 120.0
    assert row["point_relative_error"] == 0.0
    assert row["lower_service_valid_on_holdout"] is True


def test_stochastic_lcb_rejects_too_few_calibration_runs(tmp_path):
    training = [_write(tmp_path / "train.json", [_row(total=100.0, run=1)])]
    calibration = [
        _write(tmp_path / f"cal{index}.json", [_row(total=101.0, run=10 + index)])
        for index in range(8)
    ]
    holdout = _write(tmp_path / "holdout.json", [_row(total=101.0, run=99)])

    result = build_stochastic_completion_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
        miscoverage_alpha=0.10,
    )

    assert result["status"] == "FAIL"
    assert result["rows"][0]["conformal_rank"] == 9
    assert result["rows"][0]["finite_sample_conformal_ready"] is False


def test_stochastic_lcb_rejects_resource_regime_shift(tmp_path):
    training = [_write(tmp_path / "train.json", [_row(total=100.0, run=1)])]
    calibration = [
        _write(tmp_path / f"cal{index}.json", [_row(total=101.0, run=10 + index)])
        for index in range(9)
    ]
    holdout = _write(
        tmp_path / "holdout.json",
        [
            _row(
                total=101.0,
                run=99,
                bucket="external_c17_32",
                regime="cpu_external_heavy",
            )
        ],
    )

    result = build_stochastic_completion_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert result["status"] == "FAIL"
    assert result["rows"][0]["statewise_match"] is False
    assert result["rows"][0]["ready"] is False


def test_stochastic_lcb_stratifies_other_observed_load_states(tmp_path):
    training = [
        _write(tmp_path / f"train{index}.json", [_row(total=100.0, run=index)])
        for index in range(3)
    ]
    training.append(
        _write(
            tmp_path / "train-other.json",
            [
                _row(
                    total=140.0,
                    run=8,
                    bucket="external_c3_16",
                    regime="cpu_external_moderate",
                )
            ],
        )
    )
    calibration = [
        _write(
            tmp_path / f"cal{index}.json",
            [_row(total=101.0, run=10 + index)],
        )
        for index in range(9)
    ]
    calibration.append(
        _write(
            tmp_path / "cal-other.json",
            [
                _row(
                    total=150.0,
                    run=30,
                    bucket="external_c3_16",
                    regime="cpu_external_moderate",
                )
            ],
        )
    )
    holdout = _write(tmp_path / "holdout.json", [_row(total=101.0, run=99)])

    result = build_stochastic_completion_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert result["status"] == "PASS"
    row = result["rows"][0]
    assert row["training_sample_count"] == 3
    assert row["calibration_sample_count"] == 9
    assert row["excluded_training_state_count"] == 1
    assert row["excluded_calibration_state_count"] == 1
    assert row["statewise_match"] is True


def test_stochastic_lcb_rejects_too_few_outer_intervals(tmp_path):
    training_rows = [_row(total=100.0, run=1)]
    training_rows[0]["completion_model"]["interval_sample_count"] = 3
    training = [_write(tmp_path / "train.json", training_rows)]
    calibration = [
        _write(tmp_path / f"cal{index}.json", [_row(total=101.0, run=10 + index)])
        for index in range(9)
    ]
    holdout = _write(tmp_path / "holdout.json", [_row(total=101.0, run=99)])

    result = build_stochastic_completion_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert result["status"] == "FAIL"
    assert result["rows"][0]["all_completion_ready"] is False
    assert result["rows"][0]["ready"] is False


def test_stochastic_lcb_rejects_legacy_run_window_only_state(tmp_path):
    training = [
        _write(tmp_path / f"train{index}.json", [_row(total=100.0, run=index)])
        for index in range(3)
    ]
    calibration = [
        _write(
            tmp_path / f"cal{index}.json",
            [_row(total=101.0, run=10 + index)],
        )
        for index in range(9)
    ]
    holdout_row = _row(total=101.0, run=99)
    holdout_row.pop("dispatch_state_ready")
    holdout_row.pop("measurement_protocol")
    holdout_row.pop("dispatch_sample_count")
    holdout_row.pop("dispatch_external_cpu_regime")
    holdout_row.pop("dispatch_external_cpu_core_equiv")
    holdout_row.pop("dispatch_external_cpu_bucket")
    holdout = _write(tmp_path / "holdout.json", [holdout_row])

    result = build_stochastic_completion_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert result["status"] == "FAIL"
    assert result["rows"][0]["all_completion_ready"] is False
    assert result["rows"][0]["ready"] is False


def test_stochastic_lcb_rejects_return_139_despite_valid_flag(tmp_path):
    training = [
        _write(tmp_path / f"train{index}.json", [_row(total=100.0, run=index)])
        for index in range(3)
    ]
    calibration = [
        _write(
            tmp_path / f"cal{index}.json",
            [_row(total=101.0, run=10 + index)],
        )
        for index in range(9)
    ]
    holdout_row = _row(total=101.0, run=99)
    holdout_row["returncode"] = 139
    holdout_row["measurement_valid"] = True
    holdout = _write(tmp_path / "holdout.json", [holdout_row])

    result = build_stochastic_completion_lcb_gate(
        training_paths=training,
        calibration_paths=calibration,
        holdout_path=holdout,
    )

    assert result["status"] == "FAIL"
    assert result["rows"][0]["all_completion_ready"] is False
    assert result["rows"][0]["ready"] is False


def test_native_cells_change_seed_without_changing_physical_cell():
    base = smoke_cells(seed_offset=0)
    shifted = smoke_cells(seed_offset=100)

    assert [cell.node for cell in base] == [cell.node for cell in shifted]
    assert [cell.workload_key for cell in base] == [cell.workload_key for cell in shifted]
    assert [cell.seed + 100 for cell in base] == [cell.seed for cell in shifted]
    assert [cell.cell_id for cell in base] != [cell.cell_id for cell in shifted]
    assert all(cell.total_outer_units >= 6 for cell in shifted)
    assert len({cell.seed for cell in base if cell.kind == "freqduet"}) == 1
    assert len({cell.seed for cell in base if cell.kind == "sumo"}) == 1


def test_stochastic_campaign_has_disjoint_training_calibration_holdout_waves():
    waves = campaign_waves(tag="test")

    assert len(waves) == 13
    assert [wave["role"] for wave in waves].count("training") == 3
    assert [wave["role"] for wave in waves].count("calibration") == 9
    assert [wave["role"] for wave in waves].count("holdout") == 1
    assert len({wave["seed_offset"] for wave in waves}) == 13
    assert len({wave["output"] for wave in waves}) == 13
