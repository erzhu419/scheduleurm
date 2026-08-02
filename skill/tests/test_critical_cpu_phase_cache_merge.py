from __future__ import annotations

import json

import pytest

from algorithm.experiments.critical_cpu_phase_cache_merge import (
    build_critical_cpu_phase_cache_merge,
)
from simulation.service_cache import ServiceRateCache


def _row(workload: str, env: str, node: str, profile: int, units: float) -> dict:
    state = "empty" if profile == 1 else "controlled_colocation"
    startup, unit_s, terminal = 1.0, 0.1, 0.5
    point = startup + units * unit_s + terminal
    return {
        "workload_key": workload,
        "workload_env": env,
        "node": node,
        "allocation_workers": 1,
        "colocation_count": profile,
        "observed_effective_resource_state": state,
        "run_window_effective_resource_state": state,
        "dispatch_external_cpu_bucket": "external_c0_2",
        "run_window_external_cpu_bucket": "external_c0_2",
        "aggregate_service_units": units,
        "simultaneous_lower_service_units_per_s": units / (point + 1.0),
        "point_predicted_makespan_s": point,
        "simultaneous_conservative_upper_makespan_s": point + 1.0,
        "operational_point_group_phase_model": {
            "startup_overhead_s": startup,
            "aggregate_completion_unit_s": unit_s,
            "terminal_overhead_s": terminal,
        },
        "simultaneous_lower_service_valid_on_holdout": True,
        "training_sample_count": 3,
        "calibration_sample_count": 9,
        "point_relative_error": 0.01,
    }


def _gate(rows: list[dict]) -> dict:
    return {
        "pass": True,
        "all_rows_ready": True,
        "all_lower_service_valid_on_holdout": True,
        "simultaneous_wave_max_bound": {"ready": True},
        "rows": rows,
    }


def _expected_gate(cpu_rows: list[dict]) -> dict:
    rows = []
    for source in cpu_rows:
        units = float(source["aggregate_service_units"])
        startup, unit_s, terminal = 1.0, 0.1, 0.5
        rows.append(
            {
                **{
                    key: source[key]
                    for key in (
                        "workload_key",
                        "workload_env",
                        "node",
                        "allocation_workers",
                        "colocation_count",
                        "observed_effective_resource_state",
                        "run_window_effective_resource_state",
                        "dispatch_external_cpu_bucket",
                        "run_window_external_cpu_bucket",
                    )
                },
                "frozen_expected_phase_model": {
                    "startup_overhead_s": startup,
                    "aggregate_completion_unit_s": unit_s,
                    "terminal_overhead_s": terminal,
                },
                "predicted_expected_makespan_s": startup + units * unit_s + terminal,
                "expected_makespan_relative_error": 0.01,
                "fit_sample_count": 12,
            }
        )
    return {
        "pass": True,
        "all_rows_ready": True,
        "estimand": "expected_group_natural_completion_makespan",
        "rows": rows,
    }


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_critical_merge_is_exact_and_does_not_touch_legacy_index(tmp_path):
    base = tmp_path / "base.json"
    ServiceRateCache().save(base)
    cpu = tmp_path / "cpu.json"
    light = tmp_path / "light.json"
    expected = tmp_path / "expected.json"
    cpu_rows = [
        _row("cpu_heavy_local_bench", "cpu", node, profile, 120.0 * profile)
        for node in ("node001", "node002")
        for profile in (1, 8, 9)
    ]
    _write(
        cpu,
        _gate(cpu_rows),
    )
    _write(expected, _expected_gate(cpu_rows))
    _write(
        light,
        _gate(
            [
                _row(
                    "light_control_local",
                    "light_control",
                    node,
                    profile,
                    300.0 * profile,
                )
                for node in ("node003", "node005")
                for profile in (1, 13)
            ]
        ),
    )

    report = build_critical_cpu_phase_cache_merge(
        base_cache_path=base,
        cpu_gate_path=cpu,
        light_gate_path=light,
        cpu_expected_gate_path=expected,
    )

    assert report["pass"] is True
    assert report["inserted_count"] == 10
    cache = ServiceRateCache.from_snapshot(report["service_cache_snapshot"])
    assert cache.get("cpu_heavy_local_bench", 8) is None
    exact = cache.lookup_statewise_exact(
        "cpu_heavy_local_bench",
        workload_env="cpu",
        node_bucket="node001:cpu_hpc_192c",
        resource_state="controlled_colocation",
        allocation_workers=1,
        colocation_count=8,
        resident_mix="",
    )
    assert exact.is_exact
    assert exact.record is not None
    assert exact.record.completion_model_sample_count == 12


def test_critical_merge_rejects_run_window_state_shift(tmp_path):
    base = tmp_path / "base.json"
    ServiceRateCache().save(base)
    cpu_rows = [
        _row("cpu_heavy_local_bench", "cpu", node, profile, 120.0 * profile)
        for node in ("node001", "node002")
        for profile in (1, 8, 9)
    ]
    cpu_rows[0]["run_window_effective_resource_state"] = "cpu_resident_external"
    cpu = tmp_path / "cpu.json"
    light = tmp_path / "light.json"
    expected = tmp_path / "expected.json"
    _write(cpu, _gate(cpu_rows))
    _write(expected, _expected_gate(cpu_rows))
    _write(
        light,
        _gate(
            [
                _row("light_control_local", "light_control", node, profile, 300.0 * profile)
                for node in ("node003", "node005")
                for profile in (1, 13)
            ]
        ),
    )

    with pytest.raises(ValueError, match="dispatch/run state mismatch"):
        build_critical_cpu_phase_cache_merge(
            base_cache_path=base,
            cpu_gate_path=cpu,
            light_gate_path=light,
            cpu_expected_gate_path=expected,
        )
