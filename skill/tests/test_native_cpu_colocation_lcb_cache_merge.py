import json
from pathlib import Path

from algorithm.experiments.native_cpu_colocation_eta_closure_gate import (
    build_native_cpu_colocation_eta_closure_gate,
)
from algorithm.experiments.native_cpu_colocation_lcb_cache_merge import (
    build_native_cpu_colocation_lcb_cache_merge,
)
from simulation.service_cache import ServiceRateCache


WORKLOADS = {
    "freqduet_cpu_native": "freqduet",
    "sumo_eval_cpu_native": "sumo",
}
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _gate(profile: int) -> dict:
    rows = []
    for workload_key, workload_env in WORKLOADS.items():
        for node in NODES:
            external = node in {"node004", "node006"}
            if profile == 1:
                state = "cpu_resident_external" if external else "empty"
            else:
                state = (
                    "controlled_colocation_external"
                    if external
                    else "controlled_colocation"
                )
            total_units = 6.0 * profile
            startup = 2.0
            completion_unit = 3.0
            terminal = 1.0
            point_eta = startup + total_units * completion_unit + terminal
            rows.append(
                {
                    "workload_key": workload_key,
                    "workload_env": workload_env,
                    "node": node,
                    "allocation_workers": 1,
                    "colocation_count": profile,
                    "observed_effective_resource_state": state,
                    "run_window_effective_resource_state": state,
                    "dispatch_external_cpu_regime": (
                        "cpu_external_light"
                        if external
                        else "cpu_external_idle"
                    ),
                    "aggregate_service_units": total_units,
                    "point_predicted_makespan_s": point_eta,
                    "actual_makespan_s": point_eta * 0.98,
                    "point_relative_error": 0.02,
                    "simultaneous_conservative_upper_makespan_s": (
                        point_eta * 1.2
                    ),
                    "simultaneous_lower_service_units_per_s": (
                        total_units / (point_eta * 1.2)
                    ),
                    "simultaneous_lower_service_valid_on_holdout": True,
                    "training_sample_count": 3,
                    "calibration_sample_count": 9,
                    "operational_point_group_phase_model": {
                        "startup_overhead_s": startup,
                        "aggregate_completion_unit_s": completion_unit,
                        "terminal_overhead_s": terminal,
                    },
                }
            )
    return {
        "pass": True,
        "row_count": 12,
        "all_rows_ready": True,
        "all_lower_service_valid_on_holdout": True,
        "simultaneous_wave_max_bound": {
            "ready": True,
            "conformal_ratio_margin": 0.2,
        },
        "rows": rows,
    }


def _write_gate(path: Path, profile: int) -> Path:
    path.write_text(json.dumps(_gate(profile)), encoding="utf-8")
    return path


def test_native_merge_preserves_exact_node_state_and_profile_axes(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.json"
    ServiceRateCache().save(base)
    gates = tuple(
        _write_gate(tmp_path / f"p{profile}.json", profile)
        for profile in (1, 2, 4)
    )

    report = build_native_cpu_colocation_lcb_cache_merge(
        base_cache_path=base,
        gate_paths=gates,
    )
    cache = ServiceRateCache.from_snapshot(report["service_cache_snapshot"])

    assert report["status"] == "PASS"
    assert report["inserted_count"] == 36
    assert cache.get("freqduet_cpu_native", 2) is None

    idle = cache.lookup_statewise_exact(
        "freqduet_cpu_native",
        workload_env="freqduet",
        node_bucket="node001:cpu_hpc_192c",
        resource_state="controlled_colocation",
        allocation_workers=1,
        colocation_count=2,
        resident_mix="",
    )
    external = cache.lookup_statewise_exact(
        "sumo_eval_cpu_native",
        workload_env="sumo",
        node_bucket="node004:cpu_hpc_192c",
        resource_state="controlled_colocation_external",
        allocation_workers=1,
        colocation_count=4,
        resident_mix="organic_external_measured",
    )

    assert idle.is_exact
    assert idle.record is not None
    assert idle.record.completion_model_ready
    assert idle.record.total_units == 12.0
    assert len(idle.record.per_task_rates) == 2
    assert external.is_exact
    assert external.record is not None
    assert external.record.resource_state == "controlled_colocation_external"


def test_native_closure_gate_requires_all_36_exact_cache_rows(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.json"
    ServiceRateCache().save(base)
    gates = tuple(
        _write_gate(tmp_path / f"p{profile}.json", profile)
        for profile in (1, 2, 4)
    )
    merge = build_native_cpu_colocation_lcb_cache_merge(
        base_cache_path=base,
        gate_paths=gates,
    )
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(merge["service_cache_snapshot"]),
        encoding="utf-8",
    )
    merge_path = tmp_path / "merge.json"
    merge_path.write_text(
        json.dumps(
            {
                key: value
                for key, value in merge.items()
                if key != "service_cache_snapshot"
            }
        ),
        encoding="utf-8",
    )

    result = build_native_cpu_colocation_eta_closure_gate(
        gate_paths=gates,
        merge_report_path=merge_path,
        cache_path=cache_path,
    )

    assert result["pass"]
    assert result["exact_row_count"] == 36
    assert all(result["checks"].values())


def test_native_merge_rejects_dispatch_and_run_window_state_mismatch(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.json"
    ServiceRateCache().save(base)
    payload = _gate(1)
    payload["rows"][0]["run_window_effective_resource_state"] = (
        "cpu_resident_external"
    )
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    gates = (
        bad,
        _write_gate(tmp_path / "p2.json", 2),
        _write_gate(tmp_path / "p4.json", 4),
    )

    try:
        build_native_cpu_colocation_lcb_cache_merge(
            base_cache_path=base,
            gate_paths=gates,
        )
    except ValueError as exc:
        assert "dispatch/run-window state mismatch" in str(exc)
    else:
        raise AssertionError("state mismatch must fail closed")
