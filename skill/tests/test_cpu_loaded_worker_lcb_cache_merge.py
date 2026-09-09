import json
from pathlib import Path

from algorithm.experiments.cpu_loaded_worker_lcb_cache_merge import (
    build_cpu_loaded_worker_lcb_cache_merge,
)
from simulation.service_cache import ServiceRateCache


def _gate_row(state, workers, *, resident=None, nodes=None):
    row = {
        "resource_state": state,
        "allocation_workers": workers,
        "lower_service_units_per_s": 8.0,
        "lower_service_valid_on_holdout": True,
        "training_sample_count": 3,
        "calibration_wave_count": 9,
        "point_eta_s": 18.0,
        "point_relative_error": 0.03,
        "startup_overhead_s": 1.0,
        "completion_unit_s": 0.08,
        "finalization_overhead_s": 1.0,
        "checkpoint_observed_s": 0.4,
        "save_observed_s": 0.5,
    }
    if resident is not None:
        row["resident_workers"] = resident
    if nodes is not None:
        row["measured_nodes"] = nodes
    return row


def test_loaded_merge_keeps_state_and_worker_axes_out_of_legacy_index(
    tmp_path: Path,
):
    base = tmp_path / "base.json"
    ServiceRateCache().save(base)
    controlled = tmp_path / "controlled.json"
    controlled.write_text(
        json.dumps(
            {
                "pass": True,
                "rows": [_gate_row("half_loaded", 4, resident=96)],
            }
        )
    )
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps(
            {
                "pass": True,
                "rows": [
                    _gate_row(
                        "cpu_external_light",
                        8,
                        nodes=["node006"],
                    )
                ],
            }
        )
    )

    report = build_cpu_loaded_worker_lcb_cache_merge(
        base_cache_path=base,
        controlled_gate_path=controlled,
        external_gate_path=external,
    )
    snapshot = report["service_cache_snapshot"]

    assert report["status"] == "PASS"
    assert report["inserted_count"] == 7
    assert report["inserted_boundary_count"] == 12
    rebuilt = ServiceRateCache.from_snapshot(snapshot)
    assert rebuilt.get("cpu_heavy_local_bench", 1) is None
    exact = rebuilt.get_statewise(
        "cpu_heavy_local_bench",
        allocation_workers=4,
        colocation_count=1,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="half_loaded",
        resident_mix="controlled_resident_workers=96",
        strict=True,
    )
    assert exact is not None
    assert exact.aggregate_rate == 8.0
    states = {row["resource_state"] for row in snapshot["records"]}
    assert states == {"half_loaded", "cpu_external_light"}
    external_rows = [
        row
        for row in snapshot["records"]
        if row["resource_state"] == "cpu_external_light"
    ]
    assert [row["node_bucket"] for row in external_rows] == [
        "node006:cpu_hpc_192c"
    ]
    assert len(snapshot["capacity_boundaries"]) == 12
