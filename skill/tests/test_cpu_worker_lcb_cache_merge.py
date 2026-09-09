import json

from algorithm.experiments.cpu_worker_lcb_cache_merge import (
    build_cpu_worker_lcb_cache_merge,
)
from simulation.service_cache import ServiceRateCache


def test_worker_gate_merges_exact_allocation_axis_without_legacy_overwrite(
    tmp_path,
):
    gate = {
        "pass": True,
        "rows": [
            {
                "allocation_workers": 32,
                "lower_service_units_per_s": 4.0,
                "lower_service_valid_on_holdout": True,
                "training_sample_count": 3,
                "calibration_wave_count": 9,
                "point_eta_s": 40.0,
                "point_relative_error": 0.02,
                "startup_overhead_s": 1.0,
                "completion_unit_s": 0.2,
                "finalization_overhead_s": 2.0,
                "checkpoint_observed_s": 0.5,
                "save_observed_s": 0.1,
            }
        ],
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    report = build_cpu_worker_lcb_cache_merge(
        base_cache_path=tmp_path / "missing.json",
        gate_path=gate_path,
    )
    cache = ServiceRateCache.from_snapshot(report["service_cache_snapshot"])
    record = cache.get_statewise(
        "cpu_heavy_local_bench",
        allocation_workers=32,
        colocation_count=1,
        node_bucket="node003:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
        strict=True,
    )
    assert record is not None
    assert record.aggregate_rate == 4.0
    assert record.completion_model_ready
    assert cache.get("cpu_heavy_local_bench", 1) is None
