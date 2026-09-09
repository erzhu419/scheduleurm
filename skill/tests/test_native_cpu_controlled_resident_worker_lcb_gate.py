import json

from algorithm.experiments.native_cpu_controlled_resident_worker_lcb_gate import (
    build_native_cpu_controlled_resident_worker_lcb_gate,
    _resident_state_ready,
)
from algorithm.experiments.native_cpu_controlled_resident_worker_campaign import (
    FULL_LOAD,
    HALF_LOAD,
    LOAD_SPECS,
)


def test_controlled_resident_state_requires_expected_regime_and_capacity():
    row = {
        "load_class": HALF_LOAD,
        "resident_workers": 96,
        "workers": 64,
        "dispatch_preflight": {
            "ready": True,
            "dispatch_external_cpu_regime": "cpu_external_heavy",
        },
        "observed_resource_state": {"external_cpu_core_equiv": 100.0},
    }
    assert _resident_state_ready(row)
    row["workers"] = 128
    assert not _resident_state_ready(row)


def _completion_row(load_class, workers, replicate, wall_s):
    spec = LOAD_SPECS[load_class]
    return {
        "node": "node001" if load_class == HALF_LOAD else "node003",
        "load_class": load_class,
        "resident_workers": spec["resident_workers"],
        "workers": workers,
        "replicate": replicate,
        "measurement_valid": True,
        "completion_model_ready": True,
        "dispatch_preflight": {
            "ready": True,
            "dispatch_external_cpu_regime": spec["expected_regime"],
        },
        "observed_resource_state": {
            "external_cpu_core_equiv": spec["resident_workers"],
        },
        "completion_model": {
            "completion_model_ready": True,
            "natural_exit": True,
            "stopped_on_stable": False,
            "total_wall_s": wall_s,
            "startup_overhead_s": 1.0,
            "completion_unit_s": 0.04,
            "terminal_overhead_s": 1.0,
            "checkpoint_observed_s": 0.4,
            "save_observed_s": 0.5,
            "progress_observation_count": 10,
        },
    }


def test_controlled_gate_merges_targeted_p1_with_main_campaign(tmp_path):
    main_rows = []
    p1_rows = []
    for load_class in (HALF_LOAD, FULL_LOAD):
        for workers in LOAD_SPECS[load_class]["profiles"]:
            target = p1_rows if workers == 1 else main_rows
            for replicate in range(13):
                target.append(
                    _completion_row(
                        load_class,
                        workers,
                        replicate,
                        (10.0 + workers) * (1.0 + 0.002 * (replicate % 5)),
                    )
                )
    main = tmp_path / "main.json"
    p1 = tmp_path / "p1.json"
    main.write_text(json.dumps({"status": "PASS", "rows": main_rows}))
    p1.write_text(json.dumps({"status": "PASS", "rows": p1_rows}))

    result = build_native_cpu_controlled_resident_worker_lcb_gate(
        campaign_paths=(main, p1),
    )

    assert result["status"] == "PASS"
    assert result["sample_counts_ready"] is True
    assert {row["allocation_workers"] for row in result["rows"]} >= {1, 96}
