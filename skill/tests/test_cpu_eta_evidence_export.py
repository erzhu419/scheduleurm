import json

from algorithm.experiments.cpu_eta_evidence_export import (
    _markdown,
    build_cpu_eta_evidence_rows,
)


def test_markdown_distinguishes_worker_and_colocation_axes():
    rows = [
        {
            "evidence_family": "native_colocation",
            "node": "node001",
            "workload_key": "freqduet",
            "resource_state": "controlled_colocation",
            "profile_axis": "colocation_count",
            "allocation_workers": 1,
            "colocation_count": 4,
            "point_eta_s": 10.0,
            "holdout_actual_s": 11.0,
            "conservative_eta_s": 12.0,
            "lower_service_units_per_s": 0.5,
            "lower_service_valid": True,
        },
        {
            "evidence_family": "allocation_worker_curve",
            "node": "node001-node006",
            "workload_key": "cpu",
            "resource_state": "empty",
            "profile_axis": "allocation_workers",
            "allocation_workers": 128,
            "colocation_count": 1,
            "point_eta_s": 20.0,
            "holdout_actual_s": 21.0,
            "conservative_eta_s": 22.0,
            "lower_service_units_per_s": 0.25,
            "lower_service_valid": True,
        },
    ]
    text = _markdown(rows)
    assert "colocation_count=p4" in text
    assert "allocation_workers=p128" in text


def test_export_includes_passed_loaded_state_gate(tmp_path):
    worker = tmp_path / "worker.json"
    worker.write_text(json.dumps({"rows": []}))
    loaded = tmp_path / "loaded.json"
    loaded.write_text(
        json.dumps(
            {
                "gate": "native_cpu_controlled_resident_worker_lcb_gate",
                "pass": True,
                "rows": [
                    {
                        "resource_state": "half_loaded",
                        "dispatch_external_cpu_regime": "cpu_external_heavy",
                        "allocation_workers": 8,
                        "point_eta_s": 12.0,
                        "holdout_actual_completion_s": 12.5,
                        "simultaneous_conservative_eta_s": 14.0,
                        "lower_service_units_per_s": 11.4,
                        "lower_service_valid_on_holdout": True,
                        "training_sample_count": 3,
                        "calibration_wave_count": 9,
                    }
                ],
            }
        )
    )

    rows = build_cpu_eta_evidence_rows(
        native_gate_paths=(),
        worker_gate_path=worker,
        loaded_gate_paths=(loaded,),
    )

    assert len(rows) == 1
    assert rows[0]["evidence_family"] == "controlled_loaded_worker_curve"
    assert rows[0]["resource_state"] == "half_loaded"
