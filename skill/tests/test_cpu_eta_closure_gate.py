import json
from pathlib import Path

from algorithm.experiments.cpu_eta_closure_gate import (
    build_cpu_eta_closure_gate,
)


def _write(path: Path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_cpu_eta_closure_gate_requires_all_formal_states(tmp_path):
    gate_rows = [{"lower_service_valid_on_holdout": True}]
    empty = _write(
        tmp_path / "empty.json",
        {"pass": True, "rows": gate_rows * 11},
    )
    controlled = _write(
        tmp_path / "controlled.json",
        {
            "pass": True,
            "rows": gate_rows * 12,
            "missing_cells": [],
            "duplicate_cells": [],
            "node_mismatches": [],
        },
    )
    external = _write(
        tmp_path / "external.json",
        {
            "pass": True,
            "rows": gate_rows * 16,
            "missing_cells": [],
            "duplicate_cells": [],
            "node_mismatches": [],
        },
    )
    merge = _write(
        tmp_path / "merge.json",
        {
            "status": "PASS",
            "inserted_count": 88,
            "inserted_boundary_count": 12,
            "inserted_rows": [
                {
                    "eta_source": "task_native_tqdm",
                    "stable_rate_ready": True,
                    "completion_model_ready": True,
                }
            ],
        },
    )
    design_rows = [
        {
            "hardware_group": "cpu_hpc_192c",
            "workload_key": "cpu_heavy_local_bench",
            "resource_state": state,
            "resident_mix": mix,
            "missing_profiles": "",
            "status": status,
            "eta_missing_profiles": "",
            "service_only_profiles": "",
            "eta_status": (
                "eta_measured"
                if status == "measured"
                else "eta_measured_to_capacity_boundary"
            ),
        }
        for state, mix, status in (
            ("empty", "none", "measured"),
            (
                "half_loaded",
                "controlled_resident_workers=96",
                "measured_to_capacity_boundary",
            ),
            (
                "full_loaded",
                "controlled_resident_workers=180",
                "measured_to_capacity_boundary",
            ),
        )
    ]
    design = _write(tmp_path / "design.json", {"eta_rows": design_rows})

    result = build_cpu_eta_closure_gate(
        empty_gate_path=empty,
        controlled_gate_path=controlled,
        external_gate_path=external,
        merge_report_path=merge,
        design_path=design,
    )

    assert result["status"] == "PASS"
    assert all(result["checks"].values())
