from __future__ import annotations

import json

from algorithm.experiments.phase_aware_completion_holdout_gate import (
    build_completion_holdout_gate,
)
from algorithm.experiments.phase_aware_cpu_completion_matrix import (
    _completion_output_root,
    _external_cpu_bucket,
    _parse_resource_state_log,
    jtl311_allocation_curve_cells,
)


def _completion_model(total_wall_s: float, unit_s: float = 1.0) -> dict[str, object]:
    return {
        "completion_model_ready": True,
        "startup_overhead_s": 1.0,
        "completion_unit_s": unit_s,
        "terminal_overhead_s": 1.0,
        "total_units": 8,
        "total_wall_s": total_wall_s,
    }


def test_resource_state_log_parser_uses_last_emitted_state(tmp_path):
    path = tmp_path / "probe.log"
    path.write_text(
        "noise\n"
        'ScheduleurmResourceState {"external_cpu_core_equiv":1.0}\n'
        'ScheduleurmResourceState {"external_cpu_core_equiv":17.5}\n',
        encoding="utf-8",
    )

    assert _parse_resource_state_log(path) == {"external_cpu_core_equiv": 17.5}
    assert _external_cpu_bucket(17.5) == "external_c17_32"


def test_jtl311_phase_curve_respects_eight_core_capacity():
    cells = jtl311_allocation_curve_cells()

    assert [cell.workers for cell in cells] == [2, 4, 8]
    assert {cell.node for cell in cells} == {"jtl311linux"}
    assert all(cell.resource_state == "empty" for cell in cells)
    assert all(cell.work_items == 1_800_000 for cell in cells)
    assert _completion_output_root(
        "jtl311linux", "run"
    ) == "/home/erzhu419/scheduleurm_work/eta_completion/run"
    assert _completion_output_root(
        "node001", "run"
    ) == "/home/zhengliang01/scheduleurm_work/eta_completion/run"


def test_holdout_gate_rejects_different_observed_load_buckets(tmp_path):
    fit_path = tmp_path / "fit.json"
    holdout_path = tmp_path / "holdout.json"
    output_row = {
        "cell_id": "cpu_p64_external",
        "node": "node004",
        "workload_key": "cpu_heavy_local_bench",
        "workers": 64,
        "resource_state": "cpu_resident_external",
        "resource_telemetry_ready": True,
        "dispatch_state_ready": True,
        "completion_model": _completion_model(10.0),
    }
    fit_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        **output_row,
                        "run_id": "fit",
                        "dispatch_external_cpu_core_equiv": 12.0,
                        "dispatch_external_cpu_bucket": "external_c3_16",
                        "external_cpu_core_equiv": 24.0,
                        "external_cpu_bucket": "external_c17_32",
                        "observed_resource_state": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    holdout_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        **output_row,
                        "run_id": "holdout",
                        "dispatch_external_cpu_core_equiv": 24.0,
                        "dispatch_external_cpu_bucket": "external_c17_32",
                        "external_cpu_core_equiv": 24.0,
                        "external_cpu_bucket": "external_c17_32",
                        "observed_resource_state": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_completion_holdout_gate(
        fit_path=fit_path,
        holdout_path=holdout_path,
    )

    assert result["status"] == "FAIL"
    assert result["rows"][0]["completion_ready"] is True
    assert result["rows"][0]["resource_state_match"] is False
    assert result["rows"][0]["ready"] is False


def test_holdout_gate_accepts_same_observed_load_bucket(tmp_path):
    fit_path = tmp_path / "fit.json"
    holdout_path = tmp_path / "holdout.json"
    common = {
        "cell_id": "cpu_p64_external",
        "node": "node004",
        "workload_key": "cpu_heavy_local_bench",
        "workers": 64,
        "resource_state": "cpu_resident_external",
        "resource_telemetry_ready": True,
        "dispatch_state_ready": True,
        "dispatch_external_cpu_bucket": "external_c3_16",
        "external_cpu_bucket": "external_c17_32",
        "observed_resource_state": {},
    }
    fit_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        **common,
                        "run_id": "fit",
                        "dispatch_external_cpu_core_equiv": 8.0,
                        "external_cpu_core_equiv": 8.0,
                        "completion_model": _completion_model(10.0),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    holdout_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        **common,
                        "run_id": "holdout",
                        "dispatch_external_cpu_core_equiv": 13.0,
                        "external_cpu_core_equiv": 13.0,
                        "completion_model": _completion_model(10.0),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_completion_holdout_gate(
        fit_path=fit_path,
        holdout_path=holdout_path,
    )

    assert result["status"] == "PASS"
    assert result["rows"][0]["resource_state_match"] is True
    assert result["rows"][0]["ready"] is True


def test_holdout_gate_rejects_run_window_only_v2_telemetry(tmp_path):
    fit_path = tmp_path / "fit.json"
    holdout_path = tmp_path / "holdout.json"
    row = {
        "cell_id": "cpu_p64_external",
        "node": "node004",
        "workload_key": "cpu_heavy_local_bench",
        "workers": 64,
        "resource_state": "cpu_resident_external",
        "resource_telemetry_ready": True,
        "external_cpu_core_equiv": 12.0,
        "external_cpu_bucket": "external_c3_16",
        "observed_resource_state": {},
        "completion_model": _completion_model(10.0),
    }
    _payload = {"rows": [row]}
    fit_path.write_text(json.dumps(_payload), encoding="utf-8")
    holdout_path.write_text(json.dumps(_payload), encoding="utf-8")

    result = build_completion_holdout_gate(
        fit_path=fit_path,
        holdout_path=holdout_path,
    )

    assert result["status"] == "FAIL"
    assert result["rows"][0]["resource_state_match"] is False
    assert result["rows"][0]["ready"] is False
