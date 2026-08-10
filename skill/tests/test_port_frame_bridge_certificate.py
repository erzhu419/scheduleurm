from __future__ import annotations

from algorithm.experiments.port_frame_bridge_certificate import (
    build_certificate,
    inspect_lean_source,
    markdown_report,
)


LEAN_PASS = {
    "command": "lake env lean Scheduleurm/FrameBasedStability.lean",
    "lean_version_command_exit_code": 0,
    "lean_version": "test",
    "compile_exit_code": 0,
    "stdout": "",
    "stderr": "",
    "compile_ready": True,
}


def test_frame_bridge_closes_only_the_deterministic_virtual_interface():
    report = build_certificate(lean_check=LEAN_PASS)

    assert report["gate"]["pass"] is True
    assert report["gate"]["deterministic_virtual_frame_interface_ready"] is True
    assert report["gate"]["physical_service_frame_interface_ready"] is False
    assert report["gate"]["physical_port_positive_recurrence_claim_ready"] is False
    assert report["finite_family"]["candidate_count"] == 31
    assert report["finite_family"]["candidate_context_pair_count"] == 124
    assert len(report["frame_rows"]) == 4
    assert all(row["candidate_count"] == 31 for row in report["frame_rows"])
    assert all(
        row["observed_duration_max"] <= row["common_frame_horizon_H"]
        for row in report["frame_rows"]
    )
    assert "not physical departures" in markdown_report(report)


def test_frame_lean_source_has_all_required_theorems_and_no_placeholders():
    source = inspect_lean_source()

    assert source["source_contract_ready"] is True
    assert source["missing_theorems"] == []
    assert source["forbidden_tokens"] == []
