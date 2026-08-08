from __future__ import annotations

import hashlib
import json
from pathlib import Path

from algorithm.experiments import critical_gpu_p10_transport_completion_campaign as campaign
from algorithm.experiments import critical_gpu_p10_transport_correction_gate as gate


def test_correction_scope_is_exactly_node007_distilgpt2_p10():
    cell = campaign.correction_cell(4)

    assert cell["node"] == "node007"
    assert cell["workload_key"] == "gpu_llm_distilgpt2"
    assert cell["profile"] == 10
    assert cell["total_task_count"] == 40
    assert campaign.EXPECTED_WAVES == tuple(range(1, 14))


def test_transport_integrity_audit_requires_exact_file_and_child_support(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(campaign, "RUN_ROOT", tmp_path)
    run_id = "unit-p10"
    raw_dir = tmp_path / run_id / "raw" / "profile_10_per_gpu"
    raw_dir.mkdir(parents=True)
    payload = b"".join(
        f"__SCHEDULEURM_LOG_BEGIN__ gpu{index // 10}_{index % 10}\n"
        f"__SCHEDULEURM_LOG_END__ gpu{index // 10}_{index % 10}\n"
        f"__SCHEDULEURM_RC__ gpu{index // 10}_{index % 10} 0\n".encode("ascii")
        for index in range(40)
    )
    digest = hashlib.sha256(payload).hexdigest()
    (raw_dir / "coordinated_profile_run_fetch_01.reconstructed.log").write_bytes(payload)
    (raw_dir / "coordinated_profile_run.combined.log").write_bytes(payload)
    (raw_dir / "coordinated_profile_run_fetch_01_meta.stdout").write_text(
        f"__SCHEDULEURM_FILE__ {len(payload)} {digest}\n",
        encoding="utf-8",
    )
    summary = {
        "rows": [
            {
                "completion_model": {
                    "progress_observation_count": 120,
                    "interval_sample_count": 119,
                }
            }
            for _ in range(40)
        ]
    }

    audit = campaign._transport_integrity_audit(run_id=run_id, summary=summary)
    assert audit["ready"] is True
    assert audit["combined_log_exact_match"] is True
    assert audit["log_begin_count"] == 40

    summary["rows"][0]["completion_model"]["progress_observation_count"] = 11
    assert campaign._transport_integrity_audit(
        run_id=run_id,
        summary=summary,
    )["ready"] is False


def test_source_audit_requires_the_registered_transport_defect(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            {
                "gate": "critical_gpu_completion_campaign",
                "measurement_protocol": campaign.SOURCE_PROTOCOL,
                "node": "node007",
                "wave": 1,
                "status": "PASS",
                "pass": True,
                "rows": [
                    {
                        "workload_key": campaign.WORKLOAD_KEY,
                        "profile": campaign.PROFILE,
                        "ready": True,
                        "summary": {
                            "rows": [
                                {
                                    "global_index": 0,
                                    "completion_model": {
                                        "progress_observation_count": 2,
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = campaign._source_v8_audit(path, wave=1)
    assert audit["ready"] is True
    assert audit["source_deficient_child_count"] == 1


def test_composite_gate_substitutes_only_registered_p10_transport_rows(
    tmp_path,
    monkeypatch,
):
    p10_id = "p10"
    expected_cells = {
        **{
            f"cell-{index}": {
                "service_cell_id": f"cell-{index}",
                "workload_key": f"workload-{index}",
                "profile": 1,
            }
            for index in range(8)
        },
        p10_id: {
            "service_cell_id": p10_id,
            "workload_key": campaign.WORKLOAD_KEY,
            "profile": campaign.PROFILE,
        },
    }
    source_paths = {}
    correction_paths = {}
    for wave in campaign.EXPECTED_WAVES:
        source = tmp_path / f"source-{wave}.json"
        correction = tmp_path / f"correction-{wave}.json"
        source.write_text(
            json.dumps({"measurement_code_manifest": {"sha256": "a" * 64}}),
            encoding="utf-8",
        )
        correction.write_text(
            json.dumps({"measurement_code_manifest": {"sha256": "b" * 64}}),
            encoding="utf-8",
        )
        source_paths[wave] = source
        correction_paths[wave] = correction

    def fake_audit_wave(*, node, wave, payload, expected_cells):
        observations = {
            f"cell-{index}": {"wave": wave, "cell": index}
            for index in range(8)
        }
        return (
            {"terminal_pass": True},
            [],
            [
                {
                    "wave": wave,
                    "cell": p10_id,
                    "code": "PROGRESS_SOURCE_INVALID",
                    "detail": "child 0 has too few progress observations",
                }
            ],
            observations,
        )

    def fake_correction_wave(**kwargs):
        return [], {"wave": kwargs["wave"], "cell": p10_id}, {"pass": True}

    monkeypatch.setattr(gate, "_expected_cells", lambda node: expected_cells)
    monkeypatch.setattr(gate, "_audit_wave", fake_audit_wave)
    monkeypatch.setattr(gate, "_audit_correction_wave", fake_correction_wave)
    monkeypatch.setattr(
        gate,
        "_build_certificate",
        lambda **kwargs: {
            "constructed": True,
            "finite_sample_rank_ready": True,
            "all_holdout_bounds_valid": True,
            "rows": [],
        },
    )

    report = gate.build_critical_gpu_p10_transport_correction_gate(
        artifact_root=tmp_path,
        source_paths=source_paths,
        correction_paths=correction_paths,
    )

    assert report["status"] == "PASS"
    assert report["ready_observation_count"] == 117
    assert all(row["composite_observation_count"] == 9 for row in report["wave_audits"])


def test_unregistered_source_error_cannot_be_hidden_by_correction(
    tmp_path,
    monkeypatch,
):
    expected_cells = {
        **{
            f"cell-{index}": {
                "service_cell_id": f"cell-{index}",
                "workload_key": f"workload-{index}",
                "profile": 1,
            }
            for index in range(8)
        },
        "p10": {
            "service_cell_id": "p10",
            "workload_key": campaign.WORKLOAD_KEY,
            "profile": campaign.PROFILE,
        },
    }
    source_paths = {}
    correction_paths = {}
    for wave in campaign.EXPECTED_WAVES:
        source = tmp_path / f"source-{wave}.json"
        correction = tmp_path / f"correction-{wave}.json"
        source.write_text(json.dumps({"measurement_code_manifest": {"sha256": "a" * 64}}), encoding="utf-8")
        correction.write_text(json.dumps({"measurement_code_manifest": {"sha256": "b" * 64}}), encoding="utf-8")
        source_paths[wave] = source
        correction_paths[wave] = correction

    def fake_audit_wave(*, node, wave, payload, expected_cells):
        observations = {f"cell-{index}": {} for index in range(8)}
        errors = [
            {
                "wave": wave,
                "cell": "p10",
                "code": "PROGRESS_SOURCE_INVALID",
                "detail": "child 0 has too few progress observations",
            }
        ]
        if wave == 7:
            errors.append(
                {
                    "wave": wave,
                    "cell": "cell-0",
                    "code": "CHECKPOINT_INVALID",
                    "detail": "checkpoint missing",
                }
            )
        return {"terminal_pass": True}, [], errors, observations

    monkeypatch.setattr(gate, "_expected_cells", lambda node: expected_cells)
    monkeypatch.setattr(gate, "_audit_wave", fake_audit_wave)
    monkeypatch.setattr(gate, "_audit_correction_wave", lambda **kwargs: ([], {}, {}))

    report = gate.build_critical_gpu_p10_transport_correction_gate(
        artifact_root=tmp_path,
        source_paths=source_paths,
        correction_paths=correction_paths,
    )

    assert report["status"] == "FAIL_VALIDATION"
    assert any(error["code"] == "CHECKPOINT_INVALID" for error in report["validation_errors"])
