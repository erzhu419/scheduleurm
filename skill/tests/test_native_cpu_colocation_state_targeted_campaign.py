from __future__ import annotations

import json

from algorithm.experiments import (
    native_cpu_colocation_state_targeted_campaign as targeted,
)
from algorithm.experiments import native_cpu_colocation_completion_matrix as matrix


def _ready_row(cell_id: str, *, regime: str, makespan_s: float) -> dict:
    return {
        "calibration_cell_id": cell_id,
        "node": "node001",
        "workload_key": "freqduet_cpu_native",
        "workload_env": "freqduet",
        "observed_effective_resource_state": "empty",
        "dispatch_external_cpu_regime": regime,
        "required_dispatch_regime": regime,
        "allocation_workers": 1,
        "colocation_count": 1,
        "aggregate_service_units": 6.0,
        "init_cache_state": "warm",
        "profile_axis": "colocation_count",
        "measurement_protocol": matrix.MEASUREMENT_PROTOCOL,
        "measurement_valid": True,
        "all_tasks_ready": True,
        "near_synchronous_start": True,
        "dispatch_state_ready": True,
        "dispatch_sample_count": matrix.DISPATCH_CPU_SAMPLE_COUNT,
        "eta_source": "task_native_durable_csv",
        "makespan_s": makespan_s,
        "task_results": [
            {
                "measurement_valid": True,
                "returncode": 0,
                "completion_model": {
                    "completion_model_ready": True,
                    "natural_exit": True,
                    "interval_sample_count": 5,
                },
            }
        ],
    }


def test_cell_retry_freezes_seed_and_accepts_without_outcome_selection(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(targeted, "ARTIFACT_ROOT", tmp_path)
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        cell = matrix.colocation_cells(
            run_id=kwargs["run_id"],
            nodes=kwargs["nodes"],
            profiles=kwargs["profiles"],
            kinds=kwargs["kinds"],
            tasks_per_node=kwargs["tasks_per_node"],
            seed_base=kwargs["seed_base"],
        )[0]
        cell_id = matrix._calibration_cell_id_for_cell(cell)
        if len(calls) == 1:
            row = {
                **_ready_row(
                    cell_id,
                    regime="cpu_external_light",
                    makespan_s=1.0,
                ),
                "required_dispatch_regime": "cpu_external_idle",
                "measurement_valid": False,
                "all_tasks_ready": False,
            }
            return {
                "measurement_protocol": matrix.MEASUREMENT_PROTOCOL,
                "status": "RUN_FINISHED_WITH_GAPS",
                "rows": [row],
            }
        return {
            "measurement_protocol": matrix.MEASUREMENT_PROTOCOL,
            "status": "COMPLETE",
            "rows": [
                _ready_row(
                    cell_id,
                    regime="cpu_external_idle",
                    makespan_s=999.0,
                )
            ],
        }

    monkeypatch.setattr(
        targeted,
        "build_native_cpu_colocation_completion_matrix",
        fake_build,
    )

    selected, attempts = targeted._run_cell_until_accepted(
        node="node001",
        kind="freqduet",
        profile=1,
        role="training",
        role_index=0,
        replicate=1,
        tag="test",
        target_regime="cpu_external_idle",
        max_attempts=3,
        seed_base=100,
        init_cache_state="warm",
    )

    assert selected is not None
    assert selected["makespan_s"] == 999.0
    assert len(attempts) == 2
    assert attempts[0]["accepted"] is False
    assert attempts[1]["accepted"] is True
    assert selected["performance_outcome_fields_used_for_selection"] == []
    assert calls[0]["seed_base"] == calls[1]["seed_base"]
    assert calls[0]["deploy_bundle"] is False
    assert set(calls[1]["required_dispatch_regimes"].values()) == {
        "cpu_external_idle"
    }


def test_composite_artifacts_require_exact_cell_set(monkeypatch, tmp_path):
    monkeypatch.setattr(targeted, "ARTIFACT_ROOT", tmp_path)
    cell_ids = ("cell-a", "cell-b")
    selected = {
        role: {
            replicate: {
                cell_id: {
                    "calibration_cell_id": cell_id,
                    "node": "node001" if cell_id == "cell-a" else "node002",
                }
                for cell_id in cell_ids
            }
            for replicate in range(
                1,
                {
                    "training": targeted.TRAINING_REPEATS,
                    "calibration": targeted.CALIBRATION_REPEATS,
                    "holdout": targeted.HOLDOUT_REPEATS,
                }[role]
                + 1,
            )
        }
        for role in ("training", "calibration", "holdout")
    }
    lanes = [
        {
            "node": "combined",
            "ready": True,
            "selected": selected,
        }
    ]

    paths = targeted._write_composite_artifacts(
        tag="test",
        profile=1,
        lane_results=lanes,
        expected_cell_ids=cell_ids,
        target_regimes={
            "node001": "cpu_external_idle",
            "node002": "cpu_external_light",
        },
    )

    assert len(paths["training"]) == 3
    assert len(paths["calibration"]) == 9
    assert len(paths["holdout"]) == 1
    payload = json.loads(paths["holdout"][0].read_text(encoding="utf-8"))
    assert payload["all_completion_models_ready"] is True
    assert [row["calibration_cell_id"] for row in payload["rows"]] == [
        "cell-a",
        "cell-b",
    ]
    assert payload["performance_outcome_fields_used_for_selection"] == []
