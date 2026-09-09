from __future__ import annotations

import json

import pytest

from algorithm.experiments import (
    native_cpu_state_targeted_extension_campaign as extension,
)
from algorithm.experiments.phase_aware_stochastic_lcb_gate import (
    _state_signature,
)
from algorithm.experiments.file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
)


def _row(
    *,
    node: str = "node001",
    bucket: str = "external_c0_2",
    regime: str = "cpu_external_light",
    total_wall_s: float = 100.0,
) -> dict:
    return {
        "node": node,
        "workload_key": "freqduet_cpu_native",
        "workload_env": "freqduet",
        "resource_state": "empty",
        "observed_effective_resource_state": "empty",
        "dispatch_state_ready": True,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "dispatch_sample_count": DISPATCH_CPU_SAMPLE_COUNT,
        "dispatch_external_cpu_regime": regime,
        "dispatch_external_cpu_fraction": 0.01,
        "dispatch_external_cpu_fraction_p95": 0.02,
        "dispatch_external_cpu_bucket": bucket,
        "allocation_workers": 1,
        "colocation_count": 1,
        "init_cache_state": "warm",
        "eta_source": "task_native_durable_csv",
        "measurement_valid": True,
        "returncode": 0,
        "completion_model": {
            "completion_model_ready": True,
            "child_returncode": 0,
            "natural_exit": True,
            "interval_sample_count": 5,
            "total_units": 6,
            "total_wall_s": total_wall_s,
        },
    }


def test_minimum_conformal_count_is_nine_at_ten_percent_alpha():
    assert extension._minimum_conformal_calibration_count(0.10) == 9
    with pytest.raises(ValueError):
        extension._minimum_conformal_calibration_count(0.0)


def test_training_target_requires_all_three_pre_stable_signatures():
    nodes = tuple(f"node{index:03d}" for index in range(1, 7))
    stable = [
        {"rows": [_row(node=node) for node in nodes]}
        for _ in range(3)
    ]
    targets = extension._freeze_training_targets(stable)
    assert _state_signature(targets["node001"])[4] == "cpu_external_light"

    shifted = list(stable)
    shifted[-1] = {
        "rows": [
            _row(
                node=node,
                bucket=(
                    "external_c3_16"
                    if node == "node001"
                    else "external_c0_2"
                ),
                regime=(
                    "cpu_external_moderate"
                    if node == "node001"
                    else "cpu_external_light"
                ),
            )
            for node in nodes
        ]
    }
    with pytest.raises(ValueError, match="not pre-stable"):
        extension._freeze_training_targets(shifted)


def test_holdout_acceptance_uses_state_not_completion_time(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(extension, "ARTIFACT_ROOT", tmp_path)
    target = _state_signature(_row())

    def fake_attempt(
        *,
        artifact_path,
        run_id,
        node,
        seed_offset,
    ):
        attempt = 1 if "_a01_" in run_id else 2
        row = _row(
            bucket=(
                "external_c3_16"
                if attempt == 1
                else "external_c0_2"
            ),
            regime=(
                "cpu_external_moderate"
                if attempt == 1
                else "cpu_external_light"
            ),
            total_wall_s=1.0 if attempt == 1 else 999.0,
        )
        payload = {
            "run_id": run_id,
            "selected_count": 1,
            "all_completion_models_ready": True,
            "rows": [row],
        }
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload, "LAUNCHED"

    monkeypatch.setattr(
        extension,
        "_read_or_launch_one_node",
        fake_attempt,
    )
    result = extension._select_holdout_node(
        node="node001",
        extension_tag="unit",
        target_signature=target,
        max_attempts=2,
    )

    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["dispatch_state_match"] is False
    assert result["attempts"][1]["dispatch_state_match"] is True
    assert result["selected_row"]["completion_model"]["total_wall_s"] == 999.0
    assert result["performance_outcome_fields_used_for_selection"] == []
