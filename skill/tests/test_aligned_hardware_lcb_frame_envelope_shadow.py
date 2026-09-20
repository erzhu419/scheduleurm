"""Contracts for the registered local LCB frame-duration shadow."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil

import pytest

import algorithm.experiments.aligned_hardware_lcb_frame_envelope_shadow as shadow
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


def _forbidden(name: str, calls: list[tuple[str, tuple, dict]]):
    def fail(*args, **kwargs):
        calls.append((name, args, kwargs))
        pytest.fail(f"registration unexpectedly called {name}")

    return fail


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lcb_frame_envelope_registration")
    output = directory / "shadow.json"
    calls: list[tuple[str, tuple, dict]] = []
    patcher = pytest.MonkeyPatch()
    patcher.setattr(shadow, "linprog", _forbidden("linprog", calls))
    patcher.setattr(
        shadow,
        "capacity_certificate",
        _forbidden("capacity_certificate", calls),
    )
    patcher.setattr(
        shadow,
        "_capacity_analysis",
        _forbidden("_capacity_analysis", calls),
    )
    try:
        protocol = shadow.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


@pytest.fixture(scope="module")
def executed_bundle(registered_bundle):
    output, protocol = registered_bundle
    result = shadow.run_registered(output)
    return output, protocol, result


def _copy_registration(source_output: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target_output = directory / source_output.name
    for source, target in zip(
        shadow._paths(source_output)[1:],
        shadow._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


def _cell(workload_env: str, profile: int) -> str:
    return (
        f"workload_env={workload_env}|"
        "node_bucket=gpu_3080ti_12gb_dual|resource_state=empty|"
        f"profile={profile}"
    )


EXPECTED_ENVELOPES = {
    "gpu_loaded:jtl110gpu2:cnn_after_llm": {
        "action_kind": "loaded",
        "source_row_key": "cnn_after_llm",
        "resident_source_row_key": _cell(
            "distilgpt2_forward_fp16_b2_s128", 1
        ),
        "point_duration_s": 11.514578111707017,
        "target_upper_duration_s": 13.25209242005076,
        "resident_fresh_p1_upper_duration_s": 8.103991160594,
        "upper_duration_s": 21.35608358064476,
        "composition": "serial_target_upper_plus_fresh_resident_p1_drain_upper",
    },
    "gpu_loaded:jtl110gpu2:llm_after_cnn": {
        "action_kind": "loaded",
        "source_row_key": "llm_after_cnn",
        "resident_source_row_key": _cell(
            "resnet50_synthetic_train_amp_b32_224", 1
        ),
        "point_duration_s": 11.936669718453734,
        "target_upper_duration_s": 14.771618074855004,
        "resident_fresh_p1_upper_duration_s": 9.178786404480242,
        "upper_duration_s": 23.950404479335246,
        "composition": "serial_target_upper_plus_fresh_resident_p1_drain_upper",
    },
    (
        "profile:gpu_cnn_torch_resnet50:gpu_3080ti_12gb_dual_jtl110gpu2:"
        "empty:none:w1:p1"
    ): {
        "action_kind": "dedicated",
        "source_row_key": _cell("resnet50_synthetic_train_amp_b32_224", 1),
        "point_duration_s": 8.093888054475512,
        "upper_duration_s": 9.178786404480242,
        "composition": "empty_simultaneous_upper_drain_completion",
    },
    (
        "profile:gpu_cnn_torch_resnet50:gpu_3080ti_12gb_dual_jtl110gpu2:"
        "empty:none:w1:p3"
    ): {
        "action_kind": "dedicated",
        "source_row_key": _cell("resnet50_synthetic_train_amp_b32_224", 3),
        "point_duration_s": 14.972417070959636,
        "upper_duration_s": 16.6071014979644,
        "composition": "empty_simultaneous_upper_drain_completion",
    },
    (
        "profile:gpu_heavy_jax_matmul:gpu_3080ti_12gb_dual_jtl110gpu2:"
        "empty:none:w1:p1"
    ): {
        "action_kind": "dedicated",
        "source_row_key": _cell("jax_matmul_8192", 1),
        "point_duration_s": 2.7928254592821196,
        "upper_duration_s": 3.12295819477921,
        "composition": "empty_simultaneous_upper_drain_completion",
    },
    (
        "profile:gpu_heavy_jax_matmul:gpu_3080ti_12gb_dual_jtl110gpu2:"
        "empty:none:w1:p3"
    ): {
        "action_kind": "dedicated",
        "source_row_key": _cell("jax_matmul_8192", 3),
        "point_duration_s": 6.820477375410869,
        "upper_duration_s": 7.621514394508295,
        "composition": "empty_simultaneous_upper_drain_completion",
    },
    (
        "profile:gpu_llm_distilgpt2:gpu_3080ti_12gb_dual_jtl110gpu2:"
        "empty:none:w1:p1"
    ): {
        "action_kind": "dedicated",
        "source_row_key": _cell("distilgpt2_forward_fp16_b2_s128", 1),
        "point_duration_s": 7.4174495042607305,
        "upper_duration_s": 8.103991160594,
        "composition": "empty_simultaneous_upper_drain_completion",
    },
    (
        "profile:gpu_llm_distilgpt2:gpu_3080ti_12gb_dual_jtl110gpu2:"
        "empty:none:w1:p3"
    ): {
        "action_kind": "dedicated",
        "source_row_key": _cell("distilgpt2_forward_fp16_b2_s128", 3),
        "point_duration_s": 9.462480184740949,
        "upper_duration_s": 10.530514595181701,
        "composition": "empty_simultaneous_upper_drain_completion",
    },
    (
        "profile:gpu_llm_distilgpt2:gpu_3080ti_12gb_dual_jtl110gpu2:"
        "empty:none:w1:p8"
    ): {
        "action_kind": "dedicated",
        "source_row_key": _cell("distilgpt2_forward_fp16_b2_s128", 8),
        "point_duration_s": 14.84929022616052,
        "upper_duration_s": 16.880782200650074,
        "composition": "empty_simultaneous_upper_drain_completion",
    },
}


def test_registration_is_pre_capacity_and_freezes_local_scope(registered_bundle):
    output, protocol = registered_bundle
    result_path, protocol_path, fixed_sources_path = shadow._paths(output)

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and fixed_sources_path.exists()
    assert protocol["experiment"] == shadow.EXPERIMENT
    assert protocol["loads"] == list(shadow.LOADS)
    assert protocol["primary_arrival_basis"] == (
        "point_model_frozen_arrival_vectors"
    )
    assert protocol["shadow_relative_loads_are_diagnostic_only"] is True
    assert protocol["expected_dedicated_action_count"] == 7
    assert protocol["expected_loaded_action_count"] == 2
    assert protocol["trajectory_runs"] == 0
    assert protocol["server_pulls"] == 0
    assert protocol["rng_draws"] == 0


def test_all_action_rows_have_exact_sources_and_numeric_durations(
    registered_bundle,
):
    _output, protocol = registered_bundle
    rows = {row["action_id"]: row for row in protocol["action_envelopes"]}
    shadow_options = {
        row["action_id"]: row for row in protocol["shadow_model"]["options"]
    }

    assert set(rows) == set(EXPECTED_ENVELOPES)
    for action_id, expected in EXPECTED_ENVELOPES.items():
        row = rows[action_id]
        for key, value in expected.items():
            if key.endswith("_duration_s"):
                assert row[key] == pytest.approx(value, abs=1e-12)
            else:
                assert row[key] == value
        assert row["upper_duration_s"] >= row["point_duration_s"]
        assert row["overlap_credit_units"] == 0.0
        option = shadow_options[action_id]
        assert option["duration"] == pytest.approx(
            row["upper_duration_s"], abs=1e-12
        )
        assert all(
            offset == pytest.approx(row["upper_duration_s"], abs=1e-12)
            for offsets in option["completion_offsets"]
            for offset in offsets
        )

        if row["action_kind"] == "dedicated":
            assert row["source_gate"] == shadow.EMPTY_GATE.name
            assert row["source_functional"] == "drain_completion"
        else:
            assert row["source_gate"] == shadow.LOADED_GATE.name
            assert row["resident_source_gate"] == shadow.EMPTY_GATE.name
            assert row["source_functionals"] == [
                "target_completion_time",
                "drain_completion",
            ]
            assert row["upper_duration_s"] == pytest.approx(
                row["target_upper_duration_s"]
                + row["resident_fresh_p1_upper_duration_s"],
                abs=1e-12,
            )


def test_combined_coverage_is_point_eight_and_not_a_95_percent_claim(
    registered_bundle,
):
    _output, protocol = registered_bundle
    coverage = protocol["coverage"]

    assert coverage == {
        "empty_source_nominal_simultaneous_coverage": 0.9,
        "loaded_source_nominal_simultaneous_coverage": 0.9,
        "combined_event_lower_bound": 0.8,
        "combination_rule": "P(empty and loaded) >= 1-alpha_empty-alpha_loaded",
        "independence_assumed": False,
        "product_0p81_claimed": False,
        "aspirational_combined_coverage": 0.95,
        "aspirational_0p95_ready": False,
        "current_calibration_wave_count_per_source": 9,
        "minimum_joint_paired_calibration_waves_for_finite_0p95_rank": 19,
        "minimum_per_source_calibration_waves_for_symmetric_bonferroni_0p95": 39,
        "coverage_scope": "one future exchangeable source-wave event, not all future frames",
    }


def _decision_rows(etas):
    return [
        {"load": load, "effective_slack_eta": eta}
        for load, eta in zip(shadow.LOADS, etas, strict=True)
    ]


def test_decision_has_advance_high_load_stop_lower_load_stop_and_invalid_branches():
    assert shadow._decision(_decision_rows((0.1, 0.01, 1e-5))) == (
        "ALL_FIXED_LOADS_LCB_CAPACITY_PASS_ADVANCE"
    )
    assert shadow._decision(_decision_rows((0.1, 0.01, 0.0))) == (
        "HIGH_LOAD_LCB_CAPACITY_FAIL_STOP"
    )
    assert shadow._decision(_decision_rows((0.1, 0.0, 0.1))) == (
        "LCB_CAPACITY_FAIL_STOP"
    )
    invalid = _decision_rows((0.1, 0.1, 0.1))
    invalid[-1]["load"] = 0.96
    with pytest.raises(ValueError, match="registered loads"):
        shadow._decision(invalid)


def test_small_real_run_uses_fixed_point_arrivals_for_the_decision(executed_bundle):
    _output, _protocol, result = executed_bundle
    fixed = result["fixed_arrival_capacity"]
    relative = result["shadow_relative_capacity_diagnostic"]
    point_model = load_model()

    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert [row["load"] for row in fixed] == list(shadow.LOADS)
    assert [row["load"] for row in relative] == list(shadow.LOADS)
    assert [row["effective_slack_eta"] for row in fixed] == pytest.approx(
        [
            0.0862781789804414,
            0.020239834409225166,
            -0.012779337876382968,
        ],
        abs=1e-12,
    )
    assert [row["conditional_negative_drift_certified"] for row in fixed] == [
        True,
        True,
        False,
    ]
    for load, fixed_row, relative_row in zip(
        shadow.LOADS, fixed, relative, strict=True
    ):
        point = capacity_certificate(point_model, load)
        assert fixed_row["arrival_basis"] == (
            "point_model_frozen_arrival_vector"
        )
        assert fixed_row["arrival_rates"] == pytest.approx(
            point["arrival_rates"], abs=1e-12
        )
        assert relative_row["arrival_basis"] == (
            "shadow_capacity_renormalized_diagnostic_only"
        )
    assert result["maximum_fixed_arrival_load"][
        "maximum_point_model_load_factor"
    ] == pytest.approx(0.891945828778605, abs=1e-12)
    assert result["scientific_decision"] == shadow._decision(fixed) == (
        "HIGH_LOAD_LCB_CAPACITY_FAIL_STOP"
    )
    assert result["development_advance"] is False
    assert result["coverage_decision"] == (
        "COMBINED_0P8_ONLY_0P95_NOT_CERTIFIED"
    )


def test_runner_has_no_trajectory_server_or_rng_path(executed_bundle):
    _output, protocol, result = executed_bundle
    source = Path(shadow.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "random" not in imported_modules
    assert "numpy.random" not in imported_modules
    assert not {
        "simulate_frames",
        "generate_arrival_tape",
        "run_trajectory",
        "run_drain",
        "pull_server_data",
    }.intersection(called_names)
    assert protocol["trajectory_runs"] == protocol["server_pulls"] == 0
    assert protocol["rng_draws"] == 0
    assert result["work"]["trajectory_runs"] == 0
    assert result["work"]["server_pulls"] == 0
    assert result["work"]["rng_draws"] == 0


def test_write_new_registration_result_and_tamper_guards(
    executed_bundle, tmp_path, monkeypatch,
):
    output, _protocol, _result = executed_bundle
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text('{"preserve": true}\n', encoding="utf-8")
    with pytest.raises(FileExistsError):
        shadow._write_new(sentinel, {"preserve": False})
    assert json.loads(sentinel.read_text(encoding="utf-8")) == {"preserve": True}

    with pytest.raises(ValueError, match="artifacts already exist"):
        shadow.register(output)
    with pytest.raises(ValueError, match="result already exists"):
        shadow.run_registered(output)

    def forbidden(*args, **kwargs):
        pytest.fail("tampered registration reached capacity analysis")

    monkeypatch.setattr(shadow, "_capacity_analysis", forbidden)
    protocol_output = _copy_registration(output, tmp_path / "protocol")
    protocol_path = shadow._paths(protocol_output)[1]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["loads"][-1] = 0.96
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol or sources changed"):
        shadow.run_registered(protocol_output)

    source_output = _copy_registration(output, tmp_path / "fixed_sources")
    fixed_sources_path = shadow._paths(source_output)[2]
    fixed_sources = json.loads(fixed_sources_path.read_text(encoding="utf-8"))
    fixed_sources[shadow.THIS_SOURCE] += "\n# tampered\n"
    fixed_sources_path.write_text(json.dumps(fixed_sources), encoding="utf-8")
    with pytest.raises(ValueError, match="code sources changed"):
        shadow.run_registered(source_output)
