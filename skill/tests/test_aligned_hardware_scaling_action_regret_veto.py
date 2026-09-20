"""Behavioral contracts for the post-result scaling action-regret veto."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import algorithm.experiments.aligned_hardware_scaling_action_regret_veto as veto


def _forbid_registration_execution(patcher: pytest.MonkeyPatch):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"registration unexpectedly called {name}")

        return fail

    for name in (
        "select_budgeted_frame",
        "capacity_certificate",
        "load_model",
    ):
        patcher.setattr(veto, name, forbidden(name))
    return calls


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("scaling_action_regret_registration")
    output = directory / "result.json"
    patcher = pytest.MonkeyPatch()
    calls = _forbid_registration_execution(patcher)
    try:
        protocol = veto.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def _copy_registration(source_output: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target_output = directory / source_output.name
    for source, target in zip(
        veto._paths(source_output)[1:],
        veto._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


def test_registration_freezes_two_cases_without_executing_diagnostic(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = veto._paths(output)

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert protocol["post_result_diagnostic"] is True
    assert protocol["independent_preregistration"] is False
    assert protocol["expected_selector_calls"] == 2
    assert protocol["expected_capacity_certificate_calls"] == 6
    contracts = protocol["source_contract"]["cases"]
    assert [row["case"]["case_id"] for row in contracts] == [
        "R8_K4096_alpha_witness",
        "R16_K4096_alpha_witness",
    ]
    assert [row["case"]["lanes"] for row in contracts] == [8, 16]
    assert all(row["case"]["budget"] == 4096 for row in contracts)

    frozen = json.loads(sources_path.read_text())
    assert veto.THIS_SOURCE in frozen
    assert all((veto.ROOT / name).read_text() == text for name, text in frozen.items())


def test_decision_rule_keeps_execution_and_scientific_outcomes_separate():
    assert veto._decision(False, []) == "INVALID_EXECUTION"
    assert veto._decision(
        True, [{"action_regret_veto": True}]
    ) == "ACTION_REGRET_VETO_CERTIFICATE_ONLY_STOP"
    assert veto._decision(
        True, [{"action_regret_veto": False}]
    ) == "NO_ACTION_REGRET_VETO_ADVANCE_TO_CERTIFICATE_BOUND_TEST"


def test_registered_run_replays_two_selectors_and_establishes_both_vetoes(
    registered_bundle,
    tmp_path,
    monkeypatch,
):
    source_output, _ = registered_bundle
    output = _copy_registration(source_output, tmp_path / "run")
    selector = veto.select_budgeted_frame
    certificate = veto.capacity_certificate
    calls = {"selector": 0, "capacity": 0}

    def counted_selector(*args, **kwargs):
        calls["selector"] += 1
        return selector(*args, **kwargs)

    def counted_certificate(*args, **kwargs):
        calls["capacity"] += 1
        return certificate(*args, **kwargs)

    monkeypatch.setattr(veto, "select_budgeted_frame", counted_selector)
    monkeypatch.setattr(veto, "capacity_certificate", counted_certificate)
    result = veto.run_registered(output)

    assert calls == {"selector": 2, "capacity": 6}
    assert result["status"] == "PASS"
    assert result["scientific_decision"] == (
        "ACTION_REGRET_VETO_CERTIFICATE_ONLY_STOP"
    )
    assert result["veto_case_count"] == 2
    assert all(all(row["checks"].values()) for row in result["cases"])
    assert all(row["action_regret_veto"] for row in result["cases"])

    r8, r16 = result["cases"]
    assert r8["challenger"]["service"] == [0, 6, 18]
    assert r8["challenger"]["score"] == pytest.approx(45.65399256493415)
    assert r8["normalized_action_regret_lower_bound"] == pytest.approx(
        0.14632689924658385
    )
    assert r8["best_possible_action_preserving_residual_margin_upper_bound"] < 0
    assert r16["challenger"]["service"] == [36, 0, 32]
    assert r16["challenger"]["score"] == pytest.approx(164.56928686411976)
    assert r16["normalized_action_regret_lower_bound"] == pytest.approx(
        0.6861101466811369
    )
    assert r16["best_possible_action_preserving_residual_margin_upper_bound"] < 0
    assert result["work_accounting"] == {
        "selector_calls": 2,
        "capacity_certificate_calls": 6,
        "fixed_challenger_reconstructions": 2,
        "full_grid_states": 0,
        "full_action_or_oracle_enumerations": 0,
        "arrival_tapes": 0,
        "trajectory_frames": 0,
        "rng_initializations": 0,
        "random_draws": 0,
    }


def test_registered_run_rejects_frozen_source_tampering_before_selector(
    registered_bundle,
    tmp_path,
    monkeypatch,
):
    source_output, _ = registered_bundle
    output = _copy_registration(source_output, tmp_path / "tampered")
    _, _, sources_path = veto._paths(output)
    frozen = json.loads(sources_path.read_text())
    frozen[veto.THIS_SOURCE] += "\n# tampered\n"
    sources_path.write_text(json.dumps(frozen))

    def forbidden(*args, **kwargs):
        pytest.fail("tampered registration reached selector execution")

    monkeypatch.setattr(veto, "select_budgeted_frame", forbidden)
    with pytest.raises(ValueError, match="registered action-regret audit sources changed"):
        veto.run_registered(output)
