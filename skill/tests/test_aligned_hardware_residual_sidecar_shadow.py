"""Focused tests for the registered legacy-priority residual sidecar state shadow."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_residual_sidecar_shadow as shadow


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("residual_sidecar_registration")
    output = directory / "shadow.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("registration must not execute a selector or search trace")

    patcher = pytest.MonkeyPatch()
    patcher.setattr(shadow, "select_residual_sidecar_frame", forbidden)
    patcher.setattr(shadow, "trace_residual_sidecar_frame_search", forbidden)
    try:
        protocol = shadow.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def test_registration_reuses_compact_roster_without_copying_or_running(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = shadow._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")
    roster = json.loads(shadow.SOURCE_STATE_ROSTER.read_text())

    assert result_path == output and not output.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert protocol["reused_registered_state_roster"] == shadow.SOURCE_STATE_ROSTER.name
    assert protocol["role"].startswith("post-outcome development")
    assert protocol["registration_context"][
        "integrated_regression_state_indices"
    ] == [424, 425, 437, 438]
    assert protocol["rule"]["heap_priority"] == (
        "(-legacy_v4_upper_bound, virtual_legacy_serial)"
    )
    assert protocol["rule"]["pruning_rule"] == (
        "residual_upper_bound < incumbent"
    )
    sources = json.loads(sources_path.read_text())
    assert len(sources) == 11
    assert roster["state_count"] == len(roster["states"]) == shadow.STATE_COUNT
    assert len({tuple(row["queue"]) for row in roster["states"]}) == shadow.STATE_COUNT


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    for source, target in zip(
        shadow._paths(source_output)[1:], shadow._paths(target_output)[1:], strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


@pytest.mark.parametrize("tamper", ("rule", "roster", "source"))
def test_tampering_fails_before_selector_calls(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, sources_path = shadow._paths(output)

    if tamper == "rule":
        payload = json.loads(protocol_path.read_text())
        payload["rule"]["ordered_key"] = "tampered"
        protocol_path.write_text(json.dumps(payload))
        message = "registered residual-sidecar order changed"
    elif tamper == "roster":
        roster_path = tmp_path / shadow.SOURCE_STATE_ROSTER.name
        payload = json.loads(shadow.SOURCE_STATE_ROSTER.read_text())
        payload["states"][0]["queue"][0] += 1
        roster_path.write_text(json.dumps(payload))
        monkeypatch.setattr(shadow, "SOURCE_STATE_ROSTER", roster_path)
        message = "registered residual-sidecar states or model changed"
    else:
        payload = json.loads(sources_path.read_text())
        name = "algorithm/residual_sidecar_frame_candidates.py"
        payload[name] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "registered residual-sidecar sources changed"

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("invalid registered input reached search execution")

    monkeypatch.setattr(shadow, "select_residual_sidecar_frame", forbidden)
    monkeypatch.setattr(shadow, "trace_residual_sidecar_frame_search", forbidden)
    with pytest.raises(ValueError, match=message):
        shadow.run_registered(output)
    assert calls == []
    assert not output.exists()


@dataclass(frozen=True)
class _Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _option(requirements, duration):
    return _Option(
        requirements=requirements,
        duration=duration,
        completion_offsets=tuple((duration,) * count for count in requirements),
    )


def test_crafted_state_preserves_order_and_tightens_valid_support_bound(
    monkeypatch,
):
    model = SimpleNamespace(
        lanes=3,
        workloads=("a", "b", "c"),
        options=(
            _option((1, 2, 1), 4.0),
            _option((2, 2, 3), 4.0),
            _option((2, 2, 1), 4.0),
            _option((1, 2, 2), 2.0),
            _option((3, 0, 3), 2.0),
        ),
    )
    monkeypatch.setattr(shadow, "SMALL_BUDGET", 6)
    state = {
        "queue": [6, 7, 5],
        "population_visits": [],
        "legacy_k26": {
            "option_indices": [3, 3, None],
            "service": [2, 4, 4],
            "duration": 2.0,
            "score": 30.0,
            "upper_bound": 49.5,
            "gap": 19.5,
            "epsilon": 19.5 / 18.0,
            "attempts": 6,
            "exact": False,
        },
        "support_reference_score": 31.5,
        "legacy_first_exact_score_attempt": 7,
    }

    evaluated = shadow._evaluate_state(model, 0, state)
    selected = evaluated["variant_k26"]

    assert selected["ordered_option_indices"] == (1, 4, 2, 3, 0, None)
    assert selected["score"] == 30.0
    assert selected["upper_bound"] == 42.5
    assert selected["gap"] == 12.5
    assert evaluated["certified_gap_delta_over_legacy_k26"] == -7.0
    assert evaluated["k26_bound_strictly_activated"]
    assert evaluated["k26_bound_strict_tightening_count"] == 1
    assert evaluated["support_outcome"] == "unchanged"
    assert evaluated["legacy_v4_order_preserved"]
    assert evaluated["work_structure_checks_pass"]
    assert evaluated["selection_and_bound_checks_pass"]
    assert evaluated["variant_k120"]["exact"]
    assert evaluated["variant_k120"]["score"] == 31.5


def test_four_integrated_regressions_restore_registered_legacy_support():
    model = shadow.load_model()
    roster = json.loads(shadow.SOURCE_STATE_ROSTER.read_text())
    for index in (424, 425, 437, 438):
        evaluated = shadow._evaluate_state(model, index, roster["states"][index])
        assert evaluated["support_outcome"] == "unchanged"
        assert evaluated["support_score_delta_over_legacy_k26"] == 0.0
        assert evaluated["variant_k26"]["option_indices"] == (6, 7)
        assert evaluated["upper_bound_delta_over_legacy_k26"] <= 1e-10
        assert evaluated["certified_gap_delta_over_legacy_k26"] <= 1e-10
        assert evaluated["selection_and_bound_checks_pass"]


def _population_rows(*, support_improvement=True, activation=3):
    rows = []
    for load in shadow.LOADS:
        rows.append({
            "load": load,
            "strict_bound_activation_frame_count": activation,
            "safe_pruning_frame_count": activation,
            "legacy": {
                "mean_finite_budget_support_regret": 1.0,
                "strict_support_loss_frame_count": 10,
                "mean_certified_gap": 0.4,
                "exact_certified_frame_fraction": 0.7,
                "exact_support_score_frame_fraction": 0.8,
                "mean_candidate_attempts": 26.0,
            },
            "variant": {
                "mean_finite_budget_support_regret": (
                    0.5 if support_improvement else 1.0
                ),
                "strict_support_loss_frame_count": (
                    8 if support_improvement else 10
                ),
                "mean_certified_gap": 0.3,
                "exact_certified_frame_fraction": 0.75,
                "exact_support_score_frame_fraction": (
                    0.85 if support_improvement else 0.8
                ),
                "mean_candidate_attempts": 25.0,
            },
        })
    return rows


def test_advance_gate_uses_primary_and_sensitivity_vetoes():
    primary = _population_rows()
    sensitivity = _population_rows()
    evaluations = [{
        "support_outcome": "improved",
        "upper_bound_delta_over_legacy_k26": -0.1,
        "certified_gap_delta_over_legacy_k26": -0.1,
    }]
    checks = shadow._advance_checks(primary, sensitivity, evaluations, True)
    assert all(checks.values())

    primary[1]["variant"]["mean_certified_gap"] = 0.41
    checks = shadow._advance_checks(primary, sensitivity, evaluations, True)
    assert not checks["primary_mean_certified_gap_nonworse_at_every_load"]

    primary = _population_rows()
    sensitivity[2]["variant"]["exact_support_score_frame_fraction"] = 0.79
    checks = shadow._advance_checks(primary, sensitivity, evaluations, True)
    assert not checks[
        "sensitivity_exact_support_fraction_nonworse_at_every_load"
    ]

    checks = shadow._advance_checks(
        primary, _population_rows(),
        [{
            "support_outcome": "regressed",
            "upper_bound_delta_over_legacy_k26": 0.0,
            "certified_gap_delta_over_legacy_k26": 0.0,
        }], True,
    )
    assert not checks["no_registered_state_has_lower_k26_support"]


def test_decision_distinguishes_invalid_advance_certificate_only_and_closed():
    passing = shadow._advance_checks(
        _population_rows(), _population_rows(),
        [{
            "support_outcome": "improved",
            "upper_bound_delta_over_legacy_k26": -0.1,
            "certified_gap_delta_over_legacy_k26": -0.1,
        }], True,
    )
    advance, certificate_only, decision, next_step = shadow._decision(passing)
    assert advance and not certificate_only
    assert decision == "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"
    assert next_step.startswith("preregister full development trajectories")

    certificate = shadow._advance_checks(
        _population_rows(support_improvement=False),
        _population_rows(support_improvement=False),
        [{
            "support_outcome": "unchanged",
            "upper_bound_delta_over_legacy_k26": -0.1,
            "certified_gap_delta_over_legacy_k26": -0.1,
        }],
        True,
    )
    advance, certificate_only, decision, next_step = shadow._decision(certificate)
    assert not advance and certificate_only
    assert decision == "CERTIFICATE_OR_WORK_ONLY_STOP"
    assert next_step.startswith("record certificate or candidate-attempt")

    closed = dict(passing)
    closed["primary_mean_certified_gap_nonworse_at_every_load"] = False
    advance, certificate_only, decision, next_step = shadow._decision(closed)
    assert not advance and not certificate_only
    assert decision == "CLOSE_WITHOUT_TRAJECTORIES"
    assert next_step.startswith("close the legacy-priority")

    invalid = dict(passing)
    invalid["execution_passed"] = False
    advance, certificate_only, decision, next_step = shadow._decision(invalid)
    assert not advance and not certificate_only
    assert decision == "INVALID_EXECUTION"
    assert next_step.startswith("invalid execution")
