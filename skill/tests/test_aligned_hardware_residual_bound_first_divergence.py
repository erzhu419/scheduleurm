"""Tests for the registered residual-bound first-divergence diagnosis."""
from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

import algorithm.experiments.aligned_hardware_residual_bound_first_divergence as diagnosis
from algorithm.experiments.residual_bound_first_divergence_trace import (
    trace_first_bound_divergence,
)


@dataclass(frozen=True)
class _Option:
    requirements: tuple[int, ...]
    duration: float


def _priority_fixture_trace():
    options = (
        _Option((0, 3, 2), 7.0),
        _Option((3, 1, 2), 5.0),
        _Option((1, 0, 3), 2.0),
        _Option((1, 0, 1), 2.0),
    )
    return trace_first_bound_divergence(
        options, (4, 7, 2), lanes=3, budget=20
    )


def test_priority_classification_requires_cross_live_frontier_evidence():
    traced = _priority_fixture_trace()
    assert diagnosis._classification_evidence_valid(traced)

    traced["divergence"]["legacy_top_present_in_residual_frontier"] = False
    assert not diagnosis._classification_evidence_valid(traced)


def test_registration_freezes_all_and_only_four_source_regressions(
    tmp_path, monkeypatch
):
    output = tmp_path / "diagnosis.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("registration must not execute a selector or divergence trace")

    monkeypatch.setattr(diagnosis, "trace_first_bound_divergence", forbidden)
    monkeypatch.setattr(diagnosis, "select_budgeted_frame", forbidden)
    monkeypatch.setattr(diagnosis, "select_residual_capacity_frame", forbidden)
    protocol = diagnosis.register(output)

    assert calls == []
    assert protocol["target_count"] == 4
    assert tuple(row["state_index"] for row in protocol["targets"]) == (
        424, 425, 437, 438
    )
    assert protocol["budget"] == 26
    assert protocol["role"].startswith("post-outcome deterministic")
    assert not output.exists()
    assert {
        path.name for path in tmp_path.iterdir()
    } == {
        "diagnosis_protocol.json",
        "diagnosis_fixed_sources.json",
    }


def test_registered_protocol_tamper_fails_before_search(
    tmp_path, monkeypatch
):
    output = tmp_path / "diagnosis.json"
    diagnosis.register(output)
    _, protocol_path, _ = diagnosis._paths(output)
    protocol = json.loads(protocol_path.read_text())
    protocol["budget"] = 27
    protocol_path.write_text(json.dumps(protocol))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("invalid registered input reached search execution")

    monkeypatch.setattr(diagnosis, "trace_first_bound_divergence", forbidden)
    monkeypatch.setattr(diagnosis, "select_budgeted_frame", forbidden)
    monkeypatch.setattr(diagnosis, "select_residual_capacity_frame", forbidden)
    with pytest.raises(
        ValueError, match="registered residual first-divergence inputs changed"
    ):
        diagnosis.run_registered(output)
    assert calls == []
