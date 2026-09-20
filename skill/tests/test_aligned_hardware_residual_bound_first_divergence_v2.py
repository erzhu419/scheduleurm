"""Tests for corrected physical reporting in first-divergence v2."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_residual_bound_first_divergence_v2 as diagnosis


@dataclass(frozen=True)
class _Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def test_lane_permutations_have_equal_physical_signature():
    model = SimpleNamespace(options=(
        _Option((1, 0), 2.0, ((1.0,), ())),
        _Option((0, 1), 3.0, ((), (2.5,))),
    ))
    first = {
        "prefix": {
            "indices": [0], "service": [1, 0], "duration": 2.0,
            "next_option_index": 1,
        }
    }
    second = {
        "prefix": {
            "indices": [1], "service": [0, 1], "duration": 3.0,
            "next_option_index": 0,
        }
    }

    first_candidate = diagnosis._attempted_candidate(model, first, (1, 1))
    second_candidate = diagnosis._attempted_candidate(model, second, (1, 1))

    assert first_candidate["search_indices"] == (0, 1)
    assert second_candidate["search_indices"] == (1, 0)
    assert (
        first_candidate["physical_signature"]
        == second_candidate["physical_signature"]
    )


def test_infeasible_candidate_has_no_physical_signature():
    model = SimpleNamespace(options=(
        _Option((2,), 2.0, ((1.0, 2.0),)),
    ))
    node = {
        "prefix": {
            "indices": [], "service": [0], "duration": 1.0,
            "next_option_index": 0,
        }
    }
    candidate = diagnosis._attempted_candidate(model, node, (1,))

    assert not candidate["feasible"]
    assert candidate["physical_signature"] is None


def test_v2_registration_freezes_correction_without_running_search(
    tmp_path, monkeypatch
):
    output = tmp_path / "diagnosis_v2.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("v2 registration must not execute search")

    monkeypatch.setattr(diagnosis, "trace_first_bound_divergence", forbidden)
    monkeypatch.setattr(diagnosis, "select_budgeted_frame", forbidden)
    monkeypatch.setattr(diagnosis, "select_residual_capacity_frame", forbidden)
    protocol = diagnosis.register(output)

    assert calls == []
    assert protocol["experiment"] == diagnosis.EXPERIMENT
    assert protocol["target_count"] == 4
    assert protocol["supersedes_result"] == diagnosis.SOURCE_V1_RESULT.name
    assert "excludes lane-ordered option indices" in protocol["correction"]
    assert not output.exists()
    assert {
        path.name for path in tmp_path.iterdir()
    } == {
        "diagnosis_v2_protocol.json",
        "diagnosis_v2_fixed_sources.json",
    }
