"""Focused tests for the registered duration-order fixed-state shadow."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_duration_tie_order_shadow as shadow


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    """Register once while proving registration cannot execute either search."""
    directory = tmp_path_factory.mktemp("duration_order_registration")
    output = directory / "shadow.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("registration must not execute a selector or search trace")

    patcher = pytest.MonkeyPatch()
    patcher.setattr(shadow, "select_duration_tie_frame", forbidden)
    patcher.setattr(shadow, "trace_duration_tie_frame_search", forbidden)
    try:
        protocol = shadow.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def test_registration_freezes_compact_5003_state_roster_and_18_cells(
    registered_bundle,
):
    output, protocol = registered_bundle
    _, protocol_path, states_path, sources_path = shadow._paths(output)
    roster = json.loads(states_path.read_text())

    assert not output.exists()
    assert protocol_path.exists() and states_path.exists() and sources_path.exists()
    assert roster["state_count"] == shadow.STATE_COUNT == 5003
    assert len(roster["states"]) == 5003
    assert len({tuple(row["queue"]) for row in roster["states"]}) == 5003
    assert set(roster) == {"source_experiment", "state_count", "states"}
    assert all(
        set(row) == {
            "queue", "population_visits", "legacy_k26",
            "support_reference_score", "legacy_first_exact_score_attempt",
        }
        for row in roster["states"]
    )
    assert "legacy_first_exact_score_attempt" in protocol["registered_state_schema"]
    assert "unique physical optimum" in protocol["variant_physical_rank"]
    # The registered input is the state projection, not another copy of the
    # 14 MB diagnostic with trajectories and verbose per-state diagnostics.
    assert states_path.stat().st_size < shadow.SOURCE_DIAGNOSTIC.stat().st_size / 4

    observed = defaultdict(lambda: [0, 0.0])
    for state in roster["states"]:
        for load, seed, source_budget, frames, visited_time in state["population_visits"]:
            cell = (load, seed, source_budget)
            observed[cell][0] += frames
            observed[cell][1] += visited_time
    expected = {
        (row["load"], row["seed"], row["source_policy_budget"]): (
            row["frame_count"], row["legacy_visited_time_s"],
        )
        for row in protocol["source_population_cells"]
    }
    assert len(observed) == len(expected) == 18
    assert set(observed) == set(expected)
    for cell, (frames, visited_time) in observed.items():
        assert frames == expected[cell][0]
        assert visited_time == pytest.approx(expected[cell][1], abs=1e-8)


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    source_paths = shadow._paths(source_output)
    target_paths = shadow._paths(target_output)
    for source, target in zip(source_paths[1:], target_paths[1:], strict=True):
        shutil.copyfile(source, target)
    return target_output


@pytest.mark.parametrize("tamper", ("rule", "roster", "source"))
def test_tampering_fails_before_any_selector_call(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, states_path, sources_path = shadow._paths(output)

    if tamper == "rule":
        payload = json.loads(protocol_path.read_text())
        payload["rule"]["ordered_key"] = "tampered"
        protocol_path.write_text(json.dumps(payload))
        message = "registered duration-order rule changed"
    elif tamper == "roster":
        payload = json.loads(states_path.read_text())
        payload["states"][0]["queue"][0] += 1
        states_path.write_text(json.dumps(payload))
        message = "registered duration-order states or model changed"
    else:
        payload = json.loads(sources_path.read_text())
        name = "algorithm/duration_tie_frame_candidates.py"
        payload[name] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "registered duration-order sources changed"

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("invalid registered input reached search execution")

    monkeypatch.setattr(shadow, "select_duration_tie_frame", forbidden)
    monkeypatch.setattr(shadow, "trace_duration_tie_frame_search", forbidden)
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
        completion_offsets=tuple(
            (duration,) * count for count in requirements
        ),
    )


def test_crafted_state_checks_support_bound_without_using_physical_tie_as_gate(
    monkeypatch,
):
    # The score certificate can close while equality pruning leaves a different
    # physical support tie than the score->duration->index Cartesian choice.
    model = SimpleNamespace(
        lanes=2,
        workloads=("a", "b"),
        options=(
            _option((3, 1), 4.0),
            _option((1, 0), 6.0),
            _option((1, 3), 2.0),
            _option((2, 0), 3.0),
        ),
    )
    monkeypatch.setattr(shadow, "SMALL_BUDGET", 8)
    state = {
        "queue": [5, 4],
        "population_visits": [],
        # This shorter action has the same support as the duration-order
        # selector's eventual action. Physical equality is deliberately not an
        # execution or advancement condition.
        "legacy_k26": {
            "option_indices": [2, 3],
            "service": [3, 3],
            "duration": 3.0,
            "score": 9.0,
            "upper_bound": 13.0,
            "gap": 4.0,
            "epsilon": 4.0 / 9.0,
            "attempts": 8,
            "exact": False,
        },
        "support_reference_score": 9.0,
        "legacy_first_exact_score_attempt": 8,
    }

    evaluated = shadow._evaluate_state(model, 0, state)
    selected = evaluated["variant_k26"]
    exact = evaluated["variant_k120"]

    assert selected["score"] <= state["support_reference_score"]
    assert state["support_reference_score"] <= selected["upper_bound"]
    assert evaluated["variant_gap_identity_error"] == pytest.approx(0.0)
    assert exact["exact"]
    assert exact["score"] == state["support_reference_score"]
    assert exact["option_indices"] == (2, 0)
    assert exact["duration"] == 4.0
    assert evaluated["selected_physical_signature_changed_from_legacy_k26"]
    assert evaluated["support_outcome"] == "unchanged"
    assert evaluated["selection_and_bound_checks_pass"]


def _passing_primary_summaries():
    return [
        {
            "load": load,
            "legacy": {
                "mean_finite_budget_support_regret": 1.0,
                "strict_support_loss_frame_count": 10,
                "mean_certified_gap_per_visible_job": 0.20,
                "mean_certified_gap": 0.40,
                "exact_certified_frame_fraction": 0.70,
                "exact_support_score_frame_fraction": 0.80,
            },
            "variant": {
                "mean_finite_budget_support_regret": 0.5,
                "strict_support_loss_frame_count": 8,
                "mean_certified_gap_per_visible_job": 0.15,
                "mean_certified_gap": 0.30,
                "exact_certified_frame_fraction": 0.75,
                "exact_support_score_frame_fraction": 0.85,
            },
        }
        for load in shadow.LOADS
    ]


def test_advance_gate_rejects_any_state_score_regression():
    checks = shadow._advance_checks(
        _passing_primary_summaries(),
        [{"support_outcome": "improved"}, {"support_outcome": "regressed"}],
        execution_passed=True,
    )
    assert not checks["no_registered_state_has_lower_k26_support"]
    assert not all(checks.values())


def test_advance_gate_rejects_gap_worsening_at_any_primary_load():
    summaries = _passing_primary_summaries()
    summaries[1]["variant"]["mean_certified_gap"] = 0.41

    checks = shadow._advance_checks(
        summaries,
        [{"support_outcome": "improved"}],
        execution_passed=True,
    )
    assert not checks[
        "primary_mean_certified_gap_nonworse_at_every_load"
    ]
    assert not all(checks.values())
