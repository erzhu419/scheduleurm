"""Focused tests for the registered quotient authored-transition screen."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
import algorithm.experiments.aligned_hardware_permutation_quotient_transition as transition


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    """Register once while proving registration cannot execute either arm."""
    directory = tmp_path_factory.mktemp("permutation_quotient_transition_registration")
    output = directory / "transition.json"
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"registration unexpectedly called {name}")

        return fail

    patcher = pytest.MonkeyPatch()
    patcher.setattr(transition, "select_budgeted_frame", forbidden("legacy selector"))
    patcher.setattr(
        transition,
        "select_permutation_quotient_frame",
        forbidden("quotient selector"),
    )
    patcher.setattr(
        transition,
        "drain_initial_population",
        forbidden("authored drain"),
    )
    try:
        protocol = transition.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def test_registration_embeds_only_the_16_compact_cases(registered_bundle):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = transition._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")
    cases = protocol["case_roster"]
    compact_keys = {
        "state_index",
        "queue",
        "primary_population_visits",
        "legacy_option_indices",
        "variant_option_indices",
        "registered_support_score_delta",
    }

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert "states" not in protocol
    assert protocol["registered_state_roster"] == transition.REGISTERED_STATES.name
    assert protocol["state_count"] == len(cases) == 16
    assert protocol["registered_drain_count"] == 64
    assert len({row["state_index"] for row in cases}) == 16
    assert len({tuple(row["queue"]) for row in cases}) == 16
    assert all(set(row) == compact_keys for row in cases)


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    for source, target in zip(
        transition._paths(source_output)[1:],
        transition._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


@pytest.mark.parametrize("tamper", ("protocol", "case", "source", "predecessor"))
def test_tampering_fails_before_selector_or_drain_execution(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, sources_path = transition._paths(output)

    if tamper == "protocol":
        payload = json.loads(protocol_path.read_text())
        payload["budget"] += 1
        protocol_path.write_text(json.dumps(payload))
        message = "registered permutation-quotient transition protocol changed"
    elif tamper == "case":
        payload = json.loads(protocol_path.read_text())
        payload["case_roster"][0]["queue"][0] += 1
        protocol_path.write_text(json.dumps(payload))
        message = "registered permutation-quotient transition protocol changed"
    elif tamper == "source":
        payload = json.loads(sources_path.read_text())
        payload[transition.NEW_SOURCES[0]] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "registered permutation-quotient transition sources changed"
    else:
        predecessor_path = tmp_path / transition.PREDECESSOR.name
        payload = json.loads(transition.PREDECESSOR.read_text())
        payload["scientific_decision"] = "TAMPERED"
        predecessor_path.write_text(json.dumps(payload))
        monkeypatch.setattr(transition, "PREDECESSOR", predecessor_path)
        message = "transition requires the completed advancing quotient shadow"

    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"tampered registration unexpectedly called {name}")

        return fail

    monkeypatch.setattr(
        transition, "select_budgeted_frame", forbidden("legacy selector")
    )
    monkeypatch.setattr(
        transition,
        "select_permutation_quotient_frame",
        forbidden("quotient selector"),
    )
    monkeypatch.setattr(
        transition,
        "drain_initial_population",
        forbidden("authored drain"),
    )
    with pytest.raises(ValueError, match=message):
        transition.run_registered(output)
    assert calls == []
    assert not output.exists()


def _toy_model() -> FrameModel:
    return FrameModel(
        "toy-node",
        "toy-transition",
        2,
        ("a", "b"),
        (1.0, 1.0),
        (
            FrameOption("a", (2, 1), 5.0, ((1.0, 5.0), (4.0,))),
            FrameOption("b", (0, 1), 5.0, ((), (2.0,))),
            FrameOption(
                "blocked",
                (3, 3),
                3.0,
                ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
            ),
        ),
    )


def test_synthetic_case_preserves_execution_and_two_profile_mean_identity(
    monkeypatch,
):
    model = _toy_model()
    case = {
        "state_index": 0,
        "queue": [2, 5],
        "primary_population_visits": [[0.5, 41, 2, 2.0]],
        "legacy_option_indices": [1, 1],
        "variant_option_indices": [0, 1],
        "registered_support_score_delta": 0.8,
    }
    monkeypatch.setattr(transition, "BUDGET", 7)

    evaluated = transition._run_case(model, case, maximum_duration=5.0)
    profiles = {row["profile"]: row for row in evaluated["profiles"]}

    assert all(evaluated["checks"].values())
    assert evaluated["unique_state_flow_area_delta_job_s"] == pytest.approx(-15.0)
    assert evaluated["unique_state_mean_flow_delta_s"] == pytest.approx(-15.0 / 7.0)
    assert profiles["coincident"]["flow_area_delta_job_s"] == pytest.approx(-15.0)
    assert profiles["unit_fifo"]["flow_area_delta_job_s"] == pytest.approx(-15.0)
    assert profiles["coincident"]["mean_flow_delta_s"] == pytest.approx(
        profiles["unit_fifo"]["mean_flow_delta_s"]
    )
    assert profiles["coincident"]["p90_flow_delta_s"] == pytest.approx(0.0)
    assert profiles["unit_fifo"]["p90_flow_delta_s"] == pytest.approx(-1.0)
    for row in profiles.values():
        assert tuple(row["baseline_first_action"]["option_indices"]) == (1, 1)
        assert tuple(row["variant_first_action"]["option_indices"]) == (0, 1)
        for arm in ("baseline", "variant"):
            assert transition._branch_execution_ok(row[arm], (2, 5), 5.0)

    corrupted = deepcopy(profiles["coincident"]["variant"])
    corrupted["completed_jobs"] -= 1
    assert not transition._branch_execution_ok(corrupted, (2, 5), 5.0)


def _aggregate_case(area, mean, visits, coincident_p90, unit_fifo_p90):
    return {
        "unique_state_flow_area_delta_job_s": area,
        "unique_state_mean_flow_delta_s": mean,
        "primary_population_visits": visits,
        "profiles": [
            {"profile": "coincident", "p90_flow_delta_s": coincident_p90},
            {"profile": "unit_fifo", "p90_flow_delta_s": unit_fifo_p90},
        ],
    }


def test_cell_aggregation_uses_frozen_visit_counts(monkeypatch):
    monkeypatch.setattr(transition, "LOADS", (0.5, 0.8))
    monkeypatch.setattr(transition, "SEEDS", (41, 42))
    cases = [
        _aggregate_case(
            -4.0,
            -1.0,
            [[0.5, 41, 2, 2.0], [0.5, 42, 1, 1.0],
             [0.8, 41, 4, 4.0], [0.8, 42, 1, 1.0]],
            -2.0,
            -1.0,
        ),
        _aggregate_case(
            -2.0,
            -0.5,
            [[0.5, 41, 1, 1.0], [0.5, 42, 3, 3.0],
             [0.8, 41, 1, 1.0], [0.8, 42, 2, 2.0]],
            0.0,
            -0.25,
        ),
    ]

    cells = transition._cell_summaries(cases)
    by_cell = {(row["load"], row["seed"]): row for row in cells}

    assert [(row["load"], row["seed"]) for row in cells] == [
        (0.5, 41), (0.5, 42), (0.8, 41), (0.8, 42)
    ]
    assert all(row["visited_changed_state_count"] == 2 for row in cells)
    assert by_cell[(0.5, 41)]["changed_state_visit_count"] == 3
    assert by_cell[(0.5, 41)][
        "occurrence_weighted_flow_area_delta_job_s_per_changed_state_visit"
    ] == pytest.approx(-10.0 / 3.0)
    assert by_cell[(0.5, 41)]["occurrence_weighted_mean_flow_delta_s"] == pytest.approx(
        -5.0 / 6.0
    )
    assert by_cell[(0.5, 41)]["occurrence_weighted_p90_delta_s_by_profile"] == {
        "coincident": pytest.approx(-4.0 / 3.0),
        "unit_fifo": pytest.approx(-0.75),
    }
    expected = {
        (0.5, 42): (4, -2.5, -0.625),
        (0.8, 41): (5, -3.6, -0.9),
        (0.8, 42): (3, -8.0 / 3.0, -2.0 / 3.0),
    }
    for key, (count, area, mean) in expected.items():
        assert by_cell[key]["changed_state_visit_count"] == count
        assert by_cell[key][
            "occurrence_weighted_flow_area_delta_job_s_per_changed_state_visit"
        ] == pytest.approx(area)
        assert by_cell[key]["occurrence_weighted_mean_flow_delta_s"] == pytest.approx(mean)


def _gate_case(mean=-1.0, p90=(-1.0, -0.5)):
    return {
        "unique_state_mean_flow_delta_s": mean,
        "profiles": [
            {"profile": "coincident", "p90_flow_delta_s": p90[0]},
            {"profile": "unit_fifo", "p90_flow_delta_s": p90[1]},
        ],
    }


def _gate_cell(area=-2.0):
    return {
        "occurrence_weighted_flow_area_delta_job_s_per_changed_state_visit": area
    }


def test_scientific_gates_cover_every_decision_branch():
    passing_cases = [_gate_case()]
    passing_cells = [_gate_cell()]

    checks = transition._scientific_checks(passing_cases, passing_cells, True)
    assert all(checks.values())
    assert transition._decision(checks) == "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"

    invalid = transition._scientific_checks(passing_cases, passing_cells, False)
    assert transition._decision(invalid) == "INVALID_EXECUTION"

    local_tradeoff = transition._scientific_checks(
        [_gate_case(mean=0.25)], passing_cells, True
    )
    assert transition._decision(local_tradeoff) == "LOCAL_FLOW_TRADEOFF_STOP"

    tail_tradeoff = transition._scientific_checks(
        [_gate_case(p90=(-1.0, 0.25))], passing_cells, True
    )
    assert transition._decision(tail_tradeoff) == "TAIL_TRADEOFF_STOP"

    no_benefit = transition._scientific_checks(
        [_gate_case(mean=0.0, p90=(0.0, 0.0))], [_gate_cell(area=0.0)], True
    )
    assert transition._decision(no_benefit) == "NO_LOCAL_FLOW_BENEFIT_STOP"
