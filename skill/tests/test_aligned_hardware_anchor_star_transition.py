"""Contracts for the evidence-only anchor-star transition closure."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import algorithm.anchor_star_frame_candidates as anchor_candidates
import algorithm.authored_transition_drain as authored_drain
import algorithm.budgeted_frame_candidates as budgeted_candidates
import algorithm.experiments.aligned_hardware_anchor_star_shadow as anchor_shadow
import algorithm.experiments.aligned_hardware_anchor_star_transition as transition
from algorithm.experiments.aligned_hardware_frame_model import load_model


def _forbid_execution(patcher: pytest.MonkeyPatch):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"evidence-only closure unexpectedly called {name}")

        return fail

    hooks = {
        "select_budgeted_frame": forbidden("legacy selector"),
        "select_anchor_star_frame": forbidden("anchor-star selector"),
        "authored_arrival_times": forbidden("authored arrival profile"),
        "drain_initial_population": forbidden("authored drain"),
    }
    patcher.setattr(
        budgeted_candidates, "select_budgeted_frame", hooks["select_budgeted_frame"]
    )
    patcher.setattr(
        anchor_candidates,
        "select_anchor_star_frame",
        hooks["select_anchor_star_frame"],
    )
    patcher.setattr(
        anchor_shadow,
        "select_anchor_star_frame",
        hooks["select_anchor_star_frame"],
    )
    patcher.setattr(
        authored_drain,
        "authored_arrival_times",
        hooks["authored_arrival_times"],
    )
    patcher.setattr(
        authored_drain,
        "drain_initial_population",
        hooks["drain_initial_population"],
    )
    # Also intercept future local imports without requiring the closure module
    # to expose execution symbols that it deliberately does not use.
    for name, replacement in hooks.items():
        patcher.setattr(transition, name, replacement, raising=False)
    return calls


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("anchor_star_transition_registration")
    output = directory / "closure.json"
    patcher = pytest.MonkeyPatch()
    calls = _forbid_execution(patcher)
    try:
        protocol = transition.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    for source, target in zip(
        transition._paths(source_output)[1:],
        transition._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


@pytest.fixture(scope="module")
def completed_bundle(registered_bundle, tmp_path_factory):
    source_output, _ = registered_bundle
    directory = tmp_path_factory.mktemp("anchor_star_transition_execution")
    output = _copy_registered_bundle(source_output, directory)
    patcher = pytest.MonkeyPatch()
    calls = _forbid_execution(patcher)
    try:
        report = transition.run_registered(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, report


def _partition_indices_from_shadow():
    shadow = json.loads(transition.ANCHOR_SHADOW.read_text())
    aliases, known, novel = [], [], []
    for change in shadow["state_changes"]:
        if not change["fifo_execution_signature_changed_from_legacy_k26"]:
            aliases.append(change["state_index"])
        elif change["known_quotient_fifo_pair"]:
            known.append(change["state_index"])
        else:
            novel.append(change["state_index"])
    return aliases, known, novel


def test_registration_is_zero_execution_and_freezes_the_exact_partition(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = transition._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")
    expected_aliases, expected_known, expected_novel = _partition_indices_from_shadow()

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert "states" not in protocol
    assert protocol["registered_state_roster"] == transition.REGISTERED_STATES.name
    assert protocol["closed_transition_fixed_sources"] == (
        transition.CLOSED_TRANSITION_SOURCES.name
    )
    assert protocol["registered_selector_call_count"] == 0
    assert protocol["registered_drain_count"] == 0
    assert protocol["changed_state_count"] == 1413
    assert protocol["fifo_alias_count"] == len(expected_aliases) == 912
    assert protocol["known_reuse_count"] == len(expected_known) == 14
    assert protocol["novel_unexecuted_count"] == len(expected_novel) == 487
    assert [row["state_index"] for row in protocol["fifo_alias_roster"]] == (
        expected_aliases
    )
    assert [row["state_index"] for row in protocol["known_reuse_roster"]] == (
        expected_known
    )
    assert [row["state_index"] for row in protocol["novel_unexecuted_roster"]] == (
        expected_novel
    )
    assert all(
        row["baseline_fifo_execution_signature"]
        == row["variant_fifo_execution_signature"]
        for row in protocol["fifo_alias_roster"]
    )
    assert all(
        row["baseline_fifo_execution_signature"]
        != row["variant_fifo_execution_signature"]
        for row in (
            *protocol["known_reuse_roster"],
            *protocol["novel_unexecuted_roster"],
        )
    )


@pytest.mark.parametrize(
    "tamper", ("protocol", "source", "anchor_shadow", "known_transition")
)
def test_every_tamper_fails_before_output_or_execution(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, sources_path = transition._paths(output)

    if tamper == "protocol":
        payload = json.loads(protocol_path.read_text())
        payload["budget"] += 1
        protocol_path.write_text(json.dumps(payload))
    elif tamper == "source":
        payload = json.loads(sources_path.read_text())
        payload[transition.NEW_SOURCES[0]] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
    elif tamper == "anchor_shadow":
        path = tmp_path / transition.ANCHOR_SHADOW.name
        payload = json.loads(transition.ANCHOR_SHADOW.read_text())
        payload["scientific_decision"] = "TAMPERED"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(transition, "ANCHOR_SHADOW", path)
    else:
        path = tmp_path / transition.CLOSED_TRANSITION.name
        payload = json.loads(transition.CLOSED_TRANSITION.read_text())
        known = next(
            row for row in payload["cases"] if row["state_index"] == 423
        )
        known["unique_state_mean_flow_delta_s"] += 1.0
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(transition, "CLOSED_TRANSITION", path)

    calls = _forbid_execution(monkeypatch)
    with pytest.raises(ValueError):
        transition.run_registered(output)
    assert calls == []
    assert not output.exists()


def test_known_fifo_cases_reuse_completed_outcomes_exactly(completed_bundle):
    _, report = completed_bundle
    model = load_model()
    source = json.loads(transition.CLOSED_TRANSITION.read_text())
    source_by_state = {row["state_index"]: row for row in source["cases"]}
    roster_by_state = {
        row["state_index"]: row for row in report["protocol"]["known_reuse_roster"]
    }
    reused_by_state = {
        row["state_index"]: row for row in report["known_reused_cases"]
    }

    assert set(roster_by_state) == set(reused_by_state) == {
        423, 582, 583, 595, 607, 608, 621,
        752, 753, 763, 764, 777, 778, 792,
    }
    for state_index, reused in reused_by_state.items():
        original = source_by_state[state_index]
        roster = roster_by_state[state_index]
        assert roster["queue"] == original["queue"] == reused["queue"]
        assert roster["baseline_option_indices"] == original[
            "legacy_option_indices"
        ]
        assert roster["variant_option_indices"] == original[
            "variant_option_indices"
        ]
        baseline_physical, baseline_fifo = transition._action_signatures(
            model, original["legacy_option_indices"]
        )
        variant_physical, variant_fifo = transition._action_signatures(
            model, original["variant_option_indices"]
        )
        assert roster["baseline_physical_signature"] == baseline_physical
        assert roster["variant_physical_signature"] == variant_physical
        assert roster["baseline_fifo_execution_signature"] == baseline_fifo
        assert roster["variant_fifo_execution_signature"] == variant_fifo
        assert reused["unique_state_flow_area_delta_job_s"] == original[
            "unique_state_flow_area_delta_job_s"
        ]
        assert reused["unique_state_mean_flow_delta_s"] == original[
            "unique_state_mean_flow_delta_s"
        ]
        expected_profiles = [
            {
                "profile": row["profile"],
                "flow_area_delta_job_s": row["flow_area_delta_job_s"],
                "mean_flow_delta_s": row["mean_flow_delta_s"],
                "p90_flow_delta_s": row["p90_flow_delta_s"],
                "makespan_delta_s": row["makespan_delta_s"],
            }
            for row in original["profiles"]
        ]
        assert reused["profiles"] == expected_profiles


def test_fifo_identity_marks_raw_swaps_as_dangerous_or_safe_without_a_drain():
    model = load_model()

    dangerous_forward = transition._action_signatures(model, (0, 7))
    dangerous_reverse = transition._action_signatures(model, (7, 0))
    assert dangerous_forward[0] == dangerous_reverse[0]
    assert dangerous_forward[1] != dangerous_reverse[1]

    safe_forward = transition._action_signatures(model, (3, 8))
    safe_reverse = transition._action_signatures(model, (8, 3))
    assert safe_forward[0] == safe_reverse[0]
    assert safe_forward[1] == safe_reverse[1]


def test_reused_mean_and_p90_witnesses_irreversibly_fail_the_advance_gate(
    completed_bundle,
):
    _, report = completed_bundle
    witnesses = report["hard_veto_witnesses"]
    mean = witnesses["positive_mean_flow"]
    p90 = witnesses["positive_p90"]
    gate = report["transition_advance_gate_status"]

    assert {row["state_index"] for row in mean} == {
        423, 583, 595, 608, 763, 792,
    }
    assert len(mean) == 6
    assert len(p90) == 12
    assert max(row["p90_flow_delta_s"] for row in p90) == pytest.approx(
        4.636049095222894
    )
    assert {
        row["state_index"]
        for row in p90
        if row["p90_flow_delta_s"] == max(
            witness["p90_flow_delta_s"] for witness in p90
        )
    } == {582}
    assert not gate["all_1413_changed_state_mean_flow_deltas_nonpositive"]
    assert not gate[
        "all_2826_changed_state_profile_nearest_rank_p90_deltas_nonpositive"
    ]
    assert gate[
        "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive"
    ] == "NOT_EVALUATED_AFTER_HARD_VETO"
    assert gate["strict_local_and_weighted_gain_conditions"] == (
        "NOT_EVALUATED_AFTER_HARD_VETO"
    )
    assert not gate["advance_conjunction_satisfied"]

    cells = report["summary_by_primary_load_seed"]
    assert [(row["load"], row["seed"]) for row in cells] == [
        (load, seed) for load in transition.LOADS for seed in transition.SEEDS
    ]
    assert all(row["known_reuse_frame_count"] > 0 for row in cells)
    assert all(
        row["complete_anchor_weighted_effect"]
        == "NOT_EVALUATED_AFTER_HARD_VETO"
        for row in cells
    )


def test_real_decision_is_fixed_and_all_new_execution_accounting_is_zero(
    completed_bundle,
):
    _, report = completed_bundle
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert report["scientific_decision"] == "KNOWN_LOCAL_FLOW_TRADEOFF_STOP"
    assert report["development_advance"] is False
    assert report["evidence_partition"] == {
        "changed_state_count": 1413,
        "fifo_alias_zero_effect_count": 912,
        "known_reused_outcome_count": 14,
        "novel_unexecuted_count": 487,
    }
    assert report["work_accounting"] == {
        "selector_calls": 0,
        "authored_drain_calls": 0,
        "arrival_tapes": 0,
        "trajectory_frames": 0,
        "novel_fifo_pairs_executed": 0,
        "completed_case_outcomes_reused": 14,
        "completed_drain_calls_reused": 56,
        "skipped_novel_drain_calls": 1948,
        "fifo_alias_zero_effect_proofs": 912,
        "rng_initializations": 0,
        "random_draws": 0,
    }


def test_decision_helper_preserves_all_logical_branches():
    mean = {"positive_mean_flow": [{"state_index": 1}], "positive_p90": []}
    tail = {"positive_mean_flow": [], "positive_p90": [{"state_index": 2}]}
    none = {"positive_mean_flow": [], "positive_p90": []}

    assert transition._decision(False, mean) == "INVALID_EXECUTION"
    assert transition._decision(True, mean) == "KNOWN_LOCAL_FLOW_TRADEOFF_STOP"
    assert transition._decision(True, tail) == "KNOWN_TAIL_TRADEOFF_STOP"
    assert transition._decision(True, none) == "NOVEL_EVIDENCE_REQUIRED"


def test_write_new_and_public_entry_points_never_overwrite(
    registered_bundle, completed_bundle, tmp_path,
):
    registered_output, _ = registered_bundle
    completed_output, completed_report = completed_bundle

    with pytest.raises(ValueError, match="artifacts already exist"):
        transition.register(registered_output)
    with pytest.raises(ValueError, match="result already exists"):
        transition.run_registered(completed_output)
    assert json.loads(completed_output.read_text()) == completed_report

    target = tmp_path / "existing.json"
    target.write_text('{"sentinel": true}\n')
    with pytest.raises(FileExistsError):
        transition._write_new(target, {"sentinel": False})
    assert json.loads(target.read_text()) == {"sentinel": True}
