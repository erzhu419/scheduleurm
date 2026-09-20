"""Contracts for the FIFO-execution quotient evidence-reuse closure."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import algorithm.authored_transition_drain as authored_drain
import algorithm.budgeted_frame_candidates as budgeted_candidates
import algorithm.experiments.aligned_hardware_fifo_execution_quotient_evidence_reuse as reuse
import algorithm.experiments.aligned_hardware_fifo_execution_quotient_shadow as fifo_shadow
import algorithm.experiments.aligned_hardware_permutation_quotient_transition as transition
import algorithm.fifo_execution_quotient_frame_candidates as fifo_candidates
import algorithm.permutation_quotient_frame_candidates as quotient_candidates


def _forbid_new_execution(patcher: pytest.MonkeyPatch):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"evidence reuse unexpectedly called {name}")

        return fail

    hooks = {
        "FIFO selector": forbidden("FIFO selector"),
        "legacy selector": forbidden("legacy selector"),
        "quotient selector": forbidden("quotient selector"),
        "authored arrivals": forbidden("authored arrivals"),
        "authored drain": forbidden("authored drain"),
    }
    patcher.setattr(
        fifo_candidates,
        "select_fifo_execution_quotient_frame",
        hooks["FIFO selector"],
    )
    patcher.setattr(
        fifo_shadow,
        "select_fifo_execution_quotient_frame",
        hooks["FIFO selector"],
    )
    patcher.setattr(
        budgeted_candidates, "select_budgeted_frame", hooks["legacy selector"]
    )
    patcher.setattr(
        quotient_candidates,
        "select_permutation_quotient_frame",
        hooks["quotient selector"],
    )
    patcher.setattr(
        transition, "select_budgeted_frame", hooks["legacy selector"]
    )
    patcher.setattr(
        transition,
        "select_permutation_quotient_frame",
        hooks["quotient selector"],
    )
    patcher.setattr(
        authored_drain, "authored_arrival_times", hooks["authored arrivals"]
    )
    patcher.setattr(
        authored_drain, "drain_initial_population", hooks["authored drain"]
    )
    patcher.setattr(
        transition, "authored_arrival_times", hooks["authored arrivals"]
    )
    patcher.setattr(
        transition, "drain_initial_population", hooks["authored drain"]
    )
    for name, function in (
        ("select_fifo_execution_quotient_frame", hooks["FIFO selector"]),
        ("select_budgeted_frame", hooks["legacy selector"]),
        ("select_permutation_quotient_frame", hooks["quotient selector"]),
        ("authored_arrival_times", hooks["authored arrivals"]),
        ("drain_initial_population", hooks["authored drain"]),
    ):
        patcher.setattr(reuse, name, function, raising=False)
    return calls


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fifo_evidence_reuse_registration")
    output = directory / "reuse.json"
    reads = []
    patcher = pytest.MonkeyPatch()
    calls = _forbid_new_execution(patcher)
    original_read_text = Path.read_text
    closed_outcome = reuse.CLOSED_TRANSITION.resolve()

    def guarded_read_text(path, *args, **kwargs):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved == closed_outcome:
            pytest.fail("registration opened the completed transition outcome")
        return original_read_text(path, *args, **kwargs)

    patcher.setattr(Path, "read_text", guarded_read_text)
    try:
        protocol = reuse.register(output)
    finally:
        patcher.undo()
    assert calls == []
    assert closed_outcome not in reads
    return output, protocol, reads


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    for source, target in zip(
        reuse._paths(source_output)[1:],
        reuse._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


@pytest.fixture(scope="module")
def completed_bundle(registered_bundle, tmp_path_factory):
    source_output, _, _ = registered_bundle
    output = _copy_registered_bundle(
        source_output, tmp_path_factory.mktemp("fifo_evidence_reuse_execution")
    )
    patcher = pytest.MonkeyPatch()
    calls = _forbid_new_execution(patcher)
    try:
        report = reuse.run_registered(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, report


def test_registration_reads_protocol_not_outcome_and_freezes_four_matches(
    registered_bundle,
):
    output, protocol, reads = registered_bundle
    result_path, protocol_path, sources_path = reuse._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")

    assert result_path == output and not output.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert reuse.CLOSED_TRANSITION_PROTOCOL.resolve() in reads
    assert reuse.CLOSED_TRANSITION_SOURCES.resolve() in reads
    assert protocol["registered_selector_call_count"] == 0
    assert protocol["registered_drain_count"] == 0
    assert protocol["state_count"] == 5003
    assert protocol["raw_unchanged_count"] == 4999
    assert protocol["known_reuse_count"] == 4
    assert protocol["novel_count"] == 0
    assert [row["state_index"] for row in protocol["known_reuse_roster"]] == [
        582, 621, 752, 753,
    ]
    assert len(protocol["primary_load_seed_weight_roster"]) == 9
    assert "states" not in protocol
    assert all(
        row["pair_id"] == f"quotient_case_{row['state_index']}"
        and row["baseline"]["raw"] != row["variant"]["raw"]
        and row["baseline"]["physical"] != row["variant"]["physical"]
        and row["baseline"]["fifo_execution"]
        != row["variant"]["fifo_execution"]
        for row in protocol["known_reuse_roster"]
    )


def test_four_completed_outcomes_are_reused_exactly(completed_bundle):
    _, report = completed_bundle
    source = json.loads(reuse.CLOSED_TRANSITION.read_text())
    source_by_state = {row["state_index"]: row for row in source["cases"]}
    reused_by_state = {
        row["state_index"]: row for row in report["known_reused_cases"]
    }

    assert set(reused_by_state) == {582, 621, 752, 753}
    for state_index, row in reused_by_state.items():
        original = source_by_state[state_index]
        assert row["queue"] == original["queue"]
        assert row["baseline_option_indices"] == original[
            "legacy_option_indices"
        ]
        assert row["variant_option_indices"] == original[
            "variant_option_indices"
        ]
        assert row["unique_state_flow_area_delta_job_s"] == original[
            "unique_state_flow_area_delta_job_s"
        ]
        assert row["unique_state_mean_flow_delta_s"] == original[
            "unique_state_mean_flow_delta_s"
        ]
        assert row["evidence_source"] == reuse.CLOSED_TRANSITION.name
        assert row["profiles"] == [
            {
                "profile": profile["profile"],
                "flow_area_delta_job_s": profile["flow_area_delta_job_s"],
                "mean_flow_delta_s": profile["mean_flow_delta_s"],
                "p90_flow_delta_s": profile["p90_flow_delta_s"],
                "makespan_delta_s": profile["makespan_delta_s"],
                "baseline_frame_count": profile["baseline"]["frame_count"],
                "variant_frame_count": profile["variant"]["frame_count"],
                "baseline_continuation_selector_calls": profile["baseline"][
                    "continuation_selector_calls"
                ],
                "variant_continuation_selector_calls": profile["variant"][
                    "continuation_selector_calls"
                ],
            }
            for profile in original["profiles"]
        ]


def test_three_positive_p90_witnesses_close_the_frozen_tail_gate(completed_bundle):
    _, report = completed_bundle
    witnesses = report["positive_p90_witnesses"]

    assert [
        (row["state_index"], row["profile"])
        for row in witnesses
    ] == [
        (582, "coincident"),
        (582, "unit_fifo"),
        (753, "unit_fifo"),
    ]
    assert witnesses[0]["p90_flow_delta_s"] == pytest.approx(
        4.636049095222894
    )
    assert witnesses[1]["p90_flow_delta_s"] == pytest.approx(
        4.636049095222894
    )
    assert witnesses[2]["p90_flow_delta_s"] == pytest.approx(2.0)
    assert report["positive_mean_flow_witnesses"] == []

    gates = report["transition_advance_gate_status"]
    assert gates["execution_passed"]
    assert gates["all_5003_unique_state_mean_flow_deltas_nonpositive"]
    assert not gates[
        "all_10006_state_profile_nearest_rank_p90_deltas_nonpositive"
    ]
    assert gates[
        "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive"
    ]
    assert gates["at_least_one_unique_state_has_strict_mean_flow_improvement"]
    assert gates[
        "at_least_one_primary_load_seed_cell_has_strict_weighted_flow_area_improvement"
    ]
    assert report["scientific_decision"] == "KNOWN_TAIL_TRADEOFF_STOP"
    assert report["development_advance"] is False


def test_nine_cells_include_two_zero_occurrence_null_summaries(completed_bundle):
    _, report = completed_bundle
    cells = report["summary_by_primary_load_seed"]
    zero = [row for row in cells if row["matched_change_frame_count"] == 0]
    nonzero = [row for row in cells if row["matched_change_frame_count"] > 0]

    assert [(row["load"], row["seed"]) for row in cells] == [
        (load, seed) for load in reuse.LOADS for seed in reuse.SEEDS
    ]
    assert [(row["load"], row["seed"]) for row in zero] == [
        (0.95, 41), (0.95, 42),
    ]
    assert len(nonzero) == 7
    assert all(
        row["occurrence_weighted_flow_area_delta_job_s"] == 0.0
        and row["occurrence_weighted_flow_area_delta_job_s_per_source_frame"]
        == 0.0
        and row["mean_flow_delta_s_per_matched_occurrence"] is None
        and row["makespan_delta_s_per_matched_occurrence"] is None
        and set(row["p90_delta_s_per_matched_occurrence_by_profile"].values())
        == {None}
        for row in zero
    )
    assert all(
        row["occurrence_weighted_flow_area_delta_job_s"] < 0.0
        for row in nonzero
    )
    assert sum(row["matched_change_frame_count"] for row in cells) == 61


def _passing_gate():
    return {
        "execution_passed": True,
        "all_5003_unique_state_mean_flow_deltas_nonpositive": True,
        "all_10006_state_profile_nearest_rank_p90_deltas_nonpositive": True,
        "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive": True,
        "at_least_one_unique_state_has_strict_mean_flow_improvement": True,
        "at_least_one_primary_load_seed_cell_has_strict_weighted_flow_area_improvement": True,
    }


def test_decision_helper_preserves_every_frozen_branch():
    passing = _passing_gate()
    assert reuse._decision(passing) == "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"

    invalid = {**passing, "execution_passed": False}
    assert reuse._decision(invalid) == "INVALID_EXECUTION"

    local = {
        **passing,
        "all_5003_unique_state_mean_flow_deltas_nonpositive": False,
    }
    assert reuse._decision(local) == "KNOWN_LOCAL_FLOW_TRADEOFF_STOP"

    tail = {
        **passing,
        "all_10006_state_profile_nearest_rank_p90_deltas_nonpositive": False,
    }
    assert reuse._decision(tail) == "KNOWN_TAIL_TRADEOFF_STOP"

    no_benefit = {
        **passing,
        "at_least_one_unique_state_has_strict_mean_flow_improvement": False,
    }
    assert reuse._decision(no_benefit) == "NO_LOCAL_FLOW_BENEFIT_STOP"


@pytest.mark.parametrize(
    "tamper", ("protocol", "source", "shadow", "transition_protocol", "outcome")
)
def test_tampering_fails_before_output_or_new_execution(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, sources_path = reuse._paths(output)

    if tamper == "protocol":
        payload = json.loads(protocol_path.read_text())
        payload["known_reuse_count"] += 1
        protocol_path.write_text(json.dumps(payload))
    elif tamper == "source":
        payload = json.loads(sources_path.read_text())
        payload[reuse.NEW_SOURCES[0]] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
    elif tamper == "shadow":
        path = tmp_path / reuse.FIFO_SHADOW.name
        payload = json.loads(reuse.FIFO_SHADOW.read_text())
        payload["scientific_decision"] = "TAMPERED"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(reuse, "FIFO_SHADOW", path)
    elif tamper == "transition_protocol":
        path = tmp_path / reuse.CLOSED_TRANSITION_PROTOCOL.name
        payload = json.loads(reuse.CLOSED_TRANSITION_PROTOCOL.read_text())
        payload["budget"] += 1
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(reuse, "CLOSED_TRANSITION_PROTOCOL", path)
    else:
        path = tmp_path / reuse.CLOSED_TRANSITION.name
        payload = json.loads(reuse.CLOSED_TRANSITION.read_text())
        case = next(row for row in payload["cases"] if row["state_index"] == 582)
        case["profiles"][0]["p90_flow_delta_s"] += 1.0
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(reuse, "CLOSED_TRANSITION", path)

    calls = _forbid_new_execution(monkeypatch)
    with pytest.raises(ValueError):
        reuse.run_registered(output)
    assert calls == []
    assert not output.exists()


def test_result_partition_and_work_accounting_are_zero_execution(completed_bundle):
    _, report = completed_bundle
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert report["evidence_partition"] == {
        "raw_unchanged_zero_effect_count": 4999,
        "known_reused_outcome_count": 4,
        "novel_unexecuted_count": 0,
    }
    assert report["work_accounting"] == {
        "selector_calls": 0,
        "authored_drain_calls": 0,
        "arrival_tapes": 0,
        "trajectory_frames": 0,
        "completed_case_outcomes_reused": 4,
        "completed_drain_calls_reused": 16,
        "novel_fifo_pairs_executed": 0,
        "rng_initializations": 0,
        "random_draws": 0,
    }


def test_write_new_and_registration_refuse_overwrite(registered_bundle, tmp_path):
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text('{"preserve": true}\n')
    with pytest.raises(FileExistsError):
        reuse._write_new(sentinel, {"preserve": False})
    assert json.loads(sentinel.read_text()) == {"preserve": True}

    output, _, _ = registered_bundle
    with pytest.raises(ValueError, match="FIFO evidence-reuse artifacts already exist"):
        reuse.register(output)
