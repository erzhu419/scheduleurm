"""Focused tests for the registered residual-sidecar trajectories."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import algorithm.experiments.aligned_hardware_residual_sidecar_trajectories as trajectory
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("residual_sidecar_trajectories_registration")
    output = directory / "result.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("registration must not generate arrivals or execute a selector")

    patcher = pytest.MonkeyPatch()
    patcher.setattr(trajectory, "poisson_arrivals", forbidden)
    patcher.setattr(trajectory, "simulate_frames", forbidden)
    patcher.setattr(trajectory, "select_residual_sidecar_frame", forbidden)
    patcher.setattr(trajectory, "ProcessPoolExecutor", forbidden)
    try:
        protocol = trajectory.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def _copy_registration(source_output: Path, directory: Path) -> Path:
    target = directory / source_output.name
    for source, destination in zip(
        trajectory._paths(source_output)[1:],
        trajectory._paths(target)[1:],
        strict=True,
    ):
        shutil.copyfile(source, destination)
    return target


def test_registration_freezes_controls_gates_and_only_nine_new_runs(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = trajectory._paths(output)

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert len(protocol["new_run_roster"]) == 9
    assert {
        (row["load"], row["seed"], row["budget"])
        for row in protocol["new_run_roster"]
    } == {
        (load, seed, trajectory.SMALL_BUDGET)
        for load in trajectory.LOADS
        for seed in trajectory.SEEDS
    }
    assert len(protocol["frozen_v4_rows"]) == 18
    assert len(protocol["frozen_v4_flow_decompositions"]) == 18
    assert protocol["source_shadow"]["v1_recorded_certificate_scans"] == 5003
    assert protocol["source_shadow"]["v2_corrected_certificate_scans"] == 5189
    assert protocol["source_shadow"]["v2_certificate_entries_scanned"] == 569
    assert protocol["source_shadow"]["fixed_state_support_regressions"] == 0
    assert protocol["new_run_budget"] == 26
    assert protocol["frozen_baseline_budgets"] == [26, 120]
    assert "every individual" in protocol["guardrail"]
    assert "all 9" in protocol["stopping_rule"]
    sources = json.loads(sources_path.read_text())
    assert len(sources) == 16
    assert "algorithm/experiments/aligned_hardware_frame_mechanism.py" in sources


@pytest.mark.parametrize("tamper", ("protocol", "source", "tape"))
def test_tampering_fails_before_any_selector(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _ = registered_bundle
    output = _copy_registration(source_output, tmp_path)
    _, protocol_path, sources_path = trajectory._paths(output)
    original_arrivals = trajectory.poisson_arrivals

    if tamper == "protocol":
        payload = json.loads(protocol_path.read_text())
        payload["seeds"][0] += 1
        protocol_path.write_text(json.dumps(payload))
        message = "roster, controls or gates changed"
    elif tamper == "source":
        payload = json.loads(sources_path.read_text())
        payload["algorithm/residual_sidecar_frame_candidates.py"] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "sources changed"
    else:
        def changed_arrivals(rates, horizon, seed):
            return original_arrivals(rates, horizon, seed) + (
                (horizon - 0.1, 0, 10**9),
            )

        monkeypatch.setattr(trajectory, "poisson_arrivals", changed_arrivals)
        message = "arrival population changed before selection"

    selector_calls = []

    def forbidden(*args, **kwargs):
        selector_calls.append((args, kwargs))
        pytest.fail("invalid registration reached sidecar execution")

    monkeypatch.setattr(trajectory, "select_residual_sidecar_frame", forbidden)
    monkeypatch.setattr(trajectory, "simulate_frames", forbidden)
    monkeypatch.setattr(trajectory, "ProcessPoolExecutor", forbidden)
    if tamper != "tape":
        monkeypatch.setattr(trajectory, "poisson_arrivals", forbidden)
    with pytest.raises(ValueError, match=message):
        trajectory.run_registered(output)
    assert selector_calls == []
    assert not output.exists()


def test_frozen_v4_rows_and_decompositions_reconstruct_one_for_one(
    registered_bundle,
):
    _, protocol = registered_bundle
    expected = {
        (load, seed, budget)
        for load in trajectory.LOADS
        for seed in trajectory.SEEDS
        for budget in trajectory.BASELINE_BUDGETS
    }
    rows = {
        (row["load"], row["seed"], row["budget"]): row
        for row in protocol["frozen_v4_rows"]
    }
    decompositions = {
        (row["load"], row["seed"], row["budget"]): row
        for row in protocol["frozen_v4_flow_decompositions"]
    }
    assert set(rows) == set(decompositions) == expected
    for key in expected:
        assert abs(rows[key]["mean_flow_s"] - decompositions[key]["mean_flow_s"]) <= 1e-8
        assert rows[key]["job_count"] == rows[(key[0], key[1], 26)]["job_count"]


def _authored_tape():
    return tuple(
        [(0.0, coordinate, index) for coordinate, count in enumerate((1, 4, 8)) for index in range(count)]
        + [(0.5, 2, 8), (1.0, 0, 1)]
    )


def test_authored_tape_closes_flow_and_sidecar_work_identities():
    model = load_model()
    certificate = capacity_certificate(model, 0.5)
    recorder = trajectory.SidecarRecorder(26)
    tape = _authored_tape()
    metrics = simulate_frames(
        model, tape, 2.0, certificate, 26, selector=recorder,
    )
    decomposition = trajectory.decompose_flow(model, tape, recorder.frames, 2.0)
    accounting = recorder.accounting()
    work = accounting["sidecar"]
    frames = metrics["frame_count"]
    positions = len(model.options) + 1
    thresholds = len({1.0, *(max(1.0, option.duration) for option in model.options)})
    masks = 1 << len(model.workloads)

    assert metrics["completed_jobs"] == metrics["job_count"] == len(tape)
    assert abs(metrics["mean_flow_s"] - decomposition["mean_flow_s"]) <= 1e-8
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert metrics["maximum_expected_drift_bound_violation"] <= 1e-8
    assert accounting["work_identity_failure_count"] == 0
    assert work["certificate_frontier_scans"] == frames
    assert work["bound_price_vector_count"] == frames * masks
    assert work["bound_option_probes"] == frames * thresholds * positions
    assert work["bound_dual_table_option_probes"] == frames * thresholds * masks * positions
    assert work["bound_dual_mask_evaluations"] == work["bound_threshold_evaluations"] * masks
    assert work["bound_dual_mask_sums"] == work["upper_bound_calls"] * masks
    assert work["bound_dual_coordinate_products"] == len(model.workloads) * (
        frames * masks * positions + work["upper_bound_calls"]
    )
    assert work["attempts"] == frames * positions + work["paid_search_pops"]
    assert metrics["candidate_attempts"] == work["attempts"]
    assert all(
        all(departure <= queued for departure, queued in zip(frame["departures"], frame["queue_before"], strict=True))
        for frame in recorder.frames
    )


def test_same_released_prefix_ignores_different_future_arrivals():
    model = load_model()
    certificate = capacity_certificate(model, 0.5)
    prefix = tuple(
        (0.0, coordinate, index)
        for coordinate, count in enumerate((3, 2, 8))
        for index in range(count)
    )
    suffix_time = 30.25
    changed = prefix + tuple(
        (suffix_time, coordinate, 100 + index)
        for coordinate, count in enumerate((4, 1, 6))
        for index in range(count)
    )
    recorders = []
    for tape in (prefix, changed):
        recorder = trajectory.SidecarRecorder(26)
        simulate_frames(model, tape, 31.0, certificate, 26, selector=recorder)
        recorders.append(recorder)

    histories = [
        [frame for frame in recorder.frames if frame["start_s"] < suffix_time]
        for recorder in recorders
    ]
    assert len(histories[0]) >= 4
    assert histories[0] == histories[1]


def _synthetic_protocol_and_runs(flow_ratios, *, p90_ratios=None, safe_prunes=1):
    p90_ratios = p90_ratios or flow_ratios
    rows = []
    decompositions = []
    runs = []
    index = 0
    for load in trajectory.LOADS:
        for seed in trajectory.SEEDS:
            baseline_flow = 100.0 * (1 + index)
            baseline_p90 = 150.0 * (1 + index)
            for budget in trajectory.BASELINE_BUDGETS:
                rows.append({
                    "load": load,
                    "seed": seed,
                    "budget": budget,
                    "mean_flow_s": baseline_flow,
                    "p90_flow_s": baseline_p90,
                    "makespan_s": 1000.0,
                    "time_average_unfinished_jobs": 10.0,
                    "frame_boundary_mean_backlog": 5.0,
                    "candidate_attempts": 100,
                })
                decompositions.append({
                    "load": load,
                    "seed": seed,
                    "budget": budget,
                    "mean_deferred_visible_wait_s": 40.0,
                    "mean_arrival_barrier_wait_s": 30.0,
                    "mean_dispatch_to_completion_s": baseline_flow - 70.0,
                })
            ratio = flow_ratios[index]
            p90_ratio = p90_ratios[index]
            runs.append({
                "load": load,
                "seed": seed,
                "budget": 26,
                "metrics": {
                    "job_count": 1,
                    "mean_flow_s": baseline_flow * ratio,
                    "p90_flow_s": baseline_p90 * p90_ratio,
                    "makespan_s": 1000.0,
                    "time_average_unfinished_jobs": 10.0,
                    "frame_boundary_mean_backlog": 5.0,
                    "candidate_attempts": 90,
                },
                "decomposition": {
                    "mean_deferred_visible_wait_s": 40.0 + baseline_flow * (ratio - 1.0),
                    "mean_arrival_barrier_wait_s": 30.0,
                    "mean_dispatch_to_completion_s": baseline_flow - 70.0,
                },
                "selector_accounting": {"safe_pruning_frame_count": safe_prunes},
            })
            index += 1
    return {
        "frozen_v4_rows": rows,
        "frozen_v4_flow_decompositions": decompositions,
    }, runs


def test_pairing_uses_mean_seed_ratios_and_decision_branches():
    ratios = [0.9, 1.1, 0.9, 0.99, 0.99, 0.99, 1.0, 1.0, 1.0]
    protocol, runs = _synthetic_protocol_and_runs(ratios)
    paired, summaries, _ = trajectory.summarize_runs(runs, protocol)
    assert summaries[0]["mean_paired_flow_ratio_over_v4"] == pytest.approx(
        (0.9 + 1.1 + 0.9) / 3
    )
    assert all("sidecar_over_v4_exact" in row for row in paired)
    assert all(
        abs(row["exact_component_delta_identity_error_s"]) <= 1e-8
        for row in paired
    )
    checks = trajectory._scientific_checks(paired, summaries, True)
    assert not checks["all_nine_k26_seed_pairs_mean_flow_nonworse"]
    assert trajectory._decision({"ok": True}, checks)["scientific_decision"] == (
        "TRAJECTORY_TRADEOFF_STOP"
    )

    protocol, runs = _synthetic_protocol_and_runs([0.99] * 9)
    paired, summaries, _ = trajectory.summarize_runs(runs, protocol)
    checks = trajectory._scientific_checks(paired, summaries, True)
    assert all(checks.values())
    assert trajectory._decision({"ok": True}, checks)["scientific_decision"] == (
        "ADVANCE_TO_FRESH_SEED_CONFIRMATION"
    )
    assert trajectory._decision({"ok": False}, checks)["scientific_decision"] == (
        "INVALID_EXECUTION"
    )

    protocol, runs = _synthetic_protocol_and_runs([1.0] * 9)
    paired, summaries, _ = trajectory.summarize_runs(runs, protocol)
    checks = trajectory._scientific_checks(paired, summaries, True)
    assert trajectory._decision({"ok": True}, checks)["scientific_decision"] == (
        "NO_FLOW_BENEFIT_STOP"
    )

    protocol, runs = _synthetic_protocol_and_runs(
        [0.99] * 9, p90_ratios=[1.001] * 3 + [0.99] * 6,
    )
    paired, summaries, _ = trajectory.summarize_runs(runs, protocol)
    checks = trajectory._scientific_checks(paired, summaries, True)
    assert not checks["k26_p90_flow_nonworse_at_every_load"]
    assert trajectory._decision({"ok": True}, checks)["scientific_decision"] == (
        "TRAJECTORY_TRADEOFF_STOP"
    )
