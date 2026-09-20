from itertools import product
from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_residual_fill_scaling_screen as screen
import algorithm.residual_fill_frame_candidates as residual
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.residual_fill_frame_candidates import (
    certified_fractional_frame_upper,
    select_residual_fill_frame,
)


def test_registration_does_not_construct_grid_solve_lp_or_call_selector(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("registration ran an execution-only operation")

    for name in (
        "_base_queues",
        "certified_fractional_frame_upper",
        "select_residual_fill_frame",
        "select_budgeted_frame",
        "capacity_certificate",
    ):
        monkeypatch.setattr(screen, name, forbidden)
    monkeypatch.setattr(screen, "product", forbidden)
    monkeypatch.setattr(residual, "linprog", forbidden)

    output = tmp_path / "screen.json"
    protocol = screen.register(output)

    assert not output.exists()
    assert output.with_name("screen_protocol.json").exists()
    assert output.with_name("screen_fixed_sources.json").exists()
    assert protocol["post_result_development_screen"] is True
    assert protocol["independent_preregistration"] is False
    assert protocol["total_candidate_budgets"] == [4096, 8192]
    scaling = json.loads(screen.SCALING_RESULT.read_text())
    for lanes in screen.LANE_COUNTS:
        source_row = screen._source_budget_row(scaling, lanes, 4096)
        contract = protocol["source_contract"]
        assert contract["scaling_k4096_source_rows_by_lane"][str(lanes)] == source_row
        assert contract["scaling_k4096_alpha_by_lane"][str(lanes)] == (
            source_row["max_certified_normalized_gap"]
        )


def test_cli_registration_prints_one_compact_summary_line(tmp_path, capsys):
    output = tmp_path / "screen.json"

    screen.main(["--register", "--output", str(output)])

    stdout = capsys.readouterr().out
    assert stdout.count("\n") == 1
    assert len(stdout) < 300
    assert json.loads(stdout) == {
        "experiment": screen.EXPERIMENT,
        "output": str(output),
        "scientific_decision": None,
        "status": "REGISTERED",
    }


@pytest.mark.parametrize(
    "queue",
    ((0, 0, 0), (4, 2, 5), (5, 3, 5), (40, 25, 24)),
)
def test_repaired_fractional_dual_bounds_all_100_two_lane_actions(queue):
    model = load_model()
    choices = tuple(range(len(model.options))) + (None,)
    oracle = 0.0
    for indices in product(choices, repeat=2):
        score = screen._score_action(model, queue, 2, indices)
        if score is not None:
            oracle = max(oracle, score)

    certificate = certified_fractional_frame_upper(model.options, queue, 2)

    assert certificate.exact_dual_feasible
    assert certificate.upper_bound + screen.SCORE_TOLERANCE >= oracle
    assert certificate.threshold_lp_count == 9
    assert certificate.variable_constraint_cells == (
        certificate.total_variables * 4
    )


@pytest.mark.parametrize(
    ("lanes", "queue", "expected_indices", "expected_pair_attempts"),
    (
        (8, (20, 12, 20), (5, 5, 7, 7, 7, 7, 7, 7), 247),
        (16, (40, 16, 32), (3,) * 12 + (8,) * 4, 523),
    ),
)
def test_residual_fill_recovers_fixed_veto_challenger_with_total_budget(
    lanes, queue, expected_indices, expected_pair_attempts
):
    model = load_model()
    selected = select_residual_fill_frame(model.options, queue, lanes, 4096)

    assert selected.option_indices == expected_indices
    assert selected.pair_candidate_attempts == expected_pair_attempts
    assert selected.attempts == (
        selected.legacy_attempts + selected.pair_candidate_attempts
    )
    assert selected.legacy_budget == 4096 - selected.pair_candidate_attempts
    assert selected.attempts <= 4096
    assert selected.pair_family_exhausted
    assert not selected.pair_budget_blocked_unique_candidate
    assert selected.lp_exact_dual_feasible
    assert selected.upper_bound <= selected.legacy_upper_bound
    assert math.isclose(
        selected.gap,
        selected.upper_bound - selected.score,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert selected.charged_logical_work > selected.attempts
    assert selected.pair_generation_probes >= selected.pair_candidate_attempts


def test_fixed_budget_curve_can_plateau_without_passing_high_load_gate():
    model = load_model()
    queue = (20, 12, 20)
    small = select_residual_fill_frame(model.options, queue, 8, 4096)
    large = select_residual_fill_frame(model.options, queue, 8, 8192)

    assert large.score + 1e-12 >= small.score
    assert large.upper_bound <= small.upper_bound + 1e-12
    assert large.gap <= small.gap + 1e-12
    assert math.isclose(large.epsilon, small.epsilon, rel_tol=0.0, abs_tol=1e-12)
    assert small.epsilon > 0.043857439770326634


@pytest.mark.parametrize(
    ("execution", "tradeoff", "improved", "capacity", "expected"),
    (
        (False, False, False, False, "INVALID_EXECUTION"),
        (True, True, True, True, "RESIDUAL_FILL_SCALING_ADVANCE"),
        (
            True,
            False,
            True,
            False,
            "CERTIFIED_IMPROVEMENT_WITH_HIGH_LOAD_FAIL_STOP",
        ),
        (True, False, False, False, "NO_CERTIFIED_SCALING_IMPROVEMENT_STOP"),
    ),
)
def test_decision_preserves_all_registered_branches(
    execution, tradeoff, improved, capacity, expected
):
    lane = {
        "budget_summaries": [
            {
                "tradeoff_passed": tradeoff,
                "same_budget_alpha_improved": improved,
                "capacity_margin_passed": capacity,
            }
        ]
    }
    assert screen._decision(execution, [lane, lane]) == expected


def test_registered_protocol_tamper_stops_before_execution(tmp_path, monkeypatch):
    output = tmp_path / "screen.json"
    screen.register(output)
    protocol_path = output.with_name("screen_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    protocol["total_candidate_budgets"] = [4096]
    protocol_path.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("tampered protocol reached execution")

    monkeypatch.setattr(screen, "_base_queues", forbidden)
    monkeypatch.setattr(screen, "select_residual_fill_frame", forbidden)
    monkeypatch.setattr(screen, "certified_fractional_frame_upper", forbidden)
    monkeypatch.setattr(screen, "capacity_certificate", forbidden)
    monkeypatch.setattr(residual, "linprog", forbidden)

    with pytest.raises(ValueError, match="protocol changed"):
        screen.run_registered(output)


def test_temp_registered_run_preserves_call_and_work_identities(
    tmp_path, monkeypatch
):
    output = tmp_path / "screen.json"
    screen.register(output)
    calls = {
        "r2_oracle": 0,
        "fractional": 0,
        "residual": 0,
        "legacy": 0,
        "capacity": 0,
    }

    def fake_r2_oracle(model, queue):
        calls["r2_oracle"] += 1
        return 0.0, screen.R2_ORDERED_ACTION_COUNT

    def fake_fractional(options, queue, lanes, idle_duration=1.0):
        calls["fractional"] += 1
        return residual.FractionalFrameUpperCertificate(
            upper_bound=0.0,
            raw_primal_max_diagnostic=0.0,
            threshold_lp_count=screen.OPTION_COUNT,
            total_variables=45,
            total_constraint_rows=36,
            variable_constraint_cells=180,
            objective_terms=45,
            repaired_threshold_count=0,
            exact_dual_feasible=True,
        )

    def fake_residual(options, queue, lanes, budget, idle_duration=1.0):
        calls["residual"] += 1
        gap = 0.1 * sum(queue)
        return SimpleNamespace(
            option_indices=(None,) * lanes,
            service=(0, 0, 0),
            duration=1.0,
            score=0.0,
            upper_bound=gap,
            gap=gap,
            epsilon=gap / sum(queue) if sum(queue) else 0.0,
            attempts=10,
            seed_attempts=10,
            feasible_candidate_count=1,
            local_score_evaluations=9,
            bound_option_probes=9,
            bound_threshold_evaluations=9,
            frontier_peak=1,
            exact=gap == 0.0,
            legacy_budget=budget,
            legacy_attempts=10,
            legacy_score=0.0,
            legacy_upper_bound=gap,
            pair_score=0.0,
            pair_candidate_attempts=0,
            pair_candidate_score_evaluations=0,
            pair_local_score_evaluations=9,
            pair_generation_probes=0,
            zero_fill_probes=0,
            duplicate_multiset_probes=0,
            pair_family_exhausted=True,
            pair_budget_blocked_unique_candidate=False,
            anchor_capacity_divisions=0,
            fill_capacity_divisions=0,
            residual_coordinate_evaluations=0,
            dedup_key_item_reads=0,
            fractional_upper_bound=gap,
            raw_fractional_primal_max_diagnostic=gap,
            threshold_lp_count=9,
            lp_total_variables=45,
            lp_total_constraint_rows=36,
            lp_variable_constraint_cells=180,
            lp_objective_terms=45,
            lp_repaired_threshold_count=0,
            lp_exact_dual_feasible=True,
            charged_logical_work=100,
        )

    def fake_legacy(options, queue, lanes, budget, idle_duration=1.0):
        calls["legacy"] += 1
        gap = 0.2 * sum(queue)
        return SimpleNamespace(
            option_indices=(None,) * lanes,
            score=0.0,
            upper_bound=gap,
            gap=gap,
        )

    def fake_capacity(model, load):
        calls["capacity"] += 1
        return {
            "load_factor": load,
            "effective_slack_eta": 0.04,
            "capacity_projection_passed": True,
            "conditional_model_drift_certified": True,
            "physical_service_confidence_increased": False,
        }

    monkeypatch.setattr(screen, "_r2_oracle", fake_r2_oracle)
    monkeypatch.setattr(screen, "certified_fractional_frame_upper", fake_fractional)
    monkeypatch.setattr(screen, "select_residual_fill_frame", fake_residual)
    monkeypatch.setattr(screen, "select_budgeted_frame", fake_legacy)
    monkeypatch.setattr(screen, "capacity_certificate", fake_capacity)

    result = screen.run_registered(output)

    expected = result["protocol"]["expected_calls"]
    assert calls == {
        "r2_oracle": screen.BASE_STATE_COUNT,
        "fractional": expected["r2_fractional_upper"],
        "residual": expected["residual_fill_selector"],
        "legacy": expected["fresh_legacy_k8192_selector"],
        "capacity": expected["capacity_certificate"],
    }
    assert result["status"] == "PASS"
    assert result["scientific_decision"] == (
        "CERTIFIED_IMPROVEMENT_WITH_HIGH_LOAD_FAIL_STOP"
    )
    assert result["work_accounting"] == {
        "residual_fill_selector_calls": expected["residual_fill_selector"],
        "fresh_legacy_k8192_selector_calls": expected[
            "fresh_legacy_k8192_selector"
        ],
        "capacity_certificate_calls": expected["capacity_certificate"],
        "r2_fractional_upper_calls": expected["r2_fractional_upper"],
        "r2_oracle_action_evaluations": expected[
            "r2_oracle_action_evaluations"
        ],
        "scaled_cartesian_action_enumerations": 0,
        "arrival_tapes": 0,
        "trajectory_frames": 0,
        "rng_initializations": 0,
        "random_draws": 0,
    }
    for lane in result["lane_summaries"]:
        for row in lane["budget_summaries"]:
            assert row["candidate_attempts"] == screen.BASE_STATE_COUNT * 10
            assert row["charged_logical_work"] == screen.BASE_STATE_COUNT * 100
            assert row["threshold_lp_count"] == screen.BASE_STATE_COUNT * 9
            assert row["lp_variables"] == screen.BASE_STATE_COUNT * 45
            assert row["lp_constraint_rows"] == screen.BASE_STATE_COUNT * 36
            assert row["lp_variable_constraint_cells"] == (
                screen.BASE_STATE_COUNT * 180
            )


def test_registered_source_baseline_drift_stops_before_grid(tmp_path, monkeypatch):
    output = tmp_path / "screen.json"
    screen.register(output)
    contract, model = screen._source_contract()
    changed = deepcopy(contract)
    changed["scaling_k4096_source_rows_by_lane"]["8"][
        "max_certified_normalized_gap"
    ] += 1.0
    monkeypatch.setattr(
        screen, "_source_contract", lambda: (changed, model)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("drifted source baseline reached grid execution")

    monkeypatch.setattr(screen, "_base_queues", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        screen.run_registered(output)
