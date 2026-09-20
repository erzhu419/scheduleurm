"""Contracts for the frozen finite-grid Cartesian-scaling experiment."""
from __future__ import annotations

import ast
from itertools import product
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_cartesian_scaling as scaling


def _forbid_registration_execution(patcher: pytest.MonkeyPatch):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"registration unexpectedly called {name}")

        return fail

    for name in (
        "base_queues",
        "scaled_queues",
        "capacity_certificate",
        "trace_budgeted_frame_search_curve",
        "_decision",
    ):
        patcher.setattr(scaling, name, forbidden(name))
    return calls


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cartesian_scaling_registration")
    output = directory / "scaling.json"
    patcher = pytest.MonkeyPatch()
    calls = _forbid_registration_execution(patcher)
    try:
        protocol = scaling.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def _copy_registration(source_output: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target_output = directory / source_output.name
    for source, target in zip(
        scaling._paths(source_output)[1:],
        scaling._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


def test_registration_freezes_scaling_protocol_without_experiment_execution(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = scaling._paths(output)

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert protocol["base_state_count"] == 1000
    assert protocol["base_levels"] == {
        "cnn": list(scaling.BASE_CNN_LEVELS),
        "jax": list(scaling.BASE_JAX_LEVELS),
        "llm": list(scaling.BASE_LLM_LEVELS),
    }
    assert protocol["lane_counts"] == [2, 4, 8, 16]
    assert protocol["loads"] == [0.50, 0.80, 0.95]
    assert protocol["checkpoint_budgets"] == list(scaling.CHECKPOINT_BUDGETS)
    assert protocol["raw_ordered_action_count_by_lane"] == {
        str(lanes): 10 ** lanes for lanes in scaling.LANE_COUNTS
    }
    assert protocol["expected_trace_calls"] == 4000
    assert protocol["expected_capacity_certificate_calls"] == 12
    assert protocol["rule"] == scaling.RULE
    assert protocol["decision_rule"] == scaling.DECISION_RULE
    assert protocol["timing_is_scientific_gate"] is False

    frozen = json.loads(sources_path.read_text())
    assert scaling.THIS_SOURCE in frozen
    assert all(
        (scaling.ROOT / name).read_text() == text
        for name, text in frozen.items()
    )


def test_base_grid_is_the_exact_ordered_ten_by_ten_by_ten_product():
    expected = tuple(
        product(
            scaling.BASE_CNN_LEVELS,
            scaling.BASE_JAX_LEVELS,
            scaling.BASE_LLM_LEVELS,
        )
    )
    queues = scaling.base_queues()

    assert all(
        len(levels) == len(set(levels)) == 10
        for levels in (
            scaling.BASE_CNN_LEVELS,
            scaling.BASE_JAX_LEVELS,
            scaling.BASE_LLM_LEVELS,
        )
    )
    assert queues == expected
    assert len(queues) == len(set(queues)) == scaling.BASE_STATE_COUNT == 1000
    assert queues[0] == (0, 0, 0)
    assert queues[-1] == (40, 25, 24)


@pytest.mark.parametrize("lanes", (2, 4, 8, 16))
def test_scaled_grid_is_exact_integer_multiplication(lanes):
    factor = lanes // 2
    expected = tuple(
        tuple(factor * coordinate for coordinate in queue)
        for queue in scaling.base_queues()
    )

    assert scaling.scaled_queues(lanes) == expected
    assert len(expected) == len(set(expected)) == 1000
    assert expected[-1] == (40 * factor, 25 * factor, 24 * factor)


def test_scaled_grid_rejects_unregistered_lane_counts():
    with pytest.raises(ValueError, match="lanes must be one of"):
        scaling.scaled_queues(6)


def test_analytic_action_counts_are_exact_powers_and_never_materialized():
    counts = scaling._raw_action_counts()

    assert counts == {
        str(lanes): 10 ** lanes for lanes in scaling.LANE_COUNTS
    }
    assert counts == {
        "2": 100,
        "4": 10_000,
        "8": 100_000_000,
        "16": 10_000_000_000_000_000,
    }


def _lane_summary(lanes, *, quality=None, compute=None, tradeoff=None):
    return {
        "lanes": lanes,
        "first_quality_budget": quality,
        "first_compute_budget": compute,
        "first_tradeoff_budget": tradeoff,
    }


def test_decision_rule_has_all_frozen_branches():
    advance = [
        _lane_summary(2),
        *(
            _lane_summary(lanes, quality=26, compute=26, tradeoff=26)
            for lanes in (4, 8, 16)
        ),
    ]
    loose = [
        _lane_summary(2),
        _lane_summary(4, compute=26),
        _lane_summary(8, compute=26),
        _lane_summary(16, compute=26),
    ]
    expensive = [
        _lane_summary(2),
        _lane_summary(4, quality=26),
        _lane_summary(8, quality=26),
        _lane_summary(16, quality=26),
    ]
    crossed = [
        _lane_summary(2),
        _lane_summary(4, quality=26, compute=54),
        _lane_summary(8, quality=26, compute=54),
        _lane_summary(16, quality=26, compute=54),
    ]

    assert scaling._decision(False, advance) == "INVALID_EXECUTION"
    assert scaling._decision(True, advance) == "MEANINGFUL_LAZY_SCALING_ADVANCE"
    assert scaling._decision(True, loose) == (
        "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP"
    )
    assert scaling._decision(True, expensive) == (
        "CERTIFICATE_WITHOUT_COMPUTE_SAVING_STOP"
    )
    assert scaling._decision(True, crossed) == "NO_SCALING_TRADEOFF_STOP"


def _selection(*, gap, upper, exact, attempts=10):
    return SimpleNamespace(
        option_indices=(None,) * 4,
        service=(0, 0, 0),
        duration=1.0,
        score=upper - gap,
        upper_bound=upper,
        gap=gap,
        epsilon=0.0,
        attempts=attempts,
        seed_attempts=10,
        feasible_candidate_count=1,
        local_score_evaluations=9,
        bound_option_probes=3,
        bound_threshold_evaluations=2,
        frontier_peak=1,
        exact=exact,
    )


def test_budget_summary_uses_10_to_the_lane_denominator_and_same_k_tradeoff():
    accumulator = scaling._new_budget_accumulator(26)
    scaling._update_budget(
        accumulator,
        (0, 0, 0),
        (0, 0, 0),
        _selection(gap=0.0, upper=0.0, exact=True),
    )
    selection = _selection(gap=1.0, upper=2.0, exact=False)
    scaling._update_budget(
        accumulator,
        (1, 0, 0),
        (2, 0, 0),
        selection,
    )
    rows = scaling._finalize_budget(
        accumulator,
        lanes=4,
        capacity_rows=[{"load": 0.95, "effective_slack_eta": 0.6}],
    )

    assert rows["state_count"] == 2
    assert rows["nonempty_state_count"] == 1
    assert rows["max_certified_normalized_gap"] == pytest.approx(0.5)
    assert rows["mean_certified_normalized_gap"] == pytest.approx(0.5)
    assert rows["max_relative_gap_diagnostic"] == pytest.approx(0.5)
    assert rows["raw_ordered_actions_per_state"] == 10 ** 4
    assert rows["candidate_attempt_fraction_of_raw_enumeration"] == pytest.approx(
        20 / (2 * 10 ** 4)
    )
    expected_charged = 20 + 18 + 6 + 4
    assert rows["charged_logical_work"] == expected_charged
    assert rows["charged_work_fraction_of_raw_enumeration"] == pytest.approx(
        expected_charged / (2 * 10 ** 4)
    )
    assert rows["capacity_margin_passed"]
    assert rows["computation_passed"]
    assert rows["tradeoff_passed"]


def test_small_monkeypatched_registered_run_reconstructs_curves_and_work(
    tmp_path, monkeypatch,
):
    small_base = (
        (0, 0, 0),
        (4, 2, 5),
        (5, 3, 5),
        (40, 25, 24),
    )
    monkeypatch.setattr(scaling, "BASE_STATE_COUNT", len(small_base))
    monkeypatch.setattr(scaling, "LANE_COUNTS", (2, 4))
    monkeypatch.setattr(scaling, "CHECKPOINT_BUDGETS", (10, 120))
    monkeypatch.setattr(scaling, "base_queues", lambda: small_base)
    output = tmp_path / "small_scaling.json"

    protocol = scaling.register(output)
    result = scaling.run_registered(output)

    assert protocol["expected_trace_calls"] == 8
    assert protocol["expected_capacity_certificate_calls"] == 6
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["validation_work"] == {
        "continuous_trace_calls": 8,
        "capacity_certificate_calls": 6,
        "full_action_enumeration_calls": 0,
        "oracle_calls": 0,
        "arrival_or_trajectory_calls": 0,
        "per_state_rows_retained": 0,
    }
    assert [row["lanes"] for row in result["lane_summaries"]] == [2, 4]
    for lane_summary, trace_work in zip(
        result["lane_summaries"], result["trace_work_by_lane"], strict=True
    ):
        rows = lane_summary["budget_summaries"]
        assert [row["budget"] for row in rows] == [10, 120]
        assert lane_summary["domain_state_count"] == len(small_base)
        assert lane_summary["raw_ordered_actions_per_state"] == (
            10 ** lane_summary["lanes"]
        )
        assert lane_summary["first_quality_budget"] == scaling._first_budget(
            rows, "capacity_margin_passed"
        )
        assert lane_summary["first_compute_budget"] == scaling._first_budget(
            rows, "computation_passed"
        )
        assert lane_summary["first_tradeoff_budget"] == scaling._first_budget(
            rows, "tradeoff_passed"
        )
        assert trace_work["lanes"] == lane_summary["lanes"]
        assert trace_work["trace_calls"] == len(small_base)
        assert trace_work["checkpoint_reads"] == 2 * len(small_base)
    assert result["scientific_decision"] == scaling._decision(
        True, result["lane_summaries"]
    )
    assert result["development_advance"] == (
        result["scientific_decision"] == "MEANINGFUL_LAZY_SCALING_ADVANCE"
    )


def test_registration_and_result_are_write_new_and_tamper_checked(
    registered_bundle, tmp_path, monkeypatch,
):
    output, _protocol = registered_bundle
    with pytest.raises(ValueError, match="already exist"):
        scaling.register(output)

    existing = _copy_registration(output, tmp_path / "existing")
    existing.write_text("{}\n")
    with pytest.raises(ValueError, match="already exists"):
        scaling.run_registered(existing)

    def forbidden(*args, **kwargs):
        pytest.fail("tampered registration reached experiment execution")

    protocol_target = _copy_registration(output, tmp_path / "protocol")
    protocol_path = scaling._paths(protocol_target)[1]
    protocol = json.loads(protocol_path.read_text())
    protocol["base_state_count"] += 1
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(scaling, "capacity_certificate", forbidden)
    monkeypatch.setattr(scaling, "trace_budgeted_frame_search_curve", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        scaling.run_registered(protocol_target)

    sources_target = _copy_registration(output, tmp_path / "sources")
    sources_path = scaling._paths(sources_target)[2]
    sources = json.loads(sources_path.read_text())
    sources[scaling.THIS_SOURCE] += "\n# tampered\n"
    sources_path.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match="sources changed"):
        scaling.run_registered(sources_target)


def test_runner_has_no_oracle_full_action_rng_or_trajectory_execution_path():
    source = Path(scaling.__file__).read_text()
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    product_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "product"
    ]

    assert not {
        "select_budgeted_frame",
        "_raw_action_family",
        "_independent_oracle",
    }.intersection(imported_names | called_names)
    assert "random" not in imported_names
    assert "numpy.random" not in imported_names
    assert not {
        "simulate_frames",
        "generate_arrival_tape",
        "run_trajectory",
        "run_drain",
    }.intersection(called_names)
    assert len(product_calls) == 1
    assert not product_calls[0].keywords
    assert all(
        phrase in scaling.PROHIBITED
        for phrase in ("full Cartesian", "oracle", "trajectories", "RNG")
    )
