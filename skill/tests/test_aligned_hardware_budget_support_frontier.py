"""Contracts for the finite-box legacy-v4 budget/support frontier."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import shutil

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_budgeted_frame_search_curve import (
    trace_budgeted_frame_search_curve,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)
import algorithm.experiments.aligned_hardware_budget_support_frontier as frontier


def _forbid_execution(patcher: pytest.MonkeyPatch):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"registration unexpectedly called {name}")

        return fail

    for name in (
        "trace_budgeted_frame_search_curve",
        "select_budgeted_frame",
        "_raw_action_family",
        "_independent_oracle",
        "_domain_queues",
    ):
        patcher.setattr(frontier, name, forbidden(name))
    return calls


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("budget_support_frontier_registration")
    output = directory / "frontier.json"
    patcher = pytest.MonkeyPatch()
    calls = _forbid_execution(patcher)
    try:
        protocol = frontier.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target_output = destination / source_output.name
    for source, target in zip(
        frontier._paths(source_output)[1:],
        frontier._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


def test_registration_freezes_complete_box_and_calls_no_search(registered_bundle):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = frontier._paths(output)

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert protocol["checkpoint_budgets"] == list(range(10, 121))
    assert protocol["anchor_budgets"] == [10, 14, 26, 120]
    assert protocol["envelope_min"] == [0, 0, 0]
    assert protocol["envelope_max"] == [40, 25, 24]
    assert protocol["envelope_shape"] == [41, 26, 25]
    assert protocol["envelope_state_count"] == 41 * 26 * 25 == 26_650
    assert protocol["envelope_max_l1"] == 89
    assert protocol["raw_action_count"] == 100
    assert protocol["rule"] == frontier.RULE
    assert protocol["advance_gate"] == frontier.ADVANCE_GATE
    assert protocol["decision_rule"] == frontier.DECISION_RULE
    assert protocol["timing_is_scientific_gate"] is False

    frozen = json.loads(sources_path.read_text())
    assert set(frontier.NEW_SOURCES).issubset(frozen)
    assert all(
        (frontier.ROOT / name).read_text() == text
        for name, text in frozen.items()
    )
    for certificate in protocol["capacity_certificates"]:
        assert certificate["drift_constant"] == pytest.approx(
            certificate["second_moment_B"]
            + certificate["tau_max"] * certificate["availability_loss_C"]
        )


def test_domain_generator_is_the_complete_registered_coordinate_box():
    queues = tuple(frontier._domain_queues())

    assert len(queues) == len(set(queues)) == frontier.ENVELOPE_STATE_COUNT
    assert queues[0] == frontier.ENVELOPE_MIN
    assert queues[-1] == frontier.ENVELOPE_MAX
    assert max(map(sum, queues)) == frontier.ENVELOPE_MAX_L1 == 89
    assert (40, 0, 0) in queues
    assert (0, 25, 0) in queues
    assert (0, 0, 24) in queues


def test_independent_oracle_covers_support_while_global_tie_is_diagnostic():
    model = load_model()
    raw_actions = frontier._raw_action_family(model)
    queue = (0, 0, 4)
    oracle = frontier._independent_oracle(queue, raw_actions)
    selected = select_budgeted_frame(
        model.options, queue, model.lanes, frontier.EXACT_BUDGET
    )

    independent_scores = [
        sum(q * done for q, done in zip(queue, service, strict=True)) / duration
        for _indices, service, duration, _key in raw_actions
        if all(done <= waiting for done, waiting in zip(service, queue, strict=True))
    ]
    assert len(raw_actions) == frontier.RAW_ACTION_COUNT == 100
    assert oracle["score"] == max(independent_scores)
    assert selected.exact
    assert selected.score == pytest.approx(oracle["score"])
    assert tuple(selected.option_indices) != tuple(oracle["option_indices"])
    assert frontier._selection_reconstructs(model, queue, selected)
    assert "diagnostic" in frontier.RULE["independent_oracle"]


def _certificate(*, eta=2.0, drift_constant=100.0):
    tau_max = 3.0
    availability_loss = 20.0
    second_moment = drift_constant - tau_max * availability_loss
    return {
        "load_factor": 0.5,
        "support_slack_delta": eta,
        "projection_support_loss_rate": 0.0,
        "effective_slack_eta": eta,
        "second_moment_B": second_moment,
        "availability_loss_C": availability_loss,
        "tau_min": 1.0,
        "tau_max": tau_max,
        "drift_constant": drift_constant,
        "negative_drift_backlog_threshold": drift_constant / eta,
    }


def test_capacity_bridge_uses_residual_eta_and_same_drift_constant():
    certificate = _certificate()
    exact = frontier._capacity_bridge_row(certificate, alpha=0.0)

    assert frontier._capacity_bridge_reconstructs(exact)
    assert exact["drift_constant"] == exact["drift_constant_reconstructed"]
    assert exact["generic_finite_box_residual_margin"] == 2.0
    assert exact["algebraic_threshold_if_alpha_extended_globally"] == 50.0
    assert exact["minimum_integer_backlog_above_threshold"] == 51
    assert exact["threshold_over_seed_specific_ratio"] == 1.0
    assert exact["threshold_relative_increase"] == 0.0
    assert exact["threshold_absolute_increase"] == 0.0

    approximate = frontier._capacity_bridge_row(certificate, alpha=0.5)
    assert frontier._capacity_bridge_reconstructs(approximate)
    expected_threshold = 100.0 / 1.5
    assert approximate["generic_finite_box_residual_margin"] == 1.5
    assert approximate["algebraic_threshold_if_alpha_extended_globally"] == pytest.approx(
        expected_threshold
    )
    assert approximate["threshold_over_seed_specific_ratio"] == pytest.approx(
        2.0 / 1.5
    )
    assert approximate["threshold_relative_increase"] == pytest.approx(
        0.5 / 1.5
    )
    assert approximate["threshold_absolute_increase"] == pytest.approx(
        expected_threshold - 50.0
    )
    assert approximate["generic_drift_upper_bound_at_domain_max_l1"] == pytest.approx(
        100.0 - 1.5 * 89
    )


@pytest.mark.parametrize("alpha", (2.0, 2.5))
def test_nonpositive_residual_margin_has_no_backlog_threshold(alpha):
    row = frontier._capacity_bridge_row(_certificate(), alpha=alpha)

    assert frontier._capacity_bridge_reconstructs(row)
    assert not row["generic_residual_margin_positive"]
    assert row["algebraic_threshold_if_alpha_extended_globally"] is None
    assert row["minimum_integer_backlog_above_threshold"] is None
    assert row["threshold_over_seed_specific_ratio"] is None
    assert row["threshold_relative_increase"] is None
    assert row["threshold_absolute_increase"] is None
    assert not row["negative_drift_region_intersects_domain"]


def test_finite_box_intersection_uses_a_strict_threshold():
    certificate = _certificate(drift_constant=89.0)
    at_boundary = frontier._capacity_bridge_row(certificate, alpha=1.0)
    just_inside = frontier._capacity_bridge_row(certificate, alpha=0.999)

    assert at_boundary["algebraic_threshold_if_alpha_extended_globally"] == 89.0
    assert not at_boundary["negative_drift_region_intersects_domain"]
    assert at_boundary["generic_drift_upper_bound_at_domain_max_l1"] == 0.0
    assert just_inside["algebraic_threshold_if_alpha_extended_globally"] < 89.0
    assert just_inside["negative_drift_region_intersects_domain"]
    assert just_inside["generic_drift_upper_bound_at_domain_max_l1"] < 0.0


def test_empty_queue_is_excluded_from_normalized_maximum_witnesses():
    model = load_model()
    accumulator = frontier._new_budget_accumulator(frontier.EXACT_BUDGET)
    empty = (0, 0, 0)
    empty_selection = select_budgeted_frame(
        model.options, empty, model.lanes, frontier.EXACT_BUDGET
    )
    frontier._update_budget(accumulator, empty, empty_selection, optimum=0.0)

    for field, witness in (
        ("max_certified_normalized_gap", "alpha_witness"),
        ("max_actual_normalized_regret_diagnostic", "actual_regret_witness"),
        ("max_frontier_normalized_slack_diagnostic", "frontier_slack_witness"),
    ):
        assert accumulator[field] == -1.0
        assert accumulator[witness] is None

    nonempty = (0, 0, 4)
    raw_actions = frontier._raw_action_family(model)
    optimum = frontier._independent_oracle(nonempty, raw_actions)["score"]
    selection = select_budgeted_frame(
        model.options, nonempty, model.lanes, frontier.EXACT_BUDGET
    )
    frontier._update_budget(accumulator, nonempty, selection, optimum)
    finalized = frontier._finalize_budget(accumulator, certificates=[])

    assert finalized["nonempty_state_count"] == 1
    assert finalized["max_certified_normalized_gap"] == pytest.approx(0.0)
    assert finalized["alpha_witness"]["queue"] == nonempty
    assert finalized["actual_regret_witness"]["queue"] == nonempty
    assert finalized["frontier_slack_witness"]["queue"] == nonempty


def test_certified_maximum_never_rounds_down_within_aggregate_tolerance():
    accumulator = frontier._new_budget_accumulator(26)
    lower = {"queue": (0, 0, 2)}
    higher = {"queue": (0, 0, 3)}
    frontier._update_maximum(
        accumulator,
        "max_certified_normalized_gap",
        "alpha_witness",
        1.0,
        lower,
    )
    frontier._update_maximum(
        accumulator,
        "max_certified_normalized_gap",
        "alpha_witness",
        1.0 + frontier.AGGREGATE_TOLERANCE / 2.0,
        higher,
    )

    assert accumulator["max_certified_normalized_gap"] > 1.0
    assert accumulator["alpha_witness"] is higher


def _summary(budget, *, positive, attempts):
    return {
        "budget": budget,
        "capacity_bridge": [
            {"generic_residual_margin_positive": positive} for _ in frontier.LOADS
        ],
        "selector_work": {"candidate_attempts": attempts},
    }


def test_decision_has_four_unambiguous_branches():
    vacuous = [
        _summary(26, positive=False, attempts=50),
        _summary(120, positive=True, attempts=100),
    ]
    no_saving = [
        _summary(26, positive=True, attempts=100),
        _summary(120, positive=True, attempts=100),
    ]
    advance = [
        _summary(14, positive=False, attempts=30),
        _summary(26, positive=True, attempts=90),
        _summary(120, positive=True, attempts=100),
    ]

    assert frontier._decision(False, advance) == ("INVALID_EXECUTION", None)
    assert frontier._decision(True, vacuous) == (
        "FINITE_BOX_MARGIN_VACUOUS_STOP",
        None,
    )
    assert frontier._decision(True, no_saving) == (
        "POSITIVE_MARGIN_WITHOUT_COMPUTE_SAVING_STOP",
        26,
    )
    assert frontier._decision(True, advance) == (
        "POSITIVE_FINITE_BOX_MARGIN_ADVANCE_TO_GLOBAL_EXTENSION",
        26,
    )


def test_runner_functions_reconstruct_a_small_curve_without_domain_execution(monkeypatch):
    model = load_model()
    queue = (4, 3, 5)
    budgets = (10, 26, 120)
    monkeypatch.setattr(frontier, "CHECKPOINT_BUDGETS", budgets)
    curve = trace_budgeted_frame_search_curve(
        model.options, queue, model.lanes, checkpoint_budgets=budgets
    )
    oracle = frontier._independent_oracle(
        queue, frontier._raw_action_family(model)
    )

    assert frontier._curve_monotone(curve)
    assert frontier._work_identities(curve, len(model.options))
    assert all(
        frontier._selection_reconstructs(model, queue, checkpoint.selection)
        for checkpoint in curve.checkpoints
    )
    assert curve.selection_at(120).score == pytest.approx(oracle["score"])

    accumulator = frontier._new_budget_accumulator(26)
    frontier._update_budget(
        accumulator, queue, curve.selection_at(26), oracle["score"]
    )
    summary = frontier._finalize_budget(
        accumulator, [capacity_certificate(model, 0.5)]
    )
    assert summary["domain_state_count"] == 1
    assert summary["nonempty_state_count"] == 1
    assert summary["alpha_witness"]["queue"] == queue
    assert summary["capacity_bridge"][0]["finite_box_approximate_oracle_alpha_1"] == (
        pytest.approx(curve.selection_at(26).epsilon)
    )


def test_registered_artifacts_are_write_new_and_tamper_checked(
    registered_bundle, tmp_path, monkeypatch,
):
    output, _protocol = registered_bundle
    with pytest.raises(ValueError, match="already exist"):
        frontier.register(output)

    existing = _copy_registered_bundle(output, tmp_path / "existing")
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("{}\n")
    with pytest.raises(ValueError, match="already exists"):
        frontier.run_registered(existing)

    def forbidden(*args, **kwargs):
        pytest.fail("tampered registration reached domain execution")

    protocol_target = _copy_registered_bundle(output, tmp_path / "protocol")
    protocol_path = frontier._paths(protocol_target)[1]
    protocol = json.loads(protocol_path.read_text())
    protocol["envelope_max_l1"] += 1
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(frontier, "trace_budgeted_frame_search_curve", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        frontier.run_registered(protocol_target)

    sources_target = _copy_registered_bundle(output, tmp_path / "sources")
    sources_path = frontier._paths(sources_target)[2]
    sources = json.loads(sources_path.read_text())
    first = next(iter(sources))
    sources[first] += "\n# tampered\n"
    sources_path.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match="sources changed"):
        frontier.run_registered(sources_target)


def test_runner_has_no_arrival_trajectory_drain_or_rng_execution_path():
    source = Path(frontier.__file__).read_text()
    tree = ast.parse(source)
    imported_modules = {
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

    assert "random" not in imported_modules
    assert "numpy.random" not in imported_modules
    assert not {
        "simulate_frames",
        "generate_arrival_tape",
        "conditional_lyapunov_drift",
        "run_trajectory",
        "run_drain",
    }.intersection(called_names)
    assert all(
        word in frontier.PROHIBITED
        for word in ("arrivals", "trajectory", "drain", "RNG")
    )
