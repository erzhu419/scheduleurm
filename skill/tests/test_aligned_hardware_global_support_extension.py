"""Contracts for the analytic all-integer-queue support extension."""
from __future__ import annotations

import ast
from fractions import Fraction
import json
from pathlib import Path
import shutil

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
import algorithm.budgeted_frame_candidates as budgeted
import algorithm.experiments.aligned_hardware_budget_support_frontier as finite
import algorithm.experiments.aligned_hardware_global_support_extension as extension
from algorithm.experiments.aligned_hardware_frame_model import load_model


def _forbidden(name, calls):
    def fail(*args, **kwargs):
        calls.append((name, args, kwargs))
        pytest.fail(f"registration unexpectedly called {name}")

    return fail


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("global_support_extension_registration")
    output = directory / "extension.json"
    patcher = pytest.MonkeyPatch()
    calls = []
    patcher.setattr(
        budgeted,
        "select_budgeted_frame",
        _forbidden("select_budgeted_frame", calls),
    )
    patcher.setattr(
        finite,
        "_independent_oracle",
        _forbidden("_independent_oracle", calls),
    )
    patcher.setattr(
        extension,
        "_evaluate",
        _forbidden("_evaluate", calls),
    )
    try:
        protocol = extension.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


@pytest.fixture(scope="module")
def executed_bundle(registered_bundle):
    output, protocol = registered_bundle
    result = extension.run_registered(output)
    return output, protocol, result


@pytest.fixture(scope="module")
def dominance_and_kernel():
    dominance = extension._pure_dominance_certificate(load_model())
    kernel = tuple(extension._kernel_queues(dominance))
    return dominance, kernel


@pytest.fixture(scope="module")
def kernel_search_evidence(dominance_and_kernel):
    _dominance, kernel = dominance_and_kernel
    model = load_model()
    positive_k73 = []
    max_k76_attempts = 0
    max_k120_attempts = 0
    for queue in kernel:
        k73 = select_budgeted_frame(model.options, queue, model.lanes, 73)
        k76 = select_budgeted_frame(model.options, queue, model.lanes, 76)
        k120 = select_budgeted_frame(model.options, queue, model.lanes, 120)
        assert k73.score == pytest.approx(k120.score, abs=extension.SCORE_TOLERANCE)
        assert k76.exact and k76.gap == 0.0
        assert k120.exact and k120.gap == 0.0
        assert k76.score == pytest.approx(k120.score, abs=extension.SCORE_TOLERANCE)
        assert k76.attempts == k120.attempts
        max_k76_attempts = max(max_k76_attempts, k76.attempts)
        max_k120_attempts = max(max_k120_attempts, k120.attempts)
        if k73.gap > extension.SCORE_TOLERANCE:
            positive_k73.append((queue, k73))
    return tuple(positive_k73), max_k76_attempts, max_k120_attempts


def _copy_registration(source_output: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target_output = directory / source_output.name
    for source, target in zip(
        extension._paths(source_output)[1:],
        extension._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


def _as_fraction(row):
    return Fraction(row["numerator"], row["denominator"])


def test_registration_freezes_proof_without_selector_or_oracle_work(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = extension._paths(output)

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert protocol["checkpoint_budgets"] == list(range(10, 121))
    assert protocol["pure_option_indices"] == [3, 5, 8]
    assert protocol["expected_replication_thresholds"] == [6, 6, 16]
    assert protocol["expected_tail_maximum"] == [40, 18, 15]
    assert protocol["expected_tail_state_count"] == 41 * 19 * 16
    assert protocol["expected_kernel_state_count"] == 4471
    assert protocol["global_margin_budget"] == 73
    assert protocol["global_exact_budget"] == 76
    assert protocol["tree_exact_budget"] == 120
    assert protocol["actual_exact_budget"] == 51
    assert protocol["timing_is_scientific_gate"] is False

    frozen = json.loads(sources_path.read_text())
    assert extension.THIS_SOURCE in frozen
    assert all(
        (extension.ROOT / name).read_text() == text
        for name, text in frozen.items()
    )


def test_fraction_dominance_excludes_loaded_and_lower_width_maximizers(
    dominance_and_kernel,
):
    dominance, _kernel = dominance_and_kernel
    model = load_model()
    rates = tuple(_as_fraction(row) for row in dominance["pure_rate_fractions"])

    assert dominance["pure_option_indices"] == [3, 5, 8]
    assert dominance["all_options_dominated"]
    assert dominance["replication_thresholds"] == [6, 6, 16]
    assert all(
        _as_fraction(row["coefficient_sum"]) <= 1
        for row in dominance["all_option_dominance"]
    )
    assert all(
        _as_fraction(dominance["all_option_dominance"][index]["coefficient_sum"])
        < 1
        for index in (0, 1, 2, 4, 6, 7)
    )
    assert all(
        _as_fraction(dominance["all_option_dominance"][index]["coefficient_sum"])
        == 1
        for index in extension.PURE_OPTION_INDICES
    )

    for index in (0, 1):
        loaded_rate = Fraction(1, 1) / extension._fraction(
            max(1.0, model.options[index].duration)
        )
        assert loaded_rate / rates[0] + loaded_rate / rates[2] < 1


def test_kernel_membership_preserves_all_tied_maximizers():
    certificate = {
        "pure_rate_fractions": [
            extension._fraction_row(Fraction(1, 1)),
            extension._fraction_row(Fraction(1, 1)),
            extension._fraction_row(Fraction(1, 1)),
        ],
        "replication_thresholds": [3, 2, 2],
        "tail_maximum": [2, 2, 0],
    }
    kernel = set(extension._kernel_queues(certificate))

    # Coordinates 0 and 1 tie. Coordinate 0 is underfilled, but coordinate 1
    # can replicate, so existence of that tied maximizer closes the seed bound.
    assert (2, 2, 0) not in kernel
    assert (2, 1, 0) in kernel
    assert (0, 0, 0) in kernel


def test_tail_box_and_kernel_are_exact_fraction_constructions(
    dominance_and_kernel,
):
    dominance, kernel = dominance_and_kernel
    source, _source_protocol = extension._load_source()
    checks, independently_generated = extension._tail_lemma_checks(dominance, source)
    rates = tuple(_as_fraction(row) for row in dominance["pure_rate_fractions"])
    thresholds = tuple(dominance["replication_thresholds"])
    underfilled_support = max(
        rate * (threshold - 1)
        for rate, threshold in zip(rates, thresholds, strict=True)
    )

    assert tuple(dominance["tail_maximum"]) == (40, 18, 15)
    assert dominance["tail_state_count"] == 41 * 19 * 16 == 12_464
    assert tuple(
        int(underfilled_support // rate) for rate in rates
    ) == (40, 18, 15)
    assert len(kernel) == len(set(kernel)) == 4471
    assert kernel == independently_generated
    assert kernel[0] == (0, 0, 0)
    assert max(map(sum, kernel)) == 73
    assert all(checks.values())
    assert all(
        tail <= source_max
        for tail, source_max in zip(
            dominance["tail_maximum"], source["domain"]["maximum"], strict=True
        )
    )


@pytest.mark.parametrize("queue", ((41, 0, 0), (0, 19, 0), (0, 0, 16)))
def test_outside_tail_replicated_seed_meets_root_upper_bound(queue):
    model = load_model()
    dominance = extension._pure_dominance_certificate(model)
    pure_rates = tuple(
        float(_as_fraction(row)) for row in dominance["pure_rate_fractions"]
    )
    local_envelope = max(q * rate for q, rate in zip(queue, pure_rates, strict=True))
    raw_actions = finite._raw_action_family(model)
    raw_upper = max(
        sum(q * done for q, done in zip(queue, service, strict=True)) / duration
        for _indices, service, duration, _key in raw_actions
    )
    selected = select_budgeted_frame(model.options, queue, model.lanes, 10)

    assert raw_upper <= 2.0 * local_envelope + extension.SCORE_TOLERANCE
    assert selected.attempts == 10
    assert selected.exact and selected.gap == 0.0
    assert selected.score == pytest.approx(2.0 * local_envelope)
    assert selected.upper_bound == pytest.approx(selected.score)


def test_global_curve_is_exactly_inherited_from_the_containing_source_box():
    source, _source_protocol = extension._load_source()
    curve = extension._global_curve(source)

    assert [row["budget"] for row in curve] == list(range(10, 121))
    for global_row, source_row in zip(
        curve, source["budget_summaries"], strict=True
    ):
        assert global_row["global_max_returned_normalized_gap"] == source_row[
            "max_certified_normalized_gap"
        ]
        assert global_row["global_max_actual_normalized_regret_offline"] == source_row[
            "max_actual_normalized_regret_diagnostic"
        ]
        assert global_row["returned_gap_witness"] == source_row["alpha_witness"]
        assert global_row["source_selector_work"] == source_row["selector_work"]


def test_k73_has_only_two_bound_slack_witnesses_and_zero_actual_regret(
    kernel_search_evidence,
):
    positive, _max76, _max120 = kernel_search_evidence

    assert [queue for queue, _selection in positive] == [(3, 2, 5), (4, 2, 5)]
    first, second = (selection for _queue, selection in positive)
    assert first.option_indices == second.option_indices == (6, 7)
    assert first.gap == pytest.approx(0.04137438812762806)
    assert first.epsilon == pytest.approx(0.004137438812762806)
    assert second.gap == pytest.approx(0.10871767422208833)
    assert second.epsilon == pytest.approx(0.009883424929280758)
    assert first.attempts == second.attempts == 73


def test_k76_and_k120_are_globally_exact_with_same_actual_work(
    kernel_search_evidence,
):
    _positive, max76, max120 = kernel_search_evidence

    assert max76 == max120 == 76


def test_registered_result_reports_bounded_claim_and_no_new_runtime_work(
    executed_bundle,
):
    _output, _protocol, result = executed_bundle

    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["theorem"]["queue_domain"] == "all q in Z_+^3"
    assert result["analytic_tail"]["maximum"] == [40, 18, 15]
    assert result["analytic_tail"]["kernel_state_count_including_zero"] == 4471
    assert result["primary_k73"]["global_max_returned_normalized_gap"] == pytest.approx(
        0.009883424929280758
    )
    assert result["primary_k73"]["global_max_actual_normalized_regret_offline"] == 0.0
    assert result["primary_k73"]["returned_gap_witness"]["queue"] == [4, 2, 5]
    assert result["returned_certificate_exact_k76"][
        "global_max_returned_normalized_gap"
    ] == 0.0
    assert result["finite_tree_exact_k120"]["complete_tree_attempt_bound"] == 120
    assert result["finite_tree_exact_k120"][
        "global_max_returned_normalized_gap"
    ] == 0.0
    assert result["compute_comparison"]["paid_attempts_saved"] == 6
    assert result["compute_comparison"]["scientific_interpretation"] == "negligible"
    assert result["proof_work"] == {
        "fraction_dominance_rows": 9,
        "tail_integer_states_classified": 12_464,
        "kernel_states_retained": 4471,
        "source_budget_rows_reused": 111,
        "selector_calls": 0,
        "oracle_action_checks": 0,
        "arrival_tapes": 0,
        "trajectory_frames": 0,
        "rng_or_random_draws": 0,
    }
    assert result["scientific_decision"] == (
        "GLOBAL_K73_RETURNED_GAP_CERTIFIED_K76_EXACT_COMPUTE_GAIN_NEGLIGIBLE_CLOSE"
    )
    assert not result["development_advance"]
    assert all(
        row["global_residual_margin_positive"]
        and row["stronger_k_independent_homogeneous_seed_threshold"]
        < row["generic_global_negative_drift_backlog_threshold"]
        for row in result["capacity_bridge_at_k73"]
    )


def test_registration_and_result_are_write_new_and_tamper_checked(
    executed_bundle, tmp_path, monkeypatch,
):
    output, _protocol, _result = executed_bundle
    with pytest.raises(ValueError, match="already exist"):
        extension.register(output)
    with pytest.raises(ValueError, match="already exists"):
        extension.run_registered(output)

    def forbidden(*args, **kwargs):
        pytest.fail("tampered registration reached proof evaluation")

    protocol_target = _copy_registration(output, tmp_path / "protocol")
    protocol_path = extension._paths(protocol_target)[1]
    protocol = json.loads(protocol_path.read_text())
    protocol["expected_kernel_state_count"] += 1
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(extension, "_evaluate", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        extension.run_registered(protocol_target)

    sources_target = _copy_registration(output, tmp_path / "sources")
    sources_path = extension._paths(sources_target)[2]
    sources = json.loads(sources_path.read_text())
    sources[extension.THIS_SOURCE] += "\n# tampered\n"
    sources_path.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match="sources changed"):
        extension.run_registered(sources_target)


def test_runner_has_no_selector_oracle_arrival_trajectory_or_rng_path():
    source = Path(extension.__file__).read_text()
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

    assert "select_budgeted_frame" not in imported_modules | called_names
    assert "_independent_oracle" not in imported_modules | called_names
    assert "random" not in imported_modules
    assert "numpy.random" not in imported_modules
    assert not {
        "simulate_frames",
        "generate_arrival_tape",
        "run_trajectory",
        "run_drain",
    }.intersection(called_names)
