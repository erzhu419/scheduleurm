"""Pre-run contracts for the action-preserving FIFO certificate sidecar."""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
import algorithm.experiments.aligned_hardware_fifo_execution_certificate_sidecar as sidecar
from algorithm.fifo_execution_quotient_frame_candidates import (
    select_fifo_execution_quotient_frame,
)


def _forbid_registered_execution(patcher: pytest.MonkeyPatch):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"registration unexpectedly called {name}")

        return fail

    patcher.setattr(sidecar, "select_budgeted_frame", forbidden("legacy selector"))
    patcher.setattr(
        sidecar,
        "select_fifo_execution_quotient_frame",
        forbidden("FIFO selector"),
    )
    patcher.setattr(sidecar, "_evaluate_state", forbidden("state evaluation"))
    return calls


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fifo_certificate_registration")
    output = directory / "sidecar.json"
    patcher = pytest.MonkeyPatch()
    calls = _forbid_registered_execution(patcher)
    try:
        protocol = sidecar.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    for source, target in zip(
        sidecar._paths(source_output)[1:],
        sidecar._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


def test_registration_freezes_action_preserving_rule_without_running_selectors(
    registered_bundle,
):
    output, protocol = registered_bundle
    result_path, protocol_path, sources_path = sidecar._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert "states" not in protocol
    assert protocol["state_count"] == sidecar.STATE_COUNT == 5003
    assert protocol["loads"] == [0.5, 0.8, 0.95]
    assert protocol["seeds"] == [41, 42, 43]
    assert protocol["source_policy_budgets"] == [26, 120]
    assert protocol["small_checkpoint_budget"] == 26
    assert protocol["rule"] == sidecar.RULE
    assert protocol["advance_gate"] == sidecar.ADVANCE_GATE
    assert protocol["timing_protocol"] == sidecar.TIMING_PROTOCOL
    assert protocol["predecessor_facts"]["fifo_shadow"] == {
        "experiment": "aligned_hardware_fifo_execution_quotient_state_shadow_v1",
        "status": "PASS",
        "state_evaluation_count": 5003,
        "k71_support_oracle_check": True,
        "k71_witness_replay_check": True,
    }
    assert "minimum" in protocol["rule"]["sidecar_upper_bound"]
    assert "upper bound only" in protocol["rule"]["raw_fifo_upper_bound"]
    assert protocol["timing_protocol"]["scientific_gate"] is False


def _real_state_selections(queue=(2, 2, 5)):
    model = load_model()
    legacy = select_budgeted_frame(
        model.options, queue, model.lanes, sidecar.SMALL_BUDGET
    )
    fifo = select_fifo_execution_quotient_frame(
        model.options, queue, model.lanes, sidecar.SMALL_BUDGET
    )
    return model, legacy, fifo


def _registered_state(queue, legacy, reference):
    fields = (
        "option_indices",
        "service",
        "duration",
        "score",
        "upper_bound",
        "gap",
        "epsilon",
        "attempts",
        "exact",
    )
    selected = asdict(legacy)
    return {
        "queue": list(queue),
        "population_visits": [],
        "legacy_k26": json.loads(
            json.dumps({field: selected[field] for field in fields})
        ),
        "support_reference_score": reference,
    }


def _evaluate_with(monkeypatch, model, index, state, legacy, fifo):
    monkeypatch.setattr(sidecar, "select_budgeted_frame", lambda *args: legacy)
    monkeypatch.setattr(
        sidecar, "select_fifo_execution_quotient_frame", lambda *args: fifo
    )
    ticks = iter((0.0, 0.1, 0.2, 0.4))
    monkeypatch.setattr(sidecar.time, "perf_counter", lambda: next(ticks))
    return sidecar._evaluate_state(model, index, state)


def test_crafted_fifo_action_cannot_escape_the_certificate_sidecar(monkeypatch):
    queue = (2, 2, 5)
    model, legacy, fifo = _real_state_selections(queue)
    alternate = next(
        candidate
        for candidate in (
            (6, 7),
            (7, 6),
            (None, None),
        )
        if candidate != legacy.option_indices
    )
    # The FIFO search result remains realizable, but its raw witness is made
    # visibly different so this test catches accidental action forwarding.
    alt_fifo = select_fifo_execution_quotient_frame(
        model.options, queue, model.lanes, sidecar.SMALL_BUDGET
    )
    if alt_fifo.option_indices == legacy.option_indices:
        service = tuple(
            sum(
                model.options[index].requirements[coordinate]
                for index in alternate
                if index is not None
            )
            for coordinate in range(len(queue))
        )
        duration = max(
            1.0,
            *(model.options[index].duration for index in alternate if index is not None),
        )
        score = sum(q * count for q, count in zip(queue, service)) / duration
        # Realizability is deliberately secondary here: output identity must be
        # independent of the diagnostic witness even if another check rejects it.
        alt_fifo = replace(
            alt_fifo,
            option_indices=alternate,
            service=service,
            duration=duration,
            score=score,
        )
    reference = max(legacy.score, min(legacy.upper_bound, fifo.upper_bound))
    state = _registered_state(queue, legacy, reference)

    evaluated = _evaluate_with(
        monkeypatch, model, 7, state, legacy, alt_fifo
    )

    assert evaluated["fifo_certificate_k26"]["option_indices"] != list(
        legacy.option_indices
    )
    assert evaluated["fixed_action"] == {
        "option_indices": state["legacy_k26"]["option_indices"],
        "service": state["legacy_k26"]["service"],
        "duration": state["legacy_k26"]["duration"],
        "score": state["legacy_k26"]["score"],
    }


@pytest.mark.parametrize("fifo_is_tighter", (False, True))
def test_combined_bound_and_fixed_action_gap_use_the_registered_legacy_score(
    fifo_is_tighter, monkeypatch,
):
    queue = (2, 2, 5)
    model, legacy, fifo = _real_state_selections(queue)
    base = max(legacy.score, fifo.score)
    legacy_upper = base + (3.0 if fifo_is_tighter else 1.0)
    fifo_upper = base + (1.0 if fifo_is_tighter else 3.0)
    legacy = replace(
        legacy,
        upper_bound=legacy_upper,
        gap=legacy_upper - legacy.score,
        epsilon=(legacy_upper - legacy.score) / sum(queue),
        exact=False,
    )
    fifo = replace(fifo, upper_bound=fifo_upper, gap=fifo_upper - fifo.score)
    reference = base
    state = _registered_state(queue, legacy, reference)

    evaluated = _evaluate_with(monkeypatch, model, 3, state, legacy, fifo)

    expected_upper = min(legacy_upper, fifo_upper)
    expected_gap = expected_upper - legacy.score
    assert evaluated["legacy_reconstruction_matches_registered"]
    assert all(evaluated["upper_bound_validity"].values())
    assert evaluated["exact_definition_checks_pass"]
    assert evaluated["work_accounting_checks_pass"]
    assert evaluated["raw_fifo_upper_bound"] == fifo_upper
    assert evaluated["sidecar_upper_bound"] == expected_upper
    assert evaluated["fixed_action_gap"] == pytest.approx(expected_gap)
    assert evaluated["fixed_action_epsilon"] == pytest.approx(
        expected_gap / sum(queue)
    )
    assert evaluated["fixed_action_support_regret"] == pytest.approx(
        reference - legacy.score
    )
    assert evaluated["sidecar_frontier_slack"] == pytest.approx(
        expected_upper - reference
    )
    assert evaluated["fixed_action_gap"] == pytest.approx(
        evaluated["fixed_action_support_regret"]
        + evaluated["sidecar_frontier_slack"]
    )
    assert evaluated["gap_identity_error"] == pytest.approx(0.0)


def test_invalid_fifo_upper_is_rejected_instead_of_hidden_by_gap_clamping(
    monkeypatch,
):
    queue = (2, 2, 5)
    model, legacy, fifo = _real_state_selections(queue)
    reference = max(legacy.score, fifo.score)
    legacy = replace(
        legacy,
        upper_bound=reference + 1.0,
        gap=reference + 1.0 - legacy.score,
        epsilon=(reference + 1.0 - legacy.score) / sum(queue),
        exact=False,
    )
    # This is below both the registered support reference and fixed action.
    fifo = replace(fifo, upper_bound=legacy.score - 1.0)
    state = _registered_state(queue, legacy, reference)

    evaluated = _evaluate_with(monkeypatch, model, 4, state, legacy, fifo)

    assert evaluated["fixed_action_gap"] == -1.0
    assert not evaluated["fixed_action_exact"]
    assert not all(evaluated["upper_bound_validity"].values())
    assert not evaluated["exact_definition_checks_pass"]
    assert reference > evaluated["raw_fifo_upper_bound"]
    assert evaluated["raw_fifo_frontier_slack"] < 0.0


def test_fixed_action_exact_and_fifo_search_exact_have_independent_meanings(
    monkeypatch,
):
    queue = (2, 2, 5)
    model, legacy, fifo = _real_state_selections(queue)

    # An exact FIFO search can expose that the fixed legacy action is suboptimal.
    positive_gap_upper = max(legacy.score, fifo.score) + 1.0
    nonexact_legacy = replace(
        legacy,
        upper_bound=positive_gap_upper,
        gap=positive_gap_upper - legacy.score,
        epsilon=(positive_gap_upper - legacy.score) / sum(queue),
        exact=False,
    )
    fifo_exact = replace(
        fifo,
        upper_bound=positive_gap_upper,
        gap=positive_gap_upper - fifo.score,
        exact=True,
    )
    first = _evaluate_with(
        monkeypatch,
        model,
        0,
        _registered_state(queue, nonexact_legacy, positive_gap_upper),
        nonexact_legacy,
        fifo_exact,
    )
    assert first["fifo_search_exact"]
    assert not first["fixed_action_exact"]

    # Conversely, a fixed action can have a zero combined gap regardless of a
    # diagnostic flag. The inconsistent diagnostic object makes execution fail,
    # while the two output semantics must still remain separate.
    empty_queue = (0, 0, 0)
    model, empty_legacy, empty_fifo = _real_state_selections(empty_queue)
    fifo_not_exact = replace(empty_fifo, exact=False)
    second = _evaluate_with(
        monkeypatch,
        model,
        1,
        _registered_state(empty_queue, empty_legacy, 0.0),
        empty_legacy,
        fifo_not_exact,
    )
    assert not second["fifo_search_exact"]
    assert second["fixed_action_exact"]


def test_timing_alternates_selector_order_and_accounts_each_interval(monkeypatch):
    queue = (2, 2, 5)
    model, legacy_result, fifo_result = _real_state_selections(queue)
    state = _registered_state(
        queue,
        legacy_result,
        max(legacy_result.score, fifo_result.score),
    )
    order = []

    def legacy(_options, observed, _lanes, _budget):
        order.append(("legacy", tuple(observed)))
        return legacy_result

    def fifo(_options, observed, _lanes, _budget):
        order.append(("fifo", tuple(observed)))
        return fifo_result

    monkeypatch.setattr(sidecar, "select_budgeted_frame", legacy)
    monkeypatch.setattr(sidecar, "select_fifo_execution_quotient_frame", fifo)
    ticks = iter((0.0, 0.01, 0.02, 0.05, 1.0, 1.04, 1.1, 1.16))
    monkeypatch.setattr(sidecar.time, "perf_counter", lambda: next(ticks))

    even = sidecar._evaluate_state(model, 0, state)
    odd = sidecar._evaluate_state(model, 1, state)

    assert order == [
        ("legacy", queue),
        ("fifo", queue),
        ("fifo", queue),
        ("legacy", queue),
    ]
    assert even["timing"] == {
        "legacy_selector_wall_s": pytest.approx(0.01),
        "fifo_selector_wall_s": pytest.approx(0.03),
    }
    assert odd["timing"] == {
        "legacy_selector_wall_s": pytest.approx(0.06),
        "fifo_selector_wall_s": pytest.approx(0.04),
    }


def test_evaluation_exposes_full_fifo_work_and_keeps_timing_diagnostic(
    monkeypatch,
):
    queue = (2, 2, 5)
    _model, legacy, fifo = _real_state_selections(queue)
    state = _registered_state(queue, legacy, max(legacy.score, fifo.score))
    evaluated = _evaluate_with(monkeypatch, _model, 2, state, legacy, fifo)

    assert evaluated["work_accounting_checks_pass"]
    assert evaluated["fifo_certificate_k26"] == asdict(fifo)
    for field in (
        "cache_key_evaluations",
        "cache_key_item_reads",
        "cache_key_requirement_coordinate_reads",
        "cache_key_duration_reads",
        "cache_key_completion_offset_reads",
        "logical_search_pops",
        "paid_search_pops",
        "free_cache_hit_pops",
        "cache_entries",
        "cache_hits",
    ):
        assert evaluated["fifo_certificate_k26"][field] == getattr(fifo, field)
    assert set(evaluated["timing"]) == {
        "legacy_selector_wall_s",
        "fifo_selector_wall_s",
    }
    assert all(math.isfinite(value) and value >= 0 for value in evaluated["timing"].values())


def test_operation_work_has_like_for_like_legacy_incremental_and_composite_groups():
    queue = (2, 2, 5)
    _model, legacy, fifo = _real_state_selections(queue)
    evaluations = [
        {"legacy_selection": asdict(legacy), "fifo_selection": asdict(fifo)},
        {"legacy_selection": asdict(legacy), "fifo_selection": asdict(fifo)},
    ]

    work = sidecar._operation_work(evaluations)

    assert set(work) == {
        "legacy_only",
        "fifo_certificate_incremental",
        "composite",
    }
    legacy_work = work["legacy_only"]
    fifo_work = work["fifo_certificate_incremental"]
    composite = work["composite"]
    assert legacy_work["selector_calls"] == fifo_work["selector_calls"] == 2
    assert composite["selector_calls"] == 4
    mapping = {
        "candidate_attempts": "attempts",
        "feasible_candidate_evaluations": "feasible_candidate_count",
        "local_score_evaluations": "local_score_evaluations",
        "bound_option_probes": "bound_option_probes",
        "bound_threshold_evaluations": "bound_threshold_evaluations",
    }
    for output_name, selection_name in mapping.items():
        assert legacy_work[output_name] == 2 * getattr(legacy, selection_name)
        assert fifo_work[output_name] == 2 * getattr(fifo, selection_name)
        assert composite[output_name] == (
            legacy_work[output_name] + fifo_work[output_name]
        )
    assert fifo_work["logical_search_pops"] == (
        fifo_work["paid_search_pops"] + fifo_work["free_cache_hit_pops"]
    )
    assert fifo_work["cache_hits"] == fifo_work["free_cache_hit_pops"]
    assert fifo_work["cache_key_item_reads"] == (
        2 * fifo_work["cache_key_evaluations"]
    )


def _decision_checks(value=True):
    return {
        "execution_passed": value,
        "fixed_action_identity_on_every_state": value,
        "registered_optimum_bounded_by_legacy_fifo_and_sidecar_on_every_state": value,
        "sidecar_upper_bound_pointwise_nonworse": value,
        "sidecar_gap_pointwise_nonworse": value,
        "primary_fixed_support_identical_at_every_load": value,
        "primary_mean_upper_bound_nonworse_at_every_load": value,
        "primary_mean_certified_gap_nonworse_at_every_load": value,
        "primary_exact_certified_fraction_nonworse_at_every_load": value,
        "sensitivity_fixed_support_identical_at_every_load": value,
        "sensitivity_mean_upper_bound_nonworse_at_every_load": value,
        "sensitivity_mean_certified_gap_nonworse_at_every_load": value,
        "sensitivity_exact_certified_fraction_nonworse_at_every_load": value,
        "primary_exposure_contains_certificate_tightening": value,
        "at_least_one_primary_load_strict_certificate_improvement": value,
    }


def test_decision_has_invalid_tradeoff_tightening_and_no_gain_branches():
    passing = _decision_checks()
    assert sidecar._decision(passing) == "CERTIFICATE_TIGHTENING_ONLY_CLOSE"

    invalid = dict(passing, execution_passed=False)
    assert sidecar._decision(invalid) == "INVALID_EXECUTION"

    tradeoff = dict(passing, primary_mean_certified_gap_nonworse_at_every_load=False)
    assert sidecar._decision(tradeoff) == "CERTIFICATE_TRADEOFF_STOP"

    no_gain = dict(
        passing,
        primary_exposure_contains_certificate_tightening=False,
        at_least_one_primary_load_strict_certificate_improvement=False,
    )
    assert sidecar._decision(no_gain) == "NO_CERTIFICATE_GAIN_STOP"

    aggregate_only = dict(
        passing,
        primary_exposure_contains_certificate_tightening=False,
        at_least_one_primary_load_strict_certificate_improvement=True,
    )
    assert sidecar._decision(aggregate_only) == "NO_CERTIFICATE_GAIN_STOP"


@pytest.mark.parametrize("tamper", ("protocol", "source", "roster"))
def test_tampering_fails_before_any_registered_state_execution(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, sources_path = sidecar._paths(output)

    if tamper == "protocol":
        payload = json.loads(protocol_path.read_text())
        payload["small_checkpoint_budget"] += 1
        protocol_path.write_text(json.dumps(payload))
        message = "registered FIFO certificate-sidecar constants changed"
    elif tamper == "source":
        payload = json.loads(sources_path.read_text())
        payload[sidecar.NEW_SOURCES[0]] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "registered FIFO certificate-sidecar sources changed"
    else:
        roster_path = tmp_path / sidecar.REGISTERED_STATES.name
        payload = json.loads(sidecar.REGISTERED_STATES.read_text())
        payload["states"][0]["queue"][0] += 1
        roster_path.write_text(json.dumps(payload))
        monkeypatch.setattr(sidecar, "REGISTERED_STATES", roster_path)
        message = "registered 5,003-state roster changed"

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("tampered registered input reached state execution")

    monkeypatch.setattr(sidecar, "_evaluate_state", forbidden)
    with pytest.raises(ValueError, match=message):
        sidecar.run_registered(output)
    assert calls == []
    assert not output.exists()


def test_write_new_and_registration_refuse_overwrite(registered_bundle, tmp_path):
    path = tmp_path / "sentinel.json"
    path.write_text('{"preserve": true}\n')
    with pytest.raises(FileExistsError):
        sidecar._write_new(path, {"preserve": False})
    assert json.loads(path.read_text()) == {"preserve": True}

    output, _ = registered_bundle
    with pytest.raises(
        ValueError, match="FIFO certificate-sidecar artifacts already exist"
    ):
        sidecar.register(output)


def test_runner_has_no_arrival_drain_trajectory_or_random_dependency():
    source = Path(sidecar.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden_fragments = (
        "arrival",
        "authored_transition_drain",
        "trajectory",
        "random",
    )

    assert not any(
        fragment in imported_name.lower()
        for imported_name in imported
        for fragment in forbidden_fragments
    )
    assert "--run-registered" in source
    assert "development_advance\": False" in source
    for field in (
        "legacy_selector_ms_per_state",
        "fifo_selector_ms_per_state",
        "composite_selector_ms_per_state",
        "composite_over_legacy_ratio",
    ):
        assert field in source
    assert sidecar.DEFAULT_OUTPUT.name.endswith("_development_20260912.json")
