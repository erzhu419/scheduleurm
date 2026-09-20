from __future__ import annotations

from dataclasses import replace
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_requirement_dominance_tie_continuation_development
    as quotient_dev,
)


def _work(value: int) -> dict[str, int]:
    return {field: value for field in quotient_dev.WORK_FIELDS}


def _policy(policy: str, *, mean_flow: float = 1.0) -> dict:
    candidate = policy == "candidate"
    frames = 2
    metrics = {
        "policy": policy,
        "job_count": 2,
        "completed_jobs": 2,
        "mean_flow_s": mean_flow,
        "frame_count": frames,
        "valid_selection_frame_count": frames,
        "exact_selection_frame_count": 0 if candidate else frames,
        "inexact_selection_frame_count": frames if candidate else 0,
        "maximum_certified_gap_per_job": 0.0,
        "maximum_actual_normalized_regret": 0.0 if candidate else None,
        "maximum_queue_recursion_violation": 0.0,
        "work": _work(1 if candidate else 2),
        "same_state_shadow_calls": frames if candidate else 0,
        "same_state_shadow_valid_count": frames if candidate else None,
        "same_state_shadow_exact_count": frames if candidate else None,
        "same_state_action_match_count": frames if candidate else None,
        "regret_bounded_by_epsilon_count": frames if candidate else None,
        "same_state_shadow_work": _work(2) if candidate else None,
        "conditional_selector_call_count": frames if candidate else 0,
        "conditional_selector_relation_valid_count": frames if candidate else 0,
        "tie_continuation_trigger_count": 0,
        "tie_continuation_final_exact_count": 0,
        "primary_score_exact_noncanonical_count": 0,
        "tie_continuation_audit_prefix": [],
        "requirement_dominance_quotient_selector_call_count": (
            frames if candidate else 0
        ),
        "full_catalogue_certificate_lift_valid_count": frames if candidate else 0,
    }
    return {
        "policy": policy,
        "metrics": metrics,
        "flow_decomposition": {"status": "PASS", "mean_flow_s": mean_flow},
    }


def _run(lanes: int, load: float, seed: int) -> dict:
    return {
        "lanes": lanes,
        "load": load,
        "seed": seed,
        "arrival_tape": {
            "job_count": 2,
            "arrival_generation_count": 1,
            "executed_policy_count": 2,
            "shadow_trajectory_runs": 0,
        },
        "policies": [_policy("candidate"), _policy("cold_k4096_reference")],
    }


def _all_runs() -> list[dict]:
    return [
        _run(lanes, load, seed)
        for lanes in quotient_dev.LANES
        for load in quotient_dev.LOADS
        for seed in quotient_dev.ARRIVAL_SEEDS
    ]


def _model_and_prepared():
    protocol = quotient_dev.build_registration()
    model = quotient_dev.raw.old_transition.baseline._model_from_snapshot(
        protocol["physical_complete_frame_model"]
    )
    prepared = (
        quotient_dev.prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
            model.options
        )
    )
    return model, prepared


def test_actual_confirmation_is_the_exact_compute_only_fail():
    contract, protocol, capacity, eta = quotient_dev._validated_confirmation_bundle(
        quotient_dev._source_paths()
    )

    assert contract["status"] == "FAIL"
    assert contract["scientific_decision"] == (
        quotient_dev.confirmation.COMPUTE_FAIL_DECISION
    )
    assert contract["all_six_case_gates_passed"] is True
    assert contract["aggregate_compute_gate_passed"] is True
    assert contract["r8_lane_compute_gate_passed"] is True
    assert contract["r16_lane_compute_gate_passed"] is False
    assert contract["r16_gate_fields_equal_to_shadow"] == list(
        quotient_dev.WORK_GATE_FIELDS
    )
    assert contract["fixed_source_count"] == 62
    assert contract["current_normal_source_drift"] == []
    assert len(protocol["physical_complete_frame_model"]["options"]) == 9
    assert set(capacity) == set(eta) == {"8", "16"}


def test_development_roster_reuses_both_observed_batches_and_freezes_next_batch():
    assert quotient_dev.ARRIVAL_SEEDS == tuple(range(12061, 12101))
    assert quotient_dev.NEXT_CONFIRMATION_SEEDS == tuple(range(12101, 12121))
    assert quotient_dev.ARRIVAL_SEEDS[:20] == quotient_dev.raw.ARRIVAL_SEEDS
    assert quotient_dev.ARRIVAL_SEEDS[20:] == quotient_dev.confirmation.ARRIVAL_SEEDS


def test_registration_prepares_reducer_once_and_executes_no_selector_or_trajectory(
    monkeypatch,
):
    original = (
        quotient_dev.prepare_canonical_count_space_bnb_requirement_dominance_catalogue
    )
    calls = []

    def counted(options, idle_duration=1.0):
        calls.append(tuple(options))
        return original(options, idle_duration)

    def forbidden(*args, **kwargs):
        raise AssertionError("registration executed experimental work")

    monkeypatch.setattr(
        quotient_dev,
        "prepare_canonical_count_space_bnb_requirement_dominance_catalogue",
        counted,
    )
    monkeypatch.setattr(quotient_dev, "_execute_tasks", forbidden)
    monkeypatch.setattr(quotient_dev, "_run_tape", forbidden)
    monkeypatch.setattr(quotient_dev.development, "poisson_arrivals", forbidden)
    monkeypatch.setattr(
        quotient_dev,
        "select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame",
        forbidden,
    )

    payload = quotient_dev.build_registration()

    assert len(calls) == 1
    assert payload["case_seed_tapes"] == 240
    assert payload["trajectory_runs"] == 480
    assert payload["registration_scope"]["catalogue_reduction_and_lift_preparations"] == 1
    assert payload["registration_scope"]["selector_calls"] == 0
    assert payload["registration_scope"]["arrival_generations"] == 0
    assert payload["registration_scope"]["trajectory_runs"] == 0


def test_registered_quotient_is_exactly_nine_to_eight_with_separate_work():
    quotient = quotient_dev.build_registration()["quotient_contract"]
    reduction = quotient["reduction"]
    proof = quotient["full_catalogue_certificate_lift_proof"]

    assert reduction["original_option_count"] == 9
    assert reduction["retained_option_count"] == 8
    assert reduction["removed_original_indices"] == [1]
    assert reduction["retained_original_indices"] == [0, 2, 3, 4, 5, 6, 7, 8]
    assert reduction["reduction_option_probes"] == 9
    assert reduction["reduction_requirement_coordinate_reads"] == 27
    assert reduction["reduction_duration_comparisons"] == 10
    assert proof["removed_to_dominator_original_indices"] == [[1, 0]]
    assert proof["replacement_requirements_equal"] is True
    assert proof["replacement_effective_duration_strictly_smaller"] is True
    assert proof["replacement_original_index_strictly_smaller"] is True
    assert quotient["reduction_work_charged_separately_from_bnb"] is True


def test_fixed_sources_preserve_predecessor_62_and_add_exactly_three_sources():
    predecessor = quotient_dev._read_object(
        quotient_dev.DEFAULT_CONFIRMATION_FIXED_SOURCES
    )
    frozen = quotient_dev._current_fixed_sources()

    assert len(predecessor) == 62
    assert len(frozen) == 65
    assert all(frozen[key] == value for key, value in predecessor.items())
    assert set(frozen) - set(predecessor) == {
        quotient_dev.REDUCER_SOURCE,
        quotient_dev.QUOTIENT_SOURCE,
        quotient_dev.THIS_SOURCE,
    }


def test_real_empty_queue_uses_eight_option_work_and_lifts_to_nine_counts():
    model, prepared = _model_and_prepared()
    lane_model = replace(model, lanes=8)

    call = quotient_dev._candidate_call(prepared, lane_model, (0, 0, 0))
    selected = call["selection"]

    assert call["valid"] is True
    assert call["full_catalogue_lift_valid"] is True
    assert len(selected.option_counts) == 9
    assert selected.option_counts[1] == 0
    assert selected.duration_root_count == 8
    assert selected.lp_calls == 8
    assert selected.lp_matrix_coefficients == 348
    assert selected.candidate_evaluations == 9
    assert selected.represented_ordered_configurations == 9**8
    assert selected.represented_count_vectors == 12870


def test_custom_work_validator_rejects_nine_option_accounting():
    model, prepared = _model_and_prepared()
    lane_model = replace(model, lanes=8)
    selection = quotient_dev._candidate_call(
        prepared, lane_model, (0, 0, 0)
    )["selection"]

    assert quotient_dev._full_catalogue_selection_valid(
        lane_model, (0, 0, 0), 288, selection
    )
    wrong = replace(
        selection,
        duration_root_count=9,
        represented_ordered_configurations=10**8,
        represented_count_vectors=24310,
    )
    assert not quotient_dev._full_catalogue_selection_valid(
        lane_model, (0, 0, 0), 288, wrong
    )


def test_quotient_contract_rejects_a_false_canonical_lift():
    _, prepared = _model_and_prepared()
    bad_proof = replace(
        prepared.certificate_lift_proof,
        replacement_original_index_strictly_smaller=False,
    )
    bad = replace(prepared, certificate_lift_proof=bad_proof)

    with pytest.raises(
        quotient_dev.RequirementDominanceDevelopmentRefusal,
        match="dominance quotient changed",
    ):
        quotient_dev._quotient_contract(bad)


def test_worker_temporarily_replaces_development_candidate_and_restores_it(
    monkeypatch,
):
    original = quotient_dev.development._candidate_call
    seen = {}

    def fake_run(task):
        seen["patched"] = quotient_dev.development._candidate_call is not original
        row = _run(task[2], task[3], task[4])
        candidate = row["policies"][0]["metrics"]
        candidate.pop("requirement_dominance_quotient_selector_call_count")
        candidate.pop("full_catalogue_certificate_lift_valid_count")
        return row

    monkeypatch.setattr(quotient_dev.development, "_run_tape", fake_run)
    task = (object(), {}, object(), 8, 0.5, 12061, 600.0)
    row = quotient_dev._run_tape(task)

    assert seen["patched"] is True
    assert quotient_dev.development._candidate_call is original
    metrics = row["policies"][0]["metrics"]
    assert metrics["requirement_dominance_quotient_selector_call_count"] == 0
    assert metrics["full_catalogue_certificate_lift_valid_count"] == 0


def test_worker_restores_development_candidate_after_exception(monkeypatch):
    original = quotient_dev.development._candidate_call

    def fail(task):
        assert quotient_dev.development._candidate_call is not original
        raise RuntimeError("boom")

    monkeypatch.setattr(quotient_dev.development, "_run_tape", fail)
    task = (object(), {}, object(), 8, 0.5, 12061, 600.0)
    with pytest.raises(RuntimeError, match="boom"):
        quotient_dev._run_tape(task)
    assert quotient_dev.development._candidate_call is original


def test_process_pool_receives_prepared_quotient_in_every_task(monkeypatch):
    observed = {}

    class FakePool:
        def __init__(self, *, max_workers):
            observed["workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, tasks):
            observed["function"] = function
            observed["tasks"] = list(tasks)
            return iter([{"ok": True}] * len(observed["tasks"]))

    monkeypatch.setattr(quotient_dev, "ProcessPoolExecutor", FakePool)
    prepared = object()
    tasks = [(object(), {}, prepared, 8, 0.5, 12061, 600.0)]
    rows = quotient_dev._execute_tasks(tasks)

    assert observed["workers"] == quotient_dev.EXECUTION_WORKERS
    assert observed["function"] is quotient_dev._run_tape
    assert observed["tasks"][0][2] is prepared
    assert rows == [{"ok": True}]


def test_case_summary_requires_complete_40_seed_roster_and_keeps_all_gates():
    rows = [_run(8, 0.5, seed) for seed in quotient_dev.ARRIVAL_SEEDS]

    summary = quotient_dev._case_summary(rows, 8, 0.5, eta=0.1)

    assert summary["seeds"] == list(quotient_dev.ARRIVAL_SEEDS)
    assert summary["seed_count"] == 40
    assert summary["candidate_bnb_option_count"] == 8
    assert summary["shadow_and_reference_option_count"] == 9
    assert summary["case_gate_passed"] is True
    assert summary["case_compute_gate_passed"] is True
    assert summary["same_state_action_and_regret_gate_passed"] is True
    assert summary["paired_mean_flow_noninferiority_passed"] is True

    rows[0]["seed"] = 12101
    with pytest.raises(
        quotient_dev.confirmation.TieContinuationConfirmationRefusal,
        match="registered seed roster",
    ):
        quotient_dev._case_summary(rows, 8, 0.5, eta=0.1)


def test_decision_preserves_every_scientific_stop():
    passing = {
        "trajectory_support_gate_passed": True,
        "same_state_action_and_regret_gate_passed": True,
        "paired_mean_flow_noninferiority_passed": True,
    }
    assert quotient_dev._decision(False, [passing], True) == quotient_dev.INVALID_DECISION
    assert quotient_dev._decision(
        True, [{**passing, "trajectory_support_gate_passed": False}], True
    ) == quotient_dev.SUPPORT_FAIL_DECISION
    assert quotient_dev._decision(
        True, [{**passing, "same_state_action_and_regret_gate_passed": False}], True
    ) == quotient_dev.ACTION_FAIL_DECISION
    assert quotient_dev._decision(True, [passing], False) == quotient_dev.COMPUTE_FAIL_DECISION
    assert quotient_dev._decision(
        True, [{**passing, "paired_mean_flow_noninferiority_passed": False}], True
    ) == quotient_dev.FLOW_FAIL_DECISION
    assert quotient_dev._decision(True, [passing], True) == quotient_dev.PASS_DECISION


def test_mocked_full_development_has_240_tapes_480_trajectories_and_only_authorizes_next_seed_batch(
    tmp_path, monkeypatch
):
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "result.json"
    quotient_dev.register(registration, fixed, output)
    rows = _all_runs()
    original_prepare = (
        quotient_dev.prepare_canonical_count_space_bnb_requirement_dominance_catalogue
    )
    preparation_calls = []

    def counted_prepare(options, idle_duration=1.0):
        preparation_calls.append(tuple(options))
        return original_prepare(options, idle_duration)

    monkeypatch.setattr(
        quotient_dev,
        "prepare_canonical_count_space_bnb_requirement_dominance_catalogue",
        counted_prepare,
    )
    monkeypatch.setattr(quotient_dev, "_execute_tasks", lambda tasks: rows)

    report = quotient_dev.run_registered(registration, fixed, output)

    assert len(preparation_calls) == 1
    assert report["status"] == "PASS"
    assert report["scientific_decision"] == quotient_dev.PASS_DECISION
    assert report["arrival_generations"] == 240
    assert report["trajectory_runs"] == 480
    assert report["work_gates"]["gate_passed"] is True
    assert report["new_seed_independent_confirmation_authorized"] is True
    assert report["authorized_confirmation_seeds"] == list(range(12101, 12121))
    assert report["independent_confirmation_eligible"] is False
    assert report["trajectory_or_flow_claim_authorized"] is False
    assert report["paper_claim_authorized"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == report
