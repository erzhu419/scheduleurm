from __future__ import annotations

import copy
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_requirement_dominance_tie_continuation_confirmation
    as quotient_confirmation,
)


def _work(value: int) -> dict[str, int]:
    return {field: value for field in quotient_confirmation.WORK_FIELDS}


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
        for lanes in quotient_confirmation.LANES
        for load in quotient_confirmation.LOADS
        for seed in quotient_confirmation.ARRIVAL_SEEDS
    ]


def test_actual_development_pass_protocol_result_and_65_sources_are_exact():
    body = quotient_confirmation.build_registration()
    contract = body["development_pass_contract"]

    assert contract["status"] == "PASS"
    assert contract["scientific_decision"] == (
        quotient_confirmation.development.PASS_DECISION
    )
    assert contract["arrival_seeds"] == list(range(12061, 12101))
    assert contract["authorized_confirmation_seeds"] == list(range(12101, 12121))
    assert contract["arrival_generations"] == 240
    assert contract["trajectory_runs"] == 480
    assert contract["tie_continuation_trigger_count"] == 0
    assert contract["tie_continuation_final_exact_count"] == 0
    assert contract["all_six_case_gates_passed"] is True
    assert contract["both_lane_compute_gates_passed"] is True
    assert contract["aggregate_compute_gate_passed"] is True
    assert contract["fixed_source_count"] == 65
    assert contract["historical_source_count"] == 30
    assert len(contract["current_normal_source_keys"]) == 35
    assert contract["current_normal_source_drift"] == []
    assert contract["independent_confirmation_authorized"] is True


def test_fresh_seed_roster_is_fixed_and_disjoint_from_all_173_prior_seeds():
    assert quotient_confirmation.ARRIVAL_SEEDS == tuple(range(12101, 12121))
    assert quotient_confirmation.development.NEXT_CONFIRMATION_SEEDS == (
        quotient_confirmation.ARRIVAL_SEEDS
    )
    assert len(quotient_confirmation.PREVIOUSLY_USED_ARRIVAL_SEEDS) == 173
    assert not (
        set(quotient_confirmation.ARRIVAL_SEEDS)
        & set(quotient_confirmation.PREVIOUSLY_USED_ARRIVAL_SEEDS)
    )


def test_registration_prepares_once_and_executes_no_rng_selector_or_trajectory(
    monkeypatch,
):
    original = (
        quotient_confirmation.prepare_canonical_count_space_bnb_requirement_dominance_catalogue
    )
    preparation_calls = []

    def counted(options, idle_duration=1.0):
        preparation_calls.append(tuple(options))
        return original(options, idle_duration)

    def forbidden(*args, **kwargs):
        raise AssertionError("registration executed experiment work")

    monkeypatch.setattr(
        quotient_confirmation,
        "prepare_canonical_count_space_bnb_requirement_dominance_catalogue",
        counted,
    )
    monkeypatch.setattr(quotient_confirmation, "_execute_tasks", forbidden)
    monkeypatch.setattr(quotient_confirmation.development, "_run_tape", forbidden)
    monkeypatch.setattr(
        quotient_confirmation.development.development,
        "poisson_arrivals",
        forbidden,
    )

    body = quotient_confirmation.build_registration()

    assert len(preparation_calls) == 1
    assert body["trajectory_contract"]["case_seed_tapes"] == 120
    assert body["trajectory_contract"]["trajectory_runs"] == 240
    assert body["trajectory_contract"]["candidate_method"] == (
        quotient_confirmation.development.METHOD_CONTRACT
    )
    assert body["registration_scope"][
        "catalogue_reduction_and_lift_preparations"
    ] == 1
    assert body["registration_scope"]["rng_initializations"] == 0
    assert body["registration_scope"]["selector_calls"] == 0
    assert body["registration_scope"]["trajectory_runs"] == 0
    assert body["registration_scope"]["test_source_in_fixed_bundle"] is False
    assert quotient_confirmation.TEST_SOURCE not in (
        body["development_pass_contract"]["fixed_source_keys"]
    )
    assert body["paper_claim_authorized"] is False


def test_registered_quotient_and_compute_claim_are_exactly_bounded():
    body = quotient_confirmation.build_registration()
    quotient = body["quotient_contract"]

    assert quotient == body["development_pass_contract"]["quotient_contract"]
    assert quotient["candidate_bnb_option_count"] == 8
    assert quotient["shadow_and_reference_option_count"] == 9
    assert quotient["reduction"]["removed_original_indices"] == [1]
    assert quotient["reduction"]["retained_original_indices"] == [
        0,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    assert quotient["reduction_work_charged_separately_from_bnb"] is True
    assert body["work_gate_fields"] == [
        "lp_calls",
        "lp_matrix_coefficients",
        "candidate_evaluations",
    ]
    assert "wall-time or total-runtime improvement" in body["claim_boundary"]


def test_fixed_sources_preserve_65_and_add_only_confirmation_runner():
    predecessor = quotient_confirmation._read_object(
        quotient_confirmation.DEFAULT_DEVELOPMENT_FIXED_SOURCES
    )
    frozen = quotient_confirmation._current_fixed_sources()

    assert len(predecessor) == 65
    assert len(frozen) == 66
    assert all(frozen[key] == value for key, value in predecessor.items())
    assert set(frozen) - set(predecessor) == {quotient_confirmation.THIS_SOURCE}
    assert frozen[quotient_confirmation.THIS_SOURCE] == (
        quotient_confirmation.ROOT / quotient_confirmation.THIS_SOURCE
    ).read_text(encoding="utf-8")


def test_register_writes_protocol_and_66_sources_but_no_result(tmp_path):
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    result = tmp_path / "result.json"

    protocol = quotient_confirmation.register(registration, fixed, result)

    assert protocol["status"] == quotient_confirmation.REGISTRATION_STATUS
    assert registration.is_file() and fixed.is_file()
    assert not result.exists()
    assert len(json.loads(fixed.read_text(encoding="utf-8"))) == 66


def test_process_pool_maps_exact_development_worker_and_prepared_object(monkeypatch):
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

    monkeypatch.setattr(quotient_confirmation, "ProcessPoolExecutor", FakePool)
    prepared = object()
    tasks = [(object(), {}, prepared, 8, 0.5, 12101, 600.0)]

    assert quotient_confirmation._execute_tasks(tasks) == [{"ok": True}]
    assert observed["workers"] == quotient_confirmation.EXECUTION_WORKERS
    assert observed["function"] is quotient_confirmation.development._run_tape
    assert observed["tasks"][0][2] is prepared


def test_case_summary_requires_20_seed_roster_and_keeps_full_gates():
    rows = [
        _run(8, 0.5, seed) for seed in quotient_confirmation.ARRIVAL_SEEDS
    ]

    summary = quotient_confirmation._case_summary(rows, 8, 0.5, eta=0.1)

    assert summary["seeds"] == list(quotient_confirmation.ARRIVAL_SEEDS)
    assert summary["seed_count"] == 20
    assert summary["candidate_bnb_option_count"] == 8
    assert summary["shadow_and_reference_option_count"] == 9
    assert summary["case_gate_passed"] is True
    assert summary["case_compute_gate_passed"] is True
    assert summary["same_state_action_and_regret_gate_passed"] is True
    assert summary["paired_mean_flow_noninferiority_passed"] is True
    assert summary["requirement_dominance_quotient_selector_call_count"] == 40
    assert summary["full_catalogue_certificate_lift_valid_count"] == 40

    rows[0]["seed"] = 12100
    with pytest.raises(
        quotient_confirmation.confirmation.TieContinuationConfirmationRefusal,
        match="registered seed roster",
    ):
        quotient_confirmation._case_summary(rows, 8, 0.5, eta=0.1)


def test_decision_preserves_every_registered_scientific_stop():
    passing = {
        "trajectory_support_gate_passed": True,
        "same_state_action_and_regret_gate_passed": True,
        "paired_mean_flow_noninferiority_passed": True,
    }
    assert quotient_confirmation._decision(False, [passing], True) == (
        quotient_confirmation.INVALID_DECISION
    )
    assert quotient_confirmation._decision(
        True, [{**passing, "trajectory_support_gate_passed": False}], True
    ) == quotient_confirmation.SUPPORT_FAIL_DECISION
    assert quotient_confirmation._decision(
        True,
        [{**passing, "same_state_action_and_regret_gate_passed": False}],
        True,
    ) == quotient_confirmation.ACTION_FAIL_DECISION
    assert quotient_confirmation._decision(True, [passing], False) == (
        quotient_confirmation.COMPUTE_FAIL_DECISION
    )
    assert quotient_confirmation._decision(
        True,
        [{**passing, "paired_mean_flow_noninferiority_passed": False}],
        True,
    ) == quotient_confirmation.FLOW_FAIL_DECISION
    assert quotient_confirmation._decision(True, [passing], True) == (
        quotient_confirmation.PASS_DECISION
    )


def test_mocked_zero_trigger_pass_is_nonvacuous_about_continuation(
    tmp_path, monkeypatch
):
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "result.json"
    quotient_confirmation.register(registration, fixed, output)
    rows = _all_runs()
    original_prepare = (
        quotient_confirmation.prepare_canonical_count_space_bnb_requirement_dominance_catalogue
    )
    preparation_calls = []

    def counted(options, idle_duration=1.0):
        preparation_calls.append(tuple(options))
        return original_prepare(options, idle_duration)

    monkeypatch.setattr(
        quotient_confirmation,
        "prepare_canonical_count_space_bnb_requirement_dominance_catalogue",
        counted,
    )
    monkeypatch.setattr(quotient_confirmation, "_execute_tasks", lambda tasks: rows)

    report = quotient_confirmation.run_registered(registration, fixed, output)

    assert len(preparation_calls) == 1
    assert report["status"] == "PASS"
    assert report["scientific_decision"] == quotient_confirmation.PASS_DECISION
    assert report["tie_continuation_trigger_count"] == 0
    assert report["tie_continuation_final_exact_count"] == 0
    assert report["continuation_branch_observed"] is False
    assert report["continuation_branch_empirically_confirmed"] is False
    assert report["continuation_branch_claim_authorized"] is False
    assert report["independent_confirmation_passed"] is True
    assert report["trajectory_transition_passed"] is True
    assert report["trajectory_or_flow_claim_authorized"] is True
    assert report[
        "fixed_protocol_action_identity_and_flow_noninferiority_authorized"
    ] is True
    assert report["registered_bnb_counter_reduction_authorized"] is True
    assert report["flow_improvement_claim_authorized"] is False
    assert report["wall_time_or_total_runtime_claim_authorized"] is False
    assert report["paper_claim_authorized"] is False
    assert report["arrival_generations"] == 120
    assert report["trajectory_runs"] == 240
    assert report["shadow_trajectory_runs"] == 0
    assert all(report["checks"].values())
    assert all(row["case_gate_passed"] for row in report["case_summaries"])
    assert report["work_gates"]["gate_passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_observed_and_resolved_trigger_marks_only_branch_evidence(
    tmp_path, monkeypatch
):
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "result.json"
    quotient_confirmation.register(registration, fixed, output)
    rows = _all_runs()
    metrics = rows[0]["policies"][0]["metrics"]
    metrics["tie_continuation_trigger_count"] = 1
    metrics["tie_continuation_final_exact_count"] = 1
    metrics["primary_score_exact_noncanonical_count"] = 1
    monkeypatch.setattr(quotient_confirmation, "_execute_tasks", lambda tasks: rows)

    report = quotient_confirmation.run_registered(registration, fixed, output)

    assert report["status"] == "PASS"
    assert report["continuation_branch_observed"] is True
    assert report["continuation_branch_empirically_confirmed"] is True
    assert report["continuation_branch_claim_authorized"] is True
    assert report["flow_improvement_claim_authorized"] is False


def test_protocol_tamper_is_rejected_before_task_execution(tmp_path, monkeypatch):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(quotient_confirmation.build_registration()),
    }
    protocol["trajectory_contract"]["arrival_seeds"][0] = 12100
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "result.json"
    registration.write_text(json.dumps(protocol), encoding="utf-8")
    fixed.write_text(
        json.dumps(quotient_confirmation._current_fixed_sources()),
        encoding="utf-8",
    )
    calls = []

    def forbidden(tasks):
        calls.append(tasks)
        raise AssertionError("tampered protocol reached execution")

    monkeypatch.setattr(quotient_confirmation, "_execute_tasks", forbidden)
    with pytest.raises(
        quotient_confirmation.RequirementDominanceConfirmationRefusal,
        match="registered protocol changed",
    ):
        quotient_confirmation.run_registered(registration, fixed, output)
    assert calls == []


def test_development_result_tamper_is_rejected(tmp_path):
    result = quotient_confirmation._read_object(
        quotient_confirmation.DEFAULT_DEVELOPMENT_RESULT
    )
    result["paper_claim_authorized"] = True
    tampered = tmp_path / "development.json"
    tampered.write_text(json.dumps(result), encoding="utf-8")
    paths = quotient_confirmation._source_paths(development_result=tampered)
    source_protocol = quotient_confirmation._read_object(
        quotient_confirmation.DEFAULT_DEVELOPMENT_REGISTRATION
    )
    _, quotient = quotient_confirmation._prepare_registered_quotient(
        source_protocol
    )

    with pytest.raises(
        quotient_confirmation.RequirementDominanceConfirmationRefusal,
        match="exact audited dominance-quotient PASS",
    ):
        quotient_confirmation._validated_development_bundle(paths, quotient)


def test_register_refuses_overwrite(tmp_path):
    registration = tmp_path / "protocol.json"
    registration.write_text("{}", encoding="utf-8")
    with pytest.raises(
        quotient_confirmation.RequirementDominanceConfirmationRefusal,
        match="already exists",
    ):
        quotient_confirmation.register(
            registration,
            tmp_path / "fixed.json",
            tmp_path / "result.json",
        )
