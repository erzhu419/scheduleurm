from __future__ import annotations

import copy
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_tie_continuation_confirmation
    as confirmation,
)


def _work(value: int) -> dict[str, int]:
    return {field: value for field in confirmation.WORK_FIELDS}


def _policy(policy: str, *, mean_flow: float = 1.0, work: int = 1):
    frame_count = 2
    candidate = policy == "candidate"
    metrics = {
        "policy": policy,
        "job_count": 2,
        "completed_jobs": 2,
        "mean_flow_s": mean_flow,
        "frame_count": frame_count,
        "valid_selection_frame_count": frame_count,
        "exact_selection_frame_count": 0 if candidate else frame_count,
        "inexact_selection_frame_count": frame_count if candidate else 0,
        "maximum_certified_gap_per_job": 0.0,
        "maximum_actual_normalized_regret": 0.0 if candidate else None,
        "maximum_queue_recursion_violation": 0.0,
        "work": _work(work),
        "same_state_shadow_calls": frame_count if candidate else 0,
        "same_state_shadow_valid_count": frame_count if candidate else None,
        "same_state_shadow_exact_count": frame_count if candidate else None,
        "same_state_action_match_count": frame_count if candidate else None,
        "regret_bounded_by_epsilon_count": frame_count if candidate else None,
        "same_state_shadow_work": _work(work + 1) if candidate else None,
        "conditional_selector_call_count": frame_count if candidate else 0,
        "conditional_selector_relation_valid_count": frame_count if candidate else 0,
        "tie_continuation_trigger_count": 0,
        "tie_continuation_final_exact_count": 0,
        "primary_score_exact_noncanonical_count": 0,
        "tie_continuation_audit_prefix": [],
    }
    return {
        "policy": policy,
        "metrics": metrics,
        "flow_decomposition": {"status": "PASS", "mean_flow_s": mean_flow},
    }


def _run(
    lanes: int, load: float, seed: int, *, shadow_work_offset: int = 1
) -> dict:
    candidate = _policy("candidate", work=1)
    candidate["metrics"]["same_state_shadow_work"] = _work(
        1 + shadow_work_offset
    )
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
        "policies": [
            candidate,
            _policy("cold_k4096_reference", work=2),
        ],
    }


def _all_runs() -> list[dict]:
    return [
        _run(lanes, load, seed)
        for lanes in confirmation.LANES
        for load in confirmation.LOADS
        for seed in confirmation.ARRIVAL_SEEDS
    ]


def _source_fixture():
    capacity = {str(lanes): [] for lanes in confirmation.LANES}
    eta = {str(lanes): 0.1 for lanes in confirmation.LANES}
    protocol = {
        "physical_complete_frame_model": {"options": [{}] * 9},
        "physical_envelope_capacity_normalized_loads": capacity,
        "lane_minimum_effective_slack_eta": eta,
        "arrival_seeds": list(confirmation.development.ARRIVAL_SEEDS),
        "arrival_horizon_s": confirmation.ARRIVAL_HORIZON_S,
        "method_contract": confirmation.development.METHOD_CONTRACT,
    }
    return {"current_normal_source_drift": []}, protocol, capacity, eta


def test_fresh_seed_roster_is_fixed_and_disjoint_from_all_153_prior_seeds():
    assert confirmation.ARRIVAL_SEEDS == tuple(range(12081, 12101))
    assert len(confirmation.raw.PREVIOUSLY_USED_ARRIVAL_SEEDS) == 133
    assert confirmation.raw.ARRIVAL_SEEDS == tuple(range(12061, 12081))
    assert len(confirmation.PREVIOUSLY_USED_ARRIVAL_SEEDS) == 153
    assert not (
        set(confirmation.ARRIVAL_SEEDS)
        & set(confirmation.PREVIOUSLY_USED_ARRIVAL_SEEDS)
    )


def test_actual_development_pass_and_all_current_normal_sources_revalidate():
    paths = confirmation._source_paths()
    contract, protocol, capacity, eta = confirmation._validated_development_bundle(
        paths
    )

    assert contract["status"] == "PASS"
    assert contract["scientific_decision"] == confirmation.development.PASS_DECISION
    assert contract["tie_continuation_trigger_count"] == 2
    assert contract["tie_continuation_final_exact_count"] == 2
    assert contract["fixed_source_count"] == 61
    assert len(contract["current_normal_source_keys"]) == 31
    assert contract["current_normal_source_drift"] == []
    assert contract["independent_confirmation_authorized"] is True
    assert protocol["method_contract"] == confirmation.development.METHOD_CONTRACT
    assert set(capacity) == set(eta) == {"8", "16"}


def test_registration_is_zero_execution_and_freezes_independent_method(monkeypatch):
    monkeypatch.setattr(
        confirmation, "_validated_development_bundle", lambda paths: _source_fixture()
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("registration executed experiment work")

    monkeypatch.setattr(confirmation, "_execute_tasks", forbidden)
    monkeypatch.setattr(confirmation.development, "_run_tape", forbidden)
    monkeypatch.setattr(confirmation.development, "poisson_arrivals", forbidden)
    payload = confirmation.build_registration()

    assert calls == []
    assert payload["independent_confirmation"] is True
    assert payload["trajectory_contract"]["arrival_seeds"] == list(
        range(12081, 12101)
    )
    assert payload["trajectory_contract"]["case_seed_tapes"] == 120
    assert payload["trajectory_contract"]["trajectory_runs"] == 240
    assert payload["trajectory_contract"]["candidate_method"] == (
        confirmation.development.METHOD_CONTRACT
    )
    assert payload["registration_scope"]["rng_initializations"] == 0
    assert payload["registration_scope"]["arrival_generations"] == 0
    assert payload["registration_scope"]["selector_calls"] == 0
    assert payload["registration_scope"]["trajectory_runs"] == 0
    assert payload["paper_claim_authorized"] is False


def test_register_writes_only_protocol_and_62_sources_without_execution(
    tmp_path, monkeypatch
):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("registration executed experiment work")

    monkeypatch.setattr(confirmation, "_execute_tasks", forbidden)
    monkeypatch.setattr(confirmation.development, "_run_tape", forbidden)
    monkeypatch.setattr(confirmation.development, "poisson_arrivals", forbidden)
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    result = tmp_path / "result.json"

    protocol = confirmation.register(registration, fixed, result)

    assert calls == []
    assert protocol["status"] == confirmation.REGISTRATION_STATUS
    assert registration.is_file() and fixed.is_file()
    assert not result.exists()
    assert len(json.loads(fixed.read_text(encoding="utf-8"))) == 62


def test_fixed_sources_preserve_predecessor_61_and_add_only_confirmation_runner():
    predecessor = confirmation._read_object(
        confirmation.DEFAULT_DEVELOPMENT_FIXED_SOURCES
    )
    frozen = confirmation._current_fixed_sources()

    assert len(predecessor) == 61
    assert len(frozen) == 62
    assert all(frozen[key] == value for key, value in predecessor.items())
    assert set(frozen) - set(predecessor) == {confirmation.THIS_SOURCE}
    assert frozen[confirmation.THIS_SOURCE] == (
        confirmation.ROOT / confirmation.THIS_SOURCE
    ).read_text(encoding="utf-8")


def test_parameterized_summary_matches_raw_field_for_field_on_old_roster():
    rows = [
        _run(8, 0.5, seed) for seed in confirmation.raw.ARRIVAL_SEEDS
    ]

    expected = confirmation.raw._paired_case_summary(rows, 8, 0.5, eta=0.1)
    observed = confirmation._paired_case_summary(
        rows, 8, 0.5, eta=0.1, expected_seeds=confirmation.raw.ARRIVAL_SEEDS
    )

    assert observed == expected


def test_case_summary_accepts_new_roster_without_mutating_raw_seed_global():
    old = confirmation.raw.ARRIVAL_SEEDS
    rows = [_run(8, 0.5, seed) for seed in confirmation.ARRIVAL_SEEDS]

    summary = confirmation._case_summary(rows, 8, 0.5, eta=0.1)

    assert confirmation.raw.ARRIVAL_SEEDS is old
    assert summary["seeds"] == list(confirmation.ARRIVAL_SEEDS)
    assert summary["seed_count"] == 20
    assert summary["candidate_method"] == (
        "score_exact_tie_conditional_continuation"
    )
    assert summary["primary_budget"] == 288
    assert summary["continuation_budget"] == 576
    assert summary["case_gate_passed"] is True


@pytest.mark.parametrize("replacement", (12080, 12082, 12101))
def test_case_summary_rejects_missing_duplicate_old_or_extra_seed(replacement):
    rows = [_run(8, 0.5, seed) for seed in confirmation.ARRIVAL_SEEDS]
    rows[0]["seed"] = replacement
    old = confirmation.raw.ARRIVAL_SEEDS

    with pytest.raises(
        confirmation.TieContinuationConfirmationRefusal,
        match="independent registered seed roster",
    ):
        confirmation._case_summary(rows, 8, 0.5, eta=0.1)
    assert confirmation.raw.ARRIVAL_SEEDS is old


def test_task_executor_maps_the_frozen_development_runner(monkeypatch):
    observed = {}

    class FakePool:
        def __init__(self, *, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, tasks):
            observed["function"] = function
            observed["tasks"] = list(tasks)
            return iter([{"ok": True} for _ in observed["tasks"]])

    monkeypatch.setattr(confirmation, "ProcessPoolExecutor", FakePool)
    tasks = [(object(), {}, 8, 0.5, 12081, 600.0)]
    rows = confirmation._execute_tasks(tasks)

    assert observed["max_workers"] == confirmation.EXECUTION_WORKERS
    assert observed["function"] is confirmation.development._run_tape
    assert observed["tasks"] == tasks
    assert rows == [{"ok": True}]


def test_decision_preserves_every_registered_scientific_stop():
    passing = {
        "trajectory_support_gate_passed": True,
        "same_state_action_and_regret_gate_passed": True,
        "paired_mean_flow_noninferiority_passed": True,
    }
    assert confirmation._decision(False, [passing], True) == (
        confirmation.INVALID_DECISION
    )
    assert confirmation._decision(
        True, [{**passing, "trajectory_support_gate_passed": False}], True
    ) == confirmation.SUPPORT_FAIL_DECISION
    assert confirmation._decision(
        True,
        [{**passing, "same_state_action_and_regret_gate_passed": False}],
        True,
    ) == confirmation.ACTION_FAIL_DECISION
    assert confirmation._decision(True, [passing], False) == (
        confirmation.COMPUTE_FAIL_DECISION
    )
    assert confirmation._decision(
        True,
        [{**passing, "paired_mean_flow_noninferiority_passed": False}],
        True,
    ) == confirmation.FLOW_FAIL_DECISION
    assert confirmation._decision(True, [passing], True) == confirmation.PASS_DECISION


def test_new_seed_work_gates_allow_case_parity_but_keep_lane_and_aggregate_strict():
    runs = [
        _run(
            lanes,
            load,
            seed,
            shadow_work_offset=0 if (lanes, load) == (8, 0.5) else 1,
        )
        for lanes in confirmation.LANES
        for load in confirmation.LOADS
        for seed in confirmation.ARRIVAL_SEEDS
    ]
    cases = [
        confirmation._case_summary(runs, lanes, load, eta=0.1)
        for lanes in confirmation.LANES
        for load in confirmation.LOADS
    ]
    gates = confirmation.raw._aggregate_work_gates(runs, cases)

    parity = next(
        row for row in cases if (row["lanes"], row["load"]) == (8, 0.5)
    )
    assert parity["case_compute_gate_passed"] is True
    assert parity["charged_work_comparison"]["gate_fields_strictly_lower"] is False
    assert all(row["gate_passed"] for row in gates["by_lane"])
    assert gates["aggregate"]["gate_passed"] is True
    assert gates["gate_passed"] is True


def test_mocked_zero_trigger_pass_marks_bounded_confirmation_but_not_paper(
    tmp_path, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(confirmation.build_registration()),
    }
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "result.json"
    registration.write_text(json.dumps(protocol), encoding="utf-8")
    fixed.write_text(
        json.dumps(confirmation._current_fixed_sources()), encoding="utf-8"
    )
    runs = _all_runs()
    monkeypatch.setattr(confirmation, "_execute_tasks", lambda tasks: runs)

    report = confirmation.run_registered(registration, fixed, output)

    assert report["status"] == "PASS"
    assert report["scientific_decision"] == confirmation.PASS_DECISION
    assert report["tie_continuation_trigger_count"] == 0
    assert report["independent_confirmation_passed"] is True
    assert report["trajectory_transition_passed"] is True
    assert report["trajectory_or_flow_claim_authorized"] is True
    assert report["fixed_protocol_claim_only"] is True
    assert report["paper_claim_authorized"] is False
    assert report["trajectory_runs"] == 240
    assert report["arrival_generations"] == 120
    assert report["shadow_trajectory_runs"] == 0
    assert all(report["checks"].values())
    assert all(row["case_gate_passed"] for row in report["case_summaries"])
    assert report["work_gates"]["gate_passed"] is True


def test_protocol_tamper_is_rejected_before_task_execution(tmp_path, monkeypatch):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(confirmation.build_registration()),
    }
    protocol["trajectory_contract"]["arrival_seeds"][0] = 12080
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "result.json"
    registration.write_text(json.dumps(protocol), encoding="utf-8")
    fixed.write_text(
        json.dumps(confirmation._current_fixed_sources()), encoding="utf-8"
    )
    calls = []

    def forbidden(tasks):
        calls.append(tasks)
        raise AssertionError("tampered protocol reached execution")

    monkeypatch.setattr(confirmation, "_execute_tasks", forbidden)
    with pytest.raises(
        confirmation.TieContinuationConfirmationRefusal,
        match="registered protocol changed",
    ):
        confirmation.run_registered(registration, fixed, output)
    assert calls == []


def test_register_refuses_overwrite(tmp_path):
    registration = tmp_path / "protocol.json"
    registration.write_text("{}", encoding="utf-8")
    with pytest.raises(
        confirmation.TieContinuationConfirmationRefusal,
        match="already exists",
    ):
        confirmation.register(
            registration,
            tmp_path / "fixed.json",
            tmp_path / "result.json",
        )
