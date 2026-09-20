from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_cumulative_continuation_development
    as continuation,
)


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _work(value: int) -> dict[str, int]:
    return {field: value for field in continuation.WORK_FIELDS}


def _variant(
    *,
    work: int,
    shadow_work: int,
    action_match: bool = True,
    certificate_valid: bool = True,
    within_eta: bool = True,
    regret_bounded: bool = True,
    score_exact: bool = True,
    canonical_exact: bool = True,
):
    return {
        "configured_cap": 1,
        "option_indices": [0],
        "score": 1.0,
        "upper_bound": 1.0,
        "epsilon": 0.0,
        "score_exact": score_exact,
        "canonical_exact": canonical_exact,
        "certificate_valid": certificate_valid,
        "same_state_shadow_action_match": action_match,
        "actual_normalized_regret": 0.0,
        "within_lane_min_eta": within_eta,
        "regret_bounded_by_epsilon": regret_bounded,
        "fixed_cap_native_work": _work(work),
        "shadow_work": _work(shadow_work),
    }


@pytest.fixture(scope="module")
def registration():
    return continuation.build_registration()


def test_registration_freezes_post_scout_raw_fixed_cap_scope(registration):
    assert registration["status"] == continuation.REGISTERED_STATUS
    assert registration["registration_scope"] == {
        "post_result_development": True,
        "exploratory_scout_preceded_registration": True,
        "post_scout_development": True,
        "independent_blind_registration": False,
        "registration_selector_calls": 0,
        "registration_rng_initializations": 0,
        "registration_arrival_generations": 0,
        "registration_trajectory_runs": 0,
        "registration_server_pulls": 0,
    }
    contract = registration["selector_contract"]
    assert contract["checkpoints_by_lane"] == {
        "8": [48, 96, 144, 288],
        "16": [9, 96, 144, 288, 576],
    }
    assert contract["decision_policy_has_runtime_trigger"] is False
    assert "no exchange" in contract["decision_work_rule"]
    assert "bulk polish" in contract["decision_work_rule"]
    assert "uniformly applied to all six cases" in contract["case_compute_gate"]
    assert registration["independent_confirmation_eligible"] is False
    assert registration["paper_claim_authorized"] is False


def test_registration_compacts_all_frozen_states_and_occurrences(registration):
    roster = registration["state_roster"]
    assert len(roster) == 4445
    assert sum(row["total_occurrences"] for row in roster) == 5411
    assert len({(row["lanes"], tuple(row["queue"])) for row in roster}) == 4445
    assert all(
        row["shadow"]["valid"]
        and row["shadow"]["score_exact"]
        and row["shadow"]["canonical_exact"]
        for row in roster
    )


def test_registration_calls_no_selector_rng_arrival_or_trajectory(
    tmp_path, monkeypatch
):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("registration executed experiment work")

    monkeypatch.setattr(
        continuation, "trace_canonical_count_space_bnb_frame", forbidden
    )
    monkeypatch.setattr(
        continuation, "select_canonical_count_space_bnb_frame", forbidden
    )
    monkeypatch.setattr(continuation.source.source, "poisson_arrivals", forbidden)
    registration_path = tmp_path / "protocol.json"
    fixed_path = tmp_path / "fixed.json"
    result_path = tmp_path / "result.json"

    payload = continuation.register(registration_path, fixed_path, result_path)

    assert calls == []
    assert registration_path.is_file() and fixed_path.is_file()
    assert not result_path.exists()
    assert payload["registration_scope"]["registration_selector_calls"] == 0


def test_changed_source_result_is_refused(tmp_path):
    payload = continuation._read_object(continuation.DEFAULT_SOURCE_RESULT)
    payload["status"] = "CHANGED"
    changed = _write(tmp_path / "changed.json", payload)

    with pytest.raises(
        continuation.CumulativeContinuationRefusal,
        match="completed negative rescue-budget development",
    ):
        continuation.build_registration(source_result=changed)


def test_work_rows_are_complete_nonnegative_integers():
    assert continuation._checked_work(_work(2)) == _work(2)
    missing = _work(2)
    missing.pop(next(iter(missing)))
    with pytest.raises(continuation.CumulativeContinuationRefusal):
        continuation._checked_work(missing)
    invalid = _work(2)
    invalid[next(iter(invalid))] = -1
    with pytest.raises(continuation.CumulativeContinuationRefusal):
        continuation._checked_work(invalid)


def test_case_compute_gate_allows_parity_but_lane_requires_strict_saving():
    state = {
        "lanes": 16,
        "total_occurrences": 3,
        "occurrences_by_case": {0.5: 3},
        "raw_variants": {576: _variant(work=4, shadow_work=4)},
    }
    case = continuation._scope_summary(
        [state],
        {16: 576},
        lambda row: row["occurrences_by_case"].get(0.5, 0),
        scope={"scope": "case", "lanes": 16, "load": 0.5},
    )
    lane = continuation._scope_summary(
        [state],
        {16: 576},
        lambda row: row["total_occurrences"],
        scope={"scope": "lane", "lanes": 16},
    )

    assert case["work_gate_fields_no_greater"] is True
    assert case["registered_scope_compute_gate_passed"] is True
    assert case["all_registered_gates_passed"] is True
    assert lane["work_gate_fields_strictly_lower"] is False
    assert lane["registered_scope_compute_gate_passed"] is False
    assert lane["all_registered_gates_passed"] is False


def test_scope_uses_occurrence_weighted_work_and_reports_state_relations():
    equal = {
        "lanes": 8,
        "total_occurrences": 2,
        "occurrences_by_case": {0.8: 2},
        "raw_variants": {288: _variant(work=3, shadow_work=3)},
    }
    saving = {
        "lanes": 8,
        "total_occurrences": 5,
        "occurrences_by_case": {0.8: 5},
        "raw_variants": {288: _variant(work=1, shadow_work=2)},
    }
    summary = continuation._scope_summary(
        [equal, saving],
        {8: 288},
        lambda row: row["total_occurrences"],
        scope={"scope": "lane", "lanes": 8},
    )

    assert summary["state_occurrence_count"] == 7
    assert summary["candidate_occurrence_weighted_work"] == _work(11)
    assert summary["same_state_k4096_occurrence_weighted_work"] == _work(16)
    relations = summary["per_state_gate_work_relation_counts"]
    assert relations["all_three_no_greater_unique_state_count"] == 2
    assert relations["all_three_no_greater_occurrence_count"] == 7
    assert relations["all_three_equal_unique_state_count"] == 1
    assert relations["all_three_equal_occurrence_count"] == 2


def test_exactness_is_reported_but_does_not_control_statewise_gate():
    state = {
        "lanes": 8,
        "total_occurrences": 4,
        "occurrences_by_case": {0.95: 4},
        "raw_variants": {
            288: _variant(
                work=1,
                shadow_work=2,
                score_exact=False,
                canonical_exact=False,
            )
        },
    }
    summary = continuation._scope_summary(
        [state],
        {8: 288},
        lambda row: row["total_occurrences"],
        scope={"scope": "lane", "lanes": 8},
    )

    assert summary["score_inexact_occurrence_count"] == 4
    assert summary["canonical_inexact_occurrence_count"] == 4
    assert summary["exactness_is_diagnostic_not_a_registered_gate"] is True
    assert summary["all_registered_gates_passed"] is True


def test_action_support_regret_and_certificate_each_remain_hard_gates():
    keyword_by_field = {
        "action_match": "all_occurrences_match_shadow_action",
        "certificate_valid": "all_occurrences_certificate_valid",
        "within_eta": "all_occurrences_within_lane_min_eta",
        "regret_bounded": "all_occurrences_regret_bounded",
    }
    for field, gate in keyword_by_field.items():
        kwargs = {field: False}
        state = {
            "lanes": 8,
            "total_occurrences": 1,
            "occurrences_by_case": {0.5: 1},
            "raw_variants": {288: _variant(work=1, shadow_work=2, **kwargs)},
        }
        summary = continuation._scope_summary(
            [state],
            {8: 288},
            lambda row: row["total_occurrences"],
            scope={"scope": "lane", "lanes": 8},
        )
        assert summary[gate] is False
        assert summary["all_registered_gates_passed"] is False


def test_smallest_passing_cap_is_selected():
    cases = []
    lanes = []
    for lane, checkpoints, passing_cap in ((8, (48, 96, 144, 288), 288), (16, (9, 96, 144, 288, 576), 576)):
        for cap in checkpoints:
            passed = cap >= passing_cap
            lanes.append(
                {
                    "lanes": lane,
                    "configured_cap": cap,
                    "all_registered_gates_passed": passed,
                }
            )
            for load in continuation.LOADS:
                cases.append(
                    {
                        "lanes": lane,
                        "configured_cap": cap,
                        "load": load,
                        "all_registered_gates_passed": passed,
                    }
                )
    assert continuation._select_lane_caps({"case": cases, "by_lane": lanes}) == {
        8: 288,
        16: 576,
    }


def test_trace_first_checkpoint_matches_standalone_native_selection(registration):
    model = continuation.source.source.baseline._model_from_snapshot(
        registration["physical_complete_frame_model"]
    )
    queue = (2, 5, 11)
    checkpoints = continuation.CHECKPOINTS[8]
    trace = continuation.trace_canonical_count_space_bnb_frame(
        model.options, queue, 8, checkpoints
    )
    standalone = continuation.select_canonical_count_space_bnb_frame(
        model.options, queue, 8, checkpoints[0]
    )

    assert trace[0] == standalone
    for earlier, later in zip(trace[:-1], trace[1:], strict=True):
        assert later.score + continuation.SCORE_TOLERANCE >= earlier.score
        assert later.upper_bound <= earlier.upper_bound + continuation.SCORE_TOLERANCE
        assert all(
            getattr(later, field) >= getattr(earlier, field)
            for field in continuation.WORK_FIELDS
        )


def test_register_refuses_overwrite(tmp_path):
    registration = tmp_path / "protocol.json"
    registration.write_text("{}", encoding="utf-8")
    with pytest.raises(
        continuation.CumulativeContinuationRefusal,
        match="already exists",
    ):
        continuation.register(
            registration,
            tmp_path / "fixed.json",
            tmp_path / "result.json",
        )
