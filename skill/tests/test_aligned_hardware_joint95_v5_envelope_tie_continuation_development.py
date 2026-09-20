from __future__ import annotations

from types import SimpleNamespace

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_tie_continuation_development
    as development,
)


def _work(value: int) -> dict[str, int]:
    return {field: value for field in development.WORK_FIELDS}


def _selection(
    *,
    budget: int,
    work: int,
    score_exact: bool,
    canonical_exact: bool,
    score: float = 1.0,
    upper: float = 1.0,
):
    return SimpleNamespace(
        option_indices=(0,),
        service=(1,),
        score=score,
        upper_bound=upper,
        score_exact=score_exact,
        canonical_exact=canonical_exact,
        lp_node_budget=budget,
        **_work(work),
    )


def _source_fixture():
    protocol = {
        "physical_complete_frame_model": {"options": [{}] * 9},
        "trajectory_contract": {
            "arrival_seeds": list(development.ARRIVAL_SEEDS),
            "case_seed_tapes": 120,
            "trajectory_runs": 240,
        },
    }
    capacity = {str(lanes): [] for lanes in development.LANES}
    eta = {str(lanes): 0.1 for lanes in development.LANES}
    return {"action_mismatch_count": 1}, protocol, capacity, eta


def test_registration_is_zero_work_and_freezes_certificate_only_method(monkeypatch):
    monkeypatch.setattr(
        development,
        "_validated_predecessor_bundle",
        lambda paths: _source_fixture(),
    )
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append((args, kwargs))
        raise AssertionError("registration executed experiment work")

    monkeypatch.setattr(development, "poisson_arrivals", forbidden)
    monkeypatch.setattr(
        development,
        "select_canonical_count_space_bnb_score_exact_tie_continuation_frame",
        forbidden,
    )
    payload = development.build_registration()

    assert forbidden_calls == []
    assert payload["method_contract"]["primary_budget_by_lane"] == {
        "8": 288,
        "16": 576,
    }
    assert payload["method_contract"]["continuation_budget_by_lane"] == {
        "8": 576,
        "16": 1152,
    }
    assert payload["method_contract"]["cold_restart_calls"] == 0
    assert payload["method_contract"]["shadow_information_read_by_candidate"] is False
    assert payload["registration_scope"]["post_result_development"] is True
    assert payload["registration_scope"]["independent_confirmation"] is False
    assert payload["registration_scope"]["selector_calls"] == 0
    assert payload["paper_claim_authorized"] is False


def test_successor_fixed_sources_keep_old_and_current_core_texts():
    historical = development._read_object(
        development.DEFAULT_PREDECESSOR_FIXED_SOURCES
    )
    successor = development._current_fixed_sources()
    old_key = development.HISTORICAL_SOURCE_PREFIX + development.CORE_SOURCE

    assert successor[old_key] == historical[development.CORE_SOURCE]
    assert successor[development.CORE_SOURCE] == (
        development.ROOT / development.CORE_SOURCE
    ).read_text(encoding="utf-8")
    assert successor[old_key] != successor[development.CORE_SOURCE]
    assert (
        "select_canonical_count_space_bnb_score_exact_tie_continuation_frame"
        not in successor[old_key]
    )
    assert (
        "select_canonical_count_space_bnb_score_exact_tie_continuation_frame"
        in successor[development.CORE_SOURCE]
    )


@pytest.mark.parametrize(
    "lanes,initial_budget,continuation_budget",
    ((8, 288, 576), (16, 576, 1152)),
)
def test_candidate_uses_one_live_conditional_search(
    monkeypatch, lanes, initial_budget, continuation_budget
):
    primary = _selection(
        budget=initial_budget,
        work=initial_budget,
        score_exact=True,
        canonical_exact=False,
    )
    selected = _selection(
        budget=continuation_budget,
        work=initial_budget + 1,
        score_exact=True,
        canonical_exact=True,
    )
    calls = []

    def select(options, queue, selected_lanes, primary_budget, maximum_budget):
        calls.append(
            (tuple(queue), selected_lanes, primary_budget, maximum_budget)
        )
        return SimpleNamespace(
            primary_selection=primary,
            selection=selected,
            continuation_triggered=True,
            initial_budget=primary_budget,
            continuation_budget=maximum_budget,
        )

    monkeypatch.setattr(
        development,
        "select_canonical_count_space_bnb_score_exact_tie_continuation_frame",
        select,
    )
    monkeypatch.setattr(development.raw, "_certificate_valid", lambda *a, **k: True)
    model = SimpleNamespace(lanes=lanes, options=(object(),))
    row = development._candidate_call(model, (3,))

    assert calls == [((3,), lanes, initial_budget, continuation_budget)]
    assert row["selection"] is selected
    assert row["primary_selection"] is primary
    assert row["continuation_triggered"] is True
    assert row["method_relation_valid"] is True
    assert row["work"] == _work(initial_budget + 1)
    assert row["valid"] is True


def test_score_inexact_primary_stops_without_continuation(monkeypatch):
    primary = _selection(
        budget=288,
        work=17,
        score_exact=False,
        canonical_exact=False,
        score=0.9,
        upper=1.0,
    )
    observed = []

    def select(*args):
        observed.append(args[2:])
        return SimpleNamespace(
            primary_selection=primary,
            selection=primary,
            continuation_triggered=False,
            initial_budget=288,
            continuation_budget=576,
        )

    monkeypatch.setattr(
        development,
        "select_canonical_count_space_bnb_score_exact_tie_continuation_frame",
        select,
    )
    monkeypatch.setattr(development.raw, "_certificate_valid", lambda *a, **k: True)
    row = development._candidate_call(
        SimpleNamespace(lanes=8, options=(object(),)), (2,)
    )

    assert observed == [(8, 288, 576)]
    assert row["selection"] is primary
    assert row["continuation_triggered"] is False
    assert row["work"] == _work(17)
    assert row["valid"] is True


def test_decision_preserves_every_registered_scientific_stop():
    base = {
        "trajectory_support_gate_passed": True,
        "same_state_action_and_regret_gate_passed": True,
        "paired_mean_flow_noninferiority_passed": True,
    }
    assert development._decision(False, [base], True) == development.INVALID_DECISION
    assert development._decision(
        True, [{**base, "trajectory_support_gate_passed": False}], True
    ) == development.SUPPORT_FAIL_DECISION
    assert development._decision(
        True, [{**base, "same_state_action_and_regret_gate_passed": False}], True
    ) == development.ACTION_FAIL_DECISION
    assert development._decision(True, [base], False) == development.COMPUTE_FAIL_DECISION
    assert development._decision(
        True, [{**base, "paired_mean_flow_noninferiority_passed": False}], True
    ) == development.FLOW_FAIL_DECISION
    assert development._decision(True, [base], True) == development.PASS_DECISION


def test_register_refuses_overwrite(tmp_path):
    registration = tmp_path / "protocol.json"
    registration.write_text("{}", encoding="utf-8")
    with pytest.raises(
        development.TieContinuationDevelopmentRefusal, match="already exists"
    ):
        development.register(
            registration,
            tmp_path / "fixed.json",
            tmp_path / "result.json",
        )
