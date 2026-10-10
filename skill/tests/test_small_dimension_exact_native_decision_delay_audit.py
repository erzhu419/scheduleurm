"""Routed replay budgets, original evidence and separate reference accounting."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.integer_capacity_decision_delay_replay import simulate_delayed_trajectory
from algorithm.experiments import small_dimension_exact_native_decision_delay_audit as audit


PROTOCOL = dict(initial_lp_budget=144, lp_node_budget=256, integer_state_budget=8192,
                reference_budget=4096, oracle_state_budget=5_000_000)


def model():
    return FrameModel("small", "cold_routed_delay", 4, ("a",), (1.,), (
        FrameOption("a1", (1,), 1., ((1.,),)),
        FrameOption("a3", (3,), 2., ((2., 2., 2.),))))


def replay(policy="candidate", tape=None, cutoff=20., delay=.125):
    current = model()
    tape = tuple((0., 0, index) for index in range(20)) if tape is None else tape
    timestamps = iter(value for index in range(100) for value in (
        index * 1_000_000_000, index * 1_000_000_000 + round(delay * 1e9)))
    budget = PROTOCOL["reference_budget"] if policy == "full_reference" else PROTOCOL["lp_node_budget"]
    trajectory, results = simulate_delayed_trajectory(
        current, tape, min(1., cutoff), cutoff, budget, audit.selectors(PROTOCOL)[policy],
        policy=policy, clock=lambda: next(timestamps))
    return current, trajectory, results


@pytest.mark.parametrize("policy", ("candidate", "direct", "full_reference"))
def test_real_prefix_caps_are_audited_without_rewriting_or_extra_native_searches(policy, monkeypatch):
    current, trajectory, results = replay(policy)
    import algorithm.small_dimension_exact_native_frame_candidates as router

    def no_extra_search(*args, **kwargs):
        raise AssertionError("offline audit must not create candidate native ladders")

    monkeypatch.setattr(router, "SmallDimensionExactNativeSearch", no_extra_search)
    reference = audit.ReplayReferenceAudit(current, PROTOCOL)
    audit.audit_trajectory(current, trajectory, results, reference, PROTOCOL)
    metrics = trajectory["selector_metrics"]
    assert metrics["calls"] == len(results) == trajectory["decision_count"]
    assert not metrics["failures"] and not metrics["reference_incomplete_count"]
    assert metrics["canonical_exact_count"] == len(results)
    assert not metrics["score_mismatch_count"] and not metrics["canonical_action_mismatch_count"]
    assert metrics["whole_work"]["calls"] == len(results)
    if policy != "full_reference":
        assert results[0].final_lp.selection.lp_node_budget == 2
        assert trajectory["decision_diagnostics"][0]["actual_lp_budget"] == 2
        assert trajectory["decision_diagnostics"][0]["root_lp_budget"] == 2
        assert metrics["whole_work"]["search_work"]["lp_calls"] == sum(result.final_lp.selection.lp_calls for result in results)
    else:
        assert reference.summary()["calls"] == 0
        assert reference.summary()["observed_policy_calls"] == len(results)
        assert all(value == 0 for value in reference.summary()["work"]["search_work"].values())


def test_same_state_reference_is_shared_but_whole_selectors_are_not_cached():
    current, first, first_results = replay("candidate")
    reference = audit.ReplayReferenceAudit(current, PROTOCOL)
    audit.audit_trajectory(current, first, first_results, reference, PROTOCOL)
    offline_calls = reference.summary()["calls"]
    _, second, second_results = replay("direct")
    audit.audit_trajectory(current, second, second_results, reference, PROTOCOL)
    assert reference.summary()["calls"] == offline_calls
    assert second["reference_audit"]["calls"] == 0
    assert second["reference_audit"]["cache_hits"] == len(second_results)
    assert second["selector_metrics"]["calls"] == len(second_results)
    assert second_results[0] is not first_results[0]


def test_cancelled_terminal_decision_retains_complete_whole_call_cost():
    current, trajectory, results = replay(cutoff=.1, delay=.125)
    assert trajectory["decision_count"] == 1 and trajectory["executed_frame_count"] == 0
    audit.audit_trajectory(current, trajectory, results, audit.ReplayReferenceAudit(current, PROTOCOL), PROTOCOL)
    assert trajectory["selector_metrics"]["calls"] == trajectory["selector_metrics"]["whole_work"]["calls"] == 1
    assert trajectory["selector_metrics"]["operational_elapsed_s"] == .125
    assert not trajectory["selector_metrics"]["failures"]


def test_exact_lp_evidence_corruption_is_a_reported_failure():
    current, trajectory, results = replay(cutoff=.1, delay=.125)
    payload = deepcopy(results[0])
    event = payload.final_lp.small_dimension_lp_evidence[0]
    damaged = replace(event, numerator_fraction=(event.numerator_fraction[0] + 1, event.numerator_fraction[1]))
    final = replace(payload.final_lp, small_dimension_lp_evidence=(damaged, *payload.final_lp.small_dimension_lp_evidence[1:]))
    corrupted = replace(payload, final_lp=final, root_lp=final, initial_lp=final)
    audit.audit_trajectory(current, trajectory, [corrupted], audit.ReplayReferenceAudit(current, PROTOCOL), PROTOCOL)
    assert trajectory["selector_metrics"]["failures"] == 1
    assert any("strong equality" in error for error in trajectory["decision_diagnostics"][0]["errors"])


def test_whole_budget_corruption_is_detected_even_when_actual_final_cap_is_valid():
    current, trajectory, results = replay(cutoff=.1, delay=.125)
    audit.audit_trajectory(current, trajectory, [replace(results[0], integer_state_budget=8193)],
                           audit.ReplayReferenceAudit(current, PROTOCOL), PROTOCOL)
    assert trajectory["selector_metrics"]["failures"] == 1
    assert "whole-controller integer_state_budget differs from registration" in trajectory["decision_diagnostics"][0]["errors"]


@pytest.mark.parametrize("policy", ("candidate", "direct"))
def test_actual_root_probe_retains_shared_integer_and_dual_face_cost(policy, monkeypatch):
    import algorithm.small_dimension_exact_native_kernel as exact
    import algorithm.direct_probe_own_tight_native_kernel as direct

    def tied_lp(*args, **kwargs):
        return SimpleNamespace(success=True, x=(0., 2., 0.),
            ineqlin=SimpleNamespace(marginals=(-3., 0., 0., 0., 0., 0., 0.)))

    monkeypatch.setattr(exact if policy == "candidate" else direct, "linprog", tied_lp)
    current = FrameModel("tie", "root_probe", 2, ("a", "b", "c"), (1., 1., 1.), tuple(
        FrameOption(str(c), tuple(int(c == d) for d in range(3)), 1.,
                    tuple((1.,) if c == d else () for d in range(3))) for c in range(3)))
    tape = tuple((0., c, index) for c, n in enumerate((3, 3, 1)) for index in range(n))
    times = iter((0, 125_000_000))
    trajectory, results = simulate_delayed_trajectory(current, tape, .1, .1, 256,
        audit.selectors(PROTOCOL)[policy], policy=policy, clock=lambda: next(times))
    assert results[0].probe_stage == "root" and results[0].status == "INTEGER_PROBE_COMPLETE"
    audit.audit_trajectory(current, trajectory, results, audit.ReplayReferenceAudit(current, PROTOCOL), PROTOCOL)
    assert not trajectory["selector_metrics"]["failures"]
    work = trajectory["selector_metrics"]["whole_work"]
    assert work["shared_integer_states"] > 0
    assert work["dual_face_work"]["capture_calls"] == 1
    assert work["root_settle_work"]["calls"] == 1
    assert trajectory["decision_diagnostics"][0]["shared_integer_states"] == work["shared_integer_states"]


def test_actual_evidence_copy_fee_cannot_be_omitted():
    current, trajectory, results = replay(cutoff=.1, delay=.125)
    payload = results[0]
    changed = replace(payload.final_lp.small_dimension_lp_work, event_snapshot_entries_copied=0)
    final = replace(payload.final_lp, small_dimension_lp_work=changed)
    corrupted = replace(payload, root_lp=final, initial_lp=final, final_lp=final)
    audit.audit_trajectory(current, trajectory, [corrupted], audit.ReplayReferenceAudit(current, PROTOCOL), PROTOCOL)
    assert trajectory["selector_metrics"]["failures"] == 1
    assert any("copy fee" in error for error in trajectory["decision_diagnostics"][0]["errors"])


@pytest.mark.parametrize("score_exact,expect_oracle", ((True, True), (False, False)))
def test_reference_oracle_uses_only_its_own_score_certificate(score_exact, expect_oracle, monkeypatch):
    current = model()
    reference = audit.ReplayReferenceAudit(current, PROTOCOL)
    import algorithm.canonical_count_space_bnb_frame_candidates as full
    import algorithm.experiments.integer_capacity_canonical_oracle as oracle

    selected = select_canonical_count_space_bnb_frame(current.options, (20,), current.lanes, 4096)
    open_reference = replace(selected, exact=False, canonical_exact=False, score_exact=score_exact)
    monkeypatch.setattr(full, "select_canonical_count_space_bnb_frame", lambda *args: open_reference)
    calls, original = [], oracle.canonical_at_certified_score

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(oracle, "canonical_at_certified_score", counted)
    record = reference.lookup((20,))
    assert bool(calls) == expect_oracle
    assert reference.summary()["oracle_calls"] == int(expect_oracle)
    assert reference.summary()["incomplete_count"] == int(not expect_oracle)
    assert not record["errors"]
    if calls:
        assert calls[0]["state_visit_budget"] == 5_000_000
        assert record["expected"] is not None


def test_incomplete_reference_is_preserved_without_rescuing_whole_policy(monkeypatch):
    current, trajectory, results = replay(cutoff=.1, delay=.125)
    import algorithm.canonical_count_space_bnb_frame_candidates as full

    selected = select_canonical_count_space_bnb_frame(current.options, (20,), current.lanes, 4096)
    unproven = replace(selected, exact=False, canonical_exact=False, score_exact=False)
    monkeypatch.setattr(full, "select_canonical_count_space_bnb_frame", lambda *args: unproven)
    original = results[0]
    reference = audit.ReplayReferenceAudit(current, PROTOCOL)
    audit.audit_trajectory(current, trajectory, results, reference, PROTOCOL)
    assert results[0] is original and original.selection.canonical_exact
    assert trajectory["selector_metrics"]["reference_incomplete_count"] == 1
    assert trajectory["reference_audit"]["incomplete_count"] == 1
    assert reference.summary()["oracle_calls"] == 0


def test_wrong_replay_budget_is_rejected_before_policy_call():
    with pytest.raises(ValueError, match="whole-controller budget"):
        audit.selectors(PROTOCOL)["candidate"](model().options, (20,), 4, 144)
