"""A fixed-root audit keeps independent draws separate from root selection."""
from copy import deepcopy
from fractions import Fraction
from random import Random

from algorithm.arrival_aware_lane_rollout import (
    ArrivalAwareLaneRollout, PlanningArrivals, sample_planning_arrivals,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments import arrival_aware_root_ranking as ranking


def _case():
    model = FrameModel("toy", "toy", 2, ("a", "b", "c"), (1., 1., 1.), (
        FrameOption("a", (1, 0, 0), 5., ((5.,), (), ())),
        FrameOption("b", (0, 1, 0), 1., ((), (1.,), ())),
        FrameOption("c", (0, 0, 1), 2., ((), (), (2.,))),
        FrameOption("a_batch", (2, 0, 0), 3., ((3., 3.), (), ())),
    ))
    state = {"tape_id": "toy:R2:load0.8:seed41", "node": "toy", "lanes": 2,
             "load": .8, "seed": 41,
             "first_change": {"frame": 0, "start": 0., "queue": [2, 1, 0],
                              "available_before": [0., 10.], "lane_ids": [0],
                              "maxweight_action_id": "a_batch", "visible_queue_action_id": "b"}}
    protocol = {"experiment": "toy", "registered_at_utc": "frozen", "arrival_horizon": 12.,
                "capacities": [{"node": "toy", "load": .8, "arrival_rates": [.05, .02, .01]}],
                "states": [state], "selection_samples": 8, "audit_samples": 32, "event_cap": 128,
                "selection_seed_template": "arrival-plan-v1:{node}:{load}:rank-selection",
                "audit_seed_template": "arrival-audit-v1:{node}:{load}:rank:{seed}", "limitations": "toy"}
    return model, state, protocol


def test_state_extraction_omits_real_futures_and_preserves_absent_state():
    _, state, _ = _case()
    first = {key: value for key, value in state["first_change"].items()
             if key not in ("maxweight_action_id", "visible_queue_action_id")}
    first.update({"old_root": {"action_id": "a_batch"}, "new_root": {"action_id": "b"},
                  "real_future_arrivals": [[.5, 0, 2]]})
    row = {**state, "first_change": first, "real_future_arrivals": [[1., 0, 3]]}
    absent = {**row, "tape_id": "toy:absent", "first_change": None}
    result = ranking.extract_states({"paired_runs": [row, absent]})
    assert result == [state, {**state, "tape_id": "toy:absent", "first_change": None}]


def test_every_audit_path_is_shared_by_fixed_roots_and_matches_core_costs():
    model, state, protocol = _case()
    row = ranking.run_state(model, state, protocol)
    assert row["status"] == "INDEPENDENT_ROOT_AUDIT_COMPLETE"
    rng = Random(protocol["audit_seed_template"].format(**state))
    paths = [sample_planning_arrivals((.05, .02, .01), 5., rng, 128) for _ in range(32)]
    core = ArrivalAwareLaneRollout(model.options, (.05, .02, .01), 12., "separate-unused-selection")
    for index, path in enumerate(paths):
        expected = core.score_roots((2, 1, 0), (0., 10.), 0., 0, (path,))
        for label, root in row["roots"].items():
            assert row["path_cost_job_s"][label][index] == float(expected.costs[root["index"]])
    assert row["audit_sampling"]["events"] == sum(len(path.events) for path in paths)
    assert row["identity_option_probes"] == len(model.options)
    assert row["audit_work"]["root_option_probes"] == len(model.options) * 32
    assert row["audit_work"]["rollout_option_probes"] == row["audit_work"]["rollout_calls"] * len(model.options)
    for label in ("arrival_aware", "visible_queue_clearing"):
        deltas = row["paired_vs_maxweight"][label]["path_cost_delta_job_s"]
        assert len(deltas) == 32
        assert abs(sum(deltas) / 32 - row["paired_vs_maxweight"][label]["mean_cost_delta_job_s"]) < 1e-12


def test_audit_seed_change_does_not_change_selected_root_or_planning_cost():
    model, state, protocol = _case()
    old = ranking.run_state(model, state, protocol)
    changed = deepcopy(protocol)
    changed["audit_seed_template"] = "independent-second-audit:{node}:{load}:rank:{seed}"
    new = ranking.run_state(model, state, changed)
    assert new["roots"] == old["roots"]
    assert new["selection"] == old["selection"]
    assert new["audit_sampling"] != old["audit_sampling"]


def test_audit_overflow_is_retained_without_truncating_or_resampling(monkeypatch):
    model, state, protocol = _case()
    calls = []

    def overflow(rates, window, rng, event_cap):
        calls.append((rates, window, event_cap))
        return PlanningArrivals(((.1, 0, 0), (.2, 0, 1)), True, 2, 1)

    monkeypatch.setattr(ranking, "sample_planning_arrivals", overflow)
    row = ranking.run_state(model, state, protocol)
    assert row["status"] == "AUDIT_EVENT_BUDGET_OVERFLOW"
    assert row["audit_sampling"]["overflow_samples"] == 32
    assert len(calls) == 32
    assert "mean_cost_job_s" not in row and "audit_work" not in row
    assert "selection" in row


def test_paired_comparison_uses_exact_difference_before_float_conversion():
    values = [Fraction(10**20) + Fraction(1, 3), Fraction(10**20) - Fraction(1, 3)]
    reference = [Fraction(10**20), Fraction(10**20)]
    result = ranking.paired_costs(values, reference)
    assert result["path_cost_delta_job_s"] == [float(Fraction(1, 3)), float(Fraction(-1, 3))]
    assert result["mean_cost_delta_job_s"] == 0.
    assert result["lower_cost_paths"] == result["higher_cost_paths"] == 1


def test_summary_retains_all_states_and_has_no_performance_promotion():
    model, state, protocol = _case()
    good = ranking.run_state(model, state, protocol)
    absent = {**state, "tape_id": "toy:absent", "first_change": None}
    overflow = {**state, "tape_id": "toy:overflow", "seed": 42}
    protocol["states"] = [state, absent, overflow]
    rows = [good, ranking.run_state(model, absent, protocol),
            {**overflow, "status": "AUDIT_EVENT_BUDGET_OVERFLOW"}]
    result = ranking.summarize(rows, protocol)
    assert result["decision"] == "INDEPENDENT_ROOT_RANKING_DIAGNOSIS_COMPLETE"
    assert result["registered_states"] == 3 and result["audited_states"] == 1
    assert result["absent_states"] == result["audit_overflow_states"] == 1
    assert result["paired_states"] == rows
    assert result["seed_groups"][1]["audited_states"] == 0
    assert result["seed_groups"][1]["paired_vs_maxweight"] is None
    assert "performance_gate_passed" not in result
