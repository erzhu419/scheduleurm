from __future__ import annotations

import json

import pytest

from algorithm.count_space_bnb_frame_candidates import (
    trace_count_space_bnb_frame,
)
from algorithm.experiments import (
    aligned_hardware_count_space_bnb_scaling_screen as screen,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


def test_fixed_grid_and_r2_oracle_match_all_checkpoint_certificates():
    base = screen.base_queues()
    assert len(base) == len(set(base)) == screen.BASE_STATE_COUNT
    assert base == tuple(sorted(base))
    assert screen.scaled_queues(16)[-1] == (320, 200, 192)

    model = load_model()
    queue = (5, 3, 5)
    optimum, evaluations = screen._r2_oracle(model, queue)
    trace = trace_count_space_bnb_frame(
        model.options,
        queue,
        2,
        screen.LP_CHECKPOINT_BUDGETS,
    )
    assert evaluations == screen.R2_ORDERED_ACTION_COUNT == 100
    assert len(trace) == len(screen.LP_CHECKPOINT_BUDGETS)
    for budget, selection in zip(
        screen.LP_CHECKPOINT_BUDGETS, trace, strict=True
    ):
        assert selection.lp_node_budget == budget
        assert selection.score <= optimum + screen.SCORE_TOLERANCE
        assert optimum <= selection.upper_bound + screen.SCORE_TOLERANCE
        assert screen._selection_reconstructs(model, queue, 2, selection)
        assert screen._selection_work_holds(queue, 2, budget, selection)


def test_registration_performs_no_grid_search_lp_or_capacity_work(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "screen.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("registration performed execution work")

    monkeypatch.setattr(screen, "base_queues", forbidden)
    monkeypatch.setattr(screen, "scaled_queues", forbidden)
    monkeypatch.setattr(screen, "trace_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(screen, "capacity_certificate", forbidden)
    monkeypatch.setattr(screen, "load_model", forbidden)

    protocol = screen.register(output)
    _, protocol_path, fixed_sources_path = screen._paths(output)
    assert not output.exists()
    assert protocol_path.exists()
    assert fixed_sources_path.exists()
    assert protocol["lp_checkpoint_budgets"] == [9, 18, 36, 72]
    assert protocol["checkpoint_execution"].startswith("one cumulative B&B trace")
    sources = json.loads(fixed_sources_path.read_text())
    assert screen.SELECTOR_SOURCE in sources
    assert screen.THIS_SOURCE in sources


def test_run_rejects_changed_protocol_before_loading_model(tmp_path, monkeypatch):
    output = tmp_path / "screen.json"
    screen.register(output)
    _, protocol_path, _ = screen._paths(output)
    protocol = json.loads(protocol_path.read_text())
    protocol["lp_checkpoint_budgets"] = [9, 18, 36, 73]
    protocol_path.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("model loaded before protocol validation")

    monkeypatch.setattr(screen, "load_model", forbidden)
    with pytest.raises(ValueError, match="protocol changed"):
        screen.run_registered(output)


def test_decision_requires_same_checkpoint_tradeoff_per_scaled_lane():
    rows = [
        {
            "lanes": lanes,
            "first_quality_budget": 18,
            "first_work_budget": 9,
            "first_tradeoff_budget": 18,
        }
        for lanes in screen.LANE_COUNTS
    ]
    assert screen._decision(True, rows) == "COUNT_SPACE_BNB_SCALING_ADVANCE"
    rows[-1]["first_tradeoff_budget"] = None
    rows[-1]["first_quality_budget"] = None
    assert screen._decision(True, rows) == "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP"
    assert screen._decision(False, rows) == "INVALID_EXECUTION"
