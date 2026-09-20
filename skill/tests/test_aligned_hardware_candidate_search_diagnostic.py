"""Focused tests for the registered legacy-v4 search diagnostic."""
from __future__ import annotations

from collections import Counter
import json

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
import algorithm.experiments.aligned_hardware_candidate_search_diagnostic as diagnostic
from algorithm.experiments.aligned_hardware_frame_model import load_model


def test_queue_recorder_preserves_selection_and_counts_visits_and_duration():
    model = load_model()
    queue = (4, 3, 5)
    expected = select_budgeted_frame(
        model.options, queue, model.lanes, diagnostic.SMALL_BUDGET,
    )
    recorder = diagnostic.QueueRecorder(model, diagnostic.SMALL_BUDGET)

    first = recorder(
        model.options, queue, model.lanes, diagnostic.SMALL_BUDGET,
    )
    second = recorder(
        model.options, queue, model.lanes, diagnostic.SMALL_BUDGET,
    )

    assert first == second == expected
    assert recorder.visits == Counter({queue: 2})
    assert recorder.duration_by_queue[queue] == pytest.approx(2 * expected.duration)


def test_population_summary_weights_gap_classes_by_policy_visits():
    strict_queue = (1, 0, 0)
    bound_open_queue = (0, 1, 0)
    certified_queue = (0, 0, 0)

    def state(
        queue,
        gap_class,
        *,
        first_score_attempt,
        first_physical_attempt,
        regret,
        slack,
        certified,
        same_physical,
    ):
        return {
            "queue": queue,
            "queue_norm": sum(queue),
            "gap_class": gap_class,
            "first_exact_score_attempt": first_score_attempt,
            "first_k120_selected_physical_signature_attempt": first_physical_attempt,
            "exact_score_found_by_k26": first_score_attempt <= diagnostic.SMALL_BUDGET,
            "finite_budget_support_regret": regret,
            "bound_relaxation_slack": slack,
            "k26_gap": regret + slack,
            "k26_exact_certified": certified,
            "k26_exact_physical_action": same_physical,
            "k26": {"option_indices": (0,)},
            "exact": {"option_indices": (1,)},
        }

    traces = {
        strict_queue: state(
            strict_queue,
            "strict_support_loss",
            first_score_attempt=28,
            first_physical_attempt=29,
            regret=4.0,
            slack=2.0,
            certified=False,
            same_physical=False,
        ),
        bound_open_queue: state(
            bound_open_queue,
            "exact_score_found_bound_open",
            first_score_attempt=10,
            first_physical_attempt=31,
            regret=0.0,
            slack=3.0,
            certified=False,
            same_physical=False,
        ),
        certified_queue: state(
            certified_queue,
            "exact_certified_by_k26",
            first_score_attempt=1,
            first_physical_attempt=1,
            regret=0.0,
            slack=0.0,
            certified=True,
            same_physical=True,
        ),
    }
    visits = Counter({strict_queue: 2, bound_open_queue: 3, certified_queue: 5})
    durations = Counter({strict_queue: 8.0, bound_open_queue: 6.0, certified_queue: 5.0})

    summary = diagnostic._population_summary(
        0.8, diagnostic.SMALL_BUDGET, visits, durations, traces,
    )

    assert summary["frame_count"] == 10
    assert summary["visited_time_s"] == 19.0
    assert summary["strict_support_loss_frame_count"] == 2
    assert summary["strict_support_loss_frame_fraction"] == pytest.approx(0.2)
    assert summary["strict_support_loss_time_fraction"] == pytest.approx(8 / 19)
    assert summary["exact_score_found_by_k26_frame_count"] == 8
    assert summary["exact_score_found_by_k26_frame_fraction"] == pytest.approx(0.8)
    assert summary["exact_score_found_bound_open_frame_count"] == 3
    assert summary["exact_score_found_bound_open_frame_fraction"] == pytest.approx(0.3)
    assert summary["k26_exact_certified_frame_fraction"] == pytest.approx(0.5)
    assert summary["exact_physical_action_frame_fraction"] == pytest.approx(0.5)
    assert summary["exact_score_first_found_after_k26_frame_fraction"] == pytest.approx(0.2)
    assert summary["k120_selected_physical_signature_first_found_after_k26_frame_fraction"] == pytest.approx(0.5)
    assert summary["mean_finite_budget_support_regret"] == pytest.approx(0.8)
    assert summary["mean_finite_budget_support_regret_per_visible_job"] == pytest.approx(0.8)
    assert summary["mean_bound_relaxation_slack"] == pytest.approx(1.3)
    assert summary["mean_certified_gap"] == pytest.approx(2.1)
    assert summary["median_first_exact_score_attempt"] == 1
    assert summary["p90_first_exact_score_attempt"] == 28
    assert summary["maximum_first_exact_score_attempt"] == 28
    assert summary["most_frequent_strict_loss_states"] == [{
        "queue": strict_queue,
        "frame_count": 2,
        "visited_time_s": 8.0,
        "first_exact_score_attempt": 28,
        "finite_budget_support_regret": 4.0,
        "bound_relaxation_slack": 2.0,
        "k26_option_indices": (0,),
        "exact_option_indices": (1,),
    }]


def test_registration_freezes_exact_18_row_roster_without_arrivals_and_rejects_tampering_before_replay(
    tmp_path, monkeypatch,
):
    output = tmp_path / "diagnostic.json"
    arrival_calls = []

    def unexpected_arrivals(*args, **kwargs):
        arrival_calls.append((args, kwargs))
        pytest.fail("registration or invalid-roster handling must not generate arrivals")

    monkeypatch.setattr(diagnostic, "poisson_arrivals", unexpected_arrivals)
    protocol = diagnostic.register(output)

    expected_roster = [
        (load, seed, budget)
        for load in diagnostic.LOADS
        for seed in diagnostic.SEEDS
        for budget in (diagnostic.SMALL_BUDGET, diagnostic.EXACT_BUDGET)
    ]
    observed_roster = [
        (row["load"], row["seed"], row["budget"])
        for row in protocol["source_rows"]
    ]
    assert observed_roster == expected_roster
    assert len(protocol["source_rows"]) == len(set(observed_roster)) == 18
    assert arrival_calls == []
    assert not output.exists()
    assert {path.name for path in tmp_path.iterdir()} == {
        "diagnostic_protocol.json", "diagnostic_fixed_sources.json",
    }

    protocol_path = tmp_path / "diagnostic_protocol.json"
    tampered = json.loads(protocol_path.read_text())
    tampered["source_rows"] = tampered["source_rows"][:-1]
    protocol_path.write_text(json.dumps(tampered))

    with pytest.raises(
        ValueError,
        match="registered candidate-search model, certificates or source rows changed",
    ):
        diagnostic.run_registered(output)
    assert arrival_calls == []
    assert not output.exists()
