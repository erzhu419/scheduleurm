from dataclasses import replace
import math

import pytest

from algorithm.experiments.aligned_hardware_frame_experiment import (
    conditional_lyapunov_drift, exact_search_budget, poisson_arrivals, simulate_frames,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    FrameModel, FrameOption, capacity_certificate, load_model,
)


def _model():
    return FrameModel(
        "fixture", "fixture", 2, ("a", "b"), (1.0, 1.0),
        (FrameOption("a", (1, 0), 2.0, ((2.0,), ())),
         FrameOption("b", (0, 1), 3.0, ((), (3.0,))),
         FrameOption("ab", (1, 1), 4.0, ((2.0,), (4.0,)), loaded=True)),
    )


def test_conditional_drift_matches_independent_poisson_expectation():
    # Directly sum the Poisson probabilities, independently of the formula.
    q, departure, arrival_mean = 5, 2, 1.3
    expected = sum(
        math.exp(-arrival_mean) * arrival_mean ** k / math.factorial(k)
        * ((q - departure + k) ** 2 - q ** 2) / 2
        for k in range(40)
    )
    assert conditional_lyapunov_drift((q,), (departure,), (arrival_mean,)) == pytest.approx(expected)


def test_same_visible_history_different_future_produces_same_decision_prefix():
    model = _model()
    certificate = capacity_certificate(model, 0.8)
    initial = ((0.0, 0, 0), (0.0, 1, 0), (0.0, 0, 1))
    first = simulate_frames(model, initial + ((12.0, 0, 2),), 15.0, certificate, 8, audit_limit=40)
    second = simulate_frames(model, initial + ((11.0, 1, 1), (13.0, 1, 2)), 15.0, certificate, 8, audit_limit=40)
    prefix = lambda run: [row for row in run["frame_audit_prefix"] if row["start_s"] + row["duration_s"] < 11.0]
    assert prefix(first) == prefix(second)
    assert first["completed_jobs"] == 4
    assert second["completed_jobs"] == 5


def test_arrivals_during_frame_wait_and_departures_match_actual_completion_offsets():
    # One lane avoids a shorter homogeneous two-lane frame replacing the loaded
    # option: Q=(1,1) makes 2/4 > 1/3 and ties 1/2; shorten loaded to 3.5.
    base = _model()
    loaded = replace(base.options[2], duration=3.5, completion_offsets=((2.0,), (3.5,)))
    model = replace(base, lanes=1, options=(*base.options[:2], loaded))
    certificate = capacity_certificate(model, 0.5)
    run = simulate_frames(
        model, ((0.0, 0, 0), (0.0, 1, 0), (1.0, 0, 1)),
        2.0, certificate, exact_search_budget(3, 1), audit_limit=10,
    )
    first = run["frame_audit_prefix"][0]
    assert first["duration_s"] == 3.5
    assert first["departures"] == [1, 1]
    assert first["arrivals_during_frame"] == [1, 0]
    assert first["queue_after"] == (1, 0)
    assert run["completed_jobs"] == 3
    # First a finishes at 2; b at 3.5; the new a waits for 3.5 and finishes at 5.5.
    assert run["mean_flow_s"] == pytest.approx((2 + 3.5 + 4.5) / 3)
    assert run["makespan_s"] == 5.5
    assert run["maximum_queue_recursion_violation"] == 0


def test_real_model_all_budgets_use_identical_poisson_jobs_and_exact_reference_closes():
    model = load_model()
    certificate = capacity_certificate(model, 0.8)
    tape = poisson_arrivals(certificate["arrival_rates"], 250.0, 41)
    assert tape == poisson_arrivals(certificate["arrival_rates"], 250.0, 41)
    budgets = (len(model.options) + 1, exact_search_budget(len(model.options), model.lanes))
    runs = [simulate_frames(model, tape, 250.0, certificate, budget) for budget in budgets]
    assert all(run["completed_jobs"] == len(tape) for run in runs)
    assert all(run["maximum_queue_recursion_violation"] == 0 for run in runs)
    assert all(run["maximum_expected_drift_bound_violation"] <= 1e-8 for run in runs)
    assert all(run["maximum_full_support_deficit"] <= certificate["availability_loss_C"] + 1e-8 for run in runs)
    assert runs[1]["exact_selection_frame_count"] == runs[1]["frame_count"]
    assert runs[1]["maximum_certified_candidate_gap"] == 0
