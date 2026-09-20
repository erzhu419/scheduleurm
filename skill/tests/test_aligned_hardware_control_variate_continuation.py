from math import sqrt

import pytest

from algorithm.experiments.aligned_hardware_control_variate_continuation import compare_estimates, physical, state_key, summarize


@pytest.mark.parametrize("orientation", (-1, 1))
def test_sample_proxy_realized_partition_keeps_cv_first_sign(orientation):
    case = {"orientation_cv_minus_raw": orientation,
            "raw_rollout_margin": -5., "corrected_rollout_margin": 3.,
            "effect": {"realized_total_delta_job_s": 20.}}
    state = {"raw_margin_mean": 10., "corrected_margin_mean": 8.,
             "raw_margin_mc_se": 1., "corrected_margin_mc_se": 5.}
    result = compare_estimates(case, state)
    assert result["raw"]["fresh_proxy_mean_delta_job_s"] == orientation * 10
    assert result["raw"]["discovery_sample_delta_job_s"] == orientation * -5
    assert result["raw"]["fresh_mean_minus_discovery_sample_job_s"] == orientation * 15
    assert result["raw"]["realized_minus_fresh_proxy_mean_job_s"] == 20 - orientation * 10
    assert result["raw"]["proxy_preference"] == ("cv" if orientation < 0 else "raw")
    assert result["corrected"]["proxy_preference"] == "unresolved"
    assert all(row["residual_identity_error_job_s"] == 0 for row in result.values())


def test_empty_source_remains_in_summary_and_json_actions_have_same_key():
    assert summarize([{"cases": []}])["source_count"] == 1
    assert summarize([{"cases": []}])["mean_realized_delta_job_s"] is None
    first = {"option_indices": (1, 2), "departures": (1, 2),
             "duration_s": 3., "completion_offsets": ((1.,), (2., 3.))}
    second = {**first, "option_indices": [2, 1], "departures": [1, 2],
              "completion_offsets": [[1.], [2., 3.]]}
    assert physical(first) == physical(second)
    assert state_key(.5, [1, 2, 3]) != state_key(.8, [1, 2, 3])


def test_fresh_paths_charge_full_work_and_use_fixed_margins(monkeypatch):
    import random
    import algorithm.experiments.aligned_hardware_control_variate_continuation as diagnostic
    from algorithm.control_variate_rollout_frame_candidates import select_control_variate_rollout_frame
    from algorithm.experiments.aligned_hardware_frame_mechanism import action_description
    from algorithm.experiments.aligned_hardware_frame_model import load_model, capacity_certificate
    from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
    from algorithm.lookahead_frame_candidates import select_lookahead_frame
    from algorithm.budgeted_frame_candidates import select_budgeted_frame

    model = load_model()
    certificate = capacity_certificate(model, .5)
    mu = terminal_service_rates(model, certificate)
    queue = (1, 4, 8)
    incumbent = select_budgeted_frame(model.options, queue, model.lanes, 22)
    challenger = select_lookahead_frame(model.options, queue, model.lanes, 22,
                                       arrival_rates=certificate["arrival_rates"], terminal_service_rates=mu)
    case = {"queue": queue, "incumbent_action": action_description(model, incumbent),
            "challenger_action": action_description(model, challenger)}
    monkeypatch.setattr(diagnostic, "SAMPLES", 4)
    state = diagnostic.sample_state(model, certificate, .5, mu, case)
    selected = select_control_variate_rollout_frame(
        model.options, queue, model.lanes, 26, arrival_rates=certificate["arrival_rates"],
        terminal_service_rates=mu,
        arrival_rngs=tuple(random.Random(diagnostic.NAMESPACE.format(sample_index=0, coordinate=i)) for i in range(3)))
    assert state["samples"][0]["raw_margin"] == selected.raw_rollout_margin
    assert state["samples"][0]["corrected_margin"] == selected.corrected_rollout_margin
    assert state["sample_count"] == state["work"]["selector_calls"] == 4
    assert state["work"]["rng_initializations"] == 12
    assert state["work"]["rollout_attempts"] == 16
    assert state["work"]["cv_terminal_value_coordinate_evaluations"] == 24
    assert state["work"]["random_draws"] == state["work"]["rollout_arrival_event_generations"] + 12
    assert state["raw_margin_mc_se"] == pytest.approx(sqrt(state["raw_margin_variance"] / 4))
