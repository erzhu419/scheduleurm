from math import sqrt
from statistics import variance

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.lookahead_frame_candidates import select_lookahead_frame
from algorithm.experiments.aligned_hardware_frame_mechanism import action_description
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments import aligned_hardware_mixed_successor_continuation_sampling as sampling


def test_mean_change_uncertainty_uses_the_paired_paths():
    raw, changes = (101., -99., 201., -199.), (1., 2., 3., 2.)
    samples = [{"raw_margin": old, "mixed_margin": old + change}
               for old, change in zip(raw, changes, strict=True)]
    summary = sampling.paired_summary(samples)
    assert summary["raw_margin_mean"] == 1
    assert summary["mixed_margin_mean"] == 3
    assert summary["paired_mean_change"] == 2
    assert summary["paired_mean_change_mc_se"] == pytest.approx(sqrt(variance(changes) / 4))
    assert summary["paired_mean_change_mc_se"] < summary["mixed_margin_mc_se"] / 100


@pytest.mark.parametrize("orientation", (-1, 1))
def test_discovery_fresh_and_realized_residuals_use_the_same_orientation(orientation):
    state = {"state_id": "toy", "raw_margin_mean": 2., "mixed_margin_mean": -4.,
             "raw_margin_mc_se": .5, "mixed_margin_mc_se": 1.,
             "raw_margin_variance": 1., "mixed_margin_variance": 4.}
    case = {"state_id": "toy", "orientation_mixed_minus_raw": orientation,
            "raw_rollout_margin": 5., "mixed_rollout_margin": -1.,
            "effect": {"realized_total_delta_job_s": -10 * orientation,
                       "outcome": "improved" if orientation == 1 else "worsened"}}
    comparison = sampling.compare_case(case, state)
    for label, discovery, proxy in (("raw", 5., 2.), ("mixed", -1., -4.)):
        row = comparison[label]
        assert row["discovery_sample_delta_job_s"] == orientation * discovery
        assert row["fresh_proxy_mean_delta_job_s"] == orientation * proxy
        assert row["fresh_mean_minus_discovery_sample_job_s"] == -3 * orientation
        assert row["residual_identity_error_job_s"] == 0
    assert comparison["raw"]["proxy_point_estimate_disagrees_with_realized"]
    assert not comparison["mixed"]["proxy_point_estimate_disagrees_with_realized"]
    case["estimate_comparison"] = comparison
    summary = sampling.summarize([{"cases": [case, case]}], [state])
    assert summary["disagreements"] == 2 and summary["unique_states"] == 1
    assert summary["state_variance_comparison"] == {"decreased": 0, "equal": 0, "increased": 1}
    assert summary["raw"]["mean_absolute_realized_minus_fresh_proxy_mean_job_s"] == 12
    assert summary["mixed"]["median_absolute_realized_minus_fresh_proxy_mean_job_s"] == 6


def _fixed_case():
    model = load_model()
    certificate = capacity_certificate(model, .5)
    mu, queue = terminal_service_rates(model, certificate), (1, 4, 8)
    incumbent = select_budgeted_frame(model.options, queue, model.lanes, 22)
    challenger = select_lookahead_frame(model.options, queue, model.lanes, 22,
                                        arrival_rates=certificate["arrival_rates"], terminal_service_rates=mu)
    case = {"queue": queue, "incumbent_action": action_description(model, incumbent),
            "challenger_action": action_description(model, challenger),
            "incumbent_analytic_cost": challenger.support_incumbent_lookahead_cost,
            "challenger_analytic_cost": challenger.lookahead_cost}
    return model, certificate, mu, case


def test_three_fresh_samples_charge_both_selectors_and_both_rng_families():
    state = sampling.sample_state(*_fixed_case(), samples=3)
    assert state["status"] == "PASS" and all(state["checks"].values())
    assert state["sample_count"] == 3
    assert [row["sample_index"] for row in state["samples"]] == [0, 1, 2]
    raw, mixed = state["raw_work"], state["mixed_work"]
    for work in (raw, mixed):
        assert work["selector_calls"] == work["action_descriptions"] == 3
        assert work["rng_initializations"] == work["rng_state_reads"] == 9
        assert work["candidate_attempts"] == work["root_attempts"] + 12
        assert work["rollout_attempts"] == 12
        assert work["random_draws"] == work["rollout_arrival_event_generations"] + 9
    assert raw["random_draws"] == mixed["random_draws"]
    assert raw["rollout_temporary_queue_updates"] == 0 < mixed["rollout_temporary_queue_updates"]
    assert mixed["rollout_mixed_priority_coordinate_evaluations"] == mixed["rollout_class_evaluations"]
    assert mixed["rollout_mixed_profile_probes"] == mixed["rollout_profile_probes"]


def test_extra_rng_consumption_cannot_pass_using_equal_reported_draw_counts(monkeypatch):
    original = sampling.select_mixed_successor_rollout_frame

    def changed(*args, **kwargs):
        selected = original(*args, **kwargs)
        kwargs["arrival_rngs"][0].random()
        return selected

    monkeypatch.setattr(sampling, "select_mixed_successor_rollout_frame", changed)
    with pytest.raises(ValueError, match="shared planning path"):
        sampling.sample_state(*_fixed_case(), samples=3)
