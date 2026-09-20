"""Fresh paired model estimates on actual raw/mixed action disagreements."""
from __future__ import annotations

from math import sqrt
import random
from statistics import mean, median, variance
from time import perf_counter

from algorithm.experiments.aligned_hardware_control_variate_continuation import physical
from algorithm.experiments.aligned_hardware_control_variate_diagnostic import _WORK_FIELDS
from algorithm.experiments.aligned_hardware_frame_mechanism import action_description
from algorithm.mixed_successor_rollout_frame_candidates import select_mixed_successor_rollout_frame
from algorithm.poisson_rollout_frame_candidates import select_poisson_rollout_frame


BUDGET = 26
NAMESPACE = "aligned-mixed-continuation:{sample_index}:{coordinate}"
WORK_FIELDS = {
    **_WORK_FIELDS,
    **{name: name for name in (
        "rollout_lane_fill_evaluations", "rollout_mixed_priority_coordinate_evaluations",
        "rollout_mixed_profile_probes", "rollout_temporary_queue_updates",
    )},
}
CURRENT_FIELDS = (
    "root_attempts", "initial_root_attempts", "recycled_candidate_attempts",
    "seed_attempts", "feasible_candidate_count", "bound_option_probes",
    "bound_threshold_evaluations", "cost_candidate_evaluations",
    "initial_cost_candidate_evaluations", "recycled_cost_candidate_evaluations",
    "root_lookahead_cost", "support_incumbent_lookahead_cost",
    "support_incumbent_score", "analytic_challenger_score",
)


def state_key(load, queues):
    return f"{load}:" + ",".join(map(str, queues))


def _empty_work():
    return {**dict.fromkeys(WORK_FIELDS, 0), "selector_calls": 0,
            "rng_initializations": 0, "rng_state_reads": 0,
            "selector_wall_s": 0.0, "rng_setup_wall_s": 0.0,
            "action_descriptions": 0, "action_description_completion_offset_reads": 0}


def paired_summary(samples):
    raw = [row["raw_margin"] for row in samples]
    mixed = [row["mixed_margin"] for row in samples]
    changes = [new - old for old, new in zip(raw, mixed, strict=True)]
    raw_variance, mixed_variance = variance(raw), variance(mixed)
    return {
        "sample_count": len(samples), "raw_margin_mean": mean(raw), "mixed_margin_mean": mean(mixed),
        "raw_margin_variance": raw_variance, "mixed_margin_variance": mixed_variance,
        "raw_margin_mc_se": sqrt(raw_variance / len(samples)),
        "mixed_margin_mc_se": sqrt(mixed_variance / len(samples)),
        "variance_ratio_mixed_over_raw": mixed_variance / raw_variance if raw_variance else None,
        "paired_mean_change": mean(changes),
        "paired_mean_change_mc_se": sqrt(variance(changes) / len(samples)),
    }


def sample_state(model, certificate, mu, case, samples=64):
    """Estimate both fixed-action margins using separate, identically seeded RNGs."""
    queues, rates = case["queue"], certificate["arrival_rates"]
    load = certificate["load_factor"]
    allowed = {physical(case[name]) for name in ("incumbent_action", "challenger_action")}
    work = {label: _empty_work() for label in ("raw", "mixed")}
    retained = []
    for sample_index in range(samples):
        selections, end_states = {}, {}
        for label, selector in (("raw", select_poisson_rollout_frame),
                                ("mixed", select_mixed_successor_rollout_frame)):
            costs = work[label]
            started = perf_counter()
            rngs = tuple(random.Random(NAMESPACE.format(sample_index=sample_index, coordinate=i))
                         for i in range(len(queues)))
            costs["rng_setup_wall_s"] += perf_counter() - started
            costs["rng_initializations"] += len(rngs)
            started = perf_counter()
            selected = selector(model.options, queues, model.lanes, BUDGET,
                                arrival_rates=rates, terminal_service_rates=mu, arrival_rngs=rngs)
            costs["selector_wall_s"] += perf_counter() - started
            costs["selector_calls"] += 1
            for name, attribute in WORK_FIELDS.items():
                costs[name] += getattr(selected, attribute, 0)
            end_states[label] = tuple(rng.getstate() for rng in rngs)
            costs["rng_state_reads"] += len(rngs)
            description = action_description(model, selected)
            costs["action_descriptions"] += 1
            costs["action_description_completion_offset_reads"] += sum(map(len, description["completion_offsets"]))
            if not selected.rollout_used or physical(description) not in allowed:
                raise ValueError("fresh sample changed the fixed current contender pair")
            if (selected.support_incumbent_score != case["incumbent_action"]["score"]
                    or selected.analytic_challenger_score != case["challenger_action"]["score"]
                    or selected.root_lookahead_cost != case["challenger_analytic_cost"]
                    or selected.support_incumbent_lookahead_cost != case["incumbent_analytic_cost"]):
                raise ValueError("fresh sample changed a discovery contender score or analytic cost")
            if (selected.attempts > BUDGET or selected.rollout_attempts != 4
                    or selected.attempts != selected.root_attempts + selected.rollout_attempts
                    or selected.rollout_random_draws != selected.rollout_arrival_event_generations + sum(r > 0 for r in rates)):
                raise ValueError("fresh sample violated the root-plus-four budget or draw identity")
            selections[label] = selected
        raw, mixed = selections["raw"], selections["mixed"]
        if (end_states["raw"] != end_states["mixed"]
                or raw.rollout_random_draws != mixed.rollout_random_draws
                or raw.rollout_arrival_event_generations != mixed.rollout_arrival_event_generations
                or raw.rollout_horizon != mixed.rollout_horizon
                or any(getattr(raw, name) != getattr(mixed, name) for name in CURRENT_FIELDS)):
            raise ValueError("paired fresh selectors changed current search or the shared planning path")
        raw_margin = raw.analytic_challenger_rollout_cost - raw.incumbent_rollout_cost
        mixed_margin = mixed.analytic_challenger_rollout_cost - mixed.incumbent_rollout_cost
        retained.append({"sample_index": sample_index, "raw_margin": raw_margin,
                         "mixed_margin": mixed_margin, "paired_change": mixed_margin - raw_margin})
    return {
        "state_id": state_key(load, queues), "load": load, "queue": list(queues),
        **{key: case[key] for key in ("incumbent_action", "challenger_action",
                                    "incumbent_analytic_cost", "challenger_analytic_cost")},
        **paired_summary(retained), "samples": retained,
        "raw_work": work["raw"], "mixed_work": work["mixed"], "status": "PASS",
        "checks": {"fixed_current_contenders_and_analytic_costs_hold": True,
                   "paired_rng_end_states_and_path_work_match": True,
                   "root_plus_four_budget_and_draw_identities_hold": True},
    }


def compare_case(case, state):
    """Partition observed residuals without treating the target mismatch as pure bias."""
    orientation = case["orientation_mixed_minus_raw"]
    if orientation not in (-1, 1):
        raise ValueError("actual intervention orientation must be plus or minus one")
    realized = case["effect"]["realized_total_delta_job_s"]
    result = {}
    for label in ("raw", "mixed"):
        discovery = orientation * case[f"{label}_rollout_margin"]
        proxy = orientation * state[f"{label}_margin_mean"]
        adjustment, mismatch = proxy - discovery, realized - proxy
        error = adjustment + mismatch - (realized - discovery)
        if abs(error) > 1e-6:
            raise ValueError("fresh-sampling and target-mismatch residual partition failed")
        result[label] = {
            "discovery_sample_delta_job_s": discovery,
            "fresh_proxy_mean_delta_job_s": proxy,
            "fresh_proxy_mean_mc_se_job_s": state[f"{label}_margin_mc_se"],
            "fresh_mean_minus_discovery_sample_job_s": adjustment,
            "realized_minus_fresh_proxy_mean_job_s": mismatch,
            "residual_identity_error_job_s": error,
            "proxy_point_estimate_disagrees_with_realized": proxy * realized < 0,
        }
    return result


def summarize(rows, states):
    """Descriptive occurrence statistics; callers filter rows/states for each load."""
    cases = [case for row in rows for case in row["cases"]]
    result = {
        "source_count": len(rows), "disagreements": len(cases),
        "unique_states": len({case["state_id"] for case in cases}),
        **{outcome: sum(case["effect"]["outcome"] == outcome for case in cases)
           for outcome in ("improved", "worsened", "unchanged")},
        "mean_realized_delta_job_s": mean(case["effect"]["realized_total_delta_job_s"] for case in cases) if cases else None,
        "state_variance_comparison": {
            "decreased": sum(s["mixed_margin_variance"] < s["raw_margin_variance"] for s in states),
            "equal": sum(s["mixed_margin_variance"] == s["raw_margin_variance"] for s in states),
            "increased": sum(s["mixed_margin_variance"] > s["raw_margin_variance"] for s in states),
        },
    }
    for label in ("raw", "mixed"):
        group = [case["estimate_comparison"][label] for case in cases]
        target_residuals = [abs(row["realized_minus_fresh_proxy_mean_job_s"]) for row in group]
        sample_adjustments = [abs(row["fresh_mean_minus_discovery_sample_job_s"]) for row in group]
        result[label] = {
            "proxy_point_estimate_disagrees_with_realized": sum(row["proxy_point_estimate_disagrees_with_realized"] for row in group),
            "mean_absolute_realized_minus_fresh_proxy_mean_job_s": mean(target_residuals) if group else None,
            "median_absolute_realized_minus_fresh_proxy_mean_job_s": median(target_residuals) if group else None,
            "mean_absolute_fresh_mean_minus_discovery_sample_job_s": mean(sample_adjustments) if group else None,
            "median_absolute_fresh_mean_minus_discovery_sample_job_s": median(sample_adjustments) if group else None,
        }
    return result
