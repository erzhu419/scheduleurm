"""Compare mixed successors on previously observed development targets.

Every new margin uses the same planning sample as its retained raw-Poisson
margin. Real trajectories and their paired v4 continuations are reused exactly;
this diagnostic performs no new real-arrival simulation.
"""
from __future__ import annotations

from math import sqrt
import random
from statistics import mean, median, variance
from time import perf_counter

from algorithm.experiments.aligned_hardware_control_variate_continuation import (
    NAMESPACE, SAMPLES, physical,
)
from algorithm.experiments.aligned_hardware_control_variate_diagnostic import _WORK_FIELDS
from algorithm.experiments.aligned_hardware_frame_mechanism import action_description
from algorithm.mixed_successor_rollout_frame_candidates import select_mixed_successor_rollout_frame


BUDGET = 26
MIXED_WORK_FIELDS = {
    **_WORK_FIELDS,
    **{name: name for name in (
        "rollout_lane_fill_evaluations", "rollout_mixed_priority_coordinate_evaluations",
        "rollout_mixed_profile_probes", "rollout_temporary_queue_updates",
    )},
}


def _empty_work():
    return {**dict.fromkeys(MIXED_WORK_FIELDS, 0), "selector_calls": 0,
            "rng_initializations": 0, "selector_wall_s": 0.0, "rng_setup_wall_s": 0.0,
            "action_descriptions": 0, "action_description_completion_offset_reads": 0}


def paired_summary(samples):
    """Summarize fixed challenger-minus-incumbent margins, before selection."""
    raw = [row["old_raw_margin"] for row in samples]
    mixed = [row["mixed_margin"] for row in samples]
    changes = [new - old for old, new in zip(raw, mixed, strict=True)]
    raw_variance, mixed_variance = variance(raw), variance(mixed)
    return {
        "sample_count": len(samples),
        "old_raw_margin_mean": mean(raw), "mixed_margin_mean": mean(mixed),
        "old_raw_margin_variance": raw_variance, "mixed_margin_variance": mixed_variance,
        "old_raw_margin_mc_se": sqrt(raw_variance / len(samples)),
        "mixed_margin_mc_se": sqrt(mixed_variance / len(samples)),
        "variance_ratio_mixed_over_old_raw": mixed_variance / raw_variance if raw_variance else None,
        "paired_mean_change": mean(changes),
        "paired_mean_change_mc_se": sqrt(variance(changes) / len(samples)),
    }


def sample_state(model, certificate, mu, frozen_state):
    queues = frozen_state["queue"]
    old_samples = frozen_state["samples"]
    if [row["sample_index"] for row in old_samples] != list(range(SAMPLES)):
        raise ValueError("frozen state does not contain the registered 64 sample indices")
    rates = certificate["arrival_rates"]
    allowed = {physical(frozen_state[name]) for name in ("incumbent_action", "challenger_action")}
    work, samples = _empty_work(), []
    maximum_budget_identity_error = maximum_draw_identity_error = 0
    for old in old_samples:
        sample_index = old["sample_index"]
        started = perf_counter()
        rngs = tuple(random.Random(NAMESPACE.format(sample_index=sample_index, coordinate=i))
                     for i in range(len(queues)))
        work["rng_setup_wall_s"] += perf_counter() - started
        started = perf_counter()
        selection = select_mixed_successor_rollout_frame(
            model.options, queues, model.lanes, BUDGET, arrival_rates=rates,
            terminal_service_rates=mu, arrival_rngs=rngs,
        )
        work["selector_wall_s"] += perf_counter() - started
        work["selector_calls"] += 1
        work["rng_initializations"] += len(rngs)
        for name, attribute in MIXED_WORK_FIELDS.items():
            work[name] += getattr(selection, attribute, 0)
        description = action_description(model, selection)
        work["action_descriptions"] += 1
        work["action_description_completion_offset_reads"] += sum(map(len, description["completion_offsets"]))
        if not selection.rollout_used or physical(description) not in allowed:
            raise ValueError("mixed successors changed the frozen current contender pair")
        if (selection.support_incumbent_score != frozen_state["incumbent_action"]["score"]
                or selection.analytic_challenger_score != frozen_state["challenger_action"]["score"]):
            raise ValueError("mixed successors changed a frozen current contender score")
        budget_error = selection.attempts - selection.root_attempts - selection.rollout_attempts
        draw_error = (selection.rollout_random_draws - selection.rollout_arrival_event_generations
                      - sum(rate > 0 for rate in rates))
        maximum_budget_identity_error = max(maximum_budget_identity_error, abs(budget_error))
        maximum_draw_identity_error = max(maximum_draw_identity_error, abs(draw_error))
        if selection.attempts > BUDGET or selection.rollout_attempts != 4 or budget_error or draw_error:
            raise ValueError("mixed diagnostic violated the registered budget or draw identity")
        margin = selection.analytic_challenger_rollout_cost - selection.incumbent_rollout_cost
        samples.append({"sample_index": sample_index, "old_raw_margin": old["raw_margin"],
                        "mixed_margin": margin, "paired_change": margin - old["raw_margin"]})
    # Successor construction changes only future actions. Current search and
    # the generators/rates/horizon retain the frozen work on every state.
    for counter in (
        "random_draws", "rollout_arrival_event_generations", "root_attempts",
        "initial_root_attempts", "recycled_candidate_attempts", "seed_attempts",
        "feasible_candidate_count", "bound_option_probes", "bound_threshold_evaluations",
        "cost_candidate_evaluations", "initial_cost_candidate_evaluations",
        "recycled_cost_candidate_evaluations",
    ):
        if work[counter] != frozen_state["work"][counter]:
            raise ValueError(f"mixed successors changed frozen current-search or path work: {counter}")
    return {
        **{key: frozen_state[key] for key in (
            "state_id", "load", "queue", "incumbent_action", "challenger_action",
        )},
        **paired_summary(samples),
        "maximum_budget_identity_error": maximum_budget_identity_error,
        "maximum_random_draw_identity_error": maximum_draw_identity_error,
        "samples": samples, "work": work,
    }


def compare_case(case, state):
    """Orient both model estimates to the retained CV-first minus raw-first target."""
    orientation = case["orientation_cv_minus_raw"]
    if orientation not in (-1, 1):
        raise ValueError("retained intervention orientation must be plus or minus one")
    realized = case["effect"]["realized_total_delta_job_s"]
    estimates = {}
    for label in ("old_raw", "mixed"):
        proxy = orientation * state[f"{label}_margin_mean"]
        estimates[label] = {
            "proxy_mean_delta_job_s": proxy,
            "proxy_mean_mc_se_job_s": state[f"{label}_margin_mc_se"],
            "realized_minus_proxy_mean_job_s": realized - proxy,
            "point_estimate_disagrees_with_realized": proxy * realized < 0,
        }
    change = orientation * state["paired_mean_change"]
    error = (estimates["old_raw"]["realized_minus_proxy_mean_job_s"]
             - estimates["mixed"]["realized_minus_proxy_mean_job_s"] - change)
    if abs(error) > 1e-6:
        raise ValueError("oriented model-change residual identity failed")
    return {
        "state_id": state["state_id"],
        **{key: case[key] for key in ("frame_index", "start_s", "phase", "orientation_cv_minus_raw")},
        "realized_total_delta_job_s": realized, "estimates": estimates,
        "oriented_paired_mean_change_job_s": change,
        "paired_mean_change_mc_se_job_s": state["paired_mean_change_mc_se"],
        "residual_identity_error_job_s": error,
    }


def summarize(states, cases):
    result = {
        "unique_state_count": len(states), "retained_case_occurrences": len(cases),
        "states_with_larger_sample_variance": sum(
            s["mixed_margin_variance"] > s["old_raw_margin_variance"] for s in states),
    }
    for label in ("old_raw", "mixed"):
        group = [case["estimates"][label] for case in cases]
        absolute = [abs(row["realized_minus_proxy_mean_job_s"]) for row in group]
        result[label] = {
            "point_estimate_disagrees_with_realized": sum(row["point_estimate_disagrees_with_realized"] for row in group),
            "mean_absolute_realized_minus_proxy_mean_job_s": mean(absolute) if absolute else None,
            "median_absolute_realized_minus_proxy_mean_job_s": median(absolute) if absolute else None,
        }
    return result


def diagnose(model, certificates, mu, frozen):
    """Reuse all 63 fixed states and 174 targets; PASS covers structural checks."""
    started = perf_counter()
    if frozen["status"] != "PASS" or len(frozen["states"]) != 63:
        raise ValueError("expected the completed frozen 63-state continuation diagnostic")
    if frozen["protocol"]["repeated_sampling"]["rng_namespace"] != NAMESPACE:
        raise ValueError("frozen planning namespace differs from the registered paired samples")
    states = [sample_state(model, certificates[s["load"]], mu, s) for s in frozen["states"]]
    by_id = {state["state_id"]: state for state in states}
    cases = []
    for source in frozen["runs"]:
        for case in source["cases"]:
            state = by_id[case["state_id"]]
            for name in ("incumbent_action", "challenger_action"):
                if physical(case[name]) != physical(state[name]):
                    raise ValueError("retained continuation targets a different contender pair")
            cases.append({**{key: source[key] for key in ("load", "seed", "planning_seed", "source_policy")},
                          **compare_case(case, state)})
    if len(cases) != 174 or len(by_id) != 63 or {case["state_id"] for case in cases} != set(by_id):
        raise ValueError("fixed target or state roster changed")
    checks = {
        "all_63_states_have_64_paired_samples": all(s["sample_count"] == SAMPLES for s in states),
        "all_174_frozen_interventions_retained": len(cases) == 174,
        "current_contenders_budget_draws_and_residual_identities_hold": True,
        "current_search_and_path_generation_work_match_frozen_samples": True,
    }
    return {
        "diagnostic": "fixed_continuation_targets_mixed_successors_v1",
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "budget": BUDGET, "samples_per_state": SAMPLES, "rng_namespace": NAMESPACE,
        "sample_margin_sign": "fixed analytic challenger minus support incumbent",
        "case_delta_sign": "retained CV-first minus raw-first intervention",
        "states": states, "cases": cases, "summary": summarize(states, cases),
        "summary_by_load": [
            {"load": load, **summarize([s for s in states if s["load"] == load],
                                      [c for c in cases if c["load"] == load])}
            for load in certificates
        ],
        "work": {key: sum(state["work"][key] for state in states) for key in _empty_work()},
        "new_real_trajectory_replays": 0, "diagnostic_wall_s": perf_counter() - started,
        "limitations": [
            "Targets reuse consumed development trajectories, with overlapping futures and repeated states; case counts are occurrences, not independent replicates or whole-policy effects.",
            "Changing successor construction changes the fixed-action proxy expectation. Variance ratios and paired Monte Carlo standard errors are descriptive, with sample ids crossed across states.",
            "Realized-minus-proxy residuals mix arrival-tape noise, horizon cutoff, and hybrid-model versus v4 continuation; they do not identify pure model bias.",
            "All new searches, path draws and mixed lane-filling work are charged. Native totals include separately reported coordinate and profile subcounts; frozen target-replay work is reused and not charged as new work.",
        ],
    }
