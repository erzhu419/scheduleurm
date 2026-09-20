"""Compare successor priorities on previously observed development targets.

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
from algorithm.pressure_rollout_frame_candidates import select_pressure_rollout_frame


BUDGET = 26
PRESSURE_WORK_FIELDS = {
    **_WORK_FIELDS,
    **{name: name for name in (
        "rollout_pressure_profile_reads", "rollout_pressure_rate_evaluations",
        "rollout_priority_coordinate_evaluations",
    )},
}


def _empty_work():
    return {**dict.fromkeys(PRESSURE_WORK_FIELDS, 0), "selector_calls": 0,
            "rng_initializations": 0, "selector_wall_s": 0.0, "rng_setup_wall_s": 0.0,
            "action_descriptions": 0, "action_description_completion_offset_reads": 0}


def paired_summary(samples):
    """Summarize fixed challenger-minus-incumbent margins, before selection."""
    raw = [row["old_raw_margin"] for row in samples]
    pressure = [row["pressure_margin"] for row in samples]
    changes = [new - old for old, new in zip(raw, pressure, strict=True)]
    raw_variance, pressure_variance = variance(raw), variance(pressure)
    return {
        "sample_count": len(samples),
        "old_raw_margin_mean": mean(raw), "pressure_margin_mean": mean(pressure),
        "old_raw_margin_variance": raw_variance, "pressure_margin_variance": pressure_variance,
        "old_raw_margin_mc_se": sqrt(raw_variance / len(samples)),
        "pressure_margin_mc_se": sqrt(pressure_variance / len(samples)),
        "variance_ratio_pressure_over_old_raw": pressure_variance / raw_variance if raw_variance else None,
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
    pressure_rates = None
    maximum_budget_identity_error = maximum_draw_identity_error = 0
    for old in old_samples:
        sample_index = old["sample_index"]
        started = perf_counter()
        rngs = tuple(random.Random(NAMESPACE.format(sample_index=sample_index, coordinate=i))
                     for i in range(len(queues)))
        work["rng_setup_wall_s"] += perf_counter() - started
        started = perf_counter()
        selection = select_pressure_rollout_frame(
            model.options, queues, model.lanes, BUDGET, arrival_rates=rates,
            terminal_service_rates=mu, arrival_rngs=rngs,
        )
        work["selector_wall_s"] += perf_counter() - started
        work["selector_calls"] += 1
        work["rng_initializations"] += len(rngs)
        for name, attribute in PRESSURE_WORK_FIELDS.items():
            work[name] += getattr(selection, attribute, 0)
        description = action_description(model, selection)
        work["action_descriptions"] += 1
        work["action_description_completion_offset_reads"] += sum(map(len, description["completion_offsets"]))
        if not selection.rollout_used or physical(description) not in allowed:
            raise ValueError("pressure priority changed the frozen current contender pair")
        budget_error = selection.attempts - selection.root_attempts - selection.rollout_attempts
        draw_error = (selection.rollout_random_draws - selection.rollout_arrival_event_generations
                      - sum(rate > 0 for rate in rates))
        maximum_budget_identity_error = max(maximum_budget_identity_error, abs(budget_error))
        maximum_draw_identity_error = max(maximum_draw_identity_error, abs(draw_error))
        if selection.attempts > BUDGET or selection.rollout_attempts != 4 or budget_error or draw_error:
            raise ValueError("pressure diagnostic violated the registered budget or draw identity")
        if pressure_rates is None:
            pressure_rates = selection.rollout_pressure_rates
        elif pressure_rates != selection.rollout_pressure_rates:
            raise ValueError("static pressure rates changed across samples")
        margin = selection.analytic_challenger_rollout_cost - selection.incumbent_rollout_cost
        samples.append({"sample_index": sample_index, "old_raw_margin": old["raw_margin"],
                        "pressure_margin": margin, "paired_change": margin - old["raw_margin"]})
    # The priority changes successors only. Identical generators/rates/horizon
    # must therefore consume the same complete collection of planning events.
    for counter in ("random_draws", "rollout_arrival_event_generations"):
        if work[counter] != frozen_state["work"][counter]:
            raise ValueError("pressure priority changed the registered planning-path work")
    return {
        **{key: frozen_state[key] for key in (
            "state_id", "load", "queue", "incumbent_action", "challenger_action",
        )},
        "pressure_rates": pressure_rates, **paired_summary(samples),
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
    for label in ("old_raw", "pressure"):
        proxy = orientation * state[f"{label}_margin_mean"]
        estimates[label] = {
            "proxy_mean_delta_job_s": proxy,
            "proxy_mean_mc_se_job_s": state[f"{label}_margin_mc_se"],
            "realized_minus_proxy_mean_job_s": realized - proxy,
            "point_estimate_disagrees_with_realized": proxy * realized < 0,
        }
    change = orientation * state["paired_mean_change"]
    error = (estimates["old_raw"]["realized_minus_proxy_mean_job_s"]
             - estimates["pressure"]["realized_minus_proxy_mean_job_s"] - change)
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
            s["pressure_margin_variance"] > s["old_raw_margin_variance"] for s in states),
    }
    for label in ("old_raw", "pressure"):
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
        "planning_path_generation_work_matches_frozen_samples": True,
    }
    return {
        "diagnostic": "fixed_continuation_targets_pressure_priority_v1",
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
            "Changing successor priority changes the fixed-action proxy expectation. Variance ratios and paired Monte Carlo standard errors are descriptive, with sample ids crossed across states.",
            "Realized-minus-proxy residuals mix arrival-tape noise, horizon cutoff, and hybrid-model versus v4 continuation; they do not identify pure model bias.",
            "All new searches, path draws and pressure preparation are charged. Native totals include separately reported coordinate subcounts; frozen target-replay work is reused and not charged as new work.",
        ],
    }
