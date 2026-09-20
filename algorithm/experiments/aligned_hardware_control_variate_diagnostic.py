"""Describe paired planning-cost variance on a fixed roster of integer queues.

This diagnostic calls only the new selector, once per retained sample. Both
raw and corrected margins come from its same two contenders and shared path.
An inactive first probe is charged and retained as an inactive state, without
63 redundant calls. No real arrival tape is constructed or consulted.
"""
from __future__ import annotations

import math
import random
from statistics import mean, variance
from time import perf_counter

from algorithm.control_variate_rollout_frame_candidates import select_control_variate_rollout_frame


DIAGNOSTIC_QUEUES = (
    (0, 0, 0), (1, 0, 0), (1, 1, 1), (3, 0, 3), (1, 4, 8),
    (4, 4, 8), (4, 1, 5), (10, 12, 16), (2, 5, 8), (1, 6, 4),
)
DIAGNOSTIC_SAMPLES = 64
DIAGNOSTIC_BUDGET = 26
IDENTITY_TOLERANCE = 1e-8

# These are operation counts, not budgets, queue values or frontier maxima.
# Native totals include the separately named subcounts; do not add them again.
_WORK_FIELDS = {
    "candidate_attempts": "attempts",
    "random_draws": "rollout_random_draws",
    **{name: name for name in (
        "seed_attempts", "feasible_candidate_count", "local_score_evaluations",
        "bound_option_probes", "bound_threshold_evaluations", "root_attempts",
        "rollout_attempts", "cost_candidate_evaluations", "completion_offset_reads",
        "next_queue_coordinate_evaluations", "terminal_value_coordinate_evaluations",
        "rollout_class_evaluations", "rollout_profile_probes", "rollout_setup_requirement_reads",
        "rollout_completion_offset_reads", "rollout_terminal_value_coordinate_evaluations",
        "rollout_cost_candidate_evaluations", "physical_action_completion_offset_reads",
        "initial_root_attempts", "recycled_candidate_attempts",
        "initial_cost_candidate_evaluations", "recycled_cost_candidate_evaluations",
        "rollout_arrival_event_generations", "rollout_arrival_rate_reads",
        "rollout_arrival_event_reads", "rollout_tail_segments",
        "rollout_queue_coordinate_evaluations", "rollout_current_completion_offset_reads",
        "cv_terminal_value_coordinate_evaluations",
    )},
}


def _empty_work():
    return {**dict.fromkeys(_WORK_FIELDS, 0), "selector_calls": 0,
            "rng_initializations": 0, "selector_wall_s": 0.0, "rng_setup_wall_s": 0.0}


def _sample_summary(samples):
    if not samples:
        return {"sample_count": 0, "raw_margin_mean": None, "corrected_margin_mean": None,
                "raw_margin_variance": None, "corrected_margin_variance": None,
                "variance_ratio_cv_over_raw": None, "paired_mean_change": None,
                "paired_mean_change_mc_se": None}
    raw = [row["raw_margin"] for row in samples]
    corrected = [row["corrected_margin"] for row in samples]
    changes = [new - old for old, new in zip(raw, corrected, strict=True)]
    raw_variance, corrected_variance = variance(raw), variance(corrected)
    return {
        "sample_count": len(samples), "raw_margin_mean": mean(raw),
        "corrected_margin_mean": mean(corrected),
        "raw_margin_variance": raw_variance, "corrected_margin_variance": corrected_variance,
        "variance_ratio_cv_over_raw": corrected_variance / raw_variance if raw_variance else None,
        "paired_mean_change": mean(changes),
        "paired_mean_change_mc_se": math.sqrt(variance(changes) / len(changes)),
    }


def diagnose(model, certificates, loads, mu):
    """Return descriptive within-state variance and fully charged selector work.

    Sample indices are crossed with every fixed queue and load. Each call gets
    fresh class RNGs in a diagnostic-only namespace. Variance uses the fixed
    challenger-minus-incumbent margin, regardless of the action selected. The
    only execution verdict is the algebraic control-variate identity check;
    neither a variance reduction nor a small Monte Carlo mean error is required.
    """
    started = perf_counter()
    loads = tuple(loads)
    total_work = _empty_work()
    states = []
    maximum_identity_error = 0.0
    for load, certificate in zip(loads, certificates, strict=True):
        rates = certificate["arrival_rates"]
        for queues in DIAGNOSTIC_QUEUES:
            state_work = _empty_work()
            samples = []
            active = False
            for sample_index in range(DIAGNOSTIC_SAMPLES):
                rng_started = perf_counter()
                rngs = tuple(random.Random(f"aligned-cv-diagnostic:{sample_index}:{coordinate}")
                             for coordinate in range(len(queues)))
                rng_wall = perf_counter() - rng_started
                call_started = perf_counter()
                selected = select_control_variate_rollout_frame(
                    model.options, queues, model.lanes, DIAGNOSTIC_BUDGET,
                    arrival_rates=rates, terminal_service_rates=mu, arrival_rngs=rngs,
                )
                selector_wall = perf_counter() - call_started
                for work in (state_work, total_work):
                    for output_key, selected_key in _WORK_FIELDS.items():
                        work[output_key] += getattr(selected, selected_key)
                    work["selector_calls"] += 1
                    work["rng_initializations"] += len(rngs)
                    work["selector_wall_s"] += selector_wall
                    work["rng_setup_wall_s"] += rng_wall
                if sample_index == 0:
                    active = selected.rollout_used
                    if not active:
                        break
                raw_margin = selected.raw_rollout_margin
                corrected_margin = selected.corrected_rollout_margin
                sampled_control_margin = (selected.analytic_challenger_sampled_control_value
                                          - selected.incumbent_sampled_control_value)
                analytic_margin = (selected.root_lookahead_cost
                                   - selected.support_incumbent_lookahead_cost)
                identity_error = corrected_margin - (raw_margin - sampled_control_margin + analytic_margin)
                maximum_identity_error = max(maximum_identity_error, abs(identity_error))
                samples.append({
                    "sample_index": sample_index, "raw_margin": raw_margin,
                    "corrected_margin": corrected_margin,
                    "sampled_control_margin": sampled_control_margin,
                    "analytic_margin": analytic_margin, "identity_error": identity_error,
                })
            states.append({
                "load": load, "queue": list(queues), "active": active,
                **_sample_summary(samples), "samples": samples, "work": state_work,
            })
    identity_holds = maximum_identity_error <= IDENTITY_TOLERANCE
    return {
        "diagnostic": "fixed_queue_paired_control_variate_variance_v1",
        "status": "PASS" if identity_holds else "FAIL",
        "checks": {"control_variate_identity_holds": identity_holds},
        "budget": DIAGNOSTIC_BUDGET, "loads": list(loads),
        "queues": [list(queue) for queue in DIAGNOSTIC_QUEUES],
        "samples_per_active_state": DIAGNOSTIC_SAMPLES,
        "rng_namespace": "aligned-cv-diagnostic:{sample_index}:{coordinate}",
        "margin_sign": "analytic challenger minus support incumbent, before selection",
        "identity_tolerance": IDENTITY_TOLERANCE,
        "maximum_identity_error": maximum_identity_error,
        "active_state_count": sum(state["active"] for state in states),
        "inactive_state_count": sum(not state["active"] for state in states),
        "states": states, "work": total_work, "diagnostic_wall_s": perf_counter() - started,
        "limitations": [
            "Within-state sample variance and paired mean-change MC standard error are descriptive; variance reduction is not an execution requirement.",
            "Sample indices share RNG initializations across queues and loads, so states are not independent replicates and their margins are not pooled.",
            "Every repeated current search, successor construction and control-variate coordinate evaluation is charged; native total counters include separately reported subcounts.",
            "The diagnostic uses no real arrival tape and does not measure complete-policy flow time; inactive states have one charged probe and no sampled margins.",
        ],
    }
