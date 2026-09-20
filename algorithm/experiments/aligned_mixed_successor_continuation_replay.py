"""Locate matched-sample action disagreements and replay one intervention.

The frozen source recorder owns the only RNG stream that affects its trajectory.
Opposite-policy calls use copied pre-call states and are charged offline. Every
physical disagreement is replayed under the same deterministic v4 continuation.
"""
from __future__ import annotations

from itertools import accumulate
import json
import random
from time import perf_counter

from algorithm.mixed_successor_rollout_frame_candidates import select_mixed_successor_rollout_frame
from algorithm.poisson_rollout_frame_candidates import select_poisson_rollout_frame
from algorithm.experiments.aligned_hardware_control_variate import _assert_control_reproduced
from algorithm.experiments.aligned_hardware_control_variate_diagnostic import _WORK_FIELDS
from algorithm.experiments.aligned_hardware_mixed_successor_rollout import (
    MIXED_COUNTERS, MixedSuccessorRolloutRecorder,
)
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_frame_mechanism import action_description, action_signature
from algorithm.experiments.aligned_hardware_lookahead_continuation import (
    _checked_run, action_audit, paired_effect,
)
from algorithm.experiments.aligned_hardware_poisson_rollout import PoissonRolloutRecorder
from algorithm.experiments.aligned_lookahead_continuation import DecisionPairs, PrefixReplay


RAW_POLICY = "poisson_rollout_cap_1pct"
MIXED_POLICY = "mixed_successor_rollout_cap_1pct"
SOURCE_POLICIES = (RAW_POLICY, MIXED_POLICY)

# Totals include the named native and mixed subcounts; do not add aliases again.
WORK_FIELDS = {
    **{key: value for key, value in _WORK_FIELDS.items()
       if key != "cv_terminal_value_coordinate_evaluations"},
    **{key: key for key in MIXED_COUNTERS},
}
SHARED_FIELDS = (
    "attempts", "rollout_attempts", "root_budget", "root_attempts",
    "initial_root_budget", "initial_root_attempts", "recycled_candidate_attempts",
    "seed_support", "support_floor", "support_incumbent_score",
    "analytic_challenger_score", "root_lookahead_cost",
    "support_incumbent_lookahead_cost", "eligible_candidate_count",
    "cost_candidate_evaluations", "rollout_horizon", "rollout_population_jobs",
    "rollout_random_draws", "rollout_arrival_event_generations",
)


def _empty_work():
    return {**dict.fromkeys(WORK_FIELDS, 0), "selector_calls": 0,
            "rng_initializations": 0, "selector_wall_s": 0.0, "rng_setup_wall_s": 0.0}


class DisagreementDiscovery:
    """Preserve native decisions and compare only active, matched planning draws."""

    def __init__(self, model, rates, service_rates, planning_seed, source_policy):
        if source_policy not in SOURCE_POLICIES:
            raise ValueError("disagreement sources must be frozen Poisson or mixed-successor policies")
        self.model = model
        self.rates, self.service_rates = rates, service_rates
        self.source_policy = source_policy
        recorder = PoissonRolloutRecorder if source_policy == RAW_POLICY else MixedSuccessorRolloutRecorder
        self.native = recorder(rates, service_rates, planning_seed)
        self.prefix, self.switches = [], []
        self.shadow_work = dict(_empty_work(), rng_state_reads=0, rng_state_writes=0,
                                physical_action_descriptions=0,
                                comparison_completion_offset_reads=0)

    @property
    def frames(self):
        return self.native.frames

    def __call__(self, options, queues, lanes, budget):
        states = tuple(rng.getstate() for rng in self.native.arrival_rngs)
        self.shadow_work["rng_state_reads"] += len(states)
        frame_index, start = len(self.prefix), self.native.now
        selected = self.native(options, queues, lanes, budget)
        self.prefix.append((tuple(queues), selected))
        if not selected.rollout_used:
            return selected

        started = perf_counter()
        shadow_rngs = tuple(random.Random(0) for _ in states)
        for rng, state in zip(shadow_rngs, states, strict=True):
            rng.setstate(state)
        self.shadow_work["rng_setup_wall_s"] += perf_counter() - started
        self.shadow_work["rng_initializations"] += len(states)
        self.shadow_work["rng_state_writes"] += len(states)
        shadow_selector = (select_mixed_successor_rollout_frame if self.source_policy == RAW_POLICY
                           else select_poisson_rollout_frame)
        started = perf_counter()
        shadow = shadow_selector(options, queues, lanes, budget,
                                 arrival_rates=self.rates,
                                 terminal_service_rates=self.service_rates,
                                 arrival_rngs=shadow_rngs)
        self.shadow_work["selector_wall_s"] += perf_counter() - started
        self.shadow_work["selector_calls"] += 1
        for output_key, selected_key in WORK_FIELDS.items():
            self.shadow_work[output_key] += getattr(shadow, selected_key, 0)

        raw, mixed = (selected, shadow) if self.source_policy == RAW_POLICY else (shadow, selected)
        native_end = tuple(rng.getstate() for rng in self.native.arrival_rngs)
        shadow_end = tuple(rng.getstate() for rng in shadow_rngs)
        self.shadow_work["rng_state_reads"] += 2 * len(states)
        if (not shadow.rollout_used or native_end != shadow_end
                or any(getattr(raw, key) != getattr(mixed, key) for key in SHARED_FIELDS)):
            raise ValueError("shadow comparison changed the frozen candidates or sampled path")
        descriptions = [action_description(self.model, action) for action in (raw, mixed)]
        self.shadow_work["physical_action_descriptions"] += 2
        self.shadow_work["comparison_completion_offset_reads"] += sum(
            len(offsets) for description in descriptions for offsets in description["completion_offsets"])
        if action_signature(descriptions[0]) != action_signature(descriptions[1]):
            if raw.rollout_vetoed == mixed.rollout_vetoed:
                raise ValueError("physical disagreement does not identify opposite frozen contenders")
            orientation = 1 if raw.rollout_vetoed else -1
            self.switches.append({
                "frame_index": frame_index, "start_s": start, "queue": tuple(queues),
                "raw_action": raw, "mixed_action": mixed, "orientation_mixed_minus_raw": orientation,
                "raw_rollout_margin": raw.analytic_challenger_rollout_cost - raw.incumbent_rollout_cost,
                "mixed_rollout_margin": mixed.analytic_challenger_rollout_cost - mixed.incumbent_rollout_cost,
                "incumbent_analytic_cost": raw.support_incumbent_lookahead_cost,
                "challenger_analytic_cost": raw.root_lookahead_cost,
                "analytic_margin": raw.root_lookahead_cost - raw.support_incumbent_lookahead_cost,
                "planning_random_draws": raw.rollout_random_draws,
            })
        return selected


def run_source(task):
    """Replay every physical disagreement from one exactly reproduced source.

    task = (model, certificate, load, seed, planning_seed, source_policy,
            protocol, original_frozen_run_row)
    All case effect signs are mixed-first minus raw-first, in job-seconds.
    """
    model, certificate, load, seed, planning_seed, source_policy, protocol, original = task
    if protocol["budget"] != 26:
        raise ValueError("this diagnosis uses the frozen K26 continuation")
    if any(original[key] != value for key, value in (
            ("load", load), ("seed", seed), ("budget", 26),
            ("planning_seed", planning_seed), ("policy", source_policy))):
        raise ValueError("frozen source identity differs from the requested trajectory")
    horizon = protocol["arrival_horizon_s"]
    rates, mu = certificate["arrival_rates"], protocol["terminal_service_rates"]
    tape = poisson_arrivals(rates, horizon, seed)
    discovery = DisagreementDiscovery(model, rates, mu, planning_seed, source_policy)
    source = {
        "load": load, "seed": seed, "budget": 26,
        "policy": source_policy, "planning_seed": planning_seed,
        **_checked_run(model, tape, horizon, certificate, discovery),
        "selector_accounting": discovery.native.accounting(),
    }
    _assert_control_reproduced(source, original)
    identity = {"load": load, "seed": seed, "planning_seed": planning_seed,
                "source_policy": source_policy}
    print(json.dumps({**identity, "disagreements": len(discovery.switches)}), flush=True)

    times = tuple(t for t, _, _ in tape)
    sums = tuple(accumulate(times, initial=0.0))
    class_times = [tuple(t for t, c, _ in tape if c == coordinate)
                   for coordinate in range(len(model.workloads))]
    continuation = DecisionPairs(model, rates, mu, budget=26)
    cases, replay_frames = [], 0
    audit_offset_reads = 0
    for switch in discovery.switches:
        index, start, queue = switch["frame_index"], switch["start_s"], switch["queue"]
        actions = (switch["raw_action"], switch["mixed_action"])
        audits = [action_audit(model, queue, action, start, horizon, rates, mu,
                               times, sums, class_times) for action in actions]
        descriptions = [action_description(model, action) for action in actions]
        # Each audit reads offsets for both expected and realized current area.
        audit_offset_reads += 2 * sum(len(offsets) for description in descriptions
                                      for offsets in description["completion_offsets"])
        branches = []
        for action in actions:
            recorder = PrefixReplay(discovery.prefix, index, action, continuation.support)
            branches.append(_checked_run(model, tape, horizon, certificate, recorder))
            replay_frames += len(recorder.frames)
        full_effect = paired_effect(*branches, audits, len(tape))
        if (abs(full_effect["residual_identity_error_job_s"]) > 1e-5
                or abs(full_effect["flow_component_identity_error_job_s"]) > 1e-5):
            raise ValueError("mixed-minus-raw paired flow area decomposition failed")
        effect = {key: full_effect[key] for key in (
            "realized_total_delta_job_s", "mean_flow_delta_s", "realized_current_delta_job_s",
            "realized_future_delta_job_s", "flow_component_deltas_job_s",
            "flow_component_identity_error_job_s", "outcome",
        )}
        orientation = switch["orientation_mixed_minus_raw"]
        incumbent, challenger = descriptions if orientation == 1 else descriptions[::-1]
        cases.append({
            **{key: value for key, value in switch.items() if key not in ("raw_action", "mixed_action")},
            "phase": ("drain" if start >= horizon else "crosses_cutoff"
                      if start + max(action.duration for action in actions) > horizon else "arrival_window"),
            "raw_action": descriptions[0], "mixed_action": descriptions[1],
            "incumbent_action": incumbent, "challenger_action": challenger,
            "oriented_raw_sample_delta_job_s": orientation * switch["raw_rollout_margin"],
            "oriented_mixed_sample_delta_job_s": orientation * switch["mixed_rollout_margin"],
            "realized_current_area_job_s": [audit["realized_current_area"] for audit in audits],
            "realized_next_queues": [audit["realized_next_queue"] for audit in audits],
            "branch_mean_flow_s": [branch["metrics"]["mean_flow_s"] for branch in branches],
            "effect": effect,
        })
        if len(cases) % 25 == 0:
            print(json.dumps({**identity, "pairs_completed": len(cases)}), flush=True)
    return {
        **identity, "job_count": len(tape), "source_frame_count": len(discovery.frames),
        "source": source, "source_reproduced": True,
        "substitution_count": len(discovery.switches), "paired_replay_count": len(cases),
        "cases": cases,
        "diagnostic_work": {
            "shadow": discovery.shadow_work, "continuation": continuation.accounting(),
            "action_audit_evaluations": 2 * len(cases),
            "action_audit_completion_offset_reads": audit_offset_reads,
            "replay_action_descriptions": 2 * len(cases),
            "replay_action_description_completion_offset_reads": audit_offset_reads // 2,
            "replay_runs": 2 * len(cases), "replay_frames": replay_frames,
        },
    }
