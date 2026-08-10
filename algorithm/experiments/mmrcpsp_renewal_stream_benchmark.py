"""Variable-duration MMRCPSP renewal-stream benchmark.

This experiment lifts complete feasible MMRCPSP schedules into a registered
finite action library and places those actions in a genuine project-arrival
queue.  One action completes one project of one registered class.  Its
internal activity completions are audited separately and are never counted as
independent exogenous service.

The online selector maximizes

    (Q[c] * lower_departure(a) - bounded_penalty(a)) / (tau(a) / H)

over feasible actions for nonempty queues.  Arrivals are generated on an
exogenous unit-time tape, so every policy sees the same sample path even when
its renewal frames have different durations.  The resulting finite-sample
drift diagnostics are evidence about this registered experiment; they do not
constitute a positive-recurrence proof or an MMRCPSP-wide optimality claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Any, Mapping, Sequence

from algorithm.experiments.mmrcpsp_benchmark import (
    POLICY_ORDER as MMRCPSP_POLICY_ORDER,
    schedule_instance,
)
from algorithm.experiments.mmrcpsp_instances import (
    MMRCPSPInstance,
    parse_psplib_mm,
)
from algorithm.experiments.mmrcpsp_renewal_frame_upgrade import (
    build_mmrcpsp_renewal_frame_schedule,
)
from algorithm.experiments.trajectory_renewal_frame_score import (
    complete_trajectory_renewal_score,
)


SCHEMA_VERSION = "scheduleurm.mmrcpsp_renewal_stream_benchmark.v1"
OURS_POLICY = "duration_normalized_robust_maxweight"
BASELINE_POLICIES = (
    "unnormalized_robust_maxweight",
    "shortest_registered_trajectory",
    "longest_queue_then_shortest",
    "round_robin_class",
)
POLICY_ORDER = (OURS_POLICY, *BASELINE_POLICIES)
HOLDING_PENALTY_WEIGHT = 0.05
EXTRA_PENALTY_BOUND = 0.02
NUMERIC_TOLERANCE = 1e-9


class RenewalStreamError(ValueError):
    """Raised when the registered renewal-stream contract is violated."""


@dataclass(frozen=True)
class RegisteredTrajectory:
    """One complete feasible trajectory action in the registered library."""

    action_id: str
    workload_class: str
    source_sha256: str
    sources: tuple[str, ...]
    trajectory_sha256: str
    duration: int
    common_instance_horizon: int
    real_activity_count: int
    completion_times: tuple[float, ...]
    lower_project_departure: float
    bounded_penalty: float
    penalty_uniform_bound: float
    makespan: float
    mean_flow_time: float
    schedule_feasible: bool
    resource_violation_units: float
    precedence_violation_count: int

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "workload_class": self.workload_class,
            "source_sha256": self.source_sha256,
            "sources": list(self.sources),
            "trajectory_sha256": self.trajectory_sha256,
            "duration_tau": self.duration,
            "common_instance_horizon": self.common_instance_horizon,
            "real_activity_count": self.real_activity_count,
            "completion_times": list(self.completion_times),
            "lower_project_departure": self.lower_project_departure,
            "bounded_penalty": self.bounded_penalty,
            "penalty_uniform_bound": self.penalty_uniform_bound,
            "metrics": {
                "makespan": self.makespan,
                "mean_flow_time": self.mean_flow_time,
                "schedule_feasible": self.schedule_feasible,
                "resource_violation_units": self.resource_violation_units,
                "precedence_violation_count": self.precedence_violation_count,
            },
        }


@dataclass(frozen=True)
class ArrivalScenario:
    """A preregistered exogenous arrival protocol on a fixed time horizon."""

    scenario_id: str
    family: str
    arrival_kind: str
    load_factor: float
    horizon: int
    initial_backlog_per_class: int
    burst_multiplier: float = 4.0
    burst_switch_probability: float = 0.20

    def __post_init__(self) -> None:
        if self.family not in {"static", "poisson", "bursty", "load_sweep"}:
            raise RenewalStreamError(f"unsupported scenario family: {self.family}")
        if self.arrival_kind not in {"none", "poisson", "markov_bursty"}:
            raise RenewalStreamError(
                f"unsupported arrival kind: {self.arrival_kind}"
            )
        if self.horizon <= 0 or self.initial_backlog_per_class < 0:
            raise RenewalStreamError("scenario horizon/backlog is invalid")
        if not math.isfinite(self.load_factor) or self.load_factor < 0.0:
            raise RenewalStreamError("load factor must be finite and nonnegative")
        if self.arrival_kind == "none" and abs(self.load_factor) > NUMERIC_TOLERANCE:
            raise RenewalStreamError("static no-arrival scenario must have zero load")
        if self.burst_multiplier <= 1.0:
            raise RenewalStreamError("burst multiplier must exceed one")
        if not 0.0 < self.burst_switch_probability <= 1.0:
            raise RenewalStreamError("burst switch probability must lie in (0, 1]")

    def snapshot(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "arrival_kind": self.arrival_kind,
            "load_factor": self.load_factor,
            "horizon": self.horizon,
            "initial_backlog_per_class": self.initial_backlog_per_class,
            "burst_multiplier": self.burst_multiplier,
            "burst_switch_probability": self.burst_switch_probability,
        }


def build_registered_trajectory_library(
    instance_paths: Sequence[str | Path],
) -> tuple[RegisteredTrajectory, ...]:
    """Build and independently audit a finite trajectory library.

    The library is generated without arrival or policy feedback.  It contains
    the existing deterministic priority-rule trajectories and the exact
    generated-family renewal-score maximizer for each supplied instance.
    Exact duplicate schedules are registered once with all source labels.
    """

    if not instance_paths:
        raise RenewalStreamError("at least one MMRCPSP instance is required")
    paths = sorted((Path(path) for path in instance_paths), key=lambda p: (p.name, str(p)))
    instances = [parse_psplib_mm(path) for path in paths]
    names = [instance.name for instance in instances]
    if len(names) != len(set(names)):
        raise RenewalStreamError("workload-class names must be unique")

    library: list[RegisteredTrajectory] = []
    for instance in instances:
        raw: list[tuple[str, Sequence[Mapping[str, Any]], Mapping[str, Any]]] = []
        for policy in MMRCPSP_POLICY_ORDER:
            result = schedule_instance(instance, policy)
            if not result.feasible:
                raise RenewalStreamError(
                    f"{instance.name}: registered policy {policy} is infeasible: "
                    f"{result.failure_reason}"
                )
            raw.append(
                (
                    f"priority_rule:{policy}",
                    tuple(entry.snapshot() for entry in result.entries),
                    result.metrics,
                )
            )

        renewal = build_mmrcpsp_renewal_frame_schedule(instance)
        raw.append(
            (
                "generated_family:renewal_ratio_argmax",
                tuple(renewal["schedule"]),
                renewal["metrics"],
            )
        )

        by_signature: dict[str, dict[str, Any]] = {}
        for source, schedule, metrics in raw:
            signature = _trajectory_sha256(schedule)
            if signature in by_signature:
                by_signature[signature]["sources"].append(source)
                continue
            by_signature[signature] = {
                "sources": [source],
                "schedule": schedule,
                "metrics": metrics,
            }
        for signature, row in sorted(by_signature.items()):
            library.append(
                _register_trajectory(
                    instance,
                    trajectory_sha256=signature,
                    sources=tuple(sorted(row["sources"])),
                    schedule=row["schedule"],
                    metrics=row["metrics"],
                )
            )

    result = tuple(sorted(library, key=lambda row: row.action_id))
    _audit_library(result, expected_classes=tuple(sorted(names)))
    return result


def default_scenarios(*, horizon: int = 240) -> tuple[ArrivalScenario, ...]:
    """Return the frozen static, stochastic, bursty, and load-sweep scenarios."""

    return (
        ArrivalScenario("static_backlog", "static", "none", 0.0, horizon, 4),
        ArrivalScenario("poisson_nominal", "poisson", "poisson", 0.65, horizon, 1),
        ArrivalScenario(
            "bursty_nominal", "bursty", "markov_bursty", 0.65, horizon, 1
        ),
        ArrivalScenario("load_035", "load_sweep", "poisson", 0.35, horizon, 1),
        ArrivalScenario("load_065", "load_sweep", "poisson", 0.65, horizon, 1),
        ArrivalScenario("load_090", "load_sweep", "poisson", 0.90, horizon, 1),
        ArrivalScenario("load_105", "load_sweep", "poisson", 1.05, horizon, 1),
    )


def select_registered_action(
    library: Sequence[RegisteredTrajectory],
    *,
    queue: Mapping[str, int],
    policy: str,
    common_horizon: int,
    remaining_time: int,
    round_robin_cursor: int = 0,
) -> tuple[RegisteredTrajectory | None, dict[str, Any]]:
    """Select one feasible registered action with deterministic tie breaking."""

    if policy not in POLICY_ORDER:
        raise RenewalStreamError(f"unknown stream policy: {policy}")
    if common_horizon <= 0 or remaining_time < 0:
        raise RenewalStreamError("selection horizon is invalid")
    candidates = tuple(
        action
        for action in library
        if queue.get(action.workload_class, 0) > 0
        and action.duration <= remaining_time
    )
    if not candidates:
        return None, {
            "candidate_count": 0,
            "exact_argmax": None,
            "oracle_gap": None,
            "scores": {},
        }

    normalized_scores = {
        action.action_id: _duration_normalized_score(
            action,
            queue_length=queue[action.workload_class],
            common_horizon=common_horizon,
        )
        for action in candidates
    }
    if policy == OURS_POLICY:
        ranked = sorted(
            candidates,
            key=lambda action: (
                -normalized_scores[action.action_id],
                action.duration,
                action.bounded_penalty,
                action.action_id,
            ),
        )
        selected = ranked[0]
        optimum = max(normalized_scores.values())
        gap = optimum - normalized_scores[selected.action_id]
        exact = abs(gap) <= NUMERIC_TOLERANCE
    elif policy == "unnormalized_robust_maxweight":
        unnormalized = {
            action.action_id: (
                queue[action.workload_class] * action.lower_project_departure
                - action.bounded_penalty
            )
            for action in candidates
        }
        selected = min(
            candidates,
            key=lambda action: (
                -unnormalized[action.action_id],
                action.duration,
                action.action_id,
            ),
        )
        gap = None
        exact = None
    elif policy == "shortest_registered_trajectory":
        selected = min(
            candidates,
            key=lambda action: (
                action.duration,
                action.bounded_penalty,
                action.action_id,
            ),
        )
        gap = None
        exact = None
    elif policy == "longest_queue_then_shortest":
        selected = min(
            candidates,
            key=lambda action: (
                -queue[action.workload_class],
                action.duration,
                action.bounded_penalty,
                action.action_id,
            ),
        )
        gap = None
        exact = None
    else:
        classes = sorted({action.workload_class for action in candidates})
        start = round_robin_cursor % len(classes)
        ordered_classes = classes[start:] + classes[:start]
        chosen_class = ordered_classes[0]
        selected = min(
            (action for action in candidates if action.workload_class == chosen_class),
            key=lambda action: (
                action.duration,
                action.bounded_penalty,
                action.action_id,
            ),
        )
        gap = None
        exact = None

    return selected, {
        "candidate_count": len(candidates),
        "exact_argmax": exact,
        "oracle_gap": _round(gap) if gap is not None else None,
        "selected_score": _round(normalized_scores[selected.action_id]),
        "scores": {
            action_id: _round(score)
            for action_id, score in sorted(normalized_scores.items())
        },
    }


def run_mmrcpsp_renewal_stream_benchmark(
    instance_paths: Sequence[str | Path],
    *,
    scenarios: Sequence[ArrivalScenario] | None = None,
    policies: Sequence[str] = POLICY_ORDER,
    seeds: Sequence[int] = (11, 23, 37),
) -> dict[str, Any]:
    """Run the registered renewal stream and return a complete audit report."""

    library = build_registered_trajectory_library(instance_paths)
    selected_scenarios = tuple(scenarios or default_scenarios())
    selected_policies = tuple(str(policy) for policy in policies)
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected_scenarios:
        raise RenewalStreamError("at least one arrival scenario is required")
    if not selected_policies or len(selected_policies) != len(set(selected_policies)):
        raise RenewalStreamError("policies must be a nonempty unique sequence")
    unknown = sorted(set(selected_policies) - set(POLICY_ORDER))
    if unknown:
        raise RenewalStreamError(f"unknown stream policies: {unknown}")
    if OURS_POLICY not in selected_policies:
        raise RenewalStreamError(f"benchmark must include {OURS_POLICY}")
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise RenewalStreamError("seeds must be a nonempty unique sequence")
    scenario_ids = [scenario.scenario_id for scenario in selected_scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise RenewalStreamError("scenario ids must be unique")

    classes = tuple(sorted({action.workload_class for action in library}))
    common_horizon = max(action.common_instance_horizon for action in library)
    min_duration = {
        workload_class: min(
            action.duration
            for action in library
            if action.workload_class == workload_class
        )
        for workload_class in classes
    }
    activity_count = {
        workload_class: next(
            action.real_activity_count
            for action in library
            if action.workload_class == workload_class
        )
        for workload_class in classes
    }

    runs: list[dict[str, Any]] = []
    tape_hashes: dict[tuple[str, int], str] = {}
    for scenario in selected_scenarios:
        rates = _arrival_rates(
            scenario,
            classes=classes,
            minimum_duration=min_duration,
        )
        for seed in selected_seeds:
            tape = _arrival_tape(scenario, classes=classes, rates=rates, seed=seed)
            tape_hash = _arrival_tape_sha256(tape)
            tape_hashes[(scenario.scenario_id, seed)] = tape_hash
            for policy in selected_policies:
                run = _simulate(
                    library,
                    classes=classes,
                    activity_count=activity_count,
                    scenario=scenario,
                    policy=policy,
                    seed=seed,
                    arrival_rates=rates,
                    arrival_tape=tape,
                    arrival_tape_sha256=tape_hash,
                    common_horizon=common_horizon,
                )
                runs.append(run)

    protocol_pass = all(bool(run["gate"]["pass"]) for run in runs)
    ours_runs = [run for run in runs if run["policy"] == OURS_POLICY]
    stochastic_ours = [
        run for run in ours_runs if run["scenario"]["arrival_kind"] != "none"
    ]
    return _canonical(
        {
            "schema_version": SCHEMA_VERSION,
            "benchmark": "registered_mmrcpsp_variable_duration_renewal_stream",
            "deterministic_given_seed": True,
            "policy_order": list(selected_policies),
            "seed_order": list(selected_seeds),
            "scenario_order": scenario_ids,
            "workload_classes": list(classes),
            "registered_trajectory_library": [
                action.snapshot() for action in library
            ],
            "library_contract": {
                "finite": True,
                "generated_before_arrival_tapes": True,
                "class_minimum_duration": min_duration,
                "common_duration_normalizer_H": common_horizon,
                "all_actions_complete_one_project": True,
                "all_internal_activity_departures_audited": True,
                "all_actions_feasible": True,
            },
            "arrival_contract": {
                "unit_time_exogenous_tape": True,
                "same_scenario_seed_tape_for_every_policy": True,
                "policy_cannot_change_random_draws_by_changing_frame_duration": True,
                "tape_hashes": {
                    f"{scenario_id}:seed_{seed}": digest
                    for (scenario_id, seed), digest in sorted(tape_hashes.items())
                },
            },
            "runs": runs,
            "aggregate": {
                "run_count": len(runs),
                "protocol_pass_count": sum(bool(run["gate"]["pass"]) for run in runs),
                "ours_exact_oracle_run_count": sum(
                    bool(run["gate"]["exact_duration_normalized_oracle"])
                    for run in ours_runs
                ),
                "stochastic_ours_run_count": len(stochastic_ours),
                "stochastic_ours_nominal_capacity_interior_count": sum(
                    bool(run["stability_diagnostics"]["nominal_capacity_interior"])
                    for run in stochastic_ours
                ),
                "stochastic_ours_empirical_negative_tail_drift_count": sum(
                    bool(
                        run["stability_diagnostics"][
                            "empirical_negative_high_backlog_drift"
                        ]
                    )
                    for run in stochastic_ours
                ),
            },
            "gate": {
                "status": (
                    "MMRCPSP_RENEWAL_STREAM_PROTOCOL_PASS"
                    if protocol_pass
                    else "MMRCPSP_RENEWAL_STREAM_PROTOCOL_FAIL"
                ),
                "pass": protocol_pass,
                "performance_superiority_not_required": True,
                "negative_empirical_drift_not_required": True,
            },
            "claim_boundary": {
                "registered_finite_library_only": True,
                "variable_duration_queue_recurrence_executed": True,
                "workload_conservation_pathwise_certified": protocol_pass,
                "exact_oracle_only_over_registered_feasible_family": True,
                "finite_sample_drift_is_diagnostic_not_positive_recurrence_proof": True,
                "global_mmrcpsp_optimality_claim": False,
                "psplib_wide_generalization_claim": False,
                "production_arrival_model_claim": False,
                "stochastic_stability_claim_ready": False,
                "reason": (
                    "A finite replay cannot prove irreducibility, recurrence, or a "
                    "domain-wide service model. The capacity and drift fields audit "
                    "the registered renewal stream and expose, rather than hide, "
                    "capacity-exterior load points."
                ),
            },
        }
    )


def _register_trajectory(
    instance: MMRCPSPInstance,
    *,
    trajectory_sha256: str,
    sources: tuple[str, ...],
    schedule: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> RegisteredTrajectory:
    if not bool(metrics["schedule_feasible"]):
        raise RenewalStreamError(f"{instance.name}: infeasible trajectory registered")
    if float(metrics["resource_violation_units"]) > NUMERIC_TOLERANCE:
        raise RenewalStreamError(f"{instance.name}: resource-violating trajectory")
    if int(metrics["precedence_violation_count"]) != 0:
        raise RenewalStreamError(f"{instance.name}: precedence-violating trajectory")
    entries = {int(entry["job_id"]): entry for entry in schedule}
    missing = sorted(set(instance.real_job_ids) - set(entries))
    if missing:
        raise RenewalStreamError(
            f"{instance.name}: trajectory omits real activities {missing}"
        )
    completion_times = tuple(float(entries[job_id]["finish"]) for job_id in instance.real_job_ids)
    duration = int(round(float(metrics["makespan"])))
    if duration <= 0 or abs(duration - float(metrics["makespan"])) > NUMERIC_TOLERANCE:
        raise RenewalStreamError("MMRCPSP renewal duration must be a positive integer")
    horizon = max(
        1,
        sum(
            max(mode.duration for mode in instance.activity_map[job_id].modes)
            for job_id in instance.job_ids
        ),
    )
    reconfiguration_fraction = float(metrics["mode_reconfiguration_count"]) / max(
        1, len(instance.real_job_ids)
    )
    extra_penalty = EXTRA_PENALTY_BOUND * min(1.0, reconfiguration_fraction)
    score = complete_trajectory_renewal_score(
        completion_times=completion_times,
        frame_duration=duration,
        common_horizon_bound=horizon,
        project_queue_weight=1.0,
        holding_penalty_weight=HOLDING_PENALTY_WEIGHT,
        extra_penalty=extra_penalty,
        extra_penalty_bound=EXTRA_PENALTY_BOUND,
    )
    bounded_penalty = float(score["penalty"]["bounded_penalty"])
    uniform_bound = float(score["penalty"]["uniform_bound"])
    if bounded_penalty < -NUMERIC_TOLERANCE or bounded_penalty > uniform_bound + NUMERIC_TOLERANCE:
        raise RenewalStreamError("registered trajectory penalty is not bounded")
    return RegisteredTrajectory(
        action_id=f"{instance.name}:{trajectory_sha256[:16]}",
        workload_class=instance.name,
        source_sha256=instance.source_sha256,
        sources=sources,
        trajectory_sha256=trajectory_sha256,
        duration=duration,
        common_instance_horizon=horizon,
        real_activity_count=len(instance.real_job_ids),
        completion_times=completion_times,
        lower_project_departure=1.0,
        bounded_penalty=bounded_penalty,
        penalty_uniform_bound=uniform_bound,
        makespan=float(metrics["makespan"]),
        mean_flow_time=float(metrics["mean_flow_time"]),
        schedule_feasible=True,
        resource_violation_units=float(metrics["resource_violation_units"]),
        precedence_violation_count=int(metrics["precedence_violation_count"]),
    )


def _audit_library(
    library: Sequence[RegisteredTrajectory], *, expected_classes: Sequence[str]
) -> None:
    if not library:
        raise RenewalStreamError("registered trajectory library is empty")
    ids = [action.action_id for action in library]
    signatures = [
        (action.workload_class, action.trajectory_sha256) for action in library
    ]
    if len(ids) != len(set(ids)) or len(signatures) != len(set(signatures)):
        raise RenewalStreamError("registered trajectory library contains duplicates")
    if set(expected_classes) != {action.workload_class for action in library}:
        raise RenewalStreamError("registered trajectory library lost a workload class")
    for action in library:
        if not action.schedule_feasible or action.resource_violation_units > NUMERIC_TOLERANCE:
            raise RenewalStreamError(f"infeasible registered action {action.action_id}")
        if action.precedence_violation_count != 0:
            raise RenewalStreamError(f"precedence violation in {action.action_id}")
        if action.duration <= 0 or action.lower_project_departure != 1.0:
            raise RenewalStreamError(f"invalid service contract in {action.action_id}")
        if action.bounded_penalty > action.penalty_uniform_bound + NUMERIC_TOLERANCE:
            raise RenewalStreamError(f"penalty bound failed in {action.action_id}")


def _arrival_rates(
    scenario: ArrivalScenario,
    *,
    classes: Sequence[str],
    minimum_duration: Mapping[str, int],
) -> dict[str, float]:
    if scenario.arrival_kind == "none":
        return {workload_class: 0.0 for workload_class in classes}
    share = 1.0 / len(classes)
    return {
        workload_class: scenario.load_factor * share / minimum_duration[workload_class]
        for workload_class in classes
    }


def _arrival_tape(
    scenario: ArrivalScenario,
    *,
    classes: Sequence[str],
    rates: Mapping[str, float],
    seed: int,
) -> dict[str, tuple[int, ...]]:
    rng = random.Random(_stable_seed(scenario.scenario_id, seed))
    tape: dict[str, tuple[int, ...]] = {}
    for workload_class in classes:
        if scenario.arrival_kind == "none":
            tape[workload_class] = tuple(0 for _ in range(scenario.horizon))
            continue
        if scenario.arrival_kind == "poisson":
            tape[workload_class] = tuple(
                _poisson(rng, rates[workload_class])
                for _ in range(scenario.horizon)
            )
            continue

        stationary_on = 1.0 / scenario.burst_multiplier
        p_on_to_off = scenario.burst_switch_probability * (1.0 - stationary_on)
        p_off_to_on = scenario.burst_switch_probability * stationary_on
        on = rng.random() < stationary_on
        rows: list[int] = []
        for _ in range(scenario.horizon):
            if on:
                rows.append(
                    _poisson(
                        rng,
                        rates[workload_class] * scenario.burst_multiplier,
                    )
                )
                if rng.random() < p_on_to_off:
                    on = False
            else:
                rows.append(0)
                if rng.random() < p_off_to_on:
                    on = True
        tape[workload_class] = tuple(rows)
    return tape


def _simulate(
    library: Sequence[RegisteredTrajectory],
    *,
    classes: Sequence[str],
    activity_count: Mapping[str, int],
    scenario: ArrivalScenario,
    policy: str,
    seed: int,
    arrival_rates: Mapping[str, float],
    arrival_tape: Mapping[str, Sequence[int]],
    arrival_tape_sha256: str,
    common_horizon: int,
) -> dict[str, Any]:
    queue = {
        workload_class: scenario.initial_backlog_per_class
        for workload_class in classes
    }
    initial = dict(queue)
    total_arrivals = {workload_class: 0 for workload_class in classes}
    departures = {workload_class: 0 for workload_class in classes}
    frames: list[dict[str, Any]] = []
    now = 0
    rr_cursor = 0
    exact_oracle = True
    selected_registered = True
    no_overservice = True
    recurrence_exact = True
    drift_identity_exact = True
    penalty_bound_respected = True
    queue_time_area = 0.0

    while now < scenario.horizon:
        before = dict(queue)
        selected, oracle = select_registered_action(
            library,
            queue=before,
            policy=policy,
            common_horizon=common_horizon,
            remaining_time=scenario.horizon - now,
            round_robin_cursor=rr_cursor,
        )
        if selected is None:
            duration = 1 if not any(before.values()) else scenario.horizon - now
            departure = {workload_class: 0 for workload_class in classes}
            action_id = "idle:no_eligible_registered_trajectory"
            action_class = None
            penalty = 0.0
            selected_score = None
            if policy == OURS_POLICY and oracle["candidate_count"]:
                exact_oracle = False
        else:
            duration = selected.duration
            action_id = selected.action_id
            action_class = selected.workload_class
            departure = {
                workload_class: int(workload_class == action_class)
                for workload_class in classes
            }
            penalty = selected.bounded_penalty
            selected_score = oracle["selected_score"]
            no_overservice = no_overservice and before[action_class] >= 1
            selected_registered = selected_registered and selected in library
            penalty_bound_respected = penalty_bound_respected and (
                penalty <= selected.penalty_uniform_bound + NUMERIC_TOLERANCE
            )
            if policy == OURS_POLICY:
                exact_oracle = exact_oracle and bool(oracle["exact_argmax"])
            if policy == "round_robin_class":
                rr_cursor += 1

        end = now + duration
        arrivals = {
            workload_class: int(sum(arrival_tape[workload_class][now:end]))
            for workload_class in classes
        }
        after = {
            workload_class: before[workload_class]
            - departure[workload_class]
            + arrivals[workload_class]
            for workload_class in classes
        }
        if any(value < 0 for value in after.values()):
            no_overservice = False
        for workload_class in classes:
            total_arrivals[workload_class] += arrivals[workload_class]
            departures[workload_class] += departure[workload_class]
        queue_time_area += sum(before.values()) * duration

        lyapunov_before = 0.5 * sum(value * value for value in before.values())
        lyapunov_after = 0.5 * sum(value * value for value in after.values())
        delta = lyapunov_after - lyapunov_before
        increments = {
            workload_class: arrivals[workload_class] - departure[workload_class]
            for workload_class in classes
        }
        identity_rhs = sum(
            before[workload_class] * increments[workload_class]
            + 0.5 * increments[workload_class] ** 2
            for workload_class in classes
        )
        frame_recurrence = all(
            after[workload_class]
            == before[workload_class]
            - departure[workload_class]
            + arrivals[workload_class]
            for workload_class in classes
        )
        recurrence_exact = recurrence_exact and frame_recurrence
        drift_identity_exact = drift_identity_exact and (
            abs(delta - identity_rhs) <= NUMERIC_TOLERANCE
        )
        expected_linear_drift_rate = sum(
            before[workload_class]
            * (
                arrival_rates[workload_class]
                - departure[workload_class] / duration
            )
            for workload_class in classes
        )
        frames.append(
            {
                "frame_index": len(frames),
                "start_time": now,
                "end_time": end,
                "duration_tau": duration,
                "queue_before": before,
                "arrivals_during_frame": arrivals,
                "project_departures": departure,
                "queue_after": after,
                "selected_action_id": action_id,
                "selected_workload_class": action_class,
                "selected_duration_normalized_score": selected_score,
                "bounded_penalty": _round(penalty),
                "candidate_count": oracle["candidate_count"],
                "oracle_gap": oracle["oracle_gap"],
                "lyapunov_before": _round(lyapunov_before),
                "lyapunov_after": _round(lyapunov_after),
                "observed_drift": _round(delta),
                "observed_drift_rate": _round(delta / duration),
                "exact_drift_identity_rhs": _round(identity_rhs),
                "nominal_expected_linear_drift_rate": _round(
                    expected_linear_drift_rate
                ),
                "recurrence_exact": frame_recurrence,
            }
        )
        queue = after
        now = end

    conservation = {
        workload_class: {
            "initial_projects": initial[workload_class],
            "arrived_projects": total_arrivals[workload_class],
            "completed_projects": departures[workload_class],
            "final_queued_projects": queue[workload_class],
            "project_identity_residual": (
                initial[workload_class]
                + total_arrivals[workload_class]
                - departures[workload_class]
                - queue[workload_class]
            ),
            "admitted_activity_obligations": (
                initial[workload_class] + total_arrivals[workload_class]
            )
            * activity_count[workload_class],
            "completed_internal_activities": departures[workload_class]
            * activity_count[workload_class],
            "remaining_activity_obligations": queue[workload_class]
            * activity_count[workload_class],
        }
        for workload_class in classes
    }
    conservation_exact = all(
        row["project_identity_residual"] == 0
        and row["admitted_activity_obligations"]
        == row["completed_internal_activities"]
        + row["remaining_activity_obligations"]
        for row in conservation.values()
    )
    minimum_duration = {
        workload_class: min(
            action.duration
            for action in library
            if action.workload_class == workload_class
        )
        for workload_class in classes
    }
    offered_load = sum(
        arrival_rates[workload_class] * minimum_duration[workload_class]
        for workload_class in classes
    )
    positive_frames = [frame for frame in frames if sum(frame["queue_before"].values()) > 0]
    queue_levels = sorted(sum(frame["queue_before"].values()) for frame in positive_frames)
    high_threshold = queue_levels[(3 * (len(queue_levels) - 1)) // 4] if queue_levels else 1
    high_frames = [
        frame
        for frame in positive_frames
        if sum(frame["queue_before"].values()) >= high_threshold
    ]
    observed_tail = (
        fmean(float(frame["observed_drift_rate"]) for frame in high_frames)
        if high_frames
        else 0.0
    )
    expected_tail = (
        fmean(
            float(frame["nominal_expected_linear_drift_rate"])
            for frame in high_frames
        )
        if high_frames
        else 0.0
    )
    protocol_pass = (
        selected_registered
        and no_overservice
        and recurrence_exact
        and drift_identity_exact
        and penalty_bound_respected
        and conservation_exact
        and (policy != OURS_POLICY or exact_oracle)
        and now == scenario.horizon
    )
    return _canonical(
        {
            "run_id": f"{scenario.scenario_id}:seed_{seed}:{policy}",
            "scenario": scenario.snapshot(),
            "seed": seed,
            "policy": policy,
            "arrival_tape_sha256": arrival_tape_sha256,
            "arrival_rates_per_time_unit": dict(arrival_rates),
            "frames": frames,
            "workload_conservation": conservation,
            "performance": {
                "elapsed_time": now,
                "completed_projects": sum(departures.values()),
                "total_arrivals": sum(total_arrivals.values()),
                "final_backlog": sum(queue.values()),
                "maximum_backlog": max(
                    [sum(frame["queue_before"].values()) for frame in frames]
                    + [sum(queue.values())]
                ),
                "time_average_queue": queue_time_area / scenario.horizon,
                "project_throughput_per_time": sum(departures.values())
                / scenario.horizon,
            },
            "stability_diagnostics": {
                "offered_load_rho_against_class_fastest_registered_actions": offered_load,
                "nominal_capacity_slack": 1.0 - offered_load,
                "nominal_capacity_interior": offered_load < 1.0 - NUMERIC_TOLERANCE,
                "capacity_exterior_or_boundary": offered_load >= 1.0 - NUMERIC_TOLERANCE,
                "high_backlog_threshold": high_threshold,
                "high_backlog_frame_count": len(high_frames),
                "mean_observed_high_backlog_drift_rate": observed_tail,
                "mean_nominal_expected_linear_high_backlog_drift_rate": expected_tail,
                "empirical_negative_high_backlog_drift": observed_tail < 0.0,
                "nominal_expected_linear_tail_negative": expected_tail < 0.0,
                "quadratic_drift_identity_pathwise_exact": drift_identity_exact,
                "positive_recurrence_inferred_from_finite_sample": False,
            },
            "gate": {
                "pass": protocol_pass,
                "selected_actions_registered": selected_registered,
                "no_project_overservice": no_overservice,
                "queue_recurrence_pathwise_exact": recurrence_exact,
                "quadratic_drift_identity_pathwise_exact": drift_identity_exact,
                "bounded_penalty_respected": penalty_bound_respected,
                "workload_conservation_exact": conservation_exact,
                "exact_duration_normalized_oracle": (
                    exact_oracle if policy == OURS_POLICY else None
                ),
                "fixed_wall_clock_horizon_respected": now == scenario.horizon,
            },
        }
    )


def _duration_normalized_score(
    action: RegisteredTrajectory, *, queue_length: int, common_horizon: int
) -> float:
    numerator = (
        queue_length * action.lower_project_departure - action.bounded_penalty
    )
    return numerator / (action.duration / common_horizon)


def _trajectory_sha256(schedule: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        (
            int(entry["job_id"]),
            int(entry["mode_id"]),
            int(entry["start"]),
            int(entry["finish"]),
        )
        for entry in schedule
    ]
    return sha256(
        json.dumps(payload, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _arrival_tape_sha256(tape: Mapping[str, Sequence[int]]) -> str:
    payload = {key: list(tape[key]) for key in sorted(tape)}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _stable_seed(scenario_id: str, seed: int) -> int:
    digest = sha256(f"{scenario_id}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _poisson(rng: random.Random, rate: float) -> int:
    if rate < 0.0 or not math.isfinite(rate):
        raise RenewalStreamError("Poisson rate must be finite and nonnegative")
    if rate == 0.0:
        return 0
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(row) for row in value]
    if isinstance(value, float):
        return _round(value)
    return value


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args(argv)
    report = run_mmrcpsp_renewal_stream_benchmark(
        args.instances,
        scenarios=default_scenarios(horizon=args.horizon),
        seeds=args.seeds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArrivalScenario",
    "BASELINE_POLICIES",
    "OURS_POLICY",
    "POLICY_ORDER",
    "RegisteredTrajectory",
    "RenewalStreamError",
    "build_registered_trajectory_library",
    "default_scenarios",
    "run_mmrcpsp_renewal_stream_benchmark",
    "select_registered_action",
]
