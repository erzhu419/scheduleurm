"""Theorem-aligned finite trajectory upgrade for deterministic MMRCPSP.

The core MMRCPSP sidecar selects one activity-mode action at a time.  This
module lifts the decision object to a complete, feasible configuration
trajectory.  A fixed, instance-independent construction protocol generates a
finite family from deterministic rollout rules, bounded beam searches, and a
fixed local neighborhood.  The selected action is the exact maximizer of

    Q^T lower_service(trajectory) - bounded_penalty(trajectory)

over that generated family.  The experiment is deliberately scoped: it is a
finite deterministic trajectory-action demonstration, not an exact solver for
the full MMRCPSP feasible set and not a stochastic stability experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.mmrcpsp_benchmark import (
    OURS_POLICY,
    ScheduleEntry,
    _evaluate_schedule,
    schedule_instance,
)
from algorithm.experiments.mmrcpsp_instances import (
    MMRCPSPInstance,
    ModeAction,
    feasible_mode_actions,
    parse_psplib_mm,
)


SCHEMA_VERSION = "scheduleurm.mmrcpsp.trajectory_upgrade.v1"
TRAJECTORY_POLICY = "scheduleurm_trajectory_robust_maxweight"
BASELINE_POLICIES = (
    "shortest_processing_time",
    "minimum_slack",
    "greatest_rank_positional_weight",
    "most_total_successors",
)
ROLLOUT_POLICIES = (OURS_POLICY, *BASELINE_POLICIES)
BEAM_WIDTH = 16
LOCAL_SEED_LIMIT = 10
MODE_PENALTY_WEIGHT = 0.03
CONFIGURATION_PENALTY_WEIGHT = 0.02
PENALTY_BOUND = MODE_PENALTY_WEIGHT + CONFIGURATION_PENALTY_WEIGHT

# These are construction rules, not fitted coefficients.  Every instance uses
# the same ordered list, beam width, and local neighborhood.
BEAM_SEEDS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("critical_service", (1.00, 1.00, 0.20, 0.20)),
    ("flow_service", (0.35, 1.35, 0.55, 0.15)),
    ("resource_guard", (0.70, 0.85, 0.20, 0.70)),
    ("critical_flow", (1.20, 0.70, 0.70, 0.15)),
)


class TrajectoryUpgradeError(RuntimeError):
    """Raised when the finite-action experiment cannot be certified."""


@dataclass(frozen=True)
class _Analysis:
    descendants: Mapping[int, frozenset[int]]
    critical_path: Mapping[int, int]


@dataclass(frozen=True)
class _SearchState:
    now: int
    unscheduled: frozenset[int]
    completed: frozenset[int]
    entries: tuple[ScheduleEntry, ...]
    nonrenewable_consumed: tuple[int, ...]
    construction_score: float


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    origin: str
    entries: tuple[ScheduleEntry, ...]

    @property
    def signature(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (entry.job_id, entry.mode_id, entry.start, entry.finish)
            for entry in self.entries
        )


def build_trajectory_upgrade(
    instance_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Run the fixed trajectory-action protocol on one or more instances."""

    if not instance_paths:
        raise ValueError("at least one PSPLIB .mm instance is required")
    paths = sorted((Path(path) for path in instance_paths), key=lambda p: (p.name, str(p)))
    instances = [parse_psplib_mm(path) for path in paths]
    names = [instance.name for instance in instances]
    if len(set(names)) != len(names):
        raise ValueError(f"instance stems must be unique: {names}")

    reports: dict[str, Any] = {}
    all_feasible = True
    all_exact_oracle = True
    zero_resource_violation = True
    zero_precedence_violation = True
    selected_nondominated_count = 0
    for instance in instances:
        row = _run_instance(instance)
        reports[instance.name] = row
        all_feasible = all_feasible and bool(row["gate"]["all_evaluated_schedules_feasible"])
        all_exact_oracle = all_exact_oracle and bool(row["oracle_audit"]["exact_over_candidate_family"])
        zero_resource_violation = zero_resource_violation and bool(
            row["gate"]["zero_resource_violation"]
        )
        zero_precedence_violation = zero_precedence_violation and bool(
            row["gate"]["zero_precedence_violation"]
        )
        selected_nondominated_count += int(row["pareto"]["selected_is_nondominated"])

    gate_pass = (
        all_feasible
        and all_exact_oracle
        and zero_resource_violation
        and zero_precedence_violation
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "finite_deterministic_mmrcpsp_trajectory_configuration_actions",
        "deterministic_construction": True,
        "per_instance_tuning": False,
        "instance_count": len(instances),
        "instances": reports,
        "construction_contract": {
            "rollout_policies": list(ROLLOUT_POLICIES),
            "beam_width": BEAM_WIDTH,
            "beam_seeds": [
                {"name": name, "weights": list(weights)}
                for name, weights in BEAM_SEEDS
            ],
            "local_seed_limit": LOCAL_SEED_LIMIT,
            "local_neighborhood": (
                "all adjacent launch-order swaps and all one-job alternative-mode "
                "substitutions from the first fixed-order unique seeds"
            ),
            "deduplication": "exact (job, mode, start, finish) trajectory signature",
        },
        "objective_contract": {
            "formula": "Q^T lower_service(trajectory) - bounded_trajectory_penalty",
            "trajectory_horizon_bound": (
                "H is the sum of maximum declared mode durations and therefore a "
                "common serial-completion upper bound for every generated feasible trajectory."
            ),
            "activity_lower_service": (
                "H - completion_time: time-integrated completed-work service for "
                "each non-dummy activity over the common macro-action horizon."
            ),
            "project_lower_service": "H - makespan for the project-completion coordinate.",
            "queue_weight": "One for every activity and for the project-completion coordinate.",
            "deterministic_lower_bound": (
                "Durations are deterministic in the parsed instance, so the declared "
                "cumulative lower-service coordinates equal their evaluated values."
            ),
            "penalty": (
                "0.03 times non-nominal-mode fraction plus 0.02 times x/(1+x), "
                "where x is renewable-configuration L1."
            ),
            "penalty_bound": PENALTY_BOUND,
        },
        "gate": {
            "status": (
                "MMRCPSP_TRAJECTORY_ACTION_PASS"
                if gate_pass
                else "MMRCPSP_TRAJECTORY_ACTION_FAIL"
            ),
            "pass": gate_pass,
            "all_evaluated_schedules_feasible": all_feasible,
            "zero_resource_violation": zero_resource_violation,
            "zero_precedence_violation": zero_precedence_violation,
            "exact_oracle_over_generated_family": all_exact_oracle,
            "performance_dominance_not_required": True,
        },
        "aggregate": {
            "selected_nondominated_instance_count": selected_nondominated_count,
            "selected_nondominated_fraction": _round(
                selected_nondominated_count / len(instances)
            ),
        },
        "theorem_mapping": {
            "finite_candidate_family": (
                "Each complete activity-mode schedule is one finite configuration action."
            ),
            "robust_score": (
                "The exact enumerated selector maximizes the declared cumulative "
                "lower-service linear form minus a uniformly bounded trajectory penalty."
            ),
            "pareto_monotonicity": (
                "With unit queue weights, Q^T lower_service equals a constant minus "
                "the sum of real-job completion times and makespan.  A strict integer "
                "Pareto improvement changes this term by at least one, exceeding the "
                "declared 0.05 penalty range."
            ),
            "oracle_error": (
                "The selected-minus-best generated-family score gap is exactly zero; "
                "this does not assert zero gap to the full MMRCPSP feasible set."
            ),
            "statewise_feasibility": (
                "Every construction step calls feasible_mode_actions, including the "
                "exact nonrenewable-reserve certificate, and every terminal trajectory "
                "is independently audited for precedence and both resource classes."
            ),
        },
        "claim_boundary": {
            "strong_claim_ready": False,
            "performance_superiority_is_descriptive": True,
            "scope": (
                "Fixed finite trajectory family on explicitly supplied deterministic "
                "PSPLIB instances; no stochastic arrivals or online recourse."
            ),
            "blockers": [
                "Beam and local search generate a strict subset of the full feasible trajectory set.",
                "The exact oracle certificate applies only to the generated finite family.",
                "SPT, MINSLK, GRPW, and MTS are priority-rule baselines, not an exact or full-stack SOTA solver.",
                "Nondominance or dominance is reported descriptively and is never a feasibility gate.",
            ],
        },
    }


def _run_instance(instance: MMRCPSPInstance) -> dict[str, Any]:
    analysis = _analyze(instance)
    started = perf_counter()
    raw_candidates: list[_Candidate] = []
    construction_counts = {
        "rollout_seed_count": 0,
        "beam_seed_count": 0,
        "beam_state_expansion_count": 0,
        "local_neighbor_attempt_count": 0,
        "local_neighbor_feasible_count": 0,
    }

    for policy in ROLLOUT_POLICIES:
        result = schedule_instance(instance, policy)
        if not result.feasible:
            raise TrajectoryUpgradeError(
                f"{instance.name}: rollout seed {policy} is infeasible: {result.failure_reason}"
            )
        raw_candidates.append(
            _Candidate(
                candidate_id=f"rollout:{policy}",
                origin="rollout",
                entries=result.entries,
            )
        )
        construction_counts["rollout_seed_count"] += 1

    for seed_name, weights in BEAM_SEEDS:
        beam_candidates, expansion_count = _beam_trajectories(
            instance,
            analysis=analysis,
            seed_name=seed_name,
            weights=weights,
        )
        raw_candidates.extend(beam_candidates)
        construction_counts["beam_seed_count"] += 1
        construction_counts["beam_state_expansion_count"] += expansion_count

    base_candidates = _deduplicate(raw_candidates)[:LOCAL_SEED_LIMIT]
    local_candidates, attempted = _local_neighbors(instance, base_candidates)
    raw_candidates.extend(local_candidates)
    construction_counts["local_neighbor_attempt_count"] = attempted
    construction_counts["local_neighbor_feasible_count"] = len(local_candidates)
    candidates = _deduplicate(raw_candidates)
    if not candidates:
        raise TrajectoryUpgradeError(f"{instance.name}: empty trajectory candidate family")

    evaluated = [_evaluate_candidate(instance, candidate) for candidate in candidates]
    infeasible = [row for row in evaluated if not row["feasible"]]
    if infeasible:
        raise TrajectoryUpgradeError(
            f"{instance.name}: generated infeasible trajectory {infeasible[0]['candidate_id']}"
        )
    selected = max(
        evaluated,
        key=lambda row: (
            row["objective"]["robust_score"],
            -row["metrics"]["makespan"],
            -row["metrics"]["mean_flow_time"],
            row["candidate_id"],
        ),
    )
    maximum_score = max(row["objective"]["robust_score"] for row in evaluated)
    oracle_gap = _round(maximum_score - selected["objective"]["robust_score"])
    if oracle_gap != 0.0:
        raise TrajectoryUpgradeError(f"{instance.name}: nonzero generated-family oracle gap")

    baselines = {
        policy: schedule_instance(instance, policy).snapshot()
        for policy in BASELINE_POLICIES
    }
    pareto = _pareto_audit(selected, baselines)
    elapsed = perf_counter() - started
    total_work = (
        construction_counts["rollout_seed_count"]
        + construction_counts["beam_state_expansion_count"]
        + construction_counts["local_neighbor_attempt_count"]
        + len(evaluated)
    )
    metrics = selected["metrics"]
    all_schedules = [selected, *baselines.values()]
    all_feasible = all(bool(row["feasible"]) for row in all_schedules)
    zero_resource = all(
        float(row["metrics"]["resource_violation_units"]) == 0.0
        for row in all_schedules
    )
    zero_precedence = all(
        int(row["metrics"]["precedence_violation_count"]) == 0
        for row in all_schedules
    )
    return {
        "instance": instance.snapshot(),
        "selected_policy": TRAJECTORY_POLICY,
        "selected_candidate": selected,
        "baseline_results": baselines,
        "pareto": pareto,
        "action_family": {
            "raw_candidate_count": len(raw_candidates),
            "unique_candidate_count": len(candidates),
            "origin_counts": {
                origin: sum(candidate.origin == origin for candidate in candidates)
                for origin in ("rollout", "beam", "local_improvement")
            },
            "finite": True,
            "construction_counts": construction_counts,
        },
        "runtime": {
            "wall_clock_seconds": _round(elapsed, digits=6),
            "deterministic_work_units": total_work,
            "wall_clock_is_not_a_determinism_contract": True,
        },
        "oracle_audit": {
            "evaluated_candidate_count": len(evaluated),
            "maximum_generated_family_score": _round(maximum_score),
            "selected_score": selected["objective"]["robust_score"],
            "oracle_gap": oracle_gap,
            "exact_over_candidate_family": oracle_gap == 0.0,
            "exact_over_full_feasible_set": False,
        },
        "resource_audit": {
            "renewable_capacities": list(instance.renewable_capacities),
            "renewable_peak_excess": list(metrics["renewable_peak_excess"]),
            "nonrenewable_capacities": list(instance.nonrenewable_capacities),
            "nonrenewable_used": list(metrics["nonrenewable_used"]),
            "nonrenewable_excess": list(metrics["nonrenewable_excess"]),
        },
        "gate": {
            "all_evaluated_schedules_feasible": all_feasible,
            "zero_resource_violation": zero_resource,
            "zero_precedence_violation": zero_precedence,
            "performance_dominance_not_required": True,
        },
    }


def _beam_trajectories(
    instance: MMRCPSPInstance,
    *,
    analysis: _Analysis,
    seed_name: str,
    weights: tuple[float, float, float, float],
) -> tuple[list[_Candidate], int]:
    root = _normalize_state(
        instance,
        _SearchState(
            now=0,
            unscheduled=frozenset(instance.job_ids),
            completed=frozenset(),
            entries=(),
            nonrenewable_consumed=tuple(0 for _ in instance.nonrenewable_capacities),
            construction_score=0.0,
        ),
    )
    frontier = [root]
    terminals: list[_SearchState] = []
    expansion_count = 0
    decision_limit = max(100, 20 * len(instance.job_ids) ** 2)
    for _ in range(decision_limit):
        if not frontier:
            break
        children: list[_SearchState] = []
        for state in frontier:
            if not state.unscheduled:
                terminals.append(state)
                continue
            actions = _state_actions(instance, state)
            if not actions:
                raise TrajectoryUpgradeError(
                    f"{instance.name}: beam seed {seed_name} reached a certified deadlock"
                )
            for action in actions:
                expansion_count += 1
                step_score, components = _construction_action_score(
                    instance,
                    state,
                    action,
                    analysis=analysis,
                    weights=weights,
                )
                entry = _entry_from_action(state.now, action, components)
                child = _SearchState(
                    now=state.now,
                    unscheduled=state.unscheduled - {action.job_id},
                    completed=state.completed | ({action.job_id} if action.duration == 0 else set()),
                    entries=(*state.entries, entry),
                    nonrenewable_consumed=tuple(
                        old + demand
                        for old, demand in zip(
                            state.nonrenewable_consumed,
                            action.nonrenewable_demands,
                            strict=True,
                        )
                    ),
                    construction_score=state.construction_score + step_score,
                )
                children.append(_normalize_state(instance, child))
        if not children:
            break
        unique: dict[tuple[Any, ...], _SearchState] = {}
        for child in children:
            signature = _state_signature(child)
            incumbent = unique.get(signature)
            if incumbent is None or _beam_key(child) > _beam_key(incumbent):
                unique[signature] = child
        frontier = sorted(unique.values(), key=_beam_key, reverse=True)[:BEAM_WIDTH]
    else:
        raise TrajectoryUpgradeError(f"{instance.name}: beam decision limit exceeded")

    terminals.extend(state for state in frontier if not state.unscheduled)
    complete = sorted(
        {state.entries for state in terminals},
        key=lambda entries: _trajectory_signature(entries),
    )
    if not complete:
        raise TrajectoryUpgradeError(f"{instance.name}: beam seed {seed_name} produced no terminal")
    return (
        [
            _Candidate(
                candidate_id=f"beam:{seed_name}:{index:03d}",
                origin="beam",
                entries=entries,
            )
            for index, entries in enumerate(complete)
        ],
        expansion_count,
    )


def _normalize_state(instance: MMRCPSPInstance, state: _SearchState) -> _SearchState:
    current = state
    for _ in range(max(2, len(instance.job_ids) + 1)):
        finished = {
            entry.job_id for entry in current.entries if entry.finish <= current.now
        }
        completed = current.completed | finished
        current = _SearchState(
            now=current.now,
            unscheduled=current.unscheduled,
            completed=frozenset(completed),
            entries=current.entries,
            nonrenewable_consumed=current.nonrenewable_consumed,
            construction_score=current.construction_score,
        )
        if not current.unscheduled or _state_actions(instance, current):
            return current
        running = [entry for entry in current.entries if entry.finish > current.now]
        if not running:
            return current
        next_finish = min(entry.finish for entry in running)
        current = _SearchState(
            now=next_finish,
            unscheduled=current.unscheduled,
            completed=current.completed,
            entries=current.entries,
            nonrenewable_consumed=current.nonrenewable_consumed,
            construction_score=current.construction_score,
        )
    raise TrajectoryUpgradeError(f"{instance.name}: state normalization did not advance")


def _state_actions(instance: MMRCPSPInstance, state: _SearchState) -> tuple[ModeAction, ...]:
    activity_map = instance.activity_map
    eligible = tuple(
        sorted(
            job_id
            for job_id in state.unscheduled
            if set(activity_map[job_id].predecessors).issubset(state.completed)
        )
    )
    running = [entry for entry in state.entries if entry.start <= state.now < entry.finish]
    renewable_used = tuple(
        sum(entry.renewable_demands[index] for entry in running)
        for index in range(len(instance.renewable_capacities))
    )
    renewable_available = tuple(
        capacity - used
        for capacity, used in zip(instance.renewable_capacities, renewable_used, strict=True)
    )
    return feasible_mode_actions(
        instance,
        eligible_job_ids=eligible,
        unscheduled_job_ids=state.unscheduled,
        renewable_available=renewable_available,
        nonrenewable_consumed=state.nonrenewable_consumed,
    )


def _construction_action_score(
    instance: MMRCPSPInstance,
    state: _SearchState,
    action: ModeAction,
    *,
    analysis: _Analysis,
    weights: tuple[float, float, float, float],
) -> tuple[float, tuple[float, float, float, float]]:
    if action.duration == 0:
        queue_weight = 1e6 - action.job_id
        return queue_weight, (queue_weight, 1.0, 0.0, queue_weight)
    critical_weight, service_weight, flow_weight, pressure_weight = weights
    criticality = 1.0 + analysis.critical_path[action.job_id]
    lower_service = 1.0 / action.duration
    pressure = _resource_pressure(instance, action)
    projected_finish = state.now + action.duration
    queue_weight = critical_weight * criticality
    penalty = pressure_weight * pressure + flow_weight * projected_finish / max(1, instance.horizon)
    score = queue_weight * service_weight * lower_service - penalty
    return score, (queue_weight, lower_service, penalty, score)


def _entry_from_action(
    now: int,
    action: ModeAction,
    components: tuple[float, float, float, float],
) -> ScheduleEntry:
    queue_weight, lower_service, penalty, score = components
    return ScheduleEntry(
        job_id=action.job_id,
        mode_id=action.mode_id,
        start=now,
        finish=now + action.duration,
        duration=action.duration,
        renewable_demands=action.renewable_demands,
        nonrenewable_demands=action.nonrenewable_demands,
        action_score=_round(score),
        queue_weight_proxy=_round(queue_weight),
        lower_service=_round(lower_service),
        penalty_units=_round(penalty),
        action_type="trajectory_configuration_action",
    )


def _local_neighbors(
    instance: MMRCPSPInstance,
    seeds: Sequence[_Candidate],
) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    attempted = 0
    for seed_index, seed in enumerate(seeds):
        order = [entry.job_id for entry in seed.entries]
        modes = {entry.job_id: entry.mode_id for entry in seed.entries}
        for index in range(max(0, len(order) - 1)):
            attempted += 1
            swapped = list(order)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            entries = _replay_preference(instance, swapped, modes)
            if entries is not None:
                candidates.append(
                    _Candidate(
                        candidate_id=f"local:seed{seed_index:02d}:swap{index:02d}",
                        origin="local_improvement",
                        entries=entries,
                    )
                )
        for job_id in instance.real_job_ids:
            for mode in instance.activity_map[job_id].modes:
                if mode.mode_id == modes.get(job_id):
                    continue
                attempted += 1
                alternate_modes = dict(modes)
                alternate_modes[job_id] = mode.mode_id
                entries = _replay_preference(instance, order, alternate_modes)
                if entries is not None:
                    candidates.append(
                        _Candidate(
                            candidate_id=(
                                f"local:seed{seed_index:02d}:job{job_id:03d}:mode{mode.mode_id:02d}"
                            ),
                            origin="local_improvement",
                            entries=entries,
                        )
                    )
    return candidates, attempted


def _replay_preference(
    instance: MMRCPSPInstance,
    preferred_order: Sequence[int],
    preferred_modes: Mapping[int, int],
) -> tuple[ScheduleEntry, ...] | None:
    rank = {job_id: index for index, job_id in enumerate(preferred_order)}
    state = _normalize_state(
        instance,
        _SearchState(
            now=0,
            unscheduled=frozenset(instance.job_ids),
            completed=frozenset(),
            entries=(),
            nonrenewable_consumed=tuple(0 for _ in instance.nonrenewable_capacities),
            construction_score=0.0,
        ),
    )
    for _ in range(max(100, 20 * len(instance.job_ids) ** 2)):
        if not state.unscheduled:
            metrics = _evaluate_schedule(instance, state.entries)
            return state.entries if metrics["schedule_feasible"] else None
        actions = _state_actions(instance, state)
        if not actions:
            return None
        action = min(
            actions,
            key=lambda row: (
                rank.get(row.job_id, len(rank) + row.job_id),
                row.mode_id != preferred_modes.get(row.job_id, row.mode_id),
                abs(row.mode_id - preferred_modes.get(row.job_id, row.mode_id)),
                row.duration,
                row.job_id,
                row.mode_id,
            ),
        )
        lower_service = 1.0 / action.duration if action.duration else 1.0
        entry = _entry_from_action(
            state.now,
            action,
            (1.0, lower_service, 0.0, lower_service),
        )
        state = _normalize_state(
            instance,
            _SearchState(
                now=state.now,
                unscheduled=state.unscheduled - {action.job_id},
                completed=state.completed | ({action.job_id} if action.duration == 0 else set()),
                entries=(*state.entries, entry),
                nonrenewable_consumed=tuple(
                    old + demand
                    for old, demand in zip(
                        state.nonrenewable_consumed,
                        action.nonrenewable_demands,
                        strict=True,
                    )
                ),
                construction_score=state.construction_score,
            ),
        )
    return None


def _evaluate_candidate(
    instance: MMRCPSPInstance,
    candidate: _Candidate,
) -> dict[str, Any]:
    metrics = _evaluate_schedule(instance, candidate.entries)
    if not metrics["schedule_feasible"]:
        return {
            "candidate_id": candidate.candidate_id,
            "origin": candidate.origin,
            "feasible": False,
            "metrics": metrics,
        }
    entry_map = {entry.job_id: entry for entry in candidate.entries}
    trajectory_horizon = max(
        1,
        sum(
            max(mode.duration for mode in instance.activity_map[job_id].modes)
            for job_id in instance.job_ids
        ),
    )
    activity_coordinates = []
    cumulative_lower_service = 0.0
    for job_id in instance.real_job_ids:
        finish = entry_map[job_id].finish
        if finish <= 0:
            raise TrajectoryUpgradeError(
                f"{instance.name}: non-dummy job {job_id} has nonpositive completion"
            )
        if finish > trajectory_horizon:
            raise TrajectoryUpgradeError(
                f"{instance.name}: completion {finish} exceeds serial horizon {trajectory_horizon}"
            )
        queue_weight = 1.0
        lower_service = float(trajectory_horizon - finish)
        contribution = queue_weight * lower_service
        cumulative_lower_service += contribution
        activity_coordinates.append(
            {
                "job_id": job_id,
                "queue_weight": _round(queue_weight),
                "lower_service": _round(lower_service),
                "weighted_contribution": _round(contribution),
            }
        )
    makespan = int(metrics["makespan"])
    if makespan <= 0 and instance.real_job_ids:
        raise TrajectoryUpgradeError(f"{instance.name}: nonpositive project makespan")
    if makespan > trajectory_horizon:
        raise TrajectoryUpgradeError(
            f"{instance.name}: makespan {makespan} exceeds serial horizon {trajectory_horizon}"
        )
    project_weight = 1.0
    project_lower_service = float(trajectory_horizon - makespan)
    project_contribution = project_weight * project_lower_service
    cumulative_lower_service += project_contribution

    mode_fraction = (
        metrics["mode_reconfiguration_count"] / len(instance.real_job_ids)
        if instance.real_job_ids
        else 0.0
    )
    configuration = float(metrics["renewable_configuration_l1"])
    configuration_squash = configuration / (1.0 + configuration)
    mode_penalty = MODE_PENALTY_WEIGHT * mode_fraction
    configuration_penalty = CONFIGURATION_PENALTY_WEIGHT * configuration_squash
    penalty = mode_penalty + configuration_penalty
    if penalty < -1e-12 or penalty > PENALTY_BOUND + 1e-12:
        raise TrajectoryUpgradeError(
            f"{instance.name}: trajectory penalty {penalty} violates bound {PENALTY_BOUND}"
        )
    robust_score = cumulative_lower_service - penalty
    return {
        "candidate_id": candidate.candidate_id,
        "origin": candidate.origin,
        "trajectory_sha256": _trajectory_sha256(candidate.entries),
        "feasible": True,
        "metrics": metrics,
        "objective": {
            "trajectory_horizon_bound": trajectory_horizon,
            "activity_coordinates": activity_coordinates,
            "project_coordinate": {
                "queue_weight": _round(project_weight),
                "lower_service": _round(project_lower_service),
                "weighted_contribution": _round(project_contribution),
            },
            "cumulative_weighted_lower_service": _round(cumulative_lower_service),
            "penalty_components": {
                "mode_reconfiguration": _round(mode_penalty),
                "renewable_configuration": _round(configuration_penalty),
            },
            "bounded_trajectory_penalty": _round(penalty),
            "penalty_bound": PENALTY_BOUND,
            "robust_score": _round(robust_score),
        },
        "schedule": [entry.snapshot() for entry in candidate.entries],
    }


def _pareto_audit(
    selected: Mapping[str, Any],
    baselines: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    policies = {TRAJECTORY_POLICY: selected, **baselines}
    frontier = []
    for policy, row in policies.items():
        metrics = row["metrics"]
        dominated = False
        for other_policy, other in policies.items():
            if other_policy == policy:
                continue
            other_metrics = other["metrics"]
            weak = (
                other_metrics["makespan"] <= metrics["makespan"]
                and other_metrics["mean_flow_time"] <= metrics["mean_flow_time"]
            )
            strict = (
                other_metrics["makespan"] < metrics["makespan"]
                or other_metrics["mean_flow_time"] < metrics["mean_flow_time"]
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(policy)
    comparisons = {}
    selected_metrics = selected["metrics"]
    for policy, row in baselines.items():
        metrics = row["metrics"]
        selected_weak = (
            selected_metrics["makespan"] <= metrics["makespan"]
            and selected_metrics["mean_flow_time"] <= metrics["mean_flow_time"]
        )
        selected_strict = (
            selected_metrics["makespan"] < metrics["makespan"]
            or selected_metrics["mean_flow_time"] < metrics["mean_flow_time"]
        )
        baseline_weak = (
            metrics["makespan"] <= selected_metrics["makespan"]
            and metrics["mean_flow_time"] <= selected_metrics["mean_flow_time"]
        )
        baseline_strict = (
            metrics["makespan"] < selected_metrics["makespan"]
            or metrics["mean_flow_time"] < selected_metrics["mean_flow_time"]
        )
        comparisons[policy] = {
            "selected_pareto_dominates": bool(selected_weak and selected_strict),
            "baseline_pareto_dominates_selected": bool(baseline_weak and baseline_strict),
            "makespan_difference_selected_minus_baseline": _round(
                selected_metrics["makespan"] - metrics["makespan"]
            ),
            "mean_flow_difference_selected_minus_baseline": _round(
                selected_metrics["mean_flow_time"] - metrics["mean_flow_time"]
            ),
        }
    return {
        "policies": [TRAJECTORY_POLICY, *BASELINE_POLICIES],
        "frontier": frontier,
        "selected_is_nondominated": TRAJECTORY_POLICY in frontier,
        "selected_strictly_dominates_every_baseline": all(
            row["selected_pareto_dominates"] for row in comparisons.values()
        ),
        "comparisons": comparisons,
        "dominance_is_descriptive_not_gate": True,
    }


def _analyze(instance: MMRCPSPInstance) -> _Analysis:
    activity_map = instance.activity_map
    order = _topological_order(instance)
    minimum_duration = {
        job_id: min(mode.duration for mode in activity_map[job_id].modes)
        for job_id in instance.job_ids
    }
    descendants: dict[int, frozenset[int]] = {}
    critical_path: dict[int, int] = {}
    for job_id in reversed(order):
        nested: set[int] = set()
        successor_paths = []
        for successor in activity_map[job_id].successors:
            nested.add(successor)
            nested.update(descendants[successor])
            successor_paths.append(critical_path[successor])
        descendants[job_id] = frozenset(nested)
        critical_path[job_id] = minimum_duration[job_id] + (
            max(successor_paths) if successor_paths else 0
        )
    return _Analysis(descendants=descendants, critical_path=critical_path)


def _topological_order(instance: MMRCPSPInstance) -> tuple[int, ...]:
    activity_map = instance.activity_map
    indegree = {job_id: len(activity_map[job_id].predecessors) for job_id in instance.job_ids}
    ready = sorted(job_id for job_id, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        job_id = ready.pop(0)
        order.append(job_id)
        for successor in activity_map[job_id].successors:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(instance.job_ids):
        raise TrajectoryUpgradeError(f"{instance.name}: precedence graph is cyclic")
    return tuple(order)


def _resource_pressure(instance: MMRCPSPInstance, action: ModeAction) -> float:
    renewable = sum(
        demand / capacity if capacity else float(demand)
        for demand, capacity in zip(
            action.renewable_demands, instance.renewable_capacities, strict=True
        )
    )
    nonrenewable = sum(
        demand / capacity if capacity else float(demand)
        for demand, capacity in zip(
            action.nonrenewable_demands, instance.nonrenewable_capacities, strict=True
        )
    )
    return renewable + nonrenewable


def _beam_key(state: _SearchState) -> tuple[Any, ...]:
    completed_work = len(state.entries)
    finish_sum = sum(entry.finish for entry in state.entries)
    return (
        _round(state.construction_score),
        completed_work,
        -finish_sum,
        tuple(-value for value in state.nonrenewable_consumed),
        tuple((-entry.job_id, -entry.mode_id, -entry.start) for entry in state.entries),
    )


def _state_signature(state: _SearchState) -> tuple[Any, ...]:
    return (
        state.now,
        tuple(sorted(state.unscheduled)),
        tuple(sorted(state.completed)),
        state.nonrenewable_consumed,
        _trajectory_signature(state.entries),
    )


def _trajectory_signature(
    entries: Iterable[ScheduleEntry],
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (entry.job_id, entry.mode_id, entry.start, entry.finish)
        for entry in entries
    )


def _trajectory_sha256(entries: Sequence[ScheduleEntry]) -> str:
    payload = json.dumps(_trajectory_signature(entries), separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _deduplicate(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    unique: dict[tuple[tuple[int, int, int, int], ...], _Candidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.signature, candidate)
    return sorted(unique.values(), key=lambda row: (row.candidate_id, row.signature))


def _round(value: float, *, digits: int = 9) -> float:
    if not math.isfinite(float(value)):
        raise TrajectoryUpgradeError(f"non-finite trajectory metric: {value}")
    return round(float(value), digits)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_artifact(report: Mapping[str, Any], path: str | Path) -> None:
    """Write a deterministic artifact after removing wall-clock observations."""

    deterministic = json.loads(json.dumps(report, allow_nan=False))
    for row in deterministic.get("instances", {}).values():
        row["runtime"]["wall_clock_seconds"] = None
    _atomic_write(
        Path(path),
        json.dumps(deterministic, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed MMRCPSP finite trajectory-action upgrade."
    )
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_trajectory_upgrade(args.instances)
    write_artifact(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
