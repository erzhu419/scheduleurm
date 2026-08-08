"""Theorem-aligned trajectory/configuration-action upgrade for Brandimarte FJSP.

The original FJSP sidecar makes one myopic dispatch decision at a time.  This
module instead constructs a finite family of complete feasible trajectories
from globally fixed deterministic seeds and a fixed-width prefix beam.  It then
evaluates every generated trajectory under one frozen robust MaxWeight score

    q_bar^T lower_service(trajectory) - bounded_penalty(trajectory)

and returns the exact maximizer *within that generated family*.  The family
contains every existing sidecar policy, so the comparison does not remove a
baseline after seeing an instance.  It is not an exact oracle over all FJSP
schedules and it does not use CP-SAT.

The Brandimarte instances are deterministic.  For a common conservative frame
H, the job coordinate ``(H - C_j) / H`` and the system-drain coordinate
``(H - C_max) / H`` are therefore exact lower-service coordinates.  A bounded
latency penalty uses the same completion ledger.  With the frozen equal-weight
configuration the score is strictly decreasing in both makespan and mean flow,
so the selected member is Pareto-nondominated inside the generated family.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.fjsp_benchmark import (
    POLICY_LABELS,
    POLICY_ORDER,
    CandidateAction,
    ReassignmentConfig,
    ScheduledOperation,
    build_schedule,
    enumerate_feasible_actions,
    verify_schedule,
)
from algorithm.experiments.fjsp_instances import FJSPInstance
from algorithm.experiments.fjsp_suite_benchmark import (
    DEFAULT_MANIFEST,
    DEFAULT_SUITE_DIR,
    EXPECTED_INSTANCE_IDS,
    validate_source_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = DEFAULT_SUITE_DIR / "fjsp_trajectory_upgrade_gate.json"
DEFAULT_MARKDOWN = DEFAULT_SUITE_DIR / "fjsp_trajectory_upgrade_gate.md"
TRAJECTORY_POLICY = "ours_trajectory_configuration"
TRAJECTORY_LABEL = "Ours: trajectory robust MaxWeight"
CLASSIC_POLICIES = tuple(policy for policy in POLICY_ORDER if policy != "ours_robust_maxweight")
COMPARISON_ORDER = (TRAJECTORY_POLICY,) + POLICY_ORDER
SPLITS = {
    "development": tuple(f"mk{index:02d}" for index in range(1, 6)),
    "calibration": tuple(f"mk{index:02d}" for index in range(6, 11)),
    "holdout": tuple(f"mk{index:02d}" for index in range(11, 16)),
}


class TrajectoryGateError(ValueError):
    """Raised when a source, configuration, or feasibility contract fails."""


@dataclass(frozen=True)
class FrozenTrajectoryConfig:
    """One global configuration used unchanged for all fifteen instances."""

    prefix_depth: int = 2
    beam_width: int = 12
    rollout_policies: tuple[str, ...] = CLASSIC_POLICIES
    system_queue_weight_per_job: float = 1.0
    bounded_latency_penalty: float = 0.05

    def __post_init__(self) -> None:
        if self.prefix_depth <= 0 or self.beam_width <= 0:
            raise ValueError("prefix depth and beam width must be positive")
        if not self.rollout_policies:
            raise ValueError("at least one deterministic rollout policy is required")
        if len(set(self.rollout_policies)) != len(self.rollout_policies):
            raise ValueError("rollout policies must be unique")
        if any(policy not in CLASSIC_POLICIES for policy in self.rollout_policies):
            raise ValueError("rollout family may contain only registered classic policies")
        if not math.isfinite(self.system_queue_weight_per_job) or self.system_queue_weight_per_job <= 0:
            raise ValueError("system queue weight must be finite and positive")
        if not math.isfinite(self.bounded_latency_penalty) or not 0 <= self.bounded_latency_penalty <= 1:
            raise ValueError("bounded latency penalty must lie in [0, 1]")

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "scheduleurm.fjsp_trajectory_config.v1",
            "prefix_depth": self.prefix_depth,
            "beam_width": self.beam_width,
            "rollout_policies": list(self.rollout_policies),
            "system_queue_weight_per_job": _number(self.system_queue_weight_per_job),
            "bounded_latency_penalty": _number(self.bounded_latency_penalty),
            "candidate_seed_policies": list(POLICY_ORDER),
            "per_instance_tuning": False,
            "cp_sat_used_by_ours": False,
        }


FROZEN_CONFIG = FrozenTrajectoryConfig()
# Literal lock is filled from the canonical snapshot and checked before any suite run.
FROZEN_CONFIG_SHA256 = "b9fa529ff6218035d4fe0173b9faf09909db96fdd8b24035d8e52a292359fe07"


@dataclass(frozen=True)
class _SearchState:
    next_operation: tuple[int, ...]
    job_ready: tuple[float, ...]
    machine_timelines: tuple[tuple[ScheduledOperation, ...], ...]
    scheduled: tuple[ScheduledOperation, ...]
    decision_action_ids: tuple[str, ...]


@dataclass(frozen=True)
class _TrajectoryCandidate:
    candidate_id: str
    origins: tuple[str, ...]
    schedule: tuple[ScheduledOperation, ...]
    metrics: Mapping[str, float]
    feasibility: Mapping[str, Any]
    score_audit: Mapping[str, Any]
    fingerprint: str


def frozen_config_sha256() -> str:
    payload = json.dumps(FROZEN_CONFIG.snapshot(), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def build_trajectory_schedule(instance: FJSPInstance) -> dict[str, Any]:
    """Build and exactly score the frozen finite trajectory family."""

    _verify_frozen_config()
    started = time.perf_counter()
    raw: list[tuple[str, str, tuple[ScheduledOperation, ...]]] = []
    baseline_runs: dict[str, Mapping[str, Any]] = {}

    for policy in POLICY_ORDER:
        run = build_schedule(instance, policy)
        baseline_runs[policy] = run
        raw.append((f"seed:{policy}", f"fixed_policy:{policy}", _schedule_from_run(run)))

    beam = _prefix_beam(instance, FROZEN_CONFIG)
    for beam_index, state in enumerate(beam):
        prefix_hash = _state_fingerprint(state)[:12]
        for policy in FROZEN_CONFIG.rollout_policies:
            completed = _complete_state(instance, state, policy)
            raw.append(
                (
                    f"beam:{beam_index:02d}:{prefix_hash}:rollout:{policy}",
                    f"prefix_beam_depth_{FROZEN_CONFIG.prefix_depth}+{policy}",
                    completed.scheduled,
                )
            )

    candidates = _deduplicate_and_score(instance, raw)
    if not candidates:
        raise TrajectoryGateError("finite trajectory family is empty")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row.score_audit["score"]),
            float(row.metrics["makespan"]),
            float(row.metrics["mean_flow_time"]),
            row.fingerprint,
        ),
    )
    selected = ranked[0]
    best_score = max(float(row.score_audit["score"]) for row in candidates)
    oracle_gap = best_score - float(selected.score_audit["score"])
    if abs(oracle_gap) > 1e-12:
        raise TrajectoryGateError("selected trajectory is not the exact family maximizer")
    if not bool(selected.feasibility["pass"]):
        raise TrajectoryGateError("selected trajectory is infeasible")
    dominated_by = [
        policy
        for policy in POLICY_ORDER
        if _dominates(_metrics(baseline_runs[policy]), selected.metrics)
    ]
    if dominated_by:
        raise TrajectoryGateError(
            "monotone trajectory score selected a baseline-dominated schedule: "
            + ", ".join(dominated_by)
        )

    runner_up_gap = None
    if len(ranked) > 1:
        runner_up_gap = float(ranked[0].score_audit["score"]) - float(
            ranked[1].score_audit["score"]
        )
    return {
        "policy": TRAJECTORY_POLICY,
        "policy_label": TRAJECTORY_LABEL,
        "schedule": [row.snapshot() for row in _schedule_output_order(selected.schedule)],
        "metrics": {key: _number(value) for key, value in selected.metrics.items()},
        "feasibility": dict(selected.feasibility),
        "selection": {
            "selected_candidate_id": selected.candidate_id,
            "selected_origins": list(selected.origins),
            "selected_fingerprint": selected.fingerprint,
            "generated_candidate_count": len(raw),
            "unique_action_family_size": len(candidates),
            "beam_terminal_state_count": len(beam),
            "exact_family_argmax": True,
            "oracle_gap_within_generated_family": _number(oracle_gap),
            "runner_up_score_gap": None if runner_up_gap is None else _number(runner_up_gap),
            "global_fjsp_oracle_gap": None,
            "global_fjsp_oracle_claimed": False,
            "score_audit": dict(selected.score_audit),
        },
        "candidate_family": [
            {
                "candidate_id": row.candidate_id,
                "origins": list(row.origins),
                "fingerprint": row.fingerprint,
                "makespan": _number(row.metrics["makespan"]),
                "mean_flow_time": _number(row.metrics["mean_flow_time"]),
                "score": _number(row.score_audit["score"]),
                "feasible": bool(row.feasibility["pass"]),
            }
            for row in ranked
        ],
        "baseline_runs": baseline_runs,
        "runtime": {
            "wall_clock_seconds": _number(time.perf_counter() - started),
            "reported_but_excluded_from_policy_score": True,
        },
    }


def build_fjsp_trajectory_suite(
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Run one hash-locked internal diagnostic partition under one config."""

    config_hash = _verify_frozen_config()
    source_gate = validate_source_manifest(suite_dir, manifest_path)
    source_by_id = {str(row["id"]): row for row in source_gate["verified_instances"]}
    if set(source_by_id) != set(EXPECTED_INSTANCE_IDS):
        raise TrajectoryGateError("source gate did not return Mk01-Mk15 exactly")
    _verify_split_partition()

    rows: list[dict[str, Any]] = []
    phase_audit: list[dict[str, Any]] = []
    for split_name in ("development", "calibration", "holdout"):
        if split_name == "holdout" and config_hash != FROZEN_CONFIG_SHA256:
            raise TrajectoryGateError("holdout cannot run before frozen config verification")
        before_hash = config_hash
        for instance_id in SPLITS[split_name]:
            rows.append(_run_suite_instance(source_by_id[instance_id], split_name))
        after_hash = _verify_frozen_config()
        if after_hash != before_hash:
            raise TrajectoryGateError("configuration changed during split execution")
        phase_audit.append(
            {
                "split": split_name,
                "instances": list(SPLITS[split_name]),
                "config_sha256_before": before_hash,
                "config_sha256_after": after_hash,
                "configuration_updates_applied": False,
            }
        )

    aggregate = _aggregate(rows)
    split_aggregates = {
        split: _aggregate([row for row in rows if row["split"] == split])
        for split in SPLITS
    }
    all_feasible = all(
        bool(result["feasible"])
        for row in rows
        for result in row["policies"].values()
    )
    exact_argmax = all(bool(row["trajectory_search"]["exact_family_argmax"]) for row in rows)
    zero_oracle_gap = all(
        abs(float(row["trajectory_search"]["oracle_gap_within_generated_family"])) <= 1e-12
        for row in rows
    )
    ours_nondominated = sum(
        TRAJECTORY_POLICY in row["pareto_frontier_policies"] for row in rows
    )
    implementation = _implementation_manifest()
    return {
        "schema_version": "scheduleurm.fjsp_trajectory_upgrade.v1",
        "deterministic_algorithm": True,
        "suite": "Brandimarte Mk01-Mk15",
        "source_manifest_sha256": source_gate["manifest_sha256"],
        "implementation": implementation,
        "frozen_protocol": {
            "config": FROZEN_CONFIG.snapshot(),
            "config_sha256": config_hash,
            "split_order": ["development", "calibration", "holdout"],
            "splits": {key: list(value) for key, value in SPLITS.items()},
            "phase_audit": phase_audit,
            "per_instance_tuning": False,
            "calibration_updates_applied": False,
            "holdout_feedback_used_for_configuration": "not_auditable_before_repository_freeze",
            "holdout_execution_guard": "runtime_literal_config_hash_only",
            "internal_partition_is_prospective_holdout": False,
        },
        "theorem_mapping": _theorem_mapping(),
        "instances": rows,
        "split_aggregates": split_aggregates,
        "aggregate": aggregate,
        "gate": {
            "pass": bool(all_feasible and exact_argmax and zero_oracle_gap and ours_nondominated == 15),
            "source_manifest_valid": bool(source_gate["pass"]),
            "instance_count": len(rows),
            "all_schedules_feasible": all_feasible,
            "exact_generated_family_argmax_on_every_instance": exact_argmax,
            "zero_generated_family_oracle_gap_on_every_instance": zero_oracle_gap,
            "ours_instance_pareto_nondominated_count": ours_nondominated,
            "ours_instance_pareto_dominated_count": len(rows) - ours_nondominated,
            "performance_superiority_claim_ready": bool(
                aggregate["ours_strictly_pareto_dominates_every_classic_on_every_instance"]
            ),
            "prospective_external_holdout_ready": False,
            "claim_scope": "finite_hash_locked_fjsp_trajectory_family",
            "global_fjsp_optimality_claim": False,
            "stochastic_throughput_optimality_claim": False,
        },
    }


def write_artifacts(
    report: Mapping[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(markdown_report(report), encoding="utf-8")
    return output_json, output_markdown


def markdown_report(report: Mapping[str, Any]) -> str:
    gate = report["gate"]
    aggregate = report["aggregate"]
    lines = [
        "# FJSP trajectory/configuration-action upgrade",
        "",
        f"- Gate: `{'PASS' if gate['pass'] else 'FAIL'}`",
        f"- Frozen configuration: `{report['frozen_protocol']['config_sha256']}`",
        "- Internal diagnostic partition: Mk01-Mk05 development; Mk06-Mk10 calibration; Mk11-Mk15 evaluation.",
        "- Prospective external holdout: not yet run from a repository-frozen configuration.",
        "- Exactness: exact maximizer of the generated finite trajectory family; not all FJSP schedules.",
        f"- Ours Pareto-nondominated: `{gate['ours_instance_pareto_nondominated_count']}/15`.",
        f"- Strict all-instance classic-policy dominance ready: `{str(gate['performance_superiority_claim_ready']).lower()}`.",
        "",
        "## Fixed theorem mapping",
        "",
        "For common frame `H`, each job has lower-service `(H-C_j)/H`, and the system",
        "coordinate has `(H-C_max)/H`. The normalized queue-weighted service is scored",
        "minus a latency penalty bounded by the single frozen coefficient. The family",
        "oracle gap is zero; approximation to the full FJSP trajectory space is not claimed.",
        "",
        "## Aggregate comparison",
        "",
        "| Policy | Geo. makespan/BKS | Geo. flow/best | Pareto instances |",
        "|---|---:|---:|---:|",
    ]
    for policy in COMPARISON_ORDER:
        row = aggregate["policies"][policy]
        lines.append(
            f"| {row['policy_label']} | {_fmt(row['geomean_makespan_over_bks'])} | "
            f"{_fmt(row['geomean_mean_flow_to_instance_best'])} | {row['pareto_frontier_count']} |"
        )
    lines.extend(
        [
            "",
            "## Instance results",
            "",
            "| Split | Instance | Family | Runtime (s) | Ours makespan | Ours mean flow | Pareto frontier |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["instances"]:
        ours = row["policies"][TRAJECTORY_POLICY]
        lines.append(
            f"| {row['split']} | {row['instance_id']} | "
            f"{row['trajectory_search']['unique_action_family_size']} | "
            f"{_fmt(row['trajectory_search']['runtime_seconds'])} | "
            f"{_fmt(ours['makespan'])} | {_fmt(ours['mean_flow_time'])} | "
            f"{', '.join(row['pareto_frontier_policies'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is deterministic, hash-locked, finite-family FJSP evidence. CP-SAT is not",
            "used. A zero oracle gap refers only to the generated trajectory family. The",
            "artifact does not claim global FJSP optimality or stochastic throughput stability.",
            "",
        ]
    )
    return "\n".join(lines)


def _prefix_beam(instance: FJSPInstance, config: FrozenTrajectoryConfig) -> list[_SearchState]:
    states = [_empty_state(instance)]
    for _ in range(config.prefix_depth):
        expanded: list[_SearchState] = []
        for state in states:
            for action in _actions(instance, state):
                expanded.append(_advance(state, action))
        if not expanded:
            raise TrajectoryGateError("prefix beam exhausted before schedule completion")
        expanded.sort(key=lambda state: _beam_rank(instance, state))
        deduplicated: dict[str, _SearchState] = {}
        for state in expanded:
            deduplicated.setdefault(_state_fingerprint(state), state)
            if len(deduplicated) >= config.beam_width:
                break
        states = list(deduplicated.values())
    return states


def _empty_state(instance: FJSPInstance) -> _SearchState:
    return _SearchState(
        next_operation=tuple(0 for _ in instance.jobs),
        job_ready=tuple(0.0 for _ in instance.jobs),
        machine_timelines=tuple(tuple() for _ in range(instance.machine_count)),
        scheduled=tuple(),
        decision_action_ids=tuple(),
    )


def _actions(instance: FJSPInstance, state: _SearchState) -> list[CandidateAction]:
    return enumerate_feasible_actions(
        instance,
        next_operation_by_job=state.next_operation,
        job_ready_times=state.job_ready,
        machine_timelines=state.machine_timelines,
        reassignment=ReassignmentConfig(),
    )


def _advance(state: _SearchState, action: CandidateAction) -> _SearchState:
    operation = ScheduledOperation(
        job_id=action.operation.job_id,
        operation_id=action.operation.operation_id,
        machine_id=action.machine_id,
        start_time=action.start_time,
        end_time=action.end_time,
        processing_time=action.processing_time,
        reconfiguration_time=0.0,
        preferred_machine_id=action.operation.preferred_machine_id,
        reassignment_semantics_enabled=False,
    )
    next_operation = list(state.next_operation)
    next_operation[operation.job_id] += 1
    job_ready = list(state.job_ready)
    job_ready[operation.job_id] = operation.end_time
    timelines = [list(rows) for rows in state.machine_timelines]
    timelines[operation.machine_id].append(operation)
    timelines[operation.machine_id].sort(key=_operation_order)
    return _SearchState(
        next_operation=tuple(next_operation),
        job_ready=tuple(job_ready),
        machine_timelines=tuple(tuple(rows) for rows in timelines),
        scheduled=state.scheduled + (operation,),
        decision_action_ids=state.decision_action_ids + (action.action_id,),
    )


def _complete_state(instance: FJSPInstance, state: _SearchState, policy: str) -> _SearchState:
    current = state
    while len(current.scheduled) < instance.operation_count:
        candidates = _actions(instance, current)
        if not candidates:
            raise TrajectoryGateError("deterministic rollout reached an empty action set")
        current = _advance(current, min(candidates, key=lambda row: _policy_key(policy, row)))
    return current


def _policy_key(policy: str, item: CandidateAction) -> tuple[Any, ...]:
    common = (item.operation.job_id, item.operation.operation_id, item.machine_id)
    if policy == "earliest_completion_time":
        return (item.end_time, item.occupied_time) + common
    if policy == "shortest_processing_time":
        return (item.occupied_time, item.end_time) + common
    if policy == "most_work_remaining":
        return (-item.remaining_minimum_work, item.end_time, item.occupied_time) + common
    if policy == "most_operations_remaining":
        return (-item.remaining_operation_count, item.end_time, item.occupied_time) + common
    if policy == "fifo_job_order":
        return common + (item.end_time, item.occupied_time)
    raise TrajectoryGateError(f"unregistered rollout policy: {policy}")


def _beam_rank(instance: FJSPInstance, state: _SearchState) -> tuple[Any, ...]:
    horizon = _frame_upper_bound(instance)
    machine_finish = [max((row.end_time for row in rows), default=0.0) for rows in state.machine_timelines]
    max_finish = max(machine_finish, default=0.0)
    mean_ready = statistics.fmean(state.job_ready) if state.job_ready else 0.0
    load_spread = max(machine_finish, default=0.0) - min(machine_finish, default=0.0)
    return (
        (max_finish + mean_ready) / horizon,
        load_spread / horizon,
        tuple(state.decision_action_ids),
    )


def _deduplicate_and_score(
    instance: FJSPInstance,
    raw: Sequence[tuple[str, str, tuple[ScheduledOperation, ...]]],
) -> list[_TrajectoryCandidate]:
    by_fingerprint: dict[str, _TrajectoryCandidate] = {}
    for candidate_id, origin, schedule in raw:
        feasibility = verify_schedule(instance, schedule, reassignment=ReassignmentConfig())
        if not bool(feasibility["pass"]):
            raise TrajectoryGateError(f"candidate {candidate_id} is infeasible: {feasibility['errors']}")
        metrics = _schedule_metrics(instance, schedule)
        score = _trajectory_score(instance, metrics, schedule)
        fingerprint = _schedule_fingerprint(schedule)
        existing = by_fingerprint.get(fingerprint)
        if existing is None:
            by_fingerprint[fingerprint] = _TrajectoryCandidate(
                candidate_id=candidate_id,
                origins=(origin,),
                schedule=tuple(schedule),
                metrics=metrics,
                feasibility=feasibility,
                score_audit=score,
                fingerprint=fingerprint,
            )
        else:
            by_fingerprint[fingerprint] = replace(existing, origins=existing.origins + (origin,))
    return list(by_fingerprint.values())


def _trajectory_score(
    instance: FJSPInstance,
    metrics: Mapping[str, float],
    schedule: Sequence[ScheduledOperation],
) -> dict[str, Any]:
    horizon = _frame_upper_bound(instance)
    completion = _completion_times(instance, schedule)
    if any(value > horizon + 1e-9 for value in completion):
        raise TrajectoryGateError("conservative common frame does not cover trajectory")
    job_lower_service = [(horizon - value) / horizon for value in completion]
    system_lower_service = (horizon - float(metrics["makespan"])) / horizon
    job_queue_mass = float(instance.job_count)
    system_queue_mass = FROZEN_CONFIG.system_queue_weight_per_job * instance.job_count
    total_queue_mass = job_queue_mass + system_queue_mass
    weighted_lower_service = (
        sum(job_lower_service) + system_queue_mass * system_lower_service
    ) / total_queue_mass
    latency_normalization = (
        float(metrics["mean_flow_time"]) + float(metrics["makespan"])
    ) / (2.0 * horizon)
    bounded_penalty = FROZEN_CONFIG.bounded_latency_penalty * latency_normalization
    score = weighted_lower_service - bounded_penalty
    if not 0.0 <= bounded_penalty <= FROZEN_CONFIG.bounded_latency_penalty + 1e-12:
        raise TrajectoryGateError("trajectory penalty exceeded its declared bound")
    return {
        "score_semantics": "normalized_Q_dot_cumulative_lower_service_minus_bounded_trajectory_penalty",
        "score": _number(score),
        "common_frame_upper_bound": _number(horizon),
        "queue_vector": {
            "job_coordinate_count": instance.job_count,
            "queue_weight_per_job": 1,
            "system_drain_queue_weight": _number(system_queue_mass),
            "total_queue_mass": _number(total_queue_mass),
        },
        "lower_service": {
            "job_coordinates": [_number(value) for value in job_lower_service],
            "system_drain_coordinate": _number(system_lower_service),
            "normalized_weighted_value": _number(weighted_lower_service),
            "source": "deterministic_processing_times_exact_lower_bound",
        },
        "penalty": {
            "latency_normalization": _number(latency_normalization),
            "coefficient": _number(FROZEN_CONFIG.bounded_latency_penalty),
            "penalty_units": _number(bounded_penalty),
            "uniform_upper_bound": _number(FROZEN_CONFIG.bounded_latency_penalty),
            "queue_linear_coefficient_beta": 0,
        },
    }


def _run_suite_instance(source_row: Mapping[str, Any], split: str) -> dict[str, Any]:
    instance = source_row["instance"]
    result = build_trajectory_schedule(instance)
    policies: dict[str, dict[str, Any]] = {
        TRAJECTORY_POLICY: _policy_summary(result, source_row)
    }
    for policy in POLICY_ORDER:
        policies[policy] = _policy_summary(result["baseline_runs"][policy], source_row)
    frontier = _pareto_frontier(policies)
    if TRAJECTORY_POLICY not in frontier:
        raise TrajectoryGateError("trajectory score violated candidate-family Pareto monotonicity")
    return {
        "instance_id": str(source_row["id"]),
        "split": split,
        "source_raw_sha256": str(source_row["raw_sha256"]),
        "lower_bound": _number(source_row["lower_bound"]),
        "bks": _number(source_row["upper_bound"]),
        "job_count": instance.job_count,
        "machine_count": instance.machine_count,
        "operation_count": instance.operation_count,
        "policies": policies,
        "pareto_frontier_policies": frontier,
        "ours_pareto_dominated_by": [
            policy for policy in POLICY_ORDER if _dominates(policies[policy], policies[TRAJECTORY_POLICY])
        ],
        "trajectory_search": {
            "generated_candidate_count": result["selection"]["generated_candidate_count"],
            "unique_action_family_size": result["selection"]["unique_action_family_size"],
            "beam_terminal_state_count": result["selection"]["beam_terminal_state_count"],
            "selected_candidate_id": result["selection"]["selected_candidate_id"],
            "selected_origins": result["selection"]["selected_origins"],
            "exact_family_argmax": result["selection"]["exact_family_argmax"],
            "oracle_gap_within_generated_family": result["selection"]["oracle_gap_within_generated_family"],
            "runner_up_score_gap": result["selection"]["runner_up_score_gap"],
            "runtime_seconds": result["runtime"]["wall_clock_seconds"],
            "score_audit": result["selection"]["score_audit"],
        },
    }


def _policy_summary(run: Mapping[str, Any], source_row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics(run)
    makespan = metrics["makespan"]
    bks = float(source_row["upper_bound"])
    lower_bound = float(source_row["lower_bound"])
    return {
        "policy_label": str(run["policy_label"]),
        "makespan": _number(makespan),
        "mean_flow_time": _number(metrics["mean_flow_time"]),
        "makespan_over_bks": _number(makespan / bks),
        "bks_gap_fraction": _number((makespan - bks) / bks),
        "below_source_lower_bound": makespan < lower_bound - 1e-9,
        "matched_bks": _close(makespan, bks),
        "feasible": bool(run["feasibility"]["pass"]),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    policies: dict[str, dict[str, Any]] = {}
    for policy in COMPARISON_ORDER:
        items = [row["policies"][policy] for row in rows]
        flow_ratios = []
        for row in rows:
            best = min(float(item["mean_flow_time"]) for item in row["policies"].values())
            flow_ratios.append(float(row["policies"][policy]["mean_flow_time"]) / best)
        policies[policy] = {
            "policy_label": TRAJECTORY_LABEL if policy == TRAJECTORY_POLICY else POLICY_LABELS[policy],
            "instance_count": len(rows),
            "geomean_makespan_over_bks": _number(_geomean(float(item["makespan_over_bks"]) for item in items)),
            "geomean_mean_flow_to_instance_best": _number(_geomean(flow_ratios)),
            "pareto_frontier_count": sum(policy in row["pareto_frontier_policies"] for row in rows),
            "makespan_best_or_tie_count": sum(
                _close(
                    float(row["policies"][policy]["makespan"]),
                    min(float(item["makespan"]) for item in row["policies"].values()),
                )
                for row in rows
            ),
            "mean_flow_best_or_tie_count": sum(
                _close(
                    float(row["policies"][policy]["mean_flow_time"]),
                    min(float(item["mean_flow_time"]) for item in row["policies"].values()),
                )
                for row in rows
            ),
        }
    pairwise = {
        policy: _pairwise(rows, policy)
        for policy in CLASSIC_POLICIES
    }
    strict_all = all(
        comparison["ours_pareto_dominates_count"] == len(rows)
        for comparison in pairwise.values()
    )
    return {
        "instance_count": len(rows),
        "policies": policies,
        "ours_vs_each_classic": pairwise,
        "ours_strictly_pareto_dominates_every_classic_on_every_instance": strict_all,
    }


def _pairwise(rows: Sequence[Mapping[str, Any]], baseline: str) -> dict[str, Any]:
    ours_dominates = baseline_dominates = tradeoff_or_equal = 0
    makespan_ratios: list[float] = []
    flow_ratios: list[float] = []
    for row in rows:
        ours = row["policies"][TRAJECTORY_POLICY]
        other = row["policies"][baseline]
        makespan_ratios.append(float(ours["makespan"]) / float(other["makespan"]))
        flow_ratios.append(float(ours["mean_flow_time"]) / float(other["mean_flow_time"]))
        if _dominates(ours, other):
            ours_dominates += 1
        elif _dominates(other, ours):
            baseline_dominates += 1
        else:
            tradeoff_or_equal += 1
    return {
        "geomean_makespan_ratio": _number(_geomean(makespan_ratios)),
        "geomean_mean_flow_ratio": _number(_geomean(flow_ratios)),
        "ours_pareto_dominates_count": ours_dominates,
        "baseline_pareto_dominates_count": baseline_dominates,
        "tradeoff_or_equal_count": tradeoff_or_equal,
    }


def _theorem_mapping() -> dict[str, Any]:
    return {
        "action": "complete_nonpreemptive_fjsp_schedule_trajectory",
        "candidate_family": "finite_fixed_policy_seeds_plus_fixed_prefix_beam_rollouts",
        "queue_vector": "one_unit_per_unfinished_job_plus_fixed_system_drain_coordinate",
        "lower_service": {
            "job_j": "(H-C_j)/H",
            "system_drain": "(H-C_max)/H",
            "common_frame_H": "sum_over_operations_max_eligible_processing_time",
            "robustness": "deterministic_processing_times_make_coordinates_exact_lower_bounds",
        },
        "objective": "normalized_Q_dot_lower_service_minus_bounded_trajectory_penalty",
        "penalty": {
            "form": "gamma*(mean_flow+makespan)/(2H)",
            "gamma": _number(FROZEN_CONFIG.bounded_latency_penalty),
            "uniform_bound_P0": _number(FROZEN_CONFIG.bounded_latency_penalty),
            "queue_scaled_beta": 0,
            "slack_implication": "bounded_P0_changes_the_finite_frame_drift_constant_not_the_linear_slack_margin",
        },
        "oracle": {
            "exact_over_generated_family": True,
            "alpha0": 0,
            "alpha1": 0,
            "global_fjsp_action_space_exact": False,
            "candidate_cover_error_calibrated": False,
        },
        "claim_boundary": (
            "This is a finite deterministic draining-problem configuration mapping. "
            "It does not by itself establish stochastic-arrival recurrence or throughput optimality."
        ),
    }


def _frame_upper_bound(instance: FJSPInstance) -> float:
    bound = sum(
        max(float(option.processing_time) for option in operation.alternatives)
        for job in instance.jobs
        for operation in job.operations
    )
    if bound <= 0:
        raise TrajectoryGateError("frame upper bound must be positive")
    return bound


def _schedule_metrics(
    instance: FJSPInstance, schedule: Sequence[ScheduledOperation]
) -> dict[str, float]:
    completion = _completion_times(instance, schedule)
    return {
        "makespan": max(completion, default=0.0),
        "mean_flow_time": statistics.fmean(completion) if completion else 0.0,
    }


def _completion_times(
    instance: FJSPInstance, schedule: Sequence[ScheduledOperation]
) -> list[float]:
    completion = [0.0 for _ in instance.jobs]
    for row in schedule:
        completion[row.job_id] = max(completion[row.job_id], row.end_time)
    return completion


def _schedule_from_run(run: Mapping[str, Any]) -> tuple[ScheduledOperation, ...]:
    rows = []
    for item in run["schedule"]:
        rows.append(
            ScheduledOperation(
                job_id=int(item["job_id"]),
                operation_id=int(item["operation_id"]),
                machine_id=int(item["machine_id"]),
                start_time=float(item["start_time"]),
                end_time=float(item["end_time"]),
                processing_time=float(item["processing_time"]),
                reconfiguration_time=float(item["reconfiguration_time"]),
                preferred_machine_id=int(item["preferred_machine_id"]),
                reassignment_semantics_enabled=False,
            )
        )
    return tuple(rows)


def _metrics(run: Mapping[str, Any]) -> dict[str, float]:
    return {
        "makespan": float(run["metrics"]["makespan"]),
        "mean_flow_time": float(run["metrics"]["mean_flow_time"]),
    }


def _schedule_fingerprint(schedule: Sequence[ScheduledOperation]) -> str:
    payload = [
        (
            row.job_id,
            row.operation_id,
            row.machine_id,
            _number(row.start_time),
            _number(row.end_time),
        )
        for row in sorted(schedule, key=lambda item: (item.job_id, item.operation_id))
    ]
    return sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _state_fingerprint(state: _SearchState) -> str:
    return sha256("\n".join(state.decision_action_ids).encode("utf-8")).hexdigest()


def _schedule_output_order(
    schedule: Sequence[ScheduledOperation],
) -> list[ScheduledOperation]:
    return sorted(schedule, key=lambda row: (row.start_time, row.machine_id, row.job_id, row.operation_id))


def _operation_order(row: ScheduledOperation) -> tuple[Any, ...]:
    return (row.start_time, row.end_time, row.job_id, row.operation_id)


def _pareto_frontier(rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        policy
        for policy in COMPARISON_ORDER
        if policy in rows
        and not any(other != policy and _dominates(rows[other], rows[policy]) for other in rows)
    ]


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    pairs = [
        (float(left["makespan"]), float(right["makespan"])),
        (float(left["mean_flow_time"]), float(right["mean_flow_time"])),
    ]
    return all(lhs <= rhs + 1e-9 for lhs, rhs in pairs) and any(
        lhs < rhs - 1e-9 for lhs, rhs in pairs
    )


def _verify_frozen_config() -> str:
    actual = frozen_config_sha256()
    if actual != FROZEN_CONFIG_SHA256:
        raise TrajectoryGateError(
            f"frozen configuration hash mismatch: expected {FROZEN_CONFIG_SHA256}, got {actual}"
        )
    return actual


def _verify_split_partition() -> None:
    flattened = tuple(instance for split in SPLITS.values() for instance in split)
    if flattened != EXPECTED_INSTANCE_IDS or len(set(flattened)) != len(flattened):
        raise TrajectoryGateError("development/calibration/holdout split is not Mk01-Mk15 exactly")


def _implementation_manifest() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("fjsp_benchmark.py"),
        Path(__file__).resolve().with_name("fjsp_instances.py"),
    )
    files = []
    combined = sha256()
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(REPO_ROOT).as_posix()
        files.append({"path": relative, "sha256": sha256(data).hexdigest()})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
    return {"files": files, "combined_sha256": combined.hexdigest()}


def _geomean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items or any(value <= 0 for value in items):
        raise TrajectoryGateError("geometric mean inputs must be positive")
    return math.exp(statistics.fmean(math.log(value) for value in items))


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _number(value: Any) -> int | float:
    numeric = float(value)
    if math.isclose(numeric, round(numeric), rel_tol=0.0, abs_tol=1e-12):
        return int(round(numeric))
    return round(numeric, 12)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if math.isclose(numeric, round(numeric), rel_tol=0.0, abs_tol=1e-12):
        return str(int(round(numeric)))
    return f"{numeric:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_fjsp_trajectory_suite(args.suite_dir, args.manifest)
    json_path, markdown_path = write_artifacts(
        report, json_path=args.json, markdown_path=args.markdown
    )
    print(json.dumps({"gate": report["gate"], "json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
